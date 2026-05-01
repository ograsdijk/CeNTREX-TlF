"""Build and cache the Jmax=6 truth reference for the perf bench.

Reference configuration matches the demo notebook's calibrated magnetic-RF setup:
    - DC plateau 2 → 30 → 2 kV/cm, ramp midpoint at ±1.5 m, ramp 5 cm
    - Two magnetic-RF coils at z = ±1.25 m, B-amplitude = 0.113 G (calibrated π/2)
    - Carrier omega_rf = 2π · 119.64 kHz (Tl spin-flip resonance)
    - phi1 = 0, phi2 = +π/2  (Ramsey working point)
    - Trajectory v_z = 184 m/s, z = -2 → +2 m
    - Initial state: adiabatic ancestor of |J=1, mJ=-1, m1=-1/2, m2=-1/2⟩ at 30 kV/cm
    - Jmax = 6, segmented grid with dt_fine = 0.5 µs

The reference is cached as `_cache/reference_jmax6_dt0p5us.npz` next to this file.
Each entry in the cache:
    Psi_final    (N=196, K=1) complex128  — final state, the gold-standard target
    survival     ()           float       — return-prob to Psi0
    per_j        (1, 4)       float       — bare-J populations
    t_grid       (n_steps+1,) float
    elapsed_s    ()           float       — wall-clock to compute the reference

Re-running the script reuses the cache unless `--force` is passed.

Single-trajectory at Jmax=6 takes ~9 min on this machine (basis 196, ~14k steps,
~40 ms eigh per step in the segmented active region + 3 single-shot exact steps
in the inactive region).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # examples/ramsey_rf/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centrex_tlf.states import ElectronicState, UncoupledBasisState  # noqa: E402

from ramsey_rf import (  # noqa: E402
    AnalyticDCField,
    BallisticTrajectory,
    FieldStack,
    MagneticRFRegion,
    RamseyRFConfig,
    RamseyRFSimulator,
    adiabatic_dressed_initial_states,
    build_basis,
    build_H_func,
    build_segmented_t_grid,
)


# -------- Reference configuration (same numbers as the demo notebook) --------
JMAX = 6
DT_FINE = 0.5e-6
Z_RF1, Z_RF2 = -1.25, +1.25
RF_WIDTH, RF_EDGE = 0.30, 0.05
DC_HALF_WIDTH, DC_RAMP_LENGTH = 1.50, 0.05
E_LOW, E_HIGH = 2e3, 30e3
F_RES = 119.64e3
OMEGA_RF = 2 * np.pi * F_RES
RF_AMPLITUDE = 0.113   # Gauss
PHI1 = 0.0
PHI2 = +np.pi / 2
V_Z = 184.0
Z_START, Z_FINAL = -2.0, +2.0

CACHE_DIR = HERE / "_cache"
CACHE_FILE = CACHE_DIR / f"reference_jmax{JMAX}_dt{int(DT_FINE * 1e9)}ns.npz"


@dataclass
class ReferenceResult:
    Psi_final: np.ndarray          # (N, K)
    survival: float
    per_j: np.ndarray              # (K, n_J)
    J_values: list[int]
    t_grid: np.ndarray
    elapsed_s: float
    n_steps: int
    Jmax: int
    dt_fine: float

    @classmethod
    def load(cls, path: Path) -> "ReferenceResult":
        data = np.load(path, allow_pickle=False)
        return cls(
            Psi_final=data["Psi_final"],
            survival=float(data["survival"]),
            per_j=data["per_j"],
            J_values=[int(j) for j in data["J_values"]],
            t_grid=data["t_grid"],
            elapsed_s=float(data["elapsed_s"]),
            n_steps=int(data["n_steps"]),
            Jmax=int(data["Jmax"]),
            dt_fine=float(data["dt_fine"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            Psi_final=self.Psi_final,
            survival=np.float64(self.survival),
            per_j=self.per_j,
            J_values=np.array(self.J_values, dtype=np.int64),
            t_grid=self.t_grid,
            elapsed_s=np.float64(self.elapsed_s),
            n_steps=np.int64(self.n_steps),
            Jmax=np.int64(self.Jmax),
            dt_fine=np.float64(self.dt_fine),
        )


def build_reference_config():
    """Construct (cfg, Psi0, traj, fields, H_func, QN) for the reference run."""
    QN = build_basis(Jmax=JMAX)
    H_func = build_H_func(QN)

    # Adiabatic-ancestor initial state (single column)
    target = UncoupledBasisState(
        J=1, mJ=-1, I1=0.5, m1=-0.5, I2=0.5, m2=-0.5,
        Omega=0, P=-1, electronic_state=ElectronicState.X,
    )
    Psi0, _, _ = adiabatic_dressed_initial_states(
        H_func,
        E_init=(0.0, 0.0, E_LOW),
        E_target=(0.0, 0.0, E_HIGH),
        QN=QN,
        targets=[target],
        n_steps=80,
    )

    dc = AnalyticDCField.symmetric_plateau(
        E_low=E_LOW, E_high=E_HIGH,
        half_width=DC_HALF_WIDTH, ramp_length=DC_RAMP_LENGTH,
        direction=(0, 0, 1),
    )
    rf1 = MagneticRFRegion.rounded_rectangle(
        z_center=Z_RF1, width=RF_WIDTH, edge_length=RF_EDGE,
        amplitude=RF_AMPLITUDE, omega=OMEGA_RF, phi=PHI1, direction=(1, 0, 0),
    )
    rf2 = MagneticRFRegion.rounded_rectangle(
        z_center=Z_RF2, width=RF_WIDTH, edge_length=RF_EDGE,
        amplitude=RF_AMPLITUDE, omega=OMEGA_RF, phi=PHI2, direction=(1, 0, 0),
    )
    fields = FieldStack(E_dc=dc, rf_regions_B=[rf1, rf2])
    traj = BallisticTrajectory(
        r0=np.array([0.0, 0.0, Z_START]), v=np.array([0.0, 0.0, V_Z]),
    )

    grid = build_segmented_t_grid(
        traj, fields, traj.t_at_z(Z_START), traj.t_at_z(Z_FINAL),
        dt_fine=DT_FINE, guard_probes=3,
    )
    cfg = RamseyRFConfig(
        fields=fields, trajectory=traj,
        z_start=Z_START, z_final=Z_FINAL,
        initial_psi0=Psi0, Jmax=JMAX, t_grid=grid,
    )
    return cfg, Psi0, QN, H_func


def compute_reference() -> ReferenceResult:
    cfg, _Psi0, QN, H_func = build_reference_config()
    sim = RamseyRFSimulator(cfg, QN=QN, H_func=H_func)
    n_steps = sim.t_grid.size - 1
    print(f"Reference config: Jmax={JMAX}, basis={len(QN)}, "
          f"dt_fine={DT_FINE * 1e6:.2f} us, segmented grid n_steps={n_steps}")
    print(f"Starting reference trajectory (estimated ~9 min)...")
    t0 = time.perf_counter()
    res = sim.run()
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed / 60:.2f} min.")
    print(f"  survival       = {res.survival_weighted:.6f}")
    print(f"  per-J weighted = {dict(zip(res.J_values, res.per_j_weighted))}")
    print(f"  norm dev max   = {np.max(np.abs(res.norm_trace - 1.0)):.3e}")
    return ReferenceResult(
        Psi_final=res.Psi_final,
        survival=res.survival_weighted,
        per_j=res.per_j,
        J_values=res.J_values,
        t_grid=res.t_grid,
        elapsed_s=elapsed,
        n_steps=n_steps,
        Jmax=JMAX,
        dt_fine=DT_FINE,
    )


def get_reference(force: bool = False) -> ReferenceResult:
    """Load the cached reference, or compute and cache it if missing/--force."""
    if CACHE_FILE.exists() and not force:
        print(f"Loading cached reference from {CACHE_FILE}")
        return ReferenceResult.load(CACHE_FILE)
    result = compute_reference()
    result.save(CACHE_FILE)
    print(f"Cached reference to {CACHE_FILE}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists")
    args = parser.parse_args()
    ref = get_reference(force=args.force)
    print(f"\nReference cache OK: {CACHE_FILE}")
    print(f"  Psi_final shape = {ref.Psi_final.shape}")
    print(f"  survival = {ref.survival:.6f}")
    print(f"  elapsed = {ref.elapsed_s / 60:.2f} min")
