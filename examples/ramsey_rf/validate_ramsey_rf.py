"""Standalone numerical-invariant validation for the RF Ramsey simulator.

Runs a sequence of cheap-to-fast checks and exits 0 on pass / 1 on first
failure. Intended to be runnable as:

    .venv/Scripts/python.exe examples/ramsey_rf/validate_ramsey_rf.py

Uses small Jmax and short trajectories so the whole script finishes in
a couple of minutes. The demo notebook (`demo_ramsey_rf.ipynb`) covers the
full Jmax=6 / dt=0.5 us geometry separately.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the local `ramsey_rf` package importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from centrex_tlf.states import ElectronicState  # noqa: E402

from ramsey_rf import (  # noqa: E402
    AnalyticDCField,
    AnalyticRFRegion,
    BallisticTrajectory,
    FieldStack,
    RamseyRFConfig,
    RamseyRFSimulator,
    ScanSpec,
    build_basis,
    build_H_func,
    run_scan,
)
from ramsey_rf.states import UncoupledSelector  # noqa: E402


# -------- Small geometry used by the validation (NOT the demo geometry) --------
Z_RF1, Z_RF2 = -0.30, +0.30
RF_WIDTH = 0.10
RF_EDGE = 0.02
DC_HALF_WIDTH = 0.50
DC_RAMP_LENGTH = 0.10
E_LOW, E_HIGH = 2e3, 30e3
OMEGA_RF = 2 * np.pi * 120e3
V_Z = 184.0
Z_START, Z_FINAL = -0.6, +0.6
JMAX_VALIDATE = 3      # basis size 64 for fast eigh
DT_VALIDATE = 1.0e-6
RF_AMPLITUDE = 50.0    # V/cm — chosen to give an observable but partial transfer


def make_config(Jmax: int = JMAX_VALIDATE, rf_amp: float = RF_AMPLITUDE,
                phi2: float = 0.0, dt: float = DT_VALIDATE) -> RamseyRFConfig:
    dc = AnalyticDCField.symmetric_plateau(
        E_low=E_LOW, E_high=E_HIGH,
        half_width=DC_HALF_WIDTH, ramp_length=DC_RAMP_LENGTH,
        direction=(0, 0, 1),
    )
    rf1 = AnalyticRFRegion.rounded_rectangle(
        z_center=Z_RF1, width=RF_WIDTH, edge_length=RF_EDGE,
        amplitude=rf_amp, omega=OMEGA_RF, phi=0.0, direction=(1, 0, 0),
    )
    rf2 = AnalyticRFRegion.rounded_rectangle(
        z_center=Z_RF2, width=RF_WIDTH, edge_length=RF_EDGE,
        amplitude=rf_amp, omega=OMEGA_RF, phi=phi2, direction=(1, 0, 0),
    )
    fields = FieldStack(E_dc=dc, rf_regions=[rf1, rf2])
    traj = BallisticTrajectory(r0=np.array([0, 0, Z_START]), v=np.array([0, 0, V_Z]))
    return RamseyRFConfig(
        fields=fields,
        trajectory=traj,
        z_start=Z_START, z_final=Z_FINAL,
        initial_targets=UncoupledSelector(J=1, electronic=ElectronicState.X),
        Jmax=Jmax,
        dt=dt,
    )


# -------- Reporting helpers --------
PASS = "[ OK ]"
FAIL = "[FAIL]"


def report(label: str, ok: bool, details: str) -> None:
    tag = PASS if ok else FAIL
    print(f"{tag} {label}: {details}")


def fail(label: str, details: str) -> None:
    report(label, False, details)
    print("\nValidation failed.")
    sys.exit(1)


# -------- Checks --------
def check_hermiticity() -> None:
    cfg = make_config()
    sim = RamseyRFSimulator(cfg)
    # sample H at several random times across the trajectory
    rng = np.random.default_rng(0)
    ts = sim.t_start + (sim.t_end - sim.t_start) * rng.random(8)
    worst = 0.0
    for t in ts:
        H = sim._H_at_t(float(t))
        d = float(np.max(np.abs(H - H.conj().T)))
        worst = max(worst, d)
    ok = worst < 1e-10
    report("Hermiticity of H_mid", ok, f"max |H - H†| = {worst:.3e}")
    if not ok:
        fail("Hermiticity", "tolerance 1e-10")


def check_norm_conservation_with_rf() -> RamseyRFSimulator:
    cfg = make_config()
    sim = RamseyRFSimulator(cfg)
    res = sim.run()
    worst = float(np.max(np.abs(res.norm_trace - 1.0)))
    ok = worst < 1e-9
    report("Norm conservation (with RF)", ok,
           f"max |‖Psi‖² - 1| = {worst:.3e} over {res.norm_trace.shape[0]} steps")
    if not ok:
        fail("Norm conservation", "tolerance 1e-9")
    return sim


def check_no_rf_unitarity() -> None:
    cfg = make_config(rf_amp=0.0)
    sim = RamseyRFSimulator(cfg)
    res = sim.run()
    worst_norm = float(np.max(np.abs(res.norm_trace - 1.0)))
    ok_norm = worst_norm < 1e-9
    report("No-RF norm conservation", ok_norm, f"max |‖Psi‖² - 1| = {worst_norm:.3e}")
    if not ok_norm:
        fail("No-RF norm conservation", "tolerance 1e-9")
    # With no RF, each initial dressed eigenstate at E_low maps to a dressed
    # eigenstate at E_high. The detection target is the SAME bare state, dressed
    # at E_high. If the field ramps adiabatically, survival should be ≈ 1.
    min_surv = float(np.min(res.survival))
    mean_surv = float(np.mean(res.survival))
    ok_surv = min_surv > 0.95
    report("No-RF adiabatic survival", ok_surv,
           f"min survival = {min_surv:.4f}, mean = {mean_surv:.4f}")
    if not ok_surv:
        fail("No-RF survival", "min survival should be > 0.95 (relax bound)")


def check_dt_convergence() -> None:
    surv = {}
    for dt in (4.0e-6, 2.0e-6, 1.0e-6):
        cfg = make_config(dt=dt)
        res = RamseyRFSimulator(cfg).run()
        surv[dt] = res.survival_weighted
        print(f"   dt={dt*1e6:.2f} us → survival_weighted = {res.survival_weighted:.6f}")
    diff_high = abs(surv[1.0e-6] - surv[2.0e-6])
    diff_low = abs(surv[4.0e-6] - surv[2.0e-6])
    converging = diff_high <= diff_low + 1e-3
    report("dt convergence", converging,
           f"|surv(2)-surv(1)| = {diff_high:.3e}, |surv(4)-surv(2)| = {diff_low:.3e}")
    if not converging:
        fail("dt convergence", "halving dt should not increase the change in survival")


def check_phi2_fringe() -> None:
    cfg = make_config()
    spec = ScanSpec(axis="phi2", values=np.linspace(0, 2 * np.pi, 17, endpoint=False))
    res = run_scan(cfg, spec)
    pp = float(res.survival_weighted.max() - res.survival_weighted.min())
    print(f"   phi2 scan: survival range = "
          f"[{res.survival_weighted.min():.4f}, {res.survival_weighted.max():.4f}]")
    ok = pp > 0.01
    report("φ2 Ramsey fringe present", ok, f"peak-to-peak = {pp:.4f}")
    if not ok:
        fail("φ2 fringe", "expected non-trivial peak-to-peak amplitude in survival")


def check_jmax_convergence() -> None:
    out = {}
    for Jmax in (3, 4):
        cfg = make_config(Jmax=Jmax)
        res = RamseyRFSimulator(cfg).run()
        out[Jmax] = res.survival_weighted
        print(f"   Jmax={Jmax} → survival_weighted = {res.survival_weighted:.6f}")
    rel = abs(out[4] - out[3]) / max(abs(out[4]), 1e-12)
    ok = rel < 0.20
    report("Jmax convergence (Jmax=3 vs 4, validation geometry)", ok,
           f"rel diff = {rel*100:.2f}%")
    if not ok:
        fail("Jmax convergence",
             "validation geometry gives a coarse convergence check; relax bound to 20%")


def main() -> int:
    print("=" * 70)
    print("RF Ramsey benchmark — numerical-invariant validation")
    print(f"  Jmax = {JMAX_VALIDATE} (validation geometry; smaller than the demo)")
    print(f"  trajectory: z = {Z_START} → {Z_FINAL} m at v_z = {V_Z} m/s")
    print(f"  RF coils at z = {Z_RF1}, {Z_RF2} m (width {RF_WIDTH} m, "
          f"omega = 2π·120 kHz)")
    print(f"  dt = {DT_VALIDATE*1e6:.2f} us")
    print("=" * 70)

    t0 = time.perf_counter()
    check_hermiticity()
    check_norm_conservation_with_rf()
    check_no_rf_unitarity()
    check_dt_convergence()
    check_phi2_fringe()
    check_jmax_convergence()
    print("=" * 70)
    print(f"All checks passed in {time.perf_counter() - t0:.1f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
