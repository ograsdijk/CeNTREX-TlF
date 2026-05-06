"""Generate CPU V3b scan data for Ramsey RF analysis notebooks.

The production setting comes from the V3b basis_dt convergence sweep:
K=24, basis_dt=10 us, dt_fine=0.05 us. This script intentionally avoids GPU
paths and writes compact NPZ caches for notebooks to load.
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cloudpickle
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centrex_tlf.hamiltonian import generate_uncoupled_hamiltonian_X  # noqa: E402

from ramsey_rf import (  # noqa: E402
    BallisticTrajectory,
    build_segmented_t_grid,
    make_uniform_tracking_grid,
    propagate_midpoint_tracked_decomposed,
    select_subspace_by_overlap,
    track_subspace_bases,
)

from benchmarks.reference import (  # noqa: E402
    E_HIGH,
    E_LOW,
    F_RES,
    V_Z,
    Z_FINAL,
    Z_START,
    build_reference_config,
)


K_DEFAULT = 24
BASIS_DT_US_DEFAULT = 10.0
FRINGE_HZ = V_Z / 2.5
TWOPI = 2.0 * np.pi


def _raw_components(QN):
    Hraw = generate_uncoupled_hamiltonian_X(QN)
    return [
        Hraw.Hff,
        Hraw.HSx,
        Hraw.HSy,
        Hraw.HSz,
        Hraw.HZx,
        Hraw.HZy,
        Hraw.HZz,
    ]


def _high_field_j1_subspace(QN, H_func, K: int) -> np.ndarray:
    H_high = H_func(np.array([0.0, 0.0, E_HIGH]), np.zeros(3))
    bare_j1_idx = np.array([i for i, bs in enumerate(QN) if bs.J == 1])
    bare_j1_targets = np.zeros((len(QN), bare_j1_idx.size), dtype=np.complex128)
    for col, idx in enumerate(bare_j1_idx):
        bare_j1_targets[idx, col] = 1.0
    T_high, _idx = select_subspace_by_overlap(
        H_high, K=K, target_vectors=bare_j1_targets
    )
    return T_high


def _track_subspace_to_start(T_high: np.ndarray, H_func) -> np.ndarray:
    e_path = np.linspace(E_HIGH, E_LOW, 81)

    def H_at_e(e_z: float) -> np.ndarray:
        return H_func(np.array([0.0, 0.0, e_z]), np.zeros(3))

    return track_subspace_bases(e_path, H_at_e, T_high)[-1]


def _set_rf_omega(cfg, omega: float) -> None:
    for region in cfg.fields.rf_regions:
        region.omega = float(omega)
    for region in cfg.fields.rf_regions_B:
        region.omega = float(omega)


def _set_velocity(cfg, velocity: float) -> None:
    cfg.trajectory = BallisticTrajectory(
        r0=np.array([0.0, 0.0, Z_START], dtype=np.float64),
        v=np.array([0.0, 0.0, velocity], dtype=np.float64),
    )
    cfg.t_grid = build_segmented_t_grid(
        cfg.trajectory,
        cfg.fields,
        cfg.trajectory.t_at_z(Z_START),
        cfg.trajectory.t_at_z(Z_FINAL),
        dt_fine=0.05e-6,
        guard_probes=3,
    )


def _coeffs_builder(cfg):
    traj = cfg.trajectory
    fields = cfg.fields

    def coeffs_at_t(t: float) -> np.ndarray:
        R = traj(t)
        E = fields.E_total(R, t)
        B = fields.B_total(R, t)
        return TWOPI * np.array([1.0, E[0], E[1], E[2], B[0], B[1], B[2]])

    return coeffs_at_t


_CFG = None
_PSI0 = None
_COMPONENTS = None
_TRACK_TIMES = None
_TRACKED_BASES = None


def _worker_init(cfg_blob: bytes, K: int, basis_dt_us: float) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"

    global _CFG, _PSI0, _COMPONENTS, _TRACK_TIMES, _TRACKED_BASES
    _CFG = cloudpickle.loads(cfg_blob)

    from ramsey_rf import build_basis as _build_basis, build_H_func as _build_H_func

    QN = _build_basis(_CFG.Jmax)
    H_func = _build_H_func(QN)
    if _CFG.initial_psi0 is None:
        raise ValueError("V3b scan requires cfg.initial_psi0")
    _PSI0 = np.asarray(_CFG.initial_psi0, dtype=np.complex128)
    _COMPONENTS = _raw_components(QN)

    T_high = _high_field_j1_subspace(QN, H_func, K)
    T_start = _track_subspace_to_start(T_high, H_func)
    traj = _CFG.trajectory
    fields = _CFG.fields

    def H_basis_at_t(t: float) -> np.ndarray:
        R = traj(t)
        B = fields.B_dc(R) if fields.B_dc is not None else np.zeros(3)
        return H_func(fields.E_dc(R), B)

    _TRACK_TIMES = make_uniform_tracking_grid(
        float(_CFG.t_grid[0]),
        float(_CFG.t_grid[-1]),
        basis_dt_us * 1e-6,
    )
    _TRACKED_BASES = track_subspace_bases(_TRACK_TIMES, H_basis_at_t, T_start)


def _worker_run(omega: float) -> float:
    cfg_i = copy.deepcopy(_CFG)
    _set_rf_omega(cfg_i, omega)
    out = propagate_midpoint_tracked_decomposed(
        _PSI0,
        cfg_i.t_grid,
        _COMPONENTS,
        _coeffs_builder(cfg_i),
        _TRACK_TIMES,
        _TRACKED_BASES,
        store_norm=False,
    )
    return float(abs(np.vdot(_PSI0[:, 0], out.Psi_final[:, 0])) ** 2)


def _worker_run_phi2(phi2: float) -> float:
    """Variant of `_worker_run` that varies the second RF region's phase
    while leaving omega and phi1 at the values baked into the cfg blob.

    Module-level so spawn workers can find it via re-import.
    """
    cfg_i = copy.deepcopy(_CFG)
    n_e = len(cfg_i.fields.rf_regions)
    n_b = len(cfg_i.fields.rf_regions_B)
    if n_e + n_b < 2:
        raise RuntimeError("phi2 scan requires at least 2 RF regions")
    if n_e >= 2:
        cfg_i.fields.rf_regions[1].phi = float(phi2)
    elif n_e == 1:
        cfg_i.fields.rf_regions_B[0].phi = float(phi2)
    else:
        cfg_i.fields.rf_regions_B[1].phi = float(phi2)
    out = propagate_midpoint_tracked_decomposed(
        _PSI0,
        cfg_i.t_grid,
        _COMPONENTS,
        _coeffs_builder(cfg_i),
        _TRACK_TIMES,
        _TRACKED_BASES,
        store_norm=False,
    )
    return float(abs(np.vdot(_PSI0[:, 0], out.Psi_final[:, 0])) ** 2)


_BLAS_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _force_single_thread_blas_in_children() -> None:
    """Set BLAS thread caps in the PARENT environment so spawn child processes
    inherit them at startup (before numpy is imported by the child). The same
    keys are also set in `_worker_init`, but that's too late on the worker side
    because numpy gets re-imported during spawn before the initializer runs;
    setting them here in the parent is the only place that takes effect.
    """
    for key in _BLAS_ENV_KEYS:
        os.environ.setdefault(key, "1")


def run_frequency_scan(
    freqs: np.ndarray,
    *,
    velocity: float,
    K: int,
    basis_dt_us: float,
    n_workers: int,
) -> tuple[np.ndarray, float, int]:
    cfg, _Psi0, _QN, _H_func = build_reference_config()
    _set_velocity(cfg, velocity)
    _force_single_thread_blas_in_children()
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(cloudpickle.dumps(cfg), K, basis_dt_us),
    ) as pool:
        survival = np.array(
            list(pool.imap(_worker_run, [float(TWOPI * f) for f in freqs])),
            dtype=np.float64,
        )
    elapsed_s = time.perf_counter() - t0
    return survival, elapsed_s, int(cfg.t_grid.size - 1)


def run_phi2_scan_at_frequency(
    phi2_grid: np.ndarray,
    *,
    frequency: float,
    velocity: float,
    phi1: float = 0.0,
    K: int,
    basis_dt_us: float,
    n_workers: int,
) -> tuple[np.ndarray, float, int]:
    """Parallel scan over the second-coil RF phase at fixed carrier frequency.

    Bakes omega = 2π·frequency and phi1 into the cfg blob shipped to workers;
    each worker varies only phi2 via `_worker_run_phi2`.
    """
    cfg, _Psi0, _QN, _H_func = build_reference_config()
    _set_velocity(cfg, velocity)
    omega = float(TWOPI * frequency)
    _set_rf_omega(cfg, omega)
    n_e = len(cfg.fields.rf_regions)
    if n_e >= 1:
        cfg.fields.rf_regions[0].phi = float(phi1)
    elif cfg.fields.rf_regions_B:
        cfg.fields.rf_regions_B[0].phi = float(phi1)
    _force_single_thread_blas_in_children()
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(cloudpickle.dumps(cfg), K, basis_dt_us),
    ) as pool:
        survival = np.array(
            list(pool.imap(_worker_run_phi2, [float(p) for p in phi2_grid])),
            dtype=np.float64,
        )
    elapsed_s = time.perf_counter() - t0
    return survival, elapsed_s, int(cfg.t_grid.size - 1)


def gaussian_velocity_nodes(mean: float, std: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    velocities = mean + np.sqrt(2.0) * std * nodes
    weights = weights / np.sqrt(np.pi)
    positive = velocities > 0.0
    velocities = velocities[positive]
    weights = weights[positive]
    weights = weights / np.sum(weights)
    return velocities.astype(np.float64), weights.astype(np.float64)


def save_far_scan(args: argparse.Namespace) -> None:
    offsets = np.linspace(-args.fringes, args.fringes, args.n_freq)
    freqs = F_RES + offsets * FRINGE_HZ
    survival, elapsed_s, n_steps = run_frequency_scan(
        freqs,
        velocity=args.mean_velocity,
        K=args.K,
        basis_dt_us=args.basis_dt_us,
        n_workers=args.n_workers,
    )
    args.far_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.far_out,
        freqs=freqs,
        offsets=offsets,
        survival=survival,
        mean_velocity=np.float64(args.mean_velocity),
        velocity=np.float64(args.mean_velocity),
        fringe_hz=np.float64(FRINGE_HZ),
        K=np.int64(args.K),
        basis_dt_us=np.float64(args.basis_dt_us),
        n_workers=np.int64(args.n_workers),
        n_steps=np.int64(n_steps),
        elapsed_s=np.float64(elapsed_s),
    )
    print(
        f"far scan: {args.n_freq} points, {elapsed_s / 60:.2f} min, "
        f"survival range=({survival.min():.6g}, {survival.max():.6g})"
    )
    print(f"wrote {args.far_out}")


def save_velocity_scan(args: argparse.Namespace) -> None:
    offsets = np.linspace(-args.fringes, args.fringes, args.n_freq)
    freqs = F_RES + offsets * FRINGE_HZ
    velocities, weights = gaussian_velocity_nodes(
        args.mean_velocity, args.velocity_std, args.n_velocity_nodes
    )
    survival_by_velocity = np.empty((velocities.size, freqs.size), dtype=np.float64)
    elapsed_by_velocity = np.empty(velocities.size, dtype=np.float64)
    n_steps_by_velocity = np.empty(velocities.size, dtype=np.int64)
    for idx, velocity in enumerate(velocities):
        print(
            f"velocity node {idx + 1}/{velocities.size}: "
            f"v={velocity:.3f} m/s, weight={weights[idx]:.6f}"
        )
        survival, elapsed_s, n_steps = run_frequency_scan(
            freqs,
            velocity=float(velocity),
            K=args.K,
            basis_dt_us=args.basis_dt_us,
            n_workers=args.n_workers,
        )
        survival_by_velocity[idx] = survival
        elapsed_by_velocity[idx] = elapsed_s
        n_steps_by_velocity[idx] = n_steps
        print(
            f"  done in {elapsed_s / 60:.2f} min, "
            f"survival range=({survival.min():.6g}, {survival.max():.6g})"
        )

    averaged = weights @ survival_by_velocity
    args.velocity_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.velocity_out,
        freqs=freqs,
        offsets=offsets,
        velocities=velocities,
        weights=weights,
        survival_by_velocity=survival_by_velocity,
        survival_velocity_averaged=averaged,
        mean_velocity=np.float64(args.mean_velocity),
        velocity_std=np.float64(args.velocity_std),
        fringe_hz=np.float64(FRINGE_HZ),
        K=np.int64(args.K),
        basis_dt_us=np.float64(args.basis_dt_us),
        n_workers=np.int64(args.n_workers),
        n_velocity_nodes=np.int64(velocities.size),
        n_steps_by_velocity=n_steps_by_velocity,
        elapsed_by_velocity_s=elapsed_by_velocity,
        elapsed_total_s=np.float64(np.sum(elapsed_by_velocity)),
    )
    print(
        f"velocity averaged scan: {freqs.size} freq x {velocities.size} velocities, "
        f"{np.sum(elapsed_by_velocity) / 60:.2f} min"
    )
    print(f"wrote {args.velocity_out}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("far", "velocity", "both"), default="both"
    )
    parser.add_argument("--fringes", type=float, default=10.0)
    parser.add_argument("--n-freq", type=int, default=161)
    parser.add_argument("--mean-velocity", type=float, default=184.0)
    parser.add_argument("--velocity-std", type=float, default=16.0)
    parser.add_argument("--n-velocity-nodes", type=int, default=7)
    parser.add_argument("--K", type=int, default=K_DEFAULT)
    parser.add_argument("--basis-dt-us", type=float, default=BASIS_DT_US_DEFAULT)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument(
        "--far-out",
        type=Path,
        default=HERE / "_cache" / "v3b_far_frequency_scan_k24_basis10.npz",
    )
    parser.add_argument(
        "--velocity-out",
        type=Path,
        default=HERE / "_cache" / "v3b_velocity_averaged_scan_k24_basis10.npz",
    )
    args = parser.parse_args(argv)

    if args.mode in ("far", "both"):
        save_far_scan(args)
    if args.mode in ("velocity", "both"):
        save_velocity_scan(args)


if __name__ == "__main__":
    main()
