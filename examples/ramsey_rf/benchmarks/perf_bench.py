"""Run every variant against the cached reference and emit PERF_REPORT.md.

Each variant is a `Variant` dataclass with a callable that returns a
(Psi_final, elapsed_s) pair for the reference single-trajectory configuration,
plus an optional 41-point omega_rf scan. Accuracy metrics are computed against
the cached Jmax=6 truth from reference.py.

Run as:

    .venv/Scripts/python.exe examples/ramsey_rf/benchmarks/perf_bench.py

Add `--variants v1 v2` to filter. `--scan` enables the 41-point scan benchmark
(adds ~10–60 min depending on n_workers and variant).

The full-basis V4 Krylov variant is registered but excluded from the default
variant set because it is much slower than V1 for this Hamiltonian.
The full-basis V7 CuPy variant is also excluded from the default set because
it requires an optional CUDA/CuPy runtime.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cloudpickle
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centrex_tlf.hamiltonian import generate_uncoupled_hamiltonian_X  # noqa: E402

from ramsey_rf import (  # noqa: E402
    make_uniform_tracking_grid,
    propagate_midpoint_krylov_hybrid,
    propagate_midpoint_tracked_decomposed,
    ScanSpec,
    run_scan,
    select_subspace_by_overlap,
    track_subspace_bases,
)
from ramsey_rf.propagator_truncated import (  # noqa: E402
    propagate_midpoint_truncated_decomposed,
)

from benchmarks.reference import (  # noqa: E402
    E_HIGH, E_LOW, F_RES, ReferenceResult, build_reference_config, get_reference,
)


# -------- Variant container ---------------------------------------------------
@dataclass
class VariantResult:
    name: str
    description: str
    elapsed_traj_s: float
    fidelity: float                  # |<ref|test>|² — phase-insensitive overlap
    norm_test: float                 # ‖test_Psi_final‖² — should be ≤ 1
    max_abs_err_psi: float           # after global-phase alignment
    max_abs_err_pop: float           # population error (phase-insensitive)
    n_steps: int
    elapsed_scan_s: Optional[float] = None
    n_scan_points: Optional[int] = None
    n_workers: Optional[int] = None
    notes: str = ""


@dataclass
class Variant:
    name: str
    description: str
    # Returns (Psi_final (N, K), elapsed_seconds, n_steps).
    run_traj: Callable[[], tuple[np.ndarray, float, int]]
    # Returns (survival_array (n_points,), elapsed_seconds, n_workers_used).
    # None means the variant does not support scans (the benchmark will skip it).
    run_scan: Optional[Callable[[int], tuple[np.ndarray, float, int]]] = None


# -------- V1: baseline (full propagator, segmented grid, sequential scan) ----
def _v1_traj():
    """V1 baseline trajectory. Reuses the cached reference (10.68 min one-shot)
    rather than re-running, since V1 == reference by definition. To force a
    re-run, delete the cache file."""
    ref = get_reference(force=False)
    return ref.Psi_final, ref.elapsed_s, ref.n_steps


def _v1_scan(n_workers_unused: int):
    # n_workers ignored by V1 (sequential)
    cfg, _Psi0, _QN, _H_func = build_reference_config()
    fringe_Hz = 184.0 / 2.5
    n_points = 41
    freqs = F_RES + np.linspace(-2 * fringe_Hz, +2 * fringe_Hz, n_points)
    spec = ScanSpec(axis="omega_rf", values=2 * np.pi * freqs)
    t0 = time.perf_counter()
    out = run_scan(cfg, spec, n_workers=1)
    return out.survival_weighted, time.perf_counter() - t0, 1


V1 = Variant("v1", "Baseline (full propagator, segmented grid, sequential)",
             _v1_traj, _v1_scan)


# -------- V2: process-parallel scan (same propagator, scan parallelized) -----
def _v2_traj():
    # Single trajectory has nothing to parallelize → identical to V1
    return _v1_traj()


def _v2_scan(n_workers: int):
    cfg, _Psi0, _QN, _H_func = build_reference_config()
    fringe_Hz = 184.0 / 2.5
    n_points = 41
    freqs = F_RES + np.linspace(-2 * fringe_Hz, +2 * fringe_Hz, n_points)
    spec = ScanSpec(axis="omega_rf", values=2 * np.pi * freqs)
    t0 = time.perf_counter()
    out = run_scan(cfg, spec, n_workers=n_workers)
    return out.survival_weighted, time.perf_counter() - t0, n_workers


V2 = Variant("v2", "Process-parallel scan (cloudpickle, BLAS=1 per worker)",
             _v2_traj, _v2_scan)


# -------- V3a: static truncated propagator -----------------------------------
def _make_v3a_runner(K: int):
    """Factory: build the V3a Variant with subspace size K.

    Truncation strategy: build T at the HIGH-field plateau (where the molecule
    spends most of the trajectory), picking the K eigenstates with largest
    summed overlap to the bare J=1 sublevels. This spans the polarized J=1
    dressed manifold at high field, including its bare J=0/2/3 Stark-mixing
    admixtures. Picking T at low field instead spans only the (essentially
    pure) bare J=1 character and discards the dynamics' bare J=0/2/3 channels.
    """
    def _v3a_traj() -> tuple[np.ndarray, float, int]:
        cfg, Psi0, QN, H_func = build_reference_config()
        # H at the high-field plateau for subspace selection (R = origin)
        H_high = H_func(cfg.fields.E_dc(np.zeros(3)),
                        (cfg.fields.B_dc(np.zeros(3))
                         if cfg.fields.B_dc is not None else np.zeros(3)))
        bare_j1_idx = np.array([i for i, bs in enumerate(QN) if bs.J == 1])
        bare_j1_targets = np.zeros((len(QN), bare_j1_idx.size), dtype=np.complex128)
        for col, i in enumerate(bare_j1_idx):
            bare_j1_targets[i, col] = 1.0
        T, _idx = select_subspace_by_overlap(H_high, K=K,
                                              target_vectors=bare_j1_targets)

        # Decomposed components: H = 2π · (Hff + Σ E_α·HS_α + Σ B_α·HZ_α).
        # Use raw centrex_tlf sub-matrices (Hz units; multiply by 2π in coeffs).
        Hraw = generate_uncoupled_hamiltonian_X(QN)
        TWOPI = 2.0 * np.pi
        components = [
            Hraw.Hff,  # coefficient = 2π
            Hraw.HSx,  # coefficient = 2π · E_x
            Hraw.HSy,  # coefficient = 2π · E_y
            Hraw.HSz,  # coefficient = 2π · E_z
            Hraw.HZx,  # coefficient = 2π · B_x
            Hraw.HZy,
            Hraw.HZz,
        ]
        traj = cfg.trajectory
        fields = cfg.fields

        def coeffs_at_t(t: float) -> np.ndarray:
            R = traj(t)
            E = fields.E_total(R, t)
            B = fields.B_total(R, t)
            return TWOPI * np.array([1.0, E[0], E[1], E[2], B[0], B[1], B[2]])

        n_steps = cfg.t_grid.size - 1
        t0 = time.perf_counter()
        out = propagate_midpoint_truncated_decomposed(
            Psi0, cfg.t_grid, components, coeffs_at_t, T, store_norm=False,
        )
        elapsed = time.perf_counter() - t0
        return out.Psi_final, elapsed, n_steps

    def _v3a_scan(n_workers: int):
        # V3a scan reuses sequential run_scan but each point uses the truncated
        # propagator. For Phase A we keep this simple — V3a + V2 (combined)
        # would need the simulator to accept a "propagator factory"; not done
        # in V3a alone. Mark as unsupported for now.
        return None  # signals caller to skip

    return Variant(f"v3a_K{K}",
                   f"Static truncation with K={K} (subspace by max overlap to Psi0)",
                   _v3a_traj, run_scan=None)


V3a_K16 = _make_v3a_runner(K=16)
V3a_K24 = _make_v3a_runner(K=24)
V3a_K32 = _make_v3a_runner(K=32)
V3a_K48 = _make_v3a_runner(K=48)


# -------- V3b: adiabatic-tracked truncated propagator ------------------------
V3B_BASIS_DT = 50e-6


def _raw_components(QN):
    Hraw = generate_uncoupled_hamiltonian_X(QN)
    return [Hraw.Hff, Hraw.HSx, Hraw.HSy, Hraw.HSz,
            Hraw.HZx, Hraw.HZy, Hraw.HZz]


def _high_field_j1_subspace(QN, H_func, K: int) -> np.ndarray:
    H_high = H_func(np.array([0.0, 0.0, E_HIGH]), np.zeros(3))
    bare_j1_idx = np.array([i for i, bs in enumerate(QN) if bs.J == 1])
    bare_j1_targets = np.zeros((len(QN), bare_j1_idx.size), dtype=np.complex128)
    for col, i in enumerate(bare_j1_idx):
        bare_j1_targets[i, col] = 1.0
    T_high, _idx = select_subspace_by_overlap(
        H_high, K=K, target_vectors=bare_j1_targets
    )
    return T_high


def _track_subspace_to_start(T_high: np.ndarray, H_func) -> np.ndarray:
    e_path = np.linspace(E_HIGH, E_LOW, 81)

    def H_at_e(e_z: float) -> np.ndarray:
        return H_func(np.array([0.0, 0.0, e_z]), np.zeros(3))

    bases = track_subspace_bases(e_path, H_at_e, T_high)
    return bases[-1]


def _make_v3b_runner(K: int, basis_dt: float = V3B_BASIS_DT):
    """Factory: V3b tracked-basis Variant with subspace size K."""

    def _v3b_traj() -> tuple[np.ndarray, float, int]:
        cfg, Psi0, QN, H_func = build_reference_config()
        T_high = _high_field_j1_subspace(QN, H_func, K)
        T_start = _track_subspace_to_start(T_high, H_func)

        traj = cfg.trajectory
        fields = cfg.fields

        def H_basis_at_t(t: float) -> np.ndarray:
            R = traj(t)
            B = (fields.B_dc(R) if fields.B_dc is not None else np.zeros(3))
            return H_func(fields.E_dc(R), B)

        track_times = make_uniform_tracking_grid(
            float(cfg.t_grid[0]), float(cfg.t_grid[-1]), basis_dt
        )
        n_steps = cfg.t_grid.size - 1

        t0 = time.perf_counter()
        tracked_bases = track_subspace_bases(track_times, H_basis_at_t, T_start)

        TWOPI = 2.0 * np.pi

        def coeffs_at_t(t: float) -> np.ndarray:
            R = traj(t)
            E = fields.E_total(R, t)
            B = fields.B_total(R, t)
            return TWOPI * np.array([1.0, E[0], E[1], E[2], B[0], B[1], B[2]])

        out = propagate_midpoint_tracked_decomposed(
            Psi0,
            cfg.t_grid,
            _raw_components(QN),
            coeffs_at_t,
            track_times,
            tracked_bases,
            store_norm=False,
        )
        elapsed = time.perf_counter() - t0
        return out.Psi_final, elapsed, n_steps

    return Variant(
        f"v3b_K{K}",
        f"Tracked truncation with K={K} (basis_dt={basis_dt * 1e6:.0f} us)",
        _v3b_traj,
        run_scan=None,
    )


V3b_K24 = _make_v3b_runner(K=24)
V3b_K32 = _make_v3b_runner(K=32)
V3b_K48 = _make_v3b_runner(K=48)


# -------- V3a + V2 combined (parallel scan of truncated trajectories) -------
# Worker for the multiprocessing pool. Top-level so it pickles cleanly.
def _v3a_scan_worker_init(cfg_blob: bytes, K_arg: int) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    import cloudpickle as _cp  # noqa: E402  (deferred so children re-import)
    global _CFG, _PSI0, _QN, _H_FUNC, _T, _COMPONENTS, _K
    _CFG = _cp.loads(cfg_blob)
    _K = K_arg

    from ramsey_rf import build_basis as _bb, build_H_func as _bhf  # noqa: E402
    _QN = _bb(_CFG.Jmax)
    _H_FUNC = _bhf(_QN)

    H_high = _H_FUNC(_CFG.fields.E_dc(np.zeros(3)),
                     (_CFG.fields.B_dc(np.zeros(3))
                      if _CFG.fields.B_dc is not None else np.zeros(3)))
    bare_j1_targets = np.zeros((len(_QN),
                                sum(1 for bs in _QN if bs.J == 1)),
                               dtype=np.complex128)
    col = 0
    for i, bs in enumerate(_QN):
        if bs.J == 1:
            bare_j1_targets[i, col] = 1.0
            col += 1
    _T, _ = select_subspace_by_overlap(H_high, K=_K,
                                        target_vectors=bare_j1_targets)
    Hraw = generate_uncoupled_hamiltonian_X(_QN)
    _COMPONENTS = [Hraw.Hff, Hraw.HSx, Hraw.HSy, Hraw.HSz,
                   Hraw.HZx, Hraw.HZy, Hraw.HZz]
    # Pre-set Psi0 from the SAME logic as build_reference_config — but Psi0
    # is fixed across scan points, so we can compute it once per worker.
    from centrex_tlf.states import UncoupledBasisState as _UBS, ElectronicState as _ES
    from ramsey_rf import adiabatic_dressed_initial_states as _ad
    target = _UBS(J=1, mJ=-1, I1=0.5, m1=-0.5, I2=0.5, m2=-0.5,
                  Omega=0, P=-1, electronic_state=_ES.X)
    _PSI0, _, _ = _ad(_H_FUNC, E_init=(0.0, 0.0, 2e3),
                       E_target=(0.0, 0.0, 30e3), QN=_QN,
                       targets=[target], n_steps=80)


def _v3a_scan_worker_run(payload: tuple) -> dict:
    """One scan point: mutate the omega in cfg, build the trajectory + grid,
    run the truncated decomposed propagator, return the (Psi_final, fidelity)."""
    import copy as _copy
    omega_value = payload[0]
    cfg_i = _copy.deepcopy(_CFG)
    # Apply omega_rf to all RF regions
    for region in cfg_i.fields.rf_regions:
        region.omega = float(omega_value)
    for region in cfg_i.fields.rf_regions_B:
        region.omega = float(omega_value)
    # Trajectory grid was pre-built in cfg.t_grid using the original omega;
    # the segmented grid only depends on RF envelope (not omega), so reuse.

    TWOPI = 2.0 * np.pi
    traj = cfg_i.trajectory
    fields = cfg_i.fields
    def coeffs_at_t(t: float) -> np.ndarray:
        R = traj(t)
        E = fields.E_total(R, t)
        B = fields.B_total(R, t)
        return TWOPI * np.array([1.0, E[0], E[1], E[2], B[0], B[1], B[2]])

    out = propagate_midpoint_truncated_decomposed(
        _PSI0, cfg_i.t_grid, _COMPONENTS, coeffs_at_t, _T, store_norm=False,
    )
    # Compute survival = |<Psi0|Psi_final>|^2 (return-prob to start)
    surv = float(abs(np.vdot(_PSI0[:, 0], out.Psi_final[:, 0])) ** 2)
    return {
        "survival_weighted": surv,
        "Psi_final": out.Psi_final,
    }


def _make_v3a_parallel_runner(K: int, n_workers: int):
    """Factory: V3a + V2 combined — parallel scan of truncated trajectories."""
    base = _make_v3a_runner(K)

    def _v3a_parallel_scan(_n_workers_arg: int) -> tuple[np.ndarray, float, int]:
        cfg, _Psi0, _QN, _H_func = build_reference_config()
        fringe_Hz = 184.0 / 2.5
        n_points = 41
        freqs = F_RES + np.linspace(-2 * fringe_Hz, +2 * fringe_Hz, n_points)
        omegas = 2 * np.pi * freqs
        cfg_blob = cloudpickle.dumps(cfg)
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()
        with ctx.Pool(processes=n_workers,
                       initializer=_v3a_scan_worker_init,
                       initargs=(cfg_blob, K)) as pool:
            payload = [(float(o),) for o in omegas]
            results = list(pool.imap(_v3a_scan_worker_run, payload))
        elapsed = time.perf_counter() - t0
        surv = np.array([r["survival_weighted"] for r in results])
        return surv, elapsed, n_workers

    return Variant(
        f"v3a_K{K}_par{n_workers}",
        f"V3a (K={K}) + V2 (parallel scan, n_workers={n_workers})",
        base.run_traj, run_scan=_v3a_parallel_scan,
    )


V3a_K48_par8 = _make_v3a_parallel_runner(K=48, n_workers=8)
V3a_K24_par8 = _make_v3a_parallel_runner(K=24, n_workers=8)


# -------- V3b + V2 combined (parallel scan of tracked trajectories) ----------
def _v3b_scan_worker_init(cfg_blob: bytes, K_arg: int, basis_dt: float) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    import cloudpickle as _cp  # noqa: E402

    global _V3B_CFG, _V3B_PSI0, _V3B_QN, _V3B_H_FUNC
    global _V3B_COMPONENTS, _V3B_TRACK_TIMES, _V3B_TRACKED_BASES

    _V3B_CFG = _cp.loads(cfg_blob)

    from ramsey_rf import build_basis as _bb, build_H_func as _bhf  # noqa: E402

    _V3B_QN = _bb(_V3B_CFG.Jmax)
    _V3B_H_FUNC = _bhf(_V3B_QN)
    if _V3B_CFG.initial_psi0 is None:
        raise ValueError("V3b scan worker requires cfg.initial_psi0")
    _V3B_PSI0 = np.asarray(_V3B_CFG.initial_psi0, dtype=np.complex128)
    _V3B_COMPONENTS = _raw_components(_V3B_QN)

    T_high = _high_field_j1_subspace(_V3B_QN, _V3B_H_FUNC, K_arg)
    T_start = _track_subspace_to_start(T_high, _V3B_H_FUNC)

    traj = _V3B_CFG.trajectory
    fields = _V3B_CFG.fields

    def H_basis_at_t(t: float) -> np.ndarray:
        R = traj(t)
        B = fields.B_dc(R) if fields.B_dc is not None else np.zeros(3)
        return _V3B_H_FUNC(fields.E_dc(R), B)

    _V3B_TRACK_TIMES = make_uniform_tracking_grid(
        float(_V3B_CFG.t_grid[0]), float(_V3B_CFG.t_grid[-1]), basis_dt
    )
    _V3B_TRACKED_BASES = track_subspace_bases(
        _V3B_TRACK_TIMES, H_basis_at_t, T_start
    )


def _v3b_scan_worker_run(payload: tuple) -> dict:
    import copy as _copy

    omega_value = payload[0]
    cfg_i = _copy.deepcopy(_V3B_CFG)
    for region in cfg_i.fields.rf_regions:
        region.omega = float(omega_value)
    for region in cfg_i.fields.rf_regions_B:
        region.omega = float(omega_value)

    TWOPI = 2.0 * np.pi
    traj = cfg_i.trajectory
    fields = cfg_i.fields

    def coeffs_at_t(t: float) -> np.ndarray:
        R = traj(t)
        E = fields.E_total(R, t)
        B = fields.B_total(R, t)
        return TWOPI * np.array([1.0, E[0], E[1], E[2], B[0], B[1], B[2]])

    out = propagate_midpoint_tracked_decomposed(
        _V3B_PSI0,
        cfg_i.t_grid,
        _V3B_COMPONENTS,
        coeffs_at_t,
        _V3B_TRACK_TIMES,
        _V3B_TRACKED_BASES,
        store_norm=False,
    )
    surv = float(abs(np.vdot(_V3B_PSI0[:, 0], out.Psi_final[:, 0])) ** 2)
    return {"survival_weighted": surv, "Psi_final": out.Psi_final}


def _make_v3b_parallel_runner(
    K: int,
    n_workers: int,
    basis_dt: float = V3B_BASIS_DT,
):
    base = _make_v3b_runner(K, basis_dt=basis_dt)

    def _v3b_parallel_scan(_n_workers_arg: int) -> tuple[np.ndarray, float, int]:
        cfg, _Psi0, _QN, _H_func = build_reference_config()
        fringe_Hz = 184.0 / 2.5
        n_points = 41
        freqs = F_RES + np.linspace(-2 * fringe_Hz, +2 * fringe_Hz, n_points)
        omegas = 2 * np.pi * freqs
        cfg_blob = cloudpickle.dumps(cfg)
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()
        with ctx.Pool(
            processes=n_workers,
            initializer=_v3b_scan_worker_init,
            initargs=(cfg_blob, K, basis_dt),
        ) as pool:
            payload = [(float(o),) for o in omegas]
            results = list(pool.imap(_v3b_scan_worker_run, payload))
        elapsed = time.perf_counter() - t0
        surv = np.array([r["survival_weighted"] for r in results])
        return surv, elapsed, n_workers

    return Variant(
        f"v3b_K{K}_par{n_workers}",
        f"V3b (K={K}, basis_dt={basis_dt * 1e6:.0f} us) + V2 "
        f"(parallel scan, n_workers={n_workers})",
        base.run_traj,
        run_scan=_v3b_parallel_scan,
    )


V3b_K24_par8 = _make_v3b_parallel_runner(K=24, n_workers=8)
V3b_K32_par8 = _make_v3b_parallel_runner(K=32, n_workers=8)


# -------- V4: sparse Krylov on active steps, exact eigh on inactive steps -----
def _v4_traj() -> tuple[np.ndarray, float, int]:
    cfg, Psi0, QN, _H_func = build_reference_config()
    traj = cfg.trajectory
    fields = cfg.fields
    TWOPI = 2.0 * np.pi

    def coeffs_at_t(t: float) -> np.ndarray:
        R = traj(t)
        E = fields.E_total(R, t)
        B = fields.B_total(R, t)
        return TWOPI * np.array([1.0, E[0], E[1], E[2], B[0], B[1], B[2]])

    n_steps = cfg.t_grid.size - 1
    t0 = time.perf_counter()
    out = propagate_midpoint_krylov_hybrid(
        Psi0,
        cfg.t_grid,
        _raw_components(QN),
        coeffs_at_t,
        store_norm=False,
    )
    elapsed = time.perf_counter() - t0
    print(
        "  V4 steps: "
        f"krylov={out.stats.n_krylov_steps}, eigh={out.stats.n_eigh_steps}, "
        f"krylov_dt_max={out.stats.krylov_dt_max * 1e9:.1f} ns"
    )
    return out.Psi_final, elapsed, n_steps


V4 = Variant(
    "v4_krylov",
    "Sparse Krylov active steps + exact eigh inactive steps (full basis)",
    _v4_traj,
    run_scan=None,
)


# -------- V5: numba JIT rotate/apply inner loop ------------------------------
def _v5_traj() -> tuple[np.ndarray, float, int]:
    from ramsey_rf.propagator_numba import propagate_midpoint_numba

    cfg, Psi0, _QN, H_func = build_reference_config()
    traj = cfg.trajectory
    fields = cfg.fields

    def H_at_t(t: float) -> np.ndarray:
        R = traj(t)
        return H_func(fields.E_total(R, t), fields.B_total(R, t))

    n_steps = cfg.t_grid.size - 1
    t0 = time.perf_counter()
    out = propagate_midpoint_numba(
        Psi0,
        cfg.t_grid,
        H_at_t,
        store_norm=False,
    )
    elapsed = time.perf_counter() - t0
    return out.Psi_final, elapsed, n_steps


V5 = Variant(
    "v5_numba",
    "Dense full-basis propagator with Numba rotate/apply kernel",
    _v5_traj,
    run_scan=None,
)


# -------- V7: cupy GPU dense full-basis eigensolve ---------------------------
def _run_cupy_cfg(
    cfg,
    Psi0: np.ndarray,
    H_func,
    *,
    warm_up: bool = True,
) -> tuple[np.ndarray, str]:
    from ramsey_rf.propagator_cupy import propagate_midpoint_cupy

    traj = cfg.trajectory
    fields = cfg.fields

    def H_at_t(t: float) -> np.ndarray:
        R = traj(t)
        return H_func(fields.E_total(R, t), fields.B_total(R, t))

    out = propagate_midpoint_cupy(
        Psi0,
        cfg.t_grid,
        H_at_t,
        warm_up=warm_up,
        store_norm=False,
    )
    device = out.device_info
    device_note = (
        f"device={device.name}, id={device.device_id}, "
        f"cuda_runtime={device.runtime_version}"
    )
    return out.Psi_final, device_note


def _v7_traj() -> tuple[np.ndarray, float, int]:
    cfg, Psi0, _QN, H_func = build_reference_config()
    n_steps = cfg.t_grid.size - 1
    t0 = time.perf_counter()
    psi_final, device_note = _run_cupy_cfg(cfg, Psi0, H_func)
    elapsed = time.perf_counter() - t0
    print(f"  V7 CuPy: {device_note}")
    return psi_final, elapsed, n_steps


def _v7_scan_seq(n_workers_unused: int) -> tuple[np.ndarray, float, int]:
    import copy as _copy

    cfg, Psi0, _QN, H_func = build_reference_config()
    fringe_Hz = 184.0 / 2.5
    n_points = 41
    freqs = F_RES + np.linspace(-2 * fringe_Hz, +2 * fringe_Hz, n_points)
    omegas = 2 * np.pi * freqs
    survival = np.empty(n_points, dtype=np.float64)
    device_note = None
    t0 = time.perf_counter()
    for i, omega in enumerate(omegas):
        cfg_i = _copy.deepcopy(cfg)
        for region in cfg_i.fields.rf_regions:
            region.omega = float(omega)
        for region in cfg_i.fields.rf_regions_B:
            region.omega = float(omega)
        psi_final, device_note = _run_cupy_cfg(cfg_i, Psi0, H_func, warm_up=(i == 0))
        survival[i] = float(abs(np.vdot(Psi0[:, 0], psi_final[:, 0])) ** 2)
        print(
            f"  V7 scan point {i + 1}/{n_points}: "
            f"freq={freqs[i]:.3f} Hz, survival={survival[i]:.6f}"
        )
    elapsed = time.perf_counter() - t0
    if device_note is not None:
        print(f"  V7 CuPy scan: {device_note}")
    return survival, elapsed, 1


V7 = Variant(
    "v7_cupy",
    "Dense full-basis propagator with CuPy GPU eigh",
    _v7_traj,
    run_scan=None,
)

V7_SCAN_SEQ = Variant(
    "v7_cupy_scan_seq",
    "Sequential 41-point omega_rf scan with CuPy GPU eigh",
    _v7_traj,
    run_scan=_v7_scan_seq,
)


# -------- Variant registry ----------------------------------------------------
DEFAULT_VARIANTS: list[Variant] = [
    V1, V2,
    V3a_K16, V3a_K24, V3a_K32, V3a_K48,
    V3b_K24, V3b_K32, V3b_K48,
    V3b_K24_par8, V3b_K32_par8,
    V3a_K48_par8, V3a_K24_par8,
]
SLOW_VARIANTS: list[Variant] = [V4, V5, V7, V7_SCAN_SEQ]
ALL_VARIANTS: list[Variant] = DEFAULT_VARIANTS + SLOW_VARIANTS


# -------- Bench main ---------------------------------------------------------
def _accuracy(psi_test: np.ndarray, psi_ref: np.ndarray) -> tuple[float, float, float, float]:
    """Return (fidelity, norm_test, max |Δamp_aligned|, max |Δ|amp|²|).

    fidelity = |<ref|test>|² — phase-insensitive overlap; 1.0 = perfect match.
    norm_test = ‖test‖² — should be ≤ 1; less than 1 means population leaked
        outside the truncated subspace (V3a) or amplitude was lost (V6).
    max|Δamp_aligned|: amp diff after rotating test to align global phase with ref.
    max|Δ|amp|²|: population diff (phase-insensitive, the tightest single-component
        check).
    """
    overlap = np.vdot(psi_ref.flatten(), psi_test.flatten())
    fidelity = float(abs(overlap) ** 2 / max(1e-300, np.vdot(psi_ref.flatten(), psi_ref.flatten()).real))
    norm_test = float(np.vdot(psi_test.flatten(), psi_test.flatten()).real)
    if abs(overlap) > 1e-12:
        psi_test_aligned = psi_test * (np.conj(overlap) / abs(overlap))
    else:
        psi_test_aligned = psi_test
    amp_err = float(np.max(np.abs(psi_test_aligned - psi_ref)))
    pop_err = float(np.max(np.abs(np.abs(psi_test_aligned) ** 2
                                  - np.abs(psi_ref) ** 2)))
    return fidelity, norm_test, amp_err, pop_err


def run_variant(variant: Variant, ref: ReferenceResult, *, do_scan: bool,
                n_workers: int) -> VariantResult:
    print(f"\n--- Running variant: {variant.name} ({variant.description}) ---")
    psi, elapsed, n_steps = variant.run_traj()
    fidelity, norm_test, amp_err, pop_err = _accuracy(psi, ref.Psi_final)
    print(f"  trajectory: {elapsed / 60:.2f} min ({elapsed:.1f} s)")
    print(f"  fidelity = {fidelity:.6f}, ‖ψ_test‖² = {norm_test:.6f}")
    print(f"  max |Δψ_aligned| = {amp_err:.3e}, max |Δ|ψ|²| = {pop_err:.3e}")

    elapsed_scan = None
    n_scan_points = None
    n_workers_used = None
    if do_scan and variant.run_scan is not None:
        scan_out = variant.run_scan(n_workers)
        if scan_out is not None:
            surv_arr, elapsed_scan, n_workers_used = scan_out
            n_scan_points = len(surv_arr)
            print(f"  scan: {elapsed_scan / 60:.2f} min ({n_scan_points} points, "
                  f"n_workers={n_workers_used})")

    return VariantResult(
        name=variant.name, description=variant.description,
        elapsed_traj_s=elapsed, fidelity=fidelity, norm_test=norm_test,
        max_abs_err_psi=amp_err, max_abs_err_pop=pop_err, n_steps=n_steps,
        elapsed_scan_s=elapsed_scan, n_scan_points=n_scan_points,
        n_workers=n_workers_used,
    )


def write_report(results: list[VariantResult], ref: ReferenceResult,
                 path: Path) -> None:
    lines: list[str] = []
    lines.append("# RF Ramsey simulator — perf investigation report (Phase A)\n")
    lines.append(f"Reference: Jmax={ref.Jmax}, dt_fine={ref.dt_fine * 1e6:.2f} µs, "
                 f"segmented grid n_steps={ref.n_steps}, "
                 f"baseline elapsed = {ref.elapsed_s / 60:.2f} min\n")
    lines.append(f"Reference survival = {ref.survival:.6f}, "
                 f"per-J = {dict(zip(ref.J_values, ref.per_j[0]))}\n")
    lines.append("\n## Variant comparison\n")
    lines.append("| Variant | Description | Trajectory time | Speedup vs V1 | "
                 "Fidelity | ‖ψ‖² | max\\|Δ\\|ψ\\|²\\| | Scan (41-pt) | n_workers |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    v1_time = next((r.elapsed_traj_s for r in results if r.name == "v1"), ref.elapsed_s)
    for r in results:
        traj_min = f"{r.elapsed_traj_s / 60:.2f} min"
        speedup = f"{v1_time / r.elapsed_traj_s:.2f}×" if v1_time else "—"
        scan_str = (f"{r.elapsed_scan_s / 60:.2f} min"
                    if r.elapsed_scan_s is not None else "—")
        nw_str = str(r.n_workers) if r.n_workers is not None else "—"
        lines.append(f"| `{r.name}` | {r.description} | {traj_min} | {speedup} | "
                     f"{r.fidelity:.6f} | {r.norm_test:.4f} | "
                     f"{r.max_abs_err_pop:.2e} | "
                     f"{scan_str} | {nw_str} |\n")

    lines.append("\n## Notes\n")
    lines.append("- **Fidelity** = |⟨ref|test⟩|² is the phase-insensitive "
                 "overlap with the truth Psi_final. 1.0 means a perfect match "
                 "(up to global phase); < 1 means population went somewhere "
                 "the reference didn't.\n")
    lines.append("- **‖ψ‖²** is the norm of the test Psi_final. < 1 indicates "
                 "the variant lost amplitude (V3a leaks outside the truncated "
                 "subspace; V6 may lose to floating-point error).\n")
    lines.append("- **max |Δ|ψ|²|** is the largest single-component population "
                 "error vs the reference (phase-insensitive).\n")
    lines.append("- Trajectory speedup is per-trajectory wall-clock against V1.\n")
    lines.append("- Scan timings (41 points, omega_rf scan) include process-spawn "
                 "+ cloudpickle overhead (~5 s constant per scan).\n")
    lines.append("- All variants run on this PC (8+ cores, no GPU). GPU "
                 "variants (V7) are tested separately on the CUDA box.\n")

    path.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport written to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Filter by variant name (e.g. v1 v3a_K32). Default: all.")
    parser.add_argument("--scan", action="store_true",
                        help="Also benchmark the 41-point omega_rf scan.")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Process count for variants that support parallelism.")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "PERF_REPORT.md"),
                        help="Path to write PERF_REPORT.md.")
    args = parser.parse_args()

    print("Loading reference...")
    ref = get_reference(force=False)
    print(f"  reference loaded, survival = {ref.survival:.6f}, "
          f"elapsed at baseline = {ref.elapsed_s / 60:.2f} min")

    selected = DEFAULT_VARIANTS
    if args.variants:
        names = set(args.variants)
        selected = [v for v in ALL_VARIANTS if v.name in names]
        if not selected:
            print(f"No variants match {args.variants}; available: "
                  f"{[v.name for v in ALL_VARIANTS]}")
            sys.exit(1)

    results: list[VariantResult] = []
    for variant in selected:
        results.append(run_variant(variant, ref, do_scan=args.scan,
                                   n_workers=args.n_workers))

    write_report(results, ref, Path(args.report))


if __name__ == "__main__":
    main()
