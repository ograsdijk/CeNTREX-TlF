"""Sweep V3b tracked-basis timestep convergence against the 50 ns reference.

This benchmark is intentionally separate from perf_bench.py because it expands
one V3b implementation into a grid of K and basis_dt settings, then optionally
builds five exact V7/CuPy scan anchors and runs 41-point scan confirmations for
the best settings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cloudpickle
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centrex_tlf.hamiltonian import generate_uncoupled_hamiltonian_X  # noqa: E402

from ramsey_rf import (  # noqa: E402
    make_uniform_tracking_grid,
    propagate_midpoint_tracked_decomposed,
    select_subspace_by_overlap,
    track_subspace_bases,
)

from benchmarks.reference import (  # noqa: E402
    E_HIGH,
    E_LOW,
    F_RES,
    build_reference_config,
    get_reference,
)


DEFAULT_K = [24, 32, 48]
DEFAULT_BASIS_DT_US = [100.0, 50.0, 25.0, 10.0, 5.0, 2.5, 1.0]
FRINGE_HZ = 184.0 / 2.5
ANCHOR_OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
SCAN_POINTS = 41
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class TrajectoryRow:
    K: int
    basis_dt_us: float
    n_tracked_bases: int
    elapsed_total_s: float
    subspace_setup_s: float
    tracking_setup_s: float
    propagation_s: float
    fidelity: float
    norm2: float
    max_amp_err: float
    max_pop_err: float
    survival: float
    survival_abs_err: float
    n_steps: int
    converged_vs_next_half: str = ""


@dataclass(frozen=True)
class ScanRow:
    K: int
    basis_dt_us: float
    n_workers: int
    elapsed_scan_s: float
    n_points: int
    anchor_max_abs_survival_err: str
    anchor_rms_survival_err: str
    survival_csv: str


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

    bases = track_subspace_bases(e_path, H_at_e, T_high)
    return bases[-1]


def _set_rf_omega(cfg, omega: float) -> None:
    for region in cfg.fields.rf_regions:
        region.omega = float(omega)
    for region in cfg.fields.rf_regions_B:
        region.omega = float(omega)


def _coeffs_builder(cfg):
    traj = cfg.trajectory
    fields = cfg.fields

    def coeffs_at_t(t: float) -> np.ndarray:
        R = traj(t)
        E = fields.E_total(R, t)
        B = fields.B_total(R, t)
        return TWOPI * np.array([1.0, E[0], E[1], E[2], B[0], B[1], B[2]])

    return coeffs_at_t


def _accuracy(
    psi_test: np.ndarray,
    psi_ref: np.ndarray,
    Psi0: np.ndarray,
    ref_survival: float,
) -> tuple[float, float, float, float, float, float]:
    overlap = np.vdot(psi_ref.flatten(), psi_test.flatten())
    ref_norm = float(np.vdot(psi_ref.flatten(), psi_ref.flatten()).real)
    fidelity = float(abs(overlap) ** 2 / max(1e-300, ref_norm))
    norm2 = float(np.vdot(psi_test.flatten(), psi_test.flatten()).real)
    if abs(overlap) > 1e-12:
        psi_test_aligned = psi_test * (np.conj(overlap) / abs(overlap))
    else:
        psi_test_aligned = psi_test
    amp_err = float(np.max(np.abs(psi_test_aligned - psi_ref)))
    pop_err = float(
        np.max(np.abs(np.abs(psi_test_aligned) ** 2 - np.abs(psi_ref) ** 2))
    )
    survival = float(abs(np.vdot(Psi0[:, 0], psi_test[:, 0])) ** 2)
    survival_abs_err = abs(survival - ref_survival)
    return fidelity, norm2, amp_err, pop_err, survival, survival_abs_err


def run_v3b_trajectory(K: int, basis_dt_us: float) -> TrajectoryRow:
    cfg, Psi0, QN, H_func = build_reference_config()
    ref = get_reference(force=False)
    components = _raw_components(QN)
    basis_dt = basis_dt_us * 1e-6
    elapsed0 = time.perf_counter()

    setup0 = time.perf_counter()
    T_high = _high_field_j1_subspace(QN, H_func, K)
    T_start = _track_subspace_to_start(T_high, H_func)
    subspace_setup_s = time.perf_counter() - setup0

    traj = cfg.trajectory
    fields = cfg.fields

    def H_basis_at_t(t: float) -> np.ndarray:
        R = traj(t)
        B = fields.B_dc(R) if fields.B_dc is not None else np.zeros(3)
        return H_func(fields.E_dc(R), B)

    track_times = make_uniform_tracking_grid(
        float(cfg.t_grid[0]), float(cfg.t_grid[-1]), basis_dt
    )
    tracking0 = time.perf_counter()
    tracked_bases = track_subspace_bases(track_times, H_basis_at_t, T_start)
    tracking_setup_s = time.perf_counter() - tracking0

    prop0 = time.perf_counter()
    out = propagate_midpoint_tracked_decomposed(
        Psi0,
        cfg.t_grid,
        components,
        _coeffs_builder(cfg),
        track_times,
        tracked_bases,
        store_norm=False,
    )
    propagation_s = time.perf_counter() - prop0
    elapsed_total_s = time.perf_counter() - elapsed0

    fidelity, norm2, amp_err, pop_err, survival, survival_abs_err = _accuracy(
        out.Psi_final, ref.Psi_final, Psi0, ref.survival
    )
    return TrajectoryRow(
        K=K,
        basis_dt_us=basis_dt_us,
        n_tracked_bases=int(track_times.size),
        elapsed_total_s=elapsed_total_s,
        subspace_setup_s=subspace_setup_s,
        tracking_setup_s=tracking_setup_s,
        propagation_s=propagation_s,
        fidelity=fidelity,
        norm2=norm2,
        max_amp_err=amp_err,
        max_pop_err=pop_err,
        survival=survival,
        survival_abs_err=survival_abs_err,
        n_steps=int(cfg.t_grid.size - 1),
    )


def _row_to_dict(row: TrajectoryRow) -> dict[str, str]:
    return {
        "K": str(row.K),
        "basis_dt_us": f"{row.basis_dt_us:.10g}",
        "n_tracked_bases": str(row.n_tracked_bases),
        "elapsed_total_s": f"{row.elapsed_total_s:.9g}",
        "subspace_setup_s": f"{row.subspace_setup_s:.9g}",
        "tracking_setup_s": f"{row.tracking_setup_s:.9g}",
        "propagation_s": f"{row.propagation_s:.9g}",
        "fidelity": f"{row.fidelity:.12g}",
        "norm2": f"{row.norm2:.12g}",
        "max_amp_err": f"{row.max_amp_err:.12g}",
        "max_pop_err": f"{row.max_pop_err:.12g}",
        "survival": f"{row.survival:.12g}",
        "survival_abs_err": f"{row.survival_abs_err:.12g}",
        "n_steps": str(row.n_steps),
        "converged_vs_next_half": row.converged_vs_next_half,
    }


def _dict_to_row(data: dict[str, str]) -> TrajectoryRow:
    return TrajectoryRow(
        K=int(data["K"]),
        basis_dt_us=float(data["basis_dt_us"]),
        n_tracked_bases=int(data["n_tracked_bases"]),
        elapsed_total_s=float(data["elapsed_total_s"]),
        subspace_setup_s=float(data["subspace_setup_s"]),
        tracking_setup_s=float(data["tracking_setup_s"]),
        propagation_s=float(data["propagation_s"]),
        fidelity=float(data["fidelity"]),
        norm2=float(data["norm2"]),
        max_amp_err=float(data["max_amp_err"]),
        max_pop_err=float(data["max_pop_err"]),
        survival=float(data["survival"]),
        survival_abs_err=float(data["survival_abs_err"]),
        n_steps=int(data["n_steps"]),
        converged_vs_next_half=data.get("converged_vs_next_half", ""),
    )


def annotate_convergence(rows: list[TrajectoryRow]) -> list[TrajectoryRow]:
    annotated: list[TrajectoryRow] = []
    by_k: dict[int, list[TrajectoryRow]] = {}
    for row in rows:
        by_k.setdefault(row.K, []).append(row)

    row_status: dict[tuple[int, float], str] = {}
    for K, group in by_k.items():
        del K
        sorted_group = sorted(group, key=lambda r: r.basis_dt_us, reverse=True)
        for row in sorted_group:
            half = row.basis_dt_us / 2.0
            finer = min(
                group,
                key=lambda candidate: abs(candidate.basis_dt_us - half),
            )
            if abs(finer.basis_dt_us - half) > max(1e-9, half * 1e-6):
                row_status[(row.K, row.basis_dt_us)] = ""
                continue
            pop_ref = max(row.max_pop_err, 1e-300)
            pop_rel_change = abs(finer.max_pop_err - row.max_pop_err) / pop_ref
            is_converged = (
                abs(finer.fidelity - row.fidelity) < 1e-4
                and abs(finer.survival - row.survival) < 1e-5
                and pop_rel_change < 0.10
            )
            row_status[(row.K, row.basis_dt_us)] = "yes" if is_converged else "no"

    for row in rows:
        annotated.append(
            TrajectoryRow(
                **{
                    **row.__dict__,
                    "converged_vs_next_half": row_status.get(
                        (row.K, row.basis_dt_us), ""
                    ),
                }
            )
        )
    return annotated


def write_trajectory_csv(rows: list[TrajectoryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_row_to_dict(rows[0]).keys()) if rows else list(_row_to_dict(
        TrajectoryRow(0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    ).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


def read_trajectory_csv(path: Path) -> list[TrajectoryRow]:
    with path.open(newline="", encoding="utf-8") as f:
        return [_dict_to_row(row) for row in csv.DictReader(f)]


def selected_candidates(rows: Iterable[TrajectoryRow]) -> list[TrajectoryRow]:
    by_k: dict[int, list[TrajectoryRow]] = {}
    for row in rows:
        by_k.setdefault(row.K, []).append(row)
    selected: list[TrajectoryRow] = []
    for K in sorted(by_k):
        best = sorted(
            by_k[K],
            key=lambda row: (
                -row.fidelity,
                row.max_pop_err,
                row.survival_abs_err,
                row.elapsed_total_s,
            ),
        )[0]
        selected.append(best)

    baseline = next(
        (
            row
            for row in rows
            if row.K == 24 and abs(row.basis_dt_us - 50.0) < 1e-9
        ),
        None,
    )
    if baseline is not None and all(
        not (row.K == baseline.K and abs(row.basis_dt_us - baseline.basis_dt_us) < 1e-9)
        for row in selected
    ):
        selected.append(baseline)
    return selected


def _anchor_frequencies() -> np.ndarray:
    return F_RES + ANCHOR_OFFSETS * FRINGE_HZ


def build_v7_anchors(path: Path) -> None:
    from ramsey_rf.propagator_cupy import propagate_midpoint_cupy

    cfg, Psi0, _QN, H_func = build_reference_config()
    freqs = _anchor_frequencies()
    omegas = TWOPI * freqs
    survival = np.empty(freqs.size, dtype=np.float64)
    elapsed_s = np.empty(freqs.size, dtype=np.float64)
    psi_finals: list[np.ndarray] = []
    device_note = ""
    for idx, omega in enumerate(omegas):
        cfg_i = copy.deepcopy(cfg)
        _set_rf_omega(cfg_i, float(omega))
        traj = cfg_i.trajectory
        fields = cfg_i.fields

        def H_at_t(t: float) -> np.ndarray:
            R = traj(t)
            return H_func(fields.E_total(R, t), fields.B_total(R, t))

        t0 = time.perf_counter()
        out = propagate_midpoint_cupy(
            Psi0,
            cfg_i.t_grid,
            H_at_t,
            warm_up=(idx == 0),
            store_norm=False,
        )
        elapsed_s[idx] = time.perf_counter() - t0
        survival[idx] = float(abs(np.vdot(Psi0[:, 0], out.Psi_final[:, 0])) ** 2)
        psi_finals.append(out.Psi_final)
        device_note = (
            f"{out.device_info.name}; runtime={out.device_info.runtime_version}"
        )
        print(
            f"anchor {idx + 1}/{freqs.size}: freq={freqs[idx]:.3f} Hz, "
            f"survival={survival[idx]:.9g}, elapsed={elapsed_s[idx] / 60:.2f} min"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        offsets=ANCHOR_OFFSETS,
        freqs=freqs,
        omegas=omegas,
        survival=survival,
        elapsed_s=elapsed_s,
        psi_finals=np.stack(psi_finals, axis=0),
        device_note=np.array(device_note),
    )
    print(f"wrote V7 anchors to {path}")


def load_anchor_survival(path: Optional[Path]) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["freqs"], data["survival"]


def _scan_frequencies() -> np.ndarray:
    return F_RES + np.linspace(-2.0 * FRINGE_HZ, 2.0 * FRINGE_HZ, SCAN_POINTS)


_SCAN_CFG = None
_SCAN_PSI0 = None
_SCAN_COMPONENTS = None
_SCAN_TRACK_TIMES = None
_SCAN_TRACKED_BASES = None


def _scan_worker_init(cfg_blob: bytes, K: int, basis_dt_us: float) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"

    global _SCAN_CFG, _SCAN_PSI0, _SCAN_COMPONENTS
    global _SCAN_TRACK_TIMES, _SCAN_TRACKED_BASES

    _SCAN_CFG = cloudpickle.loads(cfg_blob)
    from ramsey_rf import build_basis as _build_basis, build_H_func as _build_H_func

    QN = _build_basis(_SCAN_CFG.Jmax)
    H_func = _build_H_func(QN)
    if _SCAN_CFG.initial_psi0 is None:
        raise ValueError("scan worker requires cfg.initial_psi0")
    _SCAN_PSI0 = np.asarray(_SCAN_CFG.initial_psi0, dtype=np.complex128)
    _SCAN_COMPONENTS = _raw_components(QN)

    T_high = _high_field_j1_subspace(QN, H_func, K)
    T_start = _track_subspace_to_start(T_high, H_func)
    traj = _SCAN_CFG.trajectory
    fields = _SCAN_CFG.fields

    def H_basis_at_t(t: float) -> np.ndarray:
        R = traj(t)
        B = fields.B_dc(R) if fields.B_dc is not None else np.zeros(3)
        return H_func(fields.E_dc(R), B)

    _SCAN_TRACK_TIMES = make_uniform_tracking_grid(
        float(_SCAN_CFG.t_grid[0]),
        float(_SCAN_CFG.t_grid[-1]),
        basis_dt_us * 1e-6,
    )
    _SCAN_TRACKED_BASES = track_subspace_bases(
        _SCAN_TRACK_TIMES, H_basis_at_t, T_start
    )


def _scan_worker_run(omega: float) -> float:
    cfg_i = copy.deepcopy(_SCAN_CFG)
    _set_rf_omega(cfg_i, omega)
    out = propagate_midpoint_tracked_decomposed(
        _SCAN_PSI0,
        cfg_i.t_grid,
        _SCAN_COMPONENTS,
        _coeffs_builder(cfg_i),
        _SCAN_TRACK_TIMES,
        _SCAN_TRACKED_BASES,
        store_norm=False,
    )
    return float(abs(np.vdot(_SCAN_PSI0[:, 0], out.Psi_final[:, 0])) ** 2)


def run_scan_candidate(
    row: TrajectoryRow,
    n_workers: int,
    anchor_cache: Optional[Path],
) -> ScanRow:
    cfg, _Psi0, _QN, _H_func = build_reference_config()
    freqs = _scan_frequencies()
    omegas = TWOPI * freqs
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(
        processes=n_workers,
        initializer=_scan_worker_init,
        initargs=(cloudpickle.dumps(cfg), row.K, row.basis_dt_us),
    ) as pool:
        survival = np.array(list(pool.imap(_scan_worker_run, [float(o) for o in omegas])))
    elapsed = time.perf_counter() - t0

    max_anchor_err = ""
    rms_anchor_err = ""
    anchors = load_anchor_survival(anchor_cache)
    if anchors is not None:
        anchor_freqs, anchor_survival = anchors
        scan_at_anchors = np.array(
            [survival[int(np.argmin(np.abs(freqs - freq)))] for freq in anchor_freqs]
        )
        diff = scan_at_anchors - anchor_survival
        max_anchor_err = f"{float(np.max(np.abs(diff))):.12g}"
        rms_anchor_err = f"{float(np.sqrt(np.mean(diff**2))):.12g}"

    print(
        f"scan K={row.K}, basis_dt={row.basis_dt_us:g} us: "
        f"{elapsed / 60:.2f} min, anchor_max_err={max_anchor_err or 'n/a'}"
    )
    return ScanRow(
        K=row.K,
        basis_dt_us=row.basis_dt_us,
        n_workers=n_workers,
        elapsed_scan_s=elapsed,
        n_points=survival.size,
        anchor_max_abs_survival_err=max_anchor_err,
        anchor_rms_survival_err=rms_anchor_err,
        survival_csv=";".join(f"{value:.12g}" for value in survival),
    )


def write_scan_csv(rows: list[ScanRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "K",
        "basis_dt_us",
        "n_workers",
        "elapsed_scan_s",
        "n_points",
        "anchor_max_abs_survival_err",
        "anchor_rms_survival_err",
        "survival_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "K": row.K,
                    "basis_dt_us": f"{row.basis_dt_us:.10g}",
                    "n_workers": row.n_workers,
                    "elapsed_scan_s": f"{row.elapsed_scan_s:.9g}",
                    "n_points": row.n_points,
                    "anchor_max_abs_survival_err": row.anchor_max_abs_survival_err,
                    "anchor_rms_survival_err": row.anchor_rms_survival_err,
                    "survival_csv": row.survival_csv,
                }
            )


def read_scan_csv(path: Path) -> list[ScanRow]:
    if not path.exists():
        return []
    rows: list[ScanRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                ScanRow(
                    K=int(row["K"]),
                    basis_dt_us=float(row["basis_dt_us"]),
                    n_workers=int(row["n_workers"]),
                    elapsed_scan_s=float(row["elapsed_scan_s"]),
                    n_points=int(row["n_points"]),
                    anchor_max_abs_survival_err=row["anchor_max_abs_survival_err"],
                    anchor_rms_survival_err=row["anchor_rms_survival_err"],
                    survival_csv=row["survival_csv"],
                )
            )
    return rows


def write_markdown_report(
    rows: list[TrajectoryRow],
    scan_rows: list[ScanRow],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# V3b basis_dt convergence\n\n")
    lines.append(
        "Reference is the cached full-basis Jmax=6, dt_fine=0.05 us trajectory.\n\n"
    )

    lines.append("## Single-trajectory sweep\n\n")
    lines.append(
        "| K | basis_dt (us) | tracked bases | total | setup | tracking | "
        "propagation | fidelity | norm^2 | max pop err | survival err | conv. vs half |\n"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in sorted(rows, key=lambda r: (r.K, -r.basis_dt_us)):
        lines.append(
            f"| {row.K} | {row.basis_dt_us:g} | {row.n_tracked_bases} | "
            f"{row.elapsed_total_s / 60:.2f} min | {row.subspace_setup_s:.2f} s | "
            f"{row.tracking_setup_s:.2f} s | {row.propagation_s:.2f} s | "
            f"{row.fidelity:.9f} | {row.norm2:.6f} | {row.max_pop_err:.3e} | "
            f"{row.survival_abs_err:.3e} | {row.converged_vs_next_half or '-'} |\n"
        )

    if rows:
        lines.append("\n## Selected settings\n\n")
        lines.append("| K | selected basis_dt (us) | fidelity | max pop err | survival err | total |\n")
        lines.append("|---:|---:|---:|---:|---:|---:|\n")
        for row in selected_candidates(rows):
            lines.append(
                f"| {row.K} | {row.basis_dt_us:g} | {row.fidelity:.9f} | "
                f"{row.max_pop_err:.3e} | {row.survival_abs_err:.3e} | "
                f"{row.elapsed_total_s / 60:.2f} min |\n"
            )

    if scan_rows:
        lines.append("\n## Scan confirmation\n\n")
        lines.append(
            "| K | basis_dt (us) | scan time | workers | anchor max err | anchor RMS err |\n"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|\n")
        for row in scan_rows:
            lines.append(
                f"| {row.K} | {row.basis_dt_us:g} | {row.elapsed_scan_s / 60:.2f} min | "
                f"{row.n_workers} | {row.anchor_max_abs_survival_err or '-'} | "
                f"{row.anchor_rms_survival_err or '-'} |\n"
            )

    lines.append("\n## Acceptance criteria\n\n")
    lines.append(
        "- Converged vs half-step means fidelity changes by < 1e-4, survival by "
        "< 1e-5, and max population error by < 10% when basis_dt is halved.\n"
    )
    lines.append(
        "- Production scan setting must stay below 15 min for a 41-point scan and "
        "must improve or preserve agreement with the full-basis reference.\n"
    )
    path.write_text("".join(lines), encoding="utf-8")
    print(f"wrote report to {path}")


def run_sweep(k_values: list[int], basis_dt_us_values: list[float]) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    total = len(k_values) * len(basis_dt_us_values)
    index = 0
    for K in k_values:
        for basis_dt_us in basis_dt_us_values:
            index += 1
            print(f"[{index}/{total}] K={K}, basis_dt={basis_dt_us:g} us")
            row = run_v3b_trajectory(K, basis_dt_us)
            rows.append(row)
            print(
                f"  total={row.elapsed_total_s / 60:.2f} min, "
                f"fidelity={row.fidelity:.9f}, "
                f"max_pop_err={row.max_pop_err:.3e}, "
                f"survival_err={row.survival_abs_err:.3e}"
            )
    return annotate_convergence(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", nargs="+", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--basis-dt-us", nargs="+", type=float, default=DEFAULT_BASIS_DT_US
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=HERE / "_cache" / "v3b_basis_dt_convergence.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=HERE / "_cache" / "v3b_basis_dt_convergence.md",
    )
    parser.add_argument(
        "--anchor-cache",
        type=Path,
        default=HERE / "_cache" / "v7_scan_anchors_50ns.npz",
    )
    parser.add_argument("--build-v7-anchors", action="store_true")
    parser.add_argument("--scan-selected", action="store_true")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument(
        "--scan-csv",
        type=Path,
        default=HERE / "_cache" / "v3b_basis_dt_scan_confirmation.csv",
    )
    args = parser.parse_args()

    if args.build_v7_anchors:
        build_v7_anchors(args.anchor_cache)

    if args.scan_selected:
        rows = read_trajectory_csv(args.csv)
        scan_rows = [
            run_scan_candidate(row, args.n_workers, args.anchor_cache)
            for row in selected_candidates(rows)
        ]
        write_scan_csv(scan_rows, args.scan_csv)
        write_markdown_report(rows, scan_rows, args.report)
        return

    if not args.build_v7_anchors:
        rows = run_sweep(args.k, args.basis_dt_us)
        write_trajectory_csv(rows, args.csv)
        scan_rows = read_scan_csv(args.scan_csv)
        write_markdown_report(rows, scan_rows, args.report)


if __name__ == "__main__":
    main()
