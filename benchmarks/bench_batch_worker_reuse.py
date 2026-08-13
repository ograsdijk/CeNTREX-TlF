"""Throughput of lindblad.parameter_scan vs grid_scan on 1200 cheap trajectories.

Uses the 2-level system from tests/lindblad/test_rust_backend.py's fixture
pattern, where trajectories are cheap and per-trajectory workspace churn in
the Rust batch path matters. parameter_scan goes through solve_batch_ode;
grid_scan goes through solve_grid_ode_direct (per-thread GridWorker reuse).

Usage: python benchmarks/bench_batch_worker_reuse.py LABEL
LABEL (e.g. "baseline" or "worker_reuse") tags the CSV rows; results are
appended to benchmarks/scan_speedup_results/batch_worker_reuse.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import sympy as smp

from centrex_tlf.lindblad.batch import grid_scan, parameter_scan
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "scan_speedup_results"
CSV_PATH = RESULTS_DIR / "batch_worker_reuse.csv"

N_OMEGA = 40
N_DELTA = 30  # 40 x 30 = 1200 trajectories
REPEATS = 9
T_SPAN = (0.0, 0.5)
COMMON = {
    "solver": "dopri5",
    "execution_mode": "expanded_sparse",
    "dt": 1e-3,
    "reltol": 1e-8,
    "abstol": 1e-10,
    "output": "photon_integral",
    "output_when": "final",
    "dense_output": False,
    "integral_weights": [(1, 0.3)],
}


def make_two_level_system() -> OBESystem:
    omega, delta = smp.symbols("Omega delta", real=True)
    h = smp.Matrix([[0, omega / 2], [smp.conjugate(omega) / 2, -delta]])
    c_array = np.zeros((1, 2, 2), dtype=np.complex128)
    c_array[0, 0, 1] = np.sqrt(0.3)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    return OBESystem(
        ground=[], excited=[], QN=[], H_int=zeros, V_ref_int=zeros,
        couplings=[], H_symbolic=h, C_array=c_array, system=None,
        coupling_symbols=[omega, delta], polarization_symbols=[],
    )


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    system = make_two_level_system()
    omega_symbol, delta_symbol = [str(s) for s in system.coupling_symbols]
    prepared = prepare_lindblad_problem(
        system,
        {omega_symbol: 0.6, delta_symbol: 0.0},
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1.0
    omega_axis = np.linspace(0.25, 1.25, N_OMEGA)
    delta_axis = np.linspace(-0.4, 0.4, N_DELTA)
    n_traj = omega_axis.size * delta_axis.size
    flat = np.asarray(
        [(o, d) for o in omega_axis for d in delta_axis], dtype=np.complex128
    )
    scan = {omega_symbol: omega_axis, delta_symbol: delta_axis}

    def run_parameter(threads):
        return parameter_scan(
            prepared, rho0, T_SPAN,
            parameter_slots=[omega_symbol, delta_symbol],
            parameter_batch=flat,
            parallel=True, threads=threads, **COMMON,
        )

    def run_grid(threads):
        return grid_scan(
            prepared, rho0, T_SPAN, scan=scan,
            parallel=True, threads=threads, **COMMON,
        )

    # correctness cross-check + warmup
    ref_grid = np.asarray(run_grid(None).values, dtype=np.float64).reshape(-1)
    ref_param = np.asarray(run_parameter(None).values, dtype=np.float64).reshape(-1)
    max_diff = float(np.max(np.abs(ref_grid - ref_param)))
    print(f"[{label}] parameter_scan vs grid_scan values max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-12, "parameter_scan and grid_scan disagree"

    rows = []
    for api_name, fn in [("parameter_scan", run_parameter), ("grid_scan", run_grid)]:
        for threads in (1, None):
            walls = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                fn(threads)
                walls.append(time.perf_counter() - t0)
            wall = float(np.median(walls))
            tlabel = "None(all)" if threads is None else str(threads)
            print(
                f"[{label}] {api_name:15s} threads={tlabel:>9s}: "
                f"median {wall*1e3:8.2f} ms  ({n_traj/wall:9.0f} traj/s)  "
                f"min {min(walls)*1e3:8.2f} ms"
            )
            rows.append(
                {
                    "label": label,
                    "api": api_name,
                    "threads": tlabel,
                    "trajectories": n_traj,
                    "repeats": REPEATS,
                    "wall_seconds_median": wall,
                    "wall_seconds_min": float(min(walls)),
                    "traj_per_second_median": n_traj / wall,
                }
            )

    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[{label}] appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
