"""Thread scaling of lindblad.grid_scan on the real r2-in-E-field scan shape.

Single prepared r2 system (detuning as a scan slot), grid_scan over 120
detunings (-5..30 MHz), output="photon_integral", output_when="final",
dense_output=False, reltol 1e-7 / abstol 1e-9 — i.e. exactly the shape a
peak-ratio notebook scan uses. Measures wall time and trajectories/s for
threads in {1, 2, 4, 8, None(all)}, 2 repeats each (median reported).

Writes benchmarks/scan_speedup_results/scan_thread_scaling.csv.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnose_step_size as diag  # noqa: E402

from centrex_tlf import hamiltonian, states  # noqa: E402
from centrex_tlf.lindblad.batch import grid_scan  # noqa: E402
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem  # noqa: E402
from centrex_tlf.utils.rabi import power_to_rabi_rectangular_beam  # noqa: E402

RESULTS_DIR = HERE / "scan_speedup_results"
GAMMA = float(getattr(hamiltonian, "Γ"))

N_DETUNINGS = 120
DETUNING_MIN_MHZ = -5.0
DETUNING_MAX_MHZ = 30.0
THREAD_SETTINGS: list[int | None] = [1, 2, 4, 8, None]
REPEATS = 2
RELTOL = 1e-7
ABSTOL = 1e-9


def excited_indices(system) -> list[int]:
    return [
        idx
        for idx, state in enumerate(system.QN)
        if state.largest.electronic_state == states.ElectronicState.B
    ]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cpu = os.cpu_count()
    print(f"logical cores: {cpu}")
    print("Building r2-in-E-field system (notebook-identical)...")
    t0 = time.perf_counter()
    system, ts = diag.build_system()
    print(f"  built in {time.perf_counter() - t0:.2f} s, n_states={len(system.QN)}")

    rabi_value = power_to_rabi_rectangular_beam(
        diag.POWER_W, abs(system.couplings[0].main_coupling), diag.BEAM_WX, diag.BEAM_WY
    )
    rho0 = diag.build_rho0(system)
    weights = [(int(idx), GAMMA) for idx in excited_indices(system)]

    # detuning as a scannable slot (base parameter "detuning" bound to delta symbol)
    params = diag.make_parameters(system, ts, rabi_value, 0.0)
    prepared = prepare_lindblad_problem(
        system, params, backend="rust", hamiltonian_representation="decomposed"
    )

    detunings_rad = (
        2 * np.pi * 1e6 * np.linspace(DETUNING_MIN_MHZ, DETUNING_MAX_MHZ, N_DETUNINGS)
    ).astype(np.complex128)

    def run(threads: int | None):
        return grid_scan(
            prepared,
            rho0,
            (0.0, diag.T_END),
            scan={"detuning": detunings_rad},
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="photon_integral",
            output_when="final",
            dense_output=False,
            integral_weights=weights,
            abstol=ABSTOL,
            reltol=RELTOL,
            dt=1e-10,
            parallel=True,
            threads=threads,
        )

    # warmup (small scan, warms rayon pool + code paths)
    warm = grid_scan(
        prepared,
        rho0,
        (0.0, diag.T_END / 50),
        scan={"detuning": detunings_rad[:8]},
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="photon_integral",
        output_when="final",
        dense_output=False,
        integral_weights=weights,
        abstol=ABSTOL,
        reltol=RELTOL,
        dt=1e-10,
        parallel=True,
        threads=None,
    )
    _ = warm.values

    rows: list[dict] = []
    baseline_1t: float | None = None
    ref_values: np.ndarray | None = None
    for threads in THREAD_SETTINGS:
        walls = []
        for rep in range(REPEATS):
            t0 = time.perf_counter()
            result = run(threads)
            walls.append(time.perf_counter() - t0)
            values = np.asarray(result.values, dtype=np.float64).reshape(-1)
            if ref_values is None:
                ref_values = values
            else:
                max_diff = float(np.max(np.abs(values - ref_values)))
                assert max_diff < 1e-12, f"thread-count changed results: {max_diff}"
        wall = float(np.median(walls))
        if threads == 1:
            baseline_1t = wall
        traj_per_s = N_DETUNINGS / wall
        speedup = baseline_1t / wall if baseline_1t else float("nan")
        n_threads_effective = threads if threads is not None else cpu
        efficiency = speedup / n_threads_effective if baseline_1t else float("nan")
        label = "None(all)" if threads is None else str(threads)
        print(
            f"  threads={label:>9s}: wall={wall:7.2f} s (runs: "
            + ", ".join(f"{w:.2f}" for w in walls)
            + f")  {traj_per_s:6.2f} traj/s  speedup={speedup:5.2f}x  "
            f"efficiency={efficiency*100:5.1f}%"
        )
        rows.append(
            {
                "threads": label,
                "effective_threads": n_threads_effective,
                "wall_seconds_median": wall,
                "wall_seconds_run1": walls[0],
                "wall_seconds_run2": walls[1],
                "trajectories": N_DETUNINGS,
                "traj_per_second": traj_per_s,
                "speedup_vs_1_thread": speedup,
                "parallel_efficiency": efficiency,
                "logical_cores": cpu,
            }
        )

    out_path = RESULTS_DIR / "scan_thread_scaling.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
