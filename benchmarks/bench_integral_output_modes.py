from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as smp

from centrex_tlf.lindblad.batch import grid_scan
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem

GAMMA = 0.3


@dataclass(frozen=True)
class TimingSummary:
    label: str
    mean_s: float
    stdev_s: float
    min_s: float
    max_s: float
    repeats: int


def make_two_level_system() -> OBESystem:
    omega, delta = smp.symbols("Omega delta", real=True)
    hamiltonian = smp.Matrix(
        [
            [0, omega / 2],
            [smp.conjugate(omega) / 2, -delta],
        ]
    )
    c_array = np.zeros((1, 2, 2), dtype=np.complex128)
    c_array[0, 0, 1] = np.sqrt(GAMMA)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    return OBESystem(
        ground=[],
        excited=[],
        QN=[],
        H_int=zeros,
        V_ref_int=zeros,
        couplings=[],
        H_symbolic=hamiltonian,
        C_array=c_array,
        system=None,
        coupling_symbols=[omega, delta],
        polarization_symbols=[],
    )


def prepare_model() -> tuple[object, np.ndarray, str, str]:
    system = make_two_level_system()
    omega_symbol, delta_symbol = [str(s) for s in system.coupling_symbols]
    params = LindbladParameters.from_kwargs(**{omega_symbol: 0.75, delta_symbol: 0.0})
    prepared = prepare_lindblad_problem(
        system,
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1.0
    return prepared, rho0, omega_symbol, delta_symbol


def time_call(fn, repeats: int) -> tuple[list[float], object]:
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def summarize(label: str, times: list[float]) -> TimingSummary:
    return TimingSummary(
        label=label,
        mean_s=statistics.mean(times),
        stdev_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_s=min(times),
        max_s=max(times),
        repeats=len(times),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    *,
    repeats: int,
    scan_points: int,
    save_points: int,
    threads: int | None,
    output_dir: Path,
) -> None:
    prepared, rho0, omega_symbol, delta_symbol = prepare_model()
    t_span = (0.0, 4.0)
    detunings = np.linspace(-2.0, 2.0, scan_points)
    saveat = np.linspace(t_span[0], t_span[1], save_points)
    weights = [(1, GAMMA)]
    common = dict(
        prepared=prepared,
        rho0=rho0,
        t_span=t_span,
        scan={delta_symbol: detunings},
        solver="dopri5",
        execution_mode="expanded_sparse",
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=True,
        threads=threads,
    )

    def in_solver_final():
        return grid_scan(
            **common,
            output="photon_integral",
            integral_weights=weights,
            output_when="final",
            saveat=None,
            dense_output=False,
        )

    def after_solve_populations():
        return grid_scan(
            **common,
            output="populations",
            output_when="saveat",
            saveat=saveat,
        )

    def in_solver_trace():
        return grid_scan(
            **common,
            output="photon_integral",
            integral_weights=weights,
            output_when="saveat",
            saveat=saveat,
        )

    # Warm up extension/import paths and Rayon pool construction before timing.
    in_solver_final()
    after_solve_populations()
    in_solver_trace()

    in_solver_times, in_solver = time_call(in_solver_final, repeats)
    post_times, post = time_call(after_solve_populations, repeats)
    trace_times, trace = time_call(in_solver_trace, repeats)

    in_solver_values = np.asarray(in_solver.values[:, 0], dtype=np.float64)
    rate_from_populations = GAMMA * np.asarray(post.values[:, :, 1], dtype=np.float64)
    post_values = np.trapezoid(rate_from_populations, x=post.t, axis=1)
    trace_final_values = np.asarray(trace.values[:, -1, 0], dtype=np.float64)

    post_diff = post_values - in_solver_values
    trace_diff = trace_final_values - post_values
    in_trace_diff = trace_final_values - in_solver_values

    timing_rows = []
    for label, times in [
        ("in_solver_final_saveat_none", in_solver_times),
        ("post_solve_populations_trapezoid", post_times),
        ("in_solver_cumulative_trace", trace_times),
    ]:
        for idx, elapsed in enumerate(times):
            timing_rows.append({"method": label, "repeat": idx, "seconds": elapsed})

    comparison_rows = [
        {
            "detuning": float(detuning),
            "in_solver_final": float(a),
            "post_solve_trapezoid": float(b),
            "in_solver_trace_final": float(c),
            "post_minus_in_solver": float(db),
            "trace_minus_post": float(dt),
            "trace_minus_in_solver": float(di),
        }
        for detuning, a, b, c, db, dt, di in zip(
            detunings,
            in_solver_values,
            post_values,
            trace_final_values,
            post_diff,
            trace_diff,
            in_trace_diff,
            strict=True,
        )
    ]

    write_csv(output_dir / "integral_output_timing_repeats.csv", timing_rows)
    write_csv(output_dir / "integral_output_value_comparison.csv", comparison_rows)

    summaries = [
        summarize("in-solver final, saveat=None", in_solver_times),
        summarize("post-solve populations + np.trapezoid", post_times),
        summarize("in-solver cumulative trace", trace_times),
    ]
    speedup = summaries[1].mean_s / summaries[0].mean_s
    trace_speedup = summaries[1].mean_s / summaries[2].mean_s

    def error_line(name: str, diff: np.ndarray) -> str:
        return (
            f"| {name} | {np.max(np.abs(diff)):.6e} | "
            f"{np.sqrt(np.mean(diff**2)):.6e} | {statistics.mean(diff):.6e} |"
        )

    report = f"""# In-Solver Integral Output Benchmark

## Setup

- Model: two-level Lindblad system with one decay channel.
- Scan: {scan_points} detuning points from {detunings[0]:.3g} to {detunings[-1]:.3g}.
- Time span: {t_span[0]:.3g} to {t_span[1]:.3g}; post-solve grid has {save_points} save points.
- Repeats per method: {repeats}.
- Threads: {"Rayon default" if threads is None else threads}.
- Integral weights: excited-state population times Gamma = {GAMMA}.

## Timing

| Method | mean (s) | stdev (s) | min (s) | max (s) | repeats |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for summary in summaries:
        report += (
            f"| {summary.label} | {summary.mean_s:.6f} | {summary.stdev_s:.6f} | "
            f"{summary.min_s:.6f} | {summary.max_s:.6f} | {summary.repeats} |\n"
        )
    report += f"""
The final scalar in-solver path is {speedup:.2f}x faster than saving populations and integrating in Python.
The cumulative in-solver trace path is {trace_speedup:.2f}x faster than saving all populations and integrating in Python.

## Value Differences

All differences are in expected emitted photons per molecule.

| Comparison | max abs diff | RMS diff | mean signed diff |
| --- | ---: | ---: | ---: |
{error_line("post-solve trapezoid - in-solver final", post_diff)}
{error_line("in-solver trace final - post-solve trapezoid", trace_diff)}
{error_line("in-solver trace final - in-solver final", in_trace_diff)}

## Notes

- `output="photon_integral", output_when="final", saveat=None` integrates on accepted solver steps and returns one scalar per trajectory.
- Post-solve integration saves all populations on `saveat`, extracts the excited-state population, and integrates with `np.trapezoid`.
- Cumulative trace uses the new in-solver trace mode at the same `saveat` points as the post-solve path.
- Raw timing repeats: `integral_output_timing_repeats.csv`.
- Raw value comparison: `integral_output_value_comparison.csv`.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "integral_output_benchmark_report.md").write_text(report, encoding="utf-8")

    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--scan-points", type=int, default=101)
    parser.add_argument("--save-points", type=int, default=401)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks") / "integral_output_results",
    )
    args = parser.parse_args()
    run_benchmark(
        repeats=args.repeats,
        scan_points=args.scan_points,
        save_points=args.save_points,
        threads=args.threads,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
