from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as smp

from centrex_tlf.lindblad.batch import grid_scan, parameter_scan
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "grid_scan_worker_reuse_results"
REPORT_PATH = RESULTS_DIR / "grid_scan_worker_reuse_report.md"
COMPARISON_REPORT_PATH = RESULTS_DIR / "grid_scan_full_reuse_comparison_report.md"
TIMINGS_PATH = RESULTS_DIR / "grid_scan_worker_reuse_timings.csv"
SUMMARY_PATH = RESULTS_DIR / "grid_scan_worker_reuse_summary.csv"
BASELINE_SUMMARY_PATH = RESULTS_DIR / "grid_scan_rhs_only_baseline_summary.csv"
REPEATS = 21


@dataclass(frozen=True)
class Case:
    name: str
    output: str
    output_when: str
    saveat: np.ndarray | None
    output_indices: list[tuple[int, int]] | None
    integral_weights: list[tuple[int, float]] | None


def make_two_level_system() -> OBESystem:
    omega, delta = smp.symbols("Omega delta", real=True)
    hamiltonian = smp.Matrix(
        [
            [0, omega / 2],
            [smp.conjugate(omega) / 2, -delta],
        ]
    )
    c_array = np.zeros((1, 2, 2), dtype=np.complex128)
    c_array[0, 0, 1] = np.sqrt(0.3)
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


def ground_state_density() -> np.ndarray:
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1.0
    return rho0


def parameter_batch(omega_axis: np.ndarray, delta_axis: np.ndarray) -> np.ndarray:
    rows = []
    for omega in omega_axis:
        for delta in delta_axis:
            rows.append((omega, delta))
    return np.asarray(rows, dtype=np.complex128)


def summarize(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    trim_count = max(1, len(sorted_values) // 10)
    trimmed = sorted_values[trim_count:-trim_count] if len(sorted_values) > 2 else sorted_values
    return {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "trimmed_mean_seconds": statistics.fmean(trimmed),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def run_repeated(label: str, fn: Callable[[], object]) -> tuple[list[float], object]:
    result = fn()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return times, result


def run_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    system = make_two_level_system()
    omega_symbol, delta_symbol = [str(symbol) for symbol in system.coupling_symbols]
    prepared = prepare_lindblad_problem(
        system,
        {omega_symbol: 0.6, delta_symbol: 0.0},
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho0 = ground_state_density()
    omega_axis = np.linspace(0.25, 1.25, 20)
    delta_axis = np.linspace(-0.4, 0.4, 20)
    trajectory_count = omega_axis.size * delta_axis.size
    flat_parameters = parameter_batch(omega_axis, delta_axis)
    scan = {
        omega_symbol: omega_axis,
        delta_symbol: delta_axis,
    }
    t_span = (0.0, 0.5)
    common = {
        "solver": "dopri5",
        "execution_mode": "expanded_sparse",
        "dt": 1e-3,
        "reltol": 1e-8,
        "abstol": 1e-10,
        "collect_stats": True,
    }
    cases = [
        Case(
            name="final_populations",
            output="populations",
            output_when="final",
            saveat=None,
            output_indices=None,
            integral_weights=None,
        ),
        Case(
            name="final_photon_integral",
            output="photon_integral",
            output_when="final",
            saveat=None,
            output_indices=None,
            integral_weights=[(1, 0.3)],
        ),
        Case(
            name="saveat_selected",
            output="selected",
            output_when="saveat",
            saveat=np.linspace(0.0, 0.5, 21),
            output_indices=[(0, 0), (0, 1), (1, 0)],
            integral_weights=None,
        ),
    ]
    raw_rows = []
    summary_rows = []
    validation_rows = []

    for case in cases:
        for threads in (1, 4):
            parallel = threads > 1

            def run_grid(
                case: Case = case,
                parallel: bool = parallel,
                threads: int = threads,
            ) -> object:
                return grid_scan(
                    prepared,
                    rho0,
                    t_span,
                    scan=scan,
                    output=case.output,
                    output_when=case.output_when,
                    saveat=case.saveat,
                    output_indices=case.output_indices,
                    integral_weights=case.integral_weights,
                    parallel=parallel,
                    threads=threads,
                    **common,
                )

            def run_parameter(
                case: Case = case,
                parallel: bool = parallel,
                threads: int = threads,
            ) -> object:
                return parameter_scan(
                    prepared,
                    rho0,
                    t_span,
                    parameter_slots=[omega_symbol, delta_symbol],
                    parameter_batch=flat_parameters,
                    output=case.output,
                    output_when=case.output_when,
                    saveat=case.saveat,
                    output_indices=case.output_indices,
                    integral_weights=case.integral_weights,
                    parallel=parallel,
                    threads=threads,
                    **common,
                )

            grid_times, grid_result = run_repeated("grid", run_grid)
            parameter_times, parameter_result = run_repeated("parameter", run_parameter)
            np.testing.assert_allclose(grid_result.values, parameter_result.values)
            np.testing.assert_allclose(grid_result.t, parameter_result.t)
            validation_rows.append(
                {
                    "case": case.name,
                    "threads": threads,
                    "max_abs_value_difference": float(
                        np.max(np.abs(grid_result.values - parameter_result.values))
                    ),
                }
            )
            for method, times in (
                ("optimized_grid_scan", grid_times),
                ("generic_parameter_scan", parameter_times),
            ):
                for repeat, seconds in enumerate(times, start=1):
                    raw_rows.append(
                        {
                            "case": case.name,
                            "method": method,
                            "threads": threads,
                            "repeat": repeat,
                            "seconds": seconds,
                            "trajectories_per_second": trajectory_count / seconds,
                            "trajectory_count": trajectory_count,
                        }
                    )
            grid_summary = summarize(grid_times)
            parameter_summary = summarize(parameter_times)
            summary_rows.append(
                {
                    "case": case.name,
                    "threads": threads,
                    "trajectory_count": trajectory_count,
                    "optimized_mean_seconds": grid_summary["mean_seconds"],
                    "optimized_median_seconds": grid_summary["median_seconds"],
                    "optimized_trimmed_mean_seconds": grid_summary[
                        "trimmed_mean_seconds"
                    ],
                    "optimized_stdev_seconds": grid_summary["stdev_seconds"],
                    "generic_mean_seconds": parameter_summary["mean_seconds"],
                    "generic_median_seconds": parameter_summary["median_seconds"],
                    "generic_trimmed_mean_seconds": parameter_summary[
                        "trimmed_mean_seconds"
                    ],
                    "generic_stdev_seconds": parameter_summary["stdev_seconds"],
                    "speedup": parameter_summary["mean_seconds"]
                    / grid_summary["mean_seconds"],
                    "median_speedup": parameter_summary["median_seconds"]
                    / grid_summary["median_seconds"],
                    "trimmed_mean_speedup": parameter_summary["trimmed_mean_seconds"]
                    / grid_summary["trimmed_mean_seconds"],
                    "optimized_trajectories_per_second": trajectory_count
                    / grid_summary["mean_seconds"],
                    "generic_trajectories_per_second": trajectory_count
                    / parameter_summary["mean_seconds"],
                }
            )

    timings = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    validation = pd.DataFrame(validation_rows)
    timings.to_csv(TIMINGS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_report(summary, validation)
    if BASELINE_SUMMARY_PATH.exists():
        write_comparison_report(pd.read_csv(BASELINE_SUMMARY_PATH), summary)
    return timings, summary


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)

    def fmt(value: object) -> str:
        if isinstance(value, float | np.floating):
            return f"{float(value):.6g}"
        if isinstance(value, int | np.integer):
            return str(int(value))
        return str(value)

    rows = [[fmt(value) for value in row] for row in df.to_numpy(dtype=object)]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        str(column).ljust(widths[idx]) for idx, column in enumerate(columns)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_report(summary: pd.DataFrame, validation: pd.DataFrame) -> None:
    best = summary.loc[summary["trimmed_mean_speedup"].idxmax()]
    worst = summary.loc[summary["trimmed_mean_speedup"].idxmin()]
    lines = [
        "# Grid Scan Worker Reuse Benchmark",
        "",
        "This benchmark compares the optimized `grid_scan` path against the current generic `parameter_scan` batch path with an equivalent flattened two-parameter grid.",
        "",
        "The optimized grid path reuses one Rust RHS workspace and one output collector per worker, and writes fixed-size grid outputs directly into preallocated result storage. The generic parameter-scan path still constructs per-trajectory RHS/output objects and collates results afterward, so it is a practical proxy for the old grid behavior.",
        "",
        "## Setup",
        "",
        "- Model: two-level Lindblad system with one decay channel.",
        "- Grid: `20 x 20 = 400` trajectories.",
        f"- Repeats per method: `{REPEATS}`.",
        "- Solvers: Rust `dopri5`, `expanded_sparse` execution.",
        "- Cases: final populations, final photon integral with `saveat=None`, and selected saveat trace.",
        "- Mean, median, and 10% trimmed mean are reported because short threaded runs can have scheduler outliers.",
        "",
        "## Results",
        "",
        markdown_table(summary),
        "",
        "## Validation",
        "",
        markdown_table(validation),
        "",
        "## Summary",
        "",
        f"- Best observed trimmed-mean speedup: `{best['trimmed_mean_speedup']:.3g}x` for `{best['case']}` with `{int(best['threads'])}` threads.",
        f"- Smallest observed trimmed-mean speedup: `{worst['trimmed_mean_speedup']:.3g}x` for `{worst['case']}` with `{int(worst['threads'])}` threads.",
        "- These numbers mainly measure grid orchestration overhead. Larger molecular OBE systems are still dominated by RHS evaluations, so the relative speedup there should be smaller, but memory traffic and allocation pressure should still improve.",
        "",
        f"- Raw timings: `{TIMINGS_PATH.name}`.",
        f"- Summary CSV: `{SUMMARY_PATH.name}`.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(baseline: pd.DataFrame, current: pd.DataFrame) -> None:
    merged = baseline.merge(
        current,
        on=["case", "threads", "trajectory_count"],
        suffixes=("_rhs_only", "_full_reuse"),
    )
    comparison = pd.DataFrame(
        {
            "case": merged["case"],
            "threads": merged["threads"],
            "rhs_only_mean_seconds": merged["optimized_mean_seconds_rhs_only"],
            "full_reuse_mean_seconds": merged["optimized_mean_seconds_full_reuse"],
            "mean_speedup_from_output_reuse": merged[
                "optimized_mean_seconds_rhs_only"
            ]
            / merged["optimized_mean_seconds_full_reuse"],
            "full_reuse_median_seconds": merged["optimized_median_seconds"],
            "full_reuse_trimmed_mean_seconds": merged[
                "optimized_trimmed_mean_seconds"
            ],
        }
    )
    best = comparison.loc[comparison["mean_speedup_from_output_reuse"].idxmax()]
    worst = comparison.loc[comparison["mean_speedup_from_output_reuse"].idxmin()]
    lines = [
        "# Grid Scan Full Reuse Comparison",
        "",
        "This report compares the saved RHS-only grid-worker baseline against the current implementation, where each grid worker reuses both its RHS workspace and output collector.",
        "",
        "The baseline file is `grid_scan_rhs_only_baseline_summary.csv`. It was produced before adding reusable output collectors. The current file is `grid_scan_worker_reuse_summary.csv`.",
        "",
        "## Output Reuse Delta",
        "",
        markdown_table(comparison),
        "",
        "## Interpretation",
        "",
        f"- Best mean speedup from output reuse alone: `{best['mean_speedup_from_output_reuse']:.3g}x` for `{best['case']}` with `{int(best['threads'])}` threads.",
        f"- Worst mean speedup from output reuse alone: `{worst['mean_speedup_from_output_reuse']:.3g}x` for `{worst['case']}` with `{int(worst['threads'])}` threads.",
        "- These are small, allocation-level changes. The expected benefit is largest when trajectories are cheap and output allocation/collation is a visible fraction of runtime.",
        "- For RHS-dominated molecular OBE solves, this should mainly reduce allocation pressure rather than produce a large wall-time change.",
        "",
        "## Validation",
        "",
        "- The benchmark still compares `grid_scan` values against equivalent flattened `parameter_scan` values with exact `np.testing.assert_allclose` checks.",
    ]
    COMPARISON_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_benchmarks()
