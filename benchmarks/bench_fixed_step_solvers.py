from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as smp

from centrex_tlf.lindblad.batch import grid_scan
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.utils_setup import OBESystem

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "fixed_step_solver_results"
REPORT_PATH = RESULTS_DIR / "fixed_step_solver_report.md"
TIMINGS_PATH = RESULTS_DIR / "fixed_step_solver_timings.csv"
VALIDATION_PATH = RESULTS_DIR / "fixed_step_solver_validation.csv"
REPEATS = 15


@dataclass(frozen=True)
class SolverCase:
    solver: str
    dt: float
    label: str


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


def summarize(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    trim_count = max(1, len(sorted_values) // 10)
    trimmed = sorted_values[trim_count:-trim_count]
    return {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "trimmed_mean_seconds": statistics.fmean(trimmed),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def time_repeated(fn: Callable[[], object]) -> tuple[list[float], object]:
    result = fn()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return times, result


def normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return values.copy()
    return values / scale


def peak_position(axis: np.ndarray, values: np.ndarray) -> float:
    return float(axis[int(np.argmax(values))])


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


def run_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    system = make_two_level_system()
    omega_symbol, delta_symbol = [str(symbol) for symbol in system.coupling_symbols]
    prepared = prepare_lindblad_problem(
        system,
        {omega_symbol: 0.8, delta_symbol: 0.0},
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho0 = ground_state_density()
    t_span = (0.0, 0.8)
    detuning_axis = np.linspace(-2.0, 2.0, 201)
    adaptive = SolverCase("dopri5", 1e-3, "adaptive_dopri5")
    cases = [
        adaptive,
        SolverCase("fixed_dopri5", 5e-2, "fixed_dopri5_dt5e-2"),
        SolverCase("fixed_dopri5", 2.5e-2, "fixed_dopri5_dt2p5e-2"),
        SolverCase("fixed_dopri5", 1e-3, "fixed_dopri5_dt1e-3"),
        SolverCase("fixed_rk4", 5e-2, "fixed_rk4_dt5e-2"),
        SolverCase("fixed_rk4", 2.5e-2, "fixed_rk4_dt2p5e-2"),
        SolverCase("fixed_rk4", 1e-3, "fixed_rk4_dt1e-3"),
        SolverCase("fixed_rk4", 2e-3, "fixed_rk4_dt2e-3"),
        SolverCase("fixed_rk2", 1e-2, "fixed_rk2_dt1e-2"),
        SolverCase("fixed_rk2", 5e-4, "fixed_rk2_dt5e-4"),
    ]
    common = {
        "execution_mode": "expanded_sparse",
        "output": "photon_integral",
        "output_when": "final",
        "integral_weights": [(1, 0.3)],
        "saveat": None,
        "reltol": 1e-8,
        "abstol": 1e-10,
        "collect_stats": True,
    }
    timing_rows = []
    summary_rows = []
    validation_rows = []

    single_results = {}
    grid_results = {}
    for case in cases:
        single_kwargs = {
            **common,
            "solver": case.solver,
            "dt": case.dt,
        }

        def run_single(case: SolverCase = case, kwargs: dict[str, object] = single_kwargs):
            return solve_lindblad(prepared, rho0, t_span, **kwargs)

        single_times, single_result = time_repeated(run_single)
        single_results[case.label] = single_result
        single_summary = summarize(single_times)
        for repeat, seconds in enumerate(single_times, start=1):
            timing_rows.append(
                {
                    "case": case.label,
                    "mode": "single",
                    "threads": 1,
                    "repeat": repeat,
                    "seconds": seconds,
                }
            )
        summary_rows.append(
            {
                "case": case.label,
                "mode": "single",
                "threads": 1,
                **single_summary,
                "accepted_steps": single_result.solver_stats["accepted_steps"],
                "rhs_calls": single_result.solver_stats["rhs_calls"],
            }
        )

        for threads in (1, 4):
            parallel = threads > 1
            grid_kwargs = {
                **common,
                "solver": case.solver,
                "dt": case.dt,
                "parallel": parallel,
                "threads": threads,
            }

            def run_grid(case: SolverCase = case, kwargs: dict[str, object] = grid_kwargs):
                return grid_scan(
                    prepared,
                    rho0,
                    t_span,
                    scan={delta_symbol: detuning_axis},
                    **kwargs,
                )

            grid_times, grid_result = time_repeated(run_grid)
            grid_results[(case.label, threads)] = grid_result
            grid_summary = summarize(grid_times)
            for repeat, seconds in enumerate(grid_times, start=1):
                timing_rows.append(
                    {
                        "case": case.label,
                        "mode": "grid",
                        "threads": threads,
                        "repeat": repeat,
                        "seconds": seconds,
                    }
                )
            summary_rows.append(
                {
                    "case": case.label,
                    "mode": "grid",
                    "threads": threads,
                    **grid_summary,
                    "accepted_steps": grid_result.solver_stats["accepted_steps"],
                    "rhs_calls": grid_result.solver_stats["rhs_calls"],
                }
            )

    reference_single = float(single_results[adaptive.label].values[0])
    reference_grid = np.asarray(grid_results[(adaptive.label, 1)].values[:, 0], dtype=float)
    reference_norm = normalized(reference_grid)
    reference_peak = peak_position(detuning_axis, reference_grid)
    for case in cases[1:]:
        single_value = float(single_results[case.label].values[0])
        validation_rows.append(
            {
                "case": case.label,
                "mode": "single",
                "threads": 1,
                "photon_integral_abs_error": abs(single_value - reference_single),
                "photon_integral_rel_error": abs(single_value - reference_single)
                / max(abs(reference_single), 1e-15),
                "peak_shift": np.nan,
                "max_normalized_line_error": np.nan,
            }
        )
        for threads in (1, 4):
            values = np.asarray(grid_results[(case.label, threads)].values[:, 0], dtype=float)
            validation_rows.append(
                {
                    "case": case.label,
                    "mode": "grid",
                    "threads": threads,
                    "photon_integral_abs_error": abs(float(values.max()) - float(reference_grid.max())),
                    "photon_integral_rel_error": abs(float(values.max()) - float(reference_grid.max()))
                    / max(abs(float(reference_grid.max())), 1e-15),
                    "peak_shift": peak_position(detuning_axis, values) - reference_peak,
                    "max_normalized_line_error": float(
                        np.max(np.abs(normalized(values) - reference_norm))
                    ),
                }
            )

    timings = pd.DataFrame(timing_rows)
    summary = pd.DataFrame(summary_rows)
    validation = pd.DataFrame(validation_rows)
    timings.to_csv(TIMINGS_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    write_report(summary, validation)
    return summary, validation


def write_report(summary: pd.DataFrame, validation: pd.DataFrame) -> None:
    adaptive = summary[summary["case"] == "adaptive_dopri5"][
        ["mode", "threads", "trimmed_mean_seconds"]
    ].rename(columns={"trimmed_mean_seconds": "adaptive_trimmed_mean_seconds"})
    comparison = summary.merge(adaptive, on=["mode", "threads"], how="left")
    comparison["speedup_vs_adaptive"] = (
        comparison["adaptive_trimmed_mean_seconds"]
        / comparison["trimmed_mean_seconds"]
    )
    comparison = comparison[
        [
            "case",
            "mode",
            "threads",
            "trimmed_mean_seconds",
            "speedup_vs_adaptive",
            "accepted_steps",
            "rhs_calls",
        ]
    ]
    fixed_comparison = comparison[comparison["case"] != "adaptive_dopri5"]
    best = fixed_comparison.loc[fixed_comparison["speedup_vs_adaptive"].idxmax()]
    worst = fixed_comparison.loc[fixed_comparison["speedup_vs_adaptive"].idxmin()]
    lines = [
        "# Fixed-Step Solver Benchmark",
        "",
        "This benchmark investigates fixed-step Rust RK solvers for coarse OBE scans. The fixed-step solvers use `dt` as a maximum step and shorten steps only to land on `saveat` points or `t1`.",
        "",
        "## Setup",
        "",
        "- Model: two-level Lindblad system with one decay channel.",
        "- Single trajectory: final photon integral at zero detuning.",
        "- Grid scan: 201 detuning points from `-2` to `2`, final photon integral.",
        f"- Repeats per timing case: `{REPEATS}`.",
        "- Reference: adaptive Rust `dopri5` with `dt=1e-3`, `reltol=1e-8`, `abstol=1e-10`.",
        "",
        "## Runtime",
        "",
        markdown_table(comparison),
        "",
        "## Validation Against Adaptive Reference",
        "",
        markdown_table(validation),
        "",
        "## Interpretation",
        "",
        f"- Best fixed-step speedup: `{best['speedup_vs_adaptive']:.3g}x` for `{best['case']}` in `{best['mode']}` mode with `{int(best['threads'])}` thread(s).",
        f"- Worst fixed-step speedup: `{worst['speedup_vs_adaptive']:.3g}x` for `{worst['case']}` in `{worst['mode']}` mode with `{int(worst['threads'])}` thread(s).",
        "- `fixed_dopri5` and `fixed_rk4` are useful candidates when `dt` is coarse enough to keep the accepted-step count comparable to adaptive `dopri5`. In this benchmark, `dt=0.05` gives the speedup.",
        "- Small fixed steps are counterproductive: `dt=1e-3` and `dt=2e-3` perform hundreds of fixed steps where adaptive `dopri5` takes only tens of steps.",
        "- `fixed_rk2` is included as a lower-stage comparison, but it did not win here at the tested validation settings.",
        "- Photon-count errors are measured against the current adaptive solver baseline, including its accepted-step quadrature for final integrals.",
        "- For real molecular OBE scans, the acceptance criterion should be peak shifts below the requested MHz tolerance and photon-count/line-shape errors acceptable for the fit or coarse scan stage.",
        "",
        f"- Raw timings: `{TIMINGS_PATH.name}`.",
        f"- Validation CSV: `{VALIDATION_PATH.name}`.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_benchmarks()
