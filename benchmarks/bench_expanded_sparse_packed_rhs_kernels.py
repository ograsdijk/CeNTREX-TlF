from __future__ import annotations

import statistics
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as smp

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.centrex_tlf_rust import (
    create_lindblad_rhs_evaluator_py,
    solve_lindblad_grid_ode_py,
    solve_lindblad_ode_py,
)
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import PreparedLindbladProblem, prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem
from centrex_tlf.utils.rabi import power_to_rabi_rectangular_beam

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "expanded_sparse_packed_rhs_results"
REPORT_PATH = RESULTS_DIR / "expanded_sparse_packed_rhs_report.md"
RHS_TIMINGS_PATH = RESULTS_DIR / "rhs_only_timings.csv"
SOLVE_TIMINGS_PATH = RESULTS_DIR / "solve_timings.csv"
GRID_TIMINGS_PATH = RESULTS_DIR / "grid_timings.csv"
VALIDATION_PATH = RESULTS_DIR / "validation.csv"
SYSTEMS_PATH = RESULTS_DIR / "systems.csv"

GAMMA = getattr(hamiltonian, "Γ")  # noqa: B009
R2_TRANSITION = transitions.R2_F1_7o2_F3
E_FIELD = np.array([0.0, 0.0, 200.0])
B_FIELD = np.array([0.0, 0.0, 1e-5])
POWER_MW = 60.0
BEAM_WX = 0.02
BEAM_WY = 0.02
VELOCITY = 184.0
INTERACTION_LENGTH = 0.02
T_END_R2 = INTERACTION_LENGTH / VELOCITY
INITIAL_F1 = 5 / 2
INITIAL_F = 2
INITIAL_MF = 0
RABI_PARAMETER_NAME = "rabi"
DETUNING_PARAMETER_NAME = "detuning"

VARIANTS = [
    ("baseline", "expanded_sparse"),
    (
        "split_coefficients",
        "experimental_expanded_sparse_split_coefficients",
    ),
    ("split_inputs", "experimental_expanded_sparse_split_inputs"),
    (
        "precomputed_inputs",
        "experimental_expanded_sparse_precomputed_inputs",
    ),
]


@dataclass(frozen=True)
class BenchmarkModel:
    name: str
    system: OBESystem
    prepared: PreparedLindbladProblem
    rho0: np.ndarray
    t_span: tuple[float, float]
    dt: float
    reltol: float
    abstol: float
    rhs_iterations: int
    rhs_repeats: int
    solve_repeats: int
    grid_repeats: int
    detuning_symbol: str
    detuning_axis: np.ndarray
    photon_integral_weights: list[tuple[int, float]]
    prep_seconds: float


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


def timed_repeats(fn: Callable[[], Any], repeats: int) -> tuple[list[float], Any]:
    result = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return times, result


def make_two_level_system() -> tuple[OBESystem, str, str]:
    omega, delta = smp.symbols("Omega delta", real=True)
    hamiltonian_matrix = smp.Matrix(
        [
            [0, omega / 2],
            [smp.conjugate(omega) / 2, -delta],
        ]
    )
    c_array = np.zeros((1, 2, 2), dtype=np.complex128)
    c_array[0, 0, 1] = np.sqrt(0.3)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    return (
        OBESystem(
            ground=[],
            excited=[],
            QN=[],
            H_int=zeros,
            V_ref_int=zeros,
            couplings=[],
            H_symbolic=hamiltonian_matrix,
            C_array=c_array,
            system=None,
            coupling_symbols=[omega, delta],
            polarization_symbols=[],
        ),
        str(omega),
        str(delta),
    )


def prepare_two_level_model() -> BenchmarkModel:
    start = time.perf_counter()
    system, omega, delta = make_two_level_system()
    prepared = prepare_lindblad_problem(
        system,
        {omega: 0.6, delta: 0.0},
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1.0
    return BenchmarkModel(
        name="two_level",
        system=system,
        prepared=prepared,
        rho0=rho0,
        t_span=(0.0, 0.5),
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        rhs_iterations=5_000,
        rhs_repeats=7,
        solve_repeats=7,
        grid_repeats=5,
        detuning_symbol=delta,
        detuning_axis=np.linspace(-0.4, 0.4, 25),
        photon_integral_weights=[(1, 0.3)],
        prep_seconds=time.perf_counter() - start,
    )


def r2_transition_selectors() -> list[couplings.TransitionSelector]:
    ground_main = 1 * next(
        iter(
            states.generate_coupled_states_X(
                states.QuantumSelector(
                    electronic=states.ElectronicState.X,
                    J=R2_TRANSITION.J_ground,
                    F1=INITIAL_F1,
                    F=INITIAL_F,
                    mF=INITIAL_MF,
                    P=R2_TRANSITION.P_ground,
                )
            )
        )
    )
    excited_main = 1 * next(
        iter(
            states.generate_coupled_states_B(
                states.QuantumSelector(
                    electronic=states.ElectronicState.B,
                    J=R2_TRANSITION.J_excited,
                    F1=R2_TRANSITION.F1_excited,
                    F=R2_TRANSITION.F_excited,
                    mF=1,
                    P=R2_TRANSITION.P_excited,
                )
            )
        )
    )
    return couplings.generate_transition_selectors(
        transitions=[R2_TRANSITION],
        polarizations=[[couplings.polarization_X]],
        ground_mains=[ground_main],
        excited_mains=[excited_main],
    )


def r2_initial_state_index(system: OBESystem) -> int:
    candidates = []
    for idx, state in enumerate(system.QN):
        qn = state.largest
        if (
            qn.electronic_state == states.ElectronicState.X
            and R2_TRANSITION.J_ground == qn.J
            and qn.F1 == INITIAL_F1
            and qn.F == INITIAL_F
            and qn.mF == INITIAL_MF
        ):
            candidates.append(idx)
    if not candidates:
        raise ValueError("Could not find the requested R(2) initial state.")
    return min(candidates, key=lambda idx: float(np.real(system.H_int[idx, idx])))


def r2_excited_indices(system: OBESystem) -> list[int]:
    return [
        idx
        for idx, state in enumerate(system.QN)
        if state.largest.electronic_state == states.ElectronicState.B
    ]


def r2_parameters(
    system: OBESystem,
    selectors: list[couplings.TransitionSelector],
) -> LindbladParameters:
    rabi_rad_s = power_to_rabi_rectangular_beam(
        POWER_MW * 1e-3,
        abs(system.couplings[0].main_coupling),
        BEAM_WX,
        BEAM_WY,
    )
    params = LindbladParameters()
    rabi = params.real(RABI_PARAMETER_NAME, rabi_rad_s)
    detuning = params.real(DETUNING_PARAMETER_NAME, 0.0)
    detuning_symbol = getattr(selectors[0], "\u03b4")  # noqa: B009
    for symbol in system.H_symbolic.free_symbols:
        if symbol in system.coupling_symbols:
            params.bind(symbol, rabi, finalize=False)
        elif symbol == detuning_symbol:
            params.bind(symbol, detuning, finalize=False)
        else:
            params.real(str(symbol), 0.0)
    for symbol_group in system.polarization_symbols:
        symbols = symbol_group if isinstance(symbol_group, list | tuple) else [symbol_group]
        for symbol in symbols:
            params.bind(symbol, 1.0, finalize=False)
    params._finalize()
    return params


def prepare_r2_model() -> BenchmarkModel:
    start = time.perf_counter()
    selectors = r2_transition_selectors()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Low overlap detected.*")
        system = lindblad.generate_OBE_system_transitions(
            [R2_TRANSITION],
            selectors,
            qn_compact=True,
            E=E_FIELD,
            B=B_FIELD,
            retain_opposite_parity_levels=True,
            method="matrix",
        )
    params = r2_parameters(system, selectors)
    prepared = prepare_lindblad_problem(
        system,
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho0 = np.zeros((len(system.QN), len(system.QN)), dtype=np.complex128)
    rho0[r2_initial_state_index(system), r2_initial_state_index(system)] = 1.0
    excited = r2_excited_indices(system)
    return BenchmarkModel(
        name="r2_retained_opposite_parity",
        system=system,
        prepared=prepared,
        rho0=rho0,
        t_span=(0.0, T_END_R2),
        dt=1e-9,
        reltol=1e-7,
        abstol=1e-9,
        rhs_iterations=100,
        rhs_repeats=5,
        solve_repeats=3,
        grid_repeats=2,
        detuning_symbol=DETUNING_PARAMETER_NAME,
        detuning_axis=2 * np.pi * 1e6 * np.linspace(-80.0, 80.0, 9),
        photon_integral_weights=[(int(idx), float(GAMMA)) for idx in excited],
        prep_seconds=time.perf_counter() - start,
    )


def packed_rho0(model: BenchmarkModel) -> np.ndarray:
    return np.ascontiguousarray(model.prepared.layout.pack(model.rho0), dtype=np.float64)


def rhs_once(model: BenchmarkModel, mode: str) -> np.ndarray:
    evaluator = create_lindblad_rhs_evaluator_py(model.prepared.rust_plan, mode)
    return np.asarray(evaluator.rhs_packed_py(packed_rho0(model), model.t_span[0]))


def benchmark_rhs_only(model: BenchmarkModel) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    validation_rows = []
    baseline = rhs_once(model, "expanded_sparse")
    for variant, mode in VARIANTS:
        evaluator = create_lindblad_rhs_evaluator_py(model.prepared.rust_plan, mode)
        state = packed_rho0(model)
        evaluator.rhs_packed_py(state, model.t_span[0])
        evaluator.reset_profile_py()
        evaluator.enable_profile_py(True)

        def run_loop(evaluator: Any = evaluator, state: np.ndarray = state) -> None:
            for _ in range(model.rhs_iterations):
                evaluator.rhs_packed_py(state, model.t_span[0])

        times, _ = timed_repeats(run_loop, model.rhs_repeats)
        profile = dict(evaluator.profile_summary_py())
        summary = summarize(times)
        per_call = summary["trimmed_mean_seconds"] / model.rhs_iterations
        rows.append(
            {
                "model": model.name,
                "variant": variant,
                "mode": mode,
                "iterations_per_repeat": model.rhs_iterations,
                **summary,
                "trimmed_mean_seconds_per_call": per_call,
                "rhs_calls_per_second": 1.0 / per_call,
                "rust_average_total_seconds": profile["average_total_seconds"],
                "rust_commutator_seconds_per_call": (
                    profile["commutator_seconds"] / profile["calls"]
                    if profile["calls"]
                    else np.nan
                ),
            }
        )
        current = rhs_once(model, mode)
        diff = current - baseline
        validation_rows.append(
            {
                "model": model.name,
                "variant": variant,
                "check": "rhs_vector",
                "max_abs_diff": float(np.max(np.abs(diff))),
                "max_rel_diff": float(
                    np.max(np.abs(diff)) / max(float(np.max(np.abs(baseline))), 1e-300)
                ),
            }
        )
    return rows, validation_rows


def solve_single(model: BenchmarkModel, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    _, values, width, stats = solve_lindblad_ode_py(
        model.prepared.rust_plan,
        packed_rho0(model),
        model.t_span[0],
        model.t_span[1],
        model.abstol,
        model.reltol,
        model.dt,
        None,
        True,
        100_000,
        mode,
        "dopri5",
        "photon_integral",
        None,
        "final",
        model.photon_integral_weights,
        None,
    )
    return np.asarray(values).reshape(-1, width), dict(stats)


def benchmark_single_solves(
    model: BenchmarkModel,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    validation_rows = []
    baseline_values, _ = solve_single(model, "expanded_sparse")
    baseline_photons = float(baseline_values[-1, 0])
    for variant, mode in VARIANTS:
        times, result = timed_repeats(
            lambda mode=mode: solve_single(model, mode), model.solve_repeats
        )
        values, stats = result
        photons = float(values[-1, 0])
        summary = summarize(times)
        rows.append(
            {
                "model": model.name,
                "variant": variant,
                "mode": mode,
                **summary,
                "accepted_steps": stats["accepted_steps"],
                "rejected_steps": stats["rejected_steps"],
                "rhs_calls": stats["rhs_calls"],
            }
        )
        validation_rows.append(
            {
                "model": model.name,
                "variant": variant,
                "check": "single_photon_integral",
                "value": photons,
                "baseline_value": baseline_photons,
                "max_abs_diff": abs(photons - baseline_photons),
                "max_rel_diff": abs(photons - baseline_photons)
                / max(abs(baseline_photons), 1e-300),
            }
        )
    return rows, validation_rows


def detuning_slot_index(model: BenchmarkModel) -> int:
    slot_names = list(model.prepared.parameter_graph["slot_names"])
    return slot_names.index(model.detuning_symbol)


def solve_grid(model: BenchmarkModel, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    _, values, width, time_count, stats = solve_lindblad_grid_ode_py(
        model.prepared.rust_plan,
        packed_rho0(model),
        model.t_span[0],
        model.t_span[1],
        model.abstol,
        model.reltol,
        model.dt,
        [detuning_slot_index(model)],
        np.ascontiguousarray(model.detuning_axis.astype(np.complex128)),
        [model.detuning_axis.size],
        None,
        True,
        100_000,
        mode,
        "dopri5",
        "photon_integral",
        None,
        "final",
        model.photon_integral_weights,
        True,
        4,
        None,
    )
    return np.asarray(values).reshape(model.detuning_axis.size, time_count, width)[:, -1, 0], dict(
        stats
    )


def benchmark_grids(model: BenchmarkModel) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    validation_rows = []
    baseline_line, _ = solve_grid(model, "expanded_sparse")
    baseline_peak_idx = int(np.argmax(baseline_line))
    baseline_norm = baseline_line / max(float(np.max(np.abs(baseline_line))), 1e-300)
    for variant, mode in VARIANTS:
        times, result = timed_repeats(
            lambda mode=mode: solve_grid(model, mode), model.grid_repeats
        )
        line, stats = result
        summary = summarize(times)
        rows.append(
            {
                "model": model.name,
                "variant": variant,
                "mode": mode,
                "trajectory_count": model.detuning_axis.size,
                "threads": 4,
                **summary,
                "trajectories_per_second": model.detuning_axis.size
                / summary["trimmed_mean_seconds"],
                "accepted_steps": stats["accepted_steps"],
                "rejected_steps": stats["rejected_steps"],
                "rhs_calls": stats["rhs_calls"],
            }
        )
        norm = line / max(float(np.max(np.abs(line))), 1e-300)
        peak_idx = int(np.argmax(line))
        validation_rows.append(
            {
                "model": model.name,
                "variant": variant,
                "check": "normalized_scan",
                "max_abs_diff": float(np.max(np.abs(norm - baseline_norm))),
                "max_rel_diff": float(
                    np.max(np.abs(line - baseline_line))
                    / max(float(np.max(np.abs(baseline_line))), 1e-300)
                ),
                "peak_axis_shift": float(
                    model.detuning_axis[peak_idx] - model.detuning_axis[baseline_peak_idx]
                ),
            }
        )
    return rows, validation_rows


def system_rows(models: list[BenchmarkModel]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        rows.append(
            {
                "model": model.name,
                "n_states": len(model.system.QN),
                "packed_len": model.prepared.layout.packed_len,
                "H_nnz": sum(1 for value in model.system.H_symbolic if value != 0),
                "C_ops": int(model.system.C_array.shape[0]),
                "C_nnz": int(np.count_nonzero(model.system.C_array)),
                "prep_seconds_not_timed": model.prep_seconds,
            }
        )
    return rows


def speedup_table(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    rows = []
    for model, group in df.groupby("model", sort=False):
        baseline = float(group.loc[group["variant"] == "baseline", value_column].iloc[0])
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "model": model,
                    "variant": row.variant,
                    "speedup_vs_baseline": baseline / float(getattr(row, value_column)),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rendered = df.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: f"{value:.6g}")
        else:
            rendered[column] = rendered[column].map(str)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rendered.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    systems: pd.DataFrame,
    rhs: pd.DataFrame,
    solves: pd.DataFrame,
    grids: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    rhs_speedups = speedup_table(rhs, "trimmed_mean_seconds_per_call")
    solve_speedups = speedup_table(solves, "trimmed_mean_seconds")
    grid_speedups = speedup_table(grids, "trimmed_mean_seconds")
    recommendation = (
        rhs_speedups.merge(
            solve_speedups,
            on=["model", "variant"],
            suffixes=("_rhs", "_single_solve"),
        )
        .merge(grid_speedups, on=["model", "variant"])
        .rename(columns={"speedup_vs_baseline": "speedup_vs_baseline_grid"})
    )
    recommendation["risk_complexity"] = recommendation["variant"].map(
        {
            "baseline": "none",
            "split_coefficients": "low",
            "split_inputs": "medium",
            "precomputed_inputs": "medium",
        }
    )
    recommendation["worth_implementing_signal"] = np.where(
        recommendation["speedup_vs_baseline_rhs"] >= 1.10,
        "candidate",
        "weak",
    )

    lines = [
        "# Expanded-Sparse Packed RHS Kernel Benchmark",
        "",
        "This report benchmarks experimental packed `expanded_sparse` RHS kernels. "
        "The default `expanded_sparse` implementation is the baseline; experimental modes are hidden and benchmark-only.",
        "",
        "## Systems",
        markdown_table(systems),
        "",
        "## RHS-Only Timing",
        markdown_table(
            rhs[
                [
                    "model",
                    "variant",
                    "trimmed_mean_seconds_per_call",
                    "rhs_calls_per_second",
                    "rust_commutator_seconds_per_call",
                ]
            ]
        ),
        "",
        "## Single-Trajectory Solve Timing",
        markdown_table(
            solves[
                [
                    "model",
                    "variant",
                    "trimmed_mean_seconds",
                    "accepted_steps",
                    "rhs_calls",
                ]
            ]
        ),
        "",
        "## Grid Timing",
        markdown_table(
            grids[
                [
                    "model",
                    "variant",
                    "trajectory_count",
                    "threads",
                    "trimmed_mean_seconds",
                    "trajectories_per_second",
                    "rhs_calls",
                ]
            ]
        ),
        "",
        "## Validation",
        markdown_table(validation),
        "",
        "## Recommendation Signals",
        markdown_table(recommendation),
        "",
        "Raw CSV files are written next to this report.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = [prepare_two_level_model(), prepare_r2_model()]
    rhs_rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    systems = pd.DataFrame(system_rows(models))
    systems.to_csv(SYSTEMS_PATH, index=False)
    for model in models:
        print(f"Benchmarking {model.name} ({len(model.system.QN)} states)", flush=True)
        rows, checks = benchmark_rhs_only(model)
        rhs_rows.extend(rows)
        validation_rows.extend(checks)
        pd.DataFrame(rhs_rows).to_csv(RHS_TIMINGS_PATH, index=False)
        pd.DataFrame(validation_rows).to_csv(VALIDATION_PATH, index=False)
        rows, checks = benchmark_single_solves(model)
        solve_rows.extend(rows)
        validation_rows.extend(checks)
        pd.DataFrame(solve_rows).to_csv(SOLVE_TIMINGS_PATH, index=False)
        pd.DataFrame(validation_rows).to_csv(VALIDATION_PATH, index=False)
        rows, checks = benchmark_grids(model)
        grid_rows.extend(rows)
        validation_rows.extend(checks)
        pd.DataFrame(grid_rows).to_csv(GRID_TIMINGS_PATH, index=False)
        pd.DataFrame(validation_rows).to_csv(VALIDATION_PATH, index=False)

    rhs = pd.DataFrame(rhs_rows)
    solves = pd.DataFrame(solve_rows)
    grids = pd.DataFrame(grid_rows)
    validation = pd.DataFrame(validation_rows)
    rhs.to_csv(RHS_TIMINGS_PATH, index=False)
    solves.to_csv(SOLVE_TIMINGS_PATH, index=False)
    grids.to_csv(GRID_TIMINGS_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    write_report(systems, rhs, solves, grids, validation)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
