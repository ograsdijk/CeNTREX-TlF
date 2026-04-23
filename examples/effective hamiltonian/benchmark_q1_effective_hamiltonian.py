from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from centrex_tlf.lindblad import prepare_lindblad_problem, solve_lindblad


RABI_RATE = 2.0 * np.pi * 1e6
DETUNING = 0.0
T_FINAL = 2e-6
T_EVAL = np.linspace(0.0, T_FINAL, 251)
ABSTOL = 1e-9
RELTOL = 1e-7
DT = 1e-10
MAXITERS = 1_000_000
B_FIELD = (0.0, 0.0, 1e-5)
MASTER_FIELD_VCM = 10.0
POLARIZATION_SCALE = 1.0


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime_modules():
    here = pathlib.Path(__file__).resolve().parent
    ehr = load_module(here / "effective_hamiltonian_runtime.py", "effective_hamiltonian_runtime")
    validation = load_module(
        here / "validate_q1_instantaneous_sink_truth.py",
        "validate_q1_instantaneous_sink_truth",
    )
    return ehr, validation


def median(values: Sequence[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def relative_error(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-30))


def active_ground_indices_from_qn(ehr, transition, qn_list) -> np.ndarray:
    indices: list[int] = []
    for idx, qn in enumerate(qn_list):
        state = qn.largest
        if state.electronic_state == transition.electronic_ground and state.J == int(
            transition.J_ground
        ):
            indices.append(idx)
    if not indices:
        raise RuntimeError(f"no active X,J={transition.J_ground} ground states found")
    return np.asarray(indices, dtype=np.int64)


def uniform_density(n_states: int, indices: np.ndarray) -> np.ndarray:
    rho = np.zeros((n_states, n_states), dtype=np.complex128)
    indices = np.asarray(indices, dtype=np.int64)
    for index in indices:
        rho[int(index), int(index)] = 1.0 / int(indices.size)
    return rho


def compact_parameters(system, *, rabi: float | complex = RABI_RATE, detuning: float = DETUNING):
    omega_char = chr(0x03A9)
    delta_char = chr(0x03B4)
    params: dict[str, float | complex] = {}
    for symbol in system.H_symbolic.free_symbols:
        name = str(symbol)
        if name.startswith(omega_char):
            params[name] = complex(rabi)
        elif name.startswith(delta_char):
            params[name] = float(detuning)
        elif name.startswith("PZ"):
            params[name] = float(POLARIZATION_SCALE)
        elif name.startswith("PX") or name.startswith("PY"):
            params[name] = 0.0
        else:
            params[name] = 0.0
    return params


def trace_summary(
    ehr,
    times: np.ndarray,
    rho_t: np.ndarray,
    excited_indices: np.ndarray,
    jump_rate_operator: np.ndarray,
) -> dict[str, float]:
    populations = np.real(np.diagonal(rho_t, axis1=1, axis2=2))
    excited_population = populations[:, np.asarray(excited_indices, dtype=np.int64)].sum(axis=1)
    rates = ehr.scattering_signal(rho_t, jump_rate_operator)
    trace = np.real(np.trace(rho_t, axis1=1, axis2=2))
    return {
        "photons": float(np.trapezoid(rates, x=times)),
        "excited_integral": float(np.trapezoid(excited_population, x=times)),
        "final_excited": float(excited_population[-1]),
        "max_excited": float(np.max(excited_population)),
        "max_trace_error": float(np.max(np.abs(trace - 1.0))),
    }


def time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, float(time.perf_counter() - start)


def solve_dense_liouvillian(
    data: dict[str, Any],
    field_at: Callable[[float], float],
    field_dot_at: Callable[[float], float],
    rho0: np.ndarray,
    *,
    method: str,
    validation,
):
    n_states = int(data["n_states"])
    liouvillian_at = (
        validation.candidate_liouvillian
        if data["kind"] == "candidate_fixed_basis"
        else validation.instantaneous_liouvillian
    )

    def rhs(t: float, rho_flat: np.ndarray) -> np.ndarray:
        field = float(field_at(float(t)))
        field_dot = float(field_dot_at(float(t)))
        return np.asarray(liouvillian_at(data, field, field_dot) @ rho_flat, dtype=np.complex128)

    solution = solve_ivp(
        rhs,
        (0.0, T_FINAL),
        y0=np.asarray(rho0, dtype=np.complex128).reshape(-1),
        t_eval=T_EVAL,
        method=method,
        atol=ABSTOL,
        rtol=RELTOL,
        first_step=DT,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    rho_t = solution.y.T.reshape((-1, n_states, n_states))
    return solution, rho_t


def collapse_sparsity(c_array: np.ndarray) -> dict[str, int]:
    nnz = [int(np.count_nonzero(np.abs(collapse) > 1e-14)) for collapse in c_array]
    return {
        "n_collapse_operators": int(len(nnz)),
        "min_nnz": int(min(nnz, default=0)),
        "median_nnz": int(statistics.median(nnz)) if nnz else 0,
        "max_nnz": int(max(nnz, default=0)),
        "n_single_jump": int(sum(value == 1 for value in nnz)),
    }


def benchmark_time_dependent(
    ehr,
    validation,
    *,
    reference_points: int,
    candidate_grid: str,
    repeats: int,
    include_rk45: bool,
    case_filter: set[str] | None,
) -> dict[str, Any]:
    reference_fields = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 50.0, int(reference_points)),
                np.asarray([MASTER_FIELD_VCM], dtype=np.float64),
            )
        )
    )
    reference_model, reference_setup_s = time_call(
        lambda: validation.build_instantaneous_model(ehr, reference_fields)
    )
    reference_data, reference_precompute_s = time_call(
        lambda: validation.precompute_instantaneous_reference(ehr, reference_model)
    )

    candidate_fields = validation.candidate_grid(candidate_grid)
    candidate_model, candidate_setup_s = time_call(
        lambda: validation.build_candidate_model(ehr, candidate_fields)
    )
    candidate_data, candidate_precompute_s = time_call(
        lambda: validation.precompute_candidate(ehr, candidate_model)
    )
    candidate_sparsity = collapse_sparsity(candidate_model.patches[0].bundle.c_array)
    interpolated_sparsity = collapse_sparsity(
        candidate_model.effective_bundle((0.0, 0.0, 25.0), B_FIELD).c_array
    )

    rows: list[dict[str, Any]] = []
    for case_name, field_at, field_dot_at in validation.default_cases():
        if case_filter is not None and case_name not in case_filter:
            continue
        reference_runs = []
        candidate_runs = []
        candidate_rk_runs = []
        reference_summary = None
        candidate_summary = None
        for _ in range(int(repeats)):
            rho0_reference = validation.make_initial_state(reference_data)
            (reference_solution, reference_rho), reference_elapsed = time_call(
                lambda: solve_dense_liouvillian(
                    reference_data,
                    field_at,
                    field_dot_at,
                    rho0_reference,
                    method="BDF",
                    validation=validation,
                )
            )
            reference_runs.append(reference_elapsed)
            reference_summary = validation.trace_summary(reference_data, field_at, reference_rho)

            rho0_candidate = validation.make_initial_state(candidate_data)
            (candidate_solution, candidate_rho), candidate_elapsed = time_call(
                lambda: solve_dense_liouvillian(
                    candidate_data,
                    field_at,
                    field_dot_at,
                    rho0_candidate,
                    method="BDF",
                    validation=validation,
                )
            )
            candidate_runs.append(candidate_elapsed)
            candidate_summary = validation.trace_summary(candidate_data, field_at, candidate_rho)

            candidate_rk_solution = None
            if include_rk45:
                (candidate_rk_solution, _candidate_rk_rho), candidate_rk_elapsed = time_call(
                    lambda: solve_dense_liouvillian(
                        candidate_data,
                        field_at,
                        field_dot_at,
                        rho0_candidate,
                        method="RK45",
                        validation=validation,
                    )
                )
                candidate_rk_runs.append(candidate_rk_elapsed)

        assert reference_summary is not None
        assert candidate_summary is not None
        row = {
            "case": case_name,
            "candidate_fixed_dense_bdf_median_s": median(candidate_runs),
            "reference_instantaneous_dense_bdf_median_s": median(reference_runs),
            "fixed_vs_reference_speedup_bdf": median(reference_runs) / median(candidate_runs),
            "photons_rel_vs_reference": relative_error(
                candidate_summary["photons"],
                reference_summary["photons"],
            ),
            "excited_integral_rel_vs_reference": relative_error(
                candidate_summary["excited_population_integral"],
                reference_summary["excited_population_integral"],
            ),
            "candidate_nfev_bdf": int(candidate_solution.nfev),
            "reference_nfev_bdf": int(reference_solution.nfev),
        }
        if include_rk45 and candidate_rk_runs and candidate_rk_solution is not None:
            row.update(
                {
                    "candidate_fixed_dense_rk45_median_s": median(candidate_rk_runs),
                    "fixed_rk45_vs_reference_bdf_speedup": median(reference_runs)
                    / median(candidate_rk_runs),
                    "candidate_nfev_rk45": int(candidate_rk_solution.nfev),
                }
            )
        rows.append(row)

    return {
        "reference_points": int(reference_points),
        "candidate_grid": candidate_grid,
        "reference_setup_s": float(reference_setup_s),
        "reference_precompute_s": float(reference_precompute_s),
        "candidate_setup_s": float(candidate_setup_s),
        "candidate_precompute_s": float(candidate_precompute_s),
        "candidate_patch_collapse_sparsity": candidate_sparsity,
        "candidate_interpolated_collapse_sparsity_at_25_vcm": interpolated_sparsity,
        "rust_fixed_basis_time_dependent_status": (
            "not benchmarked: the fixed-basis recycling-kernel factorization produces dense "
            "collapse operators, while the current Rust planner requires single-jump collapse "
            "operators for structured/expanded_sparse RHS lowering"
        ),
        "cases": rows,
    }


def benchmark_static_rust_baseline(
    ehr,
    *,
    static_field: float,
    repeats: int,
) -> dict[str, Any]:
    transition = ehr.transitions.Q1_F1_1o2_F0
    (system, bundle), setup_s = time_call(
        lambda: ehr.build_compact_reference_decomposed_bundle(
            transition=transition,
            optical_polarization=ehr.couplings.polarization_Z,
            electric_field=(0.0, 0.0, float(static_field)),
            magnetic_field=B_FIELD,
            polarization_scale=POLARIZATION_SCALE,
        )
    )
    parameters = compact_parameters(system)
    prepared, prepare_s = time_call(
        lambda: prepare_lindblad_problem(
            system,
            parameters,
            backend="rust",
            hamiltonian_representation="decomposed",
        )
    )
    ground_indices = active_ground_indices_from_qn(ehr, transition, system.QN)
    rho0 = uniform_density(len(system.QN), ground_indices)
    jump_rate_operator = bundle.jump_rate_operator()
    excited_indices = np.asarray(bundle.excited_indices, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    reference_summary = None
    for solver in ("dopri5_fast", "tsit5_fast"):
        elapsed_values: list[float] = []
        last_result = None
        for _ in range(int(repeats)):
            result, elapsed = time_call(
                lambda solver=solver: solve_lindblad(
                    prepared,
                    rho0,
                    (0.0, T_FINAL),
                    backend="rust",
                    solver=solver,
                    execution_mode="expanded_sparse",
                    abstol=ABSTOL,
                    reltol=RELTOL,
                    dt=DT,
                    saveat=T_EVAL,
                    maxiters=MAXITERS,
                    collect_stats=True,
                    dense_output=True,
                )
            )
            elapsed_values.append(elapsed)
            last_result = result
        assert last_result is not None
        rho_t = last_result.density_matrices()
        summary = trace_summary(ehr, last_result.t, rho_t, excited_indices, jump_rate_operator)
        if reference_summary is None:
            reference_summary = summary
        rows.append(
            {
                "solver": solver,
                "median_s": median(elapsed_values),
                "accepted_steps": int((last_result.solver_stats or {}).get("accepted_steps", -1)),
                "rejected_steps": int((last_result.solver_stats or {}).get("rejected_steps", -1)),
                "internal_steps": int((last_result.solver_stats or {}).get("internal_steps", -1)),
                "photons_rel_vs_dopri5_fast": relative_error(
                    summary["photons"],
                    reference_summary["photons"],
                ),
                "excited_integral_rel_vs_dopri5_fast": relative_error(
                    summary["excited_integral"],
                    reference_summary["excited_integral"],
                ),
                "max_trace_error": summary["max_trace_error"],
            }
        )

    return {
        "field_vcm": float(static_field),
        "setup_s": float(setup_s),
        "prepare_rust_s": float(prepare_s),
        "n_states": int(len(system.QN)),
        "collapse_sparsity": collapse_sparsity(np.asarray(system.C_array, dtype=np.complex128)),
        "note": (
            "static compact sparse-jump Rust baseline; this is not the time-dependent "
            "fixed-basis recycling-kernel model"
        ),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-points", type=int, default=11)
    parser.add_argument(
        "--candidate-grid",
        default="production",
        choices=["production", "uniform_5", "uniform_2p5", "uniform_1p25"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--static-field", type=float, default=25.0)
    parser.add_argument("--skip-static-rust", action="store_true")
    parser.add_argument("--include-rk45", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        choices=["linear_ramp_0_to_50_vcm", "sinusoid_25_pm_20_vcm"],
        default=None,
        help="limit to one case; can be supplied more than once",
    )
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    ehr, validation = load_runtime_modules()
    output: dict[str, Any] = {
        "config": {
            "transition": "Q1_F1_1o2_F0",
            "t_final_s": T_FINAL,
            "n_t_eval": int(T_EVAL.size),
            "rabi_rate_rad_s": float(RABI_RATE),
            "detuning_rad_s": float(DETUNING),
            "abstol": ABSTOL,
            "reltol": RELTOL,
            "dt": DT,
            "repeats": int(args.repeats),
        },
        "time_dependent": benchmark_time_dependent(
            ehr,
            validation,
            reference_points=int(args.reference_points),
            candidate_grid=args.candidate_grid,
            repeats=int(args.repeats),
            include_rk45=bool(args.include_rk45),
            case_filter=None if args.case is None else set(args.case),
        ),
    }
    if not args.skip_static_rust:
        output["static_rust_baseline"] = benchmark_static_rust_baseline(
            ehr,
            static_field=float(args.static_field),
            repeats=int(args.repeats),
        )

    text = json.dumps(output, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
