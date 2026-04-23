from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


PATCH_POINTS_VCM = [0.0, 5.0, 7.0, 7.5, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0]
MASTER_FIELD_VCM = 10.0
B_FIELD = (0.0, 0.0, 1e-5)
RABI_RATE = 2.0 * np.pi * 1e6
DETUNING = 0.0
T_FINAL = 2e-6
T_EVAL = np.linspace(0.0, T_FINAL, 251)
ABSTOL = 1e-9
RELTOL = 1e-7
FIRST_STEP = 1e-10


def candidate_grid(name: str) -> np.ndarray:
    if name == "production":
        return np.asarray(PATCH_POINTS_VCM, dtype=np.float64)
    if name == "uniform_5":
        return np.linspace(0.0, 50.0, 11)
    if name == "uniform_2p5":
        return np.linspace(0.0, 50.0, 21)
    if name == "uniform_1p25":
        return np.linspace(0.0, 50.0, 41)
    raise ValueError(f"unknown candidate grid {name!r}")


def load_runtime_module():
    runtime_path = pathlib.Path(__file__).resolve().with_name("effective_hamiltonian_runtime.py")
    spec = importlib.util.spec_from_file_location("effective_hamiltonian_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def relative_error(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-30))


def uniform_density(n_states: int, indices: np.ndarray) -> np.ndarray:
    rho = np.zeros((n_states, n_states), dtype=np.complex128)
    indices = np.asarray(indices, dtype=np.int64)
    for index in indices:
        rho[int(index), int(index)] = 1.0 / int(indices.size)
    return rho


def interpolation_indices(grid: np.ndarray, value: float) -> tuple[int, int, float]:
    if value <= grid[0]:
        return 0, 0, 0.0
    if value >= grid[-1]:
        last = int(grid.size - 1)
        return last, last, 0.0
    upper = int(np.searchsorted(grid, value, side="right"))
    lower = upper - 1
    weight = (float(value) - float(grid[lower])) / float(grid[upper] - grid[lower])
    return lower, upper, float(weight)


def interpolate_matrix(grid: np.ndarray, values: Sequence[np.ndarray], value: float) -> np.ndarray:
    lower, upper, weight = interpolation_indices(grid, value)
    if lower == upper:
        return values[lower]
    return values[lower] + weight * (values[upper] - values[lower])


def interpolate_scalar(grid: np.ndarray, values: np.ndarray, value: float) -> float:
    lower, upper, weight = interpolation_indices(grid, value)
    if lower == upper:
        return float(values[lower])
    return float(values[lower] + weight * (values[upper] - values[lower]))


def build_candidate_model(ehr, field_points_vcm: Sequence[float] | np.ndarray = PATCH_POINTS_VCM):
    return ehr.prepare_lindblad_safe_compact_interpolated_model(
        field_points=np.asarray(field_points_vcm, dtype=np.float64),
        transition=ehr.transitions.Q1_F1_1o2_F0,
        optical_polarization=ehr.couplings.polarization_Z,
        magnetic_field=B_FIELD,
        master_field=MASTER_FIELD_VCM,
    )


def build_instantaneous_model(ehr, reference_fields_vcm: np.ndarray):
    return ehr.prepare_instantaneous_interpolated_effective_model(
        field_points=np.asarray(reference_fields_vcm, dtype=np.float64),
        transition=ehr.transitions.Q1_F1_1o2_F0,
        optical_polarization=ehr.couplings.polarization_Z,
        magnetic_field=B_FIELD,
        master_field=MASTER_FIELD_VCM,
    )


def precompute_candidate(ehr, model) -> dict[str, Any]:
    fields = np.asarray(model.field_points, dtype=np.float64)
    bundles = tuple(
        model.effective_bundle((0.0, 0.0, float(field)), model.reference_magnetic_field)
        for field in fields
    )
    return {
        "kind": "candidate_fixed_basis",
        "fields": fields,
        "liouvillians": tuple(
            bundle.liouvillian_superoperator(rabi_rate=RABI_RATE, detuning=DETUNING)
            for bundle in bundles
        ),
        "jump_rates": tuple(bundle.jump_rate_operator() for bundle in bundles),
        "ground_indices": np.asarray(model.ground_indices, dtype=np.int64),
        "sink_indices": np.asarray(model.sink_indices, dtype=np.int64),
        "excited_indices": np.asarray(model.excited_indices, dtype=np.int64),
        "sink_labels": tuple(getattr(model, "union_state_keys", ())),
        "n_states": int(model.n_effective_states),
    }


def precompute_instantaneous_reference(ehr, model) -> dict[str, Any]:
    fields = np.asarray(model.field_points, dtype=np.float64)
    bundles = tuple(patch.bundle for patch in model.patches)
    h_internal_superops = tuple(
        ehr._hamiltonian_superoperator(np.asarray(bundle.h_internal, dtype=np.complex128))
        for bundle in bundles
    )
    h_opt_superops = tuple(
        ehr._hamiltonian_superoperator(np.asarray(bundle.h_opt, dtype=np.complex128))
        for bundle in bundles
    )
    h_det_superops = tuple(
        ehr._hamiltonian_superoperator(np.asarray(bundle.h_det, dtype=np.complex128))
        for bundle in bundles
    )
    gauge_superops = tuple(
        ehr._hamiltonian_superoperator(-np.asarray(patch.gauge_connection, dtype=np.complex128))
        for patch in model.patches
    )
    dissipator_superops = tuple(
        np.asarray(bundle.dissipator_superoperator(), dtype=np.complex128)
        for bundle in bundles
    )
    omega_references = np.array([float(bundle.omega_reference) for bundle in bundles], dtype=np.float64)
    master_index = int(np.argmin(np.abs(fields - MASTER_FIELD_VCM)))
    common_omega_reference = float(omega_references[master_index])
    return {
        "kind": "instantaneous_with_gauge_and_sinks",
        "include_gauge": True,
        "fields": fields,
        "h_internal_superops": h_internal_superops,
        "h_opt_superops": h_opt_superops,
        "h_det_superops": h_det_superops,
        "gauge_superops": gauge_superops,
        "dissipator_superops": dissipator_superops,
        "jump_rates": tuple(bundle.jump_rate_operator() for bundle in bundles),
        "omega_references": omega_references,
        "common_omega_reference": common_omega_reference,
        "ground_indices": np.asarray(model.ground_indices, dtype=np.int64),
        "sink_indices": np.asarray(model.sink_indices, dtype=np.int64),
        "excited_indices": np.asarray(model.excited_indices, dtype=np.int64),
        "coherent_indices": np.asarray(model.coherent_indices, dtype=np.int64),
        "sink_labels": tuple(model.union_state_keys),
        "n_states": int(model.n_effective_states),
    }


def without_gauge(data: dict[str, Any]) -> dict[str, Any]:
    copied = dict(data)
    copied["include_gauge"] = False
    return copied


def candidate_liouvillian(data: dict[str, Any], field: float, field_dot: float) -> np.ndarray:
    del field_dot
    return interpolate_matrix(data["fields"], data["liouvillians"], field)


def instantaneous_liouvillian(data: dict[str, Any], field: float, field_dot: float) -> np.ndarray:
    fields = np.asarray(data["fields"], dtype=np.float64)
    detuning = DETUNING + (
        interpolate_scalar(fields, data["omega_references"], field)
        - float(data["common_omega_reference"])
    )
    gauge_term = (
        float(field_dot) * interpolate_matrix(fields, data["gauge_superops"], field)
        if bool(data.get("include_gauge", True))
        else 0.0
    )
    return (
        interpolate_matrix(fields, data["h_internal_superops"], field)
        + 0.5 * complex(RABI_RATE) * interpolate_matrix(fields, data["h_opt_superops"], field)
        + float(detuning) * interpolate_matrix(fields, data["h_det_superops"], field)
        + gauge_term
        + interpolate_matrix(fields, data["dissipator_superops"], field)
    )


def solve_time_dependent(
    data: dict[str, Any],
    field_at: Callable[[float], float],
    field_dot_at: Callable[[float], float],
    rho0: np.ndarray,
) -> tuple[Any, np.ndarray, float]:
    n_states = int(data["n_states"])
    liouvillian_at = (
        candidate_liouvillian if data["kind"] == "candidate_fixed_basis" else instantaneous_liouvillian
    )

    def rhs(t: float, rho_flat: np.ndarray) -> np.ndarray:
        field = float(field_at(float(t)))
        field_dot = float(field_dot_at(float(t)))
        return np.asarray(liouvillian_at(data, field, field_dot) @ rho_flat, dtype=np.complex128)

    start = time.perf_counter()
    solution = solve_ivp(
        rhs,
        (0.0, T_FINAL),
        y0=rho0.reshape(-1),
        t_eval=T_EVAL,
        method="BDF",
        atol=ABSTOL,
        rtol=RELTOL,
        first_step=FIRST_STEP,
    )
    elapsed = time.perf_counter() - start
    if not solution.success:
        raise RuntimeError(solution.message)
    rho_t = solution.y.T.reshape((-1, n_states, n_states))
    return solution, rho_t, elapsed


def trace_summary(
    data: dict[str, Any],
    field_at: Callable[[float], float],
    rho_t: np.ndarray,
) -> dict[str, Any]:
    fields = np.asarray(data["fields"], dtype=np.float64)
    jump_rates = data["jump_rates"]
    ground_indices = np.asarray(data["ground_indices"], dtype=np.int64)
    sink_indices = np.asarray(data["sink_indices"], dtype=np.int64)
    excited_indices = np.asarray(data["excited_indices"], dtype=np.int64)
    populations = np.real(np.diagonal(rho_t, axis1=1, axis2=2))
    ground_population = populations[:, ground_indices].sum(axis=1)
    sink_populations = populations[:, sink_indices] if sink_indices.size else np.zeros((rho_t.shape[0], 0))
    total_sink_population = sink_populations.sum(axis=1)
    excited_population = populations[:, excited_indices].sum(axis=1)
    active_population = ground_population + excited_population
    trace = np.real(np.trace(rho_t, axis1=1, axis2=2))
    rates = np.empty(T_EVAL.shape, dtype=np.float64)
    for idx, (t, rho) in enumerate(zip(T_EVAL, rho_t)):
        jump_rate = interpolate_matrix(fields, jump_rates, float(field_at(float(t))))
        rates[idx] = float(np.real(np.einsum("ij,ji->", rho, jump_rate)))
    return {
        "photons": float(np.trapezoid(rates, x=T_EVAL)),
        "excited_population_integral": float(np.trapezoid(excited_population, x=T_EVAL)),
        "final_excited_population": float(excited_population[-1]),
        "max_excited_population": float(np.max(excited_population)),
        "final_ground_population": float(ground_population[-1]),
        "final_active_population": float(active_population[-1]),
        "final_total_sink_population": float(total_sink_population[-1]),
        "final_sink_populations": [float(value) for value in sink_populations[-1]],
        "max_trace_error": float(np.max(np.abs(trace - 1.0))),
    }


def compare_case(
    name: str,
    field_at: Callable[[float], float],
    field_dot_at: Callable[[float], float],
    candidate_data: dict[str, Any],
    reference_data: dict[str, Any],
    rho0_candidate: np.ndarray,
    rho0_reference: np.ndarray,
) -> dict[str, Any]:
    cand_solution, cand_rho, cand_elapsed = solve_time_dependent(
        candidate_data,
        field_at,
        field_dot_at,
        rho0_candidate,
    )
    ref_solution, ref_rho, ref_elapsed = solve_time_dependent(
        reference_data,
        field_at,
        field_dot_at,
        rho0_reference,
    )
    cand_summary = trace_summary(candidate_data, field_at, cand_rho)
    ref_summary = trace_summary(reference_data, field_at, ref_rho)
    cand_excited = np.real(np.diagonal(cand_rho, axis1=1, axis2=2))[
        :,
        candidate_data["excited_indices"],
    ].sum(axis=1)
    ref_excited = np.real(np.diagonal(ref_rho, axis1=1, axis2=2))[
        :,
        reference_data["excited_indices"],
    ].sum(axis=1)
    max_excited_abs = float(np.max(np.abs(cand_excited - ref_excited)))
    return {
        "case": name,
        "candidate": cand_summary,
        "reference": ref_summary,
        "errors": {
            "photons_rel": relative_error(cand_summary["photons"], ref_summary["photons"]),
            "excited_integral_rel": relative_error(
                cand_summary["excited_population_integral"],
                ref_summary["excited_population_integral"],
            ),
            "final_excited_abs": float(
                cand_summary["final_excited_population"]
                - ref_summary["final_excited_population"]
            ),
            "final_excited_rel": relative_error(
                cand_summary["final_excited_population"],
                ref_summary["final_excited_population"],
            ),
            "max_excited_abs": max_excited_abs,
            "max_excited_rel": max_excited_abs / max(ref_summary["max_excited_population"], 1e-30),
            "final_active_abs": float(
                cand_summary["final_active_population"]
                - ref_summary["final_active_population"]
            ),
            "final_sink_abs": float(
                cand_summary["final_total_sink_population"]
                - ref_summary["final_total_sink_population"]
            ),
        },
        "solver": {
            "candidate_elapsed_s": float(cand_elapsed),
            "reference_elapsed_s": float(ref_elapsed),
            "candidate_nfev": int(cand_solution.nfev),
            "reference_nfev": int(ref_solution.nfev),
        },
    }


def solve_case_data(
    name: str,
    field_at: Callable[[float], float],
    field_dot_at: Callable[[float], float],
    data: dict[str, Any],
    rho0: np.ndarray,
) -> dict[str, Any]:
    solution, rho_t, elapsed = solve_time_dependent(data, field_at, field_dot_at, rho0)
    return {
        "case": name,
        "summary": trace_summary(data, field_at, rho_t),
        "rho_t": rho_t,
        "elapsed_s": float(elapsed),
        "nfev": int(solution.nfev),
    }


def compare_solved_cases(
    candidate_case: dict[str, Any],
    reference_case: dict[str, Any],
    candidate_data: dict[str, Any],
    reference_data: dict[str, Any],
) -> dict[str, Any]:
    cand_summary = candidate_case["summary"]
    ref_summary = reference_case["summary"]
    cand_rho = candidate_case["rho_t"]
    ref_rho = reference_case["rho_t"]
    cand_excited = np.real(np.diagonal(cand_rho, axis1=1, axis2=2))[
        :,
        candidate_data["excited_indices"],
    ].sum(axis=1)
    ref_excited = np.real(np.diagonal(ref_rho, axis1=1, axis2=2))[
        :,
        reference_data["excited_indices"],
    ].sum(axis=1)
    max_excited_abs = float(np.max(np.abs(cand_excited - ref_excited)))
    return {
        "case": candidate_case["case"],
        "candidate": cand_summary,
        "reference": ref_summary,
        "errors": {
            "photons_rel": relative_error(cand_summary["photons"], ref_summary["photons"]),
            "excited_integral_rel": relative_error(
                cand_summary["excited_population_integral"],
                ref_summary["excited_population_integral"],
            ),
            "final_excited_abs": float(
                cand_summary["final_excited_population"]
                - ref_summary["final_excited_population"]
            ),
            "final_excited_rel": relative_error(
                cand_summary["final_excited_population"],
                ref_summary["final_excited_population"],
            ),
            "max_excited_abs": max_excited_abs,
            "max_excited_rel": max_excited_abs / max(ref_summary["max_excited_population"], 1e-30),
            "final_active_abs": float(
                cand_summary["final_active_population"]
                - ref_summary["final_active_population"]
            ),
            "final_sink_abs": float(
                cand_summary["final_total_sink_population"]
                - ref_summary["final_total_sink_population"]
            ),
        },
        "solver": {
            "candidate_elapsed_s": float(candidate_case["elapsed_s"]),
            "reference_elapsed_s": float(reference_case["elapsed_s"]),
            "candidate_nfev": int(candidate_case["nfev"]),
            "reference_nfev": int(reference_case["nfev"]),
        },
    }


def default_cases() -> list[tuple[str, Callable[[float], float], Callable[[float], float]]]:
    return [
        (
            "linear_ramp_0_to_50_vcm",
            lambda t: 50.0 * float(t) / T_FINAL,
            lambda t: 50.0 / T_FINAL,
        ),
        (
            "sinusoid_25_pm_20_vcm",
            lambda t: 25.0 + 20.0 * np.sin(2.0 * np.pi * float(t) / T_FINAL),
            lambda t: 20.0 * (2.0 * np.pi / T_FINAL) * np.cos(2.0 * np.pi * float(t) / T_FINAL),
        ),
    ]


def make_initial_state(data: dict[str, Any]) -> np.ndarray:
    return uniform_density(
        int(data["n_states"]),
        np.asarray(data["ground_indices"], dtype=np.int64),
    )


def run_candidate_vs_reference(
    candidate_data: dict[str, Any],
    reference_data: dict[str, Any],
    solved_reference_cases: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rho0_candidate = make_initial_state(candidate_data)
    if solved_reference_cases is None:
        rho0_reference = make_initial_state(reference_data)
        solved_reference_cases = {
            name: solve_case_data(name, field_at, field_dot_at, reference_data, rho0_reference)
            for name, field_at, field_dot_at in default_cases()
        }
    results = []
    for name, field_at, field_dot_at in default_cases():
        candidate_case = solve_case_data(name, field_at, field_dot_at, candidate_data, rho0_candidate)
        results.append(
            compare_solved_cases(
                candidate_case,
                solved_reference_cases[name],
                candidate_data,
                reference_data,
            )
        )
    return results


def compare_references(
    lhs_name: str,
    lhs_data: dict[str, Any],
    rhs_name: str,
    rhs_data: dict[str, Any],
) -> list[dict[str, Any]]:
    results = run_candidate_vs_reference(lhs_data, rhs_data)
    for result in results:
        result["lhs_reference"] = lhs_name
        result["rhs_reference"] = rhs_name
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-points", type=int, default=41)
    parser.add_argument(
        "--candidate-grids",
        nargs="+",
        default=["production"],
        choices=["production", "uniform_5", "uniform_2p5", "uniform_1p25"],
    )
    parser.add_argument("--gauge-ablation", action="store_true")
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.reference_points < 3:
        raise ValueError("--reference-points must be at least 3")

    ehr = load_runtime_module()
    reference_fields_vcm = np.linspace(0.0, 50.0, int(args.reference_points))

    start = time.perf_counter()
    reference_model = build_instantaneous_model(ehr, reference_fields_vcm)
    reference_data = precompute_instantaneous_reference(ehr, reference_model)
    reference_setup_s = time.perf_counter() - start
    rho0_reference = make_initial_state(reference_data)
    solved_reference_cases = {
        name: solve_case_data(name, field_at, field_dot_at, reference_data, rho0_reference)
        for name, field_at, field_dot_at in default_cases()
    }

    candidate_outputs: list[dict[str, Any]] = []
    for grid_name in args.candidate_grids:
        fields = candidate_grid(grid_name)
        start = time.perf_counter()
        candidate_model = build_candidate_model(ehr, fields)
        candidate_data = precompute_candidate(ehr, candidate_model)
        candidate_setup_s = time.perf_counter() - start
        candidate_outputs.append(
            {
                "grid": grid_name,
                "setup_s": float(candidate_setup_s),
                "fields_vcm": np.asarray(candidate_data["fields"]).tolist(),
                "n_states": int(candidate_data["n_states"]),
                "sink_labels": list(candidate_data["sink_labels"]),
                "results": run_candidate_vs_reference(
                    candidate_data,
                    reference_data,
                    solved_reference_cases=solved_reference_cases,
                ),
            }
        )

    reference_ablations: list[dict[str, Any]] = []
    if args.gauge_ablation:
        reference_ablations.append(
            {
                "comparison": "instantaneous_gauge_vs_no_gauge",
                "results": compare_references(
                    "with_gauge",
                    reference_data,
                    "without_gauge",
                    without_gauge(reference_data),
                ),
            }
        )

    output = {
        "setup": {
            "reference_setup_s": float(reference_setup_s),
            "n_reference_states": int(reference_data["n_states"]),
            "reference_fields_vcm": np.asarray(reference_data["fields"]).tolist(),
            "reference_sink_labels": list(reference_data["sink_labels"]),
        },
        "candidate_outputs": candidate_outputs,
        "reference_ablations": reference_ablations,
    }
    print(json.dumps(output, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
