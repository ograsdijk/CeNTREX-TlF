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


def load_runtime_module():
    runtime_path = pathlib.Path(__file__).resolve().with_name("effective_hamiltonian_runtime.py")
    spec = importlib.util.spec_from_file_location("effective_hamiltonian_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def uniform_density(n_states: int, indices: np.ndarray) -> np.ndarray:
    rho = np.zeros((n_states, n_states), dtype=np.complex128)
    for index in np.asarray(indices, dtype=np.int64):
        rho[int(index), int(index)] = 1.0 / int(indices.size)
    return rho


def relative_error(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1e-30))


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


def build_model(ehr):
    return ehr.prepare_lindblad_safe_compact_interpolated_model(
        field_points=PATCH_POINTS_VCM,
        transition=ehr.transitions.Q1_F1_1o2_F0,
        optical_polarization=ehr.couplings.polarization_Z,
        magnetic_field=B_FIELD,
        master_field=MASTER_FIELD_VCM,
    )


def bundle_liouvillian(bundle) -> np.ndarray:
    return bundle.liouvillian_superoperator(
        rabi_rate=RABI_RATE,
        detuning=DETUNING,
    )


def precompute_candidate(ehr, model):
    fields = np.asarray(model.field_points, dtype=np.float64)
    bundles = tuple(
        model.effective_bundle((0.0, 0.0, float(field)), model.reference_magnetic_field)
        for field in fields
    )
    return {
        "fields": fields,
        "liouvillians": tuple(bundle_liouvillian(bundle) for bundle in bundles),
        "jump_rates": tuple(bundle.jump_rate_operator() for bundle in bundles),
    }


def precompute_reference(ehr, model, fields: np.ndarray):
    bundles = []
    for field in fields.tolist():
        _, bundle = ehr._aligned_exact_compact_bundle_for_field(
            model,
            (0.0, 0.0, float(field)),
        )
        bundles.append(bundle)
    return {
        "fields": np.asarray(fields, dtype=np.float64),
        "liouvillians": tuple(bundle_liouvillian(bundle) for bundle in bundles),
        "jump_rates": tuple(bundle.jump_rate_operator() for bundle in bundles),
    }


def solve_time_dependent(
    data: dict[str, Any],
    field_at: Callable[[float], float],
    rho0: np.ndarray,
) -> tuple[Any, np.ndarray, float]:
    n_states = int(rho0.shape[0])
    fields = np.asarray(data["fields"], dtype=np.float64)
    liouvillians = data["liouvillians"]

    def rhs(t: float, rho_flat: np.ndarray) -> np.ndarray:
        field = float(field_at(float(t)))
        liouvillian = interpolate_matrix(fields, liouvillians, field)
        return np.asarray(liouvillian @ rho_flat, dtype=np.complex128)

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
    excited_indices: np.ndarray,
) -> dict[str, float]:
    fields = np.asarray(data["fields"], dtype=np.float64)
    jump_rates = data["jump_rates"]
    populations = np.real(np.diagonal(rho_t, axis1=1, axis2=2))
    excited_population = populations[:, np.asarray(excited_indices, dtype=np.int64)].sum(axis=1)
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
        "max_trace_error": float(np.max(np.abs(trace - 1.0))),
    }


def compare_case(
    name: str,
    field_at: Callable[[float], float],
    candidate_data: dict[str, Any],
    reference_data: dict[str, Any],
    rho0: np.ndarray,
    excited_indices: np.ndarray,
) -> dict[str, Any]:
    cand_solution, cand_rho, cand_elapsed = solve_time_dependent(candidate_data, field_at, rho0)
    ref_solution, ref_rho, ref_elapsed = solve_time_dependent(reference_data, field_at, rho0)
    cand_summary = trace_summary(candidate_data, field_at, cand_rho, excited_indices)
    ref_summary = trace_summary(reference_data, field_at, ref_rho, excited_indices)
    excited_cand = np.real(np.diagonal(cand_rho, axis1=1, axis2=2))[:, excited_indices].sum(axis=1)
    excited_ref = np.real(np.diagonal(ref_rho, axis1=1, axis2=2))[:, excited_indices].sum(axis=1)
    max_excited_abs = float(np.max(np.abs(excited_cand - excited_ref)))
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
        },
        "solver": {
            "candidate_elapsed_s": float(cand_elapsed),
            "reference_elapsed_s": float(ref_elapsed),
            "candidate_nfev": int(cand_solution.nfev),
            "reference_nfev": int(ref_solution.nfev),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-points", type=int, default=21)
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.reference_points < 2:
        raise ValueError("--reference-points must be at least 2")
    reference_fields_vcm = np.linspace(0.0, 50.0, int(args.reference_points))

    ehr = load_runtime_module()
    setup_start = time.perf_counter()
    model = build_model(ehr)
    candidate_data = precompute_candidate(ehr, model)
    candidate_setup_s = time.perf_counter() - setup_start

    reference_start = time.perf_counter()
    reference_data = precompute_reference(ehr, model, reference_fields_vcm)
    reference_setup_s = time.perf_counter() - reference_start

    rho0 = uniform_density(model.n_effective_states, np.asarray(model.ground_indices))
    excited_indices = np.asarray(model.excited_indices, dtype=np.int64)

    cases = [
        (
            "linear_ramp_0_to_50_vcm",
            lambda t: 50.0 * float(t) / T_FINAL,
        ),
        (
            "sinusoid_25_pm_20_vcm",
            lambda t: 25.0 + 20.0 * np.sin(2.0 * np.pi * float(t) / T_FINAL),
        ),
    ]
    results = [
        compare_case(
            name,
            field_at,
            candidate_data,
            reference_data,
            rho0,
            excited_indices,
        )
        for name, field_at in cases
    ]

    output = {
        "setup": {
            "candidate_setup_s": float(candidate_setup_s),
            "reference_setup_s": float(reference_setup_s),
            "n_effective_states": int(model.n_effective_states),
            "candidate_fields_vcm": np.asarray(candidate_data["fields"]).tolist(),
            "reference_fields_vcm": np.asarray(reference_data["fields"]).tolist(),
        },
        "results": results,
    }
    print(json.dumps(output, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
