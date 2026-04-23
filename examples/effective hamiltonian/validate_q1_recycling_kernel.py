from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


TRANSITION_NAME = "Q1_F1_1o2_F0"
PATCH_POINTS_VCM = [0.0, 5.0, 7.0, 7.5, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0]
STATIC_FIELDS_VCM = [0.0, 7.5, 10.0, 20.0, 50.0]
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
    script_path = pathlib.Path(__file__).resolve()
    runtime_path = script_path.with_name("effective_hamiltonian_runtime.py")
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
    for index in np.asarray(indices, dtype=np.int64):
        rho[int(index), int(index)] = 1.0 / int(indices.size)
    return rho


def solve_bundle(bundle, rho0: np.ndarray):
    n_states = int(rho0.shape[0])
    liouvillian = bundle.liouvillian_superoperator(
        rabi_rate=RABI_RATE,
        detuning=DETUNING,
    )

    def rhs(_, rho_flat):
        return liouvillian @ rho_flat

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
    return rho_t, elapsed, int(solution.nfev)


def trace_summary(ehr, rho_t: np.ndarray, bundle) -> dict[str, float]:
    populations = np.real(np.diagonal(rho_t, axis1=1, axis2=2))
    excited_population = populations[:, np.asarray(bundle.excited_indices, dtype=np.int64)].sum(
        axis=1
    )
    rates = ehr.scattering_signal(rho_t, bundle.jump_rate_operator())
    trace = np.real(np.trace(rho_t, axis1=1, axis2=2))
    return {
        "photons": float(np.trapezoid(rates, x=T_EVAL)),
        "excited_population_integral": float(np.trapezoid(excited_population, x=T_EVAL)),
        "final_excited_population": float(excited_population[-1]),
        "max_trace_error": float(np.max(np.abs(trace - 1.0))),
    }


def build_model(ehr):
    transition = getattr(ehr.transitions, TRANSITION_NAME)
    return ehr.prepare_lindblad_safe_compact_interpolated_model(
        field_points=PATCH_POINTS_VCM,
        transition=transition,
        optical_polarization=ehr.couplings.polarization_Z,
        magnetic_field=B_FIELD,
        master_field=MASTER_FIELD_VCM,
    )


def operator_diagnostics(ehr, model) -> list[dict[str, Any]]:
    active_indices = np.concatenate([model.ground_indices, model.excited_indices]).astype(np.int64)
    patch_index_by_field = {
        float(field): index for index, field in enumerate(np.asarray(model.field_points, dtype=float))
    }
    rows: list[dict[str, Any]] = []
    for field_vcm in STATIC_FIELDS_VCM:
        electric = (0.0, 0.0, float(field_vcm))
        interp_bundle = model.effective_bundle(electric, B_FIELD)
        _, aligned_bundle = ehr._aligned_exact_compact_bundle_for_field(model, electric)
        interp_dissipator = interp_bundle.dissipator_superoperator()
        aligned_c_array_dissipator = ehr._dissipator_superoperator(aligned_bundle.c_array)
        interp_jump_rate = interp_bundle.jump_rate_operator()
        aligned_jump_rate = aligned_bundle.jump_rate_operator()
        row = {
            "field_vcm": float(field_vcm),
            "n_kernel_jumps": int(interp_bundle.c_array.shape[0]),
            "uses_dissipator_superop_override": interp_bundle.dissipator_superop is not None,
            "aligned_loss_operator_norm": float(np.linalg.norm(aligned_bundle.loss_operator)),
            "interp_loss_operator_norm": float(np.linalg.norm(interp_bundle.loss_operator)),
            "h_internal_active_rel": float(
                np.linalg.norm(
                    interp_bundle.h_internal[np.ix_(active_indices, active_indices)]
                    - aligned_bundle.h_internal[np.ix_(active_indices, active_indices)]
                )
                / max(
                    np.linalg.norm(
                        aligned_bundle.h_internal[np.ix_(active_indices, active_indices)]
                    ),
                    1.0,
                )
            ),
            "h_opt_rel": float(
                np.linalg.norm(interp_bundle.h_opt - aligned_bundle.h_opt)
                / max(np.linalg.norm(aligned_bundle.h_opt), 1.0)
            ),
            "h_det_rel": float(
                np.linalg.norm(interp_bundle.h_det - aligned_bundle.h_det)
                / max(np.linalg.norm(aligned_bundle.h_det), 1.0)
            ),
            "dissipator_rel_vs_aligned_c_array": float(
                np.linalg.norm(interp_dissipator - aligned_c_array_dissipator)
                / max(np.linalg.norm(aligned_c_array_dissipator), 1.0)
            ),
            "jump_rate_rel": float(
                np.linalg.norm(interp_jump_rate - aligned_jump_rate)
                / max(np.linalg.norm(aligned_jump_rate), 1.0)
            ),
        }
        patch_index = patch_index_by_field.get(float(field_vcm))
        if patch_index is not None:
            patch = model.patches[patch_index]
            rebuilt_c_array = ehr._c_array_from_full_recycling_decay_kernel(
                target_indices=np.arange(model.n_effective_states, dtype=np.int64),
                source_indices=np.arange(model.n_effective_states, dtype=np.int64),
                kernel=patch.full_recycling_decay_kernel,
                total_dimension=model.n_effective_states,
            )
            rebuilt_dissipator = ehr._dissipator_superoperator(rebuilt_c_array)
            patch_c_array_dissipator = ehr._dissipator_superoperator(patch.bundle.c_array)
            evals = np.linalg.eigvalsh(patch.full_recycling_decay_kernel)
            row.update(
                {
                    "patch_rebuild_vs_c_array_dissipator_rel": float(
                        np.linalg.norm(rebuilt_dissipator - patch_c_array_dissipator)
                        / max(np.linalg.norm(patch_c_array_dissipator), 1.0)
                    ),
                    "patch_full_kernel_min_eig": float(np.min(np.real(evals))),
                    "patch_full_kernel_max_eig": float(np.max(np.real(evals))),
                }
            )
        rows.append(row)
    return rows


def dynamics_diagnostics(ehr, model) -> list[dict[str, Any]]:
    rho0 = uniform_density(model.n_effective_states, np.asarray(model.ground_indices))
    rows: list[dict[str, Any]] = []
    for field_vcm in STATIC_FIELDS_VCM:
        electric = (0.0, 0.0, float(field_vcm))
        interp_bundle = model.effective_bundle(electric, B_FIELD)
        _, aligned_bundle = ehr._aligned_exact_compact_bundle_for_field(model, electric)
        rho_interp, interp_elapsed, interp_nfev = solve_bundle(interp_bundle, rho0)
        rho_aligned, aligned_elapsed, aligned_nfev = solve_bundle(aligned_bundle, rho0)
        interp_summary = trace_summary(ehr, rho_interp, interp_bundle)
        aligned_summary = trace_summary(ehr, rho_aligned, aligned_bundle)
        rows.append(
            {
                "field_vcm": float(field_vcm),
                "photons_rel": relative_error(
                    interp_summary["photons"],
                    aligned_summary["photons"],
                ),
                "excited_integral_rel": relative_error(
                    interp_summary["excited_population_integral"],
                    aligned_summary["excited_population_integral"],
                ),
                "final_excited_abs": float(
                    interp_summary["final_excited_population"]
                    - aligned_summary["final_excited_population"]
                ),
                "final_excited_rel": relative_error(
                    interp_summary["final_excited_population"],
                    aligned_summary["final_excited_population"],
                ),
                "interp_elapsed_s": float(interp_elapsed),
                "aligned_elapsed_s": float(aligned_elapsed),
                "interp_nfev": int(interp_nfev),
                "aligned_nfev": int(aligned_nfev),
            }
        )
    return rows


def max_abs(rows: list[dict[str, Any]], key: str) -> float:
    return float(max(abs(float(row[key])) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics", action="store_true", help="also run BDF dynamics checks")
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    ehr = load_runtime_module()
    start = time.perf_counter()
    model = build_model(ehr)
    setup_s = time.perf_counter() - start

    operators = operator_diagnostics(ehr, model)
    result: dict[str, Any] = {
        "setup_s": float(setup_s),
        "n_effective_states": int(model.n_effective_states),
        "operator_diagnostics": operators,
        "operator_max_abs": {
            "dissipator_rel_vs_aligned_c_array": max_abs(
                operators,
                "dissipator_rel_vs_aligned_c_array",
            ),
            "jump_rate_rel": max_abs(operators, "jump_rate_rel"),
            "h_internal_active_rel": max_abs(operators, "h_internal_active_rel"),
        },
    }
    if args.dynamics:
        dynamics = dynamics_diagnostics(ehr, model)
        result["dynamics_diagnostics"] = dynamics
        result["dynamics_max_abs"] = {
            "photons_rel": max_abs(dynamics, "photons_rel"),
            "excited_integral_rel": max_abs(dynamics, "excited_integral_rel"),
            "final_excited_rel": max_abs(dynamics, "final_excited_rel"),
        }

    print(json.dumps(result, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
