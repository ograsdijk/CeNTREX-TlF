"""Pilot a matrix-free repeated-period Krylov approximation for full RC rho.

This is a density-only action: the full packed Hermitian state is propagated
through one complete 5 us period on every Arnoldi action.  It deliberately
does not augment photon yield, so its cost describes a final-density request.
The 200 us reference is a pairwise comparison only, not an independently
validated convergence reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bench_rc_propagation as driver
import numpy as np
import run_rc_observables as observables
from rc_periodic_krylov import ArnoldiPowerResult, arnoldi_repeated_action
from run_rc_comparison import PeakMemory, resource_snapshot
from threadpoolctl import threadpool_limits

PERIOD_S = 5e-6
INITIAL_LABEL = "X_J2_F1_5o2_F2_mF0"
DEFAULT_CHECKPOINTS = (4, 8, 12, 16, 24)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_paths(metadata_path: Path) -> tuple[Path, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tag = metadata_path.stem
    density_path = metadata_path.parent / f"{tag}_densities.npz"
    if not density_path.is_file():
        raise FileNotFoundError(f"density artifact not found beside metadata: {density_path}")
    return density_path, metadata


def load_endpoint_artifact(
    metadata_path: Path,
    config: driver.BenchmarkConfig,
    *,
    expected_n_states: int,
    expected_solver: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate the full-density endpoint from a single-run artifact."""
    density_path, metadata = _artifact_paths(metadata_path)
    compatibility = observables.validate_reference_metadata(
        metadata,
        config,
        initial_label=INITIAL_LABEL,
        expected_n_states=expected_n_states,
        expected_solver=expected_solver,
    )
    with np.load(density_path) as artifact:
        times = np.asarray(artifact["times"], dtype=float)
        densities = np.asarray(artifact["densities"], dtype=np.complex128)
    expected_end = config.entry_phase_s + config.duration_s
    if times.ndim != 1 or densities.shape != (times.size, expected_n_states, expected_n_states):
        raise ValueError("density artifact has inconsistent times/densities shapes")
    if not np.all(np.isfinite(times)) or not np.isclose(
        times[-1], expected_end, rtol=0.0, atol=1e-14
    ):
        raise ValueError("density artifact endpoint does not match the requested span")
    endpoint = densities[-1]
    if not np.all(np.isfinite(endpoint)):
        raise ValueError("density endpoint contains non-finite values")
    return endpoint, {
        "metadata_path": str(metadata_path),
        "density_path": str(density_path),
        "metadata_sha256": _sha256(metadata_path),
        "density_sha256": _sha256(density_path),
        "source_revision": compatibility.get("reference_source_revision"),
        "driver_sha256": compatibility.get("reference_driver_sha256"),
        "extension_sha256": compatibility.get("reference_rust_extension_sha256"),
        "runner_sha256": compatibility.get("reference_runner_sha256"),
        "sample_count": int(times.size),
        "endpoint_time_s": float(times[-1]),
    }


def _run_action(
    case: driver.RCCase,
    packed_state: np.ndarray,
    *,
    solver: str,
    tolerance: float,
    execution_mode: str,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    layout = case.prepared_dynamic.layout
    rho = layout.unpack(np.asarray(packed_state, dtype=np.float64))
    result = observables.propagate_endpoint(
        case,
        rho,
        (0.0, PERIOD_S),
        output_kind="full_density",
        solver=solver,
        reltol=tolerance,
        execution_mode=execution_mode,
    )
    endpoint = np.asarray(result["endpoint_density"], dtype=np.complex128)
    if endpoint.shape != (case.n_states, case.n_states):
        raise ValueError("period action returned an invalid full-density shape")
    return layout.pack(endpoint), float(result["segment_solver_wall_seconds_sum"]), result


def _checkpoint_report(
    result: ArnoldiPowerResult,
    reference: np.ndarray,
    case: driver.RCCase,
    action_times: list[float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    layout = case.prepared_dynamic.layout
    rows: dict[str, Any] = {}
    densities: dict[str, np.ndarray] = {}
    reference_populations = np.diag(reference).real
    for requested_dimension, checkpoint in result.checkpoints.items():
        density = layout.unpack(checkpoint.value)
        densities[f"density_q{requested_dimension}"] = density
        difference = density - reference
        population_difference = np.diag(difference).real
        populated = reference_populations >= 1e-6
        relative_population = (
            float(
                np.max(
                    np.abs(population_difference[populated])
                    / reference_populations[populated]
                )
            )
            if np.any(populated)
            else 0.0
        )
        physical = driver.full_density_metrics(density)
        density_error = float(np.max(np.abs(difference)))
        population_error = float(np.max(np.abs(population_difference)))
        physical_pass = bool(
            abs(physical["trace_real"] - 1.0) <= 1e-9
            and physical["trace_abs_imag"] <= 1e-9
            and physical["hermiticity_max_abs"] <= 1e-9
            and physical["minimum_eigenvalue"] >= -1e-8
        )
        accuracy_pass = bool(
            np.all(np.isfinite(density))
            and density_error <= 1e-7
            and population_error <= 1e-7
            and relative_population <= 1e-3
        )
        used_actions = min(requested_dimension, result.action_count)
        basis_action_times = action_times[3 : 3 + used_actions]
        rows[str(requested_dimension)] = {
            "requested_dimension": requested_dimension,
            "actual_dimension": checkpoint.actual_dimension,
            "happy_breakdown_at_this_dimension": checkpoint.happy_breakdown,
            "next_arnoldi_coefficient": checkpoint.next_arnoldi_coefficient,
            "residual_indicator_nonrigorous": checkpoint.residual_indicator,
            "density_max_abs_error_vs_pairwise_reference": density_error,
            "population_max_abs_error_vs_pairwise_reference": population_error,
            "population_max_relative_error_above_1e-6": relative_population,
            "physical_checks": physical,
            "physical_checks_pass": physical_pass,
            "pairwise_accuracy_pass": accuracy_pass,
            "combined_pass": physical_pass and accuracy_pass,
            "krylov_period_action_count": used_actions,
            "krylov_period_action_wall_seconds": basis_action_times,
            "cumulative_krylov_period_action_wall_seconds": float(sum(basis_action_times)),
            "cumulative_wall_including_linearity_probe_seconds": float(
                sum(action_times[:3]) + sum(basis_action_times)
            ),
        }
    return rows, densities


def run(args: argparse.Namespace) -> dict[str, Any]:
    resources_before = resource_snapshot()
    if resources_before["known_simulation_pids"] or max(
        resources_before["cpu_percent_samples"]
    ) > 25:
        raise RuntimeError(f"machine busy before periodic Krylov pilot: {resources_before}")
    config_template = driver.CONFIGS["nominal"]
    config = driver.BenchmarkConfig(
        name="nominal",
        optical_rabi_over_gamma=config_template.optical_rabi_over_gamma,
        microwave_rabi_over_gamma=config_template.microwave_rabi_over_gamma,
        detuning_mhz=0.0,
        duration_s=200e-6,
        entry_phase_s=0.0,
    )
    setup_started = time.perf_counter()
    case = driver.build_case(
        config,
        prepare_frozen=False,
        amplitude_envelope="constant",
    )
    observables.driver._prepare_segmented_plan(case)
    rho0 = case.rho0(INITIAL_LABEL)
    setup_seconds = time.perf_counter() - setup_started
    reference_density, reference_info = load_endpoint_artifact(
        args.reference_json,
        config,
        expected_n_states=case.n_states,
        expected_solver="dopri5",
    )
    candidate_density, candidate_info = load_endpoint_artifact(
        args.candidate_json,
        config,
        expected_n_states=case.n_states,
        expected_solver="tsit5",
    )
    period_count = round(config.duration_s / PERIOD_S)
    if not np.isclose(period_count * PERIOD_S, config.duration_s, rtol=0.0, atol=1e-15):
        raise ValueError("duration must contain an integer number of complete periods")

    action_times: list[float] = []
    action_stats: list[dict[str, Any]] = []
    action_count = 0
    progress_path = args.output_dir / f"{args.tag}_progress.json"

    def action(vector: np.ndarray) -> np.ndarray:
        nonlocal action_count
        start = time.perf_counter()
        packed, solver_wall, details = _run_action(
            case,
            vector,
            solver=args.solver,
            tolerance=args.tolerance,
            execution_mode=args.execution_mode,
        )
        wall = time.perf_counter() - start
        action_count += 1
        action_times.append(wall)
        action_stats.append(
            {
                "action_index": action_count,
                "wall_seconds_including_unpack_pack": wall,
                "solver_segment_wall_seconds_sum": solver_wall,
                "segment_count": len(details["segment_stats"]),
                "rhs_calls": int(
                    sum(stat.get("rhs_calls", 0) for stat in details["segment_stats"])
                ),
            }
        )
        progress_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "action_count": action_count,
                    "action_wall_seconds": action_times,
                    "current_action_wall_seconds": wall,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return packed

    # Keep the additivity probe inside the physical trajectory's reachable
    # support. Random full-space vectors excite irrelevant GHz spectator
    # coherences and would misrepresent the cost of this benchmark workload.
    # The three actions are A(u), A(A(u)-u), and A(A(u))=A(u+v).
    u = case.prepared_dynamic.layout.pack(rho0)
    au = action(u)
    v = au - u
    av = action(v)
    a_sum = action(u + v)
    additivity_error = float(
        np.linalg.norm(a_sum - au - av) / max(np.linalg.norm(a_sum), 1e-300)
    )

    arnoldi = arnoldi_repeated_action(
        action,
        case.prepared_dynamic.layout.pack(rho0),
        period_count,
        checkpoints=args.checkpoints,
        max_dimension=args.max_dimension,
        stop_requested=lambda: sum(action_times) >= args.action_budget_s,
    )
    checkpoint_rows, checkpoint_densities = _checkpoint_report(
        arnoldi, reference_density, case, action_times
    )
    pairwise_candidate_error = float(np.max(np.abs(candidate_density - reference_density)))
    action_budget_exceeded = sum(action_times) > args.action_budget_s
    candidate_metrics = driver.full_density_metrics(candidate_density)
    candidate_pop_error = float(
        np.max(
            np.abs(np.diag(candidate_density).real - np.diag(reference_density).real)
        )
    )
    reference_populated = np.diag(reference_density).real >= 1e-6
    candidate_relative_pop_error = float(
        np.max(
            np.abs(
                (np.diag(candidate_density).real - np.diag(reference_density).real)[
                    reference_populated
                ]
            )
            / np.diag(reference_density).real[reference_populated]
        )
    ) if np.any(reference_populated) else 0.0
    payload: dict[str, Any] = {
        "status": (
            "incomplete_budget"
            if arnoldi.stopped_early
            else "action_budget_exceeded"
            if action_budget_exceeded
            else "complete"
        ),
        "timings_accepted": False,
        "method": "matrix-free Arnoldi power of one exact-switch-segmented 5 us full-density action",
        "output_kind": "full_density",
        "output_scope": "all packed Hermitian density coordinates; full complex endpoint density reconstructed",
        "convergence_status": "diagnostic only; q-to-q differences and projected residual are not rigorous error bounds",
        "reuse_scope_limitation": (
            "the Arnoldi basis is built from this initial density and fixed nominal parameter key; "
            "reuse for another initial state requires explicit span/projection-error checks and "
            "is not a reusable full-map substitute for arbitrary rho0"
        ),
        "reference_validation_status": "pairwise only; supplied tight-Dopri artifact is not a multi-run validated reference",
        "action_budget_s": args.action_budget_s,
        "action_budget_stopped_early": arnoldi.stopped_early,
        "action_budget_exceeded": action_budget_exceeded,
        "physics": {
            "config": asdict(config),
            "amplitude_envelope": "constant",
            "entry_phase_s": 0.0,
            "period_s": PERIOD_S,
            "period_count": period_count,
            "switches_segmented_at_exact_edges": True,
            "initial_label": INITIAL_LABEL,
            "n_states": case.n_states,
            "packed_density_dimension": case.prepared_dynamic.layout.packed_len,
        },
        "arguments": vars(args),
        "setup_seconds": setup_seconds,
        "resource_conditions_before": resources_before,
        "reference_artifact": reference_info,
        "candidate_artifact": candidate_info,
        "candidate_pairwise_comparison": {
            "full_density_max_abs_error": pairwise_candidate_error,
            "population_max_abs_error": candidate_pop_error,
            "population_max_relative_error_above_1e-6": candidate_relative_pop_error,
            "physical_checks": candidate_metrics,
            "passes": bool(
                np.all(np.isfinite(candidate_density))
                and pairwise_candidate_error <= 1e-7
                and candidate_pop_error <= 1e-7
                and candidate_relative_pop_error <= 1e-3
                and abs(candidate_metrics["trace_real"] - 1.0) <= 1e-9
                and candidate_metrics["trace_abs_imag"] <= 1e-9
                and candidate_metrics["hermiticity_max_abs"] <= 1e-9
                and candidate_metrics["minimum_eigenvalue"] >= -1e-8
            ),
        },
        "linearity_probe": {
            "action_count": 3,
            "description": "relative additivity discrepancy ||A(u+v)-A(u)-A(v)||/||A(u+v)|| for signed coherent Hermitian inputs",
            "relative_additivity_error": additivity_error,
            "action_seconds": action_times[:3],
            "cost_seconds": float(sum(action_times[:3])),
        },
        "krylov": {
            "requested_max_dimension": args.max_dimension,
            "actual_dimension": arnoldi.action_count,
            "action_count": arnoldi.action_count,
            "happy_breakdown": arnoldi.happy_breakdown,
            "breakdown_dimension": arnoldi.breakdown_dimension,
            "stopped_early_at_budget": arnoldi.stopped_early,
            "total_period_action_count_including_linearity_probe": action_count,
            "total_action_wall_seconds": float(sum(action_times)),
            "action_wall_seconds": action_times,
            "action_stats": action_stats,
            "checkpoints": checkpoint_rows,
        },
        "source_hashes": {
            "krylov_helper_sha256": _sha256(Path(__file__).with_name("rc_periodic_krylov.py")),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "observable_helper_sha256": _sha256(Path(observables.__file__).resolve()),
            "driver_sha256": _sha256(Path(driver.__file__).resolve()),
        },
        "native_provenance": observables._current_provenance(case),
    }
    output_npz = args.output_dir / f"{args.tag}_checkpoint_densities.npz"
    np.savez_compressed(output_npz, **checkpoint_densities)
    payload["checkpoint_density_artifact"] = {
        "path": str(output_npz),
        "sha256": _sha256(output_npz),
        "size_bytes": output_npz.stat().st_size,
    }
    resources_after = resource_snapshot()
    payload["resource_conditions_after"] = resources_after
    payload["timings_accepted"] = bool(
        not resources_after["known_simulation_pids"]
        and max(resources_after["cpu_percent_samples"]) <= 25
        and not arnoldi.stopped_early
        and not action_budget_exceeded
    )
    if resources_after["known_simulation_pids"] or max(
        resources_after["cpu_percent_samples"]
    ) > 25:
        payload["status"] = "contended"
    progress_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=driver.RESULTS_DIR)
    parser.add_argument("--tag", default="rc_periodic_krylov_nominal_200us")
    parser.add_argument("--solver", choices=("dopri5", "tsit5"), default="tsit5")
    parser.add_argument("--tolerance", type=float, default=1e-11)
    parser.add_argument("--action-budget-s", type=float, default=300.0)
    parser.add_argument(
        "--execution-mode", choices=("structured", "expanded_sparse"), default="structured"
    )
    parser.add_argument("--max-dimension", type=int, default=24)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS)
    )
    args = parser.parse_args()
    if not 1 <= args.max_dimension <= 40:
        parser.error("--max-dimension must be between 1 and 40")
    if any(q <= 0 or q > args.max_dimension for q in args.checkpoints):
        parser.error("checkpoint dimensions must be within [1, max-dimension]")
    if args.tolerance <= 0 or not np.isfinite(args.tolerance):
        parser.error("--tolerance must be finite and positive")
    if args.action_budget_s <= 0 or not np.isfinite(args.action_budget_s):
        parser.error("--action-budget-s must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), PeakMemory() as memory:
        payload = run(args)
        payload["peak_process_rss_bytes"] = memory.peak
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
