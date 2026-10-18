"""Measure and validate a small exact population-response map for the RC model.

The pilot builds full packed-density and photon responses for a handful of
one-hot diagonal inputs, then checks a weighted response against a direct
propagation of the corresponding mixed population state. It does not build
the full 70-column map unless explicitly requested and reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import bench_rc_propagation as driver
import numpy as np
import run_rc_observables as observables
from rc_population_response_map import (
    build_diagonal_response,
    save_response_map,
    validate_mixture_response,
)
from run_rc_comparison import PeakMemory, resource_snapshot
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.batch import solve_lindblad_batch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _representative_basis(case: driver.RCCase) -> tuple[list[int], list[str]]:
    """Choose eight unique inputs spanning X J=0..3 and B."""
    groups = observables._group_indices(case)
    named = [
        case.state_labels["X_J1_F1_3o2_F1_mF0"],
        case.state_labels["X_J2_F1_5o2_F2_mF0"],
        case.state_labels["X_J3_F1_7o2_F3_mF0"],
    ]
    candidates = [
        groups["X_J0"][0],
        named[0],
        groups["X_J1"][-1],
        named[1],
        groups["X_J2"][-1],
        named[2],
        groups["X_J3"][-1],
        groups["B_total"][0],
    ]
    indices = list(dict.fromkeys(int(i) for i in candidates))
    labels = [
        next(
            (key for key, value in case.state_labels.items() if value == index),
            f"{case.system.QN[index].largest}",
        )
        for index in indices
    ]
    if len(indices) < 3:
        raise RuntimeError("could not select at least three distinct pilot basis states")
    return indices, labels


def _load_reference(
    metadata_path: Path,
    config: driver.BenchmarkConfig,
    *,
    case: driver.RCCase,
    expected_provenance: dict[str, Any],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    compatibility = observables.validate_reference_metadata(
        metadata,
        config,
        initial_label="X_J2_F1_5o2_F2_mF0",
        expected_n_states=case.n_states,
        expected_provenance=expected_provenance,
    )
    density_path = metadata_path.parent / f"{metadata_path.stem}_densities.npz"
    with np.load(density_path) as archive:
        times = np.asarray(archive["times"], dtype=float)
        densities = archive["densities"]
        if (
            times.ndim != 1
            or times.size < 2
            or densities.shape != (times.size, case.n_states, case.n_states)
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError("reference times and full-density data have invalid shape/order")
        expected_span = (config.entry_phase_s, config.entry_phase_s + config.duration_s)
        if not np.isclose(times[0], expected_span[0], rtol=0.0, atol=1e-15) or not np.isclose(
            times[-1], expected_span[1], rtol=0.0, atol=1e-15
        ):
            raise ValueError("reference start/end do not match the response-map span")
        for start in range(0, times.size, 64):
            if not np.all(np.isfinite(densities[start : start + 64])):
                raise ValueError("reference densities contain non-finite values")
        reference_density = np.asarray(densities[-1], dtype=np.complex128)
    reference_photons = float(metadata["rows"][0]["physical_checks"]["photon_yield"])
    return reference_density, reference_photons, {
        **compatibility,
        "metadata_path": str(metadata_path),
        "density_path": str(density_path),
        "metadata_sha256": _sha256(metadata_path),
        "density_sha256": _sha256(density_path),
        "endpoint_time_s": float(times[-1]),
        "sample_count": int(times.size),
    }


def _propagate_mixture_batch(
    case: driver.RCCase,
    rho0_batch: np.ndarray,
    t_span: tuple[float, float],
    *,
    solver: str,
    reltol: float,
    execution_mode: str,
    threads: int,
    parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Directly propagate several full population mixtures in one batch."""
    layout = case.prepared_segmented.layout
    packed = np.stack([layout.pack(rho) for rho in rho0_batch])
    photon_totals = np.zeros(len(packed), dtype=np.float64)
    boundaries = driver.switch_boundaries(*t_span)
    segment_records = []
    total_started = time.perf_counter()
    for lo, hi in pairwise(boundaries):
        switch_state = driver.switch_values(float((lo + hi) / 2.0))
        parameter_batch = np.repeat(
            observables._switch_parameter_batch(switch_state), len(packed), axis=0
        )
        started = time.perf_counter()
        result = solve_lindblad_batch(
            case.prepared_segmented,
            packed,
            (float(lo), float(hi)),
            solver=solver,
            execution_mode=execution_mode,
            reltol=reltol,
            abstol=reltol / 100.0,
            dt=1e-10,
            output="full_and_integral",
            output_when="final",
            save_start=False,
            collect_stats=True,
            parallel=parallel,
            threads=threads if parallel else None,
            parameter_batch=parameter_batch,
            parameter_slots=case.switch_override_slots,
            integral_weights=[(int(i), driver.GAMMA) for i, state in enumerate(case.system.QN)
                              if state.largest.electronic_state == driver.states.ElectronicState.B],
            integral_method="solver",
        )
        elapsed = time.perf_counter() - started
        values = np.asarray(result.values, dtype=np.float64)
        expected_shape = (len(packed), layout.packed_len + 1)
        if values.shape != expected_shape:
            raise RuntimeError(f"direct mixture batch returned {values.shape}; expected {expected_shape}")
        packed = np.ascontiguousarray(values[:, : layout.packed_len])
        photon_totals += values[:, -1]
        segment_records.append(
            {
                "span_s": [float(lo), float(hi)],
                "switch_state": list(switch_state),
                "wall_seconds": elapsed,
                "solver_stats": dict(result.solver_stats or {}),
            }
        )
    densities = np.stack([layout.unpack(row) for row in packed])
    return densities, photon_totals, {
        "total_wall_seconds": time.perf_counter() - total_started,
        "segment_count": len(segment_records),
        "segments": segment_records,
    }


def run(args: argparse.Namespace, resources_before: dict[str, Any]) -> dict[str, Any]:
    template = driver.CONFIGS["nominal"]
    config = driver.BenchmarkConfig(
        name="nominal",
        optical_rabi_over_gamma=template.optical_rabi_over_gamma,
        microwave_rabi_over_gamma=template.microwave_rabi_over_gamma,
        detuning_mhz=args.detuning_mhz,
        duration_s=args.duration_us * 1e-6,
        entry_phase_s=args.entry_ns * 1e-9,
    )
    setup_started = time.perf_counter()
    case = driver.build_case(config, prepare_frozen=False, amplitude_envelope="constant")
    driver._prepare_segmented_plan(case)
    current_provenance = observables._current_provenance(case)
    reference_density, reference_photons, reference_info = _load_reference(
        args.reference_json,
        config,
        case=case,
        expected_provenance=current_provenance,
    )
    setup_seconds = time.perf_counter() - setup_started
    basis_indices, basis_labels = _representative_basis(case)
    if len(basis_indices) != args.basis_count:
        raise ValueError(f"representative picker produced {len(basis_indices)} inputs, requested {args.basis_count}")
    anchor_index = int(case.state_labels["X_J2_F1_5o2_F2_mF0"])
    if anchor_index not in basis_indices:
        raise ValueError("pilot basis must include the one-hot state used by the tight reference")
    span = (config.entry_phase_s, config.entry_phase_s + config.duration_s)
    metadata = {
        "config": asdict(config),
        "model": current_provenance["model"],
        "rates": current_provenance["rates"],
        "source_revision": current_provenance.get("source_revision"),
        "driver_sha256": current_provenance.get("driver_sha256"),
        "rust_extension_sha256": current_provenance.get("rust_extension_sha256"),
        "batch_api_sha256": _sha256(driver.HERE.parent / "centrex_tlf/lindblad/batch.py"),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "response_map_helper_sha256": _sha256(Path(__file__).with_name("rc_population_response_map.py")),
        "solver": args.solver,
        "tolerance": args.map_tolerance,
        "execution_mode": args.execution_mode,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "basis_indices": basis_indices,
        "basis_labels": basis_labels,
        "span_s": list(span),
        "integral": "native solver integral of Gamma times total B-state population",
    }
    response_map, build_stats = build_diagonal_response(
        case,
        basis_indices,
        span,
        metadata=metadata,
        solver=args.solver,
        reltol=args.map_tolerance,
        execution_mode=args.execution_mode,
        batch_size=args.batch_size,
        parallel=True,
        threads=args.threads,
    )
    rng = np.random.default_rng(205)
    mixture_weights = rng.dirichlet(np.ones(len(basis_indices)), size=8)
    mixture_weights[0, :] = 1.0 / len(basis_indices)
    rho0_batch = np.zeros((len(mixture_weights), case.n_states, case.n_states), dtype=np.complex128)
    for mixture_index, weights in enumerate(mixture_weights):
        rho0_batch[mixture_index, basis_indices, basis_indices] = weights
    direct_densities, direct_photons, direct_batch_stats = _propagate_mixture_batch(
        case,
        rho0_batch,
        span,
        solver=args.solver,
        reltol=args.map_tolerance,
        execution_mode=args.execution_mode,
        threads=args.threads,
    )
    serial_densities, serial_photons, serial_batch_stats = _propagate_mixture_batch(
        case,
        rho0_batch,
        span,
        solver=args.solver,
        reltol=args.map_tolerance,
        execution_mode=args.execution_mode,
        threads=1,
        parallel=False,
    )
    anchor_rho0 = case.rho0("X_J2_F1_5o2_F2_mF0")
    tight_density, tight_photons, tight_validation_stats = _propagate_mixture_batch(
        case,
        anchor_rho0[np.newaxis, :, :],
        span,
        solver=args.validation_solver,
        reltol=args.validation_tolerance,
        execution_mode=args.execution_mode,
        threads=1,
        parallel=False,
    )
    mapped_densities = response_map.density(mixture_weights)
    mapped_photons = np.asarray(response_map.photons(mixture_weights), dtype=float)
    validations = [
        validate_mixture_response(
            response_map,
            weights,
            {"endpoint_density": rho, "photon_yield": photon},
        )
        for weights, rho, photon in zip(
            mixture_weights, direct_densities, direct_photons, strict=True
        )
    ]
    validation_rows = [asdict(item) for item in validations]
    comparison = {
        "mixture_count": len(mixture_weights),
        "all_pass": all(
            item.density_matches
            and item.photons_match
            and item.populations_match
            and item.physical_checks_pass
            for item in validations
        ),
        "per_mixture": validation_rows,
        "maximum_density_error": max(item.density_max_abs_error for item in validations),
        "maximum_population_error": max(item.population_max_abs_error for item in validations),
        "maximum_photon_error": max(item.photon_abs_error for item in validations),
        "map_outputs_finite": bool(
            np.all(np.isfinite(mapped_densities)) and np.all(np.isfinite(mapped_photons))
        ),
    }
    serial_parallel_density_error = float(np.max(np.abs(serial_densities - direct_densities)))
    serial_parallel_photon_error = float(np.max(np.abs(serial_photons - direct_photons)))
    serial_parallel_match = bool(
        serial_parallel_density_error <= 1e-7
        and serial_parallel_photon_error <= 1e-6 + 1e-5 * float(np.max(np.abs(direct_photons)))
    )
    anchor_column = basis_indices.index(anchor_index)
    anchor_weights = np.zeros(len(basis_indices), dtype=float)
    anchor_weights[anchor_column] = 1.0
    anchor_mapped_density = response_map.density(anchor_weights)
    anchor_mapped_photons = float(response_map.photons(anchor_weights))
    anchor_density_error = float(np.max(np.abs(anchor_mapped_density - tight_density[0])))
    anchor_photon_error = float(abs(anchor_mapped_photons - tight_photons[0]))
    reference_anchor_density_error = float(np.max(np.abs(anchor_mapped_density - reference_density)))
    reference_anchor_photon_error = float(abs(anchor_mapped_photons - reference_photons))
    basis_physical_checks = []
    for basis_index, basis_label in zip(basis_indices, basis_labels, strict=True):
        unit_weights = np.zeros(len(basis_indices), dtype=float)
        unit_weights[basis_indices.index(basis_index)] = 1.0
        metrics = driver.full_density_metrics(response_map.density(unit_weights))
        basis_physical_checks.append(
            {
                "basis_index": basis_index,
                "basis_label": basis_label,
                **metrics,
                "passes": bool(
                    abs(metrics["trace_real"] - 1.0) <= 1e-9
                    and metrics["trace_abs_imag"] <= 1e-9
                    and metrics["hermiticity_max_abs"] <= 1e-9
                    and metrics["minimum_eigenvalue"] >= -1e-8
                ),
            }
        )
    estimated_full70_seconds = build_stats.total_wall_seconds * math.ceil(70 / args.batch_size)
    full70_projection_with_safety = 2.0 * estimated_full70_seconds
    tight_validation = {
        "solver": args.validation_solver,
        "tolerance": args.validation_tolerance,
        "wall_seconds": tight_validation_stats["total_wall_seconds"],
        "map_density_max_abs_error": anchor_density_error,
        "map_photon_abs_error": anchor_photon_error,
        "reference_density_max_abs_error": reference_anchor_density_error,
        "reference_photon_abs_error": reference_anchor_photon_error,
        "passes_map_accuracy": bool(
            anchor_density_error <= 1e-7
            and anchor_photon_error <= 1e-6 + 1e-5 * abs(tight_photons[0])
        ),
        "passes_existing_reference": bool(
            reference_anchor_density_error <= 1e-7
            and reference_anchor_photon_error <= 1e-6 + 1e-5 * abs(reference_photons)
        ),
        "physical_checks": driver.full_density_metrics(tight_density[0]),
    }
    data_path = args.output_dir / f"{args.tag}_response_map.npz"
    metadata_path = args.output_dir / f"{args.tag}_response_map.json"
    response_record = save_response_map(response_map, data_path, metadata_path)
    resources_after = resource_snapshot()
    timing_accepted = bool(
        not resources_after["known_simulation_pids"]
        and max(resources_after["cpu_percent_samples"]) <= 25
    )
    all_basis_physical = all(row["passes"] for row in basis_physical_checks)
    complete = bool(
        comparison["all_pass"]
        and tight_validation["passes_map_accuracy"]
        and tight_validation["passes_existing_reference"]
        and all_basis_physical
        and serial_parallel_match
        and comparison["map_outputs_finite"]
    )
    return {
        "status": "complete" if complete and timing_accepted else ("contended" if complete else "failed"),
        "complete": complete,
        "timings_accepted": timing_accepted,
        "resource_conditions_before": resources_before,
        "resource_conditions_after": resources_after,
        "config": asdict(config),
        "n_states": case.n_states,
        "basis_indices": basis_indices,
        "basis_labels": basis_labels,
        "mixture_weights": mixture_weights.tolist(),
        "reference_artifact": reference_info,
        "provenance": current_provenance,
        "setup_seconds": setup_seconds,
        "response_map_build_seconds": build_stats.total_wall_seconds,
        "response_map_solve_calls": build_stats.n_solve_calls,
        "response_map_batch_records": build_stats.batch_records,
        "response_map_per_column_allocated_costs": build_stats.per_column_allocated_wall_seconds,
        "direct_mixture_batch_seconds": direct_batch_stats["total_wall_seconds"],
        "direct_mixture_batch_stats": direct_batch_stats,
        "direct_mixture_batch_serial_seconds": serial_batch_stats["total_wall_seconds"],
        "direct_mixture_batch_serial_stats": serial_batch_stats,
        "mixtures_in_direct_batch": len(mixture_weights),
        "serial_parallel_batch_consistency": {
            "density_max_abs_error": serial_parallel_density_error,
            "photon_max_abs_error": serial_parallel_photon_error,
            "passes": serial_parallel_match,
        },
        "map_build_vs_parallel_direct_batch_cost_ratio": build_stats.total_wall_seconds / direct_batch_stats["total_wall_seconds"],
        "map_build_vs_serial_direct_batch_cost_ratio": build_stats.total_wall_seconds / serial_batch_stats["total_wall_seconds"],
        "mixture_validation": comparison,
        "tight_mixture_validation": tight_validation,
        "basis_column_physical_checks": basis_physical_checks,
        "all_basis_columns_physical": all_basis_physical,
        "full70_projection": {
            "raw_seconds_linear_batch_count_projection": estimated_full70_seconds,
            "safety_factor": 2.0,
            "seconds_with_safety_factor": full70_projection_with_safety,
            "pilot_batch_size": args.batch_size,
            "full70_batch_count_projection": math.ceil(70 / args.batch_size),
            "budget_seconds": 1800.0,
            "within_budget_gate_only": full70_projection_with_safety <= 1800.0,
            "method": "pilot full-batch wall multiplied by ceil(70/batch_size), then safety factor 2; projection only, not measured full-map cost",
        },
        "response_map": {
            **response_record,
            "path": str(data_path),
            "metadata_path": str(metadata_path),
            "data_size_bytes": data_path.stat().st_size,
        },
        "peak_process_rss_bytes": args.peak_process_rss_bytes,
        "arguments": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, default=driver.RESULTS_DIR / "idle_nominal_200us_tight_dopri.json")
    parser.add_argument("--duration-us", type=float, default=200.0)
    parser.add_argument("--entry-ns", type=float, default=0.0)
    parser.add_argument("--detuning-mhz", type=float, default=0.0)
    parser.add_argument("--basis-count", type=int, choices=(8,), default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--solver", choices=("dopri5", "tsit5"), default="tsit5")
    parser.add_argument("--map-tolerance", type=float, default=1e-9)
    parser.add_argument("--validation-solver", choices=("dopri5", "tsit5"), default="dopri5")
    parser.add_argument("--validation-tolerance", type=float, default=1e-11)
    parser.add_argument("--execution-mode", choices=("structured", "expanded_sparse"), default="structured")
    parser.add_argument("--output-dir", type=Path, default=driver.RESULTS_DIR)
    parser.add_argument("--tag", default="idle_nominal_200us_response_pilot")
    args = parser.parse_args()
    if not np.isfinite(args.duration_us) or args.duration_us <= 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("duration, batch size, and threads must be positive")
    if not np.isfinite(args.map_tolerance) or args.map_tolerance <= 0:
        parser.error("map tolerance must be finite and positive")
    if not np.isfinite(args.validation_tolerance) or args.validation_tolerance <= 0:
        parser.error("validation tolerance must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources_before = resource_snapshot()
    if resources_before["known_simulation_pids"] or max(resources_before["cpu_percent_samples"]) > 25:
        raise RuntimeError(f"machine busy before response-map pilot: {resources_before}")
    with threadpool_limits(limits=1), PeakMemory() as memory:
        args.peak_process_rss_bytes = None
        payload = run(args, resources_before)
        payload["peak_process_rss_bytes"] = memory.peak
    path = args.output_dir / f"{args.tag}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
