"""Benchmark RC photon, final-density, and final-population outputs separately.

Every propagation carries the full density matrix between exact switch edges.
Only the requested endpoint observable is retained. The photon path uses the
native mixed full-state plus integral output when available, so it does not
repeat the trajectory just to carry rho.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import bench_rc_propagation as driver
import numpy as np
import tomllib
from run_rc_comparison import PeakMemory, resource_snapshot
from threadpoolctl import threadpool_limits

from centrex_tlf import states
from centrex_tlf.lindblad.batch import solve_lindblad_batch


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_provenance(case: driver.RCCase) -> dict[str, Any]:
    """Read package version from project metadata for bare editable venvs."""
    try:
        return driver.provenance(case)
    except importlib.metadata.PackageNotFoundError:
        pyproject = tomllib.loads((driver.HERE.parent / "pyproject.toml").read_text(encoding="utf-8"))
        version = str(pyproject["project"]["version"])
        original_version = driver.importlib.metadata.version
        driver.importlib.metadata.version = lambda name: (
            version if name == "centrex-tlf" else original_version(name)
        )
        try:
            provenance = driver.provenance(case)
        finally:
            driver.importlib.metadata.version = original_version
        provenance["package_version_source"] = "pyproject.toml (distribution metadata unavailable)"
        return provenance


def validate_reference_metadata(
    metadata: dict[str, Any],
    config: driver.BenchmarkConfig,
    *,
    initial_label: str,
    expected_n_states: int,
    expected_solver: str = "dopri5",
    expected_tolerance_max: float | None = None,
    expected_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject reference artifacts that merely happen to share the duration."""
    if metadata.get("status") != "complete" or metadata.get("timings_accepted") is not True:
        raise ValueError("reference metadata must be complete and timing-accepted")
    arguments = metadata.get("arguments", {})
    provenance = metadata.get("provenance", {})
    recorded_config = provenance.get("config", {})
    checks = {
        "config_name": arguments.get("config") == config.name,
        "duration_us": np.isclose(arguments.get("duration_us", np.nan), config.duration_s * 1e6),
        "entry_ns": np.isclose(arguments.get("entry_ns", np.nan), config.entry_phase_s * 1e9),
        "envelope": provenance.get("amplitude_envelope") == "constant",
        "detuning_argument": np.isclose(
            config.detuning_mhz
            if arguments.get("detuning_mhz") is None
            else arguments.get("detuning_mhz"),
            config.detuning_mhz,
        ),
        "config_record": all(
            np.isclose(recorded_config.get(key, np.nan), value)
            if isinstance(value, float)
            else recorded_config.get(key) == value
            for key, value in asdict(config).items()
        ),
    }
    model = provenance.get("model", {})
    checks["model"] = bool(
        model.get("n_states") == expected_n_states
        and model.get("E_V_cm") == driver.E_FIELD_V_CM
        and model.get("B_G") == driver.B_FIELD_G
        and model.get("retain_opposite_parity_levels") is True
        and model.get("qn_compact") is False
        and model.get("Jmax_X") == 5
        and model.get("Jmax_B") == 3
    )
    rows = metadata.get("rows", [])
    checks["initial_label"] = bool(rows) and rows[0].get("initial_label") == initial_label
    if expected_tolerance_max is None:
        expected_tolerance_max = 1e-11 if expected_solver == "dopri5" else 1e-9
    checks["solver_configuration"] = any(
        row.get("method") == expected_solver
        and row.get("execution_mode") == "structured"
        and row.get("phase_frame") is False
        and np.isfinite(row.get("tolerance", np.nan))
        and row.get("tolerance", np.inf) <= expected_tolerance_max
        and row.get("initial_label") == initial_label
        for row in rows
    )
    if expected_provenance is not None:
        expected_model = json.loads(json.dumps(expected_provenance.get("model"), default=str))
        expected_rates = json.loads(json.dumps(expected_provenance.get("rates"), default=str))
        checks["driver_hash"] = provenance.get("driver_sha256") == expected_provenance.get("driver_sha256")
        checks["model_matches_current"] = provenance.get("model") == expected_model
        checks["rates_match_current"] = provenance.get("rates") == expected_rates
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"reference metadata does not match requested case: {failed}")
    return {
        "checks": checks,
        "reference_source_revision": provenance.get("source_revision"),
        "reference_driver_sha256": provenance.get("driver_sha256"),
        "reference_rust_extension_sha256": provenance.get("rust_extension_sha256"),
        "reference_batch_api_sha256": metadata.get("batch_api_sha256"),
        "reference_runner_sha256": metadata.get("suite_sha256"),
    }


def _group_indices(case: driver.RCCase) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "X_J0": [],
        "X_J1": [],
        "X_J2": [],
        "X_J3": [],
        "B_total": [],
    }
    for index, state in enumerate(case.system.QN):
        largest = state.largest
        if largest.electronic_state == states.ElectronicState.B:
            groups["B_total"].append(index)
        elif largest.electronic_state == states.ElectronicState.X:
            j = round(float(largest.J))
            if j in (0, 1, 2, 3):
                groups[f"X_J{j}"].append(index)
    return groups


def _switch_parameter_batch(state: tuple[int, int]) -> np.ndarray:
    phase = driver.SWITCH_PHASE_OFFSET_RAD
    row = np.asarray(
        [0.0, phase if state[0] else -phase, 0.0, phase if state[1] else -phase],
        dtype=np.complex128,
    )
    return row.reshape(1, -1)


def _unpack_final(case: driver.RCCase, values: np.ndarray) -> np.ndarray:
    packed_len = case.prepared_dynamic.layout.packed_len
    packed = np.asarray(values, dtype=np.float64).reshape(1, -1)[0, :packed_len]
    return case.prepared_dynamic.layout.unpack(packed)


def propagate_endpoint(
    case: driver.RCCase,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    *,
    output_kind: str,
    solver: str,
    reltol: float,
    execution_mode: str,
    selected_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Carry full rho through switch segments and retain only one endpoint output."""
    selected_kinds = {"selected_j1", "selected_single"}
    if output_kind not in {"photons", "full_density", "populations", *selected_kinds}:
        raise ValueError(f"unsupported endpoint output {output_kind!r}")
    if output_kind in selected_kinds and not selected_indices:
        raise ValueError(f"{output_kind} requires at least one state index")
    t0, t1 = map(float, t_span)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("t_span must contain finite endpoints with t1 > t0")
    driver._prepare_segmented_plan(case)
    edges = driver.switch_boundaries(t0, t1)
    rho = np.asarray(rho0, dtype=np.complex128)
    if rho.shape != (case.n_states, case.n_states):
        raise ValueError(f"rho0 must have shape ({case.n_states}, {case.n_states})")

    groups = _group_indices(case)
    initial_index = case.state_labels["X_J2_F1_5o2_F2_mF0"]
    subset_indices = [*groups["X_J1"], initial_index]
    if output_kind in selected_kinds:
        selected = sorted(set(int(index) for index in selected_indices))
        output_indices = [(index, index) for index in selected]
    else:
        selected = []
        output_indices = None

    elapsed_segments: list[float] = []
    segment_stats: list[dict[str, Any]] = []
    photon_total = 0.0
    native_endpoint_bytes = 0
    endpoint_density: np.ndarray | None = None
    endpoint_values: np.ndarray | None = None
    for segment_index, (lo, hi) in enumerate(pairwise(edges)):
        switch_state = driver.switch_values(float((lo + hi) / 2.0))
        final_segment = segment_index == len(edges) - 2
        if output_kind == "photons":
            native_output = "full_and_integral"
        elif output_kind in {"populations", *selected_kinds} and final_segment:
            native_output = "populations" if output_kind == "populations" else "selected"
        else:
            native_output = "full"

        kwargs: dict[str, Any] = {
            "solver": solver,
            "execution_mode": execution_mode,
            "reltol": reltol,
            "abstol": reltol / 100.0,
            "dt": 1e-10,
            "output": native_output,
            "output_when": "final",
            "save_start": False,
            "collect_stats": True,
            "parallel": False,
            "parameter_batch": _switch_parameter_batch(switch_state),
            "parameter_slots": case.switch_override_slots,
        }
        if native_output in {"full_and_integral", "photon_integral"}:
            excited = [
                i
                for i, state in enumerate(case.system.QN)
                if state.largest.electronic_state == states.ElectronicState.B
            ]
            kwargs["integral_weights"] = [(int(i), driver.GAMMA) for i in excited]
            kwargs["integral_method"] = "solver"
        if native_output == "selected":
            kwargs["output_indices"] = output_indices

        started = time.perf_counter()
        result = solve_lindblad_batch(
            case.prepared_segmented,
            rho[np.newaxis, :, :],
            (float(lo), float(hi)),
            **kwargs,
        )
        elapsed = time.perf_counter() - started
        elapsed_segments.append(elapsed)
        values = np.asarray(result.values)
        native_endpoint_bytes += int(values.nbytes)
        stats = dict(result.solver_stats or {})
        stats["wall_seconds_including_python_ffi"] = elapsed
        stats["switch_state"] = list(switch_state)
        stats["time_span_s"] = [float(lo), float(hi)]
        segment_stats.append(stats)

        if native_output == "full_and_integral":
            packed_len = case.prepared_dynamic.layout.packed_len
            row = np.asarray(values, dtype=np.float64).reshape(1, -1)[0]
            rho = case.prepared_dynamic.layout.unpack(row[:packed_len])
            photon_total += float(row[packed_len])
            if final_segment:
                endpoint_density = rho
        elif native_output == "photon_integral":
            photon_total += float(np.asarray(values).reshape(-1)[0].real)
        elif native_output == "full":
            rho = _unpack_final(case, values)
            if final_segment:
                endpoint_density = rho
        elif native_output == "populations":
            endpoint_values = np.asarray(values, dtype=float).reshape(-1)
        elif native_output == "selected":
            endpoint_values = np.asarray(values).reshape(-1)

    return {
        "output_kind": output_kind,
        "endpoint_density": endpoint_density,
        "endpoint_values": endpoint_values,
        "photon_yield": photon_total if output_kind == "photons" else None,
        "groups": groups,
        "subset_j1_and_named_state_indices": subset_indices,
        "selected_indices": selected,
        "segment_wall_seconds": elapsed_segments,
        "segment_stats": segment_stats,
        "segment_solver_wall_seconds_sum": float(sum(elapsed_segments)),
        "native_endpoint_bytes_returned": native_endpoint_bytes,
        "first_segment_wall_seconds": elapsed_segments[0],
        "median_segment_wall_seconds": float(np.median(elapsed_segments)),
        "last_segment_wall_seconds": elapsed_segments[-1],
    }


def _endpoint_report(
    result: dict[str, Any],
    reference_density: np.ndarray | None,
    reference_photons: float | None,
) -> dict[str, Any]:
    kind = result["output_kind"]
    groups = result["groups"]
    report: dict[str, Any] = {
        "output_kind": kind,
        "segment_solver_wall_seconds_sum": result["segment_solver_wall_seconds_sum"],
        "first_segment_wall_seconds": result["first_segment_wall_seconds"],
        "median_segment_wall_seconds": result["median_segment_wall_seconds"],
        "last_segment_wall_seconds": result["last_segment_wall_seconds"],
        "native_endpoint_bytes_returned_across_segments": result[
            "native_endpoint_bytes_returned"
        ],
        "segment_count": len(result["segment_wall_seconds"]),
        "segment_wall_seconds": result["segment_wall_seconds"],
        "group_indices": groups,
        "reference_status": "pairwise provisional tight-Dopri5 200us artifact" if reference_density is not None else None,
    }
    if kind == "full_density":
        rho = result["endpoint_density"]
        report["physical_checks"] = driver.full_density_metrics(rho)
        if reference_density is not None:
            metrics = driver.full_density_metrics(rho)
            reference_pop = np.diag(reference_density).real
            candidate_pop = np.diag(rho).real
            populated = reference_pop >= 1e-6
            density_error = float(np.max(np.abs(rho - reference_density)))
            population_error = float(np.max(np.abs(candidate_pop - reference_pop)))
            population_relative = (
                float(
                    np.max(
                        np.abs(candidate_pop[populated] - reference_pop[populated])
                        / reference_pop[populated]
                    )
                )
                if np.any(populated)
                else 0.0
            )
            physical_pass = bool(
                abs(metrics["trace_real"] - 1.0) <= 1e-9
                and metrics["trace_abs_imag"] <= 1e-9
                and metrics["hermiticity_max_abs"] <= 1e-9
                and metrics["minimum_eigenvalue"] >= -1e-8
            )
            report["reference_comparison"] = {
                "density_max_abs": density_error,
                "population_max_abs": population_error,
                "population_max_relative_above_1e-6": population_relative,
                "physical_checks": metrics,
                "physical_checks_pass": physical_pass,
                "passes": bool(
                    density_error <= 1e-7
                    and population_error <= 1e-7
                    and population_relative <= 1e-3
                    and physical_pass
                ),
            }
        report["output"] = "full final density matrix"
    elif kind == "photons":
        photons = float(result["photon_yield"])
        report["photon_yield"] = photons
        report["physical_checks_internal_carried_rho"] = driver.full_density_metrics(
            result["endpoint_density"]
        )
        if reference_photons is not None:
            error = abs(photons - reference_photons)
            tolerance = 1e-6 + 1e-5 * abs(reference_photons)
            report["reference_comparison"] = {
                "photon_absolute_error": error,
                "photon_relative_error": error / abs(reference_photons) if reference_photons else 0.0,
                "photon_acceptance_tolerance": tolerance,
                "passes": error <= tolerance,
            }
        report["output"] = "solver-native photon integral only; full rho carried internally"
    elif kind == "populations":
        populations = np.asarray(result["endpoint_values"], dtype=float)
        report["populations_all"] = populations.tolist()
        report["group_totals"] = {
            name: float(populations[indices].sum()) for name, indices in groups.items()
        }
        named_index = result["subset_j1_and_named_state_indices"][-1]
        report["single_named_initial_state"] = {
            "label": "X_J2_F1_5o2_F2_mF0",
            "index": named_index,
            "population": float(populations[named_index]),
        }
        if reference_density is not None:
            reference_pop = np.diag(reference_density).real
            error = np.abs(populations - reference_pop)
            populated = reference_pop >= 1e-6
            relative = (
                float(np.max(error[populated] / reference_pop[populated]))
                if np.any(populated)
                else 0.0
            )
            report["reference_comparison"] = {
                "population_max_abs": float(error.max()),
                "population_max_relative_above_1e-6": relative,
                "candidate_population_sum": float(populations.sum()),
                "candidate_minimum_population": float(populations.min()),
                "candidate_values_finite": bool(np.all(np.isfinite(populations))),
                "passes": bool(
                    error.max() <= 1e-7
                    and relative <= 1e-3
                    and np.all(np.isfinite(populations))
                    and populations.min() >= -1e-8
                    and abs(float(populations.sum()) - 1.0) <= 1e-9
                ),
            }
        report["output"] = "native all-state final populations"
    else:
        values = np.asarray(result["endpoint_values"], dtype=np.complex128)
        indices = result["selected_indices"]
        report["selected_state_indices"] = indices
        report["selected_populations"] = values.real.tolist()
        if kind == "selected_j1":
            report["subset_j1_indices"] = groups["X_J1"]
            report["subset_j1_population_total"] = float(values.real.sum())
        else:
            initial_index = indices[0]
            report["single_named_initial_state"] = {
                "label": "X_J2_F1_5o2_F2_mF0",
                "index": initial_index,
                "population": float(values[0].real),
            }
        if reference_density is not None:
            reference = np.asarray([reference_density[i, i].real for i in indices])
            error = np.abs(values - reference)
            populated = reference.real >= 1e-6
            relative = (
                float(np.max(error[populated] / reference.real[populated]))
                if np.any(populated)
                else 0.0
            )
            report["reference_comparison"] = {
                "selected_population_max_abs": float(error.max()),
                "selected_population_max_relative_above_1e-6": relative,
                "candidate_values_finite": bool(np.all(np.isfinite(values))),
                "candidate_max_imaginary": float(np.max(np.abs(values.imag))),
                "candidate_minimum_population": float(np.min(values.real)),
                "passes": bool(
                    error.max() <= 1e-7
                    and relative <= 1e-3
                    and np.all(np.isfinite(values))
                    and np.max(np.abs(values.imag)) <= 1e-9
                    and np.min(values.real) >= -1e-8
                ),
            }
        report["output"] = (
            "native selected diagonal populations for all J=1 states"
            if kind == "selected_j1"
            else "native selected diagonal population for one labeled J=2 state"
        )
    return report


def run(args: argparse.Namespace, resource_conditions_before: dict[str, Any]) -> dict[str, Any]:
    config = driver.CONFIGS[args.config]
    config = driver.BenchmarkConfig(
        name=config.name,
        optical_rabi_over_gamma=config.optical_rabi_over_gamma,
        microwave_rabi_over_gamma=config.microwave_rabi_over_gamma,
        detuning_mhz=args.detuning_mhz,
        duration_s=args.duration_us * 1e-6,
        entry_phase_s=args.entry_ns * 1e-9,
    )
    setup_started = time.perf_counter()
    case = driver.build_case(config, prepare_frozen=False, amplitude_envelope="constant")
    driver._prepare_segmented_plan(case)
    setup_seconds = time.perf_counter() - setup_started
    current_provenance = _current_provenance(case)
    label = "X_J2_F1_5o2_F2_mF0"
    rho0 = case.rho0(label)
    span = (config.entry_phase_s, config.entry_phase_s + config.duration_s)

    reference_density = None
    reference_photons = None
    reference_path = args.reference_npz
    reference_metadata = None
    if reference_path is not None:
        if args.reference_json is None:
            raise ValueError("--reference-json is required with --reference-npz")
        with np.load(reference_path) as ref:
            reference_times = np.asarray(ref["times"], dtype=float)
            reference_all = ref["densities"]
            if reference_times.ndim != 1 or reference_times.size < 2 or not np.all(np.isfinite(reference_times)):
                raise ValueError("reference times must be a finite one-dimensional sequence")
            if reference_all.shape != (reference_times.size, case.n_states, case.n_states):
                raise ValueError("reference densities have an incompatible full-density shape")
            if not np.all(np.diff(reference_times) > 0):
                raise ValueError("reference times must be strictly increasing")
            if not np.isclose(reference_times[0], span[0], rtol=0.0, atol=1e-15) or not np.isclose(
                reference_times[-1], span[1], rtol=0.0, atol=1e-15
            ):
                raise ValueError("reference start/end do not match requested t_span")
            for start in range(0, reference_times.size, 64):
                if not np.all(np.isfinite(reference_all[start : start + 64])):
                    raise ValueError("reference full densities contain non-finite values")
            reference_density = np.asarray(reference_all[-1], dtype=np.complex128)
            reference_photons = float(ref["photons"])
            if not np.isfinite(reference_photons):
                raise ValueError("reference photon integral is non-finite")
        reference_metadata = json.loads(args.reference_json.read_text(encoding="utf-8"))
        reference_compatibility = validate_reference_metadata(
            reference_metadata,
            config,
            initial_label=label,
            expected_n_states=case.n_states,
            expected_provenance=current_provenance,
        )
    elif args.reference_json is not None:
        raise ValueError("--reference-npz is required with --reference-json")

    outputs = list(args.outputs)
    if "photons" in outputs and "full_and_integral" not in Path(
        solve_lindblad_batch.__code__.co_filename
    ).read_text(encoding="utf-8"):
        raise RuntimeError(
            "mixed native output 'full_and_integral' is unavailable; refusing a duplicate-solve fallback"
        )
    case_groups = _group_indices(case)
    selected_j1 = list(case_groups["X_J1"])
    selected_single = [case.state_labels[label]]
    payload: dict[str, Any] = {
        "arguments": vars(args),
        "resource_conditions_before": resource_conditions_before,
        "setup_seconds": setup_seconds,
        "duration_s": config.duration_s,
        "entry_phase_s": config.entry_phase_s,
        "initial_label": label,
        "n_states": case.n_states,
        "output_semantics": "full complex rho is carried across every switch edge; only endpoint observable is retained",
        "reference_path": str(reference_path) if reference_path else None,
        "reference_metadata_path": str(args.reference_json) if args.reference_json else None,
        "reference_status": "pairwise provisional tight-Dopri5; not a four-run validated reference" if reference_density is not None else None,
        "provenance": current_provenance,
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "batch_api_sha256": _file_sha256(driver.HERE.parent / "centrex_tlf/lindblad/batch.py"),
        "mixed_output_contract": "full_and_integral returns packed endpoint rho plus native integral; full rho is only internal carry for photon-only output",
        "timings_accepted": False,
        "complete": False,
        "status": "running",
        "outputs": {},
    }
    if reference_path is not None:
        payload["reference_sha256"] = _file_sha256(reference_path)
    if reference_metadata is not None:
        reference_provenance = reference_metadata.get("provenance", {})
        current_provenance = payload["provenance"]
        payload["reference_compatibility"] = {
            **reference_compatibility,
            "current_driver_sha256": current_provenance.get("driver_sha256"),
            "current_rust_extension_sha256": current_provenance.get("rust_extension_sha256"),
            "native_extension_hash_changed": (
                reference_provenance.get("rust_extension_sha256")
                != current_provenance.get("rust_extension_sha256")
            ),
            "current_batch_api_sha256": payload["batch_api_sha256"],
            "hash_change_interpretation": "only model/config/entry/envelope/initial-label-matched pairwise references are allowed; native/API hash differences are recorded and require the mixed-output regression before comparison",
        }
    driver_path = Path(driver.__file__).resolve()
    payload["driver_sha256"] = _file_sha256(driver_path)

    outputs_in_memory: dict[str, dict[str, Any]] = {}
    for output in outputs:
        trajectory_reports: list[dict[str, Any]] = []
        for repeat_index in range(args.repeats):
            started = time.perf_counter()
            result = propagate_endpoint(
                case,
                rho0,
                span,
                output_kind=output,
                solver=args.solver,
                reltol=args.tolerance,
                execution_mode=args.execution_mode,
                selected_indices=(
                    selected_j1 if output == "selected_j1"
                    else selected_single if output == "selected_single"
                    else None
                ),
            )
            propagation_wall_seconds = time.perf_counter() - started
            outputs_in_memory[output] = result
            report = _endpoint_report(result, reference_density, reference_photons)
            report["propagation_wall_seconds_including_python_carry"] = propagation_wall_seconds
            report["repeat_index"] = repeat_index
            trajectory_reports.append(report)
        artifact_write_started = time.perf_counter()
        metadata_bytes = 0
        if output == "photons":
            output_path = args.output_dir / f"{args.tag}_photons.npz"
            observable_bytes = np.asarray(result["photon_yield"], dtype=np.float64).nbytes
            np.savez_compressed(output_path, photon_yield=result["photon_yield"])
        elif output == "full_density":
            output_path = args.output_dir / f"{args.tag}_full_final_density.npz"
            observable_bytes = result["endpoint_density"].nbytes
            np.savez_compressed(output_path, final_density=result["endpoint_density"])
        elif output == "populations":
            output_path = args.output_dir / f"{args.tag}_populations_all.npz"
            observable_bytes = np.asarray(result["endpoint_values"]).nbytes
            np.savez_compressed(output_path, populations=result["endpoint_values"])
        elif output == "selected_j1":
            output_path = args.output_dir / f"{args.tag}_populations_subset_j1.npz"
            observable_bytes = np.asarray(result["endpoint_values"]).nbytes
            metadata_bytes = np.asarray(result["selected_indices"], dtype=np.int64).nbytes
            np.savez_compressed(
                output_path,
                indices=np.asarray(result["selected_indices"], dtype=np.int64),
                populations=np.asarray(result["endpoint_values"]),
            )
        else:
            output_path = args.output_dir / f"{args.tag}_population_named_initial.npz"
            observable_bytes = np.asarray(result["endpoint_values"]).nbytes
            metadata_bytes = np.asarray(result["selected_indices"], dtype=np.int64).nbytes
            np.savez_compressed(
                output_path,
                indices=np.asarray(result["selected_indices"], dtype=np.int64),
                populations=np.asarray(result["endpoint_values"]),
            )
        report = trajectory_reports[-1]
        report["trajectory_wall_seconds"] = [
            row["propagation_wall_seconds_including_python_carry"]
            for row in trajectory_reports
        ]
        report["trajectory_wall_median_seconds"] = float(
            np.median(report["trajectory_wall_seconds"])
        )
        report["trajectory_wall_min_seconds"] = float(np.min(report["trajectory_wall_seconds"]))
        report["trajectory_wall_max_seconds"] = float(np.max(report["trajectory_wall_seconds"]))
        report["repeat_count"] = args.repeats
        report["per_repeat_accuracy"] = [
            row.get("reference_comparison") for row in trajectory_reports
        ]
        report["artifact_path"] = str(output_path)
        report["artifact_bytes_on_disk"] = output_path.stat().st_size
        report["artifact_observable_value_bytes"] = int(observable_bytes)
        report["artifact_metadata_bytes"] = int(metadata_bytes)
        report["artifact_logical_bytes"] = int(observable_bytes + metadata_bytes)
        report["artifact_write_seconds"] = time.perf_counter() - artifact_write_started
        payload["outputs"][output] = report
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{args.tag}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    if set(outputs_in_memory) == {"photons", "full_density", "populations", "selected_j1", "selected_single"}:
        reference_matrix = outputs_in_memory["full_density"]["endpoint_density"]
        populations = np.diag(reference_matrix).real
        selected_j1_values = np.asarray(
            outputs_in_memory["selected_j1"]["endpoint_values"], dtype=np.complex128
        ).real
        selected_single_value = float(
            np.asarray(outputs_in_memory["selected_single"]["endpoint_values"]).reshape(-1)[0].real
        )
        j1_indices = outputs_in_memory["selected_j1"]["selected_indices"]
        single_index = outputs_in_memory["selected_single"]["selected_indices"][0]
        photon_density = outputs_in_memory["photons"]["endpoint_density"]
        cross_checks = {
            "photon_internal_density_max_abs_vs_full_density_output": float(
                np.max(np.abs(photon_density - reference_matrix))
            ),
            "all_populations_max_abs_vs_full_density_output": float(
                np.max(np.abs(np.asarray(outputs_in_memory["populations"]["endpoint_values"]) - populations))
            ),
            "j1_selected_max_abs_vs_full_density_output": float(
                np.max(np.abs(selected_j1_values - populations[j1_indices]))
            ),
            "named_state_selected_abs_error_vs_full_density_output": abs(
                selected_single_value - float(populations[single_index])
            ),
        }
        payload["cross_output_checks"] = cross_checks
        payload["cross_output_checks_pass"] = bool(max(cross_checks.values()) <= 1e-7)
        if not payload["cross_output_checks_pass"]:
            raise AssertionError(f"output modes disagree beyond 1e-7: {cross_checks}")
    payload["complete"] = True
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=tuple(driver.CONFIGS), default="weak")
    parser.add_argument("--duration-us", type=float, default=1.0)
    parser.add_argument("--entry-ns", type=float, default=0.0)
    parser.add_argument("--detuning-mhz", type=float, default=0.0)
    parser.add_argument("--solver", choices=("dopri5", "tsit5"), default="tsit5")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--execution-mode", choices=("structured", "expanded_sparse"), default="structured")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=("photons", "full_density", "populations", "selected_j1", "selected_single"),
        default=["photons", "full_density", "populations", "selected_j1", "selected_single"],
    )
    parser.add_argument("--reference-npz", type=Path)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=driver.RESULTS_DIR)
    parser.add_argument("--tag", default="rc_observables")
    args = parser.parse_args()
    if args.duration_us <= 0 or args.tolerance <= 0 or args.repeats < 1:
        parser.error("duration, tolerance, and repeats must be positive")
    if len(set(args.outputs)) != len(args.outputs):
        parser.error("--outputs values must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources_before = resource_snapshot()
    if resources_before["known_simulation_pids"] or max(resources_before["cpu_percent_samples"]) > 25:
        raise RuntimeError(f"machine busy before observable benchmark: {resources_before}")
    try:
        with threadpool_limits(limits=1), PeakMemory() as memory:
            payload = run(args, resources_before)
            payload["peak_process_rss_bytes"] = memory.peak
        resources_after = resource_snapshot()
        payload["resource_conditions_after"] = resources_after
        payload["timings_accepted"] = bool(
            not resources_after["known_simulation_pids"]
            and max(resources_after["cpu_percent_samples"]) <= 25
        )
        payload["status"] = "complete" if payload["timings_accepted"] else "contended"
    except Exception as exc:
        payload = {
            "arguments": vars(args),
            "resource_conditions_before": resources_before,
            "timings_accepted": False,
            "complete": False,
            "status": "failed",
            "error": repr(exc),
        }
        raise
    finally:
        (args.output_dir / f"{args.tag}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
