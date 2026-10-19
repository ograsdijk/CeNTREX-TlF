"""Bounded benchmark harness for shared-H block and scalar-adjoint pilots.

This file is an experiment driver only. It does not run on import; the
independent benchmark checker owns execution. Each method/case is capped at
five minutes by the caller's process supervisor, and every stage writes a
checkpoint without replacing existing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from centrex_tlf.lindblad.batch import solve_lindblad_batch

import bench_rc_propagation as driver
import rc_block_adjoint_prototypes as proto
import run_rc_comparison as comparison
import run_rc_observables as observables

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rc_propagation_results" / "block_adjoint_pilot"
CPU_MEDIAN_LIMIT = 50.0
METHOD_BUDGET_SECONDS = 300.0
LABELS = (
    "X_J1_F1_3o2_F1_mF0",
    "X_J2_F1_5o2_F2_mF0",
    "X_J3_F1_7o2_F3_mF0",
)
RELTOL = 1e-9
ABSTOL = 1e-11
DT = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    temporary.replace(path)


def resource_snapshot() -> dict[str, Any]:
    snapshot = comparison.resource_snapshot()
    self_process = psutil.Process()
    excluded = {self_process.pid, *(parent.pid for parent in self_process.parents())}
    pids = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (process.info.get("name") or "").lower()
            args = process.info.get("cmdline") or []
            owns_runner_script = any(
                Path(argument).name.lower() == Path(__file__).name.lower()
                for argument in args
                if isinstance(argument, str)
            )
            if process.pid not in excluded and "python" in name and owns_runner_script:
                pids.append(int(process.pid))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    snapshot["known_block_adjoint_pids"] = pids
    return snapshot


def guard(snapshot: dict[str, Any]) -> dict[str, Any]:
    cpu = np.asarray(snapshot.get("cpu_percent_samples", []), dtype=float)
    okay = bool(
        cpu.size > 0
        and float(np.median(cpu)) <= CPU_MEDIAN_LIMIT
        and not snapshot.get("known_simulation_pids")
        and not snapshot.get("known_block_adjoint_pids")
    )
    return {
        "okay": okay,
        "cpu_median_limit_percent": CPU_MEDIAN_LIMIT,
        "cpu_median_percent": float(np.median(cpu)) if cpu.size else None,
        "reason": None if okay else "resource guard failed",
    }


def provenance(case: driver.RCCase) -> dict[str, Any]:
    base = observables._current_provenance(case)
    import scipy
    import numba
    import centrex_tlf.centrex_tlf_rust as rust_extension

    return {
        **base,
        "runner_sha256": sha256(Path(__file__)),
        "prototype_sha256": sha256(Path(proto.__file__)),
        "batch_api_sha256": sha256(Path(solve_lindblad_batch.__code__.co_filename)),
        "block_adjoint_test_sha256": sha256(HERE / "test_rc_block_adjoint_prototypes.py"),
        "rust_extension_sha256": sha256(Path(rust_extension.__file__)),
        "scipy_version": scipy.__version__,
        "numba_version": numba.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "blas_threads_requested": 1,
        "resource_policy": {
            "median_cpu_percent_max": CPU_MEDIAN_LIMIT,
            "per_method_case_budget_seconds": METHOD_BUDGET_SECONDS,
        },
        "solver": {"method": "DOP853", "reltol": RELTOL, "abstol": ABSTOL, "dt_s": DT},
        "output_contract": {
            "block": "full final density for each of three one-hots plus one weighted mixture",
            "adjoint": "scalar photons and final X J=1 population, separately",
            "switches": "exact boundaries; global time retained for EOM phase",
        },
        "baseline_core_budget": {"serial_rust": 1, "parallel_rust_threads": 3},
        "duration_us": case.config.duration_s * 1e6,
    }


def accuracy_row(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    a = np.asarray(actual, dtype=np.complex128)
    b = np.asarray(expected, dtype=np.complex128)
    pa = np.diagonal(a, axis1=-2, axis2=-1).real
    pb = np.diagonal(b, axis1=-2, axis2=-1).real
    populated = pb >= 1e-6
    rel = float(np.max(np.abs((pa[populated] - pb[populated]) / pb[populated]))) if np.any(populated) else 0.0
    return {
        "density_max_abs": float(np.max(np.abs(a - b))),
        "population_max_abs": float(np.max(np.abs(pa - pb))),
        "population_max_relative_above_1e-6": rel,
        "trace_max_abs_error": float(np.max(np.abs(np.trace(a, axis1=-2, axis2=-1) - 1.0))),
        "hermiticity_max_abs": float(np.max(np.abs(a - a.conjugate().swapaxes(-1, -2)))),
    }


def _span(case: driver.RCCase) -> tuple[float, float]:
    start = float(case.config.entry_phase_s)
    return start, start + float(case.config.duration_s)


def rust_full_density_endpoint(
    case: driver.RCCase,
    rho0_batch: np.ndarray,
    span: tuple[float, float],
    *,
    solver: str,
    reltol: float,
    abstol: float,
    dt: float,
    parallel: bool,
    threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rust endpoint-only full-density propagation, with no photon solve."""
    from itertools import pairwise

    driver._prepare_segmented_plan(case)
    rho = np.asarray(rho0_batch, dtype=np.complex128).copy()
    if rho.ndim == 2:
        rho = rho[None, :, :]
    edges = driver.switch_boundaries(*span)
    segment_rows = []
    for lo, hi in pairwise(edges):
        switch = driver.switch_values(0.5 * (float(lo) + float(hi)))
        phase = driver.SWITCH_PHASE_OFFSET_RAD
        row = np.asarray(
            [0.0, phase if switch[0] else -phase, 0.0, phase if switch[1] else -phase],
            dtype=np.complex128,
        )
        started = time.perf_counter()
        result = solve_lindblad_batch(
            case.prepared_segmented,
            rho,
            (float(lo), float(hi)),
            parameter_batch=np.tile(row, (rho.shape[0], 1)),
            parameter_slots=case.switch_override_slots,
            solver=solver,
            execution_mode="structured",
            reltol=reltol,
            abstol=abstol,
            dt=dt,
            output="full",
            output_when="final",
            save_start=False,
            collect_stats=True,
            parallel=parallel,
            threads=threads,
        )
        rho = np.stack(
            [case.prepared_dynamic.layout.unpack(np.asarray(row, dtype=np.float64)) for row in np.asarray(result.values)],
            axis=0,
        )
        segment_rows.append({
            "lo_s": float(lo), "hi_s": float(hi),
            "wall_seconds_including_python_ffi": time.perf_counter() - started,
            "solver_stats": result.solver_stats,
        })
    return rho, {"segments": segment_rows, "segment_count": len(segment_rows), "output": "full_density_endpoint_only"}


def _initials(case: driver.RCCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    onehots, weights, mixture = proto.labeled_initial_block(case)
    all_inputs = np.concatenate((onehots, mixture[None, :, :]), axis=0)
    return onehots, weights, mixture, all_inputs


STAGES = (
    "reference",
    "rhs_profile",
    "rust_serial",
    "rust_parallel_3",
    "shared_block",
    "adjoint_photons",
    "adjoint_X_J1",
)


def load_verified_reference(case: driver.RCCase, output_dir: Path, current_provenance: dict[str, Any]):
    reference_json = output_dir / "reference.json"
    reference_path = output_dir / "strict_reference.npz"
    if not reference_json.exists() or not reference_path.exists():
        raise FileNotFoundError("run the reference stage first")
    ref_payload = json.loads(reference_json.read_text(encoding="utf-8"))
    if ref_payload.get("status") != "complete" or ref_payload.get("timings_accepted") is not True:
        raise ValueError("reference stage is not complete and timing-accepted")
    if ref_payload.get("accuracy_accepted") is not True:
        raise ValueError("reference stage did not pass its accuracy gates")
    if sha256(reference_path) != ref_payload.get("reference_npz_sha256"):
        raise ValueError("reference NPZ checksum does not match its JSON")
    ref_provenance = ref_payload.get("provenance", {})
    exact_keys = (
        "source_revision", "driver_sha256", "prototype_sha256",
        "batch_api_sha256", "rust_extension_sha256",
    )
    structured_keys = ("model", "config", "rates")
    exact_matches = all(ref_provenance.get(key) == current_provenance.get(key) for key in exact_keys)
    structured_matches = all(
        json.dumps(ref_provenance.get(key), sort_keys=True, default=json_default)
        == json.dumps(current_provenance.get(key), sort_keys=True, default=json_default)
        for key in structured_keys
    )
    if not exact_matches or not structured_matches:
        raise ValueError("reference provenance, native build, model, or config mismatch")
    with np.load(reference_path) as reference:
        densities = np.asarray(reference["onehot_final_densities"])
        mixture = np.asarray(reference["mixture_final_density"])
        photons = np.asarray(reference["photon_yields"])
    if densities.shape != (3, case.n_states, case.n_states) or mixture.shape != (case.n_states, case.n_states):
        raise ValueError("reference density artifact has unexpected dimensions")
    if (
        photons.shape != (3,)
        or not np.all(np.isfinite(densities))
        or not np.all(np.isfinite(mixture))
        or not np.all(np.isfinite(photons))
    ):
        raise ValueError("reference artifact contains invalid values")
    return densities, mixture, photons


def run_stage(config_name: str, duration_us: float, output_dir: Path, stage: str) -> dict[str, Any]:
    """Run one isolated case/stage in a process supervised by ``main``.

    The supervisor timeout applies to the whole stage, including model build,
    strict reference generation/validation and I/O. Method compute walls are
    reported separately and are not mislabeled as total stage elapsed time.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    from dataclasses import replace

    stage_path = output_dir / f"{stage}.json"
    if stage_path.exists():
        raise FileExistsError(f"refusing to overwrite {stage_path}")
    stage_started = time.perf_counter()
    before = resource_snapshot()
    before_guard = guard(before)
    if not before_guard["okay"]:
        result = {
            "status": "paused_resource_guard", "timings_accepted": False,
            "stage": stage, "resource_before": before, "resource_guard_before": before_guard,
        }
        write_checkpoint(stage_path, result)
        return result

    config = replace(proto.authorized_case_configs()[config_name], duration_s=duration_us * 1e-6)
    setup_started = time.perf_counter()
    case = driver.build_case(config, prepare_frozen=True)
    setup_seconds = time.perf_counter() - setup_started
    span = _span(case)
    onehots, weights, mixture, _ = _initials(case)
    payload: dict[str, Any] = {
        "status": "complete", "stage": stage, "case": asdict(config),
        "span_s": list(span), "labels": list(LABELS), "mixture_weights": weights,
        "case_setup_seconds": setup_seconds, "provenance": provenance(case),
        "resource_before": before, "resource_guard_before": before_guard,
    }
    verified_reference = None
    if stage not in {"reference", "rhs_profile"}:
        verified_reference = load_verified_reference(case, output_dir, payload["provenance"])

    if stage == "reference":
        started = time.perf_counter()
        tight, tight_stats = rust_full_density_endpoint(
            case, onehots, span, solver="dopri5", reltol=2e-11, abstol=2e-13,
            dt=5e-11, parallel=True, threads=3,
        )
        tight_wall = time.perf_counter() - started
        cross_started = time.perf_counter()
        cross, cross_stats = rust_full_density_endpoint(
            case, onehots, span, solver="tsit5", reltol=2e-11, abstol=2e-13,
            dt=5e-11, parallel=True, threads=3,
        )
        cross_wall = time.perf_counter() - cross_started
        checks = [accuracy_row(a, b) for a, b in zip(tight, cross, strict=True)]
        physical = [driver.full_density_metrics(rho) for rho in tight]
        if not all(
            row["density_max_abs"] <= 1e-8
            and row["population_max_relative_above_1e-6"] <= 1e-4
            and abs(check["trace_real"] - 1.0) <= 1e-9
            and check["trace_abs_imag"] <= 1e-9
            and check["hermiticity_max_abs"] <= 1e-9
            and check["minimum_eigenvalue"] >= -1e-8
            for row, check in zip(checks, physical, strict=True)
        ):
            raise RuntimeError("Dopri5/TSit5 full-density strict-reference crosscheck failed")
        mixture_rho, mixture_stats = rust_full_density_endpoint(
            case, mixture, span, solver="dopri5", reltol=2e-11, abstol=2e-13,
            dt=5e-11, parallel=False, threads=1,
        )
        photon_values = []
        photon_crosscheck_values = []
        for rho0 in onehots:
            result = observables.propagate_endpoint(
                case, rho0, span, output_kind="photons", solver="dopri5",
                reltol=2e-11, execution_mode="structured",
            )
            photon_values.append(float(result["photon_yield"]))
            cross_result = observables.propagate_endpoint(
                case, rho0, span, output_kind="photons", solver="tsit5",
                reltol=2e-11, execution_mode="structured",
            )
            photon_crosscheck_values.append(float(cross_result["photon_yield"]))
        photon_crosscheck_errors = np.abs(
            np.asarray(photon_values) - np.asarray(photon_crosscheck_values)
        )
        photon_reference_passes = bool(np.all(
            photon_crosscheck_errors <= 1e-6 + 1e-5 * np.abs(photon_values)
        ))
        if not photon_reference_passes:
            raise RuntimeError("Dopri5/TSit5 photon reference crosscheck failed")
        strict_npz = output_dir / "strict_reference.npz"
        if strict_npz.exists():
            raise FileExistsError(strict_npz)
        np.savez_compressed(
            strict_npz, onehot_final_densities=tight, mixture_final_density=mixture_rho[0],
            photon_yields=np.asarray(photon_values),
        )
        payload.update({
            "reference_checks_per_initial": checks,
            "reference_physical_checks_per_initial": physical,
            "dopri5_full_endpoint_wall_seconds": tight_wall,
            "tsit5_full_endpoint_crosscheck_wall_seconds": cross_wall,
            "dopri5_stats": tight_stats, "tsit5_stats": cross_stats,
            "photon_yields": photon_values,
            "photon_tsit5_crosscheck_yields": photon_crosscheck_values,
            "photon_crosscheck_absolute_errors": photon_crosscheck_errors.tolist(),
            "mixture_linearity_density_error": float(np.max(np.abs(
                mixture_rho[0] - np.einsum("k,kij->ij", weights, tight)
            ))),
            "reference_npz": strict_npz.name,
            "reference_npz_sha256": sha256(strict_npz),
            "accuracy_accepted": photon_reference_passes,
        })
    elif stage == "rhs_profile":
        warmup = proto.warm_numba_kernels()
        plan_started = time.perf_counter()
        driver._prepare_switch_plans(case, static=False)
        plan_prep = time.perf_counter() - plan_started
        context_started = time.perf_counter()
        context = proto.PreparedRHSContext.from_prepared(case.prepared_switch_states[(0, 0)])
        context_prep = time.perf_counter() - context_started
        prepared = context.prepared
        raw = np.arange(case.n_states * case.n_states, dtype=np.float64).reshape(case.n_states, case.n_states)
        dense = raw + raw.T + 1j * (raw - raw.T)
        dense /= np.trace(dense).real
        samples = {}
        for label, rho in (("initial_onehot", onehots[0]), ("dense_complex", dense)):
            packed = proto.pack_density_block(rho[None, :, :], prepared.layout)
            times = (span[0], sum(span) / 2.0)
            started = time.perf_counter()
            for index in range(20):
                context.rhs_block(packed, times[index % 2])
            samples[label] = {"calls": 20, "wall_seconds": time.perf_counter() - started}
        payload.update({
            "numba_warmup_seconds": warmup, "switch_plan_prep_seconds": plan_prep,
            "rhs_context_prep_seconds": context_prep, "probe_timings": samples,
            "timing_boundary": "Python/SciPy/Numba end-to-end RHS wall; not native-instrumented",
        })
    elif stage in {"rust_serial", "rust_parallel_3", "shared_block"}:
        reference_path = output_dir / "strict_reference.npz"
        if stage == "shared_block":
            warmup = proto.warm_numba_kernels()
            plan_started = time.perf_counter()
            driver._prepare_switch_plans(case, static=False)
            switch_plan_setup = time.perf_counter() - plan_started
            started = time.perf_counter()
            onehot_result = proto.integrate_segmented_block(
                case, onehots, span, solver="DOP853", reltol=RELTOL, abstol=ABSTOL, dt=DT,
            )
            onehot_wall = time.perf_counter() - started
            mixture_started = time.perf_counter()
            mix_result = proto.integrate_segmented_block(
                case, mixture, span, solver="DOP853", reltol=RELTOL, abstol=ABSTOL, dt=DT,
            )
            mixture_wall = time.perf_counter() - mixture_started
            actual, stats = onehot_result
            mix_actual, mix_stats = mix_result
            payload["numba_warmup_seconds"] = warmup
            payload["switch_plan_setup_seconds"] = switch_plan_setup
            payload["solver_stats"] = stats
            payload["mixture_solver_stats"] = mix_stats
            payload["onehot_block_compute_wall_seconds"] = onehot_wall
            payload["warm_amortized_method_wall_seconds"] = onehot_wall
            payload["first_use_method_total_seconds"] = warmup + switch_plan_setup + onehot_wall
            payload["first_use_total_including_case_setup_seconds"] = setup_seconds + warmup + switch_plan_setup + onehot_wall
            payload["mixture_validation_compute_wall_seconds"] = mixture_wall
            payload["mixture_linearity_error"] = float(np.max(np.abs(
                mix_actual[0] - np.einsum("k,kij->ij", weights, actual)
            )))
        else:
            started = time.perf_counter()
            parallel = stage == "rust_parallel_3"
            actual, stats = rust_full_density_endpoint(
                case, onehots, span, solver="tsit5", reltol=RELTOL, abstol=ABSTOL,
                dt=DT, parallel=parallel, threads=3 if parallel else 1,
            )
            payload["solver_stats"] = stats
            payload["rust_solver"] = "tsit5 structured"
            payload["core_budget"] = 3 if parallel else 1
            onehot_wall = time.perf_counter() - started
        wall = onehot_wall
        ref_densities, ref_mix, _ = verified_reference
        errors = [accuracy_row(a, b) for a, b in zip(actual, ref_densities, strict=True)]
        physical = [driver.full_density_metrics(rho) for rho in actual]
        passes = all(
            error["density_max_abs"] <= 1e-7
            and error["population_max_abs"] <= 1e-7
            and error["population_max_relative_above_1e-6"] <= 1e-3
            and error["trace_max_abs_error"] <= 1e-9
            and error["hermiticity_max_abs"] <= 1e-9
            and check["minimum_eigenvalue"] >= -1e-8
            for error, check in zip(errors, physical, strict=True)
        )
        if stage == "shared_block":
            mix_error = accuracy_row(mix_actual[0], ref_mix)
            passes = bool(passes and mix_error["density_max_abs"] <= 1e-7 and payload["mixture_linearity_error"] <= 1e-7)
            payload["mixture_accuracy"] = mix_error
        data_path = output_dir / f"{stage}.npz"
        if data_path.exists():
            raise FileExistsError(data_path)
        np.savez_compressed(data_path, endpoint_densities=actual)
        payload.update({
            "method_wall_seconds": wall,
            "budget_seconds": METHOD_BUDGET_SECONDS,
            "budget_exceeded": wall > METHOD_BUDGET_SECONDS,
            "per_initial_accuracy": errors,
            "per_initial_physical_checks": physical,
            "accuracy_accepted": bool(passes),
            "density_npz": data_path.name,
            "density_npz_sha256": sha256(data_path),
        })
    else:
        objective = (
            proto.photon_objective(case, driver.GAMMA)
            if stage == "adjoint_photons"
            else proto.j1_population_objective(case)
        )
        warmup = proto.warm_numba_kernels()
        plan_started = time.perf_counter()
        driver._prepare_switch_plans(case, static=False)
        switch_plan_setup = time.perf_counter() - plan_started
        started = time.perf_counter()
        lambda0, stats = proto.integrate_segmented_adjoint(
            case, objective, span, solver="DOP853", reltol=RELTOL,
            abstol=ABSTOL, dt=DT,
        )
        adjoint_wall = time.perf_counter() - started
        responses = proto.diagonal_objective_responses(lambda0)
        onehot_values = np.asarray([responses[case.state_labels[label]] for label in LABELS])
        mixture_value = float(np.dot(weights, onehot_values))
        direct_values = []
        direct_wall_started = time.perf_counter()
        for rho0 in (*onehots, mixture):
            if stage == "adjoint_photons":
                forward = observables.propagate_endpoint(
                    case, rho0, span, output_kind="photons", solver="tsit5",
                    reltol=RELTOL, execution_mode="structured",
                )
                direct_values.append(float(forward["photon_yield"]))
            else:
                selected = observables._group_indices(case)["X_J1"]
                forward = observables.propagate_endpoint(
                    case, rho0, span, output_kind="selected_j1", solver="tsit5",
                    reltol=RELTOL, execution_mode="structured", selected_indices=selected,
                )
                direct_values.append(float(np.sum(forward["endpoint_values"])))
        direct_wall = time.perf_counter() - direct_wall_started
        ref_path = output_dir / "strict_reference.npz"
        if not ref_path.exists():
            raise FileNotFoundError("run the reference stage first")
        _, strict_mix, strict_photons = verified_reference
        if stage == "adjoint_photons":
            strict = np.concatenate((strict_photons, [float(np.dot(weights, strict_photons))]))
        else:
            indices = observables._group_indices(case)["X_J1"]
            strict_density = [*ref_densities, strict_mix]
            strict = np.asarray([np.sum(np.diag(rho).real[indices]) for rho in strict_density])
        predicted = np.concatenate((onehot_values, [mixture_value]))
        abs_errors = np.abs(predicted - strict)
        if stage == "adjoint_photons":
            accuracy_accepted = bool(np.all(abs_errors <= 1e-6 + 1e-5 * np.abs(strict)))
            relative_errors = None
        else:
            relative_errors = np.zeros_like(abs_errors)
            populated = strict >= 1e-6
            relative_errors[populated] = abs_errors[populated] / strict[populated]
            accuracy_accepted = bool(np.all(abs_errors <= 1e-7) and np.all(relative_errors <= 1e-3))
        payload.update({
            "numba_warmup_seconds": warmup,
            "switch_plan_setup_seconds": switch_plan_setup,
            "solver_stats": stats,
            "adjoint_wall_seconds": adjoint_wall,
            "warm_amortized_method_wall_seconds": adjoint_wall,
            "first_use_method_total_seconds": warmup + switch_plan_setup + adjoint_wall,
            "first_use_total_including_case_setup_seconds": setup_seconds + warmup + switch_plan_setup + adjoint_wall,
            "single_output_forward_validation_wall_seconds": direct_wall,
            "single_output_forward_solver": "Rust tsit5 structured",
            "adjoint_values_onehots_and_mixture": predicted.tolist(),
            "strict_reference_values": strict.tolist(),
            "single_output_forward_values": direct_values,
            "strict_reference_absolute_errors": abs_errors.tolist(),
            "strict_reference_relative_errors": relative_errors.tolist() if relative_errors is not None else None,
            "accuracy_accepted": accuracy_accepted,
            "timings_accepted": bool(adjoint_wall <= METHOD_BUDGET_SECONDS),
        })

    after = resource_snapshot()
    after_guard = guard(after)
    payload["resource_after"] = after
    payload["resource_guard_after"] = after_guard
    payload["whole_stage_elapsed_seconds"] = time.perf_counter() - stage_started
    payload["timings_accepted"] = bool(
        after_guard["okay"]
        and payload.get("timings_accepted", True)
        and not payload.get("budget_exceeded", False)
    )
    if stage in {"rust_serial", "rust_parallel_3", "shared_block"}:
        payload["timings_accepted"] = bool(payload["timings_accepted"] and not payload.get("budget_exceeded", False))
    payload["resource_cpu_median_limit_percent"] = CPU_MEDIAN_LIMIT
    write_checkpoint(stage_path, payload)
    return payload


def supervise_case(config_name: str, duration_us: float, target: Path, timeout_s: float) -> None:
    """Run every stage in an isolated child with a hard stage timeout."""
    target.mkdir(parents=True, exist_ok=False)
    for stage in STAGES:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker-stage", stage, "--case", config_name,
            "--duration-us", str(duration_us), "--output-dir", str(target),
        ]
        try:
            completed = subprocess.run(
                command, cwd=HERE, capture_output=True, text=True, timeout=timeout_s,
            )
            if completed.returncode != 0:
                failure = {
                    "status": "stage_failed", "timings_accepted": False,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                }
                write_checkpoint(target / f"{stage}.failure.json", failure)
        except subprocess.TimeoutExpired as exc:
            timeout_result = {
                "status": "timed_out_unmeasured", "timings_accepted": False,
                "timeout_seconds": timeout_s,
                "stdout_tail": str(exc.stdout)[-2000:] if exc.stdout else "",
                "stderr_tail": str(exc.stderr)[-2000:] if exc.stderr else "",
                "interpretation": "no completed measurement; timeout alone is not a correctness failure",
            }
            write_checkpoint(target / f"{stage}.timeout.json", timeout_result)
    stage_status = {}
    for stage in STAGES:
        result_path = target / f"{stage}.json"
        timeout_path = target / f"{stage}.timeout.json"
        failure_path = target / f"{stage}.failure.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            stage_status[stage] = {
                "status": result.get("status"),
                "timings_accepted": result.get("timings_accepted", False),
                "accuracy_accepted": result.get("accuracy_accepted"),
            }
        elif timeout_path.exists():
            stage_status[stage] = {"status": "timed_out_unmeasured", "timings_accepted": False}
        elif failure_path.exists():
            stage_status[stage] = {"status": "stage_failed", "timings_accepted": False}
        else:
            stage_status[stage] = {"status": "missing", "timings_accepted": False}
    summary = {
        "status": "complete" if all(row["status"] == "complete" for row in stage_status.values()) else "partial",
        "duration_us": duration_us,
        "case": config_name,
        "per_method_timeout_seconds": timeout_s,
        "stages": stage_status,
        "timings_accepted": all(
            stage_status[name]["timings_accepted"]
            and stage_status[name].get("accuracy_accepted", False)
            for name in ("rust_serial", "rust_parallel_3", "shared_block", "adjoint_photons", "adjoint_X_J1")
        ),
        "accuracy_accepted": all(
            stage_status[name].get("accuracy_accepted", False)
            for name in ("rust_serial", "rust_parallel_3", "shared_block", "adjoint_photons", "adjoint_X_J1")
        ),
        "interpretation": "timeouts are incomplete measurements, not correctness failures",
    }
    write_checkpoint(target / "supervision.json", summary)


def worker_main(stage: str, config_name: str, duration_us: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        run_stage(config_name, duration_us, output_dir, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("nominal", "detuned_20MHz", "optical_3Gamma"), required=True)
    parser.add_argument("--duration-us", type=float, choices=(1.0, 10.0), default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=METHOD_BUDGET_SECONDS)
    parser.add_argument("--tag", help="new artifact directory name; must not exist")
    parser.add_argument("--worker-stage", choices=STAGES)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker_stage:
        if args.output_dir is None:
            parser.error("--worker-stage requires --output-dir")
        target = args.output_dir.resolve()
        if RESULTS.resolve() not in target.parents:
            raise ValueError("worker artifact directory must remain below the benchmark results directory")
        worker_main(args.worker_stage, args.case, args.duration_us, target)
    else:
        if not args.tag:
            parser.error("the supervisor requires --tag")
        target = (RESULTS / args.tag / f"{args.case}_{args.duration_us:g}us").resolve()
        if RESULTS.resolve() not in target.parents:
            raise ValueError("artifact target must remain below the benchmark results directory")
        supervise_case(args.case, args.duration_us, target, args.timeout_seconds)


if __name__ == "__main__":
    main()
