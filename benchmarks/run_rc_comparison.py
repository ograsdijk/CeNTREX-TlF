"""Measured RC method comparisons; run stages sequentially on an idle machine.

This is a benchmark-only harness. Full density and photon yield are computed
in separate solves in the ODE path, and both costs are included in wall times.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import partial
from itertools import pairwise
from pathlib import Path

import bench_rc_propagation as driver
import numpy as np
import psutil
import rc_propagation_methods as methods
from threadpoolctl import threadpool_limits


class PeakMemory:
    def __enter__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.stop.wait(0.05)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def resource_snapshot():
    simulations = []
    for process in psutil.process_iter(["pid", "name"]):
        if "python" not in (process.info["name"] or "").lower():
            continue
        try:
            if any(Path(arg).name == "simulate_all.py" for arg in process.cmdline()):
                simulations.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {
        "cpu_percent_samples": [psutil.cpu_percent(interval=1) for _ in range(3)],
        "available_memory_bytes": psutil.virtual_memory().available,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "known_simulation_pids": simulations,
    }


def accuracy(density, reference, photons, ref_photons):
    density = np.asarray(density)
    reference = np.asarray(reference)
    pops = np.diagonal(density, axis1=-2, axis2=-1).real
    ref_pops = np.diagonal(reference, axis1=-2, axis2=-1).real
    populated = ref_pops >= 1e-6
    relative = (
        float(np.max(np.abs((pops[populated] - ref_pops[populated]) / ref_pops[populated])))
        if np.any(populated)
        else 0.0
    )
    traces = np.trace(density, axis1=-2, axis2=-1)
    herm = density - density.swapaxes(-1, -2).conj()
    minimum = float(np.linalg.eigvalsh((density + density.swapaxes(-1, -2).conj()) / 2).min())
    result = {
        "density_max_abs": float(np.max(np.abs(density - reference))),
        "population_max_abs": float(np.max(np.abs(pops - ref_pops))),
        "population_max_relative_above_1e-6": relative,
        "trace_max_abs_error": float(np.max(np.abs(traces - 1))),
        "hermiticity_max_abs": float(np.max(np.abs(herm))),
        "minimum_eigenvalue": minimum,
        "photon_absolute_error": float(abs(photons - ref_photons)),
        "photon_relative_error": (
            float(abs(photons - ref_photons) / abs(ref_photons))
            if ref_photons != 0
            else (0.0 if photons == 0 else None)
        ),
        "photon_yield": float(photons),
    }
    result["passes"] = bool(
        result["density_max_abs"] <= 1e-7
        and result["population_max_abs"] <= 1e-7
        and relative <= 1e-3
        and result["trace_max_abs_error"] <= 1e-9
        and result["hermiticity_max_abs"] <= 1e-9
        and minimum >= -1e-8
        and result["photon_absolute_error"] <= 1e-6 + 1e-5 * abs(ref_photons)
    )
    return result


def timed(call, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
    return result, samples


def ode_result(
    case,
    rho,
    span,
    solver="dopri5",
    reltol=1e-9,
    execution_mode="expanded_sparse",
    frame=None,
    dt_s=1e-10,
):
    initial = rho if frame is None else methods.phase_frame_density(rho, span[0], frame)
    times, densities, stats = driver.propagate_segmented(
        case,
        initial,
        span,
        solver=solver,
        reltol=reltol,
        abstol=reltol / 100,
        dt=dt_s,
        execution_mode=execution_mode,
        save_step_s=100e-9,
    )
    if frame is not None:
        densities = np.asarray(
            [
                methods.original_frame_density(matrix, t, frame)
                for t, matrix in zip(times, densities, strict=True)
            ]
        )
    return times, densities, driver.photon_yield(stats), stats


def _reference_comparison(
    result, other, *, max_density: float, max_photons: float, max_relative_population: float
) -> dict:
    np.testing.assert_array_equal(result[0], other[0])
    measured = accuracy(result[1], other[1], result[2], other[2])
    measured["passes_reference_tolerance"] = bool(
        measured["density_max_abs"] <= max_density
        and measured["photon_absolute_error"] <= max_photons
        and measured["population_max_relative_above_1e-6"] <= max_relative_population
        and measured["passes"]
    )
    return measured


def validated_reference(
    case,
    rho,
    span,
    reference_tolerance: float,
    *,
    execution_mode: str = "expanded_sparse",
    progress_callback: Callable[[str, list[float]], None] | None = None,
) -> tuple[tuple, dict]:
    """Build and validate a refined dopri5 reference plus a tsit5 crosscheck."""
    if not np.isfinite(reference_tolerance) or not 0 < reference_tolerance < 1e-10:
        raise ValueError("reference_tolerance must be finite, positive and less than 1e-10")
    stages = (
        ("primary_dopri5", "dopri5", 1e-10, 1e-10),
        ("refined_dopri5", "dopri5", reference_tolerance, 5e-11),
        ("tight_dopri5", "dopri5", reference_tolerance / 3.0, 2.5e-11),
        ("tight_tsit5", "tsit5", reference_tolerance / 3.0, 2.5e-11),
    )
    results: dict[str, tuple] = {}
    timings: dict[str, list[float]] = {}
    settings: dict[str, dict[str, float | str]] = {}
    for name, solver, tolerance, dt_s in stages:
        result, samples = timed(
            partial(
                ode_result,
                case,
                rho,
                span,
                solver=solver,
                reltol=tolerance,
                execution_mode=execution_mode,
                dt_s=dt_s,
            ),
            1,
        )
        results[name] = result
        timings[name] = samples
        settings[name] = {"solver": solver, "reltol": tolerance, "dt_s": dt_s}
        if progress_callback is not None:
            progress_callback(name, samples)

    max_density = 1e-8
    max_photons = 1e-7
    max_relative_population = 1e-4
    primary_refined = _reference_comparison(
        results["primary_dopri5"],
        results["refined_dopri5"],
        max_density=max_density,
        max_photons=max_photons,
        max_relative_population=max_relative_population,
    )
    refined_tight = _reference_comparison(
        results["refined_dopri5"],
        results["tight_dopri5"],
        max_density=max_density,
        max_photons=max_photons,
        max_relative_population=max_relative_population,
    )
    crosscheck = _reference_comparison(
        results["tight_tsit5"],
        results["tight_dopri5"],
        max_density=max_density,
        max_photons=max_photons,
        max_relative_population=max_relative_population,
    )
    tight_physical = accuracy(
        results["tight_dopri5"][1],
        results["tight_dopri5"][1],
        results["tight_dopri5"][2],
        results["tight_dopri5"][2],
    )
    checks = {
        "primary_to_refined": primary_refined,
        "refined_to_tight": refined_tight,
        "tsit5_crosscheck": crosscheck,
        "tight_reference_physical": tight_physical,
    }
    validated = bool(
        all(
            check["passes_reference_tolerance"]
            for name, check in checks.items()
            if name != "tight_reference_physical"
        )
        and tight_physical["passes"]
    )
    metadata = {
        "validated": validated,
        "checks": checks,
        "timings": timings,
        "settings": settings,
        "total_seconds": float(sum(sum(samples) for samples in timings.values())),
    }
    return results["tight_dopri5"], metadata


def _reference_checkpoint(payload, label: str, save: Callable[[], None]):
    progress = {"initial_label": label, "stages": {}}
    payload.setdefault("reference_progress", []).append(progress)
    save()

    def checkpoint(stage: str, samples: list[float]) -> None:
        progress["stages"][stage] = samples
        save()

    return checkpoint


def split_trajectory(split, rho, times, dt_s, weights):
    densities = [np.asarray(rho).copy()]
    photons = 0.0
    for lo, hi in pairwise(times):
        final, contribution = split.propagate(
            densities[-1],
            (float(lo), float(hi)),
            switch_values=driver.switch_values,
            switch_boundaries=driver.switch_boundaries(float(lo), float(hi)),
            dt=dt_s,
            return_photons=True,
            integral_weights=weights,
        )
        densities.append(final)
        photons += contribution
    return np.asarray(densities), photons


def run_ode(case, args, payload, save):
    span = (args.entry_ns * 1e-9, args.entry_ns * 1e-9 + case.config.duration_s)
    labels = (
        list(case.state_labels)
        if args.initial == "all"
        else [args.initial or list(case.state_labels)[1]]
    )
    started = time.perf_counter()
    frame_prepared, phase = methods.phase_frame_problem(case.prepared_dynamic)
    frame_case = copy.copy(case)
    frame_case.prepared_dynamic = frame_prepared
    frame_case.prepared_segmented = None
    payload["phase_frame_preparation_seconds"] = time.perf_counter() - started
    payload["phase_convention"] = asdict(phase)
    rows = payload.setdefault("rows", [])
    raw = {}
    for label in labels:
        rho = case.rho0(label)
        print(f"Reference: {label}", flush=True)
        progress = {"initial_label": label, "stages": {}}
        payload.setdefault("reference_progress", []).append(progress)
        save()

        def checkpoint(stage, samples, progress=progress):
            progress["stages"][stage] = samples
            save()

        if args.reference_json is None:
            refined, reference_check = validated_reference(
                case, rho, span, args.reference_tolerance, progress_callback=checkpoint
            )
        else:
            refined, reference_check = load_reference(args.reference_json, payload, label)
        reference_check["initial_label"] = label
        payload.setdefault("reference_checks", []).append(reference_check)
        validated = reference_check["validated"]
        save()
        raw[label + "_times"] = refined[0]
        raw[label + "_reference"] = refined[1]
        raw[label + "_reference_photons"] = refined[2]
        candidates = [
            ("dopri5_1e-7", case, "dopri5", 1e-7, "expanded_sparse", None),
            ("tsit5_1e-7", case, "tsit5", 1e-7, "expanded_sparse", None),
            ("dopri5_1e-9", case, "dopri5", 1e-9, "expanded_sparse", None),
            ("tsit5_1e-9", case, "tsit5", 1e-9, "expanded_sparse", None),
            ("phase_dopri5_1e-9", frame_case, "dopri5", 1e-9, "expanded_sparse", phase),
            ("phase_tsit5_1e-9", frame_case, "tsit5", 1e-9, "expanded_sparse", phase),
        ]
        if args.execution_modes:
            candidates += [
                ("structured_dopri5_1e-9", case, "dopri5", 1e-9, "structured", None),
                ("structured_tsit5_1e-9", case, "tsit5", 1e-9, "structured", None),
                ("phase_structured_tsit5_1e-9", frame_case, "tsit5", 1e-9, "structured", phase),
            ]
        if args.methods:
            candidates = [item for item in candidates if item[0] in args.methods]
        for name, candidate_case, solver, tolerance, mode, frame in candidates:
            print(f"Candidate: {name}", flush=True)
            result, samples = timed(
                partial(ode_result, candidate_case, rho, span, solver, tolerance, mode, frame),
                args.repeats,
            )
            np.testing.assert_array_equal(result[0], refined[0])
            row = {
                "initial_label": label,
                "method": name,
                "wall_seconds": samples,
                "median_seconds": float(np.median(samples)),
                "reference_validated": validated,
                "reference_check": reference_check,
                "rhs_calls": sum(s.get("rhs_calls", 0) for s in result[3]),
                **accuracy(result[1], refined[1], result[2], refined[2]),
            }
            row["passes"] = bool(row["passes"] and validated)
            row["full_density_wall_seconds"] = [
                s.get("full_density_wall_seconds") for s in result[3]
            ]
            row["photon_integral_wall_seconds"] = [
                s.get("photon_integral_wall_seconds") for s in result[3]
            ]
            rows.append(row)
            raw[label + "_" + name] = result[1]
            save()
    np.savez_compressed(args.output_dir / (args.tag + "_densities.npz"), **raw)


def load_reference(path, payload, label):
    """Reuse only a validated reference with identical model and numerical code."""
    path = Path(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("status") != "complete" or not saved.get("timings_accepted"):
        raise ValueError("reference artifact must be complete and uncontended")
    for key in ("config", "model", "amplitude_envelope", "driver_sha256", "rust_extension_sha256"):
        if json.dumps(saved["provenance"][key], sort_keys=True, default=str) != json.dumps(
            payload["provenance"][key], sort_keys=True, default=str
        ):
            raise ValueError(f"reference provenance mismatch: {key}")
    for key in ("helper_sha256", "batch_api_sha256"):
        if saved[key] != payload[key]:
            raise ValueError(f"reference code mismatch: {key}")
    check = next(item for item in saved["reference_checks"] if item["initial_label"] == label)
    required = {
        "primary_to_refined",
        "refined_to_tight",
        "tsit5_crosscheck",
        "tight_reference_physical",
    }
    if not check["validated"] or not required.issubset(check["checks"]):
        raise ValueError("reference did not pass convergence checks")
    arrays_path = path.with_name(path.stem + "_densities.npz")
    with np.load(arrays_path) as arrays:
        if saved.get("arguments", {}).get("stage") == "batch":
            index = saved["initial_labels"].index(label)
            times = arrays["times"].copy()
            density = arrays["reference"][index].copy()
        else:
            times = arrays[label + "_times"].copy()
            density = arrays[label + "_reference"].copy()
    photons = check["checks"]["tight_reference_physical"]["photon_yield"]
    payload["reused_reference"] = {
        "json_path": str(path.resolve()),
        "json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
    }
    return (times, density, photons, []), copy.deepcopy(check)


def run_static(case, args, payload, save):
    if case.amplitude_envelope != "constant":
        raise ValueError("static split factors require constant amplitudes")
    started = time.perf_counter()
    driver._prepare_switch_plans(case, static=True)
    phase = methods.infer_eom_phase(case.system, case.params_dynamic)
    payload["static_plan_preparation_seconds"] = time.perf_counter() - started
    rho = case.rho0(args.initial or list(case.state_labels)[1])
    span = (args.entry_ns * 1e-9, args.entry_ns * 1e-9 + case.config.duration_s)
    weights = [(i, driver.GAMMA) for i in phase.excited_indices]
    checkpoint = _reference_checkpoint(payload, args.initial or list(case.state_labels)[1], save)
    reference, reference_check = validated_reference(
        case, rho, span, args.reference_tolerance, progress_callback=checkpoint
    )
    payload["reference_check"] = reference_check
    payload["reference_validated"] = reference_check["validated"]
    payload["reference_seconds"] = reference_check["total_seconds"]
    save()
    estimate = 4 * methods.estimate_dense_factor_bytes(case.n_states)
    payload["dense_four_factor_workspace_estimate_bytes"] = estimate
    if args.factor_method == "dense" and estimate > 8 * 1024**3:
        payload["gate_failure"] = "four-factor workspace estimate exceeds 8 GiB"
        save()
        return
    factors = {}
    for key, prepared in case.prepared_static_switch_states.items():
        print(f"Factor static generator {key}", flush=True)
        factor, metadata = methods.prepare_static_factor(prepared, method=args.factor_method)
        factors[key] = factor
        payload.setdefault("factorizations", []).append({"switch_state": key, **metadata})
        save()
    factor_setup = {tuple(row["switch_state"]): row for row in payload["factorizations"]}
    split = methods.StaticSplit(
        case.prepared_dynamic.layout,
        factors,
        phase,
        sum(row["estimated_bytes"] for row in payload["factorizations"]),
        factor_setup,
    )
    rows = payload.setdefault("rows", [])
    raw = {"times": reference[0], "reference": reference[1]}
    for dt_ns in args.split_steps_ns:
        print(f"Split step {dt_ns} ns", flush=True)
        result, samples = timed(
            partial(split_trajectory, split, rho, reference[0], dt_ns * 1e-9, weights), args.repeats
        )
        rows.append(
            {
                "method": args.factor_method + "_split",
                "dt_ns": dt_ns,
                "wall_seconds": samples,
                "median_seconds": float(np.median(samples)),
                "validation_scope": "all saved full density matrices and total photon yield",
                "reference_validated": reference_check["validated"],
                **accuracy(result[0], reference[1], result[1], reference[2]),
            }
        )
        rows[-1]["passes"] = bool(rows[-1]["passes"] and reference_check["validated"])
        raw["split_" + str(dt_ns)] = result[0]
        save()
    no_eom_case = copy.copy(case)
    no_eom_case.prepared_dynamic = driver._prepare(
        case.system,
        driver._params_for(case.selectors, case.config, dynamic_switches=True, dynamic_eom=False),
    )
    no_eom_case.prepared_segmented = None
    no_eom_checkpoint = _reference_checkpoint(payload, "no_eom", save)
    no_eom_reference, no_eom_reference_check = validated_reference(
        no_eom_case,
        rho,
        span,
        args.reference_tolerance,
        progress_callback=no_eom_checkpoint,
    )
    payload["no_eom_reference_check"] = no_eom_reference_check
    save()
    split.phase = replace(phase, beta=0)
    result, samples = timed(
        lambda: split_trajectory(split, rho, no_eom_reference[0], 250e-9, weights), args.repeats
    )
    rows.append(
        {
            "method": args.factor_method + "_no_eom",
            "ode_reference_seconds": no_eom_reference_check["total_seconds"],
            "wall_seconds": samples,
            "median_seconds": float(np.median(samples)),
            "reference_validated": no_eom_reference_check["validated"],
            "validation_scope": "all saved full density matrices and total photon yield",
            **accuracy(result[0], no_eom_reference[1], result[1], no_eom_reference[2]),
        }
    )
    rows[-1]["passes"] = bool(rows[-1]["passes"] and no_eom_reference_check["validated"])
    raw["no_eom_reference"] = no_eom_reference[1]
    raw["no_eom_static"] = result[0]
    no_eom_candidate, no_eom_samples = timed(
        lambda: ode_result(
            no_eom_case, rho, span, solver="tsit5", reltol=1e-9, execution_mode="structured"
        ),
        args.repeats,
    )
    np.testing.assert_array_equal(no_eom_candidate[0], no_eom_reference[0])
    rows.append(
        {
            "method": "structured_tsit5_no_eom_1e-9",
            "wall_seconds": no_eom_samples,
            "median_seconds": float(np.median(no_eom_samples)),
            "reference_validated": no_eom_reference_check["validated"],
            **accuracy(
                no_eom_candidate[1], no_eom_reference[1], no_eom_candidate[2], no_eom_reference[2]
            ),
        }
    )
    rows[-1]["passes"] = bool(rows[-1]["passes"] and no_eom_reference_check["validated"])
    np.savez_compressed(args.output_dir / (args.tag + "_densities.npz"), **raw)
    save()


def run_batch(case, args, payload, save):
    labels = list(case.state_labels)
    initial = np.stack([case.rho0(label) for label in labels])
    span = (args.entry_ns * 1e-9, args.entry_ns * 1e-9 + case.config.duration_s)
    references = []
    reference_checks = []
    for label, rho in zip(labels, initial, strict=True):
        print(f"Independent batch reference {label}", flush=True)
        progress = {"initial_label": label, "stages": {}}
        payload.setdefault("reference_progress", []).append(progress)
        save()

        def checkpoint(stage, samples, progress=progress):
            progress["stages"][stage] = samples
            save()

        result, check = validated_reference(
            case,
            rho,
            span,
            args.reference_tolerance,
            progress_callback=checkpoint,
            execution_mode=args.batch_execution_mode,
        )
        references.append(result)
        reference_checks.append(check)
        payload.setdefault("reference_checks", []).append({"initial_label": label, **check})
        save()
    for result in references[1:]:
        np.testing.assert_array_equal(result[0], references[0][0])
    reference_density = np.stack([item[1] for item in references])
    reference_photons = np.asarray([item[2] for item in references])
    payload["initial_labels"] = labels
    raw = {"reference": reference_density, "times": references[0][0]}
    for parallel, threads in ((False, 1), (True, 3), (True, 8)):
        print(f"Batch parallel={parallel}, threads={threads}", flush=True)
        result, seconds = timed(
            partial(
                driver.propagate_segmented_batch,
                case,
                initial,
                span,
                solver="tsit5",
                execution_mode=args.batch_execution_mode,
                reltol=1e-9,
                abstol=1e-11,
                save_step_s=100e-9,
                parallel=parallel,
                threads=threads,
            ),
            args.repeats,
        )
        np.testing.assert_array_equal(result[0], references[0][0])
        photons = np.sum([s["photon_integral"] for s in result[2]], axis=0)
        errors = [
            accuracy(result[1][i], reference_density[i], photons[i], reference_photons[i])
            for i in range(len(labels))
        ]
        payload["rows"].append(
            {
                "execution_mode": args.batch_execution_mode,
                "parallel": parallel,
                "threads": threads,
                "wall_seconds": seconds,
                "median_seconds": float(np.median(seconds)),
                "per_initial_accuracy": errors,
                "reference_validated": all(item["validated"] for item in reference_checks),
                "passes": all(e["passes"] for e in errors)
                and all(item["validated"] for item in reference_checks),
            }
        )
        raw[f"parallel_{parallel}_threads_{threads}"] = result[1]
        save()
    np.savez_compressed(args.output_dir / (args.tag + "_densities.npz"), **raw)


def run_sparse_pilot(case, args, payload, save):
    started = time.perf_counter()
    driver._prepare_switch_plans(case, static=True)
    phase = methods.infer_eom_phase(case.system, case.params_dynamic)
    payload["static_plan_preparation_seconds"] = time.perf_counter() - started
    layout = case.prepared_dynamic.layout
    full = {}
    for key, prepared in case.prepared_static_switch_states.items():
        full[key], metadata = methods.prepare_static_factor(prepared, method="sparse_expm")
        payload.setdefault("factorizations", []).append({"switch_state": key, **metadata})
    started = time.perf_counter()
    initials = np.stack([layout.pack(case.rho0(label)) for label in case.state_labels])
    keep, sinks = methods.common_reachable_coordinates(
        [factor.generator for factor in full.values()], initials, layout, phase
    )
    payload["common_support"] = {
        "full_dimension": layout.packed_len,
        "retained_dimension": len(keep),
        "common_sink_count": len(sinks),
        "retained_indices": keep.tolist(),
    }
    reduced = {
        key: methods.SparseExpmFactor(layout, factor.generator, 0, keep)
        for key, factor in full.items()
    }
    payload["common_support_preparation_seconds"] = time.perf_counter() - started
    for key, factor in full.items():
        payload.setdefault("generator_norms", []).append(
            {
                "switch_state": key,
                "full_one_norm": float(abs(factor.generator).sum(axis=0).max()),
                "reduced_one_norm": float(abs(reduced[key].generator).sum(axis=0).max()),
            }
        )
    rho = case.rho0(args.initial or list(case.state_labels)[1])
    key = driver.switch_values(125e-9)
    for interval in (5e-9, 25e-9):
        print(f"Full/reduced sparse pilot {interval * 1e9} ns", flush=True)
        full_result, full_seconds = timed(
            partial(full[key].density_matrices, rho[None], [interval]), 1
        )
        reduced_result, reduced_seconds = timed(
            partial(reduced[key].density_matrices, rho[None], [interval]), 1
        )
        difference = float(np.max(np.abs(full_result - reduced_result)))
        payload.setdefault("support_validation", []).append(
            {
                "interval_s": interval,
                "full_seconds": full_seconds,
                "reduced_seconds": reduced_seconds,
                "density_max_abs": difference,
            }
        )
        if difference > 1e-9:
            raise AssertionError(
                "common-support sparse exponential disagrees with full propagation"
            )
        save()
    span = (args.entry_ns * 1e-9, args.entry_ns * 1e-9 + case.config.duration_s)
    checkpoint = _reference_checkpoint(payload, args.initial or list(case.state_labels)[1], save)
    reference, reference_check = validated_reference(
        case, rho, span, args.reference_tolerance, progress_callback=checkpoint
    )
    baseline, base_seconds = timed(
        partial(
            ode_result, case, rho, span, solver="tsit5", reltol=1e-9, execution_mode="structured"
        ),
        args.repeats,
    )
    payload["baseline_execution_mode"] = "structured"
    payload["baseline_seconds"] = base_seconds
    payload["reference_seconds"] = reference_check["total_seconds"]
    payload["reference_check"] = reference_check
    payload["reference_validated"] = reference_check["validated"]
    payload["baseline_accuracy"] = accuracy(baseline[1], reference[1], baseline[2], reference[2])
    np.testing.assert_array_equal(baseline[0], reference[0])
    payload["baseline_accuracy"]["reference_validated"] = reference_check["validated"]
    payload["baseline_accuracy"]["passes"] = bool(
        payload["baseline_accuracy"]["passes"] and reference_check["validated"]
    )
    save()
    split = methods.StaticSplit(
        layout, reduced, phase, sum(f.matrix_bytes for f in reduced.values()), {}
    )
    weights = [(i, driver.GAMMA) for i in phase.excited_indices]
    raw = {"times": reference[0], "reference": reference[1]}
    for dt_ns in args.split_steps_ns:
        print(f"Reduced sparse split {dt_ns} ns", flush=True)
        result, seconds = timed(
            partial(split_trajectory, split, rho, reference[0], dt_ns * 1e-9, weights), args.repeats
        )
        row = {
            "method": "reduced_sparse_split",
            "dt_ns": dt_ns,
            "wall_seconds": seconds,
            "median_seconds": float(np.median(seconds)),
            "reference_validated": reference_check["validated"],
            **accuracy(result[0], reference[1], result[1], reference[2]),
        }
        row["passes"] = bool(row["passes"] and reference_check["validated"])
        payload["rows"].append(row)
        raw["split_" + str(dt_ns)] = result[0]
        save()
        if row["passes"]:
            break
        if row["median_seconds"] > float(np.median(base_seconds)):
            payload["refinement_stop"] = (
                "already slower than accepted ODE before achieving density accuracy"
            )
            break
    no_eom_case = copy.copy(case)
    started = time.perf_counter()
    no_eom_case.prepared_dynamic = driver._prepare(
        case.system,
        driver._params_for(case.selectors, case.config, dynamic_switches=True, dynamic_eom=False),
    )
    no_eom_case.prepared_segmented = None
    payload["no_eom_preparation_seconds"] = time.perf_counter() - started
    no_eom_reference, no_eom_check = validated_reference(
        no_eom_case,
        rho,
        span,
        args.reference_tolerance,
        progress_callback=_reference_checkpoint(payload, "no_eom", save),
    )
    payload["no_eom_reference_check"] = no_eom_check
    split.phase = replace(phase, beta=0)
    no_eom_result, no_eom_seconds = timed(
        partial(split_trajectory, split, rho, no_eom_reference[0], 250e-9, weights), args.repeats
    )
    no_eom_ode, no_eom_ode_seconds = timed(
        partial(
            ode_result,
            no_eom_case,
            rho,
            span,
            solver="tsit5",
            reltol=1e-9,
            execution_mode="structured",
        ),
        args.repeats,
    )
    for name, seconds, density, photons in (
        ("reduced_sparse_no_eom", no_eom_seconds, no_eom_result[0], no_eom_result[1]),
        ("structured_tsit5_no_eom_1e-9", no_eom_ode_seconds, no_eom_ode[1], no_eom_ode[2]),
    ):
        row = {
            "method": name,
            "wall_seconds": seconds,
            "median_seconds": float(np.median(seconds)),
            "reference_validated": no_eom_check["validated"],
            **accuracy(density, no_eom_reference[1], photons, no_eom_reference[2]),
        }
        row["passes"] = bool(row["passes"] and no_eom_check["validated"])
        payload["rows"].append(row)
        raw[name] = density
    np.testing.assert_array_equal(no_eom_ode[0], no_eom_reference[0])
    raw["no_eom_reference"] = no_eom_reference[1]
    np.savez_compressed(args.output_dir / (args.tag + "_densities.npz"), **raw)


def run_single(case, args, payload, save):
    """One measured solve, allowing long reference jobs to be run separately."""
    phase = None
    if args.phase_frame:
        case = copy.copy(case)
        case.prepared_dynamic, phase = methods.phase_frame_problem(case.prepared_dynamic)
        case.prepared_segmented = None
    label = args.initial or list(case.state_labels)[1]
    span = (args.entry_ns * 1e-9, args.entry_ns * 1e-9 + case.config.duration_s)
    result, seconds = timed(
        partial(
            ode_result,
            case,
            case.rho0(label),
            span,
            solver=args.solver,
            reltol=args.tolerance,
            execution_mode=args.single_execution_mode,
            frame=phase,
        ),
        args.repeats,
    )
    payload["rows"].append(
        {
            "method": args.solver,
            "phase_frame": args.phase_frame,
            "execution_mode": args.single_execution_mode,
            "tolerance": args.tolerance,
            "initial_label": label,
            "wall_seconds": seconds,
            "median_seconds": float(np.median(seconds)),
            "physical_checks": accuracy(result[1], result[1], result[2], result[2]),
            "reference_validated": False,
            "timing_only": True,
            "segment_stats": result[3],
        }
    )
    np.savez_compressed(
        args.output_dir / (args.tag + "_densities.npz"),
        times=result[0],
        densities=result[1],
        photons=result[2],
    )
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("ode", "static", "batch", "profile", "sparse_pilot", "single", "periodic"),
        default="ode",
    )
    parser.add_argument("--config", choices=tuple(driver.CONFIGS), default="weak")
    parser.add_argument("--envelope", choices=("constant", "gaussian"), default="constant")
    parser.add_argument("--duration-us", type=float)
    parser.add_argument("--detuning-mhz", type=float)
    parser.add_argument("--entry-ns", type=float, default=0)
    parser.add_argument("--initial", default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--execution-modes", action="store_true")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument(
        "--batch-execution-mode",
        choices=("structured", "expanded_sparse"),
        default="expanded_sparse",
    )
    parser.add_argument(
        "--single-execution-mode",
        choices=("structured", "expanded_sparse"),
        default="expanded_sparse",
    )
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--reference-tolerance", type=float, default=3e-11)
    parser.add_argument("--solver", choices=("dopri5", "tsit5"), default="tsit5")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--phase-frame", action="store_true")
    parser.add_argument("--factor-method", choices=("dense", "sparse_expm"), default="dense")
    parser.add_argument("--split-steps-ns", nargs="+", type=float, default=[20, 10, 5])
    parser.add_argument("--output-dir", type=Path, default=driver.RESULTS_DIR)
    parser.add_argument("--tag", default="comparison")
    args = parser.parse_args()
    if (
        args.repeats < 1
        or (args.duration_us is not None and args.duration_us <= 0)
        or not np.isfinite(args.reference_tolerance)
        or not 0 < args.reference_tolerance < 1e-10
    ):
        parser.error("positive repeats/duration and 0 < reference-tolerance < 1e-10 required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = resource_snapshot()
    if resources["known_simulation_pids"] or max(resources["cpu_percent_samples"]) > 25:
        raise RuntimeError(f"machine busy before benchmark: {resources}")
    config = driver.CONFIGS[args.config]
    if args.duration_us is not None:
        config = replace(config, duration_s=args.duration_us * 1e-6)
    if args.detuning_mhz is not None:
        config = replace(config, detuning_mhz=args.detuning_mhz)
    config = replace(config, entry_phase_s=args.entry_ns * 1e-9)
    payload = {
        "arguments": vars(args),
        "resource_conditions_before": resources,
        "timing_boundary": "full density plus a separate native photon-integral solve; both included",
        "memory_sampling_interval_s": 0.05,
        "timings_accepted": False,
        "rows": [],
    }
    path = args.output_dir / (args.tag + ".json")

    def save():
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    with threadpool_limits(limits=1), PeakMemory() as memory:
        start = time.perf_counter()
        case = driver.build_case(config, prepare_frozen=True, amplitude_envelope=args.envelope)
        payload["common_model_and_preparation_seconds"] = time.perf_counter() - start
        payload["provenance"] = driver.provenance(case)
        payload["helper_sha256"] = hashlib.sha256(Path(methods.__file__).read_bytes()).hexdigest()
        payload["suite_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        payload["batch_api_sha256"] = hashlib.sha256(
            (driver.HERE.parent / "centrex_tlf/lindblad/batch.py").read_bytes()
        ).hexdigest()
        save()
        try:
            if args.stage == "ode":
                run_ode(case, args, payload, save)
            elif args.stage == "batch":
                run_batch(case, args, payload, save)
            elif args.stage == "sparse_pilot":
                run_sparse_pilot(case, args, payload, save)
            elif args.stage == "single":
                run_single(case, args, payload, save)
            elif args.stage == "periodic":
                from rc_periodic_pilot import run_periodic_pilot

                pilot = run_periodic_pilot(
                    case, output_path=args.output_dir / (args.tag + "_densities.npz")
                )
                pilot.pop("coarse_final_densities")
                pilot.pop("fine_final_densities")
                payload["periodic_pilot"] = pilot
                payload["periodic_pilot_sha256"] = hashlib.sha256(
                    (driver.HERE / "rc_periodic_pilot.py").read_bytes()
                ).hexdigest()
            elif args.stage == "profile":
                from rc_rhs_profile import profile_rc_rhs

                times = (np.arange(16) + 0.5) * 250e-9
                rho = case.rho0(list(case.state_labels)[1])
                profile = profile_rc_rhs(case, rho, times, repeats=200)
                payload["rhs_profile"] = profile
                vector = np.exp(1j * np.arange(case.n_states)) / np.sqrt(case.n_states)
                dense_rho = 0.5 * rho + 0.5 * np.outer(vector, vector.conj())
                dense_profile = profile_rc_rhs(case, dense_rho, times, repeats=200)
                payload["rhs_profile_dense_state"] = dense_profile
                payload["dense_profile_state"] = (
                    "equal mixture of selected initial population and a normalized complex uniform superposition; synthetic cost probe"
                )
                if any(
                    row["max_rel_rhs_difference_vs_first_mode"] > 1e-10
                    for item in (profile, dense_profile)
                    for row in item["modes"].values()
                ):
                    raise AssertionError("RHS execution modes disagree")
            else:
                run_static(case, args, payload, save)
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = repr(exc)
            payload["peak_process_rss_bytes"] = memory.peak
            save()
            raise
    payload["status"] = "complete"
    payload["peak_process_rss_bytes"] = memory.peak
    payload["resource_conditions_after"] = resource_snapshot()
    timing_environment_clear = bool(
        not payload["resource_conditions_after"]["known_simulation_pids"]
        and max(payload["resource_conditions_after"]["cpu_percent_samples"]) <= 25
    )
    payload["timings_accepted"] = timing_environment_clear
    if not timing_environment_clear:
        payload["status"] = "contended"
    save()
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
