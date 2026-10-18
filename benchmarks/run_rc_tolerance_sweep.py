"""Short, resumable structured-Rust RC ODE tolerance sweep.

The reference and every candidate use exact polarization-switch segments,
absolute EOM time, the same saved-time grid, full density output, and the
solver-native photon integral. This driver intentionally does not extrapolate
the short case to long interaction times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import bench_rc_propagation as driver
import numpy as np
import run_rc_comparison as comparison
from threadpoolctl import threadpool_limits

SWEEP_TOLERANCES = tuple(10.0**(-power) for power in range(2, 10))
FIXED_ABS_CONTROLS = (1e-7, 1e-9)
INITIAL_LABEL = "X_J2_F1_5o2_F2_mF0"
INITIAL_STEP_S = 1e-10
SAVE_STEP_S = 100e-9
POST_POINT_STOP_SECONDS = 900.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_complex(values: np.ndarray) -> bool:
    array = np.asarray(values)
    return bool(np.all(np.isfinite(array.real)) and np.all(np.isfinite(array.imag)))


def compare_trajectory(
    times: np.ndarray,
    densities: np.ndarray,
    photons: float,
    reference_times: np.ndarray,
    reference_densities: np.ndarray,
    reference_photons: float,
) -> dict[str, Any]:
    """Score a full saved trajectory and endpoint, preserving nonfinite failures."""
    times = np.asarray(times, dtype=float)
    densities = np.asarray(densities, dtype=np.complex128)
    ref_times = np.asarray(reference_times, dtype=float)
    ref_densities = np.asarray(reference_densities, dtype=np.complex128)
    finite = bool(
        np.all(np.isfinite(times))
        and _finite_complex(densities)
        and np.isfinite(photons)
        and np.all(np.isfinite(ref_times))
        and _finite_complex(ref_densities)
        and np.isfinite(reference_photons)
    )
    grid_matches = bool(times.shape == ref_times.shape and np.array_equal(times, ref_times))
    if not finite or not grid_matches or densities.shape != ref_densities.shape:
        return {
            "candidate_finite": finite,
            "common_time_grid_matches": grid_matches,
            "density_shape": list(densities.shape),
            "reference_density_shape": list(ref_densities.shape),
            "all_saved": None,
            "endpoint": None,
            "passes": False,
            "failure": "nonfinite output, saved-grid mismatch, or density-shape mismatch",
        }
    try:
        all_saved = comparison.accuracy(densities, ref_densities, photons, reference_photons)
        endpoint = comparison.accuracy(
            densities[-1], ref_densities[-1], photons, reference_photons
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {
            "candidate_finite": finite,
            "common_time_grid_matches": grid_matches,
            "density_shape": list(densities.shape),
            "reference_density_shape": list(ref_densities.shape),
            "all_saved": None,
            "endpoint": None,
            "passes": False,
            "failure": f"accuracy evaluation failed: {type(exc).__name__}: {exc}",
        }
    return {
        "candidate_finite": finite,
        "common_time_grid_matches": grid_matches,
        "hermiticity_metric_interpretation": "packed-real Hermitian output structurally enforces Hermiticity; the value is retained as a reconstruction check but is not an independent integrator diagnostic",
        "all_saved": all_saved,
        "endpoint": endpoint,
        "passes": bool(all_saved["passes"] and endpoint["passes"]),
    }


def _numeric_sum(stats: list[dict[str, Any]], key: str) -> int:
    total = 0
    for segment in stats:
        value = segment.get(key, 0)
        array = np.asarray(value)
        if array.size == 1 and np.isfinite(array.reshape(-1)[0]):
            total += int(array.reshape(-1)[0])
    return total


def _stats_summary(stats: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "segment_count": len(stats),
        "stats_scope": "full-density solver only; the separate photon-integral solve does not collect solver counters",
        "full_density_accepted_steps": _numeric_sum(stats, "accepted_steps"),
        "full_density_rejected_steps": _numeric_sum(stats, "rejected_steps"),
        "full_density_rhs_calls": _numeric_sum(stats, "rhs_calls"),
        "full_density_solver_wall_seconds": float(
            sum(float(row.get("full_density_wall_seconds", 0.0)) for row in stats)
        ),
        "photon_integral_solver_wall_seconds": float(
            sum(float(row.get("photon_integral_wall_seconds", 0.0)) for row in stats)
        ),
        "combined_solver_call_wall_seconds": float(
            sum(float(row.get("combined_wall_seconds", 0.0)) for row in stats)
        ),
        "per_segment_full_density_stats": [
            {
                "accepted_steps": row.get("accepted_steps"),
                "rejected_steps": row.get("rejected_steps"),
                "rhs_calls": row.get("rhs_calls"),
                "full_density_solver_wall_seconds": row.get("full_density_wall_seconds"),
                "photon_integral_solver_wall_seconds": row.get("photon_integral_wall_seconds"),
                "combined_solver_call_wall_seconds": row.get("combined_wall_seconds"),
            }
            for row in stats
        ],
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.stem + ".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _resource_ok(snapshot: dict[str, Any]) -> tuple[bool, str | None]:
    if snapshot["known_simulation_pids"]:
        return False, "known simulation process detected between independent points"
    if max(snapshot["cpu_percent_samples"]) > 25:
        return False, "background CPU exceeded 25% between independent points"
    return True, None


def _run_one(
    case: driver.RCCase,
    rho0: np.ndarray,
    span: tuple[float, float],
    *,
    solver: str,
    reltol: float,
    abstol: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    started = time.perf_counter()
    times, densities, stats = driver.propagate_segmented(
        case,
        rho0,
        span,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        dt=INITIAL_STEP_S,
        execution_mode="structured",
        save_step_s=SAVE_STEP_S,
    )
    outer_wall = time.perf_counter() - started
    return (
        {
            "outer_propagation_wall_seconds": outer_wall,
            "settings": {
                "solver": solver,
                "execution_mode": "structured",
                "reltol": reltol,
                "abstol": abstol,
                "initial_dt_s": INITIAL_STEP_S,
                "save_step_s": SAVE_STEP_S,
                "switches": "exact segment stops, midpoint-frozen polarization symbols",
                "eom_time": "absolute global time preserved across segment boundaries",
            },
            "solver_stats": _stats_summary(stats),
        },
        times,
        densities,
        stats,
    )


def _point_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for solver in ("dopri5", "tsit5"):
        for tolerance in SWEEP_TOLERANCES:
            specs.append(
                {
                    "method": f"{solver}_scaled_{tolerance:.0e}",
                    "solver": solver,
                    "reltol": tolerance,
                    "abstol": tolerance / 100.0,
                    "kind": "coupled_tolerance_sweep",
                }
            )
        for tolerance in FIXED_ABS_CONTROLS:
            specs.append(
                {
                    "method": f"{solver}_fixed_atol_1e-10_rtol_{tolerance:.0e}",
                    "solver": solver,
                    "reltol": tolerance,
                    "abstol": 1e-10,
                    "kind": "fixed_absolute_tolerance_control",
                }
            )
    return specs


def _audit_existing_candidates(
    path: Path, config: driver.BenchmarkConfig, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Record whether older 1e-9 structured rows satisfy the strict reuse gate."""
    if not path.exists():
        return {"path": str(path), "found": False, "reused": False}
    saved = json.loads(path.read_text(encoding="utf-8"))
    old_provenance = saved.get("provenance", {})
    saved_args = saved.get("arguments", {})
    checks = {
        "complete": saved.get("status") == "complete" and saved.get("timings_accepted") is True,
        "nominal_config": saved_args.get("config") == "nominal",
        "duration_10us": saved_args.get("duration_us") == 10.0,
        "entry_137ns": saved_args.get("entry_ns") == 137.0,
        "constant_envelope": saved_args.get("envelope") == "constant",
        "model_matches": json.dumps(old_provenance.get("model"), sort_keys=True, default=str)
        == json.dumps(provenance.get("model"), sort_keys=True, default=str),
        "driver_hash_matches": old_provenance.get("driver_sha256") == provenance.get("driver_sha256"),
        "native_hash_matches": old_provenance.get("rust_extension_sha256")
        == provenance.get("rust_extension_sha256"),
        "batch_api_hash_matches": saved.get("batch_api_sha256")
        == sha256(driver.HERE.parent / "centrex_tlf/lindblad/batch.py"),
    }
    rows = {}
    for solver in ("dopri5", "tsit5"):
        method = f"structured_{solver}_1e-9"
        row = next(
            (candidate for candidate in saved.get("rows", []) if candidate.get("method") == method),
            None,
        )
        rows[solver] = {
            "found": row is not None,
            "reference_validated": bool(row and row.get("reference_validated")),
            "saved_solver_counters": bool(
                row
                and row.get("rhs_calls") is not None
                and row.get("accepted_steps") is not None
                and row.get("rejected_steps") is not None
            ),
        }
    checks["both_solver_rows_present"] = all(value["found"] for value in rows.values())
    checks["both_reference_validated"] = all(value["reference_validated"] for value in rows.values())
    checks["all_solver_counters_present"] = all(value["saved_solver_counters"] for value in rows.values())
    return {
        "path": str(path),
        "json_sha256": sha256(path),
        "found": True,
        "checks": checks,
        "rows": rows,
        "reuse_eligible": bool(all(checks.values())),
        "reused": False,
        "decision": "this sweep reruns the rows to collect the complete requested per-point counters and score them against its fresh current-native reference; failed compatibility/stat gates are listed above",
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    base_config = driver.CONFIGS["nominal"]
    config = replace(
        base_config,
        duration_s=10e-6,
        entry_phase_s=137e-9,
    )
    reference_json_path = args.output_dir / f"{args.tag}.json"
    densities_path = args.output_dir / f"{args.tag}_densities.npz"
    if args.resume_json is None and (reference_json_path.exists() or densities_path.exists()):
        raise FileExistsError(
            f"refusing to overwrite existing tolerance-sweep artifacts: {reference_json_path}, {densities_path}; use a new --tag or --resume-json"
        )
    progress: dict[str, Any] = {
        "status": "running",
        "complete": False,
        "timings_accepted": False,
        "arguments": vars(args),
        "config": asdict(config),
        "initial_label": INITIAL_LABEL,
        "resource_conditions_before": args.resource_conditions_before,
        "timing_boundary": "outer full-density plus separate solver-native photon integral, including segment loop and carry; output grids match exactly",
        "timing_boundary_breakdown": "full-density and photon solve-call wall times are secondary; outer propagation wall is the comparable total",
        "reference_status": "not_started",
        "rows": [],
        "resource_checks_between_points": [],
        "failed_points": [],
    }
    arrays: dict[str, np.ndarray] = {}
    runner_hash = sha256(Path(__file__).resolve())
    progress["runner_sha256"] = runner_hash

    with threadpool_limits(limits=1), comparison.PeakMemory() as memory:
        setup_started = time.perf_counter()
        case = driver.build_case(config, prepare_frozen=False, amplitude_envelope="constant")
        driver._prepare_segmented_plan(case)
        provenance = driver.provenance(case)
        rho0 = case.rho0(INITIAL_LABEL)
        span = (config.entry_phase_s, config.entry_phase_s + config.duration_s)
        times_probe = driver.switch_boundaries(*span)
        progress["common_model_preparation_seconds"] = time.perf_counter() - setup_started
        progress["provenance"] = provenance
        progress["driver_sha256"] = provenance.get("driver_sha256")
        progress["run_comparison_sha256"] = sha256(Path(comparison.__file__).resolve())
        progress["batch_api_sha256"] = sha256(driver.HERE.parent / "centrex_tlf/lindblad/batch.py")
        progress["native_extension_sha256"] = provenance.get("rust_extension_sha256")
        if args.resume_json is not None:
            saved = json.loads(args.resume_json.read_text(encoding="utf-8"))
            expected_metadata = {
                "config": asdict(config),
                "initial_label": INITIAL_LABEL,
                "runner_sha256": runner_hash,
                "driver_sha256": provenance.get("driver_sha256"),
                "run_comparison_sha256": progress["run_comparison_sha256"],
                "batch_api_sha256": progress["batch_api_sha256"],
                "native_extension_sha256": provenance.get("rust_extension_sha256"),
            }
            mismatches = [
                key
                for key, expected in expected_metadata.items()
                if json.dumps(saved.get(key), sort_keys=True, default=str)
                != json.dumps(expected, sort_keys=True, default=str)
            ]
            if mismatches:
                raise ValueError(f"cannot resume incompatible sweep artifact; mismatched fields: {mismatches}")
            if saved.get("reference_validated") is not True:
                raise ValueError("cannot resume because the stored short reference is not validated")
            stored_npz = args.resume_json.with_name(args.resume_json.stem + "_densities.npz")
            expected_npz_hash = saved.get("density_npz_sha256") or saved.get("reference_npz_sha256")
            actual_npz_hash = sha256(stored_npz)
            if not expected_npz_hash or actual_npz_hash != expected_npz_hash:
                raise ValueError(
                    "cannot resume because the stored NPZ checksum does not match its checkpoint "
                    f"(expected={expected_npz_hash}, actual={actual_npz_hash})"
                )
            with np.load(stored_npz) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            if not {"reference_times", "reference_density"}.issubset(arrays):
                raise ValueError("resume NPZ is missing the validated reference density/time arrays")
            progress = saved
            progress["status"] = "running"
            progress["complete"] = False
            progress["resumed_from"] = {
                "json_path": str(args.resume_json.resolve()),
                "json_sha256": sha256(args.resume_json),
                "npz_path": str(stored_npz.resolve()),
                "npz_sha256": sha256(stored_npz),
            }
            _write_npz(densities_path, arrays)
        else:
            _write_json(reference_json_path, progress)
        progress["old_reference_not_reused"] = {
            "path": str(args.old_reference_json),
            "reason": "saved native-extension and batch-API hashes differ from the current additive full_and_integral build; the short four-stage reference is regenerated under the current native extension",
        }
        progress["existing_1e-9_candidate_rows"] = _audit_existing_candidates(
            args.output_dir / "idle_nominal_10us_structured.json", config, provenance
        )
        if args.resume_json is None:
            progress["reference_status"] = "building_four_stage_current_native"
            _write_json(reference_json_path, progress)
        reference_progress: dict[str, Any] = {"stages": {}}

        def reference_checkpoint(stage: str, samples: list[float]) -> None:
            reference_progress["stages"][stage] = samples
            guard_started = time.perf_counter()
            snapshot = comparison.resource_snapshot()
            okay, reason = _resource_ok(snapshot)
            progress.setdefault("resource_checks_between_points", []).append(
                {
                    "before_point": f"reference_after_{stage}",
                    "snapshot": snapshot,
                    "guard_wall_seconds": time.perf_counter() - guard_started,
                    "okay": okay,
                    "stop_reason": reason,
                }
            )
            progress["reference_progress"] = reference_progress
            _write_json(reference_json_path, progress)
            if not okay:
                raise RuntimeError(f"resource guard stopped reference ladder after {stage}: {reason}")

        if args.resume_json is None:
            try:
                reference, reference_check = comparison.validated_reference(
                    case,
                    rho0,
                    span,
                    args.reference_tolerance,
                    execution_mode="structured",
                    progress_callback=reference_checkpoint,
                )
            except Exception as exc:
                progress["status"] = "reference_failed"
                progress["reference_error"] = f"{type(exc).__name__}: {exc}"
                progress["reference_progress"] = reference_progress
                _write_json(reference_json_path, progress)
                return progress
            reference_times, reference_density, reference_photons = reference[:3]
            reference_validated = bool(reference_check.get("validated"))
            for settings in reference_check.get("settings", {}).values():
                settings["abstol"] = float(settings["reltol"]) / 100.0
                settings["execution_mode"] = "structured"
                settings["output_boundary"] = "full density plus separate solver-native photon integral"
            progress["reference_check"] = reference_check
            progress["reference_validated"] = reference_validated
            progress["reference_status"] = "validated" if reference_validated else "failed_gates"
            progress["reference_photon_yield"] = float(reference_photons)
            progress["reference_sample_count"] = len(reference_times)
            progress["reference_switch_segment_count"] = int(len(times_probe) - 1)
            progress["reference_output_grid"] = "100 ns per segment plus every exact microwave switch edge and final endpoint"
            arrays["reference_times"] = np.asarray(reference_times)
            arrays["reference_density"] = np.asarray(reference_density)
            _write_npz(densities_path, arrays)
            progress["reference_npz_sha256"] = sha256(densities_path)
            _write_json(reference_json_path, progress)
            print(
                f"reference {progress['reference_status']}: "
                f"{reference_check['total_seconds']:.3f}s, "
                f"samples={len(reference_times)}, photons={reference_photons:.9g}",
                flush=True,
            )
        else:
            reference_times = arrays["reference_times"]
            reference_density = arrays["reference_density"]
            reference_check = progress["reference_check"]
            reference_photons = float(progress["reference_photon_yield"])
            reference_validated = bool(progress["reference_validated"])
        if not reference_validated:
            progress["status"] = "reference_failed"
            progress["complete"] = False
            progress["peak_process_rss_bytes"] = memory.peak
            _write_json(reference_json_path, progress)
            return progress

        for point_index, spec in enumerate(_point_specs()):
            guard_started = time.perf_counter()
            snapshot = comparison.resource_snapshot()
            okay, stop_reason = _resource_ok(snapshot)
            progress["resource_checks_between_points"].append(
                {
                    "before_point": spec["method"],
                    "snapshot": snapshot,
                    "guard_wall_seconds": time.perf_counter() - guard_started,
                    "okay": okay,
                    "stop_reason": stop_reason,
                }
            )
            if not okay:
                progress["status"] = "paused_resource_guard"
                progress["pause_reason"] = stop_reason
                break
            if any(row["method"] == spec["method"] for row in progress["rows"]):
                continue
            print(
                f"[{point_index + 1}/{len(_point_specs())}] {spec['method']} "
                f"abstol={spec['abstol']:.1e}",
                flush=True,
            )
            row: dict[str, Any] = {**spec, "repeat_count": 1, "runs": []}
            try:
                one, candidate_times, candidate_density, stats = _run_one(
                    case,
                    rho0,
                    span,
                    solver=spec["solver"],
                    reltol=spec["reltol"],
                    abstol=spec["abstol"],
                )
                photons = driver.photon_yield(stats)
                scores = compare_trajectory(
                    candidate_times,
                    candidate_density,
                    photons,
                    reference_times,
                    reference_density,
                    reference_photons,
                )
                one["photon_yield"] = float(photons)
                one["accuracy"] = scores
                row["runs"].append(one)
                row["wall_seconds"] = [one["outer_propagation_wall_seconds"]]
                row["median_seconds"] = one["outer_propagation_wall_seconds"]
                row["reference_validated"] = reference_validated
                row["passes"] = bool(scores["passes"] and reference_validated)
                arrays[spec["method"]] = np.asarray(candidate_density)
                arrays[spec["method"] + "_times"] = np.asarray(candidate_times)
            except Exception as exc:
                row["repeat_count"] = 0
                row["exception"] = f"{type(exc).__name__}: {exc}"
                row["passes"] = False
                progress["failed_points"].append(spec["method"])
            progress["rows"].append(row)
            if row.get("runs") and row["runs"][0]["outer_propagation_wall_seconds"] > args.stop_after_point_seconds:
                row["post_point_stop_threshold_seconds"] = args.stop_after_point_seconds
                row["stop_threshold_type"] = "checked after a point completes; it is not an active timeout"
                progress["status"] = "stopped_runaway_point"
                progress["stop_reason"] = (
                    f"{spec['method']} exceeded {args.stop_after_point_seconds:.0f} seconds; "
                    "the completed point is retained and later points were not launched"
                )
            _write_npz(densities_path, arrays)
            progress["density_npz_sha256"] = sha256(densities_path)
            progress["last_completed_point"] = spec["method"]
            progress["peak_process_rss_bytes"] = memory.peak
            _write_json(reference_json_path, progress)
            if row.get("runs"):
                score = row["runs"][0]["accuracy"]
                all_saved = score.get("all_saved") or {}
                print(
                    f"completed {spec['method']}: {row['median_seconds']:.3f}s; "
                    f"density={all_saved.get('density_max_abs', float('nan')):.3g}, "
                    f"pop={all_saved.get('population_max_abs', float('nan')):.3g}, "
                    f"pop_rel={all_saved.get('population_max_relative_above_1e-6', float('nan')):.3g}, "
                    f"photon={all_saved.get('photon_absolute_error', float('nan')):.3g}, "
                    f"accepted={row['runs'][0]['solver_stats']['full_density_accepted_steps']}, "
                    f"rhs={row['runs'][0]['solver_stats']['full_density_rhs_calls']}, "
                    f"passes={row['passes']}",
                    flush=True,
                )
            else:
                print(f"failed {spec['method']}: {row.get('exception')}", flush=True)
            if progress.get("status") == "stopped_runaway_point":
                break

        progress["peak_process_rss_bytes"] = memory.peak
        guard_started = time.perf_counter()
        snapshot = comparison.resource_snapshot()
        okay, reason = _resource_ok(snapshot)
        progress["resource_checks_between_points"].append(
            {
                "after_point": "final_postrun_guard",
                "snapshot": snapshot,
                "guard_wall_seconds": time.perf_counter() - guard_started,
                "okay": okay,
                "stop_reason": reason,
            }
        )
        if progress.get("status") == "running":
            progress["status"] = (
                "complete"
                if okay and len(progress["rows"]) == len(_point_specs())
                else ("contended_after_run" if not okay else "incomplete")
            )
        progress["complete"] = progress["status"] == "complete"
        progress["timings_accepted"] = bool(
            all(check["okay"] for check in progress["resource_checks_between_points"])
            and progress["status"] == "complete"
        )
        progress["all_candidate_points_pass"] = bool(
            progress["rows"] and all(row.get("passes", False) for row in progress["rows"])
        )
        progress["density_npz_sha256"] = sha256(densities_path)
        _write_json(reference_json_path, progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=driver.RESULTS_DIR)
    parser.add_argument("--tag", default="idle_nominal_10us_tolerance_sweep")
    parser.add_argument(
        "--old-reference-json",
        type=Path,
        default=driver.RESULTS_DIR / "idle_nominal_10us_entry137.json",
    )
    parser.add_argument("--reference-tolerance", type=float, default=3e-11)
    parser.add_argument(
        "--stop-after-point-seconds", type=float, default=POST_POINT_STOP_SECONDS,
        help="after a point completes, stop launching later points if its wall time exceeded this threshold",
    )
    parser.add_argument("--resume-json", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.reference_tolerance) or not 0 < args.reference_tolerance < 1e-10:
        parser.error("reference tolerance must be finite and less than 1e-10")
    if not np.isfinite(args.stop_after_point_seconds) or args.stop_after_point_seconds <= 0:
        parser.error("post-point stop threshold must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = comparison.resource_snapshot()
    okay, reason = _resource_ok(before)
    if not okay:
        raise RuntimeError(f"machine busy before tolerance sweep: {reason}; {before}")
    args.resource_conditions_before = before
    payload = run_sweep(args)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "complete": payload.get("complete"),
                "timings_accepted": payload.get("timings_accepted"),
                "reference_status": payload.get("reference_status"),
                "row_count": len(payload.get("rows", [])),
                "failed_points": payload.get("failed_points", []),
                "output_json": str(args.output_dir / f"{args.tag}.json"),
                "output_npz": str(args.output_dir / f"{args.tag}_densities.npz"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
