"""Repeat three selected points from a completed short RC tolerance sweep."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import bench_rc_propagation as driver
import numpy as np
import run_rc_comparison as comparison
import run_rc_tolerance_sweep as sweep
from threadpoolctl import threadpool_limits

REPEAT_METHODS = (
    "dopri5_scaled_1e-08",
    "tsit5_scaled_1e-08",
    "tsit5_fixed_atol_1e-10_rtol_1e-07",
)
ADDITIONAL_REPEATS = 2


def _load_and_validate_sweep(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not payload.get("timings_accepted"):
        raise ValueError("source sweep must be complete and timing-accepted")
    if payload.get("reference_validated") is not True:
        raise ValueError("source sweep does not contain a validated short reference")
    arrays_path = path.with_name(path.stem + "_densities.npz")
    expected_hash = payload.get("density_npz_sha256")
    actual_hash = sweep.sha256(arrays_path)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"source sweep NPZ checksum mismatch: expected={expected_hash}, actual={actual_hash}"
        )
    with np.load(arrays_path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if not {"reference_times", "reference_density"}.issubset(arrays):
        raise ValueError("source sweep NPZ lacks validated reference arrays")
    if (
        arrays["reference_times"].ndim != 1
        or arrays["reference_density"].shape
        != (arrays["reference_times"].size, 70, 70)
        or not np.all(np.isfinite(arrays["reference_times"]))
        or not np.all(np.isfinite(arrays["reference_density"]))
    ):
        raise ValueError("source sweep reference arrays have invalid shape or nonfinite values")
    for method in REPEAT_METHODS:
        if method not in arrays or method + "_times" not in arrays:
            raise ValueError(f"source sweep lacks required candidate arrays: {method}")
    return payload, arrays


def _guard(label: str, checks: list[dict[str, Any]]) -> bool:
    started = time.perf_counter()
    snapshot = comparison.resource_snapshot()
    okay, reason = sweep._resource_ok(snapshot)
    checks.append(
        {
            "before": label,
            "snapshot": snapshot,
            "wall_seconds": time.perf_counter() - started,
            "okay": okay,
            "stop_reason": reason,
        }
    )
    return okay


def _canonical_json(value: Any) -> str:
    """Compare provenance using the same JSON normalization as saved artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def run(args: argparse.Namespace, resources_before: dict[str, Any]) -> dict[str, Any]:
    if args.output_json.exists() or args.output_npz.exists():
        raise FileExistsError(
            f"refusing to overwrite repeat artifacts: {args.output_json}, {args.output_npz}"
        )
    source, source_arrays = _load_and_validate_sweep(args.source_json)
    base_config = driver.CONFIGS["nominal"]
    config = replace(base_config, duration_s=10e-6, entry_phase_s=137e-9)
    if source.get("config") != asdict(config) or source.get("initial_label") != sweep.INITIAL_LABEL:
        raise ValueError("source sweep physics settings do not match the selected repeat case")
    reference_times = source_arrays["reference_times"]
    reference_density = source_arrays["reference_density"]
    reference_photons = float(source["reference_photon_yield"])

    output: dict[str, Any] = {
        "status": "running",
        "complete": False,
        "timings_accepted": False,
        "source_sweep_json": str(args.source_json.resolve()),
        "source_sweep_json_sha256": sweep.sha256(args.source_json),
        "source_sweep_npz_sha256": sweep.sha256(
            args.source_json.with_name(args.source_json.stem + "_densities.npz")
        ),
        "config": asdict(config),
        "initial_label": sweep.INITIAL_LABEL,
        "additional_repeats_per_method": ADDITIONAL_REPEATS,
        "timing_boundary": "outer full-density plus separate solver-native photon-integral propagation, same exact-segment and common-grid contract as source sweep",
        "resource_conditions_before": resources_before,
        "rows": [],
        "resource_checks_between_runs": [],
    }
    arrays: dict[str, np.ndarray] = {
        "reference_times": reference_times,
        "reference_density": reference_density,
    }
    with threadpool_limits(limits=1), comparison.PeakMemory() as memory:
        setup_started = time.perf_counter()
        case = driver.build_case(config, prepare_frozen=False, amplitude_envelope="constant")
        driver._prepare_segmented_plan(case)
        provenance = driver.provenance(case)
        output["common_model_preparation_seconds"] = time.perf_counter() - setup_started
        output["provenance"] = provenance
        checks = {
            "driver_hash": provenance.get("driver_sha256") == source.get("provenance", {}).get("driver_sha256"),
            "native_hash": provenance.get("rust_extension_sha256") == source.get("native_extension_sha256"),
            "model": _canonical_json(provenance.get("model"))
            == _canonical_json(source.get("provenance", {}).get("model")),
            "batch_api_hash": sweep.sha256(driver.HERE.parent / "centrex_tlf/lindblad/batch.py")
            == source.get("batch_api_sha256"),
        }
        output["source_compatibility_checks"] = checks
        if not all(checks.values()):
            raise ValueError(f"source sweep and current repeat environment differ: {checks}")
        _write_json(args.output_json, output)
        rho0 = case.rho0(sweep.INITIAL_LABEL)
        span = (config.entry_phase_s, config.entry_phase_s + config.duration_s)
        source_rows = {row["method"]: row for row in source["rows"]}
        specs: list[dict[str, Any]] = []
        for method in REPEAT_METHODS:
            row = source_rows.get(method)
            if row is None or not row.get("passes"):
                raise ValueError(f"repeat method must exist and pass in source sweep: {method}")
            specs.append(
                {
                    "method": method,
                    "solver": row["solver"],
                    "reltol": float(row["reltol"]),
                    "abstol": float(row["abstol"]),
                }
            )

        for spec in specs:
            baseline = source_rows[spec["method"]]["runs"][0]
            candidate_runs = [baseline]
            for repeat_index in range(1, ADDITIONAL_REPEATS + 1):
                if not _guard(f"{spec['method']}_repeat_{repeat_index}", output["resource_checks_between_runs"]):
                    output["status"] = "paused_resource_guard"
                    output["pause_reason"] = output["resource_checks_between_runs"][-1]["stop_reason"]
                    break
                print(
                    f"repeat {repeat_index}/{ADDITIONAL_REPEATS}: {spec['method']}",
                    flush=True,
                )
                try:
                    one, times, density, stats = sweep._run_one(
                        case,
                        rho0,
                        span,
                        solver=spec["solver"],
                        reltol=spec["reltol"],
                        abstol=spec["abstol"],
                    )
                    photons = driver.photon_yield(stats)
                    accuracy = sweep.compare_trajectory(
                        times,
                        density,
                        photons,
                        reference_times,
                        reference_density,
                        reference_photons,
                    )
                    one["photon_yield"] = float(photons)
                    one["accuracy"] = accuracy
                    one["repeat_index"] = repeat_index
                    candidate_runs.append(one)
                    arrays[f"{spec['method']}_repeat_{repeat_index}"] = density
                    arrays[f"{spec['method']}_repeat_{repeat_index}_times"] = times
                except Exception as exc:
                    candidate_runs.append(
                        {
                            "repeat_index": repeat_index,
                            "exception": f"{type(exc).__name__}: {exc}",
                            "passes": False,
                        }
                    )
                arrays_path = args.output_npz.with_name(args.output_npz.stem + ".tmp.npz")
                with arrays_path.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                arrays_path.replace(args.output_npz)
                output["last_completed"] = f"{spec['method']}_repeat_{repeat_index}"
                output["npz_sha256"] = sweep.sha256(args.output_npz)
                output["peak_process_rss_bytes"] = memory.peak
                _write_json(args.output_json, output)
                latest = candidate_runs[-1]
                if "accuracy" in latest:
                    score = latest["accuracy"]["all_saved"] or {}
                    print(
                        f"completed {spec['method']} repeat {repeat_index}: "
                        f"{latest['outer_propagation_wall_seconds']:.3f}s, "
                        f"density={score.get('density_max_abs', float('nan')):.3g}, "
                        f"relative_population={score.get('population_max_relative_above_1e-6', float('nan')):.3g}, "
                        f"passes={latest['accuracy']['passes']}",
                        flush=True,
                    )
            method_row = {
                **spec,
                "baseline_and_repeat_runs": candidate_runs,
                "wall_seconds": [
                    row["outer_propagation_wall_seconds"]
                    for row in candidate_runs
                    if row.get("outer_propagation_wall_seconds") is not None
                ],
                "median_seconds": float(
                    np.median(
                        [
                            row["outer_propagation_wall_seconds"]
                            for row in candidate_runs
                            if row.get("outer_propagation_wall_seconds") is not None
                        ]
                    )
                ),
                "range_seconds": [
                    float(
                        np.min(
                            [row["outer_propagation_wall_seconds"] for row in candidate_runs if row.get("outer_propagation_wall_seconds") is not None]
                        )
                    ),
                    float(
                        np.max(
                            [row["outer_propagation_wall_seconds"] for row in candidate_runs if row.get("outer_propagation_wall_seconds") is not None]
                        )
                    ),
                ],
                "all_accuracy_runs_pass": all(
                    row.get("accuracy", {}).get("passes", False) for row in candidate_runs
                ),
            }
            output["rows"].append(method_row)
            _write_json(args.output_json, output)
            if output.get("status") == "paused_resource_guard":
                break

        final_snapshot = comparison.resource_snapshot()
        okay, reason = sweep._resource_ok(final_snapshot)
        output["resource_checks_between_runs"].append(
            {"after": "final_postrun_guard", "snapshot": final_snapshot, "okay": okay, "stop_reason": reason}
        )
        all_methods_present = {row["method"] for row in output["rows"]} == set(REPEAT_METHODS)
        if output.get("status") == "running":
            output["status"] = "complete" if okay and all_methods_present else (
                "contended_after_run" if not okay else "incomplete"
            )
        output["complete"] = output["status"] == "complete"
        output["timings_accepted"] = bool(
            output["complete"]
            and all(item["okay"] for item in output["resource_checks_between_runs"])
        )
        output["peak_process_rss_bytes"] = memory.peak
        output["npz_sha256"] = sweep.sha256(args.output_npz) if args.output_npz.exists() else None
        _write_json(args.output_json, output)
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.stem + ".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-json",
        type=Path,
        default=driver.RESULTS_DIR / "idle_nominal_10us_tolerance_sweep_retry2.json",
    )
    parser.add_argument("--output-json", type=Path, default=driver.RESULTS_DIR / "idle_nominal_10us_tolerance_repeats_retry2.json")
    parser.add_argument("--output-npz", type=Path, default=driver.RESULTS_DIR / "idle_nominal_10us_tolerance_repeats_retry2_densities.npz")
    args = parser.parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    resources = comparison.resource_snapshot()
    okay, reason = sweep._resource_ok(resources)
    if not okay:
        raise RuntimeError(f"machine busy before tolerance repeats: {reason}; {resources}")
    output = run(args, resources)
    print(
        json.dumps(
            {
                "status": output.get("status"),
                "complete": output.get("complete"),
                "timings_accepted": output.get("timings_accepted"),
                "methods": [row.get("method") for row in output.get("rows", [])],
                "output_json": str(args.output_json),
                "output_npz": str(args.output_npz),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
