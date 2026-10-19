"""Compare structured_upper and structured execution on the validated RC case.

The runner is intentionally split: --stage profile is an RHS-only diagnostic;
--stage trajectory performs the physical 10 us matched comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py
import run_rc_comparison as comparison
import run_rc_tolerance_sweep as sweep
from threadpoolctl import threadpool_limits

import bench_rc_propagation as driver
from bench_rc_propagation import BenchmarkConfig, build_case, propagate_segmented, provenance
from rc_rhs_profile import profile_rc_rhs

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rc_propagation_results"
REFERENCE_JSON = RESULTS / "idle_nominal_10us_tolerance_sweep_retry2.json"
REFERENCE_NPZ = RESULTS / "idle_nominal_10us_tolerance_sweep_retry2_densities.npz"
REFERENCE_NATIVE_SHA256 = "309a79ac0a7a067dbce0026f3ef6d903cbfccfe33e6332881ccf1a6a5e2283d8"
LABEL = "X_J2_F1_5o2_F2_mF0"
MODES = ("structured_upper", "structured")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference():
    metadata = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    if not metadata.get("complete") or not metadata.get("reference_validated"):
        raise RuntimeError("reference artifact is not marked complete and validated")
    source = metadata.get("provenance", {})
    native_hash = source.get("rust_extension_sha256")
    if not native_hash or not native_hash.startswith(REFERENCE_NATIVE_SHA256):
        raise RuntimeError(f"reference native hash mismatch: {native_hash}")
    if sha256(REFERENCE_NPZ) != metadata.get("density_npz_sha256"):
        raise RuntimeError("reference density NPZ checksum does not match its recorded checksum")
    source_hash = metadata.get("runner_sha256")
    source_path = HERE / "run_rc_tolerance_sweep.py"
    if source_hash and sha256(source_path) != source_hash:
        raise RuntimeError("reference runner source checksum does not match the recorded checksum")
    for key, path in (
        ("run_comparison_sha256", HERE / "run_rc_comparison.py"),
        ("batch_api_sha256", HERE.parent / "centrex_tlf/lindblad/batch.py"),
    ):
        if sha256(path) != metadata.get(key):
            raise RuntimeError(f"current {key} differs from validated reference")
    if native_hash != REFERENCE_NATIVE_SHA256:
        raise RuntimeError("current native extension hash differs from validated reference")
    expected_config = {"name": "nominal", "optical_rabi_over_gamma": 1.0,
                       "microwave_rabi_over_gamma": 1.0, "detuning_mhz": 0.0,
                       "duration_s": 10e-6, "entry_phase_s": 137e-9}
    if metadata.get("config") != expected_config or metadata.get("initial_label") != LABEL:
        raise RuntimeError("reference model/configuration/initial label differs")
    with np.load(REFERENCE_NPZ) as artifact:
        times = artifact["reference_times"].copy()
        density = artifact["reference_density"].copy()
    return metadata, times, density


def validate_case_provenance(case, metadata) -> dict:
    current = provenance(case)
    saved = metadata["provenance"]
    for key in ("rust_extension_sha256", "driver_sha256"):
        if current.get(key) != saved.get(key):
            raise RuntimeError(f"current case {key} differs from validated reference")
    canonical = lambda value: json.dumps(value, sort_keys=True, default=str)
    for key in ("model", "config", "rates"):
        if canonical(current.get(key)) != canonical(saved.get(key)):
            raise RuntimeError(f"current case {key} differs from validated reference")
    return current


def build_case_137ns():
    config = BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 10e-6, 137e-9)
    return build_case(config, prepare_frozen=False, amplitude_envelope="constant")


def run_profile():
    metadata, times, reference = load_reference()
    before = comparison.resource_snapshot()
    okay, reason = sweep._resource_ok(before)
    if not okay:
        raise RuntimeError(f"RHS profile resource guard failed: {reason}")
    with threadpool_limits(limits=1):
        case = build_case(BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 10e-6, 137e-9),
                          prepare_frozen=False, amplitude_envelope="constant")
        driver._prepare_segmented_plan(case)
        case_provenance = validate_case_provenance(case, metadata)
        rho0 = case.rho0(LABEL)
        result = {
            "initial_state": profile_rc_rhs(
                case, rho0, [137e-9, 637e-9, float(times[len(times) // 2])],
                modes=("structured", "structured_upper"), repeats=1000,
            ),
            "coherent_reference_state": profile_rc_rhs(
                case, reference[len(times) // 2], [float(times[len(times) // 2])],
                modes=("structured", "structured_upper"), repeats=1000,
            ),
            "artifact": "structured_upper_rhs_profile",
            "resource_conditions_before": before,
            "provenance": case_provenance,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "batch_api_sha256": sha256(HERE.parent / "centrex_tlf/lindblad/batch.py"),
            "comparison_source_sha256": sha256(HERE / "run_rc_comparison.py"),
        }
    after = comparison.resource_snapshot()
    okay, reason = sweep._resource_ok(after)
    result["resource_conditions_after"] = after
    result["resource_guard_after"] = {"okay": okay, "reason": reason}
    if not okay:
        raise RuntimeError(f"RHS profile post-run resource guard failed: {reason}")
    result["reference_sha256"] = sha256(REFERENCE_JSON)
    result["reference_npz_sha256"] = sha256(REFERENCE_NPZ)
    result["reference_max_density_norm"] = float(np.max(np.abs(reference)))
    return result


def accuracy(candidate: np.ndarray, reference: np.ndarray) -> dict:
    diff = np.abs(candidate - reference)
    candidate_pop = np.real(np.diagonal(candidate, axis1=-2, axis2=-1))
    reference_pop = np.real(np.diagonal(reference, axis1=-2, axis2=-1))
    mask = reference_pop >= 1e-6
    return {
        "max_abs_density_all_times": float(diff.max()),
        "max_abs_density_endpoint": float(diff[-1].max()),
        "max_abs_population_all_times": float(np.abs(candidate_pop-reference_pop).max()),
        "max_rel_population_ref_ge_1e-6": float(
            (np.abs(candidate_pop-reference_pop)[mask] / reference_pop[mask]).max()
        ),
        "trace_max_abs_error": float(np.max(np.abs(np.trace(candidate, axis1=-2, axis2=-1)-1))),
        "hermiticity_max_abs": float(np.max(np.abs(candidate-candidate.conj().transpose(0,2,1)))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh((candidate+candidate.conj().transpose(0,2,1))/2).min()),
    }


def run_trajectories(repeats: int = 3, progress_path: Path | None = None):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    metadata, reference_times, reference = load_reference()
    before = comparison.resource_snapshot()
    okay, reason = sweep._resource_ok(before)
    if not okay:
        raise RuntimeError(f"trajectory resource guard failed: {reason}")
    setup_started = time.perf_counter()
    case = build_case(BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 10e-6, 137e-9),
                      prepare_frozen=False, amplitude_envelope="constant")
    driver._prepare_segmented_plan(case)
    case_provenance = validate_case_provenance(case, metadata)
    setup_seconds = time.perf_counter() - setup_started
    rho0 = case.rho0(LABEL)
    tspan = (137e-9, 137e-9 + 10e-6)
    # Alternate order to reduce systematic thermal/order bias.
    order = [list(MODES), list(reversed(MODES)), list(MODES)]
    rows = []
    saved = {}
    reference_photons = float(metadata["reference_photon_yield"])
    with threadpool_limits(limits=1):
      packed = np.ascontiguousarray(case.prepared_dynamic.layout.pack(rho0), dtype=np.float64)
      for mode in MODES:
          warmup = create_lindblad_rhs_evaluator_py(case.prepared_dynamic.rust_plan, mode)
          warmup.rhs_packed_py(packed, float(tspan[0]))
      for repeat in range(repeats):
        for mode in order[repeat % len(order)]:
            guard_before = comparison.resource_snapshot()
            okay, reason = sweep._resource_ok(guard_before)
            if not okay:
                raise RuntimeError(f"resource guard before trial failed: {reason}")
            started = time.perf_counter()
            times, densities, stats = propagate_segmented(
                case, rho0, tspan, solver="tsit5", reltol=1e-8, abstol=1e-10,
                dt=1e-10, execution_mode=mode, save_step_s=100e-9,
            )
            elapsed = time.perf_counter() - started
            photons = float(sum(float(row["photon_integral"]) for row in stats))
            scored = sweep.compare_trajectory(
                times, densities, photons, reference_times, reference, reference_photons
            )
            row = {
                "repeat": repeat + 1, "mode": mode, "solver": "tsit5",
                "reltol": 1e-8, "abstol": 1e-10, "dt": 1e-10,
                "wall_seconds_outer": elapsed,
                "full_density_wall_seconds": float(sum(row["full_density_wall_seconds"] for row in stats)),
                "photon_integral_wall_seconds": float(sum(row["photon_integral_wall_seconds"] for row in stats)),
                "full_density_accepted_steps": sweep._numeric_sum(stats, "accepted_steps"),
                "full_density_rejected_steps": sweep._numeric_sum(stats, "rejected_steps"),
                "full_density_rhs_calls": sweep._numeric_sum(stats, "rhs_calls"),
                "stats_scope": "full-density solver only; separate photon solver has no counters",
                "photon_yield": photons,
                "accuracy_vs_reference": scored,
                "resource_conditions_before": guard_before,
            }
            rows.append(row)
            if repeat == 0:
                saved[mode] = densities.copy()
            saved[f"trial_{repeat+1}_{mode}"] = densities.copy()
            arrays = {"reference_times": reference_times}
            for name, matrix in saved.items():
                arrays[name] = matrix
                arrays[name + "_times"] = times
            np.savez_compressed(
                RESULTS / "idle_nominal_10us_structured_upper_comparison_densities.npz",
                **arrays,
            )
            guard_after = comparison.resource_snapshot()
            okay, reason = sweep._resource_ok(guard_after)
            row["resource_conditions_after"] = guard_after
            row["resource_guard_after"] = {"okay": okay, "reason": reason}
            checkpoint = {
                "status": "running", "rows": rows,
                "resource_conditions_before": before,
                "resource_check_after_trial": {"snapshot": guard_after, "okay": okay, "reason": reason},
                "common_model_preparation_seconds": setup_seconds,
            }
            if progress_path:
                progress_path.write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
            if not okay:
                raise RuntimeError(f"resource guard after trial failed: {reason}")
    mode_comparison = None
    if all(mode in saved for mode in MODES):
        first_rows = {row["mode"]: row for row in rows if row["repeat"] == 1}
        mode_comparison = comparison.accuracy(
            saved["structured_upper"], saved["structured"],
            first_rows["structured_upper"]["photon_yield"],
            first_rows["structured"]["photon_yield"],
        )
    after = comparison.resource_snapshot()
    okay, reason = sweep._resource_ok(after)
    if not okay:
        raise RuntimeError(f"trajectory post-run resource guard failed: {reason}")
    all_trials_accuracy_pass = all(
        row["accuracy_vs_reference"].get("passes", False) for row in rows
    )
    return {
        "complete": all_trials_accuracy_pass,
        "status": "complete" if all_trials_accuracy_pass else "failed_accuracy",
        "repeats": repeats,
        "config": {"name":"nominal", "duration_s":10e-6, "entry_phase_s":137e-9,
                   "optical_rabi_over_gamma":1.0, "microwave_rabi_over_gamma":1.0,
                   "detuning_mhz":0.0, "initial_label":LABEL},
        "timing_boundary": "full density plus separate native photon solve; includes all switch segments and density carry",
        "rows": rows, "provenance": case_provenance,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "all_trials_accuracy_pass": all_trials_accuracy_pass,
        "resource_conditions_before": before,
        "resource_conditions_after": after,
        "resource_guard_after": {"okay": okay, "reason": reason},
        "mode_to_mode_accuracy": mode_comparison,
        "common_model_preparation_seconds": setup_seconds,
        "reference_json_sha256": sha256(REFERENCE_JSON),
        "reference_npz_sha256": sha256(REFERENCE_NPZ),
        "reference_native_sha256": metadata["provenance"]["rust_extension_sha256"],
        "saved_times": reference_times,
    }, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("profile", "trajectory"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    target = RESULTS / (
        "idle_nominal_10us_structured_upper_profile.json"
        if args.stage == "profile"
        else "idle_nominal_10us_structured_upper_comparison.json"
    )
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {target}")
    density_path = RESULTS / "idle_nominal_10us_structured_upper_comparison_densities.npz"
    progress_path = RESULTS / "idle_nominal_10us_structured_upper_comparison.progress.json"
    if args.stage == "trajectory" and (density_path.exists() or progress_path.exists()):
        raise FileExistsError("refusing to overwrite existing density or progress artifact")
    if args.stage == "profile":
        payload = run_profile()
        out = target
    else:
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        payload, densities = run_trajectories(args.repeats, progress_path)
        out = target
        npz_path = density_path
        final_arrays = {"reference_times": payload["saved_times"]}
        for name, density in densities.items():
            final_arrays[name] = density
            final_arrays[name + "_times"] = payload["saved_times"]
        np.savez_compressed(npz_path, **final_arrays)
        payload["density_npz"] = str(npz_path)
        payload["density_npz_sha256"] = sha256(npz_path)
    out.write_text(json.dumps(payload, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
