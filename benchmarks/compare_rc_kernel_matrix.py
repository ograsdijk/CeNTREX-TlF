"""Per-case RC comparison for sparse commutator kernel variants.

This runner is separate from historical benchmark sources and artifacts. It
profiles every mode per case, gives every mode one matched physical trajectory,
then adds two paired repeats for modes within the per-case profile margin.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import bench_rc_propagation as driver
import rc_propagation_methods as methods
import run_rc_comparison as comparison
import run_rc_tolerance_sweep as sweep
from bench_rc_propagation import BenchmarkConfig
from rc_rhs_profile import profile_rc_rhs

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rc_propagation_results"
TAG_DEFAULT = "rc_kernel_matrix_v1"
LABEL = "X_J2_F1_5o2_F2_mF0"
GAMMA = driver.GAMMA
PROFILE_MODES = (
    "structured",
    "structured_upper",
    "expanded_sparse",
    "experimental_structured_sparse_dense",
    "experimental_structured_sparse_dense_diag",
    "experimental_structured_sparse_dense_soa",
    "experimental_structured_sparse_dense_single_unpack",
)
MODE_NAMES = (*PROFILE_MODES, "phase_gauge")
COMPETITIVE_MARGIN = 1.20
REFERENCE_TOLERANCE = 3e-11
CPU_CUTOFF_PERCENT = 50.0
CASE_BUDGET_SECONDS = 600.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.stem + ".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def resource_ok(snapshot: dict, cutoff_percent: float = CPU_CUTOFF_PERCENT):
    if snapshot.get("known_simulation_pids"):
        return False, "known simulation process detected"
    samples = np.asarray(snapshot.get("cpu_percent_samples", []), dtype=float)
    if samples.size == 0 or float(np.median(samples)) > cutoff_percent:
        return False, f"median background CPU exceeded {cutoff_percent:g}% cutoff"
    return True, None


def take_resource_snapshot(cutoff_percent: float = CPU_CUTOFF_PERCENT):
    snapshot = comparison.resource_snapshot()
    okay, reason = resource_ok(snapshot, cutoff_percent)
    return snapshot, {"okay": okay, "reason": reason}


def case_matrix() -> list[BenchmarkConfig]:
    cases = [BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 10e-6, 137e-9)]
    for detuning in (-100.0, -20.0, -5.0, 5.0, 20.0, 100.0):
        cases.append(
            BenchmarkConfig(f"detuning_{detuning:+g}MHz", 1.0, 1.0, detuning, 10e-6, 137e-9)
        )
    for ratio in (0.3, 3.0):
        cases.append(
            BenchmarkConfig(f"optical_{ratio:g}Gamma", ratio, 1.0, 0.0, 10e-6, 137e-9)
        )
    for ratio in (0.3, 3.0):
        cases.append(
            BenchmarkConfig(f"microwave_{ratio:g}Gamma", 1.0, ratio, 0.0, 10e-6, 137e-9)
        )
    return cases


def make_phase_case(case):
    prepared, phase = methods.phase_frame_problem(case.prepared_dynamic)
    frame_case = copy.copy(case)
    frame_case.prepared_dynamic = prepared
    frame_case.prepared_segmented = None
    driver._prepare_segmented_plan(frame_case)
    return frame_case, phase


def make_probe(case, phase_case=None, phase=None, profile_repeats=1000):
    rng = np.random.default_rng(20261010)
    raw = rng.normal(size=(case.n_states, case.n_states)) + 1j * rng.normal(
        size=(case.n_states, case.n_states)
    )
    rho = raw + raw.conj().T
    rho /= np.trace(rho).real
    times = [case.config.entry_phase_s, case.config.entry_phase_s + 0.5e-6,
             case.config.entry_phase_s + case.config.duration_s / 2.0]
    initial = case.rho0(LABEL)
    direct_profile = profile_rc_rhs(
        case, initial, times, modes=PROFILE_MODES, repeats=profile_repeats
    )
    random_profile = profile_rc_rhs(
        case, rho, times[:1], modes=PROFILE_MODES, repeats=3 * profile_repeats
    )
    phase_profile = None
    if phase_case is not None and phase is not None:
        phase_initial = methods.phase_frame_density(initial, times[0], phase)
        phase_profile = profile_rc_rhs(
            phase_case, phase_initial, times, modes=("structured",), repeats=profile_repeats
        )
    return direct_profile, random_profile, phase_profile


def _stage_call(case, rho0, span, solver, reltol, dt_s):
    return comparison.ode_result(
        case, rho0, span, solver=solver, reltol=reltol,
        execution_mode="structured", dt_s=dt_s,
    )


def build_reference(case, rho0, span, case_dir: Path, cutoff_percent: float):
    json_path = case_dir / "reference.json"
    npz_path = case_dir / "reference_densities.npz"
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"refusing to overwrite {json_path} or {npz_path}")
    before, before_guard = take_resource_snapshot(cutoff_percent)
    if not before_guard["okay"]:
        raise RuntimeError(f"reference pre-run guard failed: {before_guard['reason']}")
    settings = (
        ("tight_dopri5", "dopri5", 1e-10, 1e-10),
        ("tighter_dopri5", "dopri5", REFERENCE_TOLERANCE, 5e-11),
        ("tight_tsit5_crosscheck", "tsit5", REFERENCE_TOLERANCE, 5e-11),
    )
    results, timings, stages = {}, {}, {}
    for name, solver, reltol, dt_s in settings:
        started = time.perf_counter()
        results[name] = _stage_call(case, rho0, span, solver, reltol, dt_s)
        timings[name] = time.perf_counter() - started
        snapshot, guard = take_resource_snapshot(cutoff_percent)
        stages[name] = {
            "solver": solver, "reltol": reltol, "abstol": reltol / 100.0,
            "dt_s": dt_s, "wall_seconds": timings[name],
            "resource_snapshot_after": snapshot, "resource_guard_after": guard,
        }
        checkpoint = {
            "status": "building_reference", "stages": stages,
            "resource_conditions_before": before,
        }
        write_json(case_dir / "reference.progress.json", checkpoint)
        if not guard["okay"]:
            raise RuntimeError(f"reference guard failed after {name}: {guard['reason']}")

    comparison_1 = comparison._reference_comparison(
        results["tight_dopri5"], results["tighter_dopri5"],
        max_density=1e-8, max_photons=1e-7, max_relative_population=1e-4,
    )
    comparison_2 = comparison._reference_comparison(
        results["tighter_dopri5"], results["tight_tsit5_crosscheck"],
        max_density=1e-8, max_photons=1e-7, max_relative_population=1e-4,
    )
    tight = results["tighter_dopri5"]
    physical = comparison.accuracy(tight[1], tight[1], tight[2], tight[2])
    validation = {
        "tight_dopri5_to_tighter_dopri5": comparison_1,
        "tight_tsit5_crosscheck": comparison_2,
        "tight_reference_physical": physical,
        "validated": bool(
            comparison_1["passes_reference_tolerance"]
            and comparison_2["passes_reference_tolerance"]
            and physical["passes"]
        ),
    }
    if not validation["validated"]:
        raise RuntimeError(f"reference gates failed for {case.config.name}: {validation}")
    arrays = {"reference_times": tight[0], "reference_density": tight[1]}
    tmp_npz = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    tmp_npz.replace(npz_path)
    after, after_guard = take_resource_snapshot(cutoff_percent)
    if not after_guard["okay"]:
        raise RuntimeError(f"reference post-run guard failed: {after_guard['reason']}")
    case_provenance = driver.provenance(case)
    payload = {
        "status": "validated", "complete": True,
        "case": case.config.name, "config": case_provenance["config"],
        "initial_label": LABEL, "reference_validation": validation,
        "reference_photon_yield": float(tight[2]),
        "reference_sample_count": int(len(tight[0])),
        "stages": stages, "stage_wall_seconds": timings,
        "provenance": case_provenance,
        "native_extension_sha256": case_provenance["rust_extension_sha256"],
        "reference_npz_sha256": sha256(npz_path),
        "resource_conditions_before": before,
        "resource_conditions_after": after,
        "resource_guard_after": after_guard,
        "cpu_cutoff_percent": cutoff_percent,
        "cpu_guard_rule": "median of three one-second samples; known simulation process always stops",
    }
    payload["runner_sha256"] = sha256(Path(__file__).resolve())
    write_json(json_path, payload)
    return json_path, npz_path, arrays, payload


def _trajectory(case, rho0, span, method_name, execution_mode, phase_case, phase,
                reltol=1e-8, abstol=1e-10):
    use_phase = method_name == "phase_gauge"
    active_case = phase_case if use_phase else case
    rho = methods.phase_frame_density(rho0, span[0], phase) if use_phase else rho0
    mode = "structured" if use_phase else execution_mode
    times, densities, stats = driver.propagate_segmented(
        active_case, rho, span, solver="tsit5", reltol=reltol, abstol=abstol,
        dt=1e-10, execution_mode=mode, save_step_s=100e-9,
    )
    if use_phase:
        densities = np.asarray([
            methods.original_frame_density(matrix, t, phase)
            for t, matrix in zip(times, densities, strict=True)
        ])
    photons = driver.photon_yield(stats)
    return times, densities, photons, stats


def run_case(config, root_dir, cutoff_percent, profile_repeats):
    case_dir = root_dir / config.name
    case_dir.mkdir(parents=True, exist_ok=True)
    case_started = time.perf_counter()
    case_deadline = case_started + CASE_BUDGET_SECONDS
    started = time.perf_counter()
    case = driver.build_case(config, prepare_frozen=False, amplitude_envelope="constant")
    driver._prepare_segmented_plan(case)
    phase_case, phase = make_phase_case(case)
    setup_seconds = time.perf_counter() - started
    with threadpool_limits(limits=1):
        direct_profile, random_profile, phase_profile = make_probe(
            case, phase_case, phase, profile_repeats=profile_repeats
        )
        profile_payload = {
            "case": config.name, "config": driver.provenance(case)["config"],
            "direct_initial": direct_profile, "random_hermitian": random_profile,
            "phase_gauge_initial": phase_profile,
            "provenance": driver.provenance(case),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "profile_repeats": profile_repeats,
        }
        write_json(case_dir / "profile.json", profile_payload)
        rho0 = case.rho0(LABEL)
        span = (config.entry_phase_s, config.entry_phase_s + config.duration_s)
        ref_json, ref_npz, ref_arrays, reference = build_reference(
            case, rho0, span, case_dir, cutoff_percent
        )
        profile_means = {
            mode: float(direct_profile["modes"][mode]["native_profile"]["average_total_seconds"])
            for mode in PROFILE_MODES
        }
        profile_means["phase_gauge"] = (
            float(phase_profile["modes"]["structured"]["native_profile"]["average_total_seconds"])
            if phase_profile is not None else None
        )
        progress_state = {
            "status": "screening", "case": config.name,
            "common_model_preparation_seconds": setup_seconds,
            "profile_complete": True, "reference_complete": True,
            "profile_native_mean_seconds": profile_means,
            "reference_json": str(ref_json), "reference_npz": str(ref_npz),
            "completed_trials": [], "case_budget_seconds": CASE_BUDGET_SECONDS,
        }
        write_json(case_dir / "case.progress.json", progress_state)

        candidates = [
            (mode, mode if mode in PROFILE_MODES else "structured") for mode in MODE_NAMES
        ]
        order = candidates.copy()
        # Rotate by case index to vary the initial order and avoid always timing
        # the same candidate first in the screening pass.
        shift = len([p for p in root_dir.iterdir() if p.is_dir()]) - 1
        order = order[shift % len(order):] + order[:shift % len(order)]
        results = []
        saved = {"reference_times": ref_arrays["reference_times"]}
        not_run = []
        accepted_tolerances = {}

        def run_trial(mode_name, execution_mode, phase_name, repeat=None):
            if time.perf_counter() >= case_deadline:
                return None
            before, guard = take_resource_snapshot(cutoff_percent)
            if not guard["okay"]:
                raise RuntimeError(f"guard before {config.name}/{mode_name}: {guard['reason']}")
            wall_start = time.perf_counter()
            trial_reltol, trial_abstol = accepted_tolerances.get(
                mode_name, (1e-8, 1e-10)
            ) if phase_name != "screening" else (1e-8, 1e-10)
            times, density, photons, stats = _trajectory(
                case, rho0, span, mode_name, execution_mode, phase_case, phase,
                reltol=trial_reltol, abstol=trial_abstol,
            )
            wall = time.perf_counter() - wall_start
            score = sweep.compare_trajectory(
                times, density, photons, ref_arrays["reference_times"],
                ref_arrays["reference_density"], reference["reference_photon_yield"],
            )
            after, guard_after = take_resource_snapshot(cutoff_percent)
            row = {
                "phase": phase_name, "repeat": repeat, "case": config.name, "mode": mode_name,
                "solver": "tsit5", "reltol": trial_reltol, "abstol": trial_abstol,
                "execution_mode": execution_mode, "wall_seconds": wall,
                "full_density_accepted_steps": sweep._numeric_sum(stats, "accepted_steps"),
                "full_density_rejected_steps": sweep._numeric_sum(stats, "rejected_steps"),
                "full_density_rhs_calls": sweep._numeric_sum(stats, "rhs_calls"),
                "photon_yield": photons, "accuracy_vs_reference": score,
                "resource_before": before, "resource_guard_before": guard,
                "resource_after": after, "resource_guard_after": guard_after,
            }
            results.append(row)
            progress_state["completed_trials"] = results.copy()
            progress_state["last_completed_mode"] = mode_name
            progress_state["elapsed_case_seconds"] = time.perf_counter() - case_started
            write_json(case_dir / "case.progress.json", progress_state)
            if not guard_after["okay"]:
                raise RuntimeError(f"guard after {config.name}/{mode_name}: {guard_after['reason']}")
            if not score.get("passes"):
                # Preserve the failed first pass, then try one tighter matched
                # integration before excluding this method from repeats.
                before_retry, retry_guard = take_resource_snapshot(cutoff_percent)
                if not retry_guard["okay"]:
                    raise RuntimeError(f"guard before accuracy retry {config.name}/{mode_name}: {retry_guard['reason']}")
                retry_started = time.perf_counter()
                times, density, photons, stats = _trajectory(
                    case, rho0, span, mode_name, execution_mode, phase_case, phase,
                    reltol=1e-9, abstol=1e-11,
                )
                retry_wall = time.perf_counter() - retry_started
                retry_score = sweep.compare_trajectory(
                    times, density, photons, ref_arrays["reference_times"],
                    ref_arrays["reference_density"], reference["reference_photon_yield"],
                )
                after_retry, retry_guard_after = take_resource_snapshot(cutoff_percent)
                retry_row = {
                    "phase": "accuracy_retry", "repeat": repeat, "case": config.name,
                    "mode": mode_name, "execution_mode": execution_mode,
                    "solver": "tsit5", "reltol": 1e-9, "abstol": 1e-11,
                    "wall_seconds": retry_wall,
                    "full_density_accepted_steps": sweep._numeric_sum(stats, "accepted_steps"),
                    "full_density_rejected_steps": sweep._numeric_sum(stats, "rejected_steps"),
                    "full_density_rhs_calls": sweep._numeric_sum(stats, "rhs_calls"),
                    "photon_yield": photons, "accuracy_vs_reference": retry_score,
                    "retry_of_initial_tolerance": {"reltol": 1e-8, "abstol": 1e-10},
                    "resource_before": before_retry, "resource_guard_before": retry_guard,
                    "resource_after": after_retry, "resource_guard_after": retry_guard_after,
                }
                results.append(retry_row)
                progress_state["completed_trials"] = results.copy()
                write_json(case_dir / "case.progress.json", progress_state)
                if not retry_guard_after["okay"]:
                    raise RuntimeError(f"guard after accuracy retry {config.name}/{mode_name}: {retry_guard_after['reason']}")
                if not retry_score.get("passes"):
                    return retry_row
                accepted_tolerances[mode_name] = (1e-9, 1e-11)
                row["superseded_by_tighter_retry"] = True
                progress_state["completed_trials"] = results.copy()
                write_json(case_dir / "case.progress.json", progress_state)
                row = retry_row
            saved[mode_name] = density
            return row

        for position, (mode_name, execution_mode) in enumerate(order):
            row = run_trial(mode_name, execution_mode, "screening", repeat=1)
            if row is None:
                not_run.extend(
                    f"{name}:screening" for name, _ in order[position:]
                    if not any(r["mode"] == name and r["phase"] == "screening" for r in results)
                )
                break

        screening = [row for row in results if row["phase"] == "screening"]
        passing_by_mode = {}
        for row in results:
            if row["phase"] in ("screening", "accuracy_retry") and row["accuracy_vs_reference"].get("passes", False):
                passing_by_mode[row["mode"]] = row
        passing_screening = list(passing_by_mode.values())
        measured_walls = [row["wall_seconds"] for row in passing_screening]
        base_row = next((row for row in passing_screening if row["mode"] == "structured"), None)
        base_wall = base_row["wall_seconds"] if base_row is not None else None
        fastest_wall = min(measured_walls) if measured_walls else None
        cutoff_wall = (
            COMPETITIVE_MARGIN * min(base_wall, fastest_wall)
            if base_wall is not None and fastest_wall is not None else None
        )
        competitive_order = [
            mode for mode in MODE_NAMES
            if mode != "structured"
            and cutoff_wall is not None
            and any(row["mode"] == mode and row["wall_seconds"] <= cutoff_wall for row in passing_screening)
        ]

        # Two additional rounds include one shared structured solve per round;
        # the baseline is deliberately interleaved and its shared nature is
        # recorded rather than presented as an independent pair for every mode.
        paired = {mode: [] for mode in competitive_order}
        for repeat in (2, 3):
            if not competitive_order:
                break
            round_modes = competitive_order if (repeat + shift) % 2 == 0 else list(reversed(competitive_order))
            baseline_position = 0 if repeat == 2 else len(round_modes) // 2
            round_order = list(round_modes)
            round_order.insert(baseline_position, "structured")
            baseline_row = None
            candidate_rows = {}
            for position, method_name in enumerate(round_order):
                execution_mode = method_name if method_name in PROFILE_MODES else "structured"
                row = run_trial(method_name, execution_mode, "paired_repeat", repeat=repeat)
                if row is None:
                    not_run.extend(
                        f"{name}:repeat{repeat}" for name in round_order[position:]
                        if not any(r["mode"] == name and r["repeat"] == repeat and r["phase"] == "paired_repeat" for r in results)
                    )
                    break
                if method_name == "structured":
                    baseline_row = row
                else:
                    candidate_rows[method_name] = row
            if baseline_row is None or not baseline_row["accuracy_vs_reference"].get("passes", False):
                if baseline_row is None and time.perf_counter() >= case_deadline:
                    not_run.extend(
                        f"{mode}:repeat{repeat}" for mode in competitive_order
                        if not any(r["mode"] == mode and r["repeat"] == repeat and r["phase"] == "paired_repeat" for r in results)
                    )
                break
            failed_this_round = []
            for mode, row in candidate_rows.items():
                if row["accuracy_vs_reference"].get("passes", False):
                    paired[mode].append({
                        "repeat": repeat, "order": round_order,
                        "baseline_is_shared_within_round": True,
                        "candidate_over_structured_wall_ratio": row["wall_seconds"] / baseline_row["wall_seconds"],
                        "candidate_wall_seconds": row["wall_seconds"],
                        "structured_wall_seconds": baseline_row["wall_seconds"],
                    })
                else:
                    failed_this_round.append(mode)
            if failed_this_round:
                competitive_order = [mode for mode in competitive_order if mode not in failed_this_round]
            if time.perf_counter() >= case_deadline:
                not_run.extend(
                    f"{mode}:repeat{future_repeat}"
                    for future_repeat in range(repeat + 1, 4)
                    for mode in competitive_order
                )
                break
        npz_path = case_dir / "trajectories.npz"
        np.savez_compressed(npz_path, **saved)
        screening_complete = len({row["mode"] for row in screening}) == len(MODE_NAMES)
        screening_budget_omissions = [item for item in not_run if item.endswith(":screening")]
        all_accuracy_pass = all(
            row["accuracy_vs_reference"].get("passes", False)
            for row in results if not row.get("superseded_by_tighter_retry", False)
        )
        case_status = (
            "partial_budget" if not screening_complete or screening_budget_omissions
            else "complete" if all_accuracy_pass else "complete_with_failures"
        )
        summary = {
            "status": case_status,
            "screening_complete": screening_complete,
            "case": config.name, "config": driver.provenance(case)["config"],
            "optical_power_ratio": config.optical_rabi_over_gamma ** 2,
            "microwave_power_ratio": config.microwave_rabi_over_gamma ** 2,
            "detuning_MHz": config.detuning_mhz,
            "detuning_rad_s": config.detuning_rad_s,
            "profile_native_mean_seconds": profile_means,
            "repeat_eligibility_basis": "actual screening trajectory wall time; mode is competitive when within 20% of the faster of structured and fastest measured mode",
            "competitive_modes_by_screening_wall": competitive_order,
            "screening_rows": [row for row in results if row["phase"] == "screening"],
            "all_trial_rows": results,
            "accepted_tolerances_by_mode": accepted_tolerances,
            "paired_repeats": paired,
            "reference_json": str(ref_json), "reference_npz": str(ref_npz),
            "reference_photon_yield": reference["reference_photon_yield"],
            "reference_npz_sha256": sha256(ref_npz),
            "trajectory_npz": str(npz_path), "trajectory_npz_sha256": sha256(npz_path),
            "all_effective_trajectory_accuracy_pass": all_accuracy_pass,
            "not_run_budget": not_run,
            "case_budget_seconds": CASE_BUDGET_SECONDS,
            "elapsed_case_seconds": time.perf_counter() - case_started,
            "common_model_preparation_seconds": setup_seconds,
            "provenance": driver.provenance(case),
            "runner_sha256": sha256(Path(__file__).resolve()),
        }
        write_json(case_dir / "case_summary.json", summary)
        progress_state["status"] = case_status
        progress_state["summary_json"] = str(case_dir / "case_summary.json")
        progress_state["elapsed_case_seconds"] = time.perf_counter() - case_started
        write_json(case_dir / "case.progress.json", progress_state)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG_DEFAULT)
    parser.add_argument("--cpu-cutoff-percent", type=float, default=CPU_CUTOFF_PERCENT)
    args = parser.parse_args()
    if not math.isfinite(args.cpu_cutoff_percent) or args.cpu_cutoff_percent <= 0:
        parser.error("--cpu-cutoff-percent must be finite and positive")
    root_dir = RESULTS / args.tag
    root_dir.mkdir(parents=True, exist_ok=False)
    run_payload = {
        "status": "running", "complete": False,
        "tag": args.tag, "case_names": [config.name for config in case_matrix()],
        "modes": list(MODE_NAMES), "competitive_margin": COMPETITIVE_MARGIN,
        "reference_tolerance": REFERENCE_TOLERANCE,
        "cpu_cutoff_percent": args.cpu_cutoff_percent,
        "cpu_guard_rule": "median of three one-second samples; known simulation process always stops",
        "duration_s": 10e-6, "entry_phase_s": 137e-9,
        "solver": "tsit5", "reltol": 1e-8, "abstol": 1e-10,
        "rows": [], "runner_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(root_dir / "run.progress.json", run_payload)
    with threadpool_limits(limits=1):
        for config in case_matrix():
            before, guard = take_resource_snapshot(args.cpu_cutoff_percent)
            if not guard["okay"]:
                raise RuntimeError(f"pre-case resource guard failed: {guard['reason']}")
            summary = run_case(config, root_dir, args.cpu_cutoff_percent, profile_repeats=1000)
            run_payload["rows"].append(summary)
            run_payload["last_case_resource"] = before
            write_json(root_dir / "run.progress.json", run_payload)
    all_cases_screened = len(run_payload["rows"]) == len(case_matrix()) and all(
        row["screening_complete"] for row in run_payload["rows"]
    )
    all_pass = all(row["all_effective_trajectory_accuracy_pass"] for row in run_payload["rows"])
    run_payload["complete"] = all_cases_screened
    run_payload["status"] = (
        "partial_budget" if not all_cases_screened
        else "complete" if all_pass else "complete_with_failures"
    )
    run_payload["resource_conditions_after"] = comparison.resource_snapshot()
    write_json(root_dir / "run.json", run_payload)
    write_json(root_dir / "run.progress.json", run_payload)
    print(f"Wrote {root_dir / 'run.json'}")


if __name__ == "__main__":
    main()
