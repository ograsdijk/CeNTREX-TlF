"""Three-way alternating long-horizon finalist benchmark, K=1.

For each output and fixture, compares the production structured kernel, the
DSU factorial variant, and the fastest accepted non-DSU experimental kernel
from the completed nominal screen. Exact 10/109/200 us stages are explicit;
this script does no work without ``--run``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import run_rc_cross_system_screen_v3 as screen
except ModuleNotFoundError:
    from benchmarks import run_rc_cross_system_screen_v3 as screen

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "rc_propagation_results" / "cross_system_long_finalists_v1"
FIXTURES = screen.FIXTURES
PRODUCTS = screen.PRODUCTS
BASELINE = "structured"
DSU = "experimental_structured_sparse_dense_diag_soa_single_unpack"
PREFIX = "experimental_structured_sparse_dense"
DURATIONS = {"10us": 10e-6, "109us": 109e-6, "200us": 200e-6}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, None):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def save_cell(out: Path, index: int, value: Any) -> tuple[str, str]:
    path = out / "cells" / f"cell_{index:05d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    arrays = value if isinstance(value, dict) else {"output": value}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{name: np.asarray(array) for name, array in arrays.items()})
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, None):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)
    return str(path.relative_to(OUTROOT)), sha(path)


def guard_sample() -> dict[str, Any]:
    import psutil
    samples = [float(psutil.cpu_percent(interval=1.0)) for _ in range(3)]
    current = psutil.Process()
    exempt = {os.getpid(), *(p.pid for p in current.parents())}
    heavy = ("run_cross_system_long_finalists.py", "run_cross_system_regime_screen.py",
             "run_rc_cross_system_screen_v3.py", "run_cross_system_batch_amortization_v2.py",
             "bench_rc_propagation.py", "simulate_all.py")
    competitors = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
            if process.info["pid"] not in exempt and "python" in (process.info.get("name") or "").lower() and any(x in command for x in heavy):
                competitors.append({"pid": process.info["pid"], "cmdline": command[:250]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"cpu_samples": samples, "cpu_median_percent": float(np.median(samples)),
            "competing_processes": competitors}


def finalist_modes(screen_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = screen_json["rows"]
    chosen = {}
    for product in PRODUCTS:
        by_mode = {row["kernel"]: row for row in rows if row.get("product") == product}
        baseline = by_mode.get(BASELINE)
        dsu = by_mode.get(DSU)
        experiments = [row for mode, row in by_mode.items()
                       if mode.startswith(PREFIX) and mode not in {BASELINE, DSU}
                       and row.get("accuracy_passed") is True
                       and isinstance(row.get("accepted_wall_seconds"), (float, int))]
        fastest = min(experiments, key=lambda row: row["accepted_wall_seconds"]) if experiments else None
        chosen[product] = {
            "baseline": {"kernel": BASELINE, "screen_row": baseline},
            "dsu": {"kernel": DSU, "screen_row": dsu},
            "best_non_dsu": ({"kernel": fastest["kernel"], "screen_row": fastest} if fastest else None),
            "selection_rule": "minimum accepted 1-us K=1 wall among experimental modes except DSU",
        }
    return chosen


def budget_proxy(screen_json: dict[str, Any], selected: dict[str, dict[str, Any]],
                 duration_label: str) -> dict[str, Any]:
    ratio = DURATIONS[duration_label] / 1e-6
    sums = {}
    count = 0
    missing = []
    for product, items in selected.items():
        for method_name in ("baseline", "dsu", "best_non_dsu"):
            candidate = items[method_name]
            row = None if candidate is None else candidate.get("screen_row")
            if not row or row.get("accuracy_passed") is not True:
                missing.append({"product": product, "method": method_name})
                continue
            sums[product, method_name] = float(row.get("accepted_wall_seconds", row.get("wall_seconds", 0.0)))
            count += 1
    return {
        "estimate_kind": "linear duration-ratio proxy from accepted 1-us timing; not a bound or a performance claim",
        "duration_label": duration_label, "duration_ratio_to_short_screen": ratio,
        "timed_cells_for_three_repeats": count * 3,
        "reference_calls": 2, "candidate_modes_per_product": 3,
        "estimated_candidate_seconds": float(sum(sums.values()) * ratio * 3),
        "estimated_reference_seconds": float(sum(screen_json.get("reference_checks", {}).get("reference_wall_seconds", {}).values()) * ratio),
        "missing_short_screen_candidates": missing,
        "warning": "ODE cost need not scale linearly with duration; use this only to budget a bounded stage.",
    }


def make_reference(case: dict[str, Any], duration_s: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    spec = case["spec"]
    t0 = float(spec.t_span[0])
    history = []
    for rtol, atol in ((1e-10, 1e-12), (1e-11, 1e-13), (1e-12, 1e-14)):
        ref_spec = replace(spec, t_span=(t0, t0 + duration_s), rtol=rtol, atol=atol)
        outputs = {}
        walls = {}
        for solver_name in ("dopri5", "tsit5"):
            result, wall = screen._solve_once(case, ref_spec, "structured", "full_and_integral", solver=solver_name)
            values = screen._value_for_product(result, case, "full_and_integral")
            n = len(case["system"].QN)
            outputs[solver_name] = {"full": values[:-1].reshape(n, n),
                                    "photon_integral": np.asarray([values[-1]])}
            walls[solver_name] = wall
        rho_diff = float(np.max(np.abs(outputs["dopri5"]["full"] - outputs["tsit5"]["full"])))
        photon_diff = float(np.max(np.abs(outputs["dopri5"]["photon_integral"] - outputs["tsit5"]["photon_integral"])))
        history.append({"rtol": rtol, "atol": atol, "density_diff": rho_diff,
                        "photon_diff": photon_diff, "wall_seconds": walls})
        if rho_diff <= 1e-8 and photon_diff <= 1e-8:
            cross_population = screen._accuracy_metrics(
                "populations", np.diag(outputs["dopri5"]["full"]), case,
                {"full": outputs["tsit5"]["full"],
                 "photon_integral": outputs["tsit5"]["photon_integral"]})
            physical = {}
            for solver_name, result in outputs.items():
                rho = result["full"]
                physical[solver_name] = {
                    "trace_abs_error": float(abs(np.trace(rho) - 1.0)),
                    "hermiticity_max_abs": float(np.max(np.abs(rho-rho.conj().T))),
                    "minimum_eigenvalue": float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min()),
                    "population_metrics_vs_tsit5": cross_population,
                }
            if any(v["trace_abs_error"] > 1e-9 or v["hermiticity_max_abs"] > 1e-9
                   or v["minimum_eigenvalue"] < -1e-8 for v in physical.values()) \
                    or not cross_population.get("accuracy_passed"):
                raise RuntimeError(f"long reference physical/population gate failed: {physical}")
            ref = outputs["tsit5"]
            ref.update({"dopri5_full": outputs["dopri5"]["full"],
                        "dopri5_photon_integral": outputs["dopri5"]["photon_integral"],
                        "tsit5_full": outputs["tsit5"]["full"],
                        "tsit5_photon_integral": outputs["tsit5"]["photon_integral"]})
            return ref, {"cross_solver_density_max_abs": rho_diff,
                         "cross_solver_photon_max_abs": photon_diff,
                         "reference_wall_seconds": walls, "physical": physical,
                         "solver_names": ["dopri5", "tsit5"],
                         "reference_tolerance": {"rtol": rtol, "atol": atol},
                         "tightening_history": history}
    raise RuntimeError(f"tight long reference mismatch after tolerance tightening: {history}")


def _run_finalists(fixture_name: str, screen_path: Path, duration_label: str, tag: str) -> Path:
    source = json.loads(screen_path.read_text(encoding="utf-8"))
    if source.get("complete") is not True or source.get("fixture") != fixture_name:
        raise ValueError("nominal screen must be complete and fixture-matched")
    if not source.get("native_extension_sha256"):
        raise ValueError("nominal screen is missing native-extension provenance")
    native_path = Path(importlib.import_module("centrex_tlf.centrex_tlf_rust").__file__).resolve()
    if sha(native_path) != source["native_extension_sha256"]:
        raise ValueError("nominal screen was produced by a different native extension build")
    out = OUTROOT / tag / fixture_name / duration_label
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}; use a new tag")
    out.mkdir(parents=True, exist_ok=True)
    before = guard_sample()
    if before["cpu_median_percent"] > 50 or before["competing_processes"]:
        raise RuntimeError(f"pre-run resource guard failed: {before}")
    case, build_seconds = screen._build_fixture(fixture_name)
    selected = finalist_modes(source)
    budget = budget_proxy(source, selected, duration_label)
    atomic_json(out / "budget.json", {"fixture": fixture_name, "selection": selected, "proxy": budget})
    target_s = DURATIONS[duration_label]
    reference, reference_checks = make_reference(case, target_s)
    # Reference generation tightens case["spec"]. Restore one shared screen
    # tolerance and solver for every finalist so timing comparisons are matched.
    baseline_rows = [r for r in source["rows"] if r.get("kernel") == BASELINE
                     and r.get("product") == "full" and r.get("accuracy_passed") is True]
    if not baseline_rows:
        raise ValueError("nominal screen has no accepted structured/full baseline")
    benchmark_tolerance = baseline_rows[0].get("accepted_tolerance") or {
        "rtol": source["reltol"], "atol": source["abstol"]}
    benchmark_solver = baseline_rows[0].get("solver", source.get("solver", "dopri5"))
    t0 = float(case["spec"].t_span[0])
    benchmark_spec = replace(case["spec"], t_span=(t0, t0 + target_s),
                             rtol=float(benchmark_tolerance["rtol"]),
                             atol=float(benchmark_tolerance["atol"]))
    case["spec"] = benchmark_spec
    reference_path = out / "reference.npz"
    if reference_path.exists():
        raise FileExistsError(reference_path)
    temp_reference = reference_path.with_name("reference.tmp.npz")
    np.savez_compressed(temp_reference, **reference)
    temp_reference.replace(reference_path)

    raw_rows = []
    row_index = 0
    # Trial order reverses/rotates by repeat to reduce run-order bias.
    for product in PRODUCTS:
        candidates = selected[product]
        mode_rows = [candidates[k] for k in ("baseline", "dsu", "best_non_dsu") if candidates[k] is not None]
        mode_rows = [m for m in mode_rows if m.get("screen_row") and m["screen_row"].get("accuracy_passed") is True]
        if not mode_rows:
            raw_rows.append({"product": product, "status": "not_eligible", "reason": "no accepted finalist in nominal screen"})
            continue
        guard_before = guard_sample()
        if guard_before["cpu_median_percent"] > 50 or guard_before["competing_processes"]:
            raise RuntimeError(f"pre-product guard failed: {guard_before}")
        for repeat in range(3):
            order = mode_rows[repeat % len(mode_rows):] + mode_rows[:repeat % len(mode_rows)]
            if repeat % 2:
                order.reverse()
            for candidate in order:
                mode = candidate["kernel"]
                sr = candidate["screen_row"]
                # All contenders use the same solver and tolerance. Per-row
                # screen settings are selection evidence, not long-stage settings.
                solver_name = benchmark_solver
                tolerance = benchmark_tolerance
                started = time.perf_counter()
                result, wall = screen._solve_once(case, benchmark_spec, mode, product, solver=solver_name)
                value = screen._value_for_product(result, case, product)
                metrics = screen._accuracy_metrics(product, value, case, reference)
                initial_value = np.array(value, copy=True)
                initial_metrics = metrics
                trial = {"fixture": fixture_name, "duration": duration_label,
                         "product": product, "kernel": mode, "repeat": repeat + 1,
                         "solver": solver_name, "rtol": tolerance["rtol"], "atol": tolerance["atol"],
                         "wall_seconds": wall, "outer_wall_seconds": time.perf_counter() - started,
                         "initial_accuracy": initial_metrics,
                         "accuracy": metrics, "accepted_accuracy": metrics,
                         "accuracy_passed": bool(metrics.get("accuracy_passed")),
                         "strict_accuracy_passed": bool(metrics.get("strict_accuracy_passed")),
                         "accepted_wall_seconds": wall if metrics.get("accuracy_passed") else None,
                         "accepted_tolerance": tolerance if metrics.get("accuracy_passed") else None,
                         "status": "passed" if metrics.get("accuracy_passed") else "accuracy_failed",
                         "pre_repeat_guard": guard_before}
                if not trial["accuracy_passed"]:
                    retry_rows = []
                    for retry_rtol, retry_atol in ((1e-8, 1e-10), (1e-9, 1e-11)):
                        retry_result, retry_wall = screen._solve_once(
                            case, replace(benchmark_spec, rtol=retry_rtol, atol=retry_atol), mode, product,
                            solver=solver_name,
                        )
                        retry_value = screen._value_for_product(retry_result, case, product)
                        retry_metrics = screen._accuracy_metrics(product, retry_value, case, reference)
                        retry_rows.append({"rtol": retry_rtol, "atol": retry_atol,
                                           "wall_seconds": retry_wall, "metrics": retry_metrics,
                                           "accuracy_passed": bool(retry_metrics.get("accuracy_passed"))})
                        if retry_metrics.get("accuracy_passed"):
                            value, metrics = retry_value, retry_metrics
                            trial.update(status="passed_tight_retry", accuracy_passed=True,
                                         strict_accuracy_passed=bool(retry_metrics.get("strict_accuracy_passed")),
                                         accepted_accuracy=retry_metrics,
                                         accepted_tolerance={"rtol": retry_rtol, "atol": retry_atol},
                                         accepted_wall_seconds=retry_wall)
                            break
                    trial["tight_retries"] = retry_rows
                cell_path, cell_sha = save_cell(out, row_index,
                    {"initial_output": initial_value, "accepted_output": np.asarray(value)})
                row_index += 1
                trial["artifact"] = cell_path
                trial["artifact_sha256"] = cell_sha
                raw_rows.append(trial)
                atomic_json(out / "progress" / f"after_{len(raw_rows):05d}.json", {
                    "complete": False, "rows_completed": len(raw_rows), "rows": raw_rows,
                })
        guard_after = guard_sample()
        raw_rows.append({"product": product,
                         "status": "resource_guard_passed" if guard_after["cpu_median_percent"] <= 50 and not guard_after["competing_processes"] else "resource_guard_failed",
                         "guard": guard_after})
        atomic_json(out / "progress" / f"after_{len(raw_rows):05d}.json", {
            "complete": False, "rows_completed": len(raw_rows), "rows": raw_rows,
        })
        if guard_after["cpu_median_percent"] > 50 or guard_after["competing_processes"]:
            raise RuntimeError(f"post-product guard failed: {guard_after}")

    summary = []
    for product in PRODUCTS:
        product_rows = [r for r in raw_rows if r.get("product") == product and r.get("kernel")]
        for mode in sorted({r["kernel"] for r in product_rows}):
            mode_rows = [r for r in product_rows if r["kernel"] == mode]
            accepted = [r for r in mode_rows if r.get("accuracy_passed")]
            times = [float(r.get("accepted_wall_seconds", r["wall_seconds"])) for r in accepted]
            summary.append({"product": product, "kernel": mode,
                            "accepted_trials": len(accepted), "expected_trials": 3,
                            "timings_accepted": len(accepted) == 3,
                            "median_wall_seconds": float(np.median(times)) if times else None,
                            "range_wall_seconds": [min(times), max(times)] if times else None,
                            "trials": mode_rows})
    after = guard_sample()
    payload = {
        "status": "complete" if after["cpu_median_percent"] <= 50 and not after["competing_processes"] else "resource_guard_failed",
        "complete": after["cpu_median_percent"] <= 50 and not after["competing_processes"],
        "fixture": fixture_name, "duration_label": duration_label,
        "duration_s": target_s, "tag": tag, "reference_checks": reference_checks,
        "fixture_build_seconds": build_seconds, "reference_npz_sha256": sha(reference_path),
        "nominal_screen_path": str(screen_path.resolve()), "nominal_screen_sha256": sha(screen_path),
        "native_extension_path": str(native_path), "native_extension_sha256": sha(native_path),
        "matched_solver": benchmark_solver, "matched_tolerance": benchmark_tolerance,
        "runner_sha256": sha(Path(__file__).resolve()), "selection": selected,
        "timing_budget_proxy": budget, "raw_rows": raw_rows, "summary": summary,
        "resource_before": before, "resource_after": after,
        "timing_scope": "K=1 nominal long-horizon final-output comparisons; no batch or parameter-scan extrapolation",
    }
    result_path = out / "result.json"
    atomic_json(result_path, payload)
    atomic_json(out / "progress" / "complete.json", {
        "complete": payload["complete"], "result_sha256": sha(result_path),
    })
    if not payload["complete"]:
        raise RuntimeError(f"post-stage guard failed: {after}")
    return result_path


def run_finalists(fixture_name: str, screen_path: Path, duration_label: str, tag: str) -> Path:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _run_finalists(fixture_name, screen_path, duration_label, tag)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=FIXTURES, required=True)
    parser.add_argument("--duration", choices=tuple(DURATIONS), required=True)
    parser.add_argument("--screen-reference", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("no long finalist runs without --run")
    if args.fixture == "rc" and args.duration == "109us":
        parser.error("RC long stage uses 200us, not 109us")
    if args.fixture != "rc" and args.duration == "200us":
        parser.error("R2/R3 long stage uses 109us, not 200us")
    print(run_finalists(args.fixture, args.screen_reference, args.duration, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
