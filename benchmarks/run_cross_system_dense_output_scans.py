"""Per-output fixed-budget dense scans joined to frozen native scan timings.

No trajectories run on import. Physical collection is gated by ``--run``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

try:
    from cross_system_fixtures import build_fixture
    from cross_system_methods import evaluate_static_dense
    from run_cross_system_batch_amortization import PRODUCTS, accuracy
    from run_cross_system_parallel_dense import (
        OUTROOT as OLD_ROOT,
    )
    from run_cross_system_parallel_dense import (
        TreeMemorySampler,
        _expected_points,
        _resource_guard,
        _resource_snapshot,
        _save_memory_samples,
        _write_json,
    )
    from run_cross_system_parameter_throughput import (
        FIXTURES,
        WORKERS,
        _batches,
        _points,
        _reference,
        _sha,
    )

    from centrex_tlf.lindblad.dense import prepare_dense_lindblad_propagator
except ModuleNotFoundError:
    from benchmarks.cross_system_fixtures import build_fixture
    from benchmarks.cross_system_methods import evaluate_static_dense
    from benchmarks.run_cross_system_batch_amortization import PRODUCTS, accuracy
    from benchmarks.run_cross_system_parallel_dense import (
        OUTROOT as OLD_ROOT,
    )
    from benchmarks.run_cross_system_parallel_dense import (
        TreeMemorySampler,
        _expected_points,
        _resource_guard,
        _resource_snapshot,
        _save_memory_samples,
        _write_json,
    )
    from benchmarks.run_cross_system_parameter_throughput import (
        FIXTURES,
        WORKERS,
        _batches,
        _points,
        _reference,
        _sha,
    )
    from centrex_tlf.lindblad.dense import prepare_dense_lindblad_propagator

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "rc_propagation_results" / "cross_system_dense_output_scans_v1"
SOURCE_TAG = "suite_v1"
STATIC_FIXTURES = tuple(name for name in FIXTURES if name != "rc")
KERNEL = "experimental_structured_sparse_dense_diag_soa_single_unpack"
SOLVER = "tsit5"
RTOL, ATOL = 1e-9, 1e-11
REPEATS = 3


def _output_worker(name: str, rho_batch: np.ndarray, task_queue: Any, result_queue: Any) -> None:
    limits = threadpool_limits(limits=1)
    assert limits is not None
    fixture = build_fixture(name)
    cache: dict[int, tuple[Any, dict[str, Any]]] = {}
    result_queue.put({"kind": "ready", "pid": os.getpid()})
    while True:
        task = task_queue.get()
        if task["kind"] == "stop":
            return
        index = int(task["point_index"])
        if task["kind"] == "prepare":
            point = task["point"]
            started = time.perf_counter()
            propagator = prepare_dense_lindblad_propagator(
                fixture.prepared, rho_batch,
                parameter_values={"rabi": complex(point["rabi"]),
                                  "detuning": complex(point["detuning"])},
                execution_mode=KERNEL, reduce=True)
            elapsed = time.perf_counter() - started
            stats = dict(propagator.stats)
            stats.update({"setup_wall_seconds": elapsed,
                          "initial_count_prepared": len(rho_batch),
                          "runtime_parameter_values": {
                              key: float(point[key]) for key in ("rabi", "detuning")}})
            cache[index] = (propagator, stats)
            result_queue.put({"kind": "prepared", "point_index": index,
                              "setup": stats, "pid": os.getpid()})
        elif task["kind"] == "output":
            propagator, _stats = cache[index]
            product = task["product"]
            value, elapsed = evaluate_static_dense(
                fixture, propagator, product, rho0_batch=rho_batch)
            if isinstance(value, dict):
                value = {"rho": np.asarray(value.get("rho", value.get("full"))),
                         "photon_integral": np.asarray(value["photon_integral"])}
            else:
                value = np.asarray(value)
            result_queue.put({"kind": "output", "point_index": index,
                              "product": product, "value": value,
                              "product_seconds": elapsed, "pid": os.getpid()})
        else:
            raise ValueError(f"unknown worker task {task!r}")


def _collect(queue: Any, count: int) -> list[dict[str, Any]]:
    results, deadline = [], time.monotonic() + 180.0
    while len(results) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"only received {len(results)} of {count} worker responses")
        results.append(queue.get(timeout=remaining))
    return results


def _load_or_build_reference(fixture: Any, batch_name: str, rho_batch: np.ndarray,
                             source_tag: str, out: Path):
    source_dir = OLD_ROOT / source_tag / fixture.spec.name
    source_json = source_dir / "result.json"
    if source_json.exists():
        source = json.loads(source_json.read_text(encoding="utf-8"))
        meta = source.get("fixture_metadata", {})
        stable_keys = ("state_count", "packed_length", "main_ground_label", "main_excited_label",
                       "initial_label", "selected_labels_4", "selected_indices_4",
                       "group_label", "group_indices", "excited_indices", "main_coupling",
                       "runtime_values", "native_extension_sha256", "fixture_builder_sha256")
        current = fixture.metadata
        ref_meta = source.get("reference_meta", {}).get(batch_name)
        if (source.get("complete") and source.get("points") == _points(False)
                and all(meta.get(key) == current.get(key) for key in stable_keys)
                and ref_meta and ref_meta.get("initial_count") == len(rho_batch)):
            ref_path = source_dir / ref_meta["reference_path"]
            if _sha(ref_path) == ref_meta.get("reference_sha256"):
                with np.load(ref_path) as data:
                    rho = np.asarray(data["rho"])
                    photon = np.asarray(data["photon"])
                if (rho.shape == (len(_points(False)), len(rho_batch), len(fixture.system.QN),
                                  len(fixture.system.QN))
                        and photon.shape == (len(_points(False)), len(rho_batch))
                        and np.all(np.isfinite(rho)) and np.all(np.isfinite(photon))):
                    return {"full": rho, "photon": photon}, {
                        "source": str(ref_path.relative_to(HERE)), "sha256": _sha(ref_path),
                        "source_result_sha256": _sha(source_json), "reused": True,
                        "reference_checks": ref_meta.get("physical_checks"),
                    }
    paired, checks = _reference(fixture, rho_batch, _points(False), "structured")
    rho, photon = paired["tsit5"]["full"], paired["tsit5"]["photon"]
    path = out / f"ref_{len(list(out.glob('ref_*.npz'))):02d}.npz"
    np.savez_compressed(path, rho=rho, photon=photon)
    return {"full": rho, "photon": photon}, {
        "source": path.name, "sha256": _sha(path), "reused": False,
        "reference_checks": checks,
    }


def _dense_output_sweeps(name: str, rho_batch: np.ndarray, points: list[dict[str, float]],
                         workers: int, products: tuple[str, ...], out: Path,
                         memory_artifacts: list[dict[str, Any]]):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in range(workers)]
    processes = [context.Process(target=_output_worker,
                                 args=(name, rho_batch, task_queues[i], result_queue))
                 for i in range(workers)]
    try:
        startup = time.perf_counter()
        for process in processes:
            process.start()
        ready = _collect(result_queue, workers)
        startup_seconds = time.perf_counter() - startup
        process_ids = {item["pid"] for item in ready}
        if len(process_ids) != workers or any(item.get("kind") != "ready" for item in ready):
            raise RuntimeError(f"worker startup mismatch: {ready}")
        owner = {index: index % workers for index in range(len(points))}
        with TreeMemorySampler() as sampler:
            started = time.perf_counter()
            for index, point in enumerate(points):
                task_queues[owner[index]].put({"kind": "prepare", "point_index": index, "point": point})
            prepared = _collect(result_queue, len(points))
            preparation_scan_seconds = time.perf_counter() - started
        preparation_memory = sampler.summary()
        preparation_memory["samples_artifact"] = _save_memory_samples(out, len(memory_artifacts), sampler)
        memory_artifacts.append(preparation_memory["samples_artifact"])
        setup_by_point = {item["point_index"]: item["setup"] for item in prepared}
        output_rows = []
        for product in products:
            product_repeats = []
            for _repeat in range(REPEATS):
                with TreeMemorySampler() as sampler:
                    started = time.perf_counter()
                    for index in range(len(points)):
                        task_queues[owner[index]].put({"kind": "output", "point_index": index,
                                                       "product": product})
                    responses = _collect(result_queue, len(points))
                    scan_seconds = time.perf_counter() - started
                summary = sampler.summary()
                summary["samples_artifact"] = _save_memory_samples(out, len(memory_artifacts), sampler)
                memory_artifacts.append(summary["samples_artifact"])
                by_point = {int(item["point_index"]): item for item in responses}
                product_repeats.append({
                    "scan_seconds": scan_seconds,
                    "per_point_product_seconds": {i: by_point[i]["product_seconds"] for i in by_point},
                    "values": {i: by_point[i]["value"] for i in by_point},
                    "memory": summary,
                })
            output_rows.append({"product": product, "repeats": product_repeats})
        return {"startup_seconds": startup_seconds, "worker_pids": sorted(process_ids),
                "preparation_scan_seconds": preparation_scan_seconds,
                "preparation_memory": preparation_memory, "setup_by_point": setup_by_point,
                "products": output_rows}
    finally:
        for queue in task_queues:
            with suppress(Exception):
                queue.put_nowait({"kind": "stop"})
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for queue in [*task_queues, result_queue]:
            with suppress(Exception):
                queue.close()
                queue.join_thread()


def _native_rows(source_dir: Path, batch_name: str, workers: int,
                 current_metadata: dict[str, Any], points: list[dict[str, float]]):
    source_json = source_dir / "result.json"
    if not source_json.exists():
        return {}, {"available": False, "reason": "no frozen native scan result"}
    source = json.loads(source_json.read_text(encoding="utf-8"))
    stable_keys = ("state_count", "packed_length", "main_ground_label", "main_excited_label",
                   "initial_label", "native_extension_sha256", "fixture_builder_sha256")
    if (not source.get("complete") or source.get("points") != points
            or source.get("kernel") != KERNEL or source.get("solver") != SOLVER
            or source.get("rtol") != RTOL or source.get("atol") != ATOL
            or any(source.get("fixture_metadata", {}).get(key) != current_metadata.get(key)
                   for key in stable_keys)):
        return {}, {"available": False, "reason": "frozen native result provenance mismatch"}
    rows: dict[str, list[dict[str, Any]]] = {product: [] for product in PRODUCTS}
    for product in PRODUCTS:
        for repeat in range(REPEATS):
            matching = [row for row in source.get("rows", [])
                        if row.get("batch") == batch_name and row.get("method") == "native_forward"
                        and row.get("workers") == workers and row.get("product") == product
                        and row.get("repeat") == repeat]
            point_ids = {int(row["point_index"]) for row in matching if row.get("point_index") is not None}
            times = [float(row["wall_seconds"]) for row in matching if row.get("wall_seconds") is not None]
            valid = (len(matching) == len(points) and point_ids == set(range(len(points)))
                     and len(times) == len(points) and len(set(times)) == 1)
            rows[product].append({"repeat": repeat,
                                  "scan_seconds": times[0] if valid else None,
                                  "duplicated_point_rows": len(matching),
                                  "point_indices_complete": point_ids == set(range(len(points))),
                                  "unique_timing_count": len(set(times)),
                                  "scan_timing_consistent": len(times) == len(points) and len(set(times)) == 1,
                                  "accuracy_passed": valid and all(row.get("accuracy_passed") is True for row in matching),
                                  "available": valid})
    return rows, {"available": True, "path": str(source_json.relative_to(HERE)),
                  "sha256": _sha(source_json), "runner_sha256": source.get("runner_sha256"),
                  "timing_semantics": "frozen per-product native solve across all 8 points and K initials"}


def _pack_output_values(values_by_point: dict[int, Any]) -> dict[str, np.ndarray]:
    """Flatten joint outputs into numeric NPZ members (never object arrays)."""
    packed: dict[str, np.ndarray] = {}
    for index, value in values_by_point.items():
        if isinstance(value, dict):
            packed[f"p{index}_rho"] = np.asarray(value["rho"])
            packed[f"p{index}_photon_integral"] = np.asarray(value["photon_integral"])
        else:
            packed[f"p{index}"] = np.asarray(value)
    return packed


def run_fixture(name: str, tag: str, source_tag: str = SOURCE_TAG) -> Path:
    out = OUTROOT / tag / name
    out.mkdir(parents=True, exist_ok=False)
    points = _points(False)
    rows = []
    memory_artifacts: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        fixture = build_fixture(name)
        batches = _batches(fixture)
        batch_matches = [(batch, vals) for batch, vals in batches.items() if "all_model" in batch]
        if len(batch_matches) != 1:
            raise RuntimeError(f"expected one all-model initial batch, found {[b for b, _ in batch_matches]}")
        batch_name, (rho_batch, initial_indices) = batch_matches[0]
        reference_guard = _resource_snapshot()
        _resource_guard(reference_guard, f"{name} reference pre")
        reference, reference_meta = _load_or_build_reference(fixture, batch_name, rho_batch, source_tag, out)
        _resource_guard(_resource_snapshot(), f"{name} reference post")
        expected = _expected_points(fixture, reference)
        source_dir = OLD_ROOT / source_tag / name
        native_rows, native_meta = _native_rows(source_dir, batch_name, WORKERS[0], fixture.metadata, points)
        for workers in WORKERS:
            before = _resource_snapshot()
            _resource_guard(before, f"{name} workers={workers} pre")
            dense = _dense_output_sweeps(name, rho_batch, points, workers, tuple(PRODUCTS), out,
                                         memory_artifacts)
            after = _resource_snapshot()
            _resource_guard(after, f"{name} workers={workers} post")
            native_rows, native_meta = _native_rows(source_dir, batch_name, workers, fixture.metadata, points)
            for product_entry in dense["products"]:
                product = product_entry["product"]
                for repeat_index, repeat in enumerate(product_entry["repeats"]):
                    per_point_metrics = []
                    for point_index in range(len(points)):
                        actual = repeat["values"][point_index]
                        want = {key: val[point_index] for key, val in expected[product].items()} \
                            if isinstance(expected[product], dict) else expected[product][point_index]
                        per_point_metrics.append(accuracy(product, actual, want))
                    passed = all(metric["accuracy_passed"] for metric in per_point_metrics)
                    output_path = out / f"d_{len(rows):04d}.npz"
                    packed_outputs = _pack_output_values(repeat["values"])
                    np.savez_compressed(output_path, **packed_outputs)
                    native_repeat = next((item for item in native_rows.get(product, [])
                                          if item["repeat"] == repeat_index), None)
                    setup_sum = sum(float(item["setup_wall_seconds"])
                                    for item in dense["setup_by_point"].values())
                    row = {"fixture": name, "batch": batch_name, "initial_count": len(initial_indices),
                           "product": product, "workers": workers, "repeat": repeat_index,
                           "dense_warm_output_scan_seconds": repeat["scan_seconds"],
                           "dense_cold_output_scan_estimate_seconds":
                               dense["preparation_scan_seconds"] + repeat["scan_seconds"],
                           "dense_factorization_only_parallel_scan_seconds": dense["preparation_scan_seconds"],
                           "dense_setup_wall_seconds_sum_over_points": setup_sum,
                           "dense_per_point_output_seconds": repeat["per_point_product_seconds"],
                           "dense_process_startup_seconds": dense["startup_seconds"],
                           "dense_worker_pids": dense["worker_pids"],
                           "dense_memory": repeat["memory"],
                           "accuracy_passed": passed, "per_point_accuracy": per_point_metrics,
                           "native_forward_frozen_scan": native_repeat,
                           "native_forward_source": native_meta,
                           "reference_source": reference_meta,
                           "resource_before": before, "resource_after": after,
                           "output_path": output_path.name, "output_sha256": _sha(output_path),
                           "status": "passed" if passed else "accuracy_failed"}
                    rows.append(row)
                    _write_json(out / f"c_{len(rows):04d}.json", row)
    result = {"schema_version": 1, "complete": True, "fixture": name, "tag": tag,
              "source_tag": source_tag, "batch": batch_name, "initial_count": len(initial_indices),
              "points": points, "products": PRODUCTS, "workers": WORKERS, "repeats": REPEATS,
              "kernel": KERNEL, "solver": SOLVER, "rtol": RTOL, "atol": ATOL,
              "reference_source": reference_meta, "native_forward_source": native_meta,
              "memory_sample_artifacts": memory_artifacts, "rows": rows,
              "runner_sha256": _sha(Path(__file__).resolve()),
              "fixture_builder_sha256": _sha(HERE / "cross_system_fixtures.py"),
              "native_extension_sha256": fixture.metadata.get("native_extension_sha256")}
    path = out / "result.json"
    _write_json(path, result)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=STATIC_FIXTURES, required=True)
    parser.add_argument("--tag", default="focused_v1")
    parser.add_argument("--source-tag", default=SOURCE_TAG)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"plan_only": True, "fixture": args.fixture, "points": _points(False),
                          "products": PRODUCTS, "workers": WORKERS, "repeats": REPEATS,
                          "uses_saved_native_rows": True, "source_tag": args.source_tag}, indent=2))
        return 0
    print(run_fixture(args.fixture, args.tag, args.source_tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
