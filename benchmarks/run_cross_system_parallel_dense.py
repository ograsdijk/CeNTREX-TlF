"""Matched static dense-eig versus native-forward parameter throughput.

This is benchmark infrastructure only. A physical run requires ``--run`` and
an explicit, root-reviewed invocation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

try:
    from cross_system_fixtures import build_fixture
    from cross_system_methods import evaluate_static_dense
    from run_cross_system_batch_amortization import PRODUCTS, accuracy
    from run_cross_system_parameter_throughput import (
        FIXTURES,
        WORKERS,
        _batches,
        _forward_points,
        _points,
        _reference,
        _sha,
    )

    from centrex_tlf.lindblad.dense import prepare_dense_lindblad_propagator
except ModuleNotFoundError:
    from benchmarks.cross_system_fixtures import build_fixture
    from benchmarks.cross_system_methods import evaluate_static_dense
    from benchmarks.run_cross_system_batch_amortization import PRODUCTS, accuracy
    from benchmarks.run_cross_system_parameter_throughput import (
        FIXTURES,
        WORKERS,
        _batches,
        _forward_points,
        _points,
        _reference,
        _sha,
    )
    from centrex_tlf.lindblad.dense import prepare_dense_lindblad_propagator

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "rc_propagation_results" / "cross_system_parallel_dense_v1"
STATIC_FIXTURES = tuple(name for name in FIXTURES if name != "rc")
REPEATS = 3
KERNEL = "experimental_structured_sparse_dense_diag_soa_single_unpack"
SOLVER = "tsit5"
RTOL, ATOL = 1e-9, 1e-11
SAMPLE_INTERVAL_S = 0.05


class TreeMemorySampler:
    """Sample summed parent+descendant RSS/private bytes during a timed block."""

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.parent = psutil.Process()
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        processes = [self.parent]
        with suppress(psutil.Error):
            processes.extend(self.parent.children(recursive=True))
        rss = private = 0
        private_available = True
        alive = 0
        for process in processes:
            try:
                info = process.memory_info()
                rss += int(info.rss)
                if hasattr(info, "private"):
                    private += int(info.private)
                else:
                    private_available = False
                alive += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.samples.append({"time_monotonic": time.monotonic(), "process_count": alive,
                             "tree_rss_bytes": rss,
                             "tree_private_bytes": private if private_available else None})

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        rss = [s["tree_rss_bytes"] for s in self.samples]
        private = [s["tree_private_bytes"] for s in self.samples if s["tree_private_bytes"] is not None]
        return {"sample_interval_seconds": self.interval_s,
                "sample_count": len(self.samples),
                "tree_rss_start_bytes": rss[0] if rss else None,
                "tree_rss_peak_bytes": max(rss) if rss else None,
                "tree_rss_peak_delta_bytes": max(0, max(rss) - rss[0]) if rss else None,
                "tree_private_start_bytes": private[0] if private else None,
                "tree_private_peak_bytes": max(private) if private else None,
                "tree_private_peak_delta_bytes": max(0, max(private) - private[0]) if private else None,
                }


def _dense_worker(name: str, rho_batch: np.ndarray, task_queue: Any, result_queue: Any) -> None:
    """Persistent point-affine worker; caches each point's factorization locally."""
    _thread_limit = threadpool_limits(limits=1)
    started = time.perf_counter()
    fixture = build_fixture(name)
    build_seconds = time.perf_counter() - started
    cache: dict[int, tuple[Any, dict[str, Any]]] = {}
    result_queue.put({"kind": "ready", "pid": os.getpid(), "build_seconds": build_seconds})
    while True:
        task = task_queue.get()
        if task["kind"] == "stop":
            return
        index = int(task["point_index"])
        if task["kind"] == "cold":
            point = task["point"]
            propagator, setup = _prepare_point(fixture, rho_batch, point)
            cache[index] = (propagator, setup)
            values, walls = _evaluate_all_products(fixture, propagator, rho_batch)
            result_queue.put({"kind": "cold", "point_index": index, "values": values,
                              "walls": walls, "setup": setup, "pid": os.getpid()})
        elif task["kind"] == "warm":
            propagator, setup = cache[index]
            values, walls = _evaluate_all_products(fixture, propagator, rho_batch)
            result_queue.put({"kind": "warm", "point_index": index, "repeat": task["repeat"],
                              "values": values, "walls": walls, "pid": os.getpid()})
        else:
            raise ValueError(f"unknown dense worker task: {task!r}")


def _prepare_point(fixture: Any, rho_batch: np.ndarray, point: dict[str, float]):
    started = time.perf_counter()
    propagator = prepare_dense_lindblad_propagator(
        fixture.prepared, rho_batch,
        parameter_values={"rabi": complex(point["rabi"]),
                          "detuning": complex(point["detuning"])},
        execution_mode=KERNEL, reduce=True)
    setup_wall = time.perf_counter() - started
    setup = dict(propagator.stats)
    setup.update({"setup_wall_seconds": setup_wall,
                  "factorization_seconds": float(propagator.stats.get("factorization_seconds", 0.0)),
                  "backend": "rust_static_dense_eig_cpu_factorization",
                  "initial_count_prepared": len(rho_batch),
                  "runtime_parameter_values": {key: float(value)
                                               for key, value in point.items()
                                               if key in {"rabi", "detuning"}}})
    return propagator, setup


def _evaluate_all_products(fixture: Any, propagator: Any, rho_batch: np.ndarray):
    values, walls = {}, {}
    for product in PRODUCTS:
        value, walls[product] = evaluate_static_dense(
            fixture, propagator, product, rho0_batch=rho_batch)
        if isinstance(value, dict):
            values[product] = {"rho": np.asarray(value.get("rho", value.get("full"))),
                               "photon_integral": np.asarray(value["photon_integral"])}
        else:
            values[product] = np.asarray(value)
    return values, walls


def _as_product(product: str, value: Any, fixture: Any) -> np.ndarray | dict[str, np.ndarray]:
    if product == "full":
        raw = np.asarray(value).reshape(-1, np.asarray(value).shape[-1])
        rho = np.stack([fixture.prepared.layout.unpack(row) for row in raw])
        return rho.reshape(len(value), -1, *rho.shape[-2:])
    if product != "full_and_integral":
        return np.asarray(value)
    if isinstance(value, dict):
        return {"rho": np.asarray(value.get("rho", value.get("full"))),
                "photon_integral": np.asarray(value["photon_integral"])}
    array = np.asarray(value)
    raw = array.reshape(-1, array.shape[-1])
    plen = fixture.prepared.layout.packed_len
    rho = np.stack([fixture.prepared.layout.unpack(row[:plen]) for row in raw])
    return {"rho": rho.reshape(len(array), -1, *rho.shape[-2:]),
            "photon_integral": raw[:, plen:plen + 1].reshape(len(array), -1, 1)}


def _expected_points(fixture: Any, reference: dict[str, np.ndarray]) -> dict[str, Any]:
    rho = reference["full"]
    photon = reference["photon"][..., None]
    diagonal = np.real(np.diagonal(rho, axis1=-2, axis2=-1))
    return {
        "full": rho,
        "photon_integral": photon,
        "populations": diagonal,
        "selected_1": diagonal[..., list(fixture.selected_indices_1)],
        "selected_4": diagonal[..., list(fixture.selected_indices_4)],
        "group_sum": np.sum(diagonal[..., list(fixture.group_indices)], axis=-1, keepdims=True),
        "full_and_integral": {"rho": rho, "photon_integral": photon},
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.stem + ".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)),
                         encoding="utf-8")
    temporary.replace(path)


def _save_memory_samples(out: Path, number: int, sampler: TreeMemorySampler) -> dict[str, str]:
    path = out / f"m_{number:03d}.npz"
    samples = sampler.samples
    private = [np.nan if item["tree_private_bytes"] is None else item["tree_private_bytes"]
               for item in samples]
    np.savez_compressed(
        path, time_monotonic=np.asarray([item["time_monotonic"] for item in samples]),
        process_count=np.asarray([item["process_count"] for item in samples], dtype=np.int32),
        tree_rss_bytes=np.asarray([item["tree_rss_bytes"] for item in samples], dtype=np.int64),
        tree_private_bytes=np.asarray(private, dtype=np.float64),
    )
    return {"path": path.name, "sha256": _sha(path)}


def _resource_snapshot() -> dict[str, Any]:
    cpu = [float(psutil.cpu_percent(interval=0.5)) for _ in range(3)]
    current = psutil.Process()
    exempt = {os.getpid(), *(p.pid for p in current.parents())}
    heavy_names = ("run_cross_system_parallel_dense.py", "run_cross_system_screen.py",
                   "run_cross_system_regime_screen.py", "compare_rc_kernel_matrix.py",
                   "bench_rc_propagation.py", "simulate_all.py")
    competing = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(proc.info.get("cmdline") or []).lower()
            pname = (proc.info.get("name") or "").lower()
            if proc.info["pid"] not in exempt and "python" in pname and any(n in command for n in heavy_names):
                competing.append({"pid": proc.info["pid"], "command": command[:200]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"cpu_percent_samples": cpu, "cpu_median_percent": float(np.median(cpu)),
            "competing_simulations": competing,
            "available_memory_bytes": int(psutil.virtual_memory().available)}


def _resource_guard(snapshot: dict[str, Any], label: str) -> None:
    if snapshot["cpu_median_percent"] > 50 or snapshot["competing_simulations"]:
        raise RuntimeError(f"{label} resource guard failed: {snapshot}")


def _collect_results(result_queue: Any, expected: int) -> list[dict[str, Any]]:
    results = []
    deadline = time.monotonic() + 180.0
    while len(results) < expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"received {len(results)} of {expected} dense worker results")
        results.append(result_queue.get(timeout=remaining))
    return results


def _dense_parallel_sweeps(name: str, rho_batch: np.ndarray, points: list[dict[str, float]],
                           workers: int, out: Path, memory_artifacts: list[dict[str, str]]):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in range(workers)]
    processes = [context.Process(target=_dense_worker,
                                 args=(name, rho_batch, task_queues[i], result_queue))
                 for i in range(workers)]
    try:
        startup = time.perf_counter()
        for process in processes:
            process.start()
        ready = _collect_results(result_queue, workers)
        startup_seconds = time.perf_counter() - startup
        process_ids = {item["pid"] for item in ready}
        if len(process_ids) != workers or any(item.get("kind") != "ready" for item in ready):
            raise RuntimeError(f"dense worker startup mismatch: {ready}")
        point_worker = {i: i % workers for i in range(len(points))}
        with TreeMemorySampler() as sampler:
            started = time.perf_counter()
            for i, point in enumerate(points):
                task_queues[point_worker[i]].put({"kind": "cold", "point_index": i, "point": point})
            cold_results = _collect_results(result_queue, len(points))
            cold_scan_seconds = time.perf_counter() - started
        cold_memory = sampler.summary()
        cold_memory["samples_artifact"] = _save_memory_samples(out, len(memory_artifacts), sampler)
        memory_artifacts.append(cold_memory["samples_artifact"])
        outputs = {int(item["point_index"]): {product: [item["values"][product]] for product in PRODUCTS}
                   for item in cold_results}
        setup_by_point = {int(item["point_index"]): item["setup"] for item in cold_results}
        walls_by_point = {int(item["point_index"]): {product: [item["walls"][product]] for product in PRODUCTS}
                          for item in cold_results}
        warm_scan_seconds, warm_memories = [], []
        for repeat in range(1, REPEATS):
            with TreeMemorySampler() as sampler:
                started = time.perf_counter()
                for i in range(len(points)):
                    task_queues[point_worker[i]].put({"kind": "warm", "point_index": i, "repeat": repeat})
                warm_results = _collect_results(result_queue, len(points))
                warm_scan_seconds.append(time.perf_counter() - started)
            summary = sampler.summary()
            summary["samples_artifact"] = _save_memory_samples(out, len(memory_artifacts), sampler)
            memory_artifacts.append(summary["samples_artifact"])
            warm_memories.append(summary)
            for item in warm_results:
                i = int(item["point_index"])
                for product in PRODUCTS:
                    outputs[i][product].append(item["values"][product])
                    walls_by_point[i][product].append(item["walls"][product])
        return {"outputs": outputs, "setup_by_point": setup_by_point,
                "walls_by_point": walls_by_point, "cold_scan_seconds": cold_scan_seconds,
                "cold_memory": cold_memory, "warm_scan_seconds": warm_scan_seconds,
                "warm_memories": warm_memories, "startup_seconds": startup_seconds,
                "worker_build_seconds": [item["build_seconds"] for item in ready],
                "worker_pids": sorted(process_ids)}
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


def _forward_sweeps(fixture: Any, rho_batch: np.ndarray, points: list[dict[str, float]],
                    kernel: str, workers: int, out: Path,
                    memory_artifacts: list[dict[str, str]]):
    outputs: dict[str, list[np.ndarray | dict[str, np.ndarray]]] = {p: [] for p in PRODUCTS}
    walls: dict[str, list[float]] = {p: [] for p in PRODUCTS}
    scan_seconds, memories = [], []
    for _repeat in range(REPEATS):
        with TreeMemorySampler() as sampler:
            started = time.perf_counter()
            for product in PRODUCTS:
                started_product = time.perf_counter()
                raw, _wall, _stats = _forward_points(
                    fixture, rho_batch, points, product, kernel, SOLVER, workers,
                    rtol=RTOL, atol=ATOL)
                outputs[product].append(_as_product(product, raw, fixture))
                walls[product].append(time.perf_counter() - started_product)
            scan_seconds.append(time.perf_counter() - started)
        summary = sampler.summary()
        summary["samples_artifact"] = _save_memory_samples(out, len(memory_artifacts), sampler)
        memory_artifacts.append(summary["samples_artifact"])
        memories.append(summary)
    return {"outputs": outputs, "walls": walls, "scan_seconds": scan_seconds,
            "memories": memories}


def run_fixture(name: str, tag: str) -> Path:
    out = OUTROOT / tag / name
    out.mkdir(parents=True, exist_ok=False)
    points = _points(False)
    rows: list[dict[str, Any]] = []
    memory_artifacts: list[dict[str, Any]] = []
    references: dict[str, np.ndarray] = {}
    ref_meta: dict[str, Any] = {}
    with threadpool_limits(limits=1):
        fixture = build_fixture(name)
        batches = _batches(fixture)
        for batch_name, (rho_batch, indices) in batches.items():
            if "driven" not in batch_name and "all_model" not in batch_name:
                continue
            reference_guard_before = _resource_snapshot()
            _resource_guard(reference_guard_before, f"{batch_name} reference pre")
            paired_reference, checks = _reference(fixture, rho_batch, points, "structured")
            reference_guard_after = _resource_snapshot()
            _resource_guard(reference_guard_after, f"{batch_name} reference post")
            rho_ref = paired_reference["tsit5"]["full"]
            photon_ref = paired_reference["tsit5"]["photon"]
            references[f"{batch_name}_rho"] = rho_ref
            references[f"{batch_name}_photon"] = photon_ref
            ref_meta[batch_name] = {"physical_checks": checks, "initial_count": len(indices),
                                    "resource_before": reference_guard_before,
                                    "resource_after": reference_guard_after}
            refs = out / f"r_{len(ref_meta)-1:02d}.npz"
            np.savez_compressed(refs, rho=rho_ref, photon=photon_ref)
            ref_meta[batch_name]["reference_path"] = refs.name
            ref_meta[batch_name]["reference_sha256"] = _sha(refs)
            expected = _expected_points(fixture, {"full": rho_ref, "photon": photon_ref})
            for workers in WORKERS:
                before = _resource_snapshot()
                _resource_guard(before, f"{batch_name} workers={workers} pre")
                dense = _dense_parallel_sweeps(
                    name, rho_batch, points, workers, out, memory_artifacts)
                dense_after = _resource_snapshot()
                _resource_guard(dense_after, f"{batch_name} workers={workers} dense post")
                forward = _forward_sweeps(
                    fixture, rho_batch, points, KERNEL, workers, out, memory_artifacts)
                after = _resource_snapshot()
                _resource_guard(after, f"{batch_name} workers={workers} forward post")
                for point_index, point in enumerate(points):
                    for product in PRODUCTS:
                        want = {key: value[point_index] for key, value in expected[product].items()} \
                            if isinstance(expected[product], dict) else expected[product][point_index]
                        for method in ("dense_eig", "native_forward"):
                            for repeat in range(REPEATS):
                                if method == "dense_eig":
                                    actual = dense["outputs"][point_index][product][repeat]
                                    wall = dense["walls_by_point"][point_index][product][repeat]
                                    memory = dense["cold_memory"] if repeat == 0 else dense["warm_memories"][repeat - 1]
                                    setup = dense["setup_by_point"][point_index]
                                    scan_wall = dense["cold_scan_seconds"] if repeat == 0 else dense["warm_scan_seconds"][repeat - 1]
                                else:
                                    actual = forward["outputs"][product][repeat]
                                    if isinstance(actual, dict):
                                        actual = {key: np.asarray(value)[point_index]
                                                  for key, value in actual.items()}
                                    else:
                                        actual = np.asarray(actual)[point_index]
                                    wall = forward["walls"][product][repeat]
                                    memory = forward["memories"][repeat]
                                    setup = None
                                    scan_wall = forward["scan_seconds"][repeat]
                                if product == "full_and_integral":
                                    metrics = accuracy("full_and_integral", actual, want)
                                    passed = bool(metrics["accuracy_passed"])
                                else:
                                    metrics = accuracy(product, actual, want)
                                    passed = bool(metrics["accuracy_passed"])
                                row = {"batch": batch_name, "workers": workers, "point_index": point_index,
                                       "point": point, "product": product, "method": method,
                                       "repeat": repeat, "wall_seconds": wall,
                                       "accuracy": metrics, "accuracy_passed": passed,
                                       "status": "passed" if passed else "accuracy_failed",
                                       "dense_setup": setup, "tree_memory": memory,
                                       "scan_wall_seconds": scan_wall,
                                       "dense_process_startup_seconds": dense["startup_seconds"] if method == "dense_eig" else None,
                                       "dense_worker_fixture_build_seconds": dense["worker_build_seconds"] if method == "dense_eig" else None,
                                       "dense_cold_scan_includes_factorizations": method == "dense_eig" and repeat == 0,
                                       "resource_before": before, "resource_after_dense": dense_after,
                                       "resource_after_forward": after}
                                output = out / f"o_{len(rows):05d}.npz"
                                if isinstance(actual, dict):
                                    np.savez_compressed(output, **actual)
                                else:
                                    np.savez_compressed(output, values=actual)
                                row["output_path"], row["output_sha256"] = output.name, _sha(output)
                                rows.append(row)
                                _write_json(out / f"p_{len(rows):05d}.json", {"complete": False, "rows": rows})
    refpath = out / "refs.npz"
    np.savez_compressed(refpath, **references)
    expected_rows = len(ref_meta) * len(points) * len(WORKERS) * len(PRODUCTS) * REPEATS * 2
    complete = len(rows) == expected_rows
    payload = {"schema_version": 1, "complete": complete,
               "status": "complete" if complete else "incomplete",
               "fixture": name, "tag": tag,
               "kernel": KERNEL, "solver": SOLVER, "rtol": RTOL, "atol": ATOL,
               "points": points, "products": PRODUCTS, "worker_budgets": WORKERS,
               "repeats": REPEATS, "expected_rows": expected_rows, "resolved_rows": len(rows),
               "accuracy_passed_rows": sum(row["accuracy_passed"] for row in rows),
               "reference_meta": ref_meta, "memory_sample_artifacts": memory_artifacts,
               "fixture_metadata": fixture.metadata,
               "rows": rows, "runner_sha256": _sha(Path(__file__).resolve()),
               "fixture_builder_sha256": _sha(HERE / "cross_system_fixtures.py"),
               "native_forward_helper_sha256": _sha(HERE / "run_cross_system_parameter_throughput.py"),
               "dense_helper_sha256": _sha(HERE / "cross_system_methods.py"),
               "reference_archive_sha256": _sha(refpath)}
    result = out / "result.json"
    _write_json(result, payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=STATIC_FIXTURES, required=True)
    parser.add_argument("--tag", default="parallel_dense_v1")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"plan_only": True, "fixture": args.fixture, "points": _points(False),
                          "products": PRODUCTS, "workers": WORKERS, "repeats": REPEATS,
                          "process_tree_memory_sample_interval_seconds": SAMPLE_INTERVAL_S}, indent=2))
        return 0
    print(run_fixture(args.fixture, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
