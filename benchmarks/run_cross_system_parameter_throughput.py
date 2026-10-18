"""Matched parameter-point throughput screen: native forward batches vs scalar adjoints.

No physical work is done on import. Run only after the reviewed kernel is selected.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf import hamiltonian
from centrex_tlf.lindblad.batch import solve_lindblad_batch
from centrex_tlf.lindblad.ir import encode_runtime_value

try:
    from cross_system_fixtures import build_fixture
    from cross_system_methods import evaluate_rc_adjoint, evaluate_static_adjoint
except ModuleNotFoundError:
    from benchmarks.cross_system_fixtures import build_fixture
    from benchmarks.cross_system_methods import evaluate_rc_adjoint, evaluate_static_adjoint
try:
    from run_cross_system_batch_amortization import accuracy, state_batches
except ModuleNotFoundError:
    from benchmarks.run_cross_system_batch_amortization import accuracy, state_batches

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "rc_propagation_results" / "cross_system_parameter_throughput_v1"
FIXTURES = ("r2_f4", "r3_f1_7o2_f4", "r3_f1_9o2_f4", "rc")
PRODUCTS = ("photon_integral", "selected_1", "selected_4", "group_sum")
WORKERS = (1, 3, 8)
REPEATS = 3
RTOL = 1e-9
ATOL = 1e-11
GAMMA = float(hamiltonian.Γ)
CPU_LIMIT = 50.0
RABI_DETUNING_POINTS = (
    (1.0, 0.0),
    (1.0, -20.0), (1.0, 20.0),
    (1.0, -100.0), (1.0, 100.0),
    (0.3, 0.0), (3.0, 0.0),
    (0.3, -100.0),
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _resource() -> dict[str, Any]:
    import psutil
    values = [float(psutil.cpu_percent(interval=1.0)) for _ in range(3)]
    current = psutil.Process()
    exempt = {os.getpid(), *(p.pid for p in current.parents())}
    known = ("run_cross_system_parameter_throughput.py", "run_rc_cross_system_screen.py",
             "run_cross_system_regime_screen.py", "compare_rc_kernel_matrix.py",
             "bench_rc_propagation.py", "simulate_all.py")
    competing = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            command = " ".join(proc.info.get("cmdline") or []).lower()
            if proc.info["pid"] not in exempt and "python" in name and any(x in command for x in known):
                competing.append({"pid": proc.info["pid"], "command": command[:300]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"cpu_percent_samples": values, "cpu_median_percent": float(np.median(values)),
            "available_memory_bytes": int(psutil.virtual_memory().available),
            "competing_python_processes": competing}


def _guard(sample: dict[str, Any], where: str) -> None:
    if sample["cpu_median_percent"] > CPU_LIMIT or sample["competing_python_processes"]:
        raise RuntimeError(f"{where} resource guard failed: {sample}")


def _batches(fixture: Any) -> dict[str, tuple[np.ndarray, list[int]]]:
    return {name: (rho, indices) for name, (rho, indices, _label) in state_batches(fixture).items()
            if "driven" in name or "all_model" in name}


def _points(is_rc: bool = False) -> list[dict[str, float]]:
    return [{"rabi": float(rabi * GAMMA), "detuning": float(2 * np.pi * delta_mhz * 1e6),
             "optical_rabi_over_gamma": float(rabi), "detuning_mhz": float(delta_mhz),
             **({"microwave_rabi": GAMMA} if is_rc else {})}
            for rabi, delta_mhz in RABI_DETUNING_POINTS]


def _product_indices(fixture: Any, product: str) -> list[tuple[int, int]] | None:
    if product == "selected_1":
        return [(int(i), int(i)) for i in fixture.selected_indices_1]
    if product == "selected_4":
        return [(int(i), int(i)) for i in fixture.selected_indices_4]
    if product == "group_sum":
        return [(int(i), int(i)) for i in fixture.group_indices]
    return None


def _rc_parameter_row(point: dict[str, float], sw1: bool, sw2: bool,
                      phase_offset: float) -> np.ndarray:
    return np.asarray([
        0.0, phase_offset if sw1 else -phase_offset,
        0.0, phase_offset if sw2 else -phase_offset,
        point["rabi"], point["detuning"],
        point["microwave_rabi"], point["microwave_rabi"],
    ], dtype=np.complex128)


def _forward_points(fixture: Any, rho: np.ndarray, points: list[dict[str, float]],
                    product: str, kernel: str, solver: str, workers: int,
                    *, rtol: float = RTOL, atol: float = ATOL) -> tuple[np.ndarray, float, dict[str, Any]]:
    k = len(rho)
    repeated_rho = np.tile(rho, (len(points), 1, 1, 1)).reshape(
        len(points) * len(rho), *rho.shape[1:])
    if fixture.rc_case is not None:
        return _forward_rc_points(fixture, rho, points, product, kernel, solver, workers,
                                  rtol=rtol, atol=atol)
    slots = ["rabi", "detuning"]
    point_values = np.asarray([[point[slot] for slot in slots] for point in points], dtype=np.complex128)
    parameter_batch = np.repeat(point_values, k, axis=0)
    output = "selected" if product in {"selected_1", "selected_4", "group_sum"} else product
    kwargs: dict[str, Any] = {
        "solver": solver, "execution_mode": kernel, "reltol": rtol, "abstol": atol,
        "dt": float(fixture.spec.dt), "output": output, "output_when": "final",
        "collect_stats": True, "parallel": workers > 1, "threads": workers,
    }
    indices = _product_indices(fixture, product)
    if indices is not None:
        kwargs["output_indices"] = indices
    if product in {"photon_integral", "full_and_integral"}:
        kwargs["integral_weights"] = [(int(i), GAMMA) for i in fixture.excited_indices]
    started = time.perf_counter()
    result = solve_lindblad_batch(
        fixture.prepared, repeated_rho, fixture.spec.t_span,
        parameter_batch=parameter_batch, parameter_slots=slots, **kwargs,
    )
    values = np.asarray(result.values)
    if values.ndim == 1:
        values = values[:, None]
    values = values.reshape(len(points), k, -1)
    if product == "group_sum":
        values = np.sum(values, axis=-1, keepdims=True)
    return values, time.perf_counter() - started, result.solver_stats or {}


def _forward_rc_points(
    fixture: Any, rho_batch: np.ndarray, points: list[dict[str, float]], product: str,
    kernel: str, solver: str, workers: int, *, rtol: float, atol: float,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Exact absolute-time switch handoff with runtime override per point/state."""
    rc = importlib.import_module("benchmarks.bench_rc_propagation")
    case = fixture.rc_case
    k = len(rho_batch)
    n_points = len(points)
    n = rho_batch.shape[-1]
    slot_names = ["ω1", "φ1", "ω2", "φ2", "Ωt0", "δ0", "Ω1", "Ω2"]
    states = np.tile(np.asarray(rho_batch, dtype=np.complex128), (n_points, 1, 1, 1)).reshape(
        n_points * k, n, n)
    photon = np.zeros((n_points * k,), dtype=np.float64)
    edges = rc.switch_boundaries(*fixture.spec.t_span)
    stats = []
    started = time.perf_counter()
    for segment_index, (lo, hi) in enumerate(pairwise(edges)):
        last = segment_index == len(edges) - 2
        sw1, sw2 = rc.switch_values(0.5 * (float(lo) + float(hi)))
        params_by_point = np.asarray([
            _rc_parameter_row(point, sw1, sw2, rc.SWITCH_PHASE_OFFSET_RAD)
            for point in points
        ], dtype=np.complex128)
        parameter_batch = np.repeat(params_by_point, k, axis=0)
        if product in {"photon_integral", "full_and_integral"}:
            out_name = "full_and_integral"
            weights = [(int(i), GAMMA) for i in fixture.excited_indices]
        elif not last or product == "full":
            out_name, weights = "full", None
        else:
            out_name, weights = "selected", None
        kwargs: dict[str, Any] = {
            "parameter_batch": parameter_batch, "parameter_slots": slot_names,
            "solver": solver, "execution_mode": kernel, "reltol": rtol, "abstol": atol,
            "dt": min(float(fixture.spec.dt), float(hi - lo)), "output": out_name,
            "output_when": "final", "collect_stats": True, "parallel": workers > 1,
            "threads": workers,
        }
        if weights is not None:
            kwargs["integral_weights"] = weights
        if out_name == "selected":
            kwargs["output_indices"] = _product_indices(fixture, product)
        result = solve_lindblad_batch(case.prepared_segmented, states, (float(lo), float(hi)), **kwargs)
        raw = np.asarray(result.values)
        if raw.ndim == 1:
            raw = raw[:, None]
        stats.append(result.solver_stats or {})
        if out_name == "full_and_integral":
            plen = case.prepared_segmented.layout.packed_len
            states = np.stack([case.prepared_segmented.layout.unpack(row[:plen]) for row in raw])
            photon += raw[:, plen].real
        elif out_name == "full":
            states = np.stack([case.prepared_segmented.layout.unpack(row) for row in raw])
        else:
            selected = raw
    if product == "full_and_integral":
        values = {"rho": states.reshape(n_points, k, n, n),
                  "photon_integral": photon.reshape(n_points, k, 1)}
    elif product == "full":
        values = states.reshape(n_points, k, n, n)
    elif product == "photon_integral":
        values = photon.reshape(n_points, k, 1)
    else:
        values = selected.reshape(n_points, k, -1)
        if product == "group_sum":
            values = np.sum(values, axis=-1, keepdims=True)
    return values, time.perf_counter() - started, stats


def _fixture_for_point(fixture: Any, point: dict[str, float]) -> Any:
    if fixture.rc_case is not None:
        rc = importlib.import_module("benchmarks.bench_rc_propagation")
        case = fixture.rc_case
        config = replace(
            case.config, optical_rabi_over_gamma=point["rabi"] / GAMMA,
            detuning_mhz=point["detuning"] / (2 * np.pi * 1e6),
            microwave_rabi_over_gamma=point["microwave_rabi"] / GAMMA,
        )
        params = rc._params_for(
            case.selectors, config, dynamic_switches=True, dynamic_eom=True,
            amplitude_envelope=case.amplitude_envelope,
        )
        prepared = rc._prepare(case.system, params)
        adjusted_case = rc.RCCase(config, case.amplitude_envelope, case.system,
                                  case.selectors, params, prepared)
        rc._initial_labels(adjusted_case)
        rc._prepare_segmented_plan(adjusted_case)
        return replace(fixture, prepared=prepared, rc_case=adjusted_case)
    prepared = _prepared_with_overrides(
        fixture.prepared, {"rabi": point["rabi"], "detuning": point["detuning"]})
    return replace(fixture, prepared=prepared)


def _prepared_with_overrides(prepared_source: Any, point: dict[str, float]) -> Any:
    prepared = copy.copy(prepared_source)
    graph = copy.deepcopy(prepared_source.parameter_graph)
    names = list(graph["slot_names"])
    for name, value in point.items():
        idx = names.index(name)
        if idx >= int(graph["n_base"]):
            raise ValueError(f"adjoint parameter override {name!r} is not a base slot")
        graph["base_values"][idx] = encode_runtime_value(complex(value))
    prepared.parameter_graph = graph
    return prepared


def _adjoint_points(executor: Any, rho: np.ndarray, points: list[dict[str, float]],
                    product: str) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    started = time.perf_counter()
    tasks = [(point, product, rho) for point in points]
    rows = list(executor.map(_adjoint_worker_evaluate, tasks, chunksize=1))
    output = np.stack([row[0] for row in rows], axis=0)
    return output, time.perf_counter() - started, [row[1] for row in rows]


_WORKER_FIXTURE: Any = None
_WORKER_POOL_LIMIT: Any = None


def _init_adjoint_worker(fixture_name: str, ready_queue: Any) -> None:
    global _WORKER_FIXTURE, _WORKER_POOL_LIMIT
    _WORKER_POOL_LIMIT = threadpool_limits(limits=1)
    _WORKER_FIXTURE = build_fixture(fixture_name)
    prototype = importlib.import_module("benchmarks.rc_block_adjoint_prototypes")
    prototype.warm_numba_kernels()
    ready_queue.put({"pid": os.getpid(), "fixture": fixture_name,
                     "state_count": len(_WORKER_FIXTURE.system.QN),
                     "native_sha256": _WORKER_FIXTURE.metadata.get("rust_extension_sha256")})


def _adjoint_worker_ping(_: int) -> tuple[int, float]:
    return os.getpid(), float(_WORKER_FIXTURE.spec.dt)


def _adjoint_worker_evaluate(task: tuple[dict[str, float], str, np.ndarray]
                             ) -> tuple[np.ndarray, dict[str, Any]]:
    point, product, rho_batch = task
    case = _fixture_for_point(_WORKER_FIXTURE, point)
    evaluator = evaluate_rc_adjoint if case.rc_case is not None else evaluate_static_adjoint
    value, stats = evaluator(case, product, rho0_batch=rho_batch, solver="DOP853",
                             reltol=1e-8, atol=1e-10, dt=float(case.spec.dt))
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    return array, stats


def _reference(fixture: Any, rho: np.ndarray, points: list[dict[str, float]], kernel: str) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs = {}
    for solver in ("dopri5", "tsit5"):
        values, _, _ = _forward_points(fixture, rho, points, "full_and_integral",
                                        "structured", solver, 1, rtol=1e-10, atol=1e-12)
        if isinstance(values, dict):
            outputs[solver] = {"full": np.asarray(values["rho"]),
                               "photon": np.asarray(values["photon_integral"]).reshape(
                                   len(points), len(rho))}
        else:
            # Combined native output is packed Hermitian state plus final photon scalar.
            packed_len = fixture.prepared.layout.packed_len
            rho_rows = []
            photons = []
            for row in values.reshape(-1, values.shape[-1]):
                rho_rows.append(fixture.prepared.layout.unpack(row[:packed_len]))
                photons.append(float(row[packed_len]))
            outputs[solver] = {"full": np.stack(rho_rows).reshape(len(points), len(rho), *rho.shape[1:]),
                               "photon": np.asarray(photons).reshape(len(points), len(rho))}
    delta_rho = float(np.max(np.abs(outputs["dopri5"]["full"] - outputs["tsit5"]["full"])))
    delta_photon = float(np.max(np.abs(outputs["dopri5"]["photon"] - outputs["tsit5"]["photon"])))
    if delta_rho > 1e-8 or delta_photon > 1e-8:
        raise RuntimeError(f"parameter-point references disagree: rho={delta_rho:g} photon={delta_photon:g}")
    checks = []
    for matrix in outputs["tsit5"]["full"].reshape(-1, *rho.shape[1:]):
        trace = float(abs(np.trace(matrix) - 1.0))
        herm = float(np.max(np.abs(matrix - matrix.conj().T)))
        mineig = float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min())
        checks.append({"trace_abs_error": trace, "hermiticity_max_abs": herm,
                       "minimum_eigenvalue": mineig})
        if trace > 1e-9 or herm > 1e-9 or mineig < -1e-8:
            raise RuntimeError(f"reference physical gate failed: {checks[-1]}")
    return outputs, {"cross_solver_density_max_abs": delta_rho,
                               "cross_solver_photon_max_abs": delta_photon,
                               "physical_checks": checks,
                               "reference_solvers": ["dopri5", "tsit5"],
                               "rtol": 1e-10, "atol": 1e-12}


def _expected(product: str, reference: dict[str, np.ndarray], fixture: Any) -> np.ndarray:
    if product == "photon_integral":
        return reference["photon"][..., None]
    diagonal = np.real(np.diagonal(reference["full"], axis1=-2, axis2=-1))
    if product == "selected_1":
        return diagonal[..., list(fixture.selected_indices_1)]
    if product == "selected_4":
        return diagonal[..., list(fixture.selected_indices_4)]
    if product == "group_sum":
        return np.sum(diagonal[..., list(fixture.group_indices)], axis=-1, keepdims=True)
    raise ValueError(product)


def _score(product: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    items = [accuracy(product, a, e) for a, e in zip(actual, expected, strict=True)]
    return {"accuracy_passed": all(item["accuracy_passed"] for item in items),
            "per_parameter_point": items}


def run_fixture(name: str, kernel: str, tag: str, solver: str) -> Path:
    out = OUTROOT / tag / name
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite output: {out}")
    before = _resource()
    _guard(before, "pre-run")
    with threadpool_limits(limits=1):
        build_started = time.perf_counter()
        fixture = build_fixture(name)
        build_seconds = time.perf_counter() - build_started
        fixture.prepared.check_execution_mode(kernel)
        points = _points(fixture.rc_case is not None)
        cases = _batches(fixture)
        rows: list[dict[str, Any]] = []
        refs_dir = out / "references"
        refs_dir.mkdir()
        references: dict[str, np.ndarray] = {}
        reference_checks_by_batch: dict[str, Any] = {}
        reference_artifacts: dict[str, dict[str, str]] = {}
        for batch_name, (rho, _indices) in cases.items():
            reference, ref_checks = _reference(fixture, rho, points, kernel)
            reference_checks_by_batch[batch_name] = ref_checks
            for ref_solver in ("dopri5", "tsit5"):
                references[f"{batch_name}__{ref_solver}__rho"] = reference[ref_solver]["full"]
                references[f"{batch_name}__{ref_solver}__photons"] = reference[ref_solver]["photon"]
            ref_path = refs_dir / f"r_{len(reference_artifacts):03d}.npz"
            np.savez_compressed(ref_path,
                dopri5_rho=reference["dopri5"]["full"], dopri5_photon=reference["dopri5"]["photon"],
                tsit5_rho=reference["tsit5"]["full"], tsit5_photon=reference["tsit5"]["photon"])
            reference_artifacts[batch_name] = {
                "path": str(ref_path.relative_to(out)), "sha256": _sha(ref_path),
            }

        pool_setup: dict[str, Any] = {}
        for workers in WORKERS:
            pool_started = time.perf_counter()
            mp_context = multiprocessing.get_context("spawn")
            ready_queue = mp_context.Queue()
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp_context,
                initializer=_init_adjoint_worker, initargs=(name, ready_queue),
            ) as pool:
                # Pings trigger startup; only initializer records establish readiness.
                list(pool.map(_adjoint_worker_ping, range(workers * 2), chunksize=1))
                ready_records = {}
                deadline = time.monotonic() + 180.0
                while len(ready_records) < workers:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"only {len(ready_records)} of {workers} workers initialized")
                    record = ready_queue.get(timeout=remaining)
                    ready_records[int(record["pid"])] = record
                ready_queue.close()
                pids = sorted(ready_records)
                pool_setup[str(workers)] = {
                    "startup_and_fixture_build_seconds": time.perf_counter() - pool_started,
                    "worker_pids": pids, "requested_worker_count": workers,
                    "observed_worker_count": len(pids),
                }
                if len(pids) < workers:
                    raise RuntimeError(f"adjoint pool started {len(pids)} of {workers} workers")
                for batch_name, (rho, indices) in cases.items():
                    reference = {
                        "full": references[f"{batch_name}__tsit5__rho"],
                        "photon": references[f"{batch_name}__tsit5__photons"],
                    }
                    batch_checks = reference_checks_by_batch[batch_name]
                    for product in PRODUCTS:
                        expected = _expected(product, reference, fixture)
                        for repeat in range(REPEATS):
                            order = (("forward", "adjoint") if (workers + repeat) % 2 == 0
                                     else ("adjoint", "forward"))
                            for method in order:
                                try:
                                    if method == "forward":
                                        actual, wall, stats = _forward_points(
                                            fixture, rho, points, product, kernel, solver, workers)
                                    else:
                                        actual, wall, stats = _adjoint_points(
                                            pool, rho, points, product)
                                    gates = _score(product, actual, expected)
                                    row = {
                                        "fixture": name, "batch": batch_name,
                                        "initial_count": len(indices), "product": product,
                                        "method": method, "kernel": kernel, "workers": workers,
                                        "repeat": repeat,
                                        "solver": solver if method == "forward" else "DOP853",
                                        "rtol": RTOL if method == "forward" else 1e-8,
                                        "atol": ATOL if method == "forward" else 1e-10,
                                        "wall_seconds": wall, "accuracy": gates,
                                        "accuracy_passed": gates["accuracy_passed"],
                                        "status": "passed" if gates["accuracy_passed"] else "accuracy_failed",
                                        "solver_details": stats, "parameter_points": points,
                                        "worker_pool_setup": pool_setup[str(workers)],
                                        "reference_checks": batch_checks,
                                    }
                                    output_path = out / f"o_{len(rows) + 1:05d}.npz"
                                    if output_path.exists():
                                        raise FileExistsError(output_path)
                                    np.savez_compressed(output_path, values=actual)
                                    row["output_path"] = str(output_path.relative_to(out))
                                    row["output_sha256"] = _sha(output_path)
                                    rows.append(row)
                                except Exception as exc:
                                    rows.append({
                                        "fixture": name, "batch": batch_name, "product": product,
                                        "method": method, "workers": workers, "repeat": repeat,
                                        "status": "failed",
                                        "reason": f"{type(exc).__name__}: {exc}",
                                        "worker_pool_setup": pool_setup[str(workers)],
                                    })
                                checkpoint = out / f"p_{len(rows):05d}.json"
                                checkpoint.write_text(json.dumps({
                                    "status": "running", "complete": False,
                                    "rows_completed": len(rows), "rows": rows,
                                    "worker_pool_setup": pool_setup,
                                    "reference_artifacts": reference_artifacts,
                                }, indent=2, default=_json), encoding="utf-8")
    after = _resource()
    _guard(after, "post-run")
    refs_path = out / "references.npz"
    np.savez_compressed(refs_path, **references)
    native = importlib.import_module("centrex_tlf.centrex_tlf_rust")
    native_path = Path(native.__file__).resolve()
    expected_rows = len(cases) * len(PRODUCTS) * len(WORKERS) * REPEATS * 2
    complete = len(rows) == expected_rows
    payload = {"schema_version": 1, "status": "complete" if complete else "incomplete",
               "complete": complete,
               "fixture": name, "tag": tag, "kernel": kernel,
               "worker_budgets": list(WORKERS), "repeats": REPEATS, "products": list(PRODUCTS),
               "parameter_points": points, "t_span_s": list(fixture.spec.t_span),
               "rtol": RTOL, "atol": ATOL, "reference_checks_by_batch": reference_checks_by_batch,
               "expected_rows": expected_rows, "resolved_rows": len(rows),
               "accuracy_passed_rows": sum(row.get("accuracy_passed") is True for row in rows),
               "failed_rows": sum(row.get("status") == "failed" for row in rows),
               "rows": rows, "fixture_build_seconds": build_seconds,
               "resource_before": before, "resource_after": after,
               "native_extension_path": str(native_path), "native_extension_sha256": _sha(native_path),
               "runner_sha256": _sha(Path(__file__).resolve()),
               "fixture_builder_sha256": _sha(HERE / "cross_system_fixtures.py"),
               "methods_sha256": _sha(HERE / "cross_system_methods.py"),
               "accuracy_helper_sha256": _sha(HERE / "run_cross_system_batch_amortization.py"),
               "python": sys.version, "platform": platform.platform(),
               "git_revision": __import__("subprocess").run(
                   ["git", "rev-parse", "HEAD"], cwd=HERE.parent, capture_output=True, text=True).stdout.strip()}
    result = out / "result.json"
    result.write_text(json.dumps(payload, indent=2, default=_json), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=FIXTURES, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--solver", choices=("dopri5", "tsit5"), default="tsit5")
    parser.add_argument("--tag", default="throughput_v1")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"plan_only": True, "fixture": args.fixture, "kernel": args.kernel,
                          "products": PRODUCTS, "workers": WORKERS,
                          "parameter_points": _points(args.fixture == "rc")}, indent=2))
        return 0
    run_fixture(args.fixture, args.kernel, args.tag, args.solver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
