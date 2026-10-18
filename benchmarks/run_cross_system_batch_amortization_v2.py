"""All-initial-state batch screen for forward, dense and scalar-adjoint methods.

This is benchmark-only infrastructure. It performs no work at import time;
the checker runs an explicit fixture after source review. It measures one
parameter point, not a parameter-scan throughput claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf import hamiltonian, states
from centrex_tlf.lindblad.batch import solve_lindblad_batch
try:
    from cross_system_fixtures import build_fixture
    from cross_system_methods import (
        evaluate_rc_adjoint, evaluate_static_adjoint, evaluate_static_dense,
        prepare_static_dense,
    )
except ModuleNotFoundError:
    from benchmarks.cross_system_fixtures import build_fixture
    from benchmarks.cross_system_methods import (
        evaluate_rc_adjoint, evaluate_static_adjoint, evaluate_static_dense,
        prepare_static_dense,
    )

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "rc_propagation_results" / "cross_system_batch_amortization_v2"
FIXTURES = ("r2_f4", "r3_f1_7o2_f4", "r3_f1_9o2_f4", "rc")
PRODUCTS = ("full", "photon_integral", "populations", "selected_1", "selected_4", "group_sum", "full_and_integral")
WORKER_BUDGETS = (1, 3, 8)
GAMMA = float(hamiltonian.Γ)
CPU_LIMIT = 50.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")
    replace_retry(temp, path)


def replace_retry(source: Path, target: Path) -> None:
    """Bounded Windows antivirus/indexer retry for atomic artifact replacement."""
    import time as _time
    for attempt in range(10):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            _time.sleep(0.05)


def resource_sample() -> dict[str, Any]:
    import psutil
    samples = [float(psutil.cpu_percent(interval=1.0)) for _ in range(3)]
    proc = psutil.Process()
    exempt = {os.getpid(), *(p.pid for p in proc.parents())}
    names = ("run_rc_cross_system_screen.py", "run_cross_system_batch_amortization.py",
             "bench_rc_propagation.py", "compare_rc_kernel_matrix.py", "simulate_all.py")
    competing = []
    for item in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(item.info.get("cmdline") or []).lower()
            if item.info["pid"] not in exempt and "python" in (item.info.get("name") or "").lower() and any(n in cmd for n in names):
                competing.append({"pid": item.info["pid"], "cmdline": cmd[:300]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"cpu_percent_samples": samples, "cpu_median_percent": float(np.median(samples)),
            "available_memory_bytes": int(psutil.virtual_memory().available),
            "competing_python_processes": competing}


def record_checkpoint(out: Path, rows: list[dict[str, Any]]) -> None:
    atomic_json(out / "progress" / f"after_{len(rows):05d}.json", {
        "status": "running", "complete": False, "rows_completed": len(rows), "rows": rows,
    })


def save_output(out: Path, key: str, value: Any) -> tuple[str, str]:
    arrays = value if isinstance(value, dict) else {"output": value}
    output_dir = out / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(output_dir.glob("cell_*.npz")))
    path = output_dir / f"cell_{index:05d}.npz"
    while path.exists():
        index += 1
        path = output_dir / f"cell_{index:05d}.npz"
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp, **{name: np.asarray(array) for name, array in arrays.items()})
    replace_retry(temp, path)
    return str(path.relative_to(OUTROOT)), sha256(path)


def baseline_settings(screen_path: Path, fixture: str, kernel: str) -> dict[str, Any]:
    payload = json.loads(screen_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or payload.get("fixture") != fixture:
        raise ValueError("baseline screen must be complete and match the requested fixture")
    native_path = Path(importlib.import_module("centrex_tlf.centrex_tlf_rust").__file__).resolve()
    native_hash = sha256(native_path)
    if payload.get("native_extension_sha256") != native_hash:
        raise ValueError("baseline screen native extension hash does not match current build")
    solver = payload.get("solver")
    if solver not in {"dopri5", "tsit5"}:
        raise ValueError(f"baseline screen has unsupported solver {solver!r}")
    settings = {}
    for product in PRODUCTS:
        rows = [row for row in payload["rows"] if row.get("kernel") == kernel
                and row.get("product") == product and row.get("accuracy_passed") is True]
        if not rows:
            rows = [row for row in payload["rows"] if row.get("kernel") == "structured"
                    and row.get("product") == product and row.get("accuracy_passed") is True]
        if not rows:
            raise ValueError(f"baseline screen has no accepted tolerance row for {kernel}/{product}")
        row = rows[-1]
        tol = row.get("accepted_tolerance") or {"rtol": payload["reltol"], "atol": payload["abstol"]}
        settings[product] = {"solver": solver, "rtol": float(tol["rtol"]), "atol": float(tol["atol"]),
                             "source_kernel": row.get("kernel")}
    return {"solver": solver, "by_product": settings, "screen_sha256": sha256(screen_path),
            "screen_path": str(screen_path.resolve()), "native_extension_sha256": native_hash}


def state_batches(fixture: Any) -> dict[str, tuple[np.ndarray, list[int], str]]:
    """Return stable K-small, driven-manifold and all-model basis batches."""
    qn = fixture.system.QN
    all_indices = list(range(len(qn)))
    if fixture.rc_case is not None:
        driven = np.asarray(states.QuantumSelector(
            electronic=states.ElectronicState.X, J=2, P=1,
        ).get_indices(qn), dtype=int).ravel().tolist()
        if len(driven) != 20:
            raise RuntimeError(f"RC J=2 initial manifold expected 20 states, got {len(driven)}")
        # Stable deterministic spread over label-sorted all-model states.
        labels = [qn[i].largest.state_string_custom(["electronic", "J", "F1", "F", "mF", "P", "Ω"])
                  for i in all_indices]
        ordered = sorted(all_indices, key=lambda i: labels[i])
        small = ordered[::max(1, len(ordered) // 4)][:4]
        if len(small) < 4:
            small = ordered[:4]
        choices = [("K4_label_spread", small), ("K20_driven_X_J2", driven),
                   ("K70_all_model_states", all_indices)]
    else:
        transition = getattr(importlib.import_module("centrex_tlf.transitions"), fixture.spec.transition_name)
        j = int(transition.J_ground)
        driven = np.asarray(states.QuantumSelector(
            electronic=states.ElectronicState.X, J=j, P=(-1) ** j,
        ).get_indices(qn), dtype=int).ravel().tolist()
        small = list(fixture.selected_indices_4)
        if len(driven) != (20 if j == 2 else 28):
            raise RuntimeError(f"{fixture.spec.name}: expected J={j} manifold size {(20 if j == 2 else 28)}, got {len(driven)}")
        choices = [("K4_label_spread", small), (f"K{len(driven)}_driven_X_J{j}", driven),
                   (f"K{len(all_indices)}_all_model_states", all_indices)]
    result = {}
    for name, indices in choices:
        if len(indices) == 0 or len(set(indices)) != len(indices):
            raise RuntimeError(f"invalid state selection for {name}")
        rho = np.zeros((len(indices), len(qn), len(qn)), dtype=np.complex128)
        rho[np.arange(len(indices)), indices, indices] = 1.0
        result[name] = (rho, indices, "QuantumSelector J manifold" if "driven" in name else "deterministic state-label ordering")
    return result


def initial_labels(fixture: Any, indices: list[int]) -> list[str]:
    return [fixture.system.QN[i].largest.state_string_custom(
        ["electronic", "J", "F1", "F", "mF", "P", "Ω"]
    ) for i in indices]


def native_segmented(fixture: Any, rho_batch: np.ndarray, product: str, kernel: str,
                     solver: str, workers: int, *, rtol: float, atol: float) -> tuple[Any, dict[str, Any]]:
    """Native Rust call, with exact absolute-time RC segment carry if needed."""
    prepared = fixture.prepared
    n = len(fixture.system.QN)
    output_name = product
    selected_indices: list[int] = []
    if product == "selected_1":
        output_name, selected_indices = "selected", [int(fixture.selected_indices_1[0])]
    elif product == "selected_4":
        output_name, selected_indices = "selected", list(map(int, fixture.selected_indices_4))
    elif product == "group_sum":
        output_name, selected_indices = "selected", list(map(int, fixture.group_indices))
    kwargs: dict[str, Any] = {
        "solver": solver, "execution_mode": kernel, "reltol": rtol, "abstol": atol,
        "dt": float(fixture.spec.dt), "output_when": "final", "collect_stats": True,
        "parallel": workers > 1, "threads": workers,
    }
    if output_name in {"photon_integral", "full_and_integral"}:
        kwargs["integral_weights"] = [(int(i), GAMMA) for i in fixture.excited_indices]

    def one_segment(prepared_obj: Any, rho: np.ndarray, span: tuple[float, float],
                    *, params: np.ndarray | None = None, slots: Any = None, output: str) -> Any:
        call_kwargs = dict(kwargs)
        call_kwargs["output"] = output
        if output == "selected":
            call_kwargs["output_indices"] = [(i, i) for i in selected_indices]
        else:
            call_kwargs.pop("output_indices", None)
        if params is not None:
            call_kwargs.update(parameter_batch=params, parameter_slots=slots)
        return solve_lindblad_batch(prepared_obj, rho, span, **call_kwargs)

    if fixture.rc_case is None:
        return one_segment(prepared, rho_batch, tuple(fixture.spec.t_span), output=output_name), {
            "exact_switch_segment_handoff": False, "segment_count": 1,
        }

    try:
        rc = importlib.import_module("bench_rc_propagation")
    except ModuleNotFoundError:
        rc = importlib.import_module("benchmarks.bench_rc_propagation")
    case = fixture.rc_case
    edges = rc.switch_boundaries(*fixture.spec.t_span)
    rho = np.asarray(rho_batch, dtype=np.complex128).copy()
    photon = np.zeros(rho.shape[0], dtype=np.float64)
    last_result = None
    stats = []
    spans = list(zip(edges[:-1], edges[1:], strict=True))
    for index, (lo, hi) in enumerate(spans):
        pol = rc.switch_values(float((lo + hi) / 2.0))
        switch_row = np.asarray([0.0,
            rc.SWITCH_PHASE_OFFSET_RAD if pol[0] else -rc.SWITCH_PHASE_OFFSET_RAD,
            0.0, rc.SWITCH_PHASE_OFFSET_RAD if pol[1] else -rc.SWITCH_PHASE_OFFSET_RAD], dtype=np.complex128)
        params = np.tile(switch_row, (len(rho), 1))
        if product in {"photon_integral", "full_and_integral"}:
            result = one_segment(case.prepared_segmented, rho, (float(lo), float(hi)),
                                 params=params, slots=case.switch_override_slots, output="full_and_integral")
            packed = np.asarray(result.values, dtype=np.float64)
            packed_len = case.prepared_segmented.layout.packed_len
            rho = np.stack([case.prepared_segmented.layout.unpack(row[:packed_len]) for row in packed])
            photon += packed[:, packed_len]
            last_result = result
        else:
            last = index == len(spans) - 1
            target = output_name if last else "full"
            result = one_segment(case.prepared_segmented, rho, (float(lo), float(hi)),
                                 params=params, slots=case.switch_override_slots, output=target)
            if target == "full":
                packed = np.asarray(result.values, dtype=np.float64)
                rho = np.stack([case.prepared_segmented.layout.unpack(row) for row in packed])
            else:
                last_result = result
        stats.append(result.solver_stats or {})
    if product == "photon_integral":
        value: Any = photon[:, None]
    elif product == "full_and_integral":
        value = {"rho": rho, "photon_integral": photon[:, None]}
    elif product == "full":
        value = rho
    else:
        value = np.asarray(last_result.values)
        if product == "group_sum":
            value = np.sum(value, axis=-1, keepdims=True)
    return value, {"exact_switch_segment_handoff": True, "segment_count": len(spans),
                   "solver_stats_by_segment": stats, "native_final_output": output_name}


def normalize_native(value: Any, product: str, layout: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not hasattr(value, "values"):
        return value
    raw = np.asarray(value.values)
    if product == "full":
        packed = np.asarray(raw, dtype=np.float64).reshape(raw.shape[0], -1)
        return np.stack([layout.unpack(row) for row in packed])
    if product == "full_and_integral":
        packed = np.asarray(raw, dtype=np.float64).reshape(raw.shape[0], -1)
        n_packed = layout.packed_len
        return {"rho": np.stack([layout.unpack(row[:n_packed]) for row in packed]),
                "photon_integral": packed[:, n_packed: n_packed + 1]}
    if product in {"selected_1", "selected_4", "group_sum"}:
        values = np.asarray(raw)
        if product == "group_sum":
            values = np.sum(values, axis=-1, keepdims=True)
        return values
    return np.asarray(raw)


def run_native(fixture: Any, rho_batch: np.ndarray, product: str, kernel: str,
               solver: str, workers: int, rtol: float, atol: float) -> tuple[Any, float, dict[str, Any]]:
    started = time.perf_counter()
    result, extra = native_segmented(fixture, rho_batch, product, kernel, solver, workers,
                                     rtol=rtol, atol=atol)
    value = normalize_native(result, product, fixture.prepared.layout)
    if product == "photon_integral" and isinstance(result, dict):
        value = result["value"]
    elapsed = time.perf_counter() - started
    return value, elapsed, extra


def normalize_product(product: str, value: Any, count: int) -> Any:
    if product == "full_and_integral":
        rho = value.get("rho", value.get("full"))
        photons = value.get("photon_integral")
        if rho is None or photons is None:
            raise ValueError("joint output must contain full/rho and photon_integral")
        return {"rho": normalize_product("full", rho, count),
                "photon_integral": normalize_product("photon_integral", photons, count)}
    if product == "full":
        return np.asarray(value).reshape(count, *np.asarray(value).shape[-2:])
    if product in {"photon_integral", "selected_1", "group_sum"}:
        return np.asarray(value).reshape(count, -1)
    if product == "selected_4":
        return np.asarray(value).reshape(count, 4)
    if product == "populations":
        return np.asarray(value).reshape(count, -1)
    return np.asarray(value)


def verify_reference(fixture: Any, rho_batch: np.ndarray, kernel: str, rtol: float,
                     atol: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    outputs = {}
    for solver in ("dopri5", "tsit5"):
        value, wall, extra = run_native(fixture, rho_batch, "full_and_integral", "structured", solver, 1,
                                        rtol, atol)
        outputs[solver] = value
    rho_a, rho_b = outputs["dopri5"]["rho"], outputs["tsit5"]["rho"]
    photon_a, photon_b = outputs["dopri5"]["photon_integral"], outputs["tsit5"]["photon_integral"]
    rho_diff = float(np.max(np.abs(rho_a-rho_b)))
    photon_diff = float(np.max(np.abs(photon_a-photon_b)))
    if rho_diff > 1e-8 or photon_diff > 1e-8:
        raise RuntimeError(f"tight references disagree: density={rho_diff:g}, photons={photon_diff:g}")
    physical = []
    for row in rho_b:
        checks = (abs(np.trace(row)-1.0), np.max(np.abs(row-row.conj().T)),
                  np.linalg.eigvalsh((row+row.conj().T)/2).min())
        physical.append({"trace_abs_error": float(checks[0]), "hermiticity_max_abs": float(checks[1]),
                         "minimum_eigenvalue": float(checks[2])})
        if checks[0] > 1e-9 or checks[1] > 1e-9 or checks[2] < -1e-8:
            raise RuntimeError(f"reference physical gate failed: {physical[-1]}")
    return rho_b, photon_b, {"cross_solver_density_max_abs": rho_diff,
                             "cross_solver_photon_max_abs": photon_diff,
                             "physical_checks": physical,
                             "reference_solvers": ["dopri5", "tsit5"],
                             "reference_tolerance": {"rtol": rtol, "atol": atol}}


def expected_product(product: str, rho_ref: np.ndarray, photon_ref: np.ndarray,
                     fixture: Any) -> Any:
    if product == "full":
        return rho_ref
    if product == "photon_integral":
        return photon_ref
    if product == "full_and_integral":
        return {"rho": rho_ref, "photon_integral": photon_ref}
    pops = np.real(np.diagonal(rho_ref, axis1=1, axis2=2))
    if product == "populations":
        return pops
    if product == "selected_1":
        return pops[:, list(fixture.selected_indices_1)]
    if product == "selected_4":
        return pops[:, list(fixture.selected_indices_4)]
    if product == "group_sum":
        return np.sum(pops[:, list(fixture.group_indices)], axis=1, keepdims=True)
    raise ValueError(product)


def accuracy(product: str, actual: Any, expected: Any) -> dict[str, Any]:
    if product == "full_and_integral":
        rho = accuracy("full", actual["rho"], expected["rho"])
        photon = accuracy("photon_integral", actual["photon_integral"], expected["photon_integral"])
        return {"accuracy_passed": rho["accuracy_passed"] and photon["accuracy_passed"],
                "density": rho, "photon": photon}
    a, e = np.asarray(actual), np.asarray(expected)
    if not np.all(np.isfinite(a)):
        return {"accuracy_passed": False, "reason": "nonfinite"}
    if product == "full":
        err = np.max(np.abs(a-e), axis=(1, 2))
        diag_a = np.diagonal(a, axis1=1, axis2=2)
        diag_e = np.diagonal(e, axis1=1, axis2=2)
        pop_abs = np.abs(np.real(diag_a)-np.real(diag_e))
        mask = np.abs(np.real(diag_e)) >= 1e-6
        rel = np.where(mask, pop_abs / np.maximum(np.abs(np.real(diag_e)), 1e-300), 0.0)
        trace = np.abs(np.trace(a, axis1=1, axis2=2)-1.0)
        herm = np.max(np.abs(a-np.swapaxes(a.conj(),1,2)), axis=(1,2))
        eig = np.asarray([np.linalg.eigvalsh((r+r.conj().T)/2).min() for r in a])
        passed = np.all(err <= 1e-6) and np.all(pop_abs <= 1e-6) and np.all(rel <= 3e-3)
        physical = np.all(trace <= 1e-9) and np.all(herm <= 1e-9) and np.all(eig >= -1e-8)
        return {"accuracy_passed": bool(passed and physical), "max_density_abs_by_initial": err.tolist(),
                "max_population_abs": float(pop_abs.max()), "max_population_relative_above_floor": float(rel.max()),
                "trace_abs_max": float(trace.max()), "hermiticity_abs_max": float(herm.max()),
                "minimum_eigenvalue": float(eig.min()), "physical_passed": bool(physical)}
    absolute = np.abs(a-e)
    mask = np.abs(e) >= 1e-6
    relative = np.zeros_like(absolute, dtype=float)
    relative[mask] = absolute[mask]/np.abs(e[mask])
    imag = float(np.max(np.abs(np.imag(a)))) if np.iscomplexobj(a) else 0.0
    nonnegative = bool(np.all(np.real(a) >= -1e-7))
    if product == "photon_integral":
        passed = np.all(absolute <= 1e-6) and np.all(relative <= 1e-5) and nonnegative
    else:
        passed = np.all(absolute <= 1e-6) and np.all(relative <= 3e-3) and imag <= 1e-9 and nonnegative
    return {"accuracy_passed": bool(passed), "max_abs_by_initial": np.max(absolute, axis=tuple(range(1, absolute.ndim))).tolist(),
            "max_relative_above_floor": float(np.max(relative)), "max_imaginary_abs": imag,
            "nonnegative": nonnegative}


def mixture_validation(fixture: Any, rho_batch: np.ndarray, kernel: str, solver: str,
                       rtol: float, atol: float, rho_ref: np.ndarray, photon_ref: np.ndarray) -> dict[str, Any]:
    weights = np.arange(1, len(rho_batch)+1, dtype=float)
    weights /= weights.sum()
    mixture = np.einsum("k,kij->ij", weights, rho_batch)[None, :, :]
    direct, wall, _ = run_native(fixture, mixture, "full_and_integral", kernel, solver, 1, rtol, atol)
    predicted_rho = np.einsum("k,kij->ij", weights, rho_ref)[None, :, :]
    predicted_photon = np.einsum("k,kj->j", weights, photon_ref)
    return {"weights": weights.tolist(), "direct_validation_wall_seconds": wall,
            "density_linearity_error": float(np.max(np.abs(direct["rho"]-predicted_rho))),
            "photon_linearity_error": float(np.max(np.abs(direct["photon_integral"]-predicted_photon))),
            "density_accuracy": accuracy("full", direct["rho"], predicted_rho),
            "photon_accuracy": accuracy("photon_integral", direct["photon_integral"], predicted_photon),
            "_direct_density": direct["rho"], "_direct_photons": direct["photon_integral"]}


def run_batch_fixture(name: str, tag: str, kernel: str, screen_path: Path) -> Path:
    out = OUTROOT / tag / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}; use a new tag")
    out.mkdir(parents=True, exist_ok=True)
    before = resource_sample()
    if before["cpu_median_percent"] > CPU_LIMIT or before["competing_python_processes"]:
        raise RuntimeError(f"pre-run guard failed: {before}")
    baseline = baseline_settings(screen_path, name, kernel)
    # Native reference and direct batch runs use one BLAS thread throughout;
    # dense evaluation temporarily uses its explicitly recorded worker budget.
    threadpool_limits(limits=1)
    fixture = build_fixture(name)
    cases = state_batches(fixture)
    results = []
    ref_arrays: dict[str, np.ndarray] = {}
    for batch_name, (rho_batch, indices, selection_method) in cases.items():
        # Fine tolerances are references, not a promised performance setting.
        rho_ref, photon_ref, ref_checks = verify_reference(fixture, rho_batch, "structured", 1e-10, 1e-12)
        ref_arrays[f"{batch_name}_rho_reference"] = rho_ref
        ref_arrays[f"{batch_name}_photon_reference"] = photon_ref
        ref_path = out / "references" / f"{batch_name}.npz"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_tmp = ref_path.with_name(ref_path.stem + ".tmp.npz")
        if ref_path.exists():
            raise FileExistsError(ref_path)
        np.savez_compressed(ref_tmp, rho=rho_ref, photon_integral=photon_ref)
        replace_retry(ref_tmp, ref_path)
        ref_artifact = {"path": str(ref_path.relative_to(OUTROOT)), "sha256": sha256(ref_path)}
        labels = initial_labels(fixture, indices)
        mixture_solver = baseline["by_product"]["full_and_integral"]
        mixture = mixture_validation(fixture, rho_batch, kernel, mixture_solver["solver"],
                                     mixture_solver["rtol"], mixture_solver["atol"],
                                     rho_ref, photon_ref)
        for product in PRODUCTS:
            expected = expected_product(product, rho_ref, photon_ref, fixture)
            for workers in WORKER_BUDGETS:
                row: dict[str, Any] = {
                    "fixture": name, "batch": batch_name, "initial_count": len(indices),
                    "product": product, "kernel": kernel, "workers": workers,
                    "solver": baseline["by_product"][product]["solver"],
                    "rtol": baseline["by_product"][product]["rtol"],
                    "atol": baseline["by_product"][product]["atol"],
                    "selection_method": selection_method,
                    "status": "planned", "initial_indices": indices, "initial_labels": labels,
                }
                try:
                    actual, wall, details = run_native(
                        fixture, rho_batch, product, kernel, row["solver"], workers,
                        row["rtol"], row["atol"],
                    )
                    actual = normalize_product(product, actual, len(rho_batch))
                    metrics = accuracy(product, actual, expected)
                    row.update({"wall_seconds": wall, "accuracy": metrics,
                                "initial_tolerance_metrics": metrics,
                                "initial_tolerance_accuracy_passed": bool(metrics["accuracy_passed"]),
                                "accuracy_passed": bool(metrics["accuracy_passed"]),
                                "status": "passed" if metrics["accuracy_passed"] else "accuracy_failed",
                                "solver_details": details})
                    row["output_artifact"], row["output_sha256"] = save_output(
                        out, f"native_{batch_name}_{product}_w{workers}", actual,
                    )
                    if metrics["accuracy_passed"]:
                        row["accepted_metrics"] = metrics
                        row["accepted_wall_seconds"] = wall
                        row["accepted_tolerance"] = {"rtol": row["rtol"], "atol": row["atol"]}
                    else:
                        retry_rows = []
                        for retry_rtol, retry_atol in ((1e-8, 1e-10), (1e-9, 1e-11)):
                            retry_started = time.perf_counter()
                            try:
                                retry_actual, retry_wall, retry_details = run_native(
                                    fixture, rho_batch, product, kernel, row["solver"], workers,
                                    retry_rtol, retry_atol,
                                )
                                retry_actual = normalize_product(product, retry_actual, len(rho_batch))
                                retry_metrics = accuracy(product, retry_actual, expected)
                                retry_record = {
                                    "rtol": retry_rtol, "atol": retry_atol,
                                    "wall_seconds": retry_wall,
                                    "outer_wall_seconds": time.perf_counter() - retry_started,
                                    "accuracy": retry_metrics,
                                    "accuracy_passed": bool(retry_metrics["accuracy_passed"]),
                                    "solver_details": retry_details,
                                }
                                retry_record["output_artifact"], retry_record["output_sha256"] = save_output(
                                    out, f"native_{batch_name}_{product}_w{workers}_retry_{retry_rtol:g}",
                                    retry_actual,
                                )
                                retry_rows.append(retry_record)
                                if retry_record["accuracy_passed"]:
                                    row.update({
                                        "status": "passed_tight_retry", "accuracy_passed": True,
                                        "accepted_metrics": retry_metrics,
                                        "accepted_wall_seconds": retry_wall,
                                        "accepted_tolerance": {"rtol": retry_rtol, "atol": retry_atol},
                                        "accepted_output_artifact": retry_record["output_artifact"],
                                        "accepted_output_sha256": retry_record["output_sha256"],
                                    })
                                    break
                            except Exception as retry_exc:
                                retry_rows.append({"rtol": retry_rtol, "atol": retry_atol,
                                                   "status": "failed",
                                                   "reason": f"{type(retry_exc).__name__}: {retry_exc}"})
                        row["tight_retries"] = retry_rows
                except NotImplementedError as exc:
                    row.update(status="unsupported", reason=str(exc))
                except Exception as exc:
                    row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                results.append(row)
                record_checkpoint(out, results)

        # Reusable dense and adjoint paths apply only to static fixtures and scalar objectives.
        if fixture.rc_case is None:
            try:
                with threadpool_limits(limits=1):
                    dense, dense_setup = prepare_static_dense(
                        fixture, execution_mode=kernel, reduce=True, rho0_batch=rho_batch,
                    )
            except Exception as exc:
                dense = None
                dense_setup = {"error": f"{type(exc).__name__}: {exc}"}
            for product in PRODUCTS:
                for workers in WORKER_BUDGETS:
                    try:
                        if dense is None:
                            results.append({"fixture": name, "batch": batch_name, "product": product,
                                            "method": "dense_eig", "workers": workers,
                                            "status": "failed", "reason": dense_setup["error"]})
                            record_checkpoint(out, results)
                            continue
                        with threadpool_limits(limits=workers):
                            actual, wall = evaluate_static_dense(fixture, dense, product, rho0_batch=rho_batch)
                        actual = normalize_product(product, actual, len(rho_batch))
                        expected = expected_product(product, rho_ref, photon_ref, fixture)
                        metrics = accuracy(product, actual, expected)
                        results.append({"fixture": name, "batch": batch_name, "product": product,
                                        "method": "dense_eig", "workers": workers, "wall_seconds": wall,
                                        "setup_seconds": dense_setup.get("setup_wall_seconds"),
                                        "accuracy": metrics, "accuracy_passed": metrics["accuracy_passed"],
                                        "status": "passed" if metrics["accuracy_passed"] else "accuracy_failed",
                                        "backend": dense_setup.get("backend"), "full_density_reconstruction_timed": True})
                        results[-1]["output_artifact"], results[-1]["output_sha256"] = save_output(
                            out, f"dense_{batch_name}_{product}_w{workers}", actual,
                        )
                    except NotImplementedError as exc:
                        results.append({"fixture": name, "batch": batch_name, "product": product,
                                        "method": "dense_eig", "workers": workers,
                                        "status": "unsupported", "reason": str(exc)})
                    except Exception as exc:
                        results.append({"fixture": name, "batch": batch_name, "product": product,
                                        "method": "dense_eig", "workers": workers,
                                        "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                    record_checkpoint(out, results)
            adjoint_products = ("photon_integral", "selected_1", "selected_4", "group_sum")
            for unsupported in ("full", "populations", "full_and_integral"):
                results.append({"fixture": name, "batch": batch_name, "product": unsupported,
                                "method": "scalar_adjoint_python_prototype", "status": "unsupported",
                                "reason": "scalar adjoint returns observable responses, not density matrices or the full population vector"})
                record_checkpoint(out, results)
            for product in adjoint_products:
                try:
                    actual, stats = evaluate_static_adjoint(
                        fixture, product, rho0_batch=rho_batch, solver="DOP853",
                        reltol=1e-8, atol=1e-10, dt=float(fixture.spec.dt),
                    )
                    actual = normalize_product(product, actual, len(rho_batch))
                    expected = expected_product(product, rho_ref, photon_ref, fixture)
                    metrics = accuracy(product, actual, expected)
                    results.append({"fixture": name, "batch": batch_name, "product": product,
                                    "method": "scalar_adjoint_python_prototype", "workers": 1,
                                    "wall_seconds": stats["wall_seconds"], "setup_seconds": stats.get("rhs_context_setup_seconds", 0.0),
                                    "accuracy": metrics, "accuracy_passed": metrics["accuracy_passed"],
                                    "status": "passed" if metrics["accuracy_passed"] else "accuracy_failed",
                                    "backend": stats["backend"], "full_density_supported": False,
                                    "observable_solve_count": stats["backward_solve_count"]})
                    results[-1]["output_artifact"], results[-1]["output_sha256"] = save_output(
                        out, f"adjoint_{batch_name}_{product}", actual,
                    )
                except Exception as exc:
                    results.append({"fixture": name, "batch": batch_name, "product": product,
                                    "method": "scalar_adjoint_python_prototype", "workers": 1,
                                    "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                record_checkpoint(out, results)
        else:
            for unsupported in ("full", "populations", "full_and_integral"):
                results.append({"fixture": name, "batch": batch_name, "product": unsupported,
                                "method": "scalar_adjoint_python_prototype", "status": "unsupported",
                                "reason": "scalar adjoint returns observable responses, not density matrices or the full population vector"})
                record_checkpoint(out, results)
            for product in ("photon_integral", "selected_1", "selected_4", "group_sum"):
                try:
                    actual, stats = evaluate_rc_adjoint(
                        fixture, product, rho0_batch=rho_batch, solver="DOP853",
                        reltol=1e-8, atol=1e-10, dt=float(fixture.spec.dt),
                    )
                    actual = normalize_product(product, actual, len(rho_batch))
                    expected = expected_product(product, rho_ref, photon_ref, fixture)
                    metrics = accuracy(product, actual, expected)
                    results.append({"fixture": name, "batch": batch_name, "product": product,
                                    "method": "scalar_adjoint_python_prototype", "workers": 1,
                                    "wall_seconds": stats["wall_seconds"], "setup_seconds": stats.get("setup_seconds", 0.0),
                                    "accuracy": metrics, "accuracy_passed": metrics["accuracy_passed"],
                                    "status": "passed" if metrics["accuracy_passed"] else "accuracy_failed",
                                    "backend": stats["backend"], "full_density_supported": False,
                                    "observable_solve_count": stats["backward_solve_count"]})
                    results[-1]["output_artifact"], results[-1]["output_sha256"] = save_output(
                        out, f"adjoint_{batch_name}_{product}", actual,
                    )
                except Exception as exc:
                    results.append({"fixture": name, "batch": batch_name, "product": product,
                                    "method": "scalar_adjoint_python_prototype", "workers": 1,
                                    "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                record_checkpoint(out, results)
        # Weighted mixture validation is untimed for method comparisons and independently
        # checks that basis responses contract linearly for this requested initial-state set.
        mixture_artifact = save_output(out, f"mixture_{batch_name}_full_and_integral", {
            "rho": mixture["_direct_density"], "photon_integral": mixture["_direct_photons"],
        })
        mixture_summary = {key: value for key, value in mixture.items() if not key.startswith("_")}
        results.append({"fixture": name, "batch": batch_name, "method": "mixture_validation",
                        "product": "full_and_integral", "status": "passed" if (
                            mixture["density_accuracy"]["accuracy_passed"] and
                            mixture["photon_accuracy"]["accuracy_passed"]
                        ) else "accuracy_failed", "validation": mixture_summary,
                        "reference_checks": ref_checks})
        results[-1]["reference_artifact"] = ref_artifact
        results[-1]["output_artifact"], results[-1]["output_sha256"] = mixture_artifact
        record_checkpoint(out, results)
    after = resource_sample()
    npz = out / "references.npz"
    temp = out / "references.tmp.npz"
    np.savez_compressed(temp, **ref_arrays)
    replace_retry(temp, npz)
    native_path = Path(importlib.import_module("centrex_tlf.centrex_tlf_rust").__file__).resolve()
    if native_path.suffix.lower() != ".pyd":
        raise RuntimeError(f"expected loaded Windows native extension .pyd, got {native_path}")
    guard_passed = after["cpu_median_percent"] <= CPU_LIMIT and not after["competing_python_processes"]
    payload = {
        "status": "complete" if guard_passed else "resource_guard_failed",
        "complete": guard_passed, "fixture": name, "tag": tag,
        "kernel": kernel, "worker_budgets": list(WORKER_BUDGETS), "products": list(PRODUCTS),
        "batch_names": list(cases), "rows": results,
        "resource_before": before, "resource_after": after,
        "cpu_guard": "median of three 1s samples <=50%; no competing heavy simulation",
        "reference_solver": ["dopri5", "tsit5"], "reference_rtol": 1e-10, "reference_atol": 1e-12,
        "baseline_screen": baseline,
        "fixture_configuration": fixture.configuration, "fixture_metadata": fixture.metadata,
        "native_extension_path": str(native_path), "native_extension_sha256": sha256(native_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "fixture_source_sha256": sha256(HERE / "cross_system_fixtures.py"),
        "methods_source_sha256": sha256(HERE / "cross_system_methods.py"),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                                        capture_output=True, text=True).stdout.strip(),
        "python": sys.version, "platform": platform.platform(), "references_npz_sha256": sha256(npz),
        "timing_scope": "single-parameter-point batch throughput; setup, dense factorization and adjoint setup reported separately",
    }
    atomic_json(out / "result.json", payload)
    atomic_json(out / "progress" / "complete.json", {"status": payload["status"], "complete": payload["complete"],
                                                       "result_sha256": sha256(out / "result.json"), "rows": results})
    if not guard_passed:
        raise RuntimeError(f"post-run guard failed: {after}")
    return out / "result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=FIXTURES, required=True)
    parser.add_argument("--kernel", required=True, help="kernel selected from the reviewed short screen")
    parser.add_argument("--tag", default="screen1us")
    parser.add_argument("--screen-reference", type=Path, required=True,
                        help="complete nominal screen JSON for this fixture and native build")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("no work is performed without --run")
    print(run_batch_fixture(args.fixture, args.tag, args.kernel, args.screen_reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
