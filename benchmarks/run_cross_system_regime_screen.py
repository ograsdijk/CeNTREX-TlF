"""Executable stage-4 short regime screen for the frozen cross-system suite.

This runner is intentionally separate from the nominal screen. It builds one
frozen fixture per system, overrides only declared runtime slots, uses tight
cross-solver references at each regime, and checkpoints each regime. It does
not run unless invoked with ``--run``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf import hamiltonian
from centrex_tlf.lindblad.batch import solve_lindblad_batch

try:
    _fixtures = importlib.import_module("cross_system_fixtures")
except ModuleNotFoundError:
    _fixtures = importlib.import_module("benchmarks.cross_system_fixtures")
try:
    _regime_plan = importlib.import_module("cross_system_regime_screen")
except ModuleNotFoundError:
    _regime_plan = importlib.import_module("benchmarks.cross_system_regime_screen")
build_fixture = _fixtures.build_fixture
FIXTURES = _regime_plan.FIXTURES
KERNELS = _regime_plan.KERNELS
PRODUCTS = _regime_plan.PRODUCTS
build_parameter_catalog = _regime_plan.build_parameter_catalog
regimes_for_fixture = _regime_plan.regimes_for_fixture

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "rc_propagation_results" / "cross_system_regime_screen_v1"
GAMMA = float(hamiltonian.Γ)
CPU_LIMIT = 50.0
RTOL_REFERENCE = 1e-10
ATOL_REFERENCE = 1e-12
REFERENCE_SOLVERS = ("dopri5", "tsit5")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _resource_sample() -> dict[str, Any]:
    import psutil

    cpu = [float(psutil.cpu_percent(interval=1.0)) for _ in range(3)]
    current = psutil.Process()
    exempt = {os.getpid(), *(proc.pid for proc in current.parents())}
    known_heavy = (
        "compare_rc_kernel_matrix.py", "bench_rc_propagation.py",
        "run_rc_block_adjoint_pilot.py", "run_rc_cross_system_screen.py",
        "run_cross_system_regime_screen.py", "compare_rc_sparse_dense.py",
        "compare_rc_structured_upper.py", "simulate_all.py",
    )
    competing = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            command = " ".join(proc.info.get("cmdline") or []).lower()
            if proc.info["pid"] not in exempt and "python" in name and any(
                script in command for script in known_heavy
            ):
                competing.append({"pid": proc.info["pid"], "name": name, "cmdline": command[:300]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "cpu_percent_samples": cpu,
        "cpu_median_percent": float(np.median(cpu)),
        "available_memory_bytes": int(psutil.virtual_memory().available),
        "competing_python_processes": competing,
    }


def _guard(snapshot: dict[str, Any], where: str) -> None:
    if snapshot["cpu_median_percent"] > CPU_LIMIT or snapshot["competing_python_processes"]:
        raise RuntimeError(f"{where} resource guard failed: {snapshot}")


def _parameter_names(fixture: Any, regime: Any) -> tuple[list[str], np.ndarray]:
    params = fixture.prepared.parameters
    if fixture.rc_case is None:
        names = ["rabi", "detuning"]
        values = [regime.optical_rabi_rad_s, regime.detuning_rad_s]
    else:
        names = ["ω1", "φ1", "ω2", "φ2", "Ωt0", "δ0", "Ω1", "Ω2"]
        values = [
            0.0, 0.0, 0.0, 0.0,
            regime.optical_rabi_rad_s, regime.detuning_rad_s,
            regime.microwave_rabi_rad_s, regime.microwave_rabi_rad_s,
        ]
    missing = set(names) - set(params.base_parameters)
    if missing:
        raise KeyError(f"{fixture.spec.name}: runtime override slots are not base parameters: {sorted(missing)}")
    return names, np.asarray(values, dtype=np.complex128)


def _rho_batch(fixture: Any) -> np.ndarray:
    return np.asarray(fixture.rho0[None, ...], dtype=np.complex128)


def _run_batch(
    prepared: Any,
    rho0_batch: np.ndarray,
    t_span: tuple[float, float],
    parameter_names: list[str],
    parameter_values: np.ndarray,
    *,
    mode: str,
    solver: str,
    reltol: float,
    abstol: float,
    dt: float,
    output: str,
    output_indices: list[tuple[int, int]] | None = None,
    integral_weights: list[tuple[int, float]] | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    trajectory_count = len(rho0_batch)
    batch_values = np.tile(parameter_values[None, :], (trajectory_count, 1))
    started = time.perf_counter()
    result = solve_lindblad_batch(
        prepared, rho0_batch, t_span,
        parameter_batch=batch_values, parameter_slots=parameter_names,
        solver=solver, execution_mode=mode, reltol=reltol, abstol=abstol, dt=dt,
        output=output, output_indices=output_indices, integral_weights=integral_weights,
        output_when="final", collect_stats=True, parallel=False,
    )
    wall = time.perf_counter() - started
    values = np.asarray(result.values)
    if values.ndim >= 2 and values.shape[0] == trajectory_count:
        values = values[0]
    return values, wall, result.solver_stats or {}


def _decode_output(fixture: Any, product: str, values: np.ndarray) -> Any:
    values = np.asarray(values)
    if product in {"full", "full_and_integral"}:
        packed_len = fixture.prepared.layout.packed_len
        packed = np.asarray(values[:packed_len], dtype=np.float64)
        rho = fixture.prepared.layout.unpack(packed)
        if product == "full":
            return rho
        return {"full": rho, "photon_integral": np.asarray([float(np.real(values[packed_len]))])}
    if product == "photon_integral":
        return np.asarray(values).reshape(-1)
    if product == "populations":
        return np.asarray(values).reshape(-1)
    return np.asarray(values).reshape(-1)


def _indices_for_product(fixture: Any, product: str) -> list[tuple[int, int]] | None:
    if product == "selected_1":
        return [(fixture.selected_indices_1[0], fixture.selected_indices_1[0])]
    if product == "selected_4":
        return [(i, i) for i in fixture.selected_indices_4]
    if product == "group_sum":
        return [(i, i) for i in fixture.group_indices]
    return None


def _integral_weights(fixture: Any) -> list[tuple[int, float]]:
    return [(int(i), GAMMA) for i in fixture.excited_indices]


def _static_product(
    fixture: Any, regime: Any, mode: str, product: str, *, solver: str,
    reltol: float, abstol: float, dt: float,
) -> tuple[Any, float, dict[str, Any]]:
    names, values = _parameter_names(fixture, regime)
    output = product
    if product in {"selected_1", "selected_4", "group_sum"}:
        output = "selected"
    kwargs = {
        "mode": mode, "solver": solver, "reltol": reltol,
        "abstol": abstol, "dt": dt, "output": output,
        "output_indices": _indices_for_product(fixture, product),
        "integral_weights": _integral_weights(fixture)
        if product in {"photon_integral", "full_and_integral"} else None,
    }
    started = time.perf_counter()
    raw, _, stats = _run_batch(
        fixture.prepared, _rho_batch(fixture), fixture.spec.t_span,
        names, values, **kwargs,
    )
    if product == "group_sum":
        raw = np.asarray([np.sum(raw)])
    value = _decode_output(fixture, product, raw)
    return value, time.perf_counter() - started, stats


def _rc_segmented_product(
    fixture: Any, regime: Any, mode: str, product: str, *, solver: str,
    reltol: float, abstol: float, dt: float,
) -> tuple[Any, float, list[dict[str, Any]]]:
    try:
        rc = importlib.import_module("bench_rc_propagation")
    except ModuleNotFoundError:
        rc = importlib.import_module("benchmarks.bench_rc_propagation")
    case = fixture.rc_case
    t0, t1 = fixture.spec.t_span
    edges = rc.switch_boundaries(t0, t1)
    rho = np.asarray(fixture.rho0, dtype=np.complex128)
    names, base_values = _parameter_names(fixture, regime)
    all_stats: list[dict[str, Any]] = []
    photon_total = 0.0
    output_value: Any = None
    started = time.perf_counter()
    for segment_index, (lo, hi) in enumerate(pairwise(edges)):
        last = segment_index == len(edges) - 2
        sw1, sw2 = rc.switch_values(0.5 * (float(lo) + float(hi)))
        params = base_values.copy()
        params[1] = rc.SWITCH_PHASE_OFFSET_RAD if sw1 else -rc.SWITCH_PHASE_OFFSET_RAD
        params[3] = rc.SWITCH_PHASE_OFFSET_RAD if sw2 else -rc.SWITCH_PHASE_OFFSET_RAD
        wants_integral = product in {"photon_integral", "full_and_integral"}
        if wants_integral:
            segment_output = "full_and_integral"
        elif not last or product == "full":
            segment_output = "full"
        elif product in {"selected_1", "selected_4", "group_sum"}:
            segment_output = "selected"
        else:
            segment_output = product
        indices = _indices_for_product(fixture, product) if last else None
        weights = _integral_weights(fixture) if wants_integral else None
        raw, wall, stats = _run_batch(
            case.prepared_segmented, rho[None, ...], (float(lo), float(hi)),
            names, params, mode=mode, solver=solver, reltol=reltol, abstol=abstol,
            dt=min(dt, float(hi - lo)), output=segment_output,
            output_indices=indices, integral_weights=weights,
        )
        all_stats.append(stats)
        if segment_output == "full_and_integral":
            packed_len = case.prepared_segmented.layout.packed_len
            rho = case.prepared_segmented.layout.unpack(np.asarray(raw[:packed_len], dtype=np.float64))
            photon_total += float(np.real(raw[packed_len]))
            if last and product == "full_and_integral":
                output_value = {"full": rho, "photon_integral": np.asarray([photon_total])}
        elif not last or product == "full":
            rho = case.prepared_segmented.layout.unpack(np.asarray(raw, dtype=np.float64))
            if last:
                output_value = rho
        else:
            output_value = np.asarray(raw).reshape(-1)
            if product == "group_sum":
                output_value = np.asarray([np.sum(output_value)])
    if product == "photon_integral":
        output_value = np.asarray([photon_total])
    return output_value, time.perf_counter() - started, all_stats


def _reference(fixture: Any, regime: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    walls = {}
    for solver in REFERENCE_SOLVERS:
        start = time.perf_counter()
        if fixture.rc_case is None:
            full, _, _ = _static_product(
                fixture, regime, "structured", "full", solver=solver,
                reltol=RTOL_REFERENCE, abstol=ATOL_REFERENCE, dt=fixture.spec.dt,
            )
            photons, _, _ = _static_product(
                fixture, regime, "structured", "photon_integral", solver=solver,
                reltol=RTOL_REFERENCE, abstol=ATOL_REFERENCE, dt=fixture.spec.dt,
            )
        else:
            full, _, _ = _rc_segmented_product(
                fixture, regime, "structured", "full", solver=solver,
                reltol=RTOL_REFERENCE, abstol=ATOL_REFERENCE, dt=fixture.spec.dt,
            )
            photons, _, _ = _rc_segmented_product(
                fixture, regime, "structured", "photon_integral", solver=solver,
                reltol=RTOL_REFERENCE, abstol=ATOL_REFERENCE, dt=fixture.spec.dt,
            )
        outputs[solver] = {"full": full, "photon_integral": photons}
        walls[solver] = time.perf_counter() - start
    rho_diff = float(np.max(np.abs(outputs["dopri5"]["full"] - outputs["tsit5"]["full"])))
    photon_diff = float(np.max(np.abs(outputs["dopri5"]["photon_integral"]
                                      - outputs["tsit5"]["photon_integral"])))
    reference = {
        **outputs["tsit5"],
        "dopri5_full": outputs["dopri5"]["full"],
        "dopri5_photon_integral": outputs["dopri5"]["photon_integral"],
        "tsit5_full": outputs["tsit5"]["full"],
        "tsit5_photon_integral": outputs["tsit5"]["photon_integral"],
    }
    rho = np.asarray(reference["full"])
    trace_error = float(abs(np.trace(rho) - 1.0))
    herm_error = float(np.max(np.abs(rho - rho.conj().T)))
    minimum_eigenvalue = float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min())
    checks = {
        "cross_solver_max_density_abs": rho_diff,
        "cross_solver_max_photon_abs": photon_diff,
        "reference_wall_seconds": walls,
        "trace_abs_error": trace_error,
        "hermiticity_max_abs": herm_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }
    if rho_diff > 1e-8 or photon_diff > 1e-8:
        raise RuntimeError(f"cross-solver reference mismatch: density={rho_diff:g}, photon={photon_diff:g}")
    physical_by_solver = {}
    for solver, row in outputs.items():
        rho_solver = np.asarray(row["full"])
        physical_by_solver[solver] = {
            "trace_abs_error": float(abs(np.trace(rho_solver) - 1.0)),
            "hermiticity_max_abs": float(np.max(np.abs(rho_solver - rho_solver.conj().T))),
            "minimum_eigenvalue": float(np.linalg.eigvalsh((rho_solver + rho_solver.conj().T) / 2).min()),
        }
    if any(check["trace_abs_error"] > 1e-9 or check["hermiticity_max_abs"] > 1e-9
           or check["minimum_eigenvalue"] < -1e-8 for check in physical_by_solver.values()):
        raise RuntimeError(f"reference physical gates failed: {checks}")
    checks["physical_checks_by_solver"] = physical_by_solver
    legacy_case = _accuracy_case(fixture)
    population_check = _legacy_accuracy_metrics(
        "populations", np.real(np.diag(outputs["dopri5"]["full"])), legacy_case,
        outputs["tsit5"],
    )
    if not population_check["accuracy_passed"]:
        raise RuntimeError(f"cross-solver component population gate failed: {population_check}")
    checks["cross_solver_population_gate"] = population_check
    return reference, checks


def _reference_product(fixture: Any, reference: dict[str, Any], product: str) -> Any:
    if product == "full":
        return reference["full"]
    if product == "photon_integral":
        return reference["photon_integral"]
    if product == "full_and_integral":
        return {"full": reference["full"], "photon_integral": reference["photon_integral"]}
    populations = np.real(np.diag(reference["full"]))
    if product == "populations":
        return populations
    if product == "selected_1":
        return populations[list(fixture.selected_indices_1)]
    if product == "selected_4":
        return populations[list(fixture.selected_indices_4)]
    if product == "group_sum":
        return np.asarray([np.sum(populations[list(fixture.group_indices)])])
    raise ValueError(product)


def _legacy_accuracy_metrics(product: str, value: Any, case: dict[str, Any],
                             reference: dict[str, Any]) -> dict[str, Any]:
    module = ("run_rc_cross_system_screen" if __package__ in (None, "")
              else "benchmarks.run_rc_cross_system_screen")
    legacy = importlib.import_module(module)
    return legacy._accuracy_metrics(product, np.asarray(value), case, reference)


def _accuracy_case(fixture: Any) -> dict[str, Any]:
    return {"selected_indices_1": list(fixture.selected_indices_1),
            "selected_indices_4": list(fixture.selected_indices_4),
            "group_indices": list(fixture.group_indices)}


def _accuracy(product: str, value: Any, reference: Any, fixture: Any) -> dict[str, Any]:
    """Use frozen v3 gates for every endpoint product."""
    case = _accuracy_case(fixture)
    if product == "full_and_integral":
        rho = _legacy_accuracy_metrics("full", value["full"], case, reference)
        photon = _legacy_accuracy_metrics("photon_integral", value["photon_integral"], case, reference)
        return {"passed": bool(rho["accuracy_passed"] and photon["accuracy_passed"]),
                "products": {"full": rho, "photon_integral": photon}}
    metric = _legacy_accuracy_metrics(product, value, case, reference)
    return {"passed": bool(metric["accuracy_passed"]), "products": {product: metric}}


def _assert_population_relative_gate() -> None:
    """Cheap synthetic guard against accidentally dropping the 0.3% relative gate."""
    case = {"selected_indices_1": [0], "selected_indices_4": [0, 1, 2, 3],
            "group_indices": [0]}
    metric = _legacy_accuracy_metrics(
        "selected_1", np.asarray([0.0002008]), case,
        {"full": np.diag([0.0002, 0.9998]), "photon_integral": np.asarray([0.0])},
    )
    if metric["accuracy_passed"]:
        raise AssertionError("population relative-error gate failed to reject >0.3% deviation")


def _assert_all_product_gates() -> None:
    """Exercise the frozen gate adapter over all seven endpoint shapes without an ODE."""
    from types import SimpleNamespace
    fixture = SimpleNamespace(
        selected_indices_1=(0,), selected_indices_4=(0, 1, 2, 3),
        group_indices=(1, 2),
    )
    rho = np.diag(np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.complex128))
    photons = np.asarray([0.25])
    reference = {"full": rho, "photon_integral": photons}
    products = {
        "full": rho,
        "photon_integral": photons,
        "populations": np.real(np.diag(rho)),
        "selected_1": np.asarray([0.4]),
        "selected_4": np.asarray([0.4, 0.3, 0.2, 0.1]),
        "group_sum": np.asarray([0.5]),
        "full_and_integral": {"full": rho, "photon_integral": photons},
    }
    failures = {name: _accuracy(name, value, reference, fixture)
                for name, value in products.items()}
    if any(not result["passed"] for result in failures.values()):
        raise AssertionError(f"synthetic endpoint gate check failed: {failures}")


def _run_regime(
    fixture: Any, regime: Any, output_dir: Path, *, fixture_build_seconds: float,
) -> Path:
    regime_dir = output_dir / regime.name
    json_path = regime_dir / f"{regime.name}.json"
    npz_path = regime_dir / f"{regime.name}.npz"
    if regime_dir.exists() and any(regime_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite or resume regime artifacts for {regime.name}")
    regime_dir.mkdir(parents=True, exist_ok=True)
    before = _resource_sample()
    _guard(before, f"{fixture.spec.name}/{regime.name}/pre")
    parameter_names, parameter_values = _parameter_names(fixture, regime)
    reference, reference_checks = _reference(fixture, regime)
    rows = []
    arrays: dict[str, np.ndarray] = {
        "reference_full": np.asarray(reference["full"]),
        "reference_photons": np.asarray(reference["photon_integral"]),
        "reference_dopri5_full": np.asarray(reference["dopri5_full"]),
        "reference_dopri5_photons": np.asarray(reference["dopri5_photon_integral"]),
        "reference_tsit5_full": np.asarray(reference["tsit5_full"]),
        "reference_tsit5_photons": np.asarray(reference["tsit5_photon_integral"]),
    }
    for cell_index, (mode, product) in enumerate(
        ((mode, product) for mode in KERNELS for product in PRODUCTS), start=1
    ):
        row = {"fixture": fixture.spec.name, "regime": regime.name,
               "kernel": mode, "product": product, "solver": "tsit5", "status": "planned"}
        initial_value = None
        value = None
        try:
            if fixture.rc_case is None:
                value, wall, stats = _static_product(
                    fixture, regime, mode, product, solver="tsit5",
                    reltol=fixture.spec.rtol, abstol=fixture.spec.atol, dt=fixture.spec.dt)
            else:
                value, wall, stats = _rc_segmented_product(
                    fixture, regime, mode, product, solver="tsit5",
                    reltol=fixture.spec.rtol, abstol=fixture.spec.atol, dt=fixture.spec.dt)
            initial_value = value
            gates = _accuracy(product, value, reference, fixture)
            accepted_wall = wall if gates["passed"] else None
            accepted_tolerance = ({"rtol": fixture.spec.rtol, "atol": fixture.spec.atol}
                                  if gates["passed"] else None)
            retry_rows = []
            if not gates["passed"]:
                for retry_rtol, retry_atol in ((1e-8, 1e-10), (1e-9, 1e-11)):
                    retry_start = time.perf_counter()
                    if fixture.rc_case is None:
                        retry_value, _, retry_stats = _static_product(
                            fixture, regime, mode, product, solver="tsit5",
                            reltol=retry_rtol, abstol=retry_atol, dt=fixture.spec.dt)
                    else:
                        retry_value, _, retry_stats = _rc_segmented_product(
                            fixture, regime, mode, product, solver="tsit5",
                            reltol=retry_rtol, abstol=retry_atol, dt=fixture.spec.dt)
                    retry_wall = time.perf_counter() - retry_start
                    retry_gates = _accuracy(product, retry_value, reference, fixture)
                    retry_rows.append({"rtol": retry_rtol, "atol": retry_atol,
                                       "wall_seconds": retry_wall, "accuracy": retry_gates,
                                       "solver_stats": retry_stats})
                    if retry_gates["passed"]:
                        value, gates = retry_value, retry_gates
                        accepted_wall = retry_wall
                        accepted_tolerance = {"rtol": retry_rtol, "atol": retry_atol}
                        break
            row.update({"wall_seconds": wall, "accuracy": gates,
                        "accuracy_passed": bool(gates["passed"]),
                        "accepted_wall_seconds": accepted_wall,
                        "accepted_tolerance": accepted_tolerance,
                        "tight_retries": retry_rows,
                        "tolerance": {"rtol": fixture.spec.rtol, "atol": fixture.spec.atol},
                        "solver_stats": stats,
                        "output_boundary_note": (
                            "RC full density is handed across exact switch segments; endpoint scalar output is native in the last segment"
                            if fixture.rc_case is not None else "single static solve"),
                        "native_joint_output": product == "full_and_integral"})
            if product == "full_and_integral":
                arrays[f"{mode}__{product}__initial__rho"] = np.asarray(initial_value["full"])
                arrays[f"{mode}__{product}__initial__photons"] = np.asarray(initial_value["photon_integral"])
                arrays[f"{mode}__{product}__accepted__rho"] = np.asarray(value["full"])
                arrays[f"{mode}__{product}__accepted__photons"] = np.asarray(value["photon_integral"])
            else:
                arrays[f"{mode}__{product}__initial"] = np.asarray(initial_value)
                arrays[f"{mode}__{product}__accepted"] = np.asarray(value)
            row["status"] = "passed" if gates["passed"] else "accuracy_failed"
        except NotImplementedError as exc:
            row.update(status="unsupported", reason=str(exc))
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        cells_dir = regime_dir / "cells"
        cells_dir.mkdir(exist_ok=True)
        cell_npz = cells_dir / f"{mode}__{product}.npz"
        if cell_npz.exists():
            raise FileExistsError(f"refusing to overwrite cell checkpoint {cell_npz}")
        if row["status"] in {"passed", "accuracy_failed"}:
            if product == "full_and_integral":
                cell_arrays = {"initial__rho": np.asarray(initial_value["full"]),
                               "initial__photons": np.asarray(initial_value["photon_integral"]),
                               "accepted__rho": np.asarray(value["full"]),
                               "accepted__photons": np.asarray(value["photon_integral"])}
            else:
                cell_arrays = {"initial_output": np.asarray(initial_value),
                               "accepted_output": np.asarray(value)}
            tmp_cell = cell_npz.with_name(cell_npz.stem + ".tmp.npz")
            np.savez_compressed(tmp_cell, **cell_arrays)
            tmp_cell.replace(cell_npz)
            row["cell_artifact"] = cell_npz.name
        progress = {"status": "running", "complete": False, "fixture": fixture.spec.name,
                    "regime": asdict(regime), "completed_cells": len(rows),
                    "expected_cells": len(KERNELS) * len(PRODUCTS), "rows": rows,
                    "runner_sha256": _sha(Path(__file__).resolve())}
        progress_path = regime_dir / f"progress_after_{cell_index:03d}.json"
        progress_path.write_text(json.dumps(progress, indent=2, default=_json_default), encoding="utf-8")
    after = _resource_sample()
    tmp_npz = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    tmp_npz.replace(npz_path)
    try:
        native = importlib.import_module("centrex_tlf.centrex_tlf_rust")
        native_path = Path(native.__file__).resolve()
        native_sha = _sha(native_path)
    except Exception as exc:  # retain diagnostic artifact even if native metadata fails
        native_path, native_sha = None, f"metadata_error:{type(exc).__name__}:{exc}"
    payload = {
        "schema_version": 1, "status": "complete", "complete": True,
        "fixture": fixture.spec.name, "regime": asdict(regime),
        "t_span_s": list(fixture.spec.t_span), "solver": "tsit5",
        "rtol": fixture.spec.rtol, "atol": fixture.spec.atol, "dt_s": fixture.spec.dt,
        "parameter_slots": parameter_names,
        "parameter_slot_values": parameter_values,
        "rc_phase_override_rule": (
            "per exact switch segment, phi1/phi2 are +/- SWITCH_PHASE_OFFSET_RAD; omega slots remain fixed"
            if fixture.rc_case is not None else None
        ),
        "parameter_overrides": {"detuning_rad_s": regime.detuning_rad_s,
                                "optical_rabi_rad_s": regime.optical_rabi_rad_s,
                                "microwave_rabi_rad_s": regime.microwave_rabi_rad_s},
        "kernel_count": len(KERNELS), "product_count": len(PRODUCTS),
        "candidate_tolerance": {"rtol": fixture.spec.rtol, "atol": fixture.spec.atol},
        "accepted_tolerances": {row["kernel"] + "__" + row["product"]: row.get("accepted_tolerance")
                                for row in rows if row.get("accepted_tolerance") is not None},
        "expected_cells": len(KERNELS) * len(PRODUCTS), "rows": rows,
        "passed_cell_count": sum(row["status"] == "passed" for row in rows),
        "accuracy_failed_cell_count": sum(row["status"] == "accuracy_failed" for row in rows),
        "failed_cell_count": sum(row["status"] == "failed" for row in rows),
        "unsupported_cell_count": sum(row["status"] == "unsupported" for row in rows),
        "all_accuracy_passed": all(row["status"] == "passed" for row in rows),
        "fixture_build_seconds": fixture_build_seconds,
        "reference_checks": reference_checks,
        "fixture_metadata": fixture.metadata,
        "fixture_builder_sha256": _sha(HERE / "cross_system_fixtures.py"),
        "regime_catalog_sha256": _sha(HERE / "cross_system_regime_screen.py"),
        "resource_before": before, "resource_after": after,
        "resource_cutoff_cpu_median_percent": CPU_LIMIT,
        "timings_accepted": (before["cpu_median_percent"] <= CPU_LIMIT
                              and after["cpu_median_percent"] <= CPU_LIMIT
                              and not before["competing_python_processes"]
                              and not after["competing_python_processes"]),
        "python": sys.version, "platform": platform.platform(),
        "native_extension_path": str(native_path) if native_path else None,
        "native_extension_sha256": native_sha,
        "runner_sha256": _sha(Path(__file__).resolve()),
        "npz_sha256": _sha(npz_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    if not payload["timings_accepted"]:
        raise RuntimeError(f"post-run resource guard failed; artifacts preserved for {regime.name}")
    return json_path


def run_fixture(fixture_name: str, tag: str) -> list[Path]:
    output_dir = RESULTS_ROOT / tag / fixture_name
    output_dir.mkdir(parents=True, exist_ok=True)
    before_build = _resource_sample()
    _guard(before_build, f"{fixture_name}/build/pre")
    with threadpool_limits(limits=1):
        build_started = time.perf_counter()
        fixture = build_fixture(fixture_name)
        build_seconds = time.perf_counter() - build_started
        for mode in KERNELS:
            fixture.prepared.check_execution_mode(mode)
        completed = []
        for regime in regimes_for_fixture(fixture_name):
            completed.append(_run_regime(
                fixture, regime, output_dir, fixture_build_seconds=build_seconds,
            ))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the physical regime screen")
    parser.add_argument("--tag", default="stage4_v1")
    parser.add_argument("--fixture", choices=FIXTURES, action="append")
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    _assert_population_relative_gate()
    _assert_all_product_gates()
    catalog = build_parameter_catalog()
    if args.catalog_only or not args.run:
        print(json.dumps({"plan_only": True, "regime_cells": len(catalog),
                          "fixtures": list(FIXTURES), "kernels": list(KERNELS),
                          "products": list(PRODUCTS)}, indent=2))
        return
    if not args.run:
        parser.error("--run is required to execute any physical workload")
    for fixture_name in args.fixture or list(FIXTURES):
        run_fixture(fixture_name, args.tag)


if __name__ == "__main__":
    main()
