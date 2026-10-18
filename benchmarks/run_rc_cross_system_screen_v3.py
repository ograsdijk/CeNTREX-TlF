"""Versioned short-screen harness for cross-system Lindblad kernel studies.

No workload runs at import time.  The checker runs explicit ``--run`` stages
after source/build review.  Results are immutable per tag and checkpointed.
This first harness is deliberately limited to one nominal K=1 screen; it does
not implement long scans, production APIs, or a universal method selector.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rc_propagation_results" / "cross_system_screen_v3"
GAMMA = float(hamiltonian.Γ)
CPU_LIMIT = 50.0
PREFIX = "experimental_structured_sparse_dense"
KERNELS = (
    "structured", "expanded_sparse", "structured_upper",
    PREFIX, f"{PREFIX}_diag", f"{PREFIX}_soa", f"{PREFIX}_single_unpack",
    f"{PREFIX}_diag_soa", f"{PREFIX}_diag_single_unpack",
    f"{PREFIX}_soa_single_unpack", f"{PREFIX}_diag_soa_single_unpack",
)
FIXTURES = ("r2_f4", "r3_f1_7o2_f4", "r3_f1_9o2_f4", "rc")
PRODUCTS = ("full", "photon_integral", "populations", "selected_1", "selected_4", "group_sum", "full_and_integral")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    for attempt, delay in enumerate((*delays, None)):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def _build_fixture(name: str) -> tuple[dict[str, Any], float]:
    from cross_system_fixtures import build_fixture
    started = time.perf_counter()
    fixture = build_fixture(name)
    result = {
        "fixture": fixture,
        "spec": fixture.spec,
        "system": fixture.system,
        "prepared": fixture.prepared,
        "rho0": np.asarray(fixture.rho0, dtype=np.complex128),
        "initial_index": int(fixture.initial_index),
        "initial_label": fixture.initial_label,
        "selected_indices_1": [int(i) for i in fixture.selected_indices_1],
        "selected_indices_4": [int(i) for i in fixture.selected_indices_4],
        "selected_labels_4": list(fixture.selected_labels_4),
        "group_indices": [int(i) for i in fixture.group_indices],
        "group_label": fixture.group_label,
        "excited_indices": [int(i) for i in fixture.excited_indices],
        "main_pair_labels": fixture.main_pair_labels,
        "main_coupling": float(fixture.main_coupling),
        "runtime_values": fixture.runtime_values,
        "configuration": fixture.configuration,
        "metadata": fixture.metadata,
    }
    if fixture.rc_case is not None:
        result["rc_case"] = fixture.rc_case
        result["prepared_segmented"] = fixture.rc_case.prepared_segmented
    return result, time.perf_counter() - started


def _resource_sample() -> dict[str, Any]:
    import psutil
    cpu = []
    for _ in range(3):
        cpu.append(float(psutil.cpu_percent(interval=1.0)))
    current = psutil.Process()
    exempt = {os.getpid(), *(proc.pid for proc in current.parents())}
    known_heavy = (
        "compare_rc_kernel_matrix.py", "bench_rc_propagation.py",
        "run_rc_block_adjoint_pilot.py", "run_rc_cross_system_screen.py",
        "compare_rc_sparse_dense.py", "compare_rc_structured_upper.py", "simulate_all.py",
    )
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if proc.info["pid"] not in exempt and "python" in name and any(
                script in cmd for script in known_heavy
            ):
                processes.append({"pid": proc.info["pid"], "name": name, "cmdline": cmd[:300]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"cpu_percent_samples": cpu, "cpu_median_percent": float(np.median(cpu)),
            "available_memory_bytes": int(psutil.virtual_memory().available),
            "competing_python_processes": processes}


def _solve_once(case: dict[str, Any], spec: Any, mode: str, product: str,
                *, solver: str = "dopri5") -> tuple[Any, float]:
    prepared = case["prepared"]
    rho0 = case["rho0"]
    selected = case["selected_indices_1"]
    kwargs: dict[str, Any] = {
        "solver": solver, "execution_mode": mode, "reltol": spec.rtol, "abstol": spec.atol,
        "dt": float(getattr(spec, "dt", 2e-9)), "output_when": "final", "collect_stats": True,
    }
    if product == "full":
        kwargs.update(output="full")
    elif product == "full_and_integral":
        kwargs.update(output="full_and_integral",
                      integral_weights=[(i, GAMMA) for i in case["excited_indices"]])
    elif product == "photon_integral":
        kwargs.update(output="photon_integral", integral_weights=[(i, GAMMA) for i in case["excited_indices"]])
    elif product == "populations":
        kwargs.update(output="populations")
    elif product == "selected_1":
        kwargs.update(output="selected", output_indices=[(selected[0], selected[0])])
    elif product == "selected_4":
        picks = case["selected_indices_4"]
        kwargs.update(output="selected", output_indices=[(i, i) for i in picks])
    elif product == "group_sum":
        idx = case["group_indices"]
        kwargs.update(output="selected", output_indices=[(i, i) for i in idx])
    else:
        raise ValueError(product)
    started = time.perf_counter()
    if "rc_case" not in case:
        from centrex_tlf.lindblad.solve import solve_lindblad
        result = solve_lindblad(prepared, rho0, spec.t_span, **kwargs)
    else:
        # RC splits at exact switch edges and carries full rho between segments.
        # The final segment uses the requested native output; photon and joint
        # requests use one native full_and_integral solve per segment.
        result = rc_segmented_solve(case, rho0, spec, mode, product, kwargs)
    # Include unpacking, selected-value extraction and small reductions in
    # the end-to-end requested-product wall time.
    _value_for_product(result, case, product)
    return result, time.perf_counter() - started


def rc_segmented_solve(case: dict[str, Any], rho0: np.ndarray, spec: Any, mode: str,
                       product: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Exact-switch segmentation for the short RC fixture.

    Intermediate calls return full density. The final segment returns the
    requested native endpoint output; photon requests use ``full_and_integral``
    for one state-carry plus integral solve per segment.
    """
    import bench_rc_propagation as rc
    rc_case = case["rc_case"]
    from itertools import pairwise
    t0, t1 = spec.t_span
    edges = rc.switch_boundaries(t0, t1)
    rho = np.asarray(rho0, dtype=np.complex128)
    if rho.ndim == 2:
        rho = rho[np.newaxis, :, :]
    if rho.ndim != 3:
        raise ValueError(f"RC initial batch must have rank 2 or 3, got shape {rho.shape}")
    photon_total = 0.0
    all_stats = []
    last_result = None
    final_product = {
        "full": "full", "photon_integral": "full_and_integral", "full_and_integral": "full_and_integral",
        "populations": "populations", "selected_1": "selected",
        "selected_4": "selected", "group_sum": "selected",
    }[product]
    selected_ids = {
        "selected_1": case["selected_indices_1"],
        "selected_4": case["selected_indices_4"],
        "group_sum": case["group_indices"],
    }
    segments = list(pairwise(edges))
    for segment_index, (lo, hi) in enumerate(segments):
        pa = rc.switch_values((float(lo) + float(hi)) / 2.0)
        params = np.tile(np.asarray([0.0,
            rc.SWITCH_PHASE_OFFSET_RAD if pa[0] else -rc.SWITCH_PHASE_OFFSET_RAD,
            0.0, rc.SWITCH_PHASE_OFFSET_RAD if pa[1] else -rc.SWITCH_PHASE_OFFSET_RAD], complex), (1, 1))
        if product in {"photon_integral", "full_and_integral"}:
            result = solve_lindblad_batch(
                rc_case.prepared_segmented, rho, (float(lo), float(hi)),
                parameter_batch=params, parameter_slots=rc_case.switch_override_slots,
                solver=kwargs["solver"], execution_mode=mode,
                reltol=kwargs["reltol"], abstol=kwargs["abstol"], dt=kwargs["dt"],
                output="full_and_integral", output_when="final",
                integral_weights=[(i, GAMMA) for i in case_b_indices(rc_case)],
                collect_stats=True, parallel=False,
            )
            packed = np.asarray(result.values).reshape(-1)
            packed_len = rc_case.prepared_segmented.layout.packed_len
            rho = rc_case.prepared_segmented.layout.unpack(packed[:packed_len])[None, :, :]
            photon_total += float(np.real(packed[packed_len]))
            last_result = result
        else:
            is_final = segment_index == len(segments) - 1
            output = final_product if is_final else "full"
            output_kwargs = {}
            if is_final and output == "selected":
                output_kwargs["output_indices"] = [(i, i) for i in selected_ids[product]]
            result = solve_lindblad_batch(
                rc_case.prepared_segmented, rho, (float(lo), float(hi)),
                parameter_batch=params, parameter_slots=rc_case.switch_override_slots,
                solver=kwargs["solver"], execution_mode=mode,
                reltol=kwargs["reltol"], abstol=kwargs["abstol"], dt=kwargs["dt"],
                output=output, output_when="final", collect_stats=True, parallel=False,
                **output_kwargs,
            )
            if output == "full":
                rho = rc_case.prepared_segmented.layout.unpack(np.asarray(result.values).reshape(-1))[None, :, :]
            else:
                last_result = result
        all_stats.append(result.solver_stats or {})
    if product == "photon_integral":
        value = np.asarray([photon_total])
    elif product == "full_and_integral":
        packed = np.asarray(last_result.values).reshape(-1)
        final_rho = rc_case.prepared_segmented.layout.unpack(
            packed[:rc_case.prepared_segmented.layout.packed_len]
        )
        value = np.concatenate((final_rho.ravel(), [photon_total]))
    elif product == "full":
        value = rho[0] if rho.shape[0] == 1 else rho
    else:
        value = np.asarray(last_result.values).reshape(-1)
        if product == "group_sum":
            value = np.asarray([np.sum(np.real(value))])
    payload: dict[str, Any] = {"rho": rho if product == "full" else None,
                               "value": value, "photon_integral": photon_total,
                               "solver_stats": all_stats, "rc_exact_segment_handoff": True,
                               "native_reduced_output": True, "final_segment_output": final_product}
    return payload


def case_b_indices(rc_case: Any) -> list[int]:
    return [i for i, qn in enumerate(rc_case.system.QN)
            if qn.largest.electronic_state == states.ElectronicState.B]


def build_catalog() -> list[dict[str, Any]]:
    rows = []
    for fixture in FIXTURES:
        for kernel in KERNELS:
            for product in PRODUCTS:
                rows.append({"fixture": fixture, "kernel": kernel, "product": product,
                             "initial_count": 1, "parameter_case": "nominal",
                             "status": "planned", "eligible": True,
                             "notes": ""})
    return rows


def _value_for_product(result: Any, case: dict[str, Any], product: str) -> np.ndarray:
    if isinstance(result, dict):
        if "value" in result:
            return np.asarray(result["value"])
        rho = np.asarray(result["rho"], dtype=np.complex128)
        if product == "full":
            return rho
        if product in {"photon_integral", "full_and_integral"}:
            return np.asarray([result["photon_integral"]], dtype=np.float64)
        diagonal = np.real(np.diag(rho))
        if product == "populations":
            return diagonal
        if product == "selected_1":
            return diagonal[[case["initial_index"]]]
        if product == "selected_4":
            return diagonal[case["four_indices"]]
        if product == "group_sum":
            return np.asarray([np.sum(diagonal[case["group_indices"]])])
    elif product == "full":
        return np.asarray(result.density_matrices()[-1], dtype=np.complex128)
    elif product == "full_and_integral":
        values = np.asarray(result.values).reshape(-1)
        n_packed = case["prepared"].layout.packed_len
        rho = case["prepared"].layout.unpack(values[:n_packed])
        return np.concatenate((rho.ravel(), np.asarray([values[n_packed]])))
    else:
        values = np.asarray(result.values)
        if product == "photon_integral":
            return np.asarray(values).reshape(-1)
        if product == "full_and_integral":
            n_packed = case["prepared"].layout.packed_len
            rho = case["prepared"].layout.unpack(np.asarray(values).reshape(-1)[:n_packed])
            scalar = np.asarray(values).reshape(-1)[n_packed]
            return np.concatenate((rho.ravel(), np.asarray([scalar])))
        if product in {"populations", "selected_1", "selected_4", "group_sum"}:
            values = np.asarray(values).reshape(-1)
            if product == "populations":
                return values
            if product == "selected_1":
                return values[:1]
            if product == "selected_4":
                return values[:4]
            return np.asarray([np.sum(values)])
    raise ValueError(f"cannot extract {product} from {type(result).__name__}")


def _reference(case: dict[str, Any], spec: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Two tight independent solver references, with exact RC segmentation."""
    from dataclasses import replace
    tight = replace(spec, rtol=1e-10, atol=1e-12)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    walls: dict[str, float] = {}
    for solver in ("dopri5", "tsit5"):
        started = time.perf_counter()
        full, _ = _solve_once(case, tight, "structured", "full", solver=solver)
        photons, _ = _solve_once(case, tight, "structured", "photon_integral", solver=solver)
        outputs[solver] = {
            "full": _value_for_product(full, case, "full"),
            "photon_integral": _value_for_product(photons, case, "photon_integral"),
        }
        walls[solver] = time.perf_counter() - started
    diff_rho = float(np.max(np.abs(outputs["dopri5"]["full"] - outputs["tsit5"]["full"])))
    diff_photon = float(np.max(np.abs(outputs["dopri5"]["photon_integral"] - outputs["tsit5"]["photon_integral"])))
    cross_population = _accuracy_metrics(
        "populations", np.diag(outputs["dopri5"]["full"]), case,
        {"full": outputs["tsit5"]["full"],
         "photon_integral": outputs["tsit5"]["photon_integral"]},
    )
    physical_checks = {}
    for solver_name, output in outputs.items():
        candidate = output["full"]
        physical_checks[solver_name] = {
            "trace_abs_error": float(abs(np.trace(candidate) - 1.0)),
            "hermiticity_max_abs": float(np.max(np.abs(candidate - candidate.conj().T))),
            "minimum_eigenvalue": float(np.linalg.eigvalsh((candidate + candidate.conj().T) / 2.0).min()),
        }
    if (diff_rho > 1e-8 or diff_photon > 1e-8 or not cross_population["accuracy_passed"]
            or any(p["trace_abs_error"] > 1e-9 or p["hermiticity_max_abs"] > 1e-9
                   or p["minimum_eigenvalue"] < -1e-8 for p in physical_checks.values())):
        raise RuntimeError(f"cross-solver reference mismatch: rho={diff_rho:g}, photons={diff_photon:g}, populations={cross_population}")
    reference = {
        **outputs["tsit5"],
        "dopri5_full": outputs["dopri5"]["full"],
        "dopri5_photon_integral": outputs["dopri5"]["photon_integral"],
        "tsit5_full": outputs["tsit5"]["full"],
        "tsit5_photon_integral": outputs["tsit5"]["photon_integral"],
    }
    rho = reference["full"]
    trace_error = abs(np.trace(rho) - 1.0)
    herm_error = float(np.max(np.abs(rho - rho.conj().T)))
    min_eig = float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min())
    if trace_error > 1e-9 or herm_error > 1e-9 or min_eig < -1e-8:
        raise RuntimeError(f"reference physical gates failed: trace={trace_error:g}, herm={herm_error:g}, eig={min_eig:g}")
    return reference, {"cross_solver_max_density_abs": diff_rho,
                       "cross_solver_max_photon_abs": diff_photon,
                       "cross_solver_population_check": cross_population,
                       "per_solver_physical_checks": physical_checks,
                       "reference_wall_seconds": walls,
                       "trace_abs_error": float(trace_error),
                       "hermiticity_max_abs": herm_error,
                       "minimum_eigenvalue": min_eig}


def _accuracy_metrics(product: str, value: np.ndarray, case: dict[str, Any],
                      reference: dict[str, np.ndarray]) -> dict[str, Any]:
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        return {"accuracy_passed": False, "reason": "nonfinite output"}

    def check_populations(candidate: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
        candidate = np.asarray(candidate)
        expected = np.asarray(expected, dtype=float)
        imaginary = float(np.max(np.abs(np.imag(candidate)))) if np.iscomplexobj(candidate) else 0.0
        real = np.real(candidate)
        absolute = np.abs(real - expected)
        mask = np.abs(expected) >= 1e-6
        relative = absolute[mask] / np.abs(expected[mask]) if np.any(mask) else np.asarray([0.0])
        nonnegative = bool(np.all(real >= -1e-7))
        return {
            "population_max_abs_error": float(np.max(absolute)),
            "population_max_relative_error_above_floor": float(np.max(relative)),
            "population_max_imaginary_abs": imaginary,
            "population_minimum": float(np.min(real)),
            "population_nonnegative": nonnegative,
            "production_population_passed": bool(np.all(absolute <= 1e-6) and np.all(relative <= 3e-3)
                                                  and imaginary <= 1e-9 and nonnegative),
            "strict_population_passed": bool(np.all(absolute <= 1e-7) and np.all(relative <= 1e-3)
                                              and imaginary <= 1e-9 and nonnegative),
            "accuracy_passed": bool(np.all(absolute <= 1e-6) and np.all(relative <= 3e-3)
                                     and imaginary <= 1e-9 and nonnegative),
            "strict_accuracy_passed": bool(np.all(absolute <= 1e-7) and np.all(relative <= 1e-3)
                                            and imaginary <= 1e-9 and nonnegative),
        }

    def check_density(candidate: np.ndarray) -> dict[str, Any]:
        ref = reference["full"]
        error = float(np.max(np.abs(candidate - ref)))
        trace = float(abs(np.trace(candidate) - 1.0))
        herm = float(np.max(np.abs(candidate - candidate.conj().T)))
        mineig = float(np.linalg.eigvalsh((candidate + candidate.conj().T) / 2.0).min())
        pop = check_populations(np.diag(candidate), np.real(np.diag(ref)))
        physical = trace <= 1e-9 and herm <= 1e-9 and mineig >= -1e-8
        return {"density_max_abs_error": error, "trace_abs_error": trace,
                "hermiticity_max_abs": herm, "minimum_eigenvalue": mineig,
                "diagonal_population_metrics": pop,
                "accuracy_passed": bool(error <= 1e-6 and pop["production_population_passed"] and physical),
                "strict_accuracy_passed": bool(error <= 1e-7 and pop["strict_population_passed"] and physical)}

    if product == "full":
        return check_density(value)
    if product == "photon_integral":
        ref = reference["photon_integral"]
        abs_error = np.abs(value - ref)
        err = float(np.max(abs_error))
        nonnegative = bool(np.min(value) >= -1e-7)
        mask = np.abs(ref) >= 1e-6
        rel = abs_error[mask] / np.abs(ref[mask]) if np.any(mask) else np.asarray([0.0])
        return {"accuracy_passed": bool(np.all(abs_error <= 1e-6) and np.all(rel <= 1e-5) and nonnegative),
                "strict_accuracy_passed": bool(np.all(abs_error <= 1e-7) and np.all(rel <= 1e-6)),
                "max_abs_error": err,
                "max_relative_error_above_floor": float(np.max(rel)), "nonnegative": nonnegative}
    if product == "full_and_integral":
        ref = np.concatenate((reference["full"].ravel(), reference["photon_integral"].ravel()))
        rho_candidate = value[:-1].reshape(reference["full"].shape)
        rho_check = check_density(rho_candidate)
        err_photon = float(abs(value[-1] - ref[-1]))
        rel_photon = err_photon / abs(ref[-1]) if abs(ref[-1]) >= 1e-6 else 0.0
        return {**rho_check,
                "accuracy_passed": bool(rho_check["accuracy_passed"] and err_photon <= 1e-6 and rel_photon <= 1e-5),
                "strict_accuracy_passed": bool(rho_check["strict_accuracy_passed"] and err_photon <= 1e-7 and rel_photon <= 1e-6),
                "photon_abs_error": err_photon, "photon_relative_error": rel_photon}
    rho_ref = np.real(np.diag(reference["full"]))
    if product == "populations":
        ref = rho_ref
    elif product == "selected_1":
        ref = rho_ref[case["selected_indices_1"]]
    elif product == "selected_4":
        ref = rho_ref[case["selected_indices_4"]]
    elif product == "group_sum":
        ref = np.asarray([np.sum(rho_ref[case["group_indices"]])])
    else:
        raise ValueError(product)
    return check_populations(value, ref)


def _calibrate_solvers(case: dict[str, Any], spec: Any,
                       reference: dict[str, np.ndarray]) -> dict[str, Any]:
    """Choose fastest accepted baseline solver, retrying tighter tolerances."""
    from dataclasses import replace
    trials = {}
    eligible: list[tuple[float, str, dict[str, Any]]] = []
    tolerance_ladder = list(dict.fromkeys(((spec.rtol, spec.atol), (1e-8, 1e-10), (1e-9, 1e-11))))
    for solver in ("dopri5", "tsit5"):
        trials[solver] = []
        for rtol, atol in tolerance_ladder:
            trial_spec = replace(spec, rtol=rtol, atol=atol)
            started = time.perf_counter()
            full_result, full_wall = _solve_once(case, trial_spec, "structured", "full", solver=solver)
            photon_result, photon_wall = _solve_once(case, trial_spec, "structured", "photon_integral", solver=solver)
            full = _value_for_product(full_result, case, "full")
            photon = _value_for_product(photon_result, case, "photon_integral")
            full_check = _accuracy_metrics("full", full, case, reference)
            photon_check = _accuracy_metrics("photon_integral", photon, case, reference)
            row = {
                "rtol": rtol, "atol": atol,
                "wall_seconds": time.perf_counter() - started,
                "full_wall_seconds": full_wall, "photon_wall_seconds": photon_wall,
                "full_check": full_check, "photon_check": photon_check,
                "accepted": bool(full_check["accuracy_passed"] and photon_check["accuracy_passed"]),
            }
            trials[solver].append(row)
            if row["accepted"]:
                eligible.append((row["wall_seconds"], solver, row))
                break
    if not eligible:
        raise RuntimeError(f"no baseline solver passed fixture calibration: {trials}")
    _, selected, selected_row = min(eligible, key=lambda item: item[0])
    return {"trials": trials, "selected_solver": selected,
            "selected_rtol": selected_row["rtol"], "selected_atol": selected_row["atol"],
            "selection_rule": "minimum combined full-density and photon wall among accuracy-accepted Dopri5/Tsit5 trials, with tighter tolerance retries"}


def run_fixture_screen(fixture: str, tag: str, solver: str) -> Path:
    out_dir = RESULTS / tag / fixture
    json_path, npz_path = out_dir / "screen.json", out_dir / "screen.npz"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite or resume artifacts under {out_dir}; use a fresh tag")
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _resource_sample()
    if before["cpu_median_percent"] > CPU_LIMIT or before["competing_python_processes"]:
        raise RuntimeError(f"pre-run resource guard failed: {before}")
    # Set BLAS threads once for the whole fixture build/reference/matrix block.
    threadpool_limits(limits=1)
    case, build_seconds = _build_fixture(fixture)
    spec = case["spec"]
    if len(case["selected_indices_4"]) != 4 or len(set(case["selected_indices_4"])) != 4:
        raise RuntimeError(f"{fixture}: selected four-state labels are not distinct")
    for mode in KERNELS:
        case["prepared"].check_execution_mode(mode)
    reference, ref_checks = _reference(case, spec)
    calibration = _calibrate_solvers(case, spec, reference)
    from dataclasses import replace
    spec = replace(spec, rtol=calibration["selected_rtol"], atol=calibration["selected_atol"])
    if solver == "auto":
        solver = calibration["selected_solver"]
    reference_npz = out_dir / "reference.npz"
    tmp_reference = reference_npz.with_name(reference_npz.stem + ".tmp.npz")
    np.savez_compressed(tmp_reference, dopri5_full=reference["dopri5_full"],
                        dopri5_photon_integral=reference["dopri5_photon_integral"],
                        tsit5_full=reference["tsit5_full"],
                        tsit5_photon_integral=reference["tsit5_photon_integral"])
    tmp_reference.replace(reference_npz)
    _atomic_json(out_dir / "reference.json", {
        "status": "validated", "complete": True, "fixture": fixture,
        "configuration": case["configuration"], "metadata": case["metadata"],
        "reference_checks": ref_checks, "solver_calibration": calibration,
        "reference_npz_sha256": _sha(reference_npz),
        "fixture_source_sha256": _sha(HERE / "cross_system_fixtures.py"),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                                        capture_output=True, text=True).stdout.strip(),
    })
    rows = []
    arrays: dict[str, np.ndarray] = {
        "reference_full": reference["full"],
        "reference_photons": reference["photon_integral"],
    }
    progress_dir = out_dir / "progress"
    for mode in KERNELS:
        for product in PRODUCTS:
            row: dict[str, Any] = {"fixture": fixture, "kernel": mode, "product": product,
                                   "solver": solver, "status": "planned"}
            try:
                result, wall = _solve_once(case, spec, mode, product, solver=solver)
                value = _value_for_product(result, case, product)
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError("nonfinite output")
                metrics = _accuracy_metrics(product, value, case, reference)
                row.update(metrics)
                row["initial_tolerance_metrics"] = metrics
                row["wall_seconds"] = wall
                row["output_shape"] = list(np.asarray(value).shape)
                row["initial_tolerance_accuracy_passed"] = bool(row["accuracy_passed"])
                row["status"] = "passed" if row["accuracy_passed"] else "accuracy_failed"
                row["accepted_wall_seconds"] = wall if row["accuracy_passed"] else None
                row["accepted_tolerance"] = {"rtol": spec.rtol, "atol": spec.atol} if row["accuracy_passed"] else None
                retry_rows = []
                if not row["accuracy_passed"]:
                    from dataclasses import replace
                    for retry_rtol, retry_atol in ((1e-8, 1e-10), (1e-9, 1e-11)):
                        retry_spec = replace(spec, rtol=retry_rtol, atol=retry_atol)
                        retry_result, retry_wall = _solve_once(case, retry_spec, mode, product, solver=solver)
                        retry_value = _value_for_product(retry_result, case, product)
                        retry_metrics = _accuracy_metrics(product, retry_value, case, reference)
                        retry_record = {"rtol": retry_rtol, "atol": retry_atol,
                                        "wall_seconds": retry_wall, **retry_metrics}
                        retry_rows.append(retry_record)
                        if retry_metrics["accuracy_passed"]:
                            row["status"] = "passed_tight_retry"
                            row["accuracy_passed"] = True
                            row["accepted_metrics"] = retry_metrics
                            row["accepted_wall_seconds"] = retry_wall
                            row["accepted_tolerance"] = {"rtol": retry_rtol, "atol": retry_atol}
                            retry_key = f"{mode}__{product}__accepted_retry"
                            arrays[retry_key] = np.asarray(retry_value)
                            break
                    row["tight_retries"] = retry_rows
                arrays[f"{mode}__{product}"] = np.asarray(value)
                cell_dir = out_dir / "cells"
                cell_dir.mkdir(exist_ok=True)
                cell_npz = cell_dir / f"{mode}__{product}.npz"
                if cell_npz.exists():
                    raise FileExistsError(f"refusing to overwrite {cell_npz}")
                cell_data = {"initial_tolerance_output": np.asarray(value)}
                retry_key = f"{mode}__{product}__accepted_retry"
                if retry_key in arrays:
                    cell_data["accepted_retry_output"] = arrays[retry_key]
                cell_tmp = cell_npz.with_name(cell_npz.stem + ".tmp.npz")
                np.savez_compressed(cell_tmp, **cell_data)
                cell_tmp.replace(cell_npz)
                row["artifact"] = str(cell_npz.relative_to(RESULTS))
                row["artifact_sha256"] = _sha(cell_npz)
                if isinstance(result, dict):
                    row["rc_exact_segment_handoff"] = True
                    row["native_reduced_output"] = True
                    row["output_boundary_note"] = "exact full-density handoff between switch segments; requested native output on final segment"
            except (NotImplementedError, ValueError) as exc:
                row.update(status="unsupported", reason=f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            _atomic_json(progress_dir / f"after_{len(rows):03d}.json", {
                "status": "running", "complete": False,
                "fixture": fixture, "rows_completed": len(rows),
                "expected_cells": len(KERNELS) * len(PRODUCTS),
                "rows": rows, "solver_calibration": calibration,
                "runner_sha256": _sha(Path(__file__).resolve()),
            })
    after = _resource_sample()
    if after["cpu_median_percent"] > CPU_LIMIT or after["competing_python_processes"]:
        raise RuntimeError(f"post-run resource guard failed: {after}")
    tmp_npz = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    tmp_npz.replace(npz_path)
    native_path = Path(__import__("centrex_tlf.centrex_tlf_rust", fromlist=["__file__"]).__file__).resolve()
    if native_path.suffix.lower() != ".pyd":
        raise RuntimeError(f"loaded native extension is not a .pyd binary: {native_path}")
    payload = {
        "status": "complete" if len(rows) == len(KERNELS) * len(PRODUCTS) else "incomplete",
        "complete": len(rows) == len(KERNELS) * len(PRODUCTS), "fixture": fixture, "tag": tag,
        "kernel_count": len(KERNELS), "product_count": len(PRODUCTS),
        "expected_cells": len(KERNELS) * len(PRODUCTS), "rows": rows,
        "resolved_cells": len(rows),
        "accepted_accuracy_cells": sum(row.get("accuracy_passed") is True for row in rows),
        "fixture_build_seconds": build_seconds, "reference_checks": ref_checks,
        "solver_calibration": calibration,
        "state_count": len(case["system"].QN), "packed_length": case["prepared"].layout.packed_len,
        "configuration": case["configuration"], "fixture_metadata": case["metadata"],
        "initial_label": case["initial_label"], "main_pair_labels": case["main_pair_labels"],
        "main_coupling": case["main_coupling"], "runtime_values": case["runtime_values"],
        "selected_labels_4": case["selected_labels_4"], "group_label": case["group_label"],
        "selected_indices_4": case["selected_indices_4"], "group_indices": case["group_indices"],
        "solver": solver, "t_span_s": list(spec.t_span), "reltol": spec.rtol, "abstol": spec.atol,
        "resource_before": before, "resource_after": after,
        "cpu_guard_rule": "median of three one-second samples, cutoff 50%; competing Python simulation processes pause the stage",
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "scipy", "threadpoolctl", "centrex-tlf")},
        "threadpool_info": __import__("threadpoolctl").threadpool_info(),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                                        capture_output=True, text=True).stdout.strip(),
        "native_extension_path": str(native_path), "native_extension_sha256": _sha(native_path),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "npz_sha256": _sha(npz_path),
        "reference_json_sha256": _sha(out_dir / "reference.json"),
        "reference_npz_sha256": _sha(reference_npz),
    }
    _atomic_json(json_path, payload)
    _atomic_json(progress_dir / "complete.json", {
        "status": payload["status"], "complete": payload["complete"],
        "result_json_sha256": _sha(json_path), "resolved_cells": len(rows),
    })
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true", help="write planned nominal cell catalog only")
    parser.add_argument("--fixture", choices=FIXTURES)
    parser.add_argument("--tag", default="nominal")
    parser.add_argument("--run-fixture", action="store_true", help="run one fixture's full nominal matrix")
    parser.add_argument("--solver", choices=("auto", "dopri5", "tsit5"), default="auto")
    args = parser.parse_args()
    if args.catalog:
        out = RESULTS / "cell_catalog.json"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite {out}")
        _atomic_json(out, {"version": 1, "status": "planned", "cells": build_catalog(),
                           "fixture_count": len(FIXTURES), "kernel_count": len(KERNELS),
                           "product_count": len(PRODUCTS), "nominal_k1_cells": len(build_catalog()),
                           "runner_sha256": _sha(Path(__file__).resolve())})
        print(out)
        return 0
    if not args.run_fixture or not args.fixture:
        parser.error("select --catalog or --run-fixture with --fixture")
    out = run_fixture_screen(args.fixture, args.tag, args.solver)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
