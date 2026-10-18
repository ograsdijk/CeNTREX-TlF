"""Versioned 1-us K=1 screen for phase-modulated R2/R3 fixtures.

No integration runs without ``--run``. Each fixture contributes 77 rows
(11 kernels × 7 native output products); this module does not edit or reuse
the frozen static-screen outputs.
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

try:
    import run_rc_cross_system_screen_v3 as screen
except ModuleNotFoundError:
    from benchmarks import run_rc_cross_system_screen_v3 as screen

try:
    from cross_system_eom_fixtures import EOM_FIXTURES, build_fixture
except ModuleNotFoundError:
    from benchmarks.cross_system_eom_fixtures import EOM_FIXTURES, build_fixture

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "rc_propagation_results" / "cross_system_eom_screen_v1"
CPU_LIMIT = 50.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, None):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def resource_sample() -> dict[str, Any]:
    import psutil
    proc = psutil.Process()
    exempt = {os.getpid(), *(parent.pid for parent in proc.parents())}
    monitored = ("run_cross_system_eom_screen.py", "run_rc_cross_system_screen_v3.py",
                 "run_cross_system_regime_screen.py", "run_cross_system_batch_amortization")
    competitors = []
    for item in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (item.info.get("name") or "").lower()
            cmd = " ".join(item.info.get("cmdline") or []).lower()
            if item.info["pid"] not in exempt and "python" in name and any(x in cmd for x in monitored):
                competitors.append({"pid": item.info["pid"], "cmdline": cmd[:240]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    cpu = [float(psutil.cpu_percent(interval=1.0)) for _ in range(3)]
    return {"cpu_samples_percent": cpu, "cpu_median_percent": float(np.median(cpu)),
            "available_memory_bytes": int(psutil.virtual_memory().available),
            "competing_processes": competitors}


def case_dict(fixture: Any) -> dict[str, Any]:
    return {
        "fixture": fixture, "spec": fixture.spec, "system": fixture.system,
        "prepared": fixture.prepared, "rho0": np.asarray(fixture.rho0, dtype=np.complex128),
        "initial_index": fixture.initial_index, "four_indices": list(fixture.selected_indices_4),
        "selected_indices_1": list(fixture.selected_indices_1),
        "selected_indices_4": list(fixture.selected_indices_4),
        "group_indices": list(fixture.group_indices), "excited_indices": list(fixture.excited_indices),
        "metadata": fixture.metadata,
    }


def reference_pair(case: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    spec = replace(case["spec"], rtol=1e-10, atol=1e-12)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    walls = {}
    for solver in ("dopri5", "tsit5"):
        result, wall = screen._solve_once(case, spec, "structured", "full_and_integral", solver=solver)
        combined = screen._value_for_product(result, case, "full_and_integral")
        n = len(case["system"].QN)
        outputs[solver] = {"full": combined[:-1].reshape(n, n),
                           "photon_integral": np.asarray([combined[-1]])}
        walls[solver] = wall
    density_diff = float(np.max(np.abs(outputs["dopri5"]["full"] - outputs["tsit5"]["full"])))
    photon_diff = float(np.max(np.abs(outputs["dopri5"]["photon_integral"] - outputs["tsit5"]["photon_integral"])))
    pop_check = screen._accuracy_metrics(
        "populations", np.diag(outputs["dopri5"]["full"]), case,
        {"full": outputs["tsit5"]["full"], "photon_integral": outputs["tsit5"]["photon_integral"]})
    physical = {}
    for solver, row in outputs.items():
        rho = row["full"]
        physical[solver] = {
            "trace_abs_error": float(abs(np.trace(rho) - 1.0)),
            "hermiticity_max_abs": float(np.max(np.abs(rho - rho.conj().T))),
            "minimum_eigenvalue": float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min()),
        }
    if density_diff > 1e-8 or photon_diff > 1e-8 or not pop_check.get("accuracy_passed"):
        raise RuntimeError(f"EOM tight-reference cross-solver gate failed: rho={density_diff:g}, photon={photon_diff:g}, pop={pop_check}")
    if any(v["trace_abs_error"] > 1e-9 or v["hermiticity_max_abs"] > 1e-9
           or v["minimum_eigenvalue"] < -1e-8 for v in physical.values()):
        raise RuntimeError(f"EOM tight-reference physical gate failed: {physical}")
    ref = outputs["tsit5"]
    ref.update({"dopri5_full": outputs["dopri5"]["full"],
                "dopri5_photon_integral": outputs["dopri5"]["photon_integral"],
                "tsit5_full": outputs["tsit5"]["full"],
                "tsit5_photon_integral": outputs["tsit5"]["photon_integral"]})
    return ref, {"density_cross_solver_max_abs": density_diff,
                 "photon_cross_solver_max_abs": photon_diff, "population_cross_check": pop_check,
                 "physical_checks": physical, "solver_names": ["dopri5", "tsit5"],
                 "rtol": spec.rtol, "atol": spec.atol, "reference_wall_seconds": walls}


def run_fixture(name: str, tag: str) -> Path:
    out = OUTROOT / tag / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}; choose a new tag")
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        before = resource_sample()
        if before["cpu_median_percent"] > CPU_LIMIT or before["competing_processes"]:
            raise RuntimeError(f"pre-run resource guard failed: {before}")
        started = time.perf_counter()
        fixture = build_fixture(name)
        build_wall = time.perf_counter() - started
        case = case_dict(fixture)
        reference, reference_checks = reference_pair(case)
        ref_path = out / "reference.npz"
        np.savez_compressed(ref_path, **reference)
        rows = []
        catalog = [{"kernel": kernel, "product": product, "status": "planned"}
                   for kernel in screen.KERNELS for product in screen.PRODUCTS]
        cell_index = 0
        for product in screen.PRODUCTS:
            for kernel in screen.KERNELS:
                started = time.perf_counter()
                result, wall = screen._solve_once(case, fixture.spec, kernel, product, solver="tsit5")
                value = screen._value_for_product(result, case, product)
                metrics = screen._accuracy_metrics(product, value, case, reference)
                initial_metrics = metrics
                accepted_value = np.array(value, copy=True)
                accepted_metrics = metrics if metrics.get("accuracy_passed") else None
                accepted_wall = wall if accepted_metrics is not None else None
                accepted_tol = {"rtol": fixture.spec.rtol, "atol": fixture.spec.atol} if accepted_wall else None
                retries = []
                if not metrics.get("accuracy_passed"):
                    for rtol, atol in ((1e-8, 1e-10), (1e-9, 1e-11)):
                        retry_spec = replace(fixture.spec, rtol=rtol, atol=atol)
                        retry_result, retry_wall = screen._solve_once(case, retry_spec, kernel, product, solver="tsit5")
                        retry_value = screen._value_for_product(retry_result, case, product)
                        retry_metrics = screen._accuracy_metrics(product, retry_value, case, reference)
                        retries.append({"rtol": rtol, "atol": atol, "wall_seconds": retry_wall,
                                        "metrics": retry_metrics,
                                        "accuracy_passed": bool(retry_metrics.get("accuracy_passed"))})
                        if retry_metrics.get("accuracy_passed"):
                            accepted_value = np.array(retry_value, copy=True)
                            accepted_metrics = retry_metrics
                            accepted_wall = retry_wall
                            accepted_tol = {"rtol": rtol, "atol": atol}
                            break
                cell = out / "cells" / f"c{cell_index:03d}.npz"
                cell.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cell, initial_output=np.asarray(value), accepted_output=accepted_value)
                row = {"fixture": name, "kernel": kernel, "product": product,
                       "solver": "tsit5", "initial_tolerance": {"rtol": fixture.spec.rtol, "atol": fixture.spec.atol},
                       "wall_seconds": wall, "outer_wall_seconds": time.perf_counter() - started,
                       "initial_accuracy": initial_metrics, "accepted_accuracy": accepted_metrics,
                       "accuracy_passed": accepted_metrics is not None,
                       "strict_accuracy_passed": bool(accepted_metrics and accepted_metrics.get("strict_accuracy_passed")),
                       "accepted_wall_seconds": accepted_wall, "accepted_tolerance": accepted_tol,
                       "tight_retries": retries,
                       "status": "passed" if accepted_wall is not None else "accuracy_failed",
                       "artifact": str(cell.relative_to(OUTROOT)), "artifact_sha256": sha(cell)}
                rows.append(row)
                catalog[cell_index]["status"] = row["status"]
                cell_index += 1
                atomic_json(out / "progress" / f"after_{cell_index:03d}.json",
                            {"complete": False, "resolved_cells": cell_index, "rows": rows})
        after = resource_sample()
    payload = {
        "schema_version": 1, "status": "complete" if after["cpu_median_percent"] <= CPU_LIMIT and not after["competing_processes"] else "resource_guard_failed",
        "complete": after["cpu_median_percent"] <= CPU_LIMIT and not after["competing_processes"],
        "fixture": name, "tag": tag, "fixture_kind": "constant-amplitude complex phase modulation",
        "expected_cells": len(catalog), "resolved_cells": len(rows),
        "accepted_accuracy_cells": sum(r["accuracy_passed"] for r in rows),
        "failed_accuracy_cells": sum(not r["accuracy_passed"] for r in rows),
        "catalog": catalog, "rows": rows, "fixture_build_seconds": build_wall,
        "configuration": fixture.configuration, "runtime_values": fixture.runtime_values,
        "metadata": fixture.metadata, "reference_checks": reference_checks,
        "reference_npz_sha256": sha(ref_path),
        "native_extension_path": fixture.metadata["native_extension_path"],
        "native_extension_sha256": fixture.metadata["native_extension_sha256"],
        "runner_sha256": sha(Path(__file__).resolve()),
        "fixture_runner_sha256": sha(Path(__file__).with_name("cross_system_eom_fixtures.py")),
        "python": sys.version, "platform": platform.platform(),
        "resource_before": before, "resource_after": after,
        "cpu_guard_rule": "three 1-second samples; median <= 50%; no competing benchmark workers",
        "timing_scope": "K=1 final-output solve; preparation/reference/output extraction reported separately",
    }
    result_path = out / "screen.json"
    atomic_json(result_path, payload)
    atomic_json(out / "progress" / "complete.json", {"complete": payload["complete"],
                                                        "screen_sha256": sha(result_path)})
    if not payload["complete"]:
        raise RuntimeError(f"post-run resource guard failed: {after}")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=EOM_FIXTURES, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("no propagation without --run")
    print(run_fixture(args.fixture, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
