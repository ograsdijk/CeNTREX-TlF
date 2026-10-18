"""Retry only accuracy-failed batch rows at tighter tolerances.

Reads a completed v1 batch artifact and its saved references, verifies their
hashes, and writes a separate immutable v2 retry tree. Accepted v1 rows are
never rerun. This module performs no work without explicit ``--run``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

try:
    from run_cross_system_batch_amortization import (
        OUTROOT as V1_ROOT, accuracy, build_fixture, expected_product,
        normalize_product, run_native, state_batches,
    )
except ModuleNotFoundError:
    from benchmarks.run_cross_system_batch_amortization import (
        OUTROOT as V1_ROOT, accuracy, build_fixture, expected_product,
        normalize_product, run_native, state_batches,
    )

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "rc_propagation_results" / "cross_system_batch_amortization_retry_v2"
TIGHTER = ((1e-8, 1e-10), (1e-9, 1e-11))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def replace_retry(src: Path, dst: Path) -> None:
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, None):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=default), encoding="utf-8")
    replace_retry(tmp, path)


def save_output(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    arrays = value if isinstance(value, dict) else {"output": value}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{key: np.asarray(item) for key, item in arrays.items()})
    replace_retry(tmp, path)
    return sha(path)


def guard_snapshot() -> dict[str, Any]:
    import psutil
    samples = [float(psutil.cpu_percent(interval=1.0)) for _ in range(3)]
    current = psutil.Process()
    exempt = {os.getpid(), *(p.pid for p in current.parents())}
    known = ("run_cross_system_batch_amortization_retry_v2.py",
             "run_rc_cross_system_screen_v3.py", "run_cross_system_regime_screen.py",
             "bench_rc_propagation.py", "simulate_all.py")
    competitors = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if p.info["pid"] not in exempt and "python" in (p.info.get("name") or "").lower() and any(x in cmd for x in known):
                competitors.append({"pid": p.info["pid"], "cmdline": cmd[:250]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {"cpu_samples": samples, "cpu_median": float(np.median(samples)),
            "competing_processes": competitors}


def run_retry(source_path: Path, tag: str) -> Path:
    source_path = source_path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("complete") is not True:
        raise ValueError("source batch result must be complete")
    fixture_name = source["fixture"]
    target = OUTROOT / tag / fixture_name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite {target}; use a fresh tag")
    target.mkdir(parents=True, exist_ok=True)
    before = guard_snapshot()
    if before["cpu_median"] > 50.0 or before["competing_processes"]:
        raise RuntimeError(f"pre-run guard failed: {before}")

    source_native = source.get("native_extension_sha256")
    native_path = Path(importlib.import_module("centrex_tlf.centrex_tlf_rust").__file__).resolve()
    if sha(native_path) != source_native:
        raise ValueError("source artifact native extension hash differs from current loaded binary")
    threadpool_limits(limits=1)
    fixture = build_fixture(fixture_name)
    stable_json = lambda value: json.dumps(value, sort_keys=True, default=default)
    if stable_json(source.get("fixture_configuration")) != stable_json(fixture.configuration):
        raise ValueError("source fixture configuration differs from the rebuilt fixture")
    batches = state_batches(fixture)
    rows_by_batch = {name: (rho, indices) for name, (rho, indices, _method) in batches.items()}

    retry_rows = []
    output_dir = target / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_row in source.get("rows", []):
        if source_row.get("status") != "accuracy_failed" or source_row.get("method") not in (None, "native_forward"):
            continue
        batch_name = source_row["batch"]
        if batch_name not in rows_by_batch:
            raise ValueError(f"source batch name is unknown: {batch_name}")
        rho_batch, indices = rows_by_batch[batch_name]
        reference_path = source_path.parent / "references" / f"{batch_name}.npz"
        if not reference_path.exists():
            raise FileNotFoundError(reference_path)
        mixture_row = next((r for r in source["rows"] if r.get("batch") == batch_name
                            and r.get("method") == "mixture_validation"), None)
        expected_ref_sha = (mixture_row or {}).get("reference_artifact", {}).get("sha256")
        if expected_ref_sha is None or sha(reference_path) != expected_ref_sha:
            raise ValueError(f"reference checksum mismatch for {batch_name}")
        with np.load(reference_path) as arrays:
            rho_ref = np.asarray(arrays["rho"])
            photon_ref = np.asarray(arrays["photon_integral"])
        if rho_ref.shape != rho_batch.shape or photon_ref.shape[0] != len(rho_batch):
            raise ValueError(f"reference shapes do not match source batch {batch_name}")
        product = source_row["product"]
        expected = expected_product(product, rho_ref, photon_ref, fixture)
        mode = source_row.get("kernel")
        if not mode:
            continue
        source_output = source_row.get("output_artifact")
        source_output_sha = source_row.get("output_sha256")
        if source_output and source_output_sha:
            source_output_path = V1_ROOT / source_output
            if not source_output_path.exists() or sha(source_output_path) != source_output_sha:
                raise ValueError(f"source output artifact checksum mismatch: {source_output}")
        retry_records = []
        for rtol, atol in TIGHTER:
            started = time.perf_counter()
            actual, wall, details = run_native(
                fixture, rho_batch, product, mode, source_row["solver"],
                int(source_row["workers"]), rtol, atol,
            )
            actual = normalize_product(product, actual, len(rho_batch))
            metrics = accuracy(product, actual, expected)
            row = {
                "rtol": rtol, "atol": atol, "wall_seconds": wall,
                "outer_wall_seconds_including_decode": time.perf_counter() - started,
                "accuracy": metrics, "accuracy_passed": bool(metrics["accuracy_passed"]),
                "solver_details": details,
            }
            name = f"{len(retry_rows):04d}_{batch_name}_{mode}_{product}_{rtol:g}"
            path = output_dir / f"{name}.npz"
            row["artifact"] = str(path.relative_to(OUTROOT))
            row["artifact_sha256"] = save_output(path, actual)
            retry_records.append(row)
            if row["accuracy_passed"]:
                break
        accepted = next((row for row in retry_records if row["accuracy_passed"]), None)
        retry_rows.append({
            "source_row": {k: source_row.get(k) for k in
                           ("batch", "initial_count", "product", "kernel", "workers", "solver", "rtol", "atol", "status")},
            "source_artifact": source_row.get("output_artifact"),
            "source_artifact_sha256": source_row.get("output_sha256"),
            "reference_artifact": str(reference_path.relative_to(source_path.parent)),
            "reference_sha256": expected_ref_sha,
            "initial_indices": indices,
            "accepted_tolerance": ({"rtol": accepted["rtol"], "atol": accepted["atol"]}
                                   if accepted else None),
            "accepted_metrics": accepted["accuracy"] if accepted else None,
            "retry_rows": retry_records,
            "status": "passed_tight_retry" if accepted else "still_accuracy_failed",
        })
        atomic_json(target / "progress" / f"after_{len(retry_rows):05d}.json", {
            "complete": False, "source_result": str(source_path), "retry_rows": retry_rows,
        })

    after = guard_snapshot()
    guard_passed = after["cpu_median"] <= 50.0 and not after["competing_processes"]
    result = {
        "schema_version": 2, "status": "complete" if guard_passed else "resource_guard_failed",
        "complete": guard_passed, "source_result": str(source_path),
        "source_result_sha256": sha(source_path), "source_native_sha256": source_native,
        "fixture": fixture_name, "retries_only_accuracy_failed_native_rows": True,
        "retry_tolerances": [{"rtol": r, "atol": a} for r, a in TIGHTER],
        "rows": retry_rows, "resource_before": before, "resource_after": after,
        "runner_sha256": sha(Path(__file__).resolve()),
        "reference_physics_unchanged": True,
    }
    result_path = target / "result.json"
    atomic_json(result_path, result)
    atomic_json(target / "progress" / "complete.json", {
        "complete": guard_passed, "result_sha256": sha(result_path),
        "retried_rows": len(retry_rows),
    })
    if not guard_passed:
        raise RuntimeError(f"post-run resource guard failed: {after}")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("no retry work occurs without --run")
    print(run_retry(args.source_result, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
