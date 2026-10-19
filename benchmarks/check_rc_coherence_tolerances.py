"""Check complex entrywise RC errors against archived tight snapshots."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from cross_system_fixtures import build_fixture
from run_cross_system_long_finalists import guard_sample
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).parent / "rc_propagation_results/full_density_tolerance_check_v1"
spec = importlib.util.spec_from_file_location("executed_rc_check", ROOT / "executed_runner.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def entry_metrics(candidate, reference):
    magnitude = np.abs(reference)
    error = np.abs(candidate - reference)
    offdiag = ~np.eye(reference.shape[-1], dtype=bool)[None]
    out = {"absolute_max": float(error.max()), "by_floor": {}}
    for floor in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 0.0):
        mask = magnitude > 0 if floor == 0 else magnitude >= floor
        row = {}
        for name, selection in (("all", mask), ("coherences", mask & offdiag)):
            ratio = np.zeros_like(error)
            np.divide(error, magnitude, out=ratio, where=selection)
            idx = np.unravel_index(np.argmax(ratio), ratio.shape)
            row[name] = {
                "relative_max": float(ratio[idx]),
                "index": list(map(int, idx)),
                "reference_magnitude": float(magnitude[idx]),
                "absolute_error": float(error[idx]),
                "entries_over_one_percent": int(np.count_nonzero(selection & (ratio > 0.01))),
            }
        out["by_floor"][str(floor)] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["nominal"])
    parser.add_argument("--tight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = guard_sample()
    if before["cpu_median_percent"] > 50 or before["competing_processes"]:
        raise RuntimeError(f"resource guard failed: {before}")
    report = {
        "status": "running",
        "resource_before": before,
        "rows": [],
        "scope": "complex entrywise comparison at 42 archived snapshots; physical checks at every boundary",
        "runner_sha256": base.sha(__file__),
        "base_runner_sha256": base.sha(ROOT / "executed_runner.py"),
    }
    path = args.output / "result.json"
    with threadpool_limits(limits=1):
        fixture = build_fixture("rc")
        report["configuration"] = fixture.configuration
        report["native_sha256"] = fixture.metadata["native_extension_sha256"]
        for name, det, opt, mw in base.CASES:
            if name not in args.cases:
                continue
            with np.load(ROOT / f"{name}_reference.npz") as f:
                reference = f["density"]
                times = f["times"]
            with np.load(ROOT / f"{name}_working.npz") as f:
                report["rows"].append(
                    {
                        "case": name,
                        "rtol": 1e-8,
                        "atol": 1e-10,
                        "source": "archived_working",
                        **entry_metrics(f["density"], reference),
                    }
                )
            settings = ((1e-9, 1e-11),) if args.tight_only else ((1e-9, 1e-10), (1e-9, 1e-11))
            for rtol, atol in settings:
                print(f"{name}: tsit5 {rtol}/{atol}", flush=True)
                edges, rho, wall = base.propagate(
                    fixture, (det, opt, mw), "tsit5", rtol, atol, 200e-6
                )
                ids = np.searchsorted(edges, times)
                np.testing.assert_allclose(edges[ids], times, rtol=0, atol=1e-15)
                metrics = entry_metrics(rho[ids], reference)
                artifact = args.output / f"{name}_{rtol}_{atol}.npz"
                np.savez_compressed(artifact, times=edges, density=rho)
                row = {
                    "case": name,
                    "rtol": rtol,
                    "atol": atol,
                    "wall_seconds": wall,
                    "artifact": artifact.name,
                    "sha256": base.sha(artifact),
                    "physical": base.metrics(rho, rho),
                    **metrics,
                }
                report["rows"].append(row)
                base.write_json(path, report)
                print(
                    json.dumps(
                        {
                            "case": name,
                            "rtol": rtol,
                            "atol": atol,
                            "wall_seconds": wall,
                            "floor1e6": metrics["by_floor"]["1e-06"],
                        }
                    ),
                    flush=True,
                )
        report["status"] = "complete"
        base.write_json(path, report)


if __name__ == "__main__":
    main()
