"""Focused 200-us full-density RC tolerance validation; explicit --run required."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from itertools import pairwise
from pathlib import Path

import bench_rc_propagation as rc
import numpy as np
from cross_system_fixtures import build_fixture
from run_cross_system_long_finalists import guard_sample
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad import solve_lindblad_batch

MODE = "structured_sparse_dense_diag_soa_single_unpack"
CASES = [
    ("nominal", 0.0, 1.0, 1.0),
    ("detuning_minus20", -20.0, 1.0, 1.0),
    ("detuning_plus20", 20.0, 1.0, 1.0),
    ("optical3", 0.0, 3.0, 1.0),
    ("microwave3", 0.0, 1.0, 3.0),
]
INITIAL = "X_J2_F1_5o2_F2_mF0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def propagate(fixture, parameters, solver, rtol, atol, duration):
    case = fixture.rc_case
    prepared = case.prepared_segmented
    edges = rc.switch_boundaries(125e-9, 125e-9 + duration)
    rho = fixture.rho0.copy()
    trajectory = [rho.copy()]
    names = ("ω1", "φ1", "ω2", "φ2", "Ωt0", "δ0", "Ω1", "Ω2")
    assert set(names) <= set(prepared.parameters.base_parameters)
    detuning, optical, microwave = parameters
    started = time.perf_counter()
    for lo, hi in pairwise(edges):
        a, b = rc.switch_values(float((lo + hi) / 2))
        values = np.array(
            [
                [
                    0.0,
                    rc.SWITCH_PHASE_OFFSET_RAD * (1 if a else -1),
                    0.0,
                    rc.SWITCH_PHASE_OFFSET_RAD * (1 if b else -1),
                    optical * rc.GAMMA,
                    detuning * 2 * np.pi * 1e6,
                    microwave * rc.GAMMA,
                    microwave * rc.GAMMA,
                ]
            ],
            complex,
        )
        result = solve_lindblad_batch(
            prepared,
            rho[None],
            (float(lo), float(hi)),
            parameter_batch=values,
            parameter_slots=names,
            solver=solver,
            execution_mode=MODE,
            reltol=rtol,
            abstol=atol,
            dt=2e-9,
            output="full",
            output_when="final",
            parallel=False,
        )
        rho = prepared.layout.unpack(np.asarray(result.values).reshape(-1))
        trajectory.append(rho.copy())
    return edges, np.asarray(trajectory), time.perf_counter() - started


def metrics(candidate, reference):
    delta = candidate - reference
    pop = np.diagonal(candidate, axis1=-2, axis2=-1).real
    refpop = np.diagonal(reference, axis1=-2, axis2=-1).real
    mask = refpop >= 1e-6
    herm = (candidate + candidate.conj().transpose(0, 2, 1)) / 2
    eig = np.linalg.eigvalsh(herm)
    return {
        "finite": bool(np.isfinite(candidate).all()),
        "density_max_abs": float(np.max(np.abs(delta))),
        "final_density_max_abs": float(np.max(np.abs(delta[-1]))),
        "population_max_abs": float(np.max(np.abs(pop - refpop))),
        "population_max_relative_above_1e6_floor": float(
            np.max(np.abs(pop - refpop)[mask] / refpop[mask])
        ),
        "trace_max_abs_error": float(np.max(np.abs(np.trace(candidate, axis1=-2, axis2=-1) - 1))),
        "hermiticity_max_abs": float(
            np.max(np.abs(candidate - candidate.conj().transpose(0, 2, 1)))
        ),
        "minimum_eigenvalue": float(eig.min()),
        "final_minimum_eigenvalue": float(eig[-1].min()),
        "maximum_negative_eigenvalue_mass": float(np.max(-np.minimum(eig, 0).sum(axis=1))),
    }


def strict_pass(m):
    return (
        m["finite"]
        and m["density_max_abs"] <= 1e-7
        and m["population_max_abs"] <= 1e-7
        and m["population_max_relative_above_1e6_floor"] <= 1e-3
        and m["trace_max_abs_error"] <= 1e-9
        and m["hermiticity_max_abs"] <= 1e-9
        and m["minimum_eigenvalue"] >= -1e-8
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=[c[0] for c in CASES], default=[c[0] for c in CASES]
    )
    parser.add_argument("--duration-us", type=float, default=200.0)
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run to propagate")
    if not np.isfinite(args.duration_us) or args.duration_us <= 0:
        parser.error("positive finite duration required")
    args.output.mkdir(parents=True, exist_ok=False)
    native = Path(importlib.import_module("centrex_tlf.centrex_tlf_rust").__file__)
    report = {
        "status": "running",
        "kernel": MODE,
        "duration_us": args.duration_us,
        "initial_label": INITIAL,
        "entry_time_s": 125e-9,
        "scope": "full density at every switch boundary, including initial and final; no interior guarantee",
        "native_sha256": sha(native),
        "runner_sha256": sha(__file__),
        "fixture_sha256": sha(Path(__file__).with_name("cross_system_fixtures.py")),
        "rc_driver_sha256": sha(rc.__file__),
        "cases": [],
    }
    path = args.output / "result.json"
    write_json(path, report)
    try:
        before = guard_sample()
        report["resource_before"] = before
        if before["cpu_median_percent"] > 50 or before["competing_processes"]:
            raise RuntimeError(f"resource guard failed: {before}")
        with threadpool_limits(limits=1):
            fixture = build_fixture("rc")
            report["configuration"] = fixture.configuration
            report["state_labels"] = [str(qn) for qn in fixture.system.QN]
            assert fixture.initial_label == INITIAL and len(fixture.system.QN) == 70
            for name, det, optical, microwave in CASES:
                if name not in args.cases:
                    continue
                row = {
                    "name": name,
                    "detuning_mhz": det,
                    "optical_rabi_over_gamma": optical,
                    "microwave_rabi_over_gamma": microwave,
                    "solves": [],
                    "status": "running",
                }
                report["cases"].append(row)
                values = {}
                for label, solver, rtol, atol in [
                    ("working", "tsit5", 1e-8, 1e-10),
                    ("reference", "tsit5", 1e-10, 1e-12),
                    ("crosscheck", "dopri5", 1e-10, 1e-12),
                ]:
                    print(f"{name}: {label} {solver} {rtol}/{atol}", flush=True)
                    edges, rho, wall = propagate(
                        fixture,
                        (det, optical, microwave),
                        solver,
                        rtol,
                        atol,
                        args.duration_us * 1e-6,
                    )
                    values[label] = rho
                    # Keep sparse snapshots plus endpoints; score every boundary in memory.
                    ids = np.unique(np.r_[np.arange(0, len(edges), 20), len(edges) - 1])
                    artifact = args.output / f"{name}_{label}.npz"
                    np.savez_compressed(
                        artifact, times=edges[ids], density=rho[ids], scored_time_count=len(edges)
                    )
                    row["solves"].append(
                        {
                            "label": label,
                            "solver": solver,
                            "rtol": rtol,
                            "atol": atol,
                            "wall_seconds": wall,
                            "artifact": artifact.name,
                            "sha256": sha(artifact),
                        }
                    )
                    write_json(path, report)
                    print(f"  completed in {wall:.2f}s", flush=True)
                row["reference_crosscheck"] = metrics(values["reference"], values["crosscheck"])
                row["reference_validated"] = (
                    strict_pass(row["reference_crosscheck"])
                    and row["reference_crosscheck"]["density_max_abs"] <= 1e-8
                    and strict_pass(metrics(values["crosscheck"], values["reference"]))
                )
                row["working_vs_reference"] = metrics(values["working"], values["reference"])
                row["historical_strict_gates_passed"] = bool(
                    row["reference_validated"] and strict_pass(row["working_vs_reference"])
                )
                row["status"] = "complete"
                write_json(path, report)
                print(
                    json.dumps(
                        {
                            "case": name,
                            "reference_validated": row["reference_validated"],
                            **row["working_vs_reference"],
                        }
                    ),
                    flush=True,
                )
                del values
        report["resource_after"] = guard_sample()
        report["status"] = "complete"
        write_json(path, report)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        write_json(path, report)
        raise


if __name__ == "__main__":
    main()
