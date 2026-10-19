"""One bounded GPU measurement, launched by the watchdog suite."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from rc_gpu_sparse_methods import GPU, diagnostics, load, numpy_rhs
from threadpoolctl import threadpool_limits


def rhs_check(context, method):
    g = GPU(context, method, 12, 146e-6, 147e-6, 256, pattern="legacy")
    host = g.initial_host.copy()
    rng = np.random.default_rng(731)
    # Positive Hermitian matrices containing arbitrary coherences, including sinks.
    for i in range(3):
        x = rng.normal(size=(70, 70)) + 1j * rng.normal(size=(70, 70))
        host[i] = x @ x.conj().T
        host[i] /= np.trace(host[i])
    y = g.upload(host)
    checks = []
    for t in [
        146.125e-6,
        146.375e-6,
        146.625e-6,
        146.875e-6,
        148.125e-6,
        150.875e-6,
        146.25e-6 - 1e-11,
        146.25e-6 + 1e-11,
    ]:
        g.clock.set(np.array([t, 0.0, 147e-6, g.dt]))
        g.rhs(y, g.ks[0], 0.0)
        actual = g.download(g.ks[0])
        ref = numpy_rhs(g.f, context, host, g.det_host, t)
        diff = np.abs(actual - ref)
        scale = np.maximum(np.abs(ref), 1.0)
        check = {
            "t": t,
            "max_abs": float(diff.max()),
            "max_scaled": float((diff / scale).max()),
            "relative_frobenius": float(np.linalg.norm(actual - ref) / np.linalg.norm(ref)),
        }
        checks.append(check)
        if (
            not np.isfinite(actual).all()
            or check["max_scaled"] > 1e-6
            or check["relative_frobenius"] > 1e-11
        ):
            raise ValueError(f"RHS mismatch: {check}")
    return checks


def run(job):
    context, source, pointer = load()
    out = Path(job["output"])
    report = {
        "job": job,
        "status": "running",
        "pid": os.getpid(),
        "context_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "production_signature": pointer["request_signature"],
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "precision": "float64/complex128",
        "driver_version": cp.cuda.runtime.driverGetVersion(),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
    }
    if job["kind"] == "rhs":
        report["rhs_checks"] = rhs_check(context, job["method"])
    tick = time.perf_counter()
    g = GPU(
        context,
        job["method"],
        job["batch"],
        job.get("start", 146e-6),
        job.get("end", 147e-6),
        job.get("substeps", 256),
        job.get("velocity", 150.0),
        job.get("pattern", "grid"),
    )
    g.capture(rhs_only=job["kind"] == "rhs")
    report.update(
        setup_s=time.perf_counter() - tick,
        graph_capture=not g.eager,
        off_diagonal_nnz=len(g.f["columns"]),
        dt_s=g.dt,
        library_capture_limitation="CuPy rejects cuBLAS/cuSPARSE capture" if g.eager else None,
    )
    if job["kind"] == "rhs":
        samples = []
        for _ in range(3):
            before, after = cp.cuda.Event(), cp.cuda.Event()
            before.record()
            tick = time.perf_counter()
            for _ in range(20):
                if g.eager:
                    g.rhs(g.y, g.ks[0], 0.0)
                else:
                    g.graph.launch()
            after.record()
            after.synchronize()
            samples.append(
                {
                    "wall_s_per_rhs": (time.perf_counter() - tick) / 20,
                    "gpu_elapsed_s_per_rhs": cp.cuda.get_elapsed_time(before, after) / 20000,
                }
            )
        report["timings"] = samples
    elif job["kind"] == "full":
        rho, populations, timing = g.full_transit()
        report.update(timing)
        report["max_trace_error"] = float(np.abs(np.trace(rho, axis1=-2, axis2=-1) - 1).max())
        np.savez_compressed(
            out.with_suffix(".npz"),
            density_matrix=rho,
            populations=populations,
            detunings_MHz=g.det_host,
            initial_J=(np.arange(job["batch"]) % 3) + 1,
        )
        if job.get("reference"):
            with np.load(job["reference"]) as data:
                ref = data["density_matrix"]
            selected = rho[[4]] if len(ref) == 1 else rho
            report["strict_reference_subset_accuracy"] = diagnostics(
                selected.reshape(-1, 70, 70), ref.reshape(-1, 70, 70)
            )
        if job.get("convergence_reference"):
            with np.load(job["convergence_reference"]) as data:
                ref = data["density_matrix"]
            report["timestep_difference"] = diagnostics(
                rho.reshape(-1, 70, 70), ref.reshape(-1, 70, 70)
            )
    else:
        tick = time.perf_counter()
        rho, timings, final_time = g.evolve(job.get("repeats", 3))
        report["evolution_s"] = timings
        report["evolution_and_download_wall_s"] = time.perf_counter() - tick
        report["final_time_s"] = final_time
        if abs(final_time - g.end) > 1e-13:
            raise ValueError("ODE endpoint was not reached")
        with np.load(job["reference"]) as data:
            base = data["rho"]
        ref = base[np.arange(len(rho)) % len(base)]
        # A GPU batch prefix smaller than the reference retains the same order.
        report["accuracy"] = diagnostics(rho, ref)
        report["parameter_table"] = {
            "detunings_MHz": g.det_host.tolist(),
            "initial_J": ((np.arange(len(rho)) % 3) + 1).tolist(),
        }
        np.savez_compressed(
            out.with_suffix(".npz"),
            rho=rho[: min(483, len(rho))],
            reference=ref[: min(483, len(rho))],
            detunings_MHz=g.det_host[: min(483, len(rho))],
        )
    report["cupy_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
    report["cupy_pool_live_bytes"] = cp.get_default_memory_pool().used_bytes()
    free, total = cp.cuda.runtime.memGetInfo()
    report["device_used_bytes_at_end"] = total - free
    report["status"] = "complete"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "kind": job["kind"],
                "method": job["method"],
                "batch": job["batch"],
                "status": "complete",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    with threadpool_limits(limits=1):
        run(job)
