"""CPU reference worker run with SPB's exact production interpreter/Rust binary."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pickle
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf import lindblad
from centrex_tlf.centrex_tlf_rust import prepare_lindblad_problem_py


def run(job):
    root = Path(__file__).resolve().parents[1]
    pointer = json.loads(
        (
            root.parent
            / "CeNTREX-SPB-testing/.cache/simulations/optical_bloch/native_rc_current.json"
        ).read_text(encoding="utf-8")
    )
    source = Path(pointer["progress_path"]).parent / "prepared_context.pkl"
    context = pickle.loads(source.read_bytes())
    q = context["prepared"]
    q.rust_plan = prepare_lindblad_problem_py(q.to_payload())
    import centrex_tlf.centrex_tlf_rust as native

    report = {
        "job": job,
        "python": sys.version,
        "native_binary": native.__file__,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "production_status_at_start": json.loads(Path(pointer["progress_path"]).read_text()),
        "concurrent_production": True,
        "timings": [],
    }
    det = np.asarray(job["detunings_MHz"])
    js = np.asarray(job["initial_J"], int) - 1
    velocity = job.get("velocity", 150.0)
    slots = ["δ0", "center", "sigma_time", "ω1", "φ1", "ω2", "φ2"]
    values = np.zeros((len(det), len(slots)), complex)
    values[:, 0] = 2 * np.pi * 1e6 * det
    values[:, 1] = context["physics"]["uv_length_m"] / velocity / 2
    values[:, 2] = np.sqrt(2) * context["sigma_I"] / velocity
    start, end = job["start"], job["end"]
    sample_times = np.linspace(start, end, 201) if job.get("full") else np.array([start, end])
    boundaries = np.r_[
        sample_times, np.arange(np.floor(start / 250e-9) + 1, np.ceil(end / 250e-9)) * 250e-9
    ]
    boundaries = np.unique(boundaries[(boundaries >= start) & (boundaries <= end)])
    boundaries = boundaries[np.r_[True, np.diff(boundaries) > 1e-15]]
    boundaries[-1] = end
    populations = []
    for _ in range(job.get("repeats", 1)):
        rho = context["rho0"][js].copy()
        histories = [np.diagonal(rho, axis1=-2, axis2=-1).real.copy()]
        next_sample = 1
        tick, cpu = time.perf_counter(), time.process_time()
        for lo, hi in pairwise(boundaries):
            quarter = int(np.floor(((lo + hi) / 2) / 250e-9 + 1e-8)) % 4
            a, b = ((1, 1), (1, 0), (0, 0), (0, 1))[quarter]
            values[:, 4] = np.pi / 2 if a else -np.pi / 2
            values[:, 6] = np.pi / 2 if b else -np.pi / 2
            result = lindblad.solve_lindblad_batch(
                q,
                rho,
                (float(lo), float(hi)),
                parameter_slots=slots,
                parameter_batch=values,
                solver="tsit5",
                execution_mode="structured_sparse_dense_diag_soa_single_unpack",
                reltol=job.get("reltol", 1e-10),
                abstol=job.get("abstol", 1e-12),
                dt=1e-9,
                output="full",
                output_when="final",
                save_start=False,
                parallel=job.get("threads", 1) > 1,
                threads=job.get("threads", 1),
                maxiters=1000000,
            )
            rho = np.asarray([q.layout.unpack(row) for row in result.values])
            if (
                job.get("full")
                and next_sample < len(sample_times)
                and hi >= sample_times[next_sample] - 1e-15
            ):
                histories.append(np.diagonal(rho, axis1=-2, axis2=-1).real.copy())
                next_sample += 1
        if job.get("full"):
            from centrex_tlf import hamiltonian

            endpoint = rho.copy()
            post_end = end + context["physics"]["post_decay_lifetimes"] / float(hamiltonian.Γ)
            post_values = np.zeros((len(det), 10), complex)
            post_values[:, :7] = values
            post = lindblad.solve_lindblad_batch(
                q,
                rho,
                (end, post_end),
                parameter_slots=[*slots, "Ωt0", "mw_peak1", "mw_peak2"],
                parameter_batch=post_values,
                solver="tsit5",
                execution_mode="structured_sparse_dense_diag_soa_single_unpack",
                reltol=job.get("reltol", 1e-10),
                abstol=job.get("abstol", 1e-12),
                dt=1e-9,
                output="full",
                output_when="final",
                save_start=False,
                parallel=job.get("threads", 1) > 1,
                threads=job.get("threads", 1),
                maxiters=1000000,
            )
            rho = np.asarray([q.layout.unpack(row) for row in post.values])
            full_density = np.stack((endpoint, rho), axis=1)
        report["timings"].append(
            {"wall_s": time.perf_counter() - tick, "process_cpu_s": time.process_time() - cpu}
        )
        populations.append(np.diagonal(rho, axis1=-2, axis2=-1).real)
    output = Path(job["output"])
    arrays = {"rho": rho, "detunings_MHz": det, "initial_J": js + 1}
    if job.get("full"):
        arrays.update(density_matrix=full_density, populations=np.stack(histories, axis=1))
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    report["max_trace_error"] = float(np.abs(np.trace(rho, axis1=-2, axis2=-1) - 1).max())
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    with threadpool_limits(limits=1):
        run(job)
