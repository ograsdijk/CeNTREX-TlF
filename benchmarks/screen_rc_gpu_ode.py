"""Bounded, isolated complex128 GPU ODE screen using the production RC payload.

Experimental fixed-step RK4, not a public solver or production replacement.
Only reads the SPB run. Writes this benchmark's own output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from centrex_tlf import lindblad
from centrex_tlf.centrex_tlf_rust import prepare_lindblad_problem_py

ROOT = Path(__file__).resolve().parents[1]
SPB = ROOT.parent / "CeNTREX-SPB-testing"
OUT = ROOT / "benchmarks/rc_gpu_ode_screen_results"
QUARTER = 250e-9
START = 146e-6


def sync():
    torch.cuda.synchronize()


def load():
    pointer = json.loads(
        (SPB / ".cache/simulations/optical_bloch/native_rc_current.json").read_text()
    )
    source = Path(pointer["progress_path"]).parent / "prepared_context.pkl"
    context = pickle.loads(source.read_bytes())
    return context, source, pointer


def points(count):
    # Repeat a heterogeneous set spanning the detuning range and all J mixtures.
    det = np.resize(np.repeat([-20.0, -1.0, 0.0, 20.0], 3), count)
    js = np.resize([0, 1, 2], count)
    return det, js


def matrices(context, substeps):
    """Exact prepared upper-triangle basis, with original absolute time/phase."""
    q = context["prepared"]
    plan = q.hamiltonian_plan
    assert plan["kind"] == "decomposed" and len(plan["coefficients"]) == 8
    # Fail if the specialized prototype's expected coefficient mapping changes.
    expected = [
        [(2, 3)],
        [(2, 5)],
        [(2, 17), (2, 20), (7, None)],
        [(2, 16), (2, 20), (7, None)],
        [(2, 18), (2, 21), (7, None)],
        [(2, 19), (2, 21), (7, None)],
        [(2, 4), (2, 15), (7, None)],
        [(2, 8)],
    ]
    actual = [
        [(x["op"], x.get("index")) for x in c["expression"]["instructions"]]
        for c in plan["coefficients"]
    ]
    assert actual == expected
    basis = []
    for coefficient in plan["coefficients"]:
        b = np.zeros((q.layout.n, q.layout.n), complex)
        for x in coefficient["basis_terms"]:
            b[x["i"], x["j"]] += complex(x["re"], x["im"])
        basis.append(b)
    det_mask = np.diag(basis[0]).real
    assert np.count_nonzero(basis[0] - np.diag(det_mask)) == 0
    base = {
        k: complex(v["re"], v["im"]).real
        for k, v in zip(
            q.parameter_graph["slot_names"], q.parameter_graph["base_values"], strict=False
        )
    }
    duration = context["physics"]["uv_length_m"] / 150.0
    sigma = np.sqrt(2) * context["sigma_I"] / 150.0
    dt = QUARTER / substeps
    hs = []
    for segment in range(4):
        mid = START + (segment + 0.5) * QUARTER
        # Production switch_configuration: A,B from sin(2pi f t + phase)>0.
        a = np.sin(2 * np.pi * 1e6 * mid) > 0
        b = np.sin(2 * np.pi * 1e6 * mid + np.pi / 2) > 0
        for step in range(substeps):
            stages = []
            for fraction in [0.0, 0.5, 1.0]:
                t = START + segment * QUARTER + (step + fraction) * dt
                envelope = np.exp(-0.5 * ((t - duration / 2) / sigma) ** 2)
                c = [
                    0.0,
                    0.0,
                    (not a) * base["mw_peak1"] * envelope,
                    a * base["mw_peak1"] * envelope,
                    b * base["mw_peak2"] * envelope,
                    (not b) * base["mw_peak2"] * envelope,
                    base["PUVZ0"]
                    * base["Ωt0"]
                    * np.exp(1j * base["β0"] * np.sin(base["ωphase0"] * t)),
                    0.0,
                ]
                upper = sum(value * matrix for value, matrix in zip(c, basis, strict=False))
                h = np.asarray(plan["static_matrix"]) + upper + np.triu(upper, 1).conj().T
                stages.append(h)
            hs.append(stages)
    return np.asarray(hs), det_mask, dt


def gpu(context, count, substeps=128):
    began = time.perf_counter()
    h, mask, dt = matrices(context, substeps)
    n = context["prepared"].layout.n
    det, js = points(count)
    rho_host = context["rho0"][js]
    transfer = np.zeros((n, n))
    for target, entries in enumerate(context["prepared"].incoming_transfers_by_target):
        for item in entries:
            transfer[target, item["source"]] += item["rate"]
    rates = context["prepared"].source_decay_rates
    loss_host = -0.5 * (rates[:, None] + rates[None, :])
    with torch.inference_mode():
        state = torch.tensor(rho_host, device="cuda", dtype=torch.complex128)
        initial = state.clone()
        all_h = torch.tensor(h, device="cuda", dtype=torch.complex128)
        chunk = 16
        h_input = all_h[:chunk].clone()
        det_diff = torch.tensor(
            2 * np.pi * 1e6 * det[:, None, None] * (mask[:, None] - mask[None, :]),
            device="cuda",
            dtype=torch.float64,
        )
        loss = torch.tensor(loss_host, device="cuda", dtype=torch.float64)
        transfers = torch.tensor(transfer.T.copy(), device="cuda", dtype=torch.float64)

        def rhs(y, hh):
            out = -1j * (hh @ y - y @ hh + det_diff * y) + loss * y
            incoming = y.diagonal(dim1=-2, dim2=-1).real @ transfers
            out.diagonal(dim1=-2, dim2=-1).add_(incoming)
            return out

        def block():
            y = state
            for i in range(chunk):
                k1 = rhs(y, h_input[i, 0])
                k2 = rhs(y + (0.5 * dt) * k1, h_input[i, 1])
                k3 = rhs(y + (0.5 * dt) * k2, h_input[i, 1])
                k4 = rhs(y + dt * k3, h_input[i, 2])
                y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            state.copy_(y)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            block()
        torch.cuda.current_stream().wait_stream(stream)
        sync()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block()
        sync()
        setup = time.perf_counter() - began
        times = []
        result = None
        for _repeat in range(2):
            state.copy_(initial)
            sync()
            tick = time.perf_counter()
            for begin in range(0, len(h), chunk):
                h_input.copy_(all_h[begin : begin + chunk])
                graph.replay()
            sync()
            times.append(time.perf_counter() - tick)
            tick = time.perf_counter()
            result = state.cpu().numpy()
            download = time.perf_counter() - tick
        peak = torch.cuda.max_memory_allocated()
    return result, {
        "batch": count,
        "dt_s": dt,
        "steps": len(h),
        "setup_s": setup,
        "evolution_s": times,
        "download_s": download,
        "peak_allocated_bytes": peak,
    }


def reference(context):
    q = context["prepared"]
    q.rust_plan = prepare_lindblad_problem_py(q.to_payload())
    det, js = points(12)
    rho = context["rho0"][js]
    slots = ["δ0", "center", "sigma_time", "ω1", "φ1", "ω2", "φ2"]
    values = np.zeros((12, 7), complex)
    values[:, 0] = 2 * np.pi * 1e6 * det
    values[:, 1] = context["physics"]["uv_length_m"] / 150 / 2
    values[:, 2] = np.sqrt(2) * context["sigma_I"] / 150
    tick = time.perf_counter()
    for segment in range(4):
        lo = START + segment * QUARTER
        mid = lo + QUARTER / 2
        a = np.sin(2 * np.pi * 1e6 * mid) > 0
        b = np.sin(2 * np.pi * 1e6 * mid + np.pi / 2) > 0
        values[:, 4] = np.pi / 2 if a else -np.pi / 2
        values[:, 6] = np.pi / 2 if b else -np.pi / 2
        result = lindblad.solve_lindblad_batch(
            q,
            rho,
            (lo, lo + QUARTER),
            parameter_slots=slots,
            parameter_batch=values,
            solver="tsit5",
            execution_mode="structured_sparse_dense_diag_soa_single_unpack",
            reltol=1e-10,
            abstol=1e-12,
            dt=1e-9,
            output="full",
            output_when="final",
            save_start=False,
            parallel=False,
            threads=1,
            maxiters=100000,
        )
        rho = np.asarray([q.layout.unpack(row) for row in result.values])
    return rho, time.perf_counter() - tick


def errors(actual, expected):
    diff = np.abs(actual - expected)
    mask = np.abs(expected) >= 1e-6
    return {
        "max_abs": float(diff.max()),
        "max_relative_above_1e_minus6_cutoff": float((diff[mask] / np.abs(expected[mask])).max()),
        "max_trace_error": float(np.abs(np.trace(actual, axis1=-2, axis2=-1) - 1).max()),
        "max_hermiticity_error": float(np.abs(actual - actual.conj().swapaxes(-2, -1)).max()),
    }


def main():
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=True)
    context, source, pointer = load()
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "production_signature": pointer["request_signature"],
        "precision": "complex128",
        "interval_s": [START, START + 1e-6],
        "velocity_m_s": 150,
        "method": "experimental fixed RK4 with CUDA graph chunks; two warm timing repetitions",
        "cpu_threads": 1,
        "cpu_comparison_contended_by_production": True,
        "rows": [],
    }
    started = time.perf_counter()
    ref, cpu_s = reference(context)
    report["rust_strict_12_serial_s"] = cpu_s
    print("Rust reference:", cpu_s, flush=True)
    for count, substeps in [(12, 128), (12, 256), (32, 128), (128, 128), (512, 128), (1024, 128)]:
        if time.perf_counter() - started > 180:
            report["budget_stop"] = True
            break
        torch.cuda.reset_peak_memory_stats()
        result, row = gpu(context, count, substeps)
        row["accuracy_first_12"] = errors(result[:12], ref)
        report["rows"].append(row)
        (OUT / "results.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
        np.savez_compressed(
            OUT / f"batch_{count}_substeps_{substeps}.npz", rho=result[:12], reference=ref
        )
        torch.cuda.empty_cache()
    report["wall_s"] = time.perf_counter() - started
    (OUT / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
