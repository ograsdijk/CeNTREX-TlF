"""Isolated prepared-assembly / structural-DAG CUDA experiments.

Monkeypatches only this benchmark process. Uses current library graphs/kernels,
archived RC/R2 fixtures and full complex-density validation. No production edits.
"""
# ruff: noqa: E402
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
OUT = HERE / "cuda_prepared_assembly_results"
os.environ["CUPY_CACHE_DIR"] = str(HERE / ".cupy_cache")

import numpy as np
import sympy as smp

from centrex_tlf import lindblad
from centrex_tlf.lindblad import _cuda_lowering as lowering
from centrex_tlf.lindblad import cuda

cp = cuda._cupy()
import rc_gpu_sparse_methods as fixtures
from run_rc_gpu_integrator_suite import reference

Compiler = lowering.ScalarCompiler
Integrator = cuda._Integrator
RawModule = cp.RawModule


class SharedCompiler(Compiler):
    def __init__(self, *args):
        super().__init__(*args)
        memo = {}

        def intern(node):
            if isinstance(node, tuple):
                return tuple(intern(v) if isinstance(v, lowering.Node | tuple) else v for v in node)
            if not isinstance(node, lowering.Node):
                return node
            args = tuple(intern(v) if isinstance(v, lowering.Node | tuple) else v for v in node.args)
            key = (node.op, args)
            if key not in memo:
                memo[key] = lowering.Node(node.op, args, node.symbolic, node.dependencies)
            return memo[key]

        self.coefficients = [intern(c) for c in self.coefficients]


def assembly_source(source, model):
    offdiag, diagonal, _, _ = model
    declarations = []
    for tag, bases, kernel in [("H", offdiag, "build_h"), ("D", diagonal, "build_diag")]:
        pointers, indices = [0], []
        for entry in range(bases.shape[1]):
            indices.extend((np.flatnonzero(bases[1:, entry] != 0) + 1).tolist())
            pointers.append(len(indices))
        declarations.extend([
            f"__device__ const int {tag}_ptr[]={{" + ",".join(map(str, pointers)) + "};",
            f"__device__ const int {tag}_k[]={{" + ",".join(map(str, indices or [0])) + "};",
        ])
        if tag == "H":
            old = "for(int k=0;k<K;k++){C v=bases[k*E+e]*coeff[k*B+(varied[k]?b:0)];z+=lower?conj(v):v;}"
            new = "z=lower?conj(bases[e]):bases[e];for(int p=H_ptr[e];p<H_ptr[e+1];p++){int k=H_k[p];C v=bases[k*E+e]*coeff[k*B+(varied[k]?b:0)];z+=lower?conj(v):v;}"
        else:
            old = "for(int k=0;k<K;k++)z+=bases[k*n+i]*coeff[k*B+(varied[k]?b:0)];"
            new = "z=bases[i];for(int p=D_ptr[i];p<D_ptr[i+1];p++){int k=D_k[p];z+=bases[k*n+i]*coeff[k*B+(varied[k]?b:0)];}"
        assert source.count(old) == 1, kernel
        source = source.replace(old, new)
    return "\n".join(declarations) + "\n" + source


def r2_fixture():
    import run_r2_gpu_reuse as r2

    a = dict(np.load(HERE / "r2_gpu_reuse_results/matched_cross_v1/inputs.npz"))
    context, f, _ = r2.ode_fixture(a)
    n = int(a["n"])
    h = f["basis"][4] + np.diag(f["diagonal"])
    detuning, drive = smp.Symbol("detuning", real=True), smp.Symbol("drive")
    H = smp.zeros(n)
    for i in range(n):
        H[i, i] = float(h[i, i].real) + detuning * float(a["detuning_mask"][i])
        for j in range(i + 1, n):
            if h[i, j] != 0:
                H[i, j] = complex(h[i, j]) * drive
                H[j, i] = smp.conjugate(H[i, j])
    jumps = []
    for i in range(n):
        for k in range(f["tp"][i], f["tp"][i + 1]):
            jump = np.zeros((n, n), complex)
            jump[i, f["tc"][k]] = np.sqrt(f["tr"][k])
            jumps.append(jump)
    params = lindblad.LindbladParameters()
    params.real("detuning", 0.)
    params.bind(drive, lindblad.phase_modulation(smp.Symbol("t", real=True), 3.8, 2 * np.pi * 1.6e6))
    prepared = lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=np.asarray(jumps)), params)
    refpath = HERE / "r2_gpu_reuse_results/eom_v1/eom_swap_const_exact_0.5ns.npz"
    ref = np.load(refpath)["rho"]
    det = np.repeat([-20., 0., 20.], 20)
    return prepared, context["rho0"], det, np.arange(60) % 20, (2 * np.pi * 1e6 * det)[:, None], ["detuning"], (125e-9, 1125e-9), 2e-9, ref, str(refpath)


def rc_fixture():
    context, source, _ = fixtures.load()
    det, js = fixtures.points(483)
    end = context["physics"]["uv_length_m"] / 150
    sigma = np.sqrt(2) * context["sigma_I"] / 150
    pars = np.column_stack((2 * np.pi * 1e6 * det, np.full(len(det), end / 2),
                           np.full(len(det), sigma), np.full(len(det), 2 * np.pi * 1e6),
                           np.zeros(len(det)), np.full(len(det), 2 * np.pi * 1e6), np.full(len(det), np.pi / 2)))
    ref, refpath = reference(dict(fixture="production", span_us=1))
    return context["prepared"], context["rho0"], det, js, pars, ["δ0", "center", "sigma_time", "ω1", "φ1", "ω2", "φ2"], (146e-6, 147e-6), 250e-9 / 56, ref, str(refpath)


def setup(prepared, initial, pars, names, span, dt, rhs, variant):
    held, sources = [], []

    class Experiment(Integrator):
        def __init__(self, *args, **kwargs):
            model = args[6]

            def compile_module(*a, **kw):
                if "assembly" in variant:
                    kw["code"] = assembly_source(kw["code"], model)
                sources.append(kw["code"])
                return RawModule(*a, **kw)

            cp.RawModule = compile_module
            try:
                super().__init__(*args, **kwargs)
            finally:
                cp.RawModule = RawModule
            held.append(self)

    cuda._Integrator = Experiment
    lowering.ScalarCompiler = SharedCompiler if "dag" in variant else Compiler
    try:
        answer = lindblad.solve_lindblad_batch(
            prepared, initial, span, parameter_batch=pars, parameter_slots=names,
            solver="fixed_tsit5", evolution_device="cuda", dt=dt,
            output="full", collect_stats=True, maxiters=1000000,
            cuda_options=lindblad.CudaSolverOptions(batch_size=len(initial), rhs=rhs),
        )
    finally:
        cuda._Integrator, lowering.ScalarCompiler = Integrator, Compiler
    return held[0], answer, sources[0]


def replay(g, span):
    cp.copyto(g.y, g.initial)
    g.clock.set(np.array([span[0], 0., 0., 0., 0., 0.]))
    g.error.fill(0)
    cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    g.advance(0, g.steps)
    host = g.y.get()
    return host, time.perf_counter() - start


def main():
    OUT.mkdir(exist_ok=True)
    result = dict(device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), cases=[])
    rng = np.random.default_rng(1234)
    variants = ["base", "assembly", "dag", "assembly_dag"]
    for system, factory in [("RC", rc_fixture), ("R2_EOM", r2_fixture)]:
        prepared, rhos, det, ids, pars0, names, span, dt, ref, refpath = factory()
        for batch in [512, 1024]:
            select = np.arange(batch) % len(ids)
            initial, pars = rhos[ids[select]], pars0[select]
            for rhs in ["tiled", "direct"]:
                objects, accuracy, setup_stats, source_counts = {}, {}, {}, {}
                for variant in variants:
                    g, answer, source = setup(prepared, initial, pars, names, span, dt, rhs, variant)
                    matrices = prepared.layout.unpack_batch(answer.values)
                    errors = fixtures.diagnostics(matrices, ref[select % len(ref)])
                    assert errors["passes_1percent_above_cutoff"] and errors["near_zero_max_abs"] <= 1e-6
                    assert errors["trace_error"] < 1e-6 and errors["min_eigenvalue"] >= -1e-7
                    accuracy[variant] = {k: v for k, v in errors.items() if "trajectory" not in k}
                    setup_stats[variant] = answer.solver_stats
                    source_counts[variant] = len(re.findall(r"C v\d+=", source))
                    (OUT / f"{system}_{rhs}_{variant}.cu").write_text(source)
                    objects[variant] = g
                baseline, _ = replay(objects["base"], span)
                timings = {v: [] for v in variants}
                maxdiff = {v: 0. for v in variants}
                for repeat in range(6):
                    for variant in rng.permutation(variants):
                        host, seconds = replay(objects[variant], span)
                        np.testing.assert_allclose(host, baseline, atol=1e-10, rtol=1e-9)
                        maxdiff[variant] = max(maxdiff[variant], float(np.max(abs(host - baseline))))
                        timings[variant].append(seconds)
                    print(f"{system} batch={batch} {rhs}: round {repeat + 1}/6", flush=True)
                row = dict(system=system, batch=batch, rhs=rhs, span=span, dt=dt,
                           reference=refpath, timings=timings, max_internal_difference=maxdiff,
                           accuracy=accuracy, setup=setup_stats, coefficient_temporaries=source_counts)
                result["cases"].append(row)
                (OUT / "results.json").write_text(json.dumps(result, indent=2))
                print({v: float(np.median(t)) for v, t in timings.items()}, flush=True)
                del objects, g
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
