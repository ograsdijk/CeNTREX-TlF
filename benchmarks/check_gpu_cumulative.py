"""Matched archived full-storage Tsit5 versus current public CUDA solver.

Requires the archived local RC/R2 inputs described in PUBLICATION_INDEX.md.
No production model or solver defaults are changed.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUPY_CACHE_DIR"] = str(ROOT / "benchmarks/.cupy_cache")
# Use the publication-validated wheel, isolating unrelated working-tree physics.
SITE = ROOT / ".cache/publication/installed"
if SITE.exists():
    sys.path.insert(0, str(SITE))

import check_cuda_algorithms as alg
import numpy as np
import rc_gpu_sparse_methods as fixtures
from rc_gpu_integrators import IntegratorGPU

from centrex_tlf import lindblad
from centrex_tlf.lindblad import cuda

cp = alg.cp
OUT = ROOT / "benchmarks/gpu_cumulative_results"


def unpack(g, host, layout):
    pp, pi = g.public_pk.get(), g.public_im.get().astype(bool)
    packed = np.zeros((g.B, g.n * g.n))
    real, imag = host[: g.P].reshape(-1, g.B), host[g.P : 2 * g.P].reshape(-1, g.B)
    keep = (pp >= 0) & ~pi
    packed[:, keep] = real[pp[keep]].T
    keep = (pp >= 0) & pi
    packed[:, keep] = imag[pp[keep]].T
    return layout.unpack_batch(packed)


def original(system, p, packed, pars, span, dt):
    saved = fixtures.fixture, fixtures.points
    try:
        if system == "RC":
            context, _, _ = fixtures.load()
        else:
            import run_r2_gpu_reuse as r2

            a = dict(np.load(ROOT / "benchmarks/r2_gpu_reuse_results/matched_cross_v1/inputs.npz"))
            context, f, _ = r2.ode_fixture(a)
            f["mask"] = a["detuning_mask"].copy()
            f["base"].update({"β0": 3.8, "ωphase0": 2 * np.pi * 1.6e6})
        context = dict(context, rho0=p.layout.unpack_batch(packed))
        fixtures.points = lambda count, pattern="grid": (
            pars[:, 0] / (2 * np.pi * 1e6),
            np.arange(count),
        )
        g = IntegratorGPU(
            context,
            len(packed),
            *span,
            dt_ns=dt * 1e9,
            integrator="tsit5",
            variant="base",
            chunk=16,
            velocity=150.0,
        )
        g.capture()
        return g
    finally:
        fixtures.fixture, fixtures.points = saved


def run_case(system, batch, duration, repeats):
    print(f"{system}: loading {batch} trajectories, {duration:g} us", flush=True)
    p, packed, pars, names, span, dt = alg.fixture(system, batch, duration)
    initial = p.layout.unpack_batch(packed)

    def solve(rhos, parameters, solver, step, option):
        return lindblad.solve_lindblad_batch(
            p,
            rhos,
            span,
            parameter_batch=parameters,
            parameter_slots=names,
            solver=solver,
            evolution_device="cuda",
            dt=step,
            output="full",
            maxiters=1000000,
            collect_stats=True,
            cuda_options=option,
        )

    # Distinct detuning/initial combinations: no redundant reference integration.
    unique, inverse = np.unique(np.c_[pars, packed], axis=0, return_inverse=True)
    refpars, refinitial = unique[:, : len(names)], unique[:, len(names) :]
    fine = {}
    for divisor in [2, 4]:
        print(f"{system}: reference dt/{divisor}, {len(unique)} distinct cases", flush=True)
        fine[divisor] = solve(
            refinitial,
            refpars,
            "fixed_tsit5",
            dt / divisor,
            lindblad.CudaSolverOptions(batch_size=batch),
        ).values
    convergence = alg.compare(p.layout, fine[2], fine[4])
    reference = fine[4][inverse]
    print(f"{system}: reference convergence {convergence}", flush=True)
    objects, setup, stats = {}, {}, {}
    start = time.perf_counter()
    objects["original_full_tsit5"] = original(system, p, packed, pars, span, dt)
    setup["original_full_tsit5"] = time.perf_counter() - start
    choices = {
        "current_default_tsit5": ("fixed_tsit5", dt, False),
        "current_reduced_tsit5": ("fixed_tsit5", dt, True),
    }
    if system != "RC":
        choices["current_ck54"] = ("fixed_ck54", 1.95e-9, False)
    native = cuda._Integrator
    for name, (solver, step, reduced) in choices.items():
        held = []

        class Capture(native):
            def __init__(self, *a, _held=held, **kw):
                super().__init__(*a, **kw)
                _held.append(self)

        cuda._Integrator = Capture
        try:
            start = time.perf_counter()
            answer = solve(
                initial,
                pars,
                solver,
                step,
                lindblad.CudaSolverOptions(batch_size=batch, density_reduction=reduced),
            )
            setup[name] = time.perf_counter() - start
            stats[name] = answer.solver_stats
            objects[name] = held[0]
        finally:
            cuda._Integrator = native
        print(f"{system}: prepared {name}", flush=True)

    def replay(name):
        g = objects[name]
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        if name == "original_full_tsit5":
            matrices, _, end = g.evolve(repeats=1)
            assert abs(end - span[1]) < 1e-13
        else:
            host, _ = alg.bench.previous.replay(g, span)
            matrices = unpack(g, host, p.layout)
            assert int(g.error.get()[0]) == 0
        wall = time.perf_counter() - start
        return matrices, wall

    timings, checks, differences = {k: [] for k in objects}, {}, {}
    outputs = {}
    for name in objects:
        outputs[name], _ = replay(name)
        checks[name] = alg.compare(p.layout, p.layout.pack_batch(outputs[name]), reference)
        print(f"{system}: {name} accuracy {checks[name]}", flush=True)
    base = outputs["original_full_tsit5"]
    for name, matrices in outputs.items():
        differences[name] = float(np.max(abs(matrices - base)))
    rng = np.random.default_rng(20361)
    for repeat in range(repeats):
        for name in rng.permutation(list(objects)):
            matrices, wall = replay(name)
            timings[name].append(wall)
            np.testing.assert_allclose(matrices, outputs[name], atol=1e-12, rtol=1e-10)
            print(f"{system} round {repeat + 1}/{repeats}: {name} {wall:.3f} s", flush=True)
    row = dict(
        system=system,
        levels=p.layout.n,
        batch=batch,
        duration_us=duration,
        span=span,
        baseline_step=dt,
        timings=timings,
        checks=checks,
        reference_convergence=convergence,
        setup=setup,
        current_stats=stats,
        max_abs_difference_from_original=differences,
        distinct_parameters_and_initials=len(unique),
    )
    objects.clear()
    outputs.clear()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--duration-us", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--systems", nargs="+", default=["RC", "R2_EOM"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    result = dict(
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        package=str(Path(lindblad.__file__)),
        cases=[],
        timing_scope="Warm graph replay, reset, device-to-host transfer and full complex density reconstruction; setup reported separately. No disk writes in timings.",
        baseline="Archived full-storage split-real FP64 Tsit5 variant=base; same steps as current Tsit5. This baseline already uses sparse RHS and CUDA graphs.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for system in args.systems:
        result["cases"].append(run_case(system, args.batch, args.duration_us, args.repeats))
        (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
