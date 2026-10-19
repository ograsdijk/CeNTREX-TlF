"""Follow-up checks against best current solver and archived full transits."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import check_cuda_algorithms as alg
import numpy as np
import sympy as smp

from centrex_tlf import hamiltonian, lindblad


def run_integrator(g, p, span):
    host, elapsed = alg.bench.previous.replay(g, span)
    if hasattr(g, "compact_pairs"):
        i, j = g.compact_pairs
        z = (host[: g.P] + 1j * host[g.P :]).reshape(-1, g.B).T
        rho = np.zeros((g.B, g.n, g.n), complex)
        rho[:, i, j] = z
        rho[:, j, i] = z.conj()
    else:
        rho = alg.bench.internal_density(g, host, p.layout)
    return p.layout.pack_batch(rho), elapsed


def r2_comparison():
    p, packed, pars, names, span, dt = alg.fixture("R2_EOM", 1024, 10.0)
    components, labels = alg.matrices(p)
    keep = alg.closure(components, packed)
    generators = {}
    for method, solver, step in [
        ("tsit5", "fixed_tsit5", dt),
        ("ck54", "fixed_ck54", 1.95e-9),
        ("cf4_taylor8", "fixed_tsit5", 8e-9),
    ]:
        g, src = alg.allocate(p, packed, pars, names, span, step, solver=solver)
        if method == "cf4_taylor8":
            alg.exponential(g, src, components, labels, keep, packed, span, 8, 1)
        generators[method] = g
    distinct, inverse = np.unique(np.c_[pars, packed], axis=0, return_inverse=True)
    reference = lindblad.solve_lindblad_batch(
        p,
        distinct[:, len(names) :],
        span,
        parameter_batch=distinct[:, : len(names)],
        parameter_slots=names,
        solver="fixed_tsit5",
        evolution_device="cuda",
        dt=dt / 2,
        output="full",
    )
    timings = {name: [] for name in generators}
    checks = {}
    rng = np.random.default_rng(91245)
    for repeat in range(3):
        for name in rng.permutation(list(generators)):
            values, elapsed = run_integrator(generators[name], p, span)
            timings[name].append(elapsed)
            checks[name] = alg.compare(p.layout, values, reference.values[inverse])
        print(
            "R2 interleaved finalists",
            repeat + 1,
            {k: round(v[-1], 4) for k, v in timings.items()},
            flush=True,
        )
    result = dict(
        system="R2_EOM",
        batch=1024,
        span=span,
        timings=timings,
        checks=checks,
        ck54_step_ns=1.95,
        cf4_step_ns=8.0,
        cf4_degree=8,
    )
    alg.save("r2_finalists.json", result)
    del generators, g
    gc.collect()
    alg.cp.get_default_memory_pool().free_all_blocks()


def full_transits():
    results = []
    for system in ("RC", "R2_EOM"):
        if system == "RC":
            from run_rc_gpu_integrator_suite import PREVIOUS

            context, _, _ = alg.bench.previous.fixtures.load()
            p = context["prepared"]
            det, ids = np.repeat([-20.0, 0.0, 20.0], 3), np.tile(np.arange(3), 3)
            packed = p.layout.pack_batch(context["rho0"][ids])
            end = context["physics"]["uv_length_m"] / 150
            sigma = np.sqrt(2) * context["sigma_I"] / 150
            names = ["δ0", "center", "sigma_time", "ω1", "φ1", "ω2", "φ2"]
            pars = np.column_stack(
                (
                    2 * np.pi * 1e6 * det,
                    *[
                        np.full(9, x)
                        for x in [end / 2, sigma, 2 * np.pi * 1e6, 0.0, 2 * np.pi * 1e6, np.pi / 2]
                    ],
                )
            )
            oldrows = json.loads((PREVIOUS / "results.json").read_text())
            old = next(
                x
                for x in oldrows
                if x["job"].get("kind") == "full"
                and x["job"].get("velocity") == 150.0
                and x["job"].get("method") == "staged_soa_minor"
                and x["job"].get("substeps") == 512
            )
            reference_path = Path(old["output"]).with_suffix(".npz")
            reference = np.load(reference_path)["density_matrix"]
            dt = 250e-9 / 56
            darkp = p
            darknames = [*names, "Ωt0", "mw_peak1", "mw_peak2"]
            darkpars = np.c_[pars, np.zeros((9, 3))]
        else:
            p, rhos, _, ids, pars, names, _, _, _, _ = alg.bench.previous.r2_fixture()
            packed = p.layout.pack_batch(rhos[ids])
            end = 0.02 / 150
            reference_path = (
                alg.bench.ROOT
                / "gpu_next_results/v1_20261012_174509/052_r2_full_reference_tsit5_b60_h0.5_th256_tile32_wave0.npz"
            )
            reference = np.load(reference_path)["rho"]
            dt = 8e-9
            H = smp.diag(*p.obe_system.H_symbolic.diagonal())
            darkp = lindblad.prepare_lindblad_problem(
                SimpleNamespace(H_symbolic=H, C_array=p.dense_c_array), {"detuning": 0.0}
            )
            darknames, darkpars = names, pars
        span = (0.0, end)
        components, labels = alg.matrices(p)
        keep = alg.closure(components, packed)
        print(
            f"Full transit {system}, {len(packed)} cases, {end * 1e6:g} us + dark decay", flush=True
        )
        tick = time.perf_counter()
        g, src = alg.allocate(p, packed, pars, names, span, dt)
        if system == "RC":
            alg.compact_structured(g, src, p, keep, packed, span)
        else:
            alg.exponential(g, src, components, labels, keep, packed, span, 8, 1)
        light, _ = run_integrator(g, p, span)
        dark = lindblad.solve_lindblad_batch(
            darkp,
            light,
            (end, end + 25 / float(hamiltonian.Γ)),
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=2e-9,
            output="full",
            parameter_slots=darknames,
            parameter_batch=darkpars,
            maxiters=1000000,
        )
        actual = np.stack(
            (p.layout.unpack_batch(light), p.layout.unpack_batch(dark.values)), axis=1
        )
        checks = [
            alg.compare(
                p.layout, p.layout.pack_batch(actual[:, i]), p.layout.pack_batch(reference[:, i])
            )
            for i in range(2)
        ]
        result = dict(
            system=system,
            velocity_m_s=150.0,
            cases=len(packed),
            light_end_seconds=end,
            elapsed_seconds=time.perf_counter() - tick,
            reference=str(reference_path),
            light_check=checks[0],
            after_decay_check=checks[1],
        )
        results.append(result)
        alg.save("full_transit_finalists.json", results)
        print(result, flush=True)
        del g
        gc.collect()
        alg.cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--comparison-only", action="store_true")
    args = parser.parse_args()
    if not args.full_only:
        r2_comparison()
    if not args.comparison_only:
        full_transits()
