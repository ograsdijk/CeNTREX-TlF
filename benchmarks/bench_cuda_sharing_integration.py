"""Focused acceptance checks of the package CUDA sharing implementation."""

from __future__ import annotations

import argparse
import hashlib
import itertools

import bench_spb_cu1_gpu as bench
import numpy as np

from centrex_tlf import hamiltonian, lindblad

OUT = bench.OUT / "integrated_sharing"
MODES = {"off": ("off", "off"), "coefficients": ("auto", "off"), "combined": ("auto", "auto")}


def fixture(name, varying):
    if name == "cu1":
        p, rho, idx, meta = bench.build()
        slots = bench.SLOTS
        pars = (
            bench.rows(np.linspace(17, 22, 16), np.linspace(-1, 1, 16), np.linspace(0, 0.0085, 4))
            if varying
            else bench.rows(np.linspace(17, 22, 32), np.linspace(-1, 1, 32), [0.0])
        )
        return p, rho, idx, meta, pars, slots, (18e-6, 30e-6), 2.0
    from cross_system_fixtures import build_fixture

    f = build_fixture("rc" if name == "rc" else "r2_f4")
    p, rho = f.prepared, f.rho0
    idx = np.asarray(f.group_indices)
    meta = dict(configuration=f.configuration, metadata=f.metadata, benchmark_fixture=True)
    if name == "r2":
        # The fresh transverse-field CPU fixture has a tiny imaginary diagonal
        # from basis transformation roundoff. The prior GPU validation explicitly
        # projected this diagonal to real; apply the same bounded projection here.
        from types import SimpleNamespace

        import sympy as smp

        static = np.asarray(p.hamiltonian_plan["static_matrix"], complex)
        noise = np.diag(static).imag
        assert abs(noise).max() < 512 * np.finfo(float).eps * max(1.0, abs(static).max())
        H = f.system.H_symbolic.copy()
        for i, v in enumerate(noise):
            H[i, i] -= smp.I * float(v)
        p = lindblad.prepare_lindblad_problem(
            SimpleNamespace(H_symbolic=H, C_array=f.system.C_array),
            p.parameters,
            backend="rust",
            hamiltonian_representation="decomposed",
        )
        meta["diagonal_roundoff_projection_rad_s"] = float(abs(noise).max())
    if name == "rc":
        # Keep the existing numerical RC fixture (including its declared DC fields),
        # rather than claim this is a new experimental RC recipe.
        if varying:
            base = dict(f.rc_case.params_dynamic.base_parameters)
            compound = dict(f.rc_case.params_dynamic.compound_parameters)
            base["height"] = 0.0
            compound["Ω0"] = "(" + str(compound["Ω0"]) + ")*exp(-height**2/(4*0.0075**2))"
            params = lindblad.LindbladParameters(base, compound)
            p = lindblad.prepare_lindblad_problem(
                f.system, params, backend="rust", hamiltonian_representation="decomposed"
            )
            gamma = float(hamiltonian.Γ)
            slots = ["δ0", "height", "Ωt0", "Ω1", "Ω2"]
            pars = np.asarray(
                [
                    [
                        2 * np.pi * 1e6 * d,
                        y,
                        gamma * np.sqrt(uv),
                        gamma * np.sqrt(m1),
                        gamma * np.sqrt(m2),
                    ]
                    for d, y, uv, m1, m2 in itertools.product(
                        np.linspace(-2, 2, 16),
                        np.linspace(0, 0.0085, 4),
                        [0.25, 0.5, 0.75, 1.0],
                        [0.5, 1.0],
                        [0.5, 1.0],
                    )
                ]
            )
            meta["scan_addition"] = (
                "Static Gaussian vertical UV attenuation, sigma_y=7.5 mm; independently scanned relative UV and MW powers, shared forward/time profile. Not a production recipe."
            )
        else:
            slots = ["δ0"]
            pars = (2 * np.pi * 1e6 * np.linspace(-2, 2, 1024))[:, None]
        return p, rho, idx, meta, pars, slots, (0.125e-6, 1.125e-6), 2.0
    return (
        p,
        rho,
        idx,
        meta,
        (2 * np.pi * 1e6 * np.linspace(-2, 2, 1024))[:, None],
        ["detuning"],
        (0.0, 1e-6),
        2.0,
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--fixture", choices=["cu1", "rc", "r2"], default="cu1")
    cli.add_argument("--stage", choices=["accuracy", "timing", "full"], default="accuracy")
    cli.add_argument("--varying", action="store_true")
    cli.add_argument("--force-components", action="store_true")
    args = cli.parse_args()
    if args.force_components:
        MODES["forced_components"] = ("auto", "on")
    OUT.mkdir(exist_ok=True, parents=True)
    p, rho, idx, meta, pars, slots, span, dt = fixture(args.fixture, args.varying)
    target = OUT / f"{args.fixture}_{'varying' if args.varying else 'shared'}_{args.stage}.json"
    result = dict(
        fixture=args.fixture,
        varying=args.varying,
        stage=args.stage,
        recipe=meta,
        span=span,
        dt_ns=dt,
        modes=MODES,
        cases=[],
        source_sha256={
            n: hashlib.sha256((bench.ROOT / n).read_bytes()).hexdigest()
            for n in [
                "centrex_tlf/lindblad/cuda.py",
                "centrex_tlf/lindblad/_cuda_lowering.py",
                "centrex_tlf/lindblad/_cuda_sharing.py",
                "centrex_tlf/lindblad/_cuda_kernels.py",
            ]
        },
    )
    if args.stage == "accuracy":
        pars = pars[np.rint(np.linspace(0, len(pars) - 1, 12)).astype(int)]
        if args.fixture == "cu1":
            saved = np.load(bench.CACHE / "reference_0.5ns.npz")
            pars = saved["parameters"]
            span = (0.0, bench.T_END)
        elif args.fixture == "rc":
            span = (0.125e-6, 10.125e-6)
    elif args.stage == "full":
        assert args.fixture == "cu1" and args.varying
        span = (0.0, bench.T_END)
    result["span"] = span
    oldslots = bench.SLOTS
    oldcheckpoints = bench.CHECKPOINTS
    bench.SLOTS = slots
    if args.fixture != "cu1":
        bench.CHECKPOINTS = np.linspace(span[0], span[1], 5)[1:]
    try:

        def run(mode, full):
            a, b = MODES[mode]
            option = lindblad.CudaSolverOptions(
                batch_size=1024,
                rhs="direct",
                density_reduction=True,
                graph_steps=8,
                coefficient_sharing=a,
                hamiltonian_sharing=b,
            )
            return bench.solve(
                p, rho, pars, "fixed_tsit5", dt, option, span=span, full=full, profile=True
            )

        if args.stage == "accuracy":
            for mode in MODES:
                bench.log(f"{args.fixture} accuracy {mode}, {len(pars)} trajectories")
                a, t = run(mode, True)
                if mode == "off":
                    reference = a.values.copy()
                check = bench.accuracy(p, a.values, reference, idx)
                assert check["max_absolute"] < 1e-10, check
                strict = (
                    bench.accuracy(p, a.values, saved["values"], idx)
                    if args.fixture == "cu1"
                    else None
                )
                if strict is not None:
                    assert strict["passed"], strict
                result["cases"].append(
                    dict(mode=mode, same_step=check, strict_reference=strict, **t)
                )
                bench.atomic(target, result)
        else:
            modes = list(MODES) if args.stage == "timing" else ["off", "combined"]
            if args.stage == "timing":
                for mode in modes:
                    bench.log(f"Warmup {args.fixture} {mode}")
                    run(mode, False)
            rng = np.random.default_rng(781)
            for repeat in range(3 if args.stage == "timing" else 1):
                for mode in rng.permutation(modes):
                    bench.log(f"{args.fixture} {args.stage} {mode}, round {repeat + 1}")
                    a, t = run(mode, False)
                    trace = float(abs(a.values.sum(axis=-1) - 1).max())
                    assert trace < 1e-6, trace
                    if args.stage == "full":
                        if repeat == 0 and "full_reference" not in locals():
                            full_reference = a.values.copy()
                        np.testing.assert_allclose(a.values, full_reference, rtol=1e-10, atol=1e-12)
                    result["cases"].append(
                        dict(mode=str(mode), repeat=repeat, max_trace_error=trace, **t)
                    )
                    bench.atomic(target, result)
                    bench.log(
                        f"{mode}: {t['wall_seconds']:.3f}s; H {t['stats']['chunks'][0]['hamiltonian_sharing']}"
                    )
        result["status"] = "complete"
        bench.atomic(target, result)
    finally:
        bench.SLOTS = oldslots
        bench.CHECKPOINTS = oldcheckpoints


if __name__ == "__main__":
    main()
