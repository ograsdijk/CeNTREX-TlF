"""Isolated generic CUDA FMA experiment; never modifies package files.

Matched warm alternating pairs on cleanup-1 full/partial batches, plus
independent multi-drive density validation. This is a same-grid comparison,
not a timestep-convergence study or a full-production runtime estimate.
"""

# Configure numerical-library threads and the sibling workload path before imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "CeNTREX-SPB-testing"))
os.environ["CUPY_CACHE_DIR"] = str(ROOT / "benchmarks/.cupy_cache")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import _cuda_generic_baseline as baseline
import numpy as np
import sympy as smp
from threadpoolctl import threadpool_limits

from centrex_tlf import lindblad
from centrex_tlf.lindblad import _cuda_kernels as kernels
from centrex_tlf.lindblad.cuda import _cupy


def fma_source(source):
    replacements = {
        "r+=hr[h]*u-hi[h]*w;v+=hr[h]*w+hi[h]*u;": "r=fma(hr[h],u,r);r=fma(-hi[h],w,r);v=fma(hr[h],w,v);v=fma(hi[h],u,v);",
        "ar+=hr[h]*u-hi[h]*v;ai+=hr[h]*v+hi[h]*u;": "ar=fma(hr[h],u,ar);ar=fma(-hi[h],v,ar);ai=fma(hr[h],v,ai);ai=fma(hi[h],u,ai);",
        "jr+=hr[h]*u-hi[h]*v;ji+=hr[h]*v+hi[h]*u;": "jr=fma(hr[h],u,jr);jr=fma(-hi[h],v,jr);ji=fma(hr[h],v,ji);ji=fma(hi[h],u,ji);",
        "ar+=r*u-w*v;ai+=r*v+w*u;": "ar=fma(r,u,ar);ar=fma(-w,v,ar);ai=fma(r,v,ai);ai=fma(w,u,ai);",
        "jr+=r*u-w*v;ji+=r*v+w*u;": "jr=fma(r,u,jr);jr=fma(-w,v,jr);ji=fma(r,v,ji);ji=fma(w,u,ji);",
        "r+=hr[h]*sr[k]-hi[h]*si[k];v+=hr[h]*si[k]+hi[h]*sr[k];": "r=fma(hr[h],sr[k],r);r=fma(-hi[h],si[k],r);v=fma(hr[h],si[k],v);v=fma(hi[h],sr[k],v);",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


@contextmanager
def variant(mode):
    original = kernels.KERNELS, kernels.COMPONENT_KERNELS, kernels.tile_kernel
    if mode not in {"baseline", "fma"}:
        raise ValueError(f"Unknown CUDA arithmetic variant: {mode}")
    # FMA runs the installed package without patching it. The control is frozen
    # independently, so promoting FMA cannot silently turn this into FMA vs FMA.
    assert original[0] == fma_source(baseline.KERNELS)
    assert original[1] == fma_source(baseline.COMPONENT_KERNELS)
    assert original[2](26, 32) == fma_source(baseline.tile_kernel(26, 32))
    if mode == "baseline":
        kernels.KERNELS = baseline.KERNELS
        kernels.COMPONENT_KERNELS = baseline.COMPONENT_KERNELS
        kernels.tile_kernel = baseline.tile_kernel
    try:
        yield
    finally:
        kernels.KERNELS, kernels.COMPONENT_KERNELS, kernels.tile_kernel = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--duration-us", type=float, default=2.0)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or args.duration_us <= 0:
        parser.error("rounds must be positive/even; duration must be positive")
    out = (
        ROOT
        / "benchmarks/cuda_generic_fma_results"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True)
    print(f"Results: {out}", flush=True)
    cp = _cupy()
    paths = sorted((ROOT / "centrex_tlf/lindblad").glob("*cuda*.py"))

    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    data = dict(
        status="preparing",
        arguments=vars(args),
        records=[],
        validation=[],
        warmups=[],
        hashes_before=hashes(),
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        cupy=cp.__version__,
        numpy=np.__version__,
        started_utc=datetime.now(UTC).isoformat(),
    )

    def save():
        temp = out / "results.tmp"
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp.replace(out / "results.json")

    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out / "_cuda_generic_baseline.py").write_bytes(Path(baseline.__file__).read_bytes())
    data["arithmetic"] = {
        "fma": "unmodified installed package",
        "baseline": "frozen pre-FMA source",
    }
    for mode in ("baseline", "fma"):
        with variant(mode):
            (out / f"{mode}.cu").write_text(
                kernels.HEADER + kernels.KERNELS + kernels.COMPONENT_KERNELS, encoding="utf-8"
            )
    save()
    options = {
        "direct": dict(rhs="direct", hamiltonian_sharing="off", density_reduction=False),
        "component": dict(rhs="direct", hamiltonian_sharing="on", density_reduction=True),
        "product": dict(
            rhs="tiled", tile_size=0, hamiltonian_sharing="off", density_reduction=False
        ),
        "tiled": dict(
            rhs="tiled", tile_size=32, hamiltonian_sharing="off", density_reduction=False
        ),
    }

    def solve(p, initial, pars, slots, span, dt, path, mode, solver="fixed_tsit5", snapshots=False):
        opt = lindblad.CudaSolverOptions(batch_size=1024, rhs_specialization="off", **options[path])
        cp.cuda.Stream.null.synchronize()
        tick = time.perf_counter()
        with variant(mode):
            result = lindblad.solve_lindblad_batch(
                p,
                initial,
                span,
                parameter_slots=slots,
                parameter_batch=pars,
                solver=solver,
                evolution_device="cuda",
                dt=dt,
                maxiters=1_000_000,
                output="full",
                output_when="saveat" if snapshots else "final",
                **(dict(saveat=np.linspace(*span, 5)[1:]) if snapshots else {}),
                collect_stats=True,
                cuda_options=opt,
            )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - tick
        chunk = result.solver_stats["chunks"][0]
        assert chunk["rhs_specialization"]["kernel"] == "generic"
        if path == "component":
            assert chunk["shared_hamiltonian_components"] > 0, chunk
        return result, elapsed

    def check(p, values, ref):
        np.testing.assert_allclose(values, ref, rtol=1e-10, atol=1e-12)
        rho = p.layout.unpack_batch(values)
        trace_error = float(np.max(abs(np.trace(rho, axis1=-2, axis2=-1) - 1)))
        hermitian_error = float(np.max(abs(rho - rho.swapaxes(-1, -2).conj())))
        assert trace_error < 1e-9
        return dict(
            max_density_difference=float(np.max(abs(values - ref))),
            trace_error=trace_error,
            hermitian_error=hermitian_error,
        )

    try:
        # Multiple independent shared components, complex edges, continuous
        # diagonal variation, coherent initial state, and an odd graph tail.
        a, b, d = smp.symbols("a b d", real=True)
        H = smp.Matrix([[0, a * (1 + 0.3 * smp.I), b], [a * (1 - 0.3 * smp.I), d, 0], [b, 0, -0.2]])
        C = np.zeros((2, 3, 3), complex)
        C[0, 0, 1], C[1, 2, 1] = np.sqrt(0.3), np.sqrt(0.2)
        params = lindblad.LindbladParameters()
        t = params.time()
        pa, pb = params.real("pa", 1.0), params.real("pb", 1.0)
        params.bind(a, pa * (1 + 0.2 * lindblad.sine(t, angular_frequency=2.0)), finalize=False)
        params.bind(b, pb * (1 + 0.3 * lindblad.sine(t, angular_frequency=3.0)), finalize=False)
        params.bind(d, 0.4 + 0.1 * lindblad.sine(t, angular_frequency=1.7), finalize=False)
        params._finalize()
        p = lindblad.prepare_lindblad_problem(
            SimpleNamespace(H_symbolic=H, C_array=C),
            params,
            backend="rust",
            hamiltonian_representation="decomposed",
        )
        vectors = np.array([[1, 1j, 0.3], [0.2j, 1, 1j], [1, 0, 0]], complex)
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        initial = vectors[:, :, None] * vectors[:, None, :].conj()
        pars = np.array([[1.0, 0.7], [0.2, 1.3], [0.0, 0.0]], complex)
        for solver in ("fixed_tsit5", "fixed_ck54"):
            ref = None
            for path in options:
                for mode in ("baseline", "fma"):
                    result, elapsed = solve(
                        p,
                        initial,
                        pars,
                        ["pa", "pb"],
                        (0.0, 1.003),
                        0.003,
                        path,
                        mode,
                        solver,
                        True,
                    )
                    if ref is None:
                        ref = result.values.copy()
                    metrics = check(p, result.values, ref)
                    data["validation"].append(
                        dict(
                            workload="multi_drive",
                            solver=solver,
                            path=path,
                            mode=mode,
                            wall_seconds=elapsed,
                            **metrics,
                            stats=result.solver_stats,
                        )
                    )
                    np.savez_compressed(
                        out / f"multi_{solver}_{path}_{mode}.npz",
                        values=result.values,
                        initial=initial,
                        parameters=pars,
                    )
            print(f"Multi-drive validation passed: {solver}", flush=True)
            save()

        from itertools import product

        from simulations.optical_bloch.state_cleanup_1 import engine, run

        context = engine.prepare()
        p = context["prepared"]
        for mode in ("baseline", "fma"):
            with variant(mode):
                (out / f"{mode}_tile.cu").write_text(
                    kernels.tile_kernel(p.layout.n, 32), encoding="utf-8"
                )
        settings = list(
            product(
                run.POWER_W,
                run.DETUNING_MHZ,
                run.TRANSVERSE_M_S,
                run.Y0_M,
                range(len(context["rho0"])),
            )
        )
        tail = len(settings) % 1024
        assert tail
        selections = {
            1024: np.arange(len(settings) - tail - 1024, len(settings) - tail),
            tail: np.arange(len(settings) - tail, len(settings)),
        }
        middle = len(engine.PASS_CENTERS_M) // 2
        jump = float((engine.PASS_CENTERS_M[middle] + engine.PASS_CENTERS_M[middle + 1]) / 300.0)
        span = (jump - args.duration_us * 0.5e-6, jump + args.duration_us * 0.5e-6)
        data.update(
            status="timing",
            levels=p.layout.n,
            span=span,
            dt=2e-9,
            batch_sizes=list(selections),
            scope="Matched short cleanup-1 intervals; no timestep-convergence claim.",
        )
        save()
        rng = np.random.default_rng(20261021)
        for size, ids in selections.items():
            rows = [settings[i] for i in ids]
            pars = np.asarray(
                [
                    [2 * np.pi * d * 1e6, 2 * np.pi * vt / engine.WAVELENGTH_M, 150.0, h, power]
                    for power, d, vt, h, initial in rows
                ],
                complex,
            )
            starts = context["rho0"][[r[-1] for r in rows]]
            ref = None
            for path in options:
                for mode in ("baseline", "fma"):
                    result, elapsed = solve(p, starts, pars, engine.SLOTS, span, 2e-9, path, mode)
                    if ref is None:
                        ref = result.values.copy()
                    metrics = check(p, result.values, ref)
                    data["warmups"].append(
                        dict(
                            batch=size,
                            path=path,
                            mode=mode,
                            wall_seconds=elapsed,
                            **metrics,
                            stats=result.solver_stats,
                        )
                    )
                    np.savez_compressed(
                        out / f"cleanup_{size}_{path}_{mode}.npz",
                        values=result.values,
                        initial=starts,
                        parameters=pars,
                        selection_indices=ids,
                    )
                    print(f"Warm {size} {path} {mode}: {elapsed:.3f}s", flush=True)
                orders = [0, 1] * (args.rounds // 2)
                rng.shuffle(orders)
                for repeat, reverse in enumerate(orders):
                    for mode in ("fma", "baseline") if reverse else ("baseline", "fma"):
                        result, elapsed = solve(
                            p, starts, pars, engine.SLOTS, span, 2e-9, path, mode
                        )
                        metrics = check(p, result.values, ref)
                        data["records"].append(
                            dict(
                                batch=size,
                                path=path,
                                mode=mode,
                                repeat=repeat,
                                wall_seconds=elapsed,
                                **metrics,
                                stats=result.solver_stats,
                            )
                        )
                        save()
                    print(f"Pair {repeat + 1}/{args.rounds}: {size} {path}", flush=True)
            save()
        summaries = []
        for size in selections:
            for path in options:
                records = [r for r in data["records"] if r["batch"] == size and r["path"] == path]
                base = [r["wall_seconds"] for r in records if r["mode"] == "baseline"]
                fma = [r["wall_seconds"] for r in records if r["mode"] == "fma"]
                reductions = 1 - np.array(fma) / base
                summaries.append(
                    dict(
                        batch=size,
                        path=path,
                        baseline_median_s=float(np.median(base)),
                        fma_median_s=float(np.median(fma)),
                        paired_reduction_median=float(np.median(reductions)),
                        paired_reduction_min=float(min(reductions)),
                        paired_reduction_max=float(max(reductions)),
                        pairs_favoring_fma=int(np.sum(reductions > 0)),
                    )
                )
        data.update(status="complete", summaries=summaries)
        print(json.dumps(summaries, indent=2), flush=True)
    except BaseException as exc:
        data.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        data.update(
            hashes_after=hashes(),
            package_unchanged=data["hashes_before"] == hashes(),
            finished_utc=datetime.now(UTC).isoformat(),
        )
        save()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
