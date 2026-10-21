"""Full-flight same-grid confirmation for the generic FMA benchmark."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from bench_cuda_generic_fma import ROOT, baseline, lindblad, np, threadpool_limits, variant
from simulations.optical_bloch.state_cleanup_1 import engine

from centrex_tlf.lindblad.cuda import _cupy


def main():
    out = (
        ROOT
        / "benchmarks/cuda_generic_fma_validation_results"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True)
    print(f"Results: {out}", flush=True)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out / "bench_cuda_generic_fma.py").write_bytes(
        Path(__file__).with_name("bench_cuda_generic_fma.py").read_bytes()
    )
    (out / "_cuda_generic_baseline.py").write_bytes(Path(baseline.__file__).read_bytes())
    context = engine.prepare()
    p = context["prepared"]
    physical = [
        (0.080, 0.0, 0.0, 0.0),
        (0.080, 20.0, -1.5, 0.0),
        (0.080, 40.0, 1.5, 0.0),
        (0.080, 60.0, 0.0, 0.0),
        (0.080, 100.0, 1.5, 0.0),
        (0.010, -20.0, -1.5, 0.0085),
        (0.010, 40.0, 0.0, 0.0),
        (0.040, 60.0, 0.0, 0.0085),
    ]
    rows = [(*row, initial) for row in physical for initial in range(len(context["rho0"]))]
    starts = context["rho0"][[r[-1] for r in rows]]
    pars = np.asarray(
        [
            [2 * np.pi * d * 1e6, 2 * np.pi * vt / engine.WAVELENGTH_M, 150.0, h, power]
            for power, d, vt, h, initial in rows
        ],
        complex,
    )
    span = (0.0, float(engine.LENGTH_M / 150.0))
    times = np.linspace(*span, 5)[1:]
    cp = _cupy()
    data = dict(
        status="running",
        span=span,
        times=times.tolist(),
        rows=rows,
        dt=2e-9,
        records=[],
        arithmetic={"fma": "unmodified installed package", "baseline": "frozen pre-FMA source"},
        note="Same-grid density agreement; not timestep convergence.",
    )

    def save():
        (out / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    save()
    reference = None
    try:
        for path in ("direct", "component", "product", "tiled"):
            for mode in ("baseline", "fma"):
                options = lindblad.CudaSolverOptions(
                    batch_size=1024,
                    rhs_specialization="off",
                    graph_steps=16,
                    rhs="direct" if path in {"direct", "component"} else "tiled",
                    hamiltonian_sharing="on" if path == "component" else "off",
                    density_reduction=path == "component",
                    tile_size=0 if path == "product" else 32,
                )
                tick = time.perf_counter()
                with variant(mode):
                    result = lindblad.solve_lindblad_batch(
                        p,
                        starts,
                        span,
                        parameter_slots=engine.SLOTS,
                        parameter_batch=pars,
                        solver="fixed_tsit5",
                        evolution_device="cuda",
                        dt=2e-9,
                        maxiters=1_000_000,
                        output="full",
                        output_when="saveat",
                        saveat=times,
                        cuda_options=options,
                        collect_stats=True,
                    )
                cp.cuda.Stream.null.synchronize()
                elapsed = time.perf_counter() - tick
                if reference is None:
                    reference = result.values.copy()
                np.testing.assert_allclose(result.values, reference, rtol=1e-10, atol=1e-12)
                rho = p.layout.unpack_batch(result.values)
                trace_error = float(np.max(abs(np.trace(rho, axis1=-2, axis2=-1) - 1)))
                assert trace_error < 1e-9
                error = float(np.max(abs(result.values - reference)))
                data["records"].append(
                    dict(
                        path=path,
                        mode=mode,
                        wall_seconds=elapsed,
                        max_density_difference=error,
                        trace_error=trace_error,
                        stats=result.solver_stats,
                    )
                )
                np.savez_compressed(
                    out / f"{path}_{mode}.npz",
                    values=result.values,
                    initial=starts,
                    parameters=pars,
                    times=times,
                )
                save()
                print(f"Full flight {path} {mode}: {elapsed:.2f}s, delta={error:.3g}", flush=True)
        data["status"] = "complete"
    except BaseException as exc:
        data.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
