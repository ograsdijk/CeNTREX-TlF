"""Full RC public-API validation, using saved converged trajectories."""

import os
from pathlib import Path

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
os.environ["CUPY_CACHE_DIR"] = str(HERE / ".cupy_cache")
import json, time
import numpy as np
from centrex_tlf import lindblad, hamiltonian
from centrex_tlf.lindblad.cuda import _cupy

_cupy()
import rc_gpu_sparse_methods as m
from run_rc_gpu_integrator_suite import PREVIOUS

context, _, _ = m.load()
p = context["prepared"]
det, js = m.points(9, "full")
oldrows = json.loads((PREVIOUS / "results.json").read_text())
rows = []
for v in [150.0, 50.0, 250.0]:
    end = context["physics"]["uv_length_m"] / v
    sigma = np.sqrt(2) * context["sigma_I"] / v
    names = ["\u03b40", "center", "sigma_time", "\u03c91", "\u03c61", "\u03c92", "\u03c62"]
    pars = np.column_stack(
        [
            2 * np.pi * 1e6 * det,
            *[
                np.full(9, x)
                for x in [end / 2, sigma, 2 * np.pi * 1e6, 0.0, 2 * np.pi * 1e6, np.pi / 2]
            ],
        ]
    )
    refrow = next(
        x
        for x in oldrows
        if x["job"].get("kind") == "full"
        and x["job"].get("velocity") == v
        and x["job"].get("method") == "staged_soa_minor"
        and x["job"].get("substeps") == 512
    )
    source = Path(refrow["output"]).with_suffix(".npz")
    reference = np.load(source)
    dt = 250e-9 / 56
    tick = time.perf_counter()
    # The explicit schedule uses the tested 20% longer steps in weak wings.
    edge0 = max(0.0, end / 2 - 0.8 * sigma)
    edge1 = min(end, end / 2 + 0.8 * sigma)
    r = lindblad.solve_lindblad_batch(
        p,
        context["rho0"][js],
        (0.0, end),
        parameter_slots=names,
        parameter_batch=pars,
        solver="fixed_tsit5",
        evolution_device="cuda",
        dt=dt * 1.2,
        maxiters=1000000,
        output="populations",
        output_when="saveat",
        saveat=np.linspace(0.0, end, 201),
        collect_stats=True,
        cuda_options=lindblad.CudaSolverOptions(
            density_saveat=[end], step_schedule=[(edge0, dt), (edge1, dt * 1.2)]
        ),
    )
    exit_packed = r.metadata["density_values"][:, 0]
    darknames = names + ["\u03a9t0", "mw_peak1", "mw_peak2"]
    darkpars = np.column_stack([pars, np.zeros((9, 3))])
    dark = lindblad.solve_lindblad_batch(
        p,
        exit_packed,
        (end, end + 25 / float(hamiltonian.Γ)),
        parameter_slots=darknames,
        parameter_batch=darkpars,
        solver="fixed_tsit5",
        evolution_device="cuda",
        dt=dt,
        maxiters=1000000,
        output="full",
    )
    states = np.stack(
        [np.stack([p.layout.unpack(x) for x in a]) for a in [exit_packed, dark.values]], axis=1
    )
    errors = m.diagnostics(
        states.reshape(-1, p.layout.n, p.layout.n),
        reference["density_matrix"].reshape(-1, p.layout.n, p.layout.n),
    )
    pop_error = float(abs(r.values - reference["populations"]).max())
    row = dict(
        velocity=v,
        elapsed_seconds=time.perf_counter() - tick,
        accuracy=errors,
        population_history_max_abs=pop_error,
        reference=str(source),
        stats=r.solver_stats,
    )
    rows.append(row)
    (HERE / "cuda_library_full_rc_validation.json").write_text(json.dumps(rows, indent=2))
    print(
        v,
        row["elapsed_seconds"],
        {k: x for k, x in errors.items() if "trajectory" not in k},
        pop_error,
        flush=True,
    )
    assert (
        errors["passes_1percent_above_cutoff"]
        and errors["near_zero_max_abs"] <= 1e-6
        and errors["trace_error"] < 1e-6
        and errors["min_eigenvalue"] >= -1e-7
    )
