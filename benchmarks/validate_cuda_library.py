"""Check public CUDA APIs against saved RC/R(2) references, without CPU solves."""

import os
from pathlib import Path

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
os.environ["CUPY_CACHE_DIR"] = str(HERE / ".cupy_cache")
import json, time
from types import SimpleNamespace
import numpy as np
import sympy as smp
from centrex_tlf import lindblad
from centrex_tlf.lindblad.cuda import _cupy

_cupy()
import rc_gpu_sparse_methods as m
from run_rc_gpu_integrator_suite import reference

rows = []


def record(system, p, initial, parameters, names, span, dt, ref, source, solver):
    tick = time.perf_counter()
    r = lindblad.solve_lindblad_batch(
        p,
        initial,
        span,
        parameter_batch=parameters,
        parameter_slots=names,
        solver=solver,
        evolution_device="cuda",
        dt=dt,
        maxiters=1000000,
        output="full",
        collect_stats=True,
        cuda_options=lindblad.CudaSolverOptions(batch_size=512),
    )
    matrices = np.stack([p.layout.unpack(x) for x in r.values])
    errors = m.diagnostics(matrices, ref)
    row = dict(
        system=system,
        solver=solver,
        batch=len(initial),
        dt_seconds=dt,
        span=span,
        elapsed_seconds=time.perf_counter() - tick,
        reference=str(source),
        accuracy=errors,
        stats=r.solver_stats,
    )
    rows.append(row)
    (HERE / "cuda_library_validation.json").write_text(json.dumps(rows, indent=2))
    print(
        system,
        solver,
        len(initial),
        row["elapsed_seconds"],
        {k: v for k, v in errors.items() if "trajectory" not in k},
        flush=True,
    )
    assert (
        errors["passes_1percent_above_cutoff"]
        and errors["near_zero_max_abs"] <= 1e-6
        and errors["trace_error"] < 1e-6
        and errors["min_eigenvalue"] >= -1e-7
    )


context, source, _ = m.load()
q = context["prepared"]
det, js = m.points(483)
ref, refsource = reference(dict(fixture="production", span_us=5))
end = context["physics"]["uv_length_m"] / 150
sigma = np.sqrt(2) * context["sigma_I"] / 150
pars = np.column_stack(
    (
        2 * np.pi * 1e6 * det,
        np.full(len(det), end / 2),
        np.full(len(det), sigma),
        np.full(len(det), 2 * np.pi * 1e6),
        np.zeros(len(det)),
        np.full(len(det), 2 * np.pi * 1e6),
        np.full(len(det), np.pi / 2),
    )
)
record(
    "RC",
    q,
    context["rho0"][js],
    pars,
    ["\u03b40", "center", "sigma_time", "\u03c91", "\u03c61", "\u03c92", "\u03c62"],
    (146e-6, 151e-6),
    250e-9 / 56,
    ref,
    refsource,
    "fixed_tsit5",
)
import run_r2_gpu_reuse as r2

a = dict(np.load(HERE / "r2_gpu_reuse_results/matched_cross_v1/inputs.npz"))
c, f, _ = r2.ode_fixture(a)
n = int(a["n"])
h = f["basis"][4] + np.diag(f["diagonal"])
d = smp.Symbol("detuning", real=True)
drive = smp.Symbol("drive")
H = smp.zeros(n)
for i in range(n):
    H[i, i] = float(h[i, i].real) + d * float(a["detuning_mask"][i])
    for j in range(i + 1, n):
        if h[i, j] != 0:
            H[i, j] = complex(h[i, j]) * drive
            H[j, i] = smp.conjugate(H[i, j])
C = []
for i in range(n):
    for k in range(f["tp"][i], f["tp"][i + 1]):
        jump = np.zeros((n, n), complex)
        jump[i, f["tc"][k]] = np.sqrt(f["tr"][k])
        C.append(jump)
params = lindblad.LindbladParameters()
params.real("detuning", 0.0)
t = smp.Symbol("t", real=True)
params.bind(drive, lindblad.phase_modulation(t, 3.8, 2 * np.pi * 1.6e6))
p = lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=np.asarray(C)), params)
refsource = HERE / "r2_gpu_reuse_results/eom_v1/eom_swap_const_exact_0.5ns.npz"
ref = np.load(refsource)["rho"]
batch = len(ref)
det = np.resize(np.repeat([-20.0, 0.0, 20.0], 20), batch)
initial = c["rho0"][np.arange(batch) % 20]
for solver, dt in [("fixed_tsit5", 2e-9), ("fixed_ck54", 1.95e-9)]:
    record(
        "R2 EOM",
        p,
        initial,
        (2 * np.pi * 1e6 * det)[:, None],
        ["detuning"],
        (125e-9, 1125e-9),
        dt,
        ref,
        refsource,
        solver,
    )
