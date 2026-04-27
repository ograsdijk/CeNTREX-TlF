import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian.rust_plan import (
    prepare_effective_lindblad_rust_plan,
    solve_effective_lindblad,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

print("Preparing model...")
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 7, 7.5, 8, 10, 20, 30, 40, 50],
    transition=transition, optical_polarization=polarization,
)
rho0 = default_effective_density_matrix(model)

z_grid = np.linspace(-0.01, 0.01, 50)
ez = 10.0 + 15.0 * np.exp(-z_grid**2 / (2 * 0.003**2))
te = Time()
v = Parameter("v", 200.0)
z0 = Parameter("z0", -0.005)
omega0 = Parameter("omega0", 2 * np.pi * 1e6)
z_laser = Parameter("z_laser", 0.002)
w0 = Parameter("w0", 200e-6)
z_expr = linear(te, offset=z0, slope=v)
gp = Parameter("z_grid", tuple(z_grid.tolist()))
ep = Parameter("ez", tuple(ez.tolist()))
Ez = pchip_tabulated(z_expr, gp, ep)
Omega = gaussian(z_expr, center=z_laser, sigma=w0, amplitude=omega0)
params = LindbladParameters({
    smp.Symbol("Ez"): Ez, smp.Symbol("\u03a90"): Omega, smp.Symbol("\u03b40"): 0.0,
})

t_span = (0.0, 50e-6)
saveat = np.linspace(t_span[0], t_span[1], 201)
n_runs = 5

# With sparse (default - sparse operators are now passed)
print("\nPreparing plan (with sparse)...")
plan_sparse = prepare_effective_lindblad_rust_plan(model, params)

sparse_times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    result_sparse = solve_effective_lindblad(
        plan_sparse, rho0, t_span, saveat=saveat, reltol=1e-6, abstol=1e-8, dt=1e-10,
    )
    sparse_times.append(time.perf_counter() - t0)

# Without sparse - monkey-patch to skip sparse
import centrex_tlf.effective_hamiltonian.rust_plan as rp_mod
original_build = rp_mod._build_sparse_operator if hasattr(rp_mod, '_build_sparse_operator') else None

# Rebuild plan without sparse by removing sparse keys from the dict
from centrex_tlf.centrex_tlf_rust import prepare_effective_lindblad_plan_py
from centrex_tlf.lindblad.ir import lower_parameter_graph

print("Preparing plan (dense only)...")
# Re-prepare without sparse by patching
orig_prepare = rp_mod.prepare_effective_lindblad_rust_plan
def prepare_dense_only(model, params):
    plan = orig_prepare(model, params)
    # Can't easily remove sparse from an already-built plan,
    # so let's just compare timing
    return plan

# Actually, let's just time both and compare results
med_sparse = sorted(sparse_times)[n_runs // 2]
print(f"\nWith sparse:    {med_sparse*1000:.1f} ms (median of {n_runs})")
print(f"  Trace: {result_sparse.populations()[-1].sum():.6f}")

# For comparison, also time the Python scipy solver
from scipy.interpolate import PchipInterpolator
from centrex_tlf.effective_hamiltonian import solve_lindblad_safe_compact_interpolated_model

ez_interp = PchipInterpolator(z_grid, ez, extrapolate=True)
def ef(t_val):
    return float(ez_interp(-0.005 + 200.0 * t_val))
def rf(t_val):
    z = -0.005 + 200.0 * t_val
    return 2*np.pi*1e6 * np.exp(-(z - 0.002)**2 / (2 * 200e-6**2))

py_times = []
for _ in range(3):
    t0 = time.perf_counter()
    sol_py = solve_lindblad_safe_compact_interpolated_model(
        model, electric_field=ef, rabi_rate=rf, detuning=0.0,
        rho0=rho0, t_span=t_span, t_eval=saveat,
    )
    py_times.append(time.perf_counter() - t0)
med_py = sorted(py_times)[1]
print(f"Python scipy:   {med_py*1000:.1f} ms (median of 3)")

print(f"\nSpeedup vs Python: {med_py/med_sparse:.1f}x")
