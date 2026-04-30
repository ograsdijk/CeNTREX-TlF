import sys
sys.stdout.reconfigure(encoding="utf-8")
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
    _complex_rho_to_split_real,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)
from centrex_tlf.centrex_tlf_rust import solve_effective_lindblad_batch_py

t = transitions.Q1_F1_1o2_F0
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
    transition=t, optical_polarization=couplings.polarization_Z,
)
rho0 = default_effective_density_matrix(model)
y0 = _complex_rho_to_split_real(rho0)

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
plan = prepare_effective_lindblad_rust_plan(model, params)

print(f"y0 norm: {np.linalg.norm(y0):.6f}")
print(f"plan.slot_names: {plan.slot_names}")
print(f"plan.n_states: {plan.n_states}")
print(f"plan.real_dim: {plan.real_dim}")

# Direct Rust call with 2 trajectories
velocities = np.array([200.0, 200.0], dtype=np.float64)
param_batch = velocities.reshape(2, 1)
saveat = np.array([50e-6], dtype=np.float64)

print(f"\nparam_batch: {param_batch}")
print(f"slot_indices: [1]")

times, values, width, time_count, stats = solve_effective_lindblad_batch_py(
    plan, y0, 0.0, 50e-6,
    1e-8, 1e-6, 1e-10,
    saveat, True, 100000,
    "dopri5", "populations",
    [1], param_batch,
    2, False, None,
)
print(f"\ntimes: {np.asarray(times)}")
print(f"values shape: ({len(values)},)")
print(f"width: {width}, time_count: {time_count}")
vals = np.asarray(values, dtype=np.float64)
print(f"values: {vals}")
print(f"trace traj 0: {vals[:width].sum():.6f}")
print(f"trace traj 1: {vals[width:2*width].sum():.6f}")
