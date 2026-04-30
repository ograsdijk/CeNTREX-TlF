import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
    parameter_scan,
    grid_scan,
)
from centrex_tlf.effective_hamiltonian.rust_plan import prepare_effective_lindblad_rust_plan
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
plan = prepare_effective_lindblad_rust_plan(model, params)
t_span = (0.0, 50e-6)

print(f"\n=== parameter_scan (velocity) ===")
velocities = np.linspace(150, 250, 10)
t0 = time.perf_counter()
result = parameter_scan(
    plan, rho0, t_span,
    parameter_slots=["v"],
    parameter_batch=velocities.reshape(-1, 1),
    output="populations",
    output_when="final",
    parallel=True,
)
elapsed = time.perf_counter() - t0
print(f"  Time: {elapsed*1000:.1f} ms")
print(f"  Shape: {result.values.shape}")
print(f"  Trajectories: {result.trajectory_count}")
print(f"  Trace (first): {result.values[0].sum():.6f}")
print(f"  Trace (last): {result.values[-1].sum():.6f}")

print(f"\n=== grid_scan (velocity x Rabi rate) ===")
rabi_rates = np.array([0.5, 1.0, 2.0]) * 2 * np.pi * 1e6
t0 = time.perf_counter()
result_grid = grid_scan(
    plan, rho0, t_span,
    scan={"v": velocities, "omega0": rabi_rates},
    output="populations",
    output_when="final",
    parallel=True,
)
elapsed = time.perf_counter() - t0
print(f"  Time: {elapsed*1000:.1f} ms")
print(f"  Shape: {result_grid.values.shape}")
print(f"  Trajectories: {result_grid.trajectory_count}")
print(f"  Grid shape: {result_grid.metadata.get('grid_shape')}")
print(f"  Trace (first): {result_grid.values[0].sum():.6f}")
print(f"  Trace (last): {result_grid.values[-1].sum():.6f}")

print(f"\n=== parameter_scan with saveat ===")
saveat = np.linspace(0, 50e-6, 11)
t0 = time.perf_counter()
result_saveat = parameter_scan(
    plan, rho0, t_span,
    parameter_slots=["v"],
    parameter_batch=velocities.reshape(-1, 1),
    saveat=saveat,
    output="populations",
    output_when="saveat",
    parallel=True,
)
elapsed = time.perf_counter() - t0
print(f"  Time: {elapsed*1000:.1f} ms")
print(f"  Shape: {result_saveat.values.shape}")
print(f"  Trace at end: {result_saveat.values[0, -1].sum():.6f}")
