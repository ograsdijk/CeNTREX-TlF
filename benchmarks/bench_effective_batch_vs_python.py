import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import sympy as smp
from scipy.interpolate import PchipInterpolator
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    solve_effective_fixed_basis,
    default_effective_density_matrix,
    parameter_scan,
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
n_states = model.n_effective_states

z_field_grid = np.linspace(-0.01, 0.01, 50)
ez_profile = 10.0 + 15.0 * np.exp(-z_field_grid**2 / (2 * 0.003**2))
ez_interp = PchipInterpolator(z_field_grid, ez_profile, extrapolate=True)

z0_val = -0.005
z_laser_val = 0.002
w0_val = 200e-6
omega0_val = 2 * np.pi * 1e6
t_span = (0.0, 50e-6)

velocities = np.linspace(150, 250, 10)
n_traj = len(velocities)

print(f"n_states: {n_states}, n_trajectories: {n_traj}")

# Python batch (serial, scipy RK45)
print(f"\n=== Python scipy (serial, {n_traj} trajectories) ===")
t0 = time.perf_counter()
py_results = []
for v in velocities:
    def ef(t, _v=v):
        return float(ez_interp(z0_val + _v * t))
    def rf(t, _v=v):
        z = z0_val + _v * t
        return omega0_val * np.exp(-(z - z_laser_val)**2 / (2 * w0_val**2))
    sol = solve_effective_fixed_basis(
        model, electric_field=ef, rabi_rate=rf, detuning=0.0,
        rho0=rho0, t_span=t_span,
    )
    rho_final = sol.y[:, -1].reshape(n_states, n_states)
    py_results.append(np.real(np.diag(rho_final)))
py_time = time.perf_counter() - t0
py_pops = np.array(py_results)
print(f"  Time: {py_time*1000:.0f} ms ({py_time*1000/n_traj:.0f} ms/traj)")
print(f"  Trace (first): {py_pops[0].sum():.6f}")

# Rust batch (serial)
print(f"\n=== Rust batch (serial, {n_traj} trajectories) ===")
te = Time()
v_p = Parameter("v", 200.0)
z0_p = Parameter("z0", z0_val)
omega0_p = Parameter("omega0", omega0_val)
z_laser_p = Parameter("z_laser", z_laser_val)
w0_p = Parameter("w0", w0_val)
z_expr = linear(te, offset=z0_p, slope=v_p)
gp = Parameter("z_grid", tuple(z_field_grid.tolist()))
ep = Parameter("ez", tuple(ez_profile.tolist()))
Ez = pchip_tabulated(z_expr, gp, ep)
Omega = gaussian(z_expr, center=z_laser_p, sigma=w0_p, amplitude=omega0_p)
params = LindbladParameters({
    smp.Symbol("Ez"): Ez, smp.Symbol("\u03a90"): Omega, smp.Symbol("\u03b40"): 0.0,
})
plan = prepare_effective_lindblad_rust_plan(model, params)

t0 = time.perf_counter()
result_serial = parameter_scan(
    plan, rho0, t_span,
    parameter_slots=["v"],
    parameter_batch=velocities.reshape(-1, 1),
    output="populations", output_when="final",
    parallel=False,
)
rust_serial_time = time.perf_counter() - t0
print(f"  Time: {rust_serial_time*1000:.0f} ms ({rust_serial_time*1000/n_traj:.0f} ms/traj)")
print(f"  Trace (first): {result_serial.values[0].sum():.6f}")

# Rust batch (parallel)
print(f"\n=== Rust batch (parallel, {n_traj} trajectories) ===")
t0 = time.perf_counter()
result_parallel = parameter_scan(
    plan, rho0, t_span,
    parameter_slots=["v"],
    parameter_batch=velocities.reshape(-1, 1),
    output="populations", output_when="final",
    parallel=True,
)
rust_parallel_time = time.perf_counter() - t0
print(f"  Time: {rust_parallel_time*1000:.0f} ms ({rust_parallel_time*1000/n_traj:.0f} ms/traj)")
print(f"  Trace (first): {result_parallel.values[0].sum():.6f}")

# Agreement
diff = np.max(np.abs(result_serial.values - py_pops))
print(f"\n=== Comparison ===")
print(f"  Python vs Rust agreement: {diff:.2e}")
print(f"  Speedup (serial):   {py_time/rust_serial_time:.1f}x")
print(f"  Speedup (parallel): {py_time/rust_parallel_time:.1f}x")
