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
    solve_effective_lindblad_batch,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

print("Preparing model...")
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 7, 7.5, 8, 10, 20, 30, 40, 50],
    transition=transition,
    optical_polarization=polarization,
)
rho0 = default_effective_density_matrix(model)
n_states = model.n_effective_states

z_field_grid = np.linspace(-0.01, 0.01, 50)
ez_profile = 10.0 + 15.0 * np.exp(-z_field_grid**2 / (2 * 0.003**2))

t_expr = Time()
v_param = Parameter("v", 200.0)
z0_param = Parameter("z0", -0.005)
omega0_param = Parameter("omega0", 2 * np.pi * 1e6)
z_laser_param = Parameter("z_laser", 0.002)
w0_param = Parameter("w0", 200e-6)
z_expr = linear(t_expr, offset=z0_param, slope=v_param)
z_grid_param = Parameter("z_field_grid", tuple(z_field_grid.tolist()))
ez_vals_param = Parameter("ez_profile", tuple(ez_profile.tolist()))
Ez_expr = pchip_tabulated(z_expr, z_grid_param, ez_vals_param)
Omega_expr = gaussian(z_expr, center=z_laser_param, sigma=w0_param, amplitude=omega0_param)

params = LindbladParameters({
    smp.Symbol("Ez"): Ez_expr,
    smp.Symbol("\u03a90"): Omega_expr,
    smp.Symbol("\u03b40"): 0.0,
})

print("Preparing Rust plan...")
plan = prepare_effective_lindblad_rust_plan(model, params)
print(f"  slot_names: {plan.slot_names}")

t_span = (0.0, 50e-6)
saveat = np.linspace(t_span[0], t_span[1], 51)

velocities = np.linspace(150, 250, 20)
n_traj = len(velocities)

print(f"\n=== Single solve reference (v=200) ===")
t0 = time.perf_counter()
result_single = solve_effective_lindblad(
    plan, rho0, t_span, saveat=saveat, reltol=1e-6, abstol=1e-8, dt=1e-10,
)
single_time = time.perf_counter() - t0
print(f"  Time: {single_time*1000:.1f} ms")
print(f"  Trace: {result_single.populations()[-1].sum():.6f}")

print(f"\n=== Batch solve ({n_traj} trajectories, serial) ===")
t0 = time.perf_counter()
try:
    result_batch = solve_effective_lindblad_batch(
        plan, rho0, t_span,
        parameter_overrides={"v": velocities},
        solver="dopri5",
        saveat=saveat,
        reltol=1e-6,
        abstol=1e-8,
        output="populations",
        parallel=False,
    )
    batch_serial_time = time.perf_counter() - t0
    print(f"  Time: {batch_serial_time*1000:.1f} ms")
    print(f"  Shape: {result_batch.values.shape}")
    print(f"  Trace (first): {result_batch.values[0, -1].sum():.6f}")
    print(f"  Trace (last): {result_batch.values[-1, -1].sum():.6f}")
    print(f"  Expected vs actual time per traj: {single_time*1000:.1f} vs {batch_serial_time*1000/n_traj:.1f} ms")

    mid_idx = n_traj // 2
    single_pops = result_single.populations()[-1]
    batch_pops = result_batch.values[mid_idx, -1]
    diff = np.max(np.abs(single_pops - batch_pops))
    print(f"  v=200 agreement (mid trajectory): {diff:.2e}")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n=== Batch solve ({n_traj} trajectories, parallel) ===")
t0 = time.perf_counter()
try:
    result_parallel = solve_effective_lindblad_batch(
        plan, rho0, t_span,
        parameter_overrides={"v": velocities},
        solver="dopri5",
        saveat=saveat,
        reltol=1e-6,
        abstol=1e-8,
        output="populations",
        parallel=True,
    )
    batch_parallel_time = time.perf_counter() - t0
    print(f"  Time: {batch_parallel_time*1000:.1f} ms")
    print(f"  Speedup vs serial: {batch_serial_time/batch_parallel_time:.1f}x")
    diff_par = np.max(np.abs(result_parallel.values - result_batch.values))
    print(f"  Serial vs parallel agreement: {diff_par:.2e}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n=== Final-only output ({n_traj} trajectories) ===")
t0 = time.perf_counter()
try:
    result_final = solve_effective_lindblad_batch(
        plan, rho0, t_span,
        parameter_overrides={"v": velocities},
        solver="dopri5",
        saveat=saveat,
        reltol=1e-6,
        abstol=1e-8,
        output="populations",
        output_when="final",
        parallel=True,
    )
    final_time = time.perf_counter() - t0
    print(f"  Time: {final_time*1000:.1f} ms")
    print(f"  Shape: {result_final.values.shape}")
    print(f"  Trace (first): {result_final.values[0].sum():.6f}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
