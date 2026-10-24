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
from centrex_tlf.lindblad.ir import lower_parameter_graph

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 7, 7.5, 8, 10, 20, 30, 40, 50],
    transition=transition,
    optical_polarization=polarization,
)
rho0 = default_effective_density_matrix(model)

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

t_span = (0.0, 50e-6)
t_eval = np.linspace(t_span[0], t_span[1], 201)
n_runs = 5

# PCHIP is evaluated per interval from neighbouring secants (no precomputed
# tables; the table fast path was removed after it measured a 1.01x speedup
# and could pick the wrong values array for a shared grid).
plan = prepare_effective_lindblad_rust_plan(model, params)
times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    result = solve_effective_lindblad(plan, rho0, t_span, saveat=t_eval, reltol=1e-6, abstol=1e-8, dt=1e-10)
    times.append(time.perf_counter() - t0)

print(f"PCHIP solve: {sorted(times)[n_runs // 2] * 1000:.1f} ms (median of {n_runs})")
