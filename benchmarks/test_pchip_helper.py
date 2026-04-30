import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from centrex_tlf import transitions, couplings, lindblad
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)

trans = transitions.R0_F1_3o2_F2
ts = couplings.generate_transition_selectors([trans], [[couplings.polarization_Z]])
system = lindblad.generate_OBE_system_transitions([trans], ts, method="matrix")

Gamma = 2 * np.pi * 1.56e6

z_grid = np.linspace(-0.02, 0.02, 20)
ez_values = 10.0 + 5.0 * np.sin(np.pi * z_grid / 0.02)

import sympy as smp

t = Time()
v = Parameter("v", 400.0)
z0 = Parameter("z0", -0.02)
omega0 = Parameter("\u03a9t0", Gamma)
detuning_scale = Parameter("det_scale", 1000.0)

z = linear(t, offset=z0, slope=v)
grid_param = Parameter("z_grid", tuple(z_grid.tolist()))
vals_param = Parameter("ez_vals", tuple(ez_values.tolist()))
Ez = pchip_tabulated(z, grid_param, vals_param)
delta = detuning_scale * Ez

params = LindbladParameters({
    smp.Symbol("\u03a90"): omega0,
    smp.Symbol("\u03b40"): delta,
    smp.Symbol("PZ0"): 1.0,
})

print("Preparing problem with pchip_interp...")
prepared = prepare_lindblad_problem(system, params, backend="rust")

n = len(system.QN)
n_ground = len(system.ground)
rho0 = np.zeros((n, n), dtype=np.complex128)
for i in range(n_ground):
    rho0[i, i] = 1.0 / n_ground

t_span = (0.0, 100e-6)
saveat = np.linspace(t_span[0], t_span[1], 201)

print("Solving...")
result = solve_lindblad(
    prepared, rho0, t_span,
    solver="dopri5", execution_mode="structured_upper",
    saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
)
print(f"Trace at end: {result.populations()[-1].sum():.6f}")
print("pchip_interp through Rust solver: OK")
