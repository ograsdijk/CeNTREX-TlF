import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
    solve_effective_fixed_basis,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
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
z = linear(te, offset=z0, slope=v)
gp = Parameter("z_grid", tuple(z_grid.tolist()))
ep = Parameter("ez", tuple(ez.tolist()))
Ez = pchip_tabulated(z, gp, ep)
Omega = gaussian(z, center=z_laser, sigma=w0, amplitude=omega0)

params = LindbladParameters({
    smp.Symbol("Ez"): Ez,
    smp.Symbol("\u03a90"): Omega,
    smp.Symbol("\u03b40"): 0.0,
})

t_span = (0.0, 50e-6)
t_eval = np.linspace(t_span[0], t_span[1], 51)
n = model.n_effective_states

print("=== RuntimeExpression (individual) ===")
t0 = time.perf_counter()
sol1 = solve_effective_fixed_basis(
    model, electric_field=Ez, rabi_rate=Omega, detuning=0.0,
    rho0=rho0, t_span=t_span, t_eval=t_eval,
)
t1 = time.perf_counter()
rho1 = sol1.y[:, -1].reshape(n, n)
print(f"  Time: {(t1-t0)*1000:.0f} ms, trace: {np.real(np.trace(rho1)):.6f}")

print("\n=== LindbladParameters ===")
t0 = time.perf_counter()
sol2 = solve_effective_fixed_basis(
    model, parameters=params,
    rho0=rho0, t_span=t_span, t_eval=t_eval,
)
t2 = time.perf_counter()
rho2 = sol2.y[:, -1].reshape(n, n)
print(f"  Time: {(t2-t0)*1000:.0f} ms, trace: {np.real(np.trace(rho2)):.6f}")

pops1 = np.real(np.diag(rho1))
pops2 = np.real(np.diag(rho2))
diff = np.max(np.abs(pops1 - pops2))
print(f"\n  Max population difference: {diff:.2e}")
print(f"  {'PASSED' if diff < 1e-6 else 'FAILED'}")
