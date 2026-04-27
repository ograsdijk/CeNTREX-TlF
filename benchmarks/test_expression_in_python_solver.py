import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
    solve_effective_fixed_basis,
)
from centrex_tlf.lindblad.parameters import (
    Parameter, Time, linear, gaussian, pchip_tabulated,
)
from scipy.interpolate import PchipInterpolator

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
    transition=transition, optical_polarization=polarization,
)
rho0 = default_effective_density_matrix(model)

z_grid = np.linspace(-0.01, 0.01, 50)
ez = 10.0 + 15.0 * np.exp(-z_grid**2 / (2 * 0.003**2))

velocity = 200.0
z0_val = -0.005
z_laser_val = 0.002
w0_val = 200e-6
omega0_val = 2 * np.pi * 1e6
t_span = (0.0, 50e-6)
t_eval = np.linspace(t_span[0], t_span[1], 51)

print("=== Python callable approach ===")
ez_interp = PchipInterpolator(z_grid, ez, extrapolate=True)
def ef(t):
    return float(ez_interp(z0_val + velocity * t))
def rf(t):
    z = z0_val + velocity * t
    return omega0_val * np.exp(-(z - z_laser_val)**2 / (2 * w0_val**2))

t0 = time.perf_counter()
sol1 = solve_effective_fixed_basis(
    model, electric_field=ef, rabi_rate=rf, detuning=0.0,
    rho0=rho0, t_span=t_span, t_eval=t_eval,
)
t1 = time.perf_counter()
n = model.n_effective_states
rho1 = sol1.y[:, -1].reshape(n, n)
print(f"  Time: {(t1-t0)*1000:.0f} ms, trace: {np.real(np.trace(rho1)):.6f}")

print("\n=== RuntimeExpression approach ===")
t_expr = Time()
v = Parameter("v", velocity)
z0 = Parameter("z0", z0_val)
omega0 = Parameter("omega0", omega0_val)
z_laser = Parameter("z_laser", z_laser_val)
w0 = Parameter("w0", w0_val)
z = linear(t_expr, offset=z0, slope=v)
gp = Parameter("z_grid", tuple(z_grid.tolist()))
ep = Parameter("ez", tuple(ez.tolist()))
Ez = pchip_tabulated(z, gp, ep)
Omega = gaussian(z, center=z_laser, sigma=w0, amplitude=omega0)

t0 = time.perf_counter()
sol2 = solve_effective_fixed_basis(
    model, electric_field=Ez, rabi_rate=Omega, detuning=0.0,
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
