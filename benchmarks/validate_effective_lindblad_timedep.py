"""
Phase 3 validation: time-dependent effective Lindblad solver.

Compares the Rust effective Lindblad solver against the Python scipy reference
for a realistic trajectory: molecule flying through a spatially varying electric
field with a Gaussian laser beam.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import sympy as smp
from scipy.interpolate import PchipInterpolator

from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    solve_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian.rust_plan import (
    prepare_effective_lindblad_rust_plan,
    solve_effective_lindblad,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters,
    Parameter,
    Time,
    linear,
    gaussian,
    pchip_tabulated,
)

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

velocity = 200.0
z0 = -0.005
z_laser = 0.002
w0 = 200e-6
omega0 = 2 * np.pi * 1e6
detuning_val = 0.0

field_points = [0, 5, 7, 7.5, 8, 10, 20, 30, 40, 50]

z_field_grid = np.linspace(-0.01, 0.01, 50)
ez_profile = 10.0 + 15.0 * np.exp(-z_field_grid**2 / (2 * (0.003)**2))

t_span = (0.0, 50e-6)
n_points = 201
t_eval = np.linspace(t_span[0], t_span[1], n_points)

print("=" * 70)
print("Phase 3 Validation: Time-Dependent Effective Lindblad Solver")
print("=" * 70)
print(f"  Transition: {transition}")
print(f"  Velocity: {velocity} m/s")
print(f"  z0: {z0*1000:.1f} mm, z_laser: {z_laser*1000:.1f} mm")
print(f"  Beam waist: {w0*1e6:.0f} µm")
print(f"  Rabi rate: {omega0/(2*np.pi*1e6):.1f} MHz")
print(f"  t_span: {t_span[0]*1e6:.0f} - {t_span[1]*1e6:.0f} µs")

print("\nPreparing model...")
t0 = time.perf_counter()
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=field_points,
    transition=transition,
    optical_polarization=polarization,
)
prep_model_time = time.perf_counter() - t0
print(f"  Model: {model.n_effective_states} states, {len(field_points)} grid points")
print(f"  Preparation: {prep_model_time:.2f}s")

rho0 = default_effective_density_matrix(model)

print("\n--- Python scipy reference ---")

ez_interp = PchipInterpolator(z_field_grid, ez_profile, extrapolate=True)

def electric_field_func(t_val):
    z = z0 + velocity * t_val
    return float(ez_interp(z))

def rabi_rate_func(t_val):
    z = z0 + velocity * t_val
    return omega0 * np.exp(-(z - z_laser)**2 / (2 * w0**2))

from scipy.integrate import solve_ivp
from centrex_tlf.effective_hamiltonian._superoperators import _hamiltonian_superoperator
from centrex_tlf.effective_hamiltonian._utility import _as_field_vector, _parameter_at_time

fields = np.asarray(model.field_points, dtype=np.float64)
endpoint_bundles = tuple(
    model.effective_bundle((0.0, 0.0, float(fz)), model.reference_magnetic_field)
    for fz in fields.tolist()
)
h_int_sups = tuple(_hamiltonian_superoperator(b.h_internal) for b in endpoint_bundles)
h_opt_sups = tuple(_hamiltonian_superoperator(b.h_opt) for b in endpoint_bundles)
h_det_sups = tuple(_hamiltonian_superoperator(b.h_det) for b in endpoint_bundles)
diss_sups = tuple(b.dissipator_superoperator() for b in endpoint_bundles)

def _interp_idx(fz):
    if fz <= fields[0]: return 0, 0, 0.0
    if fz >= fields[-1]: return len(fields)-1, len(fields)-1, 0.0
    u = int(np.searchsorted(fields, fz, side="right"))
    lo = u - 1
    return lo, u, (fz - fields[lo]) / (fields[u] - fields[lo])

def py_rhs(t_val, rho_flat):
    ez = electric_field_func(t_val)
    omega = rabi_rate_func(t_val)
    delta = detuning_val
    lo, hi, w = _interp_idx(ez)
    if lo == hi:
        L = h_int_sups[lo] + diss_sups[lo] + 0.5*omega*h_opt_sups[lo] + delta*h_det_sups[lo]
    else:
        L = ((1-w)*(h_int_sups[lo]+diss_sups[lo]) + w*(h_int_sups[hi]+diss_sups[hi])
             + 0.5*omega*((1-w)*h_opt_sups[lo]+w*h_opt_sups[hi])
             + delta*((1-w)*h_det_sups[lo]+w*h_det_sups[hi]))
    return L @ rho_flat

t0 = time.perf_counter()
sol_py = solve_ivp(
    py_rhs,
    t_span=t_span,
    y0=rho0.reshape(-1),
    t_eval=t_eval,
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)
py_time = time.perf_counter() - t0
n_states = model.n_effective_states
rho_py = sol_py.y.T.reshape(-1, n_states, n_states)
pops_py = np.real(np.diagonal(rho_py, axis1=1, axis2=2))
print(f"  Time: {py_time*1000:.1f} ms")
print(f"  Trace at end: {pops_py[-1].sum():.6f}")

print("\n--- Rust effective Lindblad solver ---")

t_expr = Time()
v_param = Parameter("v", velocity)
z0_param = Parameter("z0", z0)
omega0_param = Parameter("omega0", omega0)
z_laser_param = Parameter("z_laser", z_laser)
w0_param = Parameter("w0", w0)

z_expr = linear(t_expr, offset=z0_param, slope=v_param)

z_grid_param = Parameter("z_field_grid", tuple(z_field_grid.tolist()))
ez_vals_param = Parameter("ez_profile", tuple(ez_profile.tolist()))
Ez_expr = pchip_tabulated(z_expr, z_grid_param, ez_vals_param)

Omega_expr = gaussian(z_expr, center=z_laser_param, sigma=w0_param, amplitude=omega0_param)

params = LindbladParameters({
    smp.Symbol("Ez"): Ez_expr,
    smp.Symbol("\u03a90"): Omega_expr,
    smp.Symbol("\u03b40"): detuning_val,
})

print("  Preparing Rust plan...")
t0 = time.perf_counter()
try:
    rust_plan = prepare_effective_lindblad_rust_plan(model, params)
    prep_rust_time = time.perf_counter() - t0
    print(f"  Rust plan: {prep_rust_time*1000:.1f} ms")

    print("  Solving...")
    t0 = time.perf_counter()
    result = solve_effective_lindblad(
        rust_plan,
        rho0,
        t_span,
        saveat=t_eval,
        reltol=1e-6,
        abstol=1e-8,
        dt=1e-10,
    )
    rust_time = time.perf_counter() - t0
    pops_rust = result.populations()
    print(f"  Time: {rust_time*1000:.1f} ms")
    print(f"  Trace at end: {pops_rust[-1].sum():.6f}")

    max_pop_diff = np.max(np.abs(pops_rust - pops_py))
    print(f"\n  Max population difference: {max_pop_diff:.2e}")
    print(f"  Speedup: {py_time/rust_time:.1f}x")

    if max_pop_diff > 1e-2:
        print("\n  WARNING: Large population difference! Investigating...")
        print(f"  Python pops[-1]: {pops_py[-1][:5]}")
        print(f"  Rust pops[-1]:   {pops_rust[-1][:5]}")
        for ti in [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]:
            diff_at_t = np.max(np.abs(pops_rust[ti] - pops_py[ti]))
            print(f"  t={t_eval[ti]*1e6:.1f}µs: max diff = {diff_at_t:.2e}")
    else:
        print("\n  PASSED: populations agree within tolerance")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
