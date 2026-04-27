import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian._superoperators import _hamiltonian_superoperator
from centrex_tlf.effective_hamiltonian._utility import _as_field_vector
from centrex_tlf.effective_hamiltonian.rust_plan import (
    prepare_effective_lindblad_rust_plan,
    solve_effective_lindblad,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)
import sympy as smp

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

velocity = 200.0
z0 = -0.005
z_laser = 0.002
w0 = 200e-6
omega0 = 2 * np.pi * 1e6
t_span = (0.0, 50e-6)
saveat = np.linspace(t_span[0], t_span[1], 201)

field_points = [0, 5, 7, 7.5, 8, 10, 20, 30, 40, 50]
z_field_grid = np.linspace(-0.01, 0.01, 50)
ez_profile = 10.0 + 15.0 * np.exp(-z_field_grid**2 / (2 * 0.003**2))

print("Preparing model...")
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=field_points, transition=transition, optical_polarization=polarization,
)
rho0 = default_effective_density_matrix(model)
n = model.n_effective_states
print(f"n_states: {n}")

ez_interp = PchipInterpolator(z_field_grid, ez_profile, extrapolate=True)

def ef(t_val):
    return float(ez_interp(z0 + velocity * t_val))

def rf(t_val):
    z = z0 + velocity * t_val
    return omega0 * np.exp(-(z - z_laser)**2 / (2 * w0**2))

# Precompute superoperators for Python solvers
fields = np.asarray(model.field_points, dtype=np.float64)
endpoint_bundles = tuple(
    model.effective_bundle((0.0, 0.0, float(fz)), model.reference_magnetic_field)
    for fz in fields.tolist()
)
h_int_sups = tuple(_hamiltonian_superoperator(b.h_internal) for b in endpoint_bundles)
h_opt_sups = tuple(_hamiltonian_superoperator(b.h_opt) for b in endpoint_bundles)
h_det_sups = tuple(_hamiltonian_superoperator(b.h_det) for b in endpoint_bundles)
diss_sups = tuple(b.dissipator_superoperator() for b in endpoint_bundles)

def interp_idx(fz):
    if fz <= fields[0]: return 0, 0, 0.0
    if fz >= fields[-1]:
        last = len(fields) - 1
        return last, last, 0.0
    u = int(np.searchsorted(fields, fz, side="right"))
    lo = u - 1
    return lo, u, (fz - fields[lo]) / (fields[u] - fields[lo])

def py_rhs(t_val, rho_flat):
    ez = ef(t_val)
    omega = rf(t_val)
    lo, hi, w = interp_idx(ez)
    if lo == hi:
        L = h_int_sups[lo] + diss_sups[lo] + 0.5*omega*h_opt_sups[lo]
    else:
        L = ((1-w)*(h_int_sups[lo]+diss_sups[lo]) + w*(h_int_sups[hi]+diss_sups[hi])
             + 0.5*omega*((1-w)*h_opt_sups[lo]+w*h_opt_sups[hi]))
    return L @ rho_flat

n_runs = 3
print(f"\nBenchmarking Q1 time-dependent solve ({t_span[1]*1e6:.0f} µs, {len(saveat)} save points)")
print("=" * 70)

# Python RK45
print("\n--- Python scipy RK45 ---")
rk45_times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    sol = solve_ivp(py_rhs, t_span, rho0.reshape(-1), method="RK45", t_eval=saveat, rtol=1e-6, atol=1e-8)
    rk45_times.append(time.perf_counter() - t0)
med_rk45 = sorted(rk45_times)[n_runs // 2]
rho_rk45 = sol.y[:, -1].reshape(n, n)
print(f"  Time: {med_rk45*1000:.0f} ms, trace: {np.real(np.trace(rho_rk45)):.6f}")

# Python BDF
print("\n--- Python scipy BDF ---")
bdf_times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    sol_bdf = solve_ivp(py_rhs, t_span, rho0.reshape(-1), method="BDF", t_eval=saveat, rtol=1e-6, atol=1e-8)
    bdf_times.append(time.perf_counter() - t0)
med_bdf = sorted(bdf_times)[n_runs // 2]
rho_bdf = sol_bdf.y[:, -1].reshape(n, n)
print(f"  Time: {med_bdf*1000:.0f} ms, trace: {np.real(np.trace(rho_bdf)):.6f}")

med_radau = None

# Rust DOPRI5 (sparse)
print("\n--- Rust DOPRI5 (sparse) ---")
te = Time()
v_p = Parameter("v", velocity)
z0_p = Parameter("z0", z0)
omega0_p = Parameter("omega0", omega0)
z_laser_p = Parameter("z_laser", z_laser)
w0_p = Parameter("w0", w0)
z_expr = linear(te, offset=z0_p, slope=v_p)
gp = Parameter("z_grid", tuple(z_field_grid.tolist()))
ep = Parameter("ez", tuple(ez_profile.tolist()))
Ez = pchip_tabulated(z_expr, gp, ep)
Omega = gaussian(z_expr, center=z_laser_p, sigma=w0_p, amplitude=omega0_p)
params = LindbladParameters({
    smp.Symbol("Ez"): Ez, smp.Symbol("\u03a90"): Omega, smp.Symbol("\u03b40"): 0.0,
})
plan = prepare_effective_lindblad_rust_plan(model, params)

rust_times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    result = solve_effective_lindblad(
        plan, rho0, t_span, saveat=saveat, reltol=1e-6, abstol=1e-8, dt=1e-10,
    )
    rust_times.append(time.perf_counter() - t0)
med_rust = sorted(rust_times)[n_runs // 2]
pops_rust = result.populations()
print(f"  Time: {med_rust*1000:.0f} ms, trace: {pops_rust[-1].sum():.6f}")

med_tsit5 = None

# Agreement
pops_rk45 = np.real(np.diag(rho_rk45))
pops_bdf_final = np.real(np.diag(rho_bdf))
print(f"\n{'='*70}")
print(f"{'Method':25s} {'Time (ms)':>10s} {'vs RK45':>10s} {'vs Rust':>10s}")
print(f"{'-'*70}")
print(f"{'Python RK45':25s} {med_rk45*1000:10.0f} {'1.0x':>10s} {med_rk45/med_rust:10.1f}x")
print(f"{'Python BDF':25s} {med_bdf*1000:10.0f} {med_rk45/med_bdf:10.1f}x {med_bdf/med_rust:10.1f}x")
if med_radau:
    print(f"{'Python Radau':25s} {med_radau*1000:10.0f} {med_rk45/med_radau:10.1f}x {med_radau/med_rust:10.1f}x")
else:
    print(f"{'Python Radau':25s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}  (complex not supported)")
print(f"{'Rust DOPRI5 (sparse)':25s} {med_rust*1000:10.0f} {med_rk45/med_rust:10.1f}x {'1.0x':>10s}")
if med_tsit5:
    print(f"{'Rust Tsit5 (sparse)':25s} {med_tsit5*1000:10.0f} {med_rk45/med_tsit5:10.1f}x {med_tsit5/med_rust:10.1f}x")
print(f"\nPopulation agreement (vs ultra-tight RK45 rtol=1e-10):")
sol_ref = solve_ivp(py_rhs, t_span, rho0.reshape(-1), method="RK45", rtol=1e-10, atol=1e-12)
pops_ref = np.real(np.diag(sol_ref.y[:, -1].reshape(n, n)))
print(f"  Python RK45:  {np.max(np.abs(pops_rk45 - pops_ref)):.2e}")
print(f"  Python BDF:   {np.max(np.abs(pops_bdf_final - pops_ref)):.2e}")
print(f"  Rust DOPRI5:  {np.max(np.abs(pops_rust[-1] - pops_ref)):.2e}")
if med_tsit5:
    print(f"  Rust Tsit5:   {np.max(np.abs(pops_tsit5[-1] - pops_ref)):.2e}")
