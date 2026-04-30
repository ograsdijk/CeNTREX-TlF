import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian._superoperators import _hamiltonian_superoperator
from centrex_tlf.effective_hamiltonian.rust_plan import (
    _complex_superop_to_split_real,
    _complex_rho_to_split_real,
    _split_real_to_complex_rho,
)

t = transitions.OpticalTransition(transitions.OpticalTransitionType.R, 0, 1/2, 1)
p = couplings.polarization_Z
Gamma = 2 * np.pi * 1.56e6

m = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50], transition=t, optical_polarization=p
)
n = m.n_effective_states

bundle = m.effective_bundle(10.0)
L_int = _hamiltonian_superoperator(bundle.h_internal)
L_opt = _hamiltonian_superoperator(bundle.h_opt)
L_diss = bundle.dissipator_superoperator()
L_total = (L_int + L_diss) + 0.5 * Gamma * L_opt
L_real = _complex_superop_to_split_real(L_total)

rho0 = default_effective_density_matrix(m)
y0 = _complex_rho_to_split_real(rho0)

eigs = np.linalg.eigvals(L_real)
max_imag = np.max(np.abs(np.imag(eigs)))
max_real = np.max(np.real(eigs))
print(f"L_real eigenvalues: max |imag| = {max_imag:.6e}, max real = {max_real:.6e}")
print(f"Stability limit (DOPRI5): h < {3.3/max_imag:.6e}")

for dt in [1e-10, 5e-10, 1e-11, 1e-12]:
    y1 = y0 + dt * (L_real @ y0)
    dy1 = L_real @ y1
    y2 = y1 + dt * dy1
    rho1 = _split_real_to_complex_rho(y1, n)
    trace1 = np.real(np.trace(rho1))
    pops1 = np.real(np.diag(rho1))
    norm_y1 = np.linalg.norm(y1)
    norm_dy1 = np.linalg.norm(dy1)
    
    sk = 1e-9 + 1e-7 * np.maximum(np.abs(y0), np.abs(y1))
    err_est = dt * np.abs(L_real @ y0) * 0.02
    err_rms = np.sqrt(np.mean((err_est / sk)**2))
    
    print(f"\ndt={dt:.0e}: trace={trace1:.8f} norm_y={norm_y1:.6e} norm_dy={norm_dy1:.6e} err_rms={err_rms:.6e}")
    if np.any(np.isnan(y1)) or np.any(np.isinf(y1)):
        print("  NaN/Inf detected!")
    if norm_y1 > 100:
        print("  UNSTABLE")

print("\n=== scipy RK45 test (1 step) ===")
from scipy.integrate import solve_ivp
sol = solve_ivp(lambda t, y: L_real @ y, (0, 1e-10), y0, method="RK45", max_step=1e-10, dense_output=False)
print(f"scipy t_events: {sol.t}")
print(f"scipy success: {sol.success}")
rho_sp = _split_real_to_complex_rho(sol.y[:, -1], n)
print(f"scipy trace: {np.real(np.trace(rho_sp)):.8f}")
