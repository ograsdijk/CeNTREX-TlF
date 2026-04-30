import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian._superoperators import (
    _hamiltonian_superoperator,
)
from centrex_tlf.effective_hamiltonian.rust_plan import (
    _complex_superop_to_split_real,
    _complex_rho_to_split_real,
    _split_real_to_complex_rho,
)

transition = transitions.OpticalTransition(
    transitions.OpticalTransitionType.R, 0, 1/2, 1
)
polarization = couplings.polarization_Z
Gamma = 2 * np.pi * 1.56e6
static_Ez = 10.0

print("Preparing model...")
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
    transition=transition,
    optical_polarization=polarization,
)
n = model.n_effective_states
print(f"n_states: {n}")

bundle = model.effective_bundle(static_Ez)
print(f"bundle.omega_reference: {bundle.omega_reference}")
print(f"bundle.h_internal norm: {np.linalg.norm(bundle.h_internal)}")
print(f"bundle.h_opt norm: {np.linalg.norm(bundle.h_opt)}")
print(f"bundle.h_det norm: {np.linalg.norm(bundle.h_det)}")

L_int = _hamiltonian_superoperator(bundle.h_internal)
L_opt = _hamiltonian_superoperator(bundle.h_opt)
L_det = _hamiltonian_superoperator(bundle.h_det)
L_diss = bundle.dissipator_superoperator()
L_combined = L_int + L_diss
L_total = L_combined + 0.5 * Gamma * L_opt

print(f"\nL_combined norm: {np.linalg.norm(L_combined)}")
print(f"L_opt norm: {np.linalg.norm(L_opt)}")
print(f"L_total norm: {np.linalg.norm(L_total)}")

rho0 = default_effective_density_matrix(model)
rho_flat = rho0.reshape(-1)
drho_flat = L_total @ rho_flat
drho = drho_flat.reshape(n, n)
print(f"\ndrho norm (complex): {np.linalg.norm(drho)}")
print(f"drho trace: {np.trace(drho)}")

L_real = _complex_superop_to_split_real(L_total)
y0 = _complex_rho_to_split_real(rho0)
dy = L_real @ y0
print(f"\ndy norm (split-real): {np.linalg.norm(dy)}")

rho_check = _split_real_to_complex_rho(y0, n)
print(f"rho roundtrip error: {np.linalg.norm(rho_check - rho0)}")

drho_check = _split_real_to_complex_rho(dy, n)
print(f"drho roundtrip error: {np.linalg.norm(drho_check - drho)}")

print(f"\nL_total eigenvalues (max real part): {np.max(np.real(np.linalg.eigvals(L_total)))}")
print(f"L_real eigenvalues (max real part): {np.max(np.real(np.linalg.eigvals(L_real)))}")
