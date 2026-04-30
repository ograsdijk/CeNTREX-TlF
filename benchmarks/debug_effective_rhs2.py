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
    prepare_effective_lindblad_rust_plan,
)
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.centrex_tlf_rust import debug_effective_lindblad_rhs_py

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

params = LindbladParameters({
    smp.Symbol("Ez"): static_Ez,
    smp.Symbol("\u03a90"): Gamma,
    smp.Symbol("\u03b40"): 0.0,
})

rust_plan = prepare_effective_lindblad_rust_plan(model, params)

rho0 = default_effective_density_matrix(model)
y0 = _complex_rho_to_split_real(rho0)

print(f"\ny0 norm: {np.linalg.norm(y0):.6e}")
print(f"y0 nonzero: {np.count_nonzero(np.abs(y0) > 1e-20)}")

print("\n=== Rust debug RHS ===")
result = debug_effective_lindblad_rhs_py(rust_plan, y0, 0.0)
print(f"  field_val: {result['field_val']}")
print(f"  rabi_val: {result['rabi_val']}")
print(f"  detuning_val: {result['detuning_val']}")
print(f"  last_interval: {result['last_interval']}")
print(f"  l_scratch_norm: {result['l_scratch_norm']:.6e}")
print(f"  dy_norm: {result['dy_norm']:.6e}")
print(f"  dy_max: {result['dy_max']:.6e}")
print(f"  has_nan: {result['has_nan']}")
print(f"  has_inf: {result['has_inf']}")

dy_rust = np.asarray(result['dy'])
print(f"  dy[:10]: {dy_rust[:10]}")

print("\n=== Python reference RHS ===")
bundle = model.effective_bundle(static_Ez)
L_int = _hamiltonian_superoperator(bundle.h_internal)
L_opt = _hamiltonian_superoperator(bundle.h_opt)
L_det = _hamiltonian_superoperator(bundle.h_det)
L_diss = bundle.dissipator_superoperator()
L_total = (L_int + L_diss) + 0.5 * Gamma * L_opt
L_total_real = _complex_superop_to_split_real(L_total)
dy_python = L_total_real @ y0
print(f"  dy_norm: {np.linalg.norm(dy_python):.6e}")
print(f"  dy[:10]: {dy_python[:10]}")

if not result['has_nan'] and not result['has_inf']:
    diff = np.max(np.abs(dy_rust - dy_python))
    rel_diff = diff / max(np.linalg.norm(dy_python), 1e-30)
    print(f"\n  Max absolute diff: {diff:.6e}")
    print(f"  Relative diff: {rel_diff:.6e}")
