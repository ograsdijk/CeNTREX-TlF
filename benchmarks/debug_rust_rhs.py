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

transition = transitions.OpticalTransition(
    transitions.OpticalTransitionType.R, 0, 1/2, 1
)
polarization = couplings.polarization_Z
Gamma = 2 * np.pi * 1.56e6
static_Ez = 10.0

model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
    transition=transition,
    optical_polarization=polarization,
)
n = model.n_effective_states

bundle = model.effective_bundle(static_Ez)
L_int = _hamiltonian_superoperator(bundle.h_internal)
L_opt = _hamiltonian_superoperator(bundle.h_opt)
L_diss = bundle.dissipator_superoperator()
L_combined = L_int + L_diss

L_combined_real = _complex_superop_to_split_real(L_combined)
L_opt_real = _complex_superop_to_split_real(L_opt)
L_det_real = _complex_superop_to_split_real(_hamiltonian_superoperator(bundle.h_det))

rho0 = default_effective_density_matrix(model)
y0 = _complex_rho_to_split_real(rho0)

L_total_real = L_combined_real + 0.5 * Gamma * L_opt_real
dy_python = L_total_real @ y0
print(f"Python dy norm: {np.linalg.norm(dy_python):.6e}")
print(f"Python dy[:5]: {dy_python[:5]}")

params = LindbladParameters({
    smp.Symbol("Ez"): static_Ez,
    smp.Symbol("\u03a90"): Gamma,
    smp.Symbol("\u03b40"): 0.0,
})

rust_plan = prepare_effective_lindblad_rust_plan(model, params)

print(f"\nRust plan real_dim: {rust_plan.real_dim}")
print(f"Rust plan n_grid: {rust_plan.n_grid}")
print(f"Rust plan field_coordinate_slot: {rust_plan.field_coordinate_slot}")
print(f"Rust plan rabi_rate_slot: {rust_plan.rabi_rate_slot}")
print(f"Rust plan detuning_slot: {rust_plan.detuning_slot}")

from centrex_tlf.centrex_tlf_rust import solve_effective_lindblad_py

idx = 2
mat_size = (2 * n * n) ** 2
print(f"\nChecking Rust plan assembly for grid point {idx} (Ez={model.field_points[idx]})...")
lc_flat = np.array([L_combined_real.ravel()])
lo_flat = np.array([L_opt_real.ravel()])
dy_manual = (L_combined_real + 0.5 * Gamma * L_opt_real) @ y0
print(f"Manual matvec dy norm: {np.linalg.norm(dy_manual):.6e}")
print(f"Manual matvec dy nonzero count: {np.count_nonzero(np.abs(dy_manual) > 1e-20)}")
print(f"y0 nonzero count: {np.count_nonzero(np.abs(y0) > 1e-20)}")
print(f"y0 norm: {np.linalg.norm(y0):.6e}")
print(f"L_combined_real norm: {np.linalg.norm(L_combined_real):.6e}")
print(f"L_combined_real max abs: {np.max(np.abs(L_combined_real)):.6e}")
print(f"L_combined_real shape: {L_combined_real.shape}")
print(f"L eigenvalues max real: {np.max(np.real(np.linalg.eigvals(L_combined_real + 0.5 * Gamma * L_opt_real))):.6e}")

try:
    times, states = solve_effective_lindblad_py(
        rust_plan, y0, 0.0, 1e-15,
        1e-9, 1e-7, 1e-15,
        None, True, 10,
    )
    print(f"\nTiny solve succeeded: {len(times)} points")
    if len(times) > 0:
        print(f"  dy rust approx: {(states[0] - y0)[:5] if states.shape[0] > 0 else 'N/A'}")
except Exception as e:
    print(f"\nTiny solve FAILED: {e}")
    print("Trying with even smaller dt and 1 step...")
    try:
        times, states = solve_effective_lindblad_py(
            rust_plan, y0, 0.0, 1e-18,
            1e100, 1e100, 1e-18,
            None, True, 2,
        )
        print(f"  Result: {len(times)} points")
        if states.shape[0] > 1:
            dy_approx = (states[-1] - y0) / 1e-18
            print(f"  dy/dt approx norm: {np.linalg.norm(dy_approx):.6e}")
            print(f"  dy/dt approx[:5]: {dy_approx[:5]}")
            print(f"  ratio vs python: {np.linalg.norm(dy_approx) / np.linalg.norm(dy_python):.6e}")
    except Exception as e2:
        print(f"  Also failed: {e2}")
