import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian.rust_plan import (
    prepare_effective_lindblad_rust_plan,
    _complex_rho_to_split_real,
)
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.centrex_tlf_rust import debug_effective_lindblad_rhs_py

t = transitions.OpticalTransition(transitions.OpticalTransitionType.R, 0, 1/2, 1)
p = couplings.polarization_Z
Gamma = 2 * np.pi * 1.56e6

m = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50], transition=t, optical_polarization=p
)

params = LindbladParameters({
    smp.Symbol("Ez"): 10.0,
    smp.Symbol("\u03a90"): Gamma,
    smp.Symbol("\u03b40"): 0.0,
})

import centrex_tlf.effective_hamiltonian.rust_plan as rp
import importlib
importlib.reload(rp)
plan = rp.prepare_effective_lindblad_rust_plan(m, params)

# Also check the h_internal_rwa diagonal
b = m.effective_bundle(10.0)
h_rwa = np.array(b.h_internal, dtype=np.complex128).copy()
excited_idx = np.asarray(m.excited_indices, dtype=np.int64)
rwa_s = m.common_omega_reference
print(f"common_omega_reference: {rwa_s:.6e}")
print(f"excited_indices: {excited_idx}")
print(f"h_internal diag before RWA: {np.real(np.diag(b.h_internal))}")
for idx in excited_idx:
    h_rwa[idx, idx] += rwa_s
print(f"h_internal diag after RWA:  {np.real(np.diag(h_rwa))}")
print(f"h_internal_rwa norm: {np.linalg.norm(h_rwa):.6e}")
rho0 = default_effective_density_matrix(m)
y0 = _complex_rho_to_split_real(rho0)

r = debug_effective_lindblad_rhs_py(plan, y0, 0.0)
print(f"l_scratch_norm: {r['l_scratch_norm']:.6e}")
print(f"dy_norm: {r['dy_norm']:.6e}")
print(f"dy_max: {r['dy_max']:.6e}")
print(f"has_nan: {r['has_nan']}, has_inf: {r['has_inf']}")
print(f"field_val: {r['field_val']}")
print(f"rabi_val: {r['rabi_val']:.6e}")
