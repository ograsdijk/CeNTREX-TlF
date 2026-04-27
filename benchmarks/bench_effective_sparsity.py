import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
)
from centrex_tlf.effective_hamiltonian._superoperators import _hamiltonian_superoperator
from centrex_tlf.effective_hamiltonian.rust_plan import _complex_superop_to_split_real

transition = transitions.Q1_F1_1o2_F0
polarization = couplings.polarization_Z

model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 7, 7.5, 8, 10, 20, 30, 40, 50],
    transition=transition, optical_polarization=polarization,
)
n = model.n_effective_states
print(f"n_states: {n}, n²: {n*n}, split-real dim: {2*n*n}")

for field_z in [0, 10, 50]:
    bundle = model.effective_bundle(float(field_z))
    L_int = _hamiltonian_superoperator(bundle.h_internal)
    L_opt = _hamiltonian_superoperator(bundle.h_opt)
    L_det = _hamiltonian_superoperator(bundle.h_det)
    L_diss = bundle.dissipator_superoperator()
    L_combined = L_int + L_diss

    print(f"\nAt Ez = {field_z} V/cm:")
    for name, L in [("L_combined", L_combined), ("L_opt", L_opt), ("L_det", L_det)]:
        L_real = _complex_superop_to_split_real(L)
        dim = L_real.shape[0]
        nnz = np.count_nonzero(np.abs(L_real) > 1e-15)
        total = dim * dim
        density = nnz / total * 100
        mem_dense = total * 8 / 1024
        mem_sparse = (nnz * 12 + (dim + 1) * 8) / 1024  # CSR: values + col_idx + row_ptr
        print(f"  {name:12s}: dim={dim}, nnz={nnz:6d}/{total:6d} ({density:5.1f}%), "
              f"dense={mem_dense:.0f} KB, sparse={mem_sparse:.0f} KB, ratio={mem_sparse/mem_dense:.2f}")
