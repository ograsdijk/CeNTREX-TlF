import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from centrex_tlf import couplings, lindblad, transitions
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.centrex_tlf_rust import evaluate_lindblad_hamiltonian_py

trans = transitions.R0_F1_3o2_F2
ts = couplings.generate_transition_selectors([trans], [[couplings.polarization_Z]])
system = lindblad.generate_OBE_system_transitions([trans], ts, method="matrix")

params = {str(s): 0.0 for s in system.H_symbolic.free_symbols}
for s in system.coupling_symbols:
    params[str(s)] = 2 * np.pi * 1.56e6
for group in system.polarization_symbols:
    for s in group if isinstance(group, (list, tuple)) else [group]:
        params[str(s)] = 1.0

prepared = prepare_lindblad_problem(system, params, backend="rust")
n = len(system.QN)

H = np.asarray(evaluate_lindblad_hamiltonian_py(prepared.rust_plan, 0.0))

print(f"n = {n}")
print(f"n^2 = {n*n}")
print(f"nnz (|H[i,j]| > 1e-15) = {np.count_nonzero(np.abs(H) > 1e-15)}")
print(f"nnz upper (|H[i,j]| > 1e-15, i<=j) = {np.count_nonzero(np.abs(np.triu(H)) > 1e-15)}")
print(f"density = {np.count_nonzero(np.abs(H) > 1e-15) / (n*n):.4f}")

nnz_per_row = np.array([np.count_nonzero(np.abs(H[i, :]) > 1e-15) for i in range(n)])
print(f"nnz per row: min={nnz_per_row.min()}, max={nnz_per_row.max()}, mean={nnz_per_row.mean():.1f}, median={np.median(nnz_per_row):.0f}")

print(f"\nO(n^3) = {n**3}")
print(f"O(nnz * n) = {np.count_nonzero(np.abs(H) > 1e-15) * n}")
print(f"Theoretical speedup = {n**3 / (np.count_nonzero(np.abs(H) > 1e-15) * n):.1f}x")

print("\n--- Block structure (nonzero pattern) ---")
n_ground = len(system.ground)
n_excited = len(system.excited)
blocks = {
    "ground-ground": H[:n_ground, :n_ground],
    "ground-excited": H[:n_ground, n_ground:],
    "excited-ground": H[n_ground:, :n_ground],
    "excited-excited": H[n_ground:, n_ground:],
}
for name, block in blocks.items():
    total = block.size
    nnz = np.count_nonzero(np.abs(block) > 1e-15)
    print(f"  {name}: {block.shape}, nnz={nnz}/{total} ({nnz/total*100:.1f}%)")
