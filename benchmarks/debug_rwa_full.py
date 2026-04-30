import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings as couplings_mod, lindblad, states
from centrex_tlf.lindblad.generate_hamiltonian import (
    generate_rwa_symbolic_hamiltonian,
    generate_total_symbolic_hamiltonian,
)
from centrex_tlf.lindblad.utils_setup import _generate_couplings

t = transitions.Q1_F1_1o2_F0
ts_list = couplings_mod.generate_transition_selectors([t], [[couplings_mod.polarization_Z]])

E = np.array([0, 0, 10.0])
B = np.array([0, 0, 1e-5])

from centrex_tlf.hamiltonian import generate_reduced_hamiltonian_transitions
reduced = generate_reduced_hamiltonian_transitions([t], E=E, B=B, use_omega_basis=False)

QN = list(reduced.QN)
H_int = np.asarray(reduced.H_int, dtype=np.complex128)
V_ref_int = np.asarray(reduced.V_ref_int, dtype=np.complex128)
QN_basis = reduced.QN_basis

print(f"Full system: {len(QN)} states, H_int shape: {H_int.shape}")

coupling_fields = _generate_couplings(ts_list, QN_basis, H_int, QN, V_ref_int, True)

Omegas = [ts.Ω for ts in ts_list]
Deltas = [ts.δ for ts in ts_list]
pols = []
for trans in ts_list:
    if not trans.polarization_symbols:
        pols.append(None)
    else:
        pols.append(trans.polarization_symbols)

H_rwa_full = generate_rwa_symbolic_hamiltonian(QN, H_int, coupling_fields, Omegas, Deltas, pols)
print(f"H_rwa_full shape: {H_rwa_full.shape}")

symbols = list(H_rwa_full.free_symbols)
vals = {}
for s in symbols:
    name = str(s)
    if "Ω" in name or "Omega" in name:
        vals[s] = 0
    elif "δ" in name or "delta" in name:
        vals[s] = 0
    elif "PZ" in name:
        vals[s] = 1
    else:
        vals[s] = 0

H_eval = np.array(H_rwa_full.subs(vals).tolist(), dtype=np.complex128)
print(f"\nFull RWA H diagonal at Omega=0, delta=0 (MHz):")
for i in range(len(QN)):
    qn = QN[i]
    largest = qn.largest
    F1 = getattr(largest, "F1", "?")
    F = getattr(largest, "F", "?")
    mF = getattr(largest, "mF", "?")
    e_mhz = np.real(H_eval[i, i]) / (2 * np.pi * 1e6)
    print(f"  {i}: {largest.electronic_state} J={largest.J} F1={F1} F={F} mF={mF} -> {e_mhz:.4f} MHz")
