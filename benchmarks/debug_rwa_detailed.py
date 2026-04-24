import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings, lindblad, states
from centrex_tlf.lindblad.generate_hamiltonian import (
    generate_rwa_symbolic_hamiltonian,
    symbolic_hamiltonian_to_rotating_frame,
    generate_symbolic_hamiltonian,
    generate_unitary_transformation_matrix,
)

t = transitions.Q1_F1_1o2_F0
ts_list = couplings.generate_transition_selectors([t], [[couplings.polarization_Z]])
sys_ = lindblad.generate_OBE_system_transitions(
    [t], ts_list, E=[0, 0, 10], B=[0, 0, 1e-5], qn_compact=True
)

QN = sys_.QN
H_int = np.asarray(sys_.H_int, dtype=np.complex128)
coupling_fields = sys_.couplings

print(f"QN count: {len(QN)}")
print(f"H_int shape: {H_int.shape}")
print(f"Couplings count: {len(coupling_fields)}")

for idc, cf in enumerate(coupling_fields):
    print(f"\nCoupling {idc}:")
    print(f"  ground_main: {cf.ground_main}")
    print(f"  excited_main: {cf.excited_main}")
    idg = QN.index(cf.ground_main)
    ide = QN.index(cf.excited_main)
    print(f"  ground_main index in QN: {idg}")
    print(f"  excited_main index in QN: {ide}")
    print(f"  H_int[idg,idg] (GHz): {np.real(H_int[idg,idg])/(2*np.pi*1e9):.6f}")
    print(f"  H_int[ide,ide] (GHz): {np.real(H_int[ide,ide])/(2*np.pi*1e9):.6f}")

# Now check: H_int is 37x37 but QN is 15 states (compact)
# QN.index() searches in the compact list, but H_int is the full matrix
# This is the bug!
print(f"\n=== Key issue ===")
print(f"H_int is {H_int.shape[0]}x{H_int.shape[0]} (full system)")
print(f"QN has {len(QN)} states (compact system)")
print(f"QN.index(ground_main) returns index in compact QN list")
print(f"But the RWA code uses this index to access H_int diagonal!")
print(f"If H_int is 37x37 and idg=3, it accesses the WRONG state")

# Check H_symbolic
H_sym = sys_.H_symbolic
print(f"\nH_symbolic shape: {H_sym.shape}")
print(f"H_symbolic is {H_sym.shape[0]}x{H_sym.shape[0]} (compact)")
