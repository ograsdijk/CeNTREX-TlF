import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from centrex_tlf import transitions, couplings, lindblad, states

t = transitions.Q1_F1_1o2_F0
ts = couplings.generate_transition_selectors([t], [[couplings.polarization_Z]])
sys_ = lindblad.generate_OBE_system_transitions(
    [t], ts, E=[0, 0, 10], B=[0, 0, 1e-5], qn_compact=True
)

print("Compact states:")
for i, qn in enumerate(sys_.QN):
    largest = qn.largest
    F1 = getattr(largest, "F1", "?")
    F = getattr(largest, "F", "?")
    mF = getattr(largest, "mF", "?")
    print(f"  {i}: {largest.electronic_state} J={largest.J} F1={F1} F={F} mF={mF}")

print(f"\nH_int shape: {np.asarray(sys_.H_int).shape}")
print(f"H_symbolic shape: {sys_.H_symbolic.shape}")
print(f"QN count: {len(sys_.QN)}")

# The H_int has more entries than QN - it's the full (not compact) Hamiltonian
# Check the full QN
print(f"\nQN_original count: {len(sys_.QN_original)}")
print("Full states (QN_original):")
for i, qn in enumerate(sys_.QN_original):
    largest = qn.largest
    F1 = getattr(largest, "F1", "?")
    F = getattr(largest, "F", "?")
    mF = getattr(largest, "mF", "?")
    print(f"  {i}: {largest.electronic_state} J={largest.J} F1={F1} F={F} mF={mF}")

H_int = np.asarray(sys_.H_int, dtype=np.complex128)
print(f"\nH_int diagonal (GHz) for full states:")
for i in range(min(H_int.shape[0], len(sys_.QN_original))):
    qn = sys_.QN_original[i]
    largest = qn.largest
    print(f"  {i}: J={largest.J} F1={getattr(largest, 'F1', '?')} -> {np.real(H_int[i,i])/(2*np.pi*1e9):.6f} GHz")
