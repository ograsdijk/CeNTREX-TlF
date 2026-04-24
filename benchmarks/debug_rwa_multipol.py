import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings as couplings_mod, lindblad

# Q1 with Z only (single polarization - should have the bug)
print("=== Q1 with Z polarization only ===")
t1 = transitions.Q1_F1_1o2_F0
ts1 = couplings_mod.generate_transition_selectors([t1], [[couplings_mod.polarization_Z]])
sys1 = lindblad.generate_OBE_system_transitions([t1], ts1, E=[0,0,10], B=[0,0,1e-5], qn_compact=True)
H1 = sys1.H_symbolic
syms1 = list(H1.free_symbols)
vals1 = {s: (0 if any(c in str(s) for c in ["Ω", "δ"]) else 1) for s in syms1}
H1_eval = np.array(H1.subs(vals1).tolist(), dtype=np.complex128)
diag1 = np.real(np.diag(H1_eval)) / (2*np.pi*1e6)
print(f"Max |diag| (MHz): {np.max(np.abs(diag1)):.1f}")
print(f"States > 100 MHz: {np.sum(np.abs(diag1) > 100)}")

# Q1 with Z+X polarization (should be fixed)
print("\n=== Q1 with Z+X polarization ===")
ts2 = couplings_mod.generate_transition_selectors(
    [t1], [[couplings_mod.polarization_Z, couplings_mod.polarization_X]]
)
sys2 = lindblad.generate_OBE_system_transitions([t1], ts2, E=[0,0,10], B=[0,0,1e-5], qn_compact=True)
H2 = sys2.H_symbolic
syms2 = list(H2.free_symbols)
vals2 = {}
for s in syms2:
    name = str(s)
    if "Ω" in name: vals2[s] = 0
    elif "δ" in name: vals2[s] = 0
    elif "PZ" in name: vals2[s] = 1
    elif "PX" in name: vals2[s] = 0
    else: vals2[s] = 0
H2_eval = np.array(H2.subs(vals2).tolist(), dtype=np.complex128)
diag2 = np.real(np.diag(H2_eval)) / (2*np.pi*1e6)
print(f"Max |diag| (MHz): {np.max(np.abs(diag2)):.1f}")
print(f"States > 100 MHz: {np.sum(np.abs(diag2) > 100)}")
print(f"Diag (MHz): {diag2}")
