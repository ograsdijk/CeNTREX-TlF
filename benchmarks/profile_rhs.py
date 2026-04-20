import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from centrex_tlf import couplings, lindblad, transitions
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py

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
print(f"n_states: {n}, n_collapse: {system.C_array.shape[0]}")

rho0 = np.eye(n, dtype=np.complex128) / n
packed = prepared.layout.pack(rho0)

for mode in ["reference", "structured", "structured_upper"]:
    ev = create_lindblad_rhs_evaluator_py(prepared.rust_plan, mode)
    ev.enable_profile_py(True)
    for _ in range(500):
        ev.rhs_packed_py(packed, 0.0)
    s = ev.profile_summary_py()
    print(f"\n=== {mode} ({s['calls']} calls) ===")
    for k in [
        "total_seconds",
        "unpack_seconds",
        "parameter_eval_seconds",
        "hamiltonian_fill_seconds",
        "commutator_seconds",
        "dissipator_seconds",
        "pack_seconds",
    ]:
        pct = s[k] / s["total_seconds"] * 100 if s["total_seconds"] > 0 else 0
        print(f"  {k:30s}: {s[k]*1000:8.2f} ms  ({pct:5.1f}%)")
    print(f"  avg per call: {s['average_total_seconds']*1e6:.1f} us")
