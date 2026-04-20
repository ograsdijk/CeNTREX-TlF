import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
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

evaluator = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "structured")
evaluator.enable_profile_py(True)

rho0 = np.eye(n, dtype=np.complex128) / n
packed = prepared.layout.pack(rho0)
flat = rho0.reshape(-1).astype(np.complex128)
split = np.concatenate((flat.real, flat.imag)).astype(np.float64)

n_calls = 2000

print("=== Direct Rust calls (no scipy) ===")

evaluator.reset_profile_py()
t0 = time.perf_counter()
for _ in range(n_calls):
    evaluator.rhs_packed_py(packed, 0.0)
wall = time.perf_counter() - t0
profile = evaluator.profile_summary_py()
print(f"  rhs_packed_py: {wall/n_calls*1e6:.1f} us/call (wall), {profile['average_total_seconds']*1e6:.1f} us/call (rust internal)")
print(f"  Python overhead: {(wall/n_calls - profile['average_total_seconds'])*1e6:.1f} us/call")

evaluator_upper = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "structured_upper")
evaluator_upper.enable_profile_py(True)

evaluator_upper.reset_profile_py()
t0 = time.perf_counter()
for _ in range(n_calls):
    evaluator_upper.rhs_packed_py(packed, 0.0)
wall = time.perf_counter() - t0
profile = evaluator_upper.profile_summary_py()
print(f"\n  rhs_packed_py (upper): {wall/n_calls*1e6:.1f} us/call (wall), {profile['average_total_seconds']*1e6:.1f} us/call (rust internal)")
print(f"  Python overhead: {(wall/n_calls - profile['average_total_seconds'])*1e6:.1f} us/call")

print("\n=== rhs_matrix_py (used by scipy RK45 path) ===")
evaluator2 = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "structured")
evaluator2.enable_profile_py(True)
evaluator2.reset_profile_py()
t0 = time.perf_counter()
for _ in range(n_calls):
    evaluator2.rhs_matrix_py(flat, 0.0)
wall = time.perf_counter() - t0
profile = evaluator2.profile_summary_py()
print(f"  rhs_matrix_py: {wall/n_calls*1e6:.1f} us/call (wall), {profile['average_total_seconds']*1e6:.1f} us/call (rust internal)")
print(f"  Python overhead: {(wall/n_calls - profile['average_total_seconds'])*1e6:.1f} us/call")

print("\n=== rhs_split_py (used by scipy BDF/Radau path) ===")
evaluator3 = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "structured")
evaluator3.enable_profile_py(True)
evaluator3.reset_profile_py()
t0 = time.perf_counter()
for _ in range(n_calls):
    evaluator3.rhs_split_py(split, 0.0)
wall = time.perf_counter() - t0
profile = evaluator3.profile_summary_py()
print(f"  rhs_split_py: {wall/n_calls*1e6:.1f} us/call (wall), {profile['average_total_seconds']*1e6:.1f} us/call (rust internal)")
print(f"  Python overhead: {(wall/n_calls - profile['average_total_seconds'])*1e6:.1f} us/call")

print("\n=== With numpy lambda wrapping (as scipy sees it) ===")
fn_matrix = lambda t, y: np.asarray(evaluator.rhs_matrix_py(y, t), dtype=np.complex128)
t0 = time.perf_counter()
for _ in range(n_calls):
    fn_matrix(0.0, flat)
wall = time.perf_counter() - t0
print(f"  lambda + np.asarray: {wall/n_calls*1e6:.1f} us/call")

fn_split = lambda t, y: np.asarray(evaluator3.rhs_split_py(np.ascontiguousarray(y, dtype=np.float64), t), dtype=np.float64)
t0 = time.perf_counter()
for _ in range(n_calls):
    fn_split(0.0, split)
wall = time.perf_counter() - t0
print(f"  lambda + ascontiguousarray + np.asarray (split): {wall/n_calls*1e6:.1f} us/call")
