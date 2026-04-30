import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
from centrex_tlf import couplings, lindblad, transitions
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
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
n_ground = len(system.ground)
rho0 = np.zeros((n, n), dtype=np.complex128)
for i in range(n_ground):
    rho0[i, i] = 1.0 / n_ground

t_span = (0.0, 10e-6)
saveat = np.linspace(t_span[0], t_span[1], 201)

for mode in ["structured", "structured_upper"]:
    for solver in ["scipy_bdf", "scipy_radau"]:
        evaluator = create_lindblad_rhs_evaluator_py(prepared.rust_plan, mode)
        evaluator.enable_profile_py(True)

        t0 = time.perf_counter()
        try:
            result = solve_lindblad(
                prepared, rho0, t_span,
                solver=solver, execution_mode=mode,
                saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
            )
            wall = time.perf_counter() - t0
            print(f"\n=== {solver} / {mode} ===")
            print(f"  Wall time: {wall*1000:.1f} ms")
            print(f"  Trace: {result.populations()[-1].sum():.6f}")
        except Exception as e:
            wall = time.perf_counter() - t0
            print(f"\n=== {solver} / {mode} ===")
            print(f"  FAILED after {wall*1000:.1f} ms: {e}")

print("\n=== Jacobian build cost (structured) ===")
evaluator = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "structured")
t0 = time.perf_counter()
rows, cols, values = evaluator.jacobian_split_sparse_py(0.0)
jac_time = time.perf_counter() - t0
print(f"  Time: {jac_time*1000:.1f} ms")
print(f"  nnz: {len(values)}")
dim = 2 * n * n
print(f"  Jacobian dim: {dim} x {dim}")
print(f"  Jacobian density: {len(values)/(dim*dim)*100:.2f}%")

print("\n=== Jacobian build cost (structured_upper) ===")
evaluator2 = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "structured_upper")
t0 = time.perf_counter()
rows2, cols2, values2 = evaluator2.jacobian_split_sparse_py(0.0)
jac_time2 = time.perf_counter() - t0
print(f"  Time: {jac_time2*1000:.1f} ms")
print(f"  nnz: {len(values2)}")
