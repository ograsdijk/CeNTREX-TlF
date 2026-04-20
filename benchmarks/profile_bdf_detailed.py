import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import scipy.sparse
from scipy.integrate import solve_ivp
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
n_ground = len(system.ground)

rho0 = np.zeros((n, n), dtype=np.complex128)
for i in range(n_ground):
    rho0[i, i] = 1.0 / n_ground

t_span = (0.0, 10e-6)
saveat = np.linspace(t_span[0], t_span[1], 201)

for mode in ["structured", "structured_upper"]:
    evaluator = create_lindblad_rhs_evaluator_py(prepared.rust_plan, mode)
    evaluator.enable_profile_py(True)

    flat = rho0.reshape(-1).astype(np.complex128)
    y0 = np.concatenate((flat.real, flat.imag)).astype(np.float64)

    rhs_count = [0]
    rhs_total = [0.0]
    jac_count = [0]
    jac_total = [0.0]
    jacobian_cache = [None]

    def rhs(t, y):
        t0 = time.perf_counter()
        result = np.asarray(
            evaluator.rhs_split_py(np.ascontiguousarray(y, dtype=np.float64), t),
            dtype=np.float64,
        )
        rhs_total[0] += time.perf_counter() - t0
        rhs_count[0] += 1
        return result

    def build_jacobian(t, _y=None):
        if jacobian_cache[0] is not None:
            return jacobian_cache[0]
        t0 = time.perf_counter()
        rows, cols, values = evaluator.jacobian_split_sparse_py(t)
        rows_arr = np.asarray(rows, dtype=np.int64)
        cols_arr = np.asarray(cols, dtype=np.int64)
        values_arr = np.asarray(values, dtype=np.float64)
        dim = y0.size
        matrix = scipy.sparse.csc_matrix((values_arr, (rows_arr, cols_arr)), shape=(dim, dim))
        jacobian_cache[0] = matrix
        jac_total[0] += time.perf_counter() - t0
        jac_count[0] += 1
        return matrix

    wall_start = time.perf_counter()
    solution = solve_ivp(
        rhs, t_span=t_span, y0=y0, method="BDF",
        atol=1e-9, rtol=1e-7, first_step=1e-10,
        t_eval=saveat, max_step=np.inf,
        jac=build_jacobian,
    )
    wall = time.perf_counter() - wall_start

    profile = evaluator.profile_summary_py()
    rust_rhs_total = profile["total_seconds"]

    print(f"\n=== BDF / {mode} ===")
    print(f"  Wall time:         {wall*1000:.1f} ms")
    print(f"  RHS calls:         {rhs_count[0]}")
    print(f"  RHS total (wall):  {rhs_total[0]*1000:.1f} ms")
    print(f"  RHS total (rust):  {rust_rhs_total*1000:.1f} ms")
    print(f"  RHS per call:      {rhs_total[0]/rhs_count[0]*1e6:.1f} us")
    print(f"  Jacobian calls:    {jac_count[0]}")
    print(f"  Jacobian total:    {jac_total[0]*1000:.1f} ms")
    print(f"  scipy overhead:    {(wall - rhs_total[0] - jac_total[0])*1000:.1f} ms")
    print(f"  Success: {solution.success}")
