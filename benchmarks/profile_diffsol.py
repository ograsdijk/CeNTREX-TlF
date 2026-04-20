import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
from centrex_tlf import couplings, lindblad, transitions
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad

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

for run in range(3):
    t0 = time.perf_counter()
    result = solve_lindblad(
        prepared, rho0, t_span,
        solver="bdf", execution_mode="structured_upper",
        saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
    )
    elapsed = time.perf_counter() - t0
    print(f"Run {run+1}: {elapsed*1000:.1f} ms, trace={result.populations()[-1].sum():.6f}")
