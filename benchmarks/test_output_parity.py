import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings, lindblad
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian.rust_plan import (
    prepare_effective_lindblad_rust_plan,
    solve_effective_lindblad,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)

print("=" * 60)
print("Output Mode Parity Test")
print("=" * 60)

# === OBE solver ===
print("\n--- OBE Solver ---")
trans = transitions.R0_F1_3o2_F2
ts = couplings.generate_transition_selectors([trans], [[couplings.polarization_Z]])
system = lindblad.generate_OBE_system_transitions([trans], ts, method="matrix")
Gamma = 2 * np.pi * 1.56e6
params_obe = {str(s): 0.0 for s in system.H_symbolic.free_symbols}
for s in system.coupling_symbols: params_obe[str(s)] = Gamma
for group in system.polarization_symbols:
    for s in group if isinstance(group, (list, tuple)) else [group]: params_obe[str(s)] = 1.0
prepared = prepare_lindblad_problem(system, params_obe, backend="rust")
n = len(system.QN)
rho0_obe = np.zeros((n, n), dtype=np.complex128)
for i in range(len(system.ground)): rho0_obe[i, i] = 1.0 / len(system.ground)
saveat = np.linspace(0, 10e-6, 51)

for mode in ["full", "populations", "selected"]:
    kwargs = {}
    if mode == "selected":
        kwargs["output_indices"] = [(0, 0), (1, 1), (0, 1)]
    try:
        r = solve_lindblad(prepared, rho0_obe, (0, 10e-6), solver="dopri5_fast",
                          execution_mode="structured_upper", saveat=saveat,
                          dt=1e-10, reltol=1e-7, abstol=1e-9, output=mode, **kwargs)
        print(f"  {mode:20s}: OK")
    except Exception as e:
        print(f"  {mode:20s}: FAILED - {e}")

# weighted_integral
try:
    weights = [(i, 1.0) for i in range(n)]  # sum all populations = trace
    r = solve_lindblad(prepared, rho0_obe, (0, 10e-6), solver="dopri5_fast",
                      execution_mode="structured_upper", saveat=saveat,
                      dt=1e-10, reltol=1e-7, abstol=1e-9,
                      output="weighted_integral", integral_weights=weights)
    print(f"  {'weighted_integral':20s}: OK (integral={r.values[0]:.6e})")
except Exception as e:
    print(f"  {'weighted_integral':20s}: FAILED - {e}")

# === Effective Hamiltonian solver ===
print("\n--- Effective Hamiltonian Solver ---")
transition = transitions.Q1_F1_1o2_F0
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
    transition=transition, optical_polarization=couplings.polarization_Z,
)
rho0_eff = default_effective_density_matrix(model)
n_eff = model.n_effective_states

te = Time()
v = Parameter("v", 200.0)
z0 = Parameter("z0", -0.005)
omega0 = Parameter("omega0", Gamma)
z_laser = Parameter("z_laser", 0.002)
w0 = Parameter("w0", 200e-6)
z_grid = np.linspace(-0.01, 0.01, 50)
ez = 10.0 + 15.0 * np.exp(-z_grid**2 / (2 * 0.003**2))
z = linear(te, offset=z0, slope=v)
gp = Parameter("z_grid", tuple(z_grid.tolist()))
ep = Parameter("ez", tuple(ez.tolist()))
Ez = pchip_tabulated(z, gp, ep)
Omega = gaussian(z, center=z_laser, sigma=w0, amplitude=omega0)
params_eff = LindbladParameters({
    smp.Symbol("Ez"): Ez, smp.Symbol("\u03a90"): Omega, smp.Symbol("\u03b40"): 0.0,
})
plan = prepare_effective_lindblad_rust_plan(model, params_eff)
saveat_eff = np.linspace(0, 50e-6, 51)

for mode in ["full", "populations", "selected"]:
    kwargs = {}
    if mode == "selected":
        kwargs["output_indices"] = [(0, 0), (1, 1), (0, 1)]
    try:
        r = solve_effective_lindblad(plan, rho0_eff, (0, 50e-6),
                                     saveat=saveat_eff, reltol=1e-6, abstol=1e-8,
                                     output=mode, **kwargs)
        print(f"  {mode:20s}: OK")
    except Exception as e:
        print(f"  {mode:20s}: FAILED - {e}")

# weighted_integral - integrate trace (should = t_span length for normalized state)
diag_indices = [(i * n_eff + i, 1.0) for i in range(n_eff)]
try:
    r = solve_effective_lindblad(plan, rho0_eff, (0, 50e-6),
                                 saveat=saveat_eff, reltol=1e-6, abstol=1e-8,
                                 output="weighted_integral",
                                 integral_weights=diag_indices)
    expected_integral = 50e-6  # integral of trace=1 over 50us
    print(f"  {'weighted_integral':20s}: OK (integral={r.rho[0]:.6e}, expected={expected_integral:.6e})")
except Exception as e:
    print(f"  {'weighted_integral':20s}: FAILED - {e}")

# final only
for mode in ["full", "populations"]:
    try:
        r = solve_effective_lindblad(plan, rho0_eff, (0, 50e-6),
                                     saveat=saveat_eff, reltol=1e-6, abstol=1e-8,
                                     output=mode, output_when="final")
        print(f"  {mode + '/final':20s}: OK")
    except Exception as e:
        print(f"  {mode + '/final':20s}: FAILED - {e}")

print("\nDone.")
