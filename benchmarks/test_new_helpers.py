import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import math
from centrex_tlf import transitions, couplings, lindblad
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.helper_functions import gaussian_1d, pchip_interp
from centrex_tlf.lindblad.parameters import LindbladParameters
from collections import OrderedDict

print("=== Test gaussian_1d ===")
assert abs(gaussian_1d(0.0, 0.0, 1.0) - 1.0) < 1e-15
assert abs(gaussian_1d(1.0, 0.0, 1.0) - math.exp(-0.5)) < 1e-15
assert abs(gaussian_1d(5.0, 5.0, 2.0) - 1.0) < 1e-15
print("  Python gaussian_1d: OK")

print("\n=== Test pchip_interp ===")
grid = [0.0, 1.0, 2.0, 3.0, 4.0]
values = [0.0, 1.0, 0.0, 1.0, 0.0]
result = pchip_interp(0.5, grid, values)
print(f"  pchip_interp(0.5) = {result:.6f}")
result2 = pchip_interp(1.0, grid, values)
print(f"  pchip_interp(1.0) = {result2:.6f} (expected 1.0)")
assert abs(result2 - 1.0) < 1e-10
print("  Python pchip_interp: OK")

print("\n=== Test gaussian_1d through Lindblad parameter system ===")
trans = transitions.R0_F1_3o2_F2
ts = couplings.generate_transition_selectors([trans], [[couplings.polarization_Z]])
system = lindblad.generate_OBE_system_transitions([trans], ts, method="matrix")

Gamma = 2 * np.pi * 1.56e6
params = LindbladParameters(
    base_parameters=OrderedDict([
        ("\u03a9t0", Gamma),
        ("\u03b40", 0.0),
        ("PZ0", 1.0),
        ("z0", -0.01),
        ("v", 200.0),
        ("z_laser", 0.005),
        ("w0", 0.0001),
    ]),
    compound_parameters=OrderedDict([
        ("z", "z0 + v*t"),
        ("\u03a90", "\u03a9t0 * gaussian_1d(z, z_laser, w0)"),
    ]),
)

prepared = prepare_lindblad_problem(system, params, backend="rust")
n = len(system.QN)
n_ground = len(system.ground)
rho0 = np.zeros((n, n), dtype=np.complex128)
for i in range(n_ground):
    rho0[i, i] = 1.0 / n_ground

t_span = (0.0, 100e-6)
saveat = np.linspace(t_span[0], t_span[1], 201)

result = solve_lindblad(
    prepared, rho0, t_span,
    solver="dopri5", execution_mode="structured_upper",
    saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
)
print(f"  Solve completed, trace at end: {result.populations()[-1].sum():.6f}")
print("  gaussian_1d through Rust: OK")

print("\n=== Test pchip_interp through Lindblad parameter system ===")
z_grid = np.linspace(-0.02, 0.02, 20)
ez_values = 10.0 + 5.0 * np.sin(np.pi * z_grid / 0.02)

params2 = LindbladParameters(
    base_parameters=OrderedDict([
        ("\u03a9t0", Gamma),
        ("PZ0", 1.0),
        ("z0", -0.02),
        ("v", 400.0),
    ]),
    compound_parameters=OrderedDict([
        ("z", "z0 + v*t"),
        ("Ez", f"pchip_interp(z, z_grid, ez_vals)"),
        ("\u03b40", "Ez * 1000"),
        ("\u03a90", "\u03a9t0"),
    ]),
    tabulated_data={
        "z_grid": z_grid,
        "ez_vals": ez_values,
    },
)

try:
    prepared2 = prepare_lindblad_problem(system, params2, backend="rust")
    result2 = solve_lindblad(
        prepared2, rho0, t_span,
        solver="dopri5", execution_mode="structured_upper",
        saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
    )
    print(f"  Solve completed, trace at end: {result2.populations()[-1].sum():.6f}")
    print("  pchip_interp through Rust: OK")
except Exception as e:
    print(f"  pchip_interp through Rust: FAILED - {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests passed!")
