import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import sympy as smp

from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    solve_static_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.effective_hamiltonian.rust_plan import (
    prepare_effective_lindblad_rust_plan,
    solve_effective_lindblad,
)
from centrex_tlf.lindblad.parameters import LindbladParameters, Parameter

transition = transitions.OpticalTransition(
    transitions.OpticalTransitionType.R, 0, 1/2, 1
)
polarization = couplings.polarization_Z

print("Preparing model...")
t0 = time.perf_counter()
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
    transition=transition,
    optical_polarization=polarization,
)
prep_time = time.perf_counter() - t0
print(f"  Model prepared in {prep_time:.2f}s")
print(f"  n_states: {model.n_effective_states}")
print(f"  n_grid: {len(model.field_points)}")

Gamma = 2 * np.pi * 1.56e6
static_Ez = 10.0
rabi = Gamma
detuning = 0.0

rho0 = default_effective_density_matrix(model)
t_span = (0.0, 50e-6)
t_eval = np.linspace(t_span[0], t_span[1], 201)

print("\n=== Python scipy static solve ===")
t0 = time.perf_counter()
sol_py = solve_static_lindblad_safe_compact_interpolated_model(
    model,
    electric_field=static_Ez,
    rho0=rho0,
    t_span=t_span,
    rabi_rate=rabi,
    detuning=detuning,
    t_eval=t_eval,
)
py_time = time.perf_counter() - t0
print(f"  Time: {py_time*1000:.1f} ms")
rho_py = sol_py.y.T.reshape(-1, model.n_effective_states, model.n_effective_states)
pops_py = np.real(np.diagonal(rho_py, axis1=1, axis2=2))
print(f"  Trace at end: {pops_py[-1].sum():.6f}")

print("\n=== Rust effective Lindblad solve ===")

params = LindbladParameters({
    smp.Symbol("Ez"): static_Ez,
    smp.Symbol("\u03a90"): rabi,
    smp.Symbol("\u03b40"): detuning,
})

print("  Preparing Rust plan...")
t0 = time.perf_counter()
try:
    rust_plan = prepare_effective_lindblad_rust_plan(model, params)
    plan_time = time.perf_counter() - t0
    print(f"  Plan prepared in {plan_time*1000:.1f} ms")

    print("  Solving (short test first)...")
    t0 = time.perf_counter()
    result = solve_effective_lindblad(
        rust_plan,
        rho0,
        (0.0, 1e-6),
        saveat=np.array([1e-6]),
        reltol=1e-4,
        abstol=1e-6,
        dt=1e-10,
    )
    short_time = time.perf_counter() - t0
    print(f"  Short solve time: {short_time*1000:.1f} ms")
    print(f"  Short solve trace: {result.populations()[-1].sum():.6f}")
    print(f"  Short solve pops: {result.populations()[-1]}")

    print("  Solving (full)...")
    t0 = time.perf_counter()
    result = solve_effective_lindblad(
        rust_plan,
        rho0,
        t_span,
        saveat=t_eval,
        reltol=1e-4,
        abstol=1e-6,
        dt=1e-10,
    )
    rust_time = time.perf_counter() - t0
    print(f"  Time: {rust_time*1000:.1f} ms")

    pops_rust = result.populations()
    print(f"  Trace at end: {pops_rust[-1].sum():.6f}")

    max_pop_diff = np.max(np.abs(pops_rust - pops_py))
    print(f"\n  Max population difference vs scipy: {max_pop_diff:.2e}")
    print(f"  Speedup: {py_time/rust_time:.1f}x")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
