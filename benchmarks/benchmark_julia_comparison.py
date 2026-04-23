"""
Benchmark comparison: Rust vs Julia vs scipy solvers for R(0) F1=3/2 F=2 OBE system.

Julia benchmarks use BenchmarkTools @btime for pure solve-only timing.
Rust benchmarks use Python time.perf_counter on the solve_lindblad call.
Scipy benchmarks use Python time.perf_counter on the solve_lindblad call.

Requires:
  - centrex_tlf (with Rust extension built)
  - centrex_tlf_julia_extension (with Julia installed)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
from centrex_tlf import transitions, couplings, lindblad, hamiltonian, states, utils
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.parameters import LindbladParameters

trans = transitions.R0_F1_3o2_F2

transition_selectors = couplings.generate_transition_selectors(
    [trans], [[couplings.polarization_Z]]
)

system = lindblad.generate_OBE_system_transitions(
    [trans], transition_selectors, method="matrix"
)

n = len(system.QN)
n_ground = len(system.ground)
rho0 = np.zeros((n, n), dtype=np.complex128)
for i in range(n_ground):
    rho0[i, i] = 1.0 / n_ground

Gamma = 2 * np.pi * 1.56e6
t_span = (0.0, 10e-6)
saveat = np.linspace(t_span[0], t_span[1], 201)
n_runs = 5
collect_solve_stats = True

print("=" * 80)
print(f"OBE Benchmark Comparison: R(0) F1'=3/2 F'=2")
print(f"  States: {n} ({n_ground} ground + {n - n_ground} excited)")
print(f"  t_span: {t_span[0]*1e6:.0f} - {t_span[1]*1e6:.0f} us")
print(f"  saveat: {len(saveat)} points")
print("=" * 80)

# ============================================================================
# Rust benchmarks
# ============================================================================

print("\n--- Rust Solvers ---")

params_rust = LindbladParameters()
param_values = {str(s): 0.0 for s in system.H_symbolic.free_symbols}
for s in system.coupling_symbols:
    param_values[str(s)] = Gamma
for group in system.polarization_symbols:
    for s in group if isinstance(group, (list, tuple)) else [group]:
        param_values[str(s)] = 1.0
for name, value in param_values.items():
    params_rust.real(name, value)

prepared_rust = prepare_lindblad_problem(
    system,
    params_rust,
    backend="rust",
    hamiltonian_representation="decomposed",
)

rust_configs = [
    ("dopri5", "structured"),
    ("dopri5", "structured_upper"),
    ("dopri5", "expanded_sparse"),
    ("dopri5_fast", "structured_upper"),
    ("dopri5_fast", "expanded_sparse"),
    ("tsit5_fast", "structured_upper"),
    ("tsit5_fast", "expanded_sparse"),
    ("scipy", "structured"),
    ("scipy", "structured_upper"),
    ("scipy", "expanded_sparse"),
    ("scipy_bdf", "structured"),
    ("scipy_bdf", "structured_upper"),
    ("scipy_bdf", "expanded_sparse"),
]

rust_results = {}
for solver, exec_mode in rust_configs:
    label = f"{solver}/{exec_mode}"
    times_list = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = solve_lindblad(
            prepared_rust, rho0, t_span,
            solver=solver, execution_mode=exec_mode,
            saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
        )
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
    ms = [t * 1000 for t in times_list]
    median_ms = np.median(ms)
    print(f"  {label:40s} {median_ms:8.1f} ms")
    solver_stats = None
    if collect_solve_stats and solver in {"dopri5", "dopri5_fast", "tsit5_fast"}:
        profiled_result = solve_lindblad(
            prepared_rust, rho0, t_span,
            solver=solver, execution_mode=exec_mode,
            saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
            collect_stats=True,
        )
        solver_stats = profiled_result.solver_stats
    rust_results[label] = {
        "median_ms": median_ms,
        "pops_final": result.populations()[-1],
        "solver_stats": solver_stats,
    }

# ============================================================================
# Julia benchmarks
# ============================================================================

print("\n--- Julia Solvers ---")
print("  Initializing Julia...")

julia_results = {}
try:
    from centrex_tlf_julia_extension import lindblad_julia
    from centrex_tlf_julia_extension.lindblad_julia.utils_julia import jl

    rho_julia = np.zeros((n, n), dtype=np.complex128)
    for i in range(n_ground):
        rho_julia[i, i] = 1.0 / n_ground

    saveat_jl = saveat.tolist()

    julia_packages = ["BenchmarkTools"]
    jl.seval("using Pkg")
    for pkg in julia_packages:
        if not bool(jl.seval(f'isnothing(Base.find_package("{pkg}")) ? false : true')):
            print(f"  Installing Julia package: {pkg}")
            jl.Pkg.add(pkg)
    jl.seval("using BenchmarkTools")
    jl.seval("_saveat_bench = " + f"[{', '.join(str(x) for x in saveat_jl)}]")

    # Julia uses setup_OBE_system_transitions with qn_compact=True -> smaller system
    # Need to use the Julia system's dimensions for rho, not the Rust system's
    def run_julia_method(method_name, obe_system_jl, odepars_jl, solver_list):
        n_jl = len(obe_system_jl.QN)
        n_excited_jl = len(obe_system_jl.excited)
        n_ground_jl = n_jl - n_excited_jl
        rho_jl = np.zeros((n_jl, n_jl), dtype=np.complex128)
        for i in range(n_ground_jl):
            rho_jl[i, i] = 1.0 / n_ground_jl

        problem_jl = lindblad_julia.OBEProblem(odepars_jl, rho_jl, tspan=t_span)

        for jl_method, label_method in solver_list:
            label = f"julia/{jl_method}/{label_method}"

            config_jl = lindblad_julia.OBEProblemConfig(
                method=jl_method,
                saveat=saveat_jl,
                reltol=1e-7,
                abstol=1e-9,
                dt=1e-10,
                save_everystep=False,
            )

            lindblad_julia.setup_problem(odepars_jl, t_span, rho_jl, problem_name="prob")

            # warmup solve
            lindblad_julia.solve_problem(problem_jl, config_jl)

            bench_cmd = f"""
            @belapsed solve(
                $prob,
                {jl_method};
                dt=1e-10,
                abstol=1e-9,
                reltol=1e-7,
                saveat=$_saveat_bench,
                save_everystep=false,
                maxiters=100000,
                save_start=true,
                dense=false,
            )
            """
            print(f"  Benchmarking {label}...", end="", flush=True)
            elapsed_s = float(jl.seval(bench_cmd))
            elapsed_ms = elapsed_s * 1000
            print(f" {elapsed_ms:.1f} ms")
            julia_results[label] = elapsed_ms

    # --- matrix method (65 states, same as Rust) ---
    print("  Setting up Julia OBE system (matrix, 65 states)...")
    obe_system_legacy_matrix = lindblad.setup_OBE_system_transitions(
        [trans], transition_selectors, verbose=False, qn_compact=False,
    )
    odepars_matrix = lindblad_julia.odeParameters(
        **{"\u03a90": Gamma, "\u03b40": 0.0, "PZ0": 1.0}
    )
    obe_julia_matrix = lindblad_julia.setup_OBE_system_julia(
        obe_system_legacy_matrix, transition_selectors, odepars_matrix,
        n_procs=1, method="matrix", verbose=False,
    )
    n_jl = len(obe_julia_matrix.QN)
    print(f"  Julia matrix system: {n_jl} states")
    run_julia_method("matrix", obe_julia_matrix, odepars_matrix, [
        ("Tsit5()", "matrix"),
        ("Vern7()", "matrix"),
    ])

    # --- expanded method (65 states, same as Rust) ---
    print("  Setting up Julia OBE system (expanded, 65 states)...")
    obe_system_legacy_exp = lindblad.setup_OBE_system_transitions(
        [trans], transition_selectors, verbose=False, qn_compact=False,
    )
    odepars_exp = lindblad_julia.odeParameters(
        **{"\u03a90": Gamma, "\u03b40": 0.0, "PZ0": 1.0}
    )
    obe_julia_exp = lindblad_julia.setup_OBE_system_julia(
        obe_system_legacy_exp, transition_selectors, odepars_exp,
        n_procs=1, method="expanded", verbose=False,
    )
    n_jl_exp = len(obe_julia_exp.QN)
    print(f"  Julia expanded system: {n_jl_exp} states")
    run_julia_method("expanded", obe_julia_exp, odepars_exp, [
        ("Tsit5()", "expanded"),
        ("Vern7()", "expanded"),
    ])

except ImportError as e:
    print(f"  Julia extension not available: {e}")
    print("  Skipping Julia benchmarks.")
except Exception as e:
    print(f"  Julia benchmark failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary (Rust solvers)")
print("=" * 80)
print(f"{'Config':40s} {'Median (ms)':>12s}")
print("-" * 54)
ref_key = "dopri5/structured_upper"
for label, data in rust_results.items():
    print(f"  {label:38s} {data['median_ms']:10.1f}")

if ref_key in rust_results:
    ref_pops = rust_results[ref_key]["pops_final"]
    print(f"\nPopulation agreement vs {ref_key}:")
    for label, data in rust_results.items():
        if label == ref_key:
            continue
        diff = np.max(np.abs(data["pops_final"] - ref_pops))
        print(f"  {label:38s} {diff:.2e}")

if any(data.get("solver_stats") for data in rust_results.values()):
    print("\nRust solve diagnostics from one extra profiled run:")
    print(
        f"{'Config':40s} {'RHS calls':>10s} {'Acc':>8s} {'Rej':>8s} "
        f"{'RHS ms':>10s} {'Non-RHS ms':>12s}"
    )
    print("-" * 94)
    for label, data in rust_results.items():
        stats = data.get("solver_stats")
        if not stats:
            continue
        print(
            f"  {label:38s} {stats['rhs_calls']:10d} {stats['accepted_steps']:8d} "
            f"{stats['rejected_steps']:8d} {stats['rhs_seconds']*1000:10.1f} "
            f"{stats['non_rhs_seconds']*1000:12.1f}"
        )

if julia_results:
    print(f"\n{'Julia Config':40s} {'Min (ms)':>12s}")
    print("-" * 54)
    for label, ms in julia_results.items():
        print(f"  {label:38s} {ms:10.1f}")
