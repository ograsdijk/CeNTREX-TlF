"""
Benchmark script for OBE (Optical Bloch Equation) Lindblad solvers.

Sets up a simple R(0) F1'=3/2 F'=2 transition system and benchmarks
various solver/execution_mode combinations.
"""

import time
import numpy as np

from centrex_tlf import couplings, lindblad, transitions
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.parameters import LindbladParameters


def setup_system():
    trans = transitions.R0_F1_3o2_F2

    transition_selectors = couplings.generate_transition_selectors(
        [trans], [[couplings.polarization_Z]]
    )

    system = lindblad.generate_OBE_system_transitions(
        [trans], transition_selectors, method="matrix"
    )
    return system


def make_parameters(system):
    Gamma = 2 * np.pi * 1.56e6
    values = {str(s): 0.0 for s in system.H_symbolic.free_symbols}
    parameters = LindbladParameters()
    for s in system.coupling_symbols:
        values[str(s)] = Gamma
    for group in system.polarization_symbols:
        for s in group if isinstance(group, (list, tuple)) else [group]:
            values[str(s)] = 1.0
    for name, value in values.items():
        parameters.real(name, value)
    return parameters


def make_initial_state(system):
    n_ground = len(system.ground)
    n = len(system.QN)
    rho0 = np.zeros((n, n), dtype=np.complex128)
    for i in range(n_ground):
        rho0[i, i] = 1.0 / n_ground
    return rho0


def benchmark_solve(prepared, rho0, t_span, saveat, *, solver, execution_mode, n_runs=5, **kwargs):
    times_list = []
    result = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = solve_lindblad(
            prepared,
            rho0,
            t_span,
            solver=solver,
            execution_mode=execution_mode,
            saveat=saveat,
            dt=1e-10,
            reltol=1e-7,
            abstol=1e-9,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
    return times_list, result


def run_benchmarks():
    print("=" * 70)
    print("OBE Benchmark: R(0) F1'=3/2 F'=2")
    print("=" * 70)

    print("\nSetting up system...")
    t0 = time.perf_counter()
    system = setup_system()
    setup_time = time.perf_counter() - t0
    n = len(system.QN)
    print(f"  States: {n} ({n}x{n} density matrix)")
    print(f"  Setup time: {setup_time:.2f}s")

    parameters = make_parameters(system)
    rho0 = make_initial_state(system)

    t_span = (0.0, 10e-6)
    saveat = np.linspace(t_span[0], t_span[1], 201)

    print("\nPreparing Lindblad problem (rust)...")
    t0 = time.perf_counter()
    prepared_rust = prepare_lindblad_problem(
        system,
        parameters,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    prep_time = time.perf_counter() - t0
    print(f"  Preparation time: {prep_time:.3f}s")

    print("\nPreparing Lindblad problem (python)...")
    t0 = time.perf_counter()
    prepared_python = prepare_lindblad_problem(system, parameters, backend="python")
    prep_time_py = time.perf_counter() - t0
    print(f"  Preparation time: {prep_time_py:.3f}s")

    n_runs = 5
    collect_solve_stats = True
    configs = [
        ("dopri5", "structured", "rust", prepared_rust),
        ("dopri5", "structured_upper", "rust", prepared_rust),
        ("dopri5", "expanded_sparse", "rust", prepared_rust),
        ("dopri5_fast", "structured_upper", "rust", prepared_rust),
        ("dopri5_fast", "expanded_sparse", "rust", prepared_rust),
        ("tsit5_fast", "structured_upper", "rust", prepared_rust),
        ("tsit5_fast", "expanded_sparse", "rust", prepared_rust),
        ("dopri5", "reference", "rust", prepared_rust),
        ("scipy", "structured", "rust", prepared_rust),
        ("scipy", "expanded_sparse", "rust", prepared_rust),
        ("scipy_bdf", "structured", "rust", prepared_rust),
        ("scipy_bdf", "structured_upper", "rust", prepared_rust),
        ("scipy_bdf", "expanded_sparse", "rust", prepared_rust),
        ("scipy_radau", "structured", "rust", prepared_rust),
        ("scipy_radau", "structured_upper", "rust", prepared_rust),
        ("scipy_radau", "expanded_sparse", "rust", prepared_rust),
        ("explicit", "structured", "python", prepared_python),
    ]

    print(f"\nRunning benchmarks ({n_runs} runs each)...")
    print("-" * 70)
    print(f"{'Solver':<16} {'Exec Mode':<20} {'Backend':<8} {'Median (ms)':>12} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 70)

    results = {}
    for solver, exec_mode, backend, prepared in configs:
        label = f"{solver}/{exec_mode}/{backend}"
        extra_kwargs = {}
        if backend == "python":
            extra_kwargs["backend"] = "python"
        try:
            times_list, result = benchmark_solve(
                prepared,
                rho0,
                t_span,
                saveat,
                solver=solver,
                execution_mode=exec_mode,
                n_runs=n_runs,
                **extra_kwargs,
            )
            ms = [t * 1000 for t in times_list]
            median_ms = np.median(ms)
            min_ms = min(ms)
            max_ms = max(ms)
            print(f"{solver:<16} {exec_mode:<20} {backend:<8} {median_ms:>12.2f} {min_ms:>10.2f} {max_ms:>10.2f}")
            solver_stats = None
            if collect_solve_stats and backend == "rust" and solver in {
                "dopri5",
                "dopri5_fast",
                "tsit5_fast",
            }:
                profiled = solve_lindblad(
                    prepared,
                    rho0,
                    t_span,
                    solver=solver,
                    execution_mode=exec_mode,
                    saveat=saveat,
                    dt=1e-10,
                    reltol=1e-7,
                    abstol=1e-9,
                    collect_stats=True,
                )
                solver_stats = profiled.solver_stats
            results[label] = {
                "times_ms": ms,
                "median_ms": median_ms,
                "populations_final": result.populations()[-1],
                "solver_stats": solver_stats,
            }
        except Exception as e:
            print(f"{solver:<16} {exec_mode:<20} {backend:<8} {'ERROR':>12}   {str(e)[:30]}")

    print("-" * 70)

    ref_key = "dopri5/structured/rust"
    if ref_key in results:
        ref_pops = results[ref_key]["populations_final"]
        n_ground = len(system.ground)
        print(f"\nFinal state (reference: {ref_key}):")
        print(f"  Ground pop total: {ref_pops[:n_ground].sum():.6f}")
        print(f"  Excited pop total: {ref_pops[n_ground:].sum():.6f}")
        print(f"  Trace: {ref_pops.sum():.6f}")

        print("\nPopulation agreement (max abs diff vs reference):")
        for label, data in results.items():
            if label == ref_key:
                continue
            diff = np.max(np.abs(data["populations_final"] - ref_pops))
            print(f"  {label}: {diff:.2e}")

    if any(data.get("solver_stats") for data in results.values()):
        print("\nRust solve diagnostics from one extra profiled run:")
        print(
            f"{'Config':40s} {'RHS calls':>10s} {'Acc':>8s} {'Rej':>8s} "
            f"{'RHS ms':>10s} {'Non-RHS ms':>12s}"
        )
        print("-" * 94)
        for label, data in results.items():
            stats = data.get("solver_stats")
            if not stats:
                continue
            print(
                f"  {label:38s} {stats['rhs_calls']:10d} {stats['accepted_steps']:8d} "
                f"{stats['rejected_steps']:8d} {stats['rhs_seconds']*1000:10.1f} "
                f"{stats['non_rhs_seconds']*1000:12.1f}"
            )

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    run_benchmarks()
