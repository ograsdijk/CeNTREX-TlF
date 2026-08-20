"""Benchmark the ``representation="auto"`` Hamiltonian-lowering heuristic.

``centrex_tlf/lindblad/ir.py::lower_hamiltonian_upper_triangle`` chooses between
the ``entrywise`` and ``decomposed`` lowerings with an undocumented cost model::

    entrywise_cost  = len(temps) + len(entries)
    decomposed_cost = len(coefficients) + 0.15 * basis_term_count + (1 if static)

The audit item asks whether those constants are benchmark-justified. This script
measures, per system:

* the two cost-model scores and which branch ``auto`` therefore selects,
* wall-clock lowering time for each representation,
* ``prepare_lindblad_problem`` time for each representation,
* per-RHS-call cost for each representation,

so the heuristic's choice can be compared against the representation that is
actually faster.

Systems mirror ``benchmarks/bench_setup.py``: System A is the 65-state R(0)
system, System B the 154-state r2-style R(2) system with opposite-parity levels
retained.

Usage::

    uv run python benchmarks/bench_hamiltonian_representation.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from centrex_tlf import couplings, hamiltonian, lindblad, transitions
from centrex_tlf.lindblad.ir import (
    _lower_hamiltonian_upper_triangle_decomposed,
    _lower_hamiltonian_upper_triangle_entrywise,
)
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem

RESULTS_DIR = Path(__file__).parent / "hamiltonian_representation_results"
REPEATS = 5


def make_parameters(system: OBESystem) -> LindbladParameters:
    """Mirrors benchmarks/bench_setup.py:make_parameters."""
    Gamma = hamiltonian.Γ
    values: dict[str, float] = {str(s): 0.0 for s in system.H_symbolic.free_symbols}
    params = LindbladParameters()
    for s in system.coupling_symbols:
        values[str(s)] = Gamma
    for group in system.polarization_symbols:
        for s in group if isinstance(group, (list, tuple)) else [group]:
            values[str(s)] = 1.0
    for name, value in values.items():
        params.real(name, value)
    return params


def build_system(trans: Any, retain_opposite_parity: bool) -> OBESystem:
    selectors = couplings.generate_transition_selectors([trans], [[couplings.polarization_Z]])
    return lindblad.generate_OBE_system_transitions(
        [trans],
        selectors,
        retain_opposite_parity_levels=retain_opposite_parity,
    )


def cost_scores(system: OBESystem, params: LindbladParameters) -> dict[str, Any]:
    """Recompute the ir.py cost model without invoking the "auto" branch."""
    slot_index = params.slot_index_by_name
    entrywise = _lower_hamiltonian_upper_triangle_entrywise(
        system.H_symbolic, slot_index, tuple_value_names=set()
    )
    decomposed = _lower_hamiltonian_upper_triangle_decomposed(
        system.H_symbolic, slot_index, tuple_value_names=set()
    )
    diagnostics = decomposed.get("diagnostics", {})
    basis_term_count = diagnostics.get("basis_term_count", 0)
    has_static = bool(np.any(np.abs(decomposed["static_matrix"]) > 0))

    entrywise_cost = len(entrywise["temps"]) + len(entrywise["entries"])
    decomposed_cost = len(decomposed["coefficients"]) + 0.15 * basis_term_count
    if has_static:
        decomposed_cost += 1

    return {
        "entrywise_temps": len(entrywise["temps"]),
        "entrywise_entries": len(entrywise["entries"]),
        "entrywise_cost": entrywise_cost,
        "decomposed_coefficients": len(decomposed["coefficients"]),
        "basis_term_count": basis_term_count,
        "has_static": has_static,
        "decomposed_cost": decomposed_cost,
        "auto_selects": "decomposed" if decomposed_cost < entrywise_cost else "entrywise",
    }


def time_lowering(system: OBESystem, params: LindbladParameters, representation: str) -> float:
    slot_index = params.slot_index_by_name
    fn = (
        _lower_hamiltonian_upper_triangle_entrywise
        if representation == "entrywise"
        else _lower_hamiltonian_upper_triangle_decomposed
    )
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(system.H_symbolic, slot_index, tuple_value_names=set())
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_prepare(system: OBESystem, params: LindbladParameters, representation: str) -> float:
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        prepare_lindblad_problem(
            system,
            params,
            backend="rust",
            hamiltonian_representation=representation,
        )
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_rhs(system: OBESystem, params: LindbladParameters, representation: str) -> dict[str, Any]:
    """Per-RHS-call cost, via a short solve.

    ``expanded_sparse`` requires a decomposed plan (rhs.rs errors otherwise), so
    the entrywise measurement necessarily uses a different execution mode. Both
    modes are reported for decomposed so the comparison is not confounded.
    """
    prepared = prepare_lindblad_problem(
        system, params, backend="rust", hamiltonian_representation=representation
    )
    n = len(system.QN)
    rho0 = np.zeros((n, n), dtype=np.complex128)
    rho0[0, 0] = 1.0

    out: dict[str, Any] = {}
    modes = ["structured"] if representation == "entrywise" else ["structured", "expanded_sparse"]
    for mode in modes:
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = lindblad.solve_lindblad(
                prepared,
                rho0,
                (0.0, 1e-5),
                solver="dopri5",
                execution_mode=mode,
                output="full",
            )
            samples.append(time.perf_counter() - t0)
        stats = getattr(result, "solver_stats", {}) or {}
        rhs_calls = stats.get("rhs_calls") or stats.get("function_evaluations")
        solve_seconds = statistics.median(samples)
        out[mode] = {
            "solve_seconds": solve_seconds,
            "rhs_calls": rhs_calls,
            "us_per_rhs": (solve_seconds / rhs_calls * 1e6) if rhs_calls else None,
        }
    return out


def run_system(name: str, trans: Any, retain_opposite_parity: bool) -> dict[str, Any]:
    print(f"\n{'=' * 70}\nSystem {name}: {trans.name}\n{'=' * 70}")
    system = build_system(trans, retain_opposite_parity)
    params = make_parameters(system)
    n_states = len(system.QN)
    print(f"  n_states = {n_states}")

    scores = cost_scores(system, params)
    print(
        f"  cost model: entrywise={scores['entrywise_cost']:.1f} "
        f"decomposed={scores['decomposed_cost']:.1f} -> auto picks {scores['auto_selects']}"
    )

    record: dict[str, Any] = {
        "system": name,
        "transition": trans.name,
        "n_states": n_states,
        **scores,
        "lowering_seconds": {},
        "prepare_seconds": {},
        "rhs": {},
    }
    for representation in ("entrywise", "decomposed"):
        lowering = time_lowering(system, params, representation)
        prepare = time_prepare(system, params, representation)
        record["lowering_seconds"][representation] = lowering
        record["prepare_seconds"][representation] = prepare
        print(f"  {representation:11s}: lower={lowering * 1e3:8.2f} ms  prepare={prepare * 1e3:8.2f} ms")
        try:
            record["rhs"][representation] = time_rhs(system, params, representation)
            for mode, data in record["rhs"][representation].items():
                us = data["us_per_rhs"]
                us_str = f"{us:.2f} us/rhs" if us else "n/a"
                print(
                    f"      solve[{mode}]: {data['solve_seconds'] * 1e3:8.2f} ms  "
                    f"calls={data['rhs_calls']}  {us_str}"
                )
        except Exception as exc:  # pragma: no cover - diagnostic path
            record["rhs"][representation] = {"error": str(exc)}
            print(f"      solve failed: {exc}")
    return record


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    records = [
        run_system("A", transitions.R0_F1_3o2_F2, False),
        run_system("B", transitions.R2_F1_7o2_F3, True),
    ]
    out = RESULTS_DIR / "results.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
