"""Benchmark for the OBE *setup* path (as opposed to the solve path).

The performance review in ``IMPLEMENTATION_AUDIT.md`` (section "Performance
Review (2026-07-11)") found that building an ``OBESystem`` is the dominant
cost for typical notebook workflows: ~4.4 s for a 65-state R(0) F1'=3/2 F'=2
system vs ~5 ms per solve. There was previously no benchmark isolating the
individual setup stages, which makes it hard to judge the effect of the
planned change to make ``OBESystem.system``/``OBESystem.dissipator`` lazily
built from sparse entrywise constructors.

This script times each stage of the setup path separately, mirroring exactly
what ``lindblad.generate_OBE_system_transitions`` / the private
``centrex_tlf.lindblad.utils_setup._build_obe_system`` helper do internally:

    1. ``hamiltonian.generate_reduced_hamiltonian_transitions(...)``
    2. coupling generation (``utils_setup._generate_couplings``, which calls
       ``couplings.generate_coupling_field``/``generate_coupling_field_automatic``
       per transition selector)
    3. symbolic RWA Hamiltonian (``generate_total_symbolic_hamiltonian``)
    4. ``couplings.collapse_matrices(...)``
    5. symbolic dissipator (``generate_dissipator_term(..., fast=True)``)
    6. symbolic Hamiltonian term (``generate_hamiltonian_term``) -- this is
       the *extra* cost that ``method="expanded"`` pays on top of
       ``method="matrix"``.
    7. ``prepare_lindblad_problem(..., backend="rust",
       hamiltonian_representation="decomposed")``

Wall-clock timing only (``time.perf_counter``) is used throughout -- do NOT
wrap this in cProfile, since profiling sympy-heavy code inflates timings by
~4x and would give a misleading picture of where time actually goes.

Two systems are benchmarked:

  * System A: ``transitions.R0_F1_3o2_F2`` with Z polarization, exactly like
    ``benchmarks/benchmark_obe.py:setup_system`` (~65 states). Each stage is
    run 3 times and the median is reported/stored.
  * System B: an r2-style system, ``transitions.R2_F1_7o2_F3`` (the R(2)
    transition used in the r2 example notebooks, e.g.
    ``examples/lindblad/r2_opposite_parity_retention.ipynb``), with
    ``retain_opposite_parity_levels=True``. Noticeably slower; also run
    3 times per stage, median and min reported.

Since the lazy ``OBESystem.system``/``.dissipator`` refactor, stages 5 and 6
are no longer part of the end-to-end build: they measure the standalone cost
of requesting the symbolic matrices (Julia codegen / visualization path).

Because the Wigner-3j/6j and Clebsch-Gordan helpers used throughout the
Hamiltonian/coupling code are wrapped in ``functools.lru_cache``, timings
for System B (run second) are partially warmed by System A's run -- this is
noted in the generated report rather than worked around, since the
production workflow (build one system per process) sees the same warm-cache
behavior after the first transition of a given kind is processed.

Outputs (following the ``benchmarks/<name>_results/`` convention used by the
other benchmarks in this directory):

  * ``benchmarks/setup_path_results/setup_path_timings.csv``
  * ``benchmarks/setup_path_results/setup_path_report.md``
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Any

from centrex_tlf import couplings, hamiltonian, lindblad, transitions
from centrex_tlf.lindblad import utils_setup
from centrex_tlf.lindblad.generate_hamiltonian import generate_total_symbolic_hamiltonian
from centrex_tlf.lindblad.generate_system_of_equations import (
    generate_density_matrix,
    generate_dissipator_term,
    generate_hamiltonian_term,
)
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem

RESULTS_DIR = Path(__file__).parent / "setup_path_results"

STAGE_NAMES = [
    "1_reduced_hamiltonian",
    "2_couplings",
    "3_symbolic_hamiltonian_rwa",
    "4_collapse_matrices",
    "5_symbolic_dissipator",
    "6_symbolic_hamiltonian_term",
    "7_prepare_lindblad_problem",
]

# Stages that make up "generate_OBE_system_transitions(...)" end to end --
# used for the self-check ratio. Since the lazy `OBESystem.system`/
# `.dissipator` refactor, the end-to-end build no longer constructs the
# symbolic dissipator (stage 5) or Hamiltonian term (stage 6); those are
# built on first attribute access instead and are benchmarked here as the
# standalone cost of requesting the symbolic matrices.
END_TO_END_STAGES = [
    "1_reduced_hamiltonian",
    "2_couplings",
    "3_symbolic_hamiltonian_rwa",
    "4_collapse_matrices",
]


def make_parameters(system: OBESystem) -> LindbladParameters:
    """Mirrors benchmarks/benchmark_obe.py:make_parameters."""
    Gamma = hamiltonian.Γ
    values: dict[str, float] = {str(s): 0.0 for s in system.H_symbolic.free_symbols}
    parameters = LindbladParameters()
    for s in system.coupling_symbols:
        values[str(s)] = Gamma
    for group in system.polarization_symbols:
        for s in group if isinstance(group, (list, tuple)) else [group]:
            values[str(s)] = 1.0
    for name, value in values.items():
        parameters.real(name, value)
    return parameters


def time_stages(
    trans: Any,
    transition_selectors: list[Any],
    retain_opposite_parity: bool,
) -> tuple[dict[str, float], int]:
    """Run one full setup pass, timing each stage separately.

    Mirrors the call sequence in
    ``centrex_tlf.lindblad.utils_setup.generate_OBE_system_transitions`` /
    ``_build_obe_system`` exactly (qn_compact=None, decay_channels=None,
    normalize_pol=False throughout, matching benchmark_obe.py:setup_system).
    """
    timings: dict[str, float] = {}

    # Stage 1: reduced Hamiltonian.
    t0 = time.perf_counter()
    H_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        transitions=[trans],
        retain_opposite_parity_levels=retain_opposite_parity,
    )
    timings["1_reduced_hamiltonian"] = time.perf_counter() - t0

    if H_reduced.QN_basis is None:
        raise TypeError("H_reduced.QN_basis is None")

    ground_states = H_reduced.X_states
    excited_states = H_reduced.B_states
    QN = H_reduced.QN
    H_int = H_reduced.H_int
    V_ref_int = H_reduced.V_ref_int

    # Not separately timed: cheap selector bookkeeping done by
    # generate_OBE_system_transitions before _build_obe_system is called.
    _transition_selectors = (
        utils_setup._retain_opposite_parity_transition_selectors(
            [trans], transition_selectors
        )
        if retain_opposite_parity
        else transition_selectors
    )

    # Stage 2: coupling generation.
    t0 = time.perf_counter()
    couplings_list = utils_setup._generate_couplings(
        _transition_selectors,
        H_reduced.QN_basis,
        H_int,
        QN,
        V_ref_int,
        normalize_pol=False,
    )
    timings["2_couplings"] = time.perf_counter() - t0

    # Stage 3: symbolic RWA Hamiltonian.
    t0 = time.perf_counter()
    H_symbolic = generate_total_symbolic_hamiltonian(
        QN, H_int, couplings_list, _transition_selectors
    )
    timings["3_symbolic_hamiltonian_rwa"] = time.perf_counter() - t0

    # Stage 4: collapse matrices.
    t0 = time.perf_counter()
    C_array = couplings.collapse_matrices(
        QN, ground_states, excited_states, decay_rate=hamiltonian.Γ, qn_compact=None
    )
    timings["4_collapse_matrices"] = time.perf_counter() - t0

    density_matrix = generate_density_matrix(H_symbolic.shape[0])

    # Stage 5: symbolic dissipator.
    t0 = time.perf_counter()
    dissipator = generate_dissipator_term(C_array, density_matrix, fast=True)
    timings["5_symbolic_dissipator"] = time.perf_counter() - t0

    # Stage 6: symbolic Hamiltonian term -- the extra cost of method="expanded".
    t0 = time.perf_counter()
    hamiltonian_term = generate_hamiltonian_term(H_symbolic, density_matrix)
    timings["6_symbolic_hamiltonian_term"] = time.perf_counter() - t0

    obe_system = OBESystem(
        ground=ground_states,
        excited=excited_states,
        QN=QN,
        H_int=H_int,
        V_ref_int=V_ref_int,
        couplings=couplings_list,
        H_symbolic=H_symbolic,
        C_array=C_array,
        system=hamiltonian_term + dissipator,
        coupling_symbols=[ts.Ω for ts in _transition_selectors],
        polarization_symbols=[ts.polarization_symbols for ts in _transition_selectors],
        dissipator=dissipator,
    )
    parameters = make_parameters(obe_system)

    # Stage 7: prepare_lindblad_problem (rust backend, decomposed representation).
    t0 = time.perf_counter()
    prepare_lindblad_problem(
        obe_system,
        parameters,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    timings["7_prepare_lindblad_problem"] = time.perf_counter() - t0

    return timings, len(QN)


def run_system(
    name: str,
    trans: Any,
    retain_opposite_parity: bool,
    repeats: int,
    rows: list[dict[str, Any]],
) -> tuple[int, float, float]:
    print(f"\n{'=' * 70}")
    print(f"System {name}: {trans.name} (retain_opposite_parity_levels={retain_opposite_parity})")
    print(f"{'=' * 70}")

    transition_selectors = couplings.generate_transition_selectors(
        [trans], [[couplings.polarization_Z]]
    )

    n_states = -1
    for run_index in range(repeats):
        timings, n_states = time_stages(trans, transition_selectors, retain_opposite_parity)
        for stage, seconds in timings.items():
            rows.append(
                {
                    "system": name,
                    "stage": stage,
                    "seconds": seconds,
                    "n_states": n_states,
                    "run_index": run_index,
                }
            )
        print(
            f"  run {run_index}: "
            + ", ".join(f"{stage}={timings[stage]:.4f}s" for stage in STAGE_NAMES)
        )

    stage_sum = sum(
        statistics.median(
            [r["seconds"] for r in rows if r["system"] == name and r["stage"] == stage]
        )
        for stage in END_TO_END_STAGES
    )

    # Self-check: end-to-end calls using the public API. Since the lazy
    # symbolic refactor this covers stages 1-4 only (system/dissipator are
    # built on first attribute access, not during the build; `method` is a
    # deprecated no-op). Same repeat count as the stage timings so
    # median/min are comparable.
    end_to_end_samples: list[float] = []
    for run_index in range(repeats):
        t0 = time.perf_counter()
        lindblad.generate_OBE_system_transitions(
            [trans],
            transition_selectors,
            method="matrix",
            retain_opposite_parity_levels=retain_opposite_parity,
        )
        end_to_end_samples.append(time.perf_counter() - t0)
        rows.append(
            {
                "system": name,
                "stage": "end_to_end_matrix",
                "seconds": end_to_end_samples[-1],
                "n_states": n_states,
                "run_index": run_index,
            }
        )
    end_to_end_seconds = statistics.median(end_to_end_samples)

    print(f"  sum(stages 1-4, median) = {stage_sum:.4f}s")
    print(f"  end-to-end (method='matrix', median) = {end_to_end_seconds:.4f}s")
    ratio = stage_sum / end_to_end_seconds if end_to_end_seconds else float("nan")
    print(f"  ratio (stages 1-4 / end-to-end) = {ratio:.3f}")

    return n_states, stage_sum, end_to_end_seconds


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["system", "stage", "seconds", "n_states", "run_index"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    summary: dict[tuple[str, str], dict[str, float]] = {}
    systems = sorted({r["system"] for r in rows})
    for system in systems:
        stages = [s for s in STAGE_NAMES if any(r["stage"] == s for r in rows if r["system"] == system)]
        stages += ["end_to_end_matrix"]
        for stage in stages:
            values = [r["seconds"] for r in rows if r["system"] == system and r["stage"] == stage]
            if not values:
                continue
            n_states = next(r["n_states"] for r in rows if r["system"] == system and r["stage"] == stage)
            summary[(system, stage)] = {
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "n": len(values),
                "n_states": n_states,
            }
    return summary


def print_summary_table(summary: dict[tuple[str, str], dict[str, float]]) -> None:
    print("\n" + "=" * 78)
    print("Summary (median/min over repeats)")
    print("=" * 78)
    print(
        f"{'system':<10} {'stage':<28} {'n_states':>9} {'n_runs':>7} "
        f"{'median (s)':>12} {'min (s)':>12}"
    )
    print("-" * 78)
    for (system, stage), stats in summary.items():
        print(
            f"{system:<10} {stage:<28} {stats['n_states']:>9} {stats['n']:>7} "
            f"{stats['median_seconds']:>12.4f} {stats['min_seconds']:>12.4f}"
        )
    print("-" * 78)


def write_report(
    summary: dict[tuple[str, str], dict[str, float]],
    self_check: dict[str, tuple[int, float, float]],
    path: Path,
) -> None:
    lines = []
    lines.append("# Setup-Path Benchmark")
    lines.append("")
    lines.append(
        "Wall-clock (`time.perf_counter`) timings for each stage of building an "
        "`OBESystem`, motivated by `IMPLEMENTATION_AUDIT.md` ('Performance Review "
        "(2026-07-11)'), which found setup (not solve) dominates the cost of "
        "typical notebook workflows for time-independent systems."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "- System A: `transitions.R0_F1_3o2_F2`, Z polarization, "
        "`retain_opposite_parity_levels=False` (same system as "
        "`benchmarks/benchmark_obe.py:setup_system`). 3 repeats per stage, median reported."
    )
    lines.append(
        "- System B: `transitions.R2_F1_7o2_F3` (r2-style, used in the r2 example "
        "notebooks), Z polarization, `retain_opposite_parity_levels=True`. 3 repeats "
        "per stage, median and min reported (retaining the opposite-parity excited "
        "levels roughly doubles the excited manifold, giving ~154 states here vs "
        "~65 for System A, and a full build of ~60 s vs System A's ~5.7 s)."
    )
    lines.append(
        "- Stages mirror `centrex_tlf.lindblad.utils_setup._build_obe_system` / "
        "`generate_OBE_system_transitions` exactly (qn_compact=None, "
        "decay_channels=None, normalize_pol=False, `method='expanded'` inputs)."
    )
    lines.append(
        "- Caches (Wigner-3j/6j, Clebsch-Gordan `lru_cache`s) are shared across "
        "the whole process. System A runs first, so System B's stage-1 timing "
        "benefits from an already-warm cache; this understates System B's "
        "\"cold\" cost somewhat, but matches the real workflow of building "
        "several systems in one process/notebook."
    )
    lines.append("")
    lines.append("## Timings")
    lines.append("")
    lines.append("| system | stage | n_states | n_runs | median (s) | min (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for (system, stage), stats in summary.items():
        lines.append(
            f"| {system} | {stage} | {stats['n_states']} | {stats['n']} | "
            f"{stats['median_seconds']:.4f} | {stats['min_seconds']:.4f} |"
        )
    lines.append("")
    lines.append("## Self-check: sum(stages 1-4) vs end-to-end build")
    lines.append("")
    lines.append("| system | n_states | sum(stages 1-4) (s) | end-to-end (s) | ratio |")
    lines.append("| --- | --- | --- | --- | --- |")
    for system, (n_states, stage_sum, e2e) in self_check.items():
        ratio = stage_sum / e2e if e2e else float("nan")
        lines.append(f"| {system} | {n_states} | {stage_sum:.4f} | {e2e:.4f} | {ratio:.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Since the lazy `OBESystem.system`/`.dissipator` refactor, the "
        "end-to-end build covers stages 1-4 plus `OBESystem` construction; the "
        "symbolic dissipator (stage 5) and Hamiltonian term (stage 6) are only "
        "paid on first access to `.dissipator`/`.system` (Julia code "
        "generation, visualization) and are built with sparse entrywise "
        "constructors instead of dense sympy matrix products. Stage 7 "
        "(`prepare_lindblad_problem`) is what the Rust solve path consumes "
        "(`H_symbolic` + `C_array`). The end-to-end self-check (stages 1-4) "
        "validates that the per-stage instrumentation isn't missing or "
        "double-counting work; the ratio slightly exceeds 1 because the "
        "stage-timed pass carries extra bookkeeping the fused call avoids. "
        "Pre-refactor baseline timings are preserved in "
        "`benchmarks/setup_path_results_pre_lazy_baseline/` for comparison."
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    self_check: dict[str, tuple[int, float, float]] = {}

    n_states_a, sum_a, e2e_a = run_system(
        "A_R0F2",
        transitions.R0_F1_3o2_F2,
        retain_opposite_parity=False,
        repeats=3,
        rows=rows,
    )
    self_check["A_R0F2"] = (n_states_a, sum_a, e2e_a)

    n_states_b, sum_b, e2e_b = run_system(
        "B_R2F3_opp_parity",
        transitions.R2_F1_7o2_F3,
        retain_opposite_parity=True,
        repeats=3,
        rows=rows,
    )
    self_check["B_R2F3_opp_parity"] = (n_states_b, sum_b, e2e_b)

    summary = summarize(rows)
    print_summary_table(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, RESULTS_DIR / "setup_path_timings.csv")
    write_report(summary, self_check, RESULTS_DIR / "setup_path_report.md")
    print(f"\nWrote {RESULTS_DIR / 'setup_path_timings.csv'}")
    print(f"Wrote {RESULTS_DIR / 'setup_path_report.md'}")


if __name__ == "__main__":
    main()
