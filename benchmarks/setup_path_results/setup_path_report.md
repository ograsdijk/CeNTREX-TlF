# Setup-Path Benchmark

Wall-clock (`time.perf_counter`) timings for each stage of building an `OBESystem`, motivated by `IMPLEMENTATION_AUDIT.md` ('Performance Review (2026-07-11)'), which found setup (not solve) dominates the cost of typical notebook workflows for time-independent systems.

## Setup

- System A: `transitions.R0_F1_3o2_F2`, Z polarization, `retain_opposite_parity_levels=False` (same system as `benchmarks/benchmark_obe.py:setup_system`). 3 repeats per stage, median reported.
- System B: `transitions.R2_F1_7o2_F3` (r2-style, used in the r2 example notebooks), Z polarization, `retain_opposite_parity_levels=True`. 3 repeats per stage, median and min reported (retaining the opposite-parity excited levels roughly doubles the excited manifold, giving ~154 states here vs ~65 for System A, and a full build of ~60 s vs System A's ~5.7 s).
- Stages mirror `centrex_tlf.lindblad.utils_setup._build_obe_system` / `generate_OBE_system_transitions` exactly (qn_compact=None, decay_channels=None, normalize_pol=False, `method='expanded'` inputs).
- Caches (Wigner-3j/6j, Clebsch-Gordan `lru_cache`s) are shared across the whole process. System A runs first, so System B's stage-1 timing benefits from an already-warm cache; this understates System B's "cold" cost somewhat, but matches the real workflow of building several systems in one process/notebook.

## Timings

| system | stage | n_states | n_runs | median (s) | min (s) |
| --- | --- | --- | --- | --- | --- |
| A_R0F2 | 1_reduced_hamiltonian | 65 | 3 | 0.1640 | 0.1543 |
| A_R0F2 | 2_couplings | 65 | 3 | 0.0046 | 0.0036 |
| A_R0F2 | 3_symbolic_hamiltonian_rwa | 65 | 3 | 0.2437 | 0.2198 |
| A_R0F2 | 4_collapse_matrices | 65 | 3 | 0.0321 | 0.0239 |
| A_R0F2 | 5_symbolic_dissipator | 65 | 3 | 0.0825 | 0.0792 |
| A_R0F2 | 6_symbolic_hamiltonian_term | 65 | 3 | 0.8750 | 0.7925 |
| A_R0F2 | 7_prepare_lindblad_problem | 65 | 3 | 0.0615 | 0.0602 |
| A_R0F2 | end_to_end_matrix | 65 | 3 | 0.4677 | 0.4456 |
| B_R2F3_opp_parity | 1_reduced_hamiltonian | 154 | 3 | 0.7463 | 0.6798 |
| B_R2F3_opp_parity | 2_couplings | 154 | 3 | 0.0212 | 0.0204 |
| B_R2F3_opp_parity | 3_symbolic_hamiltonian_rwa | 154 | 3 | 1.8391 | 1.7988 |
| B_R2F3_opp_parity | 4_collapse_matrices | 154 | 3 | 0.2387 | 0.1952 |
| B_R2F3_opp_parity | 5_symbolic_dissipator | 154 | 3 | 2.1655 | 2.1529 |
| B_R2F3_opp_parity | 6_symbolic_hamiltonian_term | 154 | 3 | 6.2077 | 5.3931 |
| B_R2F3_opp_parity | 7_prepare_lindblad_problem | 154 | 3 | 1.2652 | 0.9897 |
| B_R2F3_opp_parity | end_to_end_matrix | 154 | 3 | 2.8072 | 2.7876 |

## Self-check: sum(stages 1-4) vs end-to-end build

| system | n_states | sum(stages 1-4) (s) | end-to-end (s) | ratio |
| --- | --- | --- | --- | --- |
| A_R0F2 | 65 | 0.4444 | 0.4677 | 0.950 |
| B_R2F3_opp_parity | 154 | 2.8454 | 2.8072 | 1.014 |

## Interpretation

Since the lazy `OBESystem.system`/`.dissipator` refactor, the end-to-end build covers stages 1-4 plus `OBESystem` construction; the symbolic dissipator (stage 5) and Hamiltonian term (stage 6) are only paid on first access to `.dissipator`/`.system` (Julia code generation, visualization) and are built with sparse entrywise constructors instead of dense sympy matrix products. Stage 7 (`prepare_lindblad_problem`) is what the Rust solve path consumes (`H_symbolic` + `C_array`). The end-to-end self-check (stages 1-4) validates that the per-stage instrumentation isn't missing or double-counting work; the ratio slightly exceeds 1 because the stage-timed pass carries extra bookkeeping the fused call avoids. Pre-refactor baseline timings are preserved in `benchmarks/setup_path_results_pre_lazy_baseline/` for comparison.
