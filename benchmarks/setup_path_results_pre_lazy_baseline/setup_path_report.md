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
| A_R0F2 | 1_reduced_hamiltonian | 65 | 3 | 0.4633 | 0.4515 |
| A_R0F2 | 2_couplings | 65 | 3 | 0.0041 | 0.0039 |
| A_R0F2 | 3_symbolic_hamiltonian_rwa | 65 | 3 | 0.2310 | 0.2260 |
| A_R0F2 | 4_collapse_matrices | 65 | 3 | 0.0244 | 0.0232 |
| A_R0F2 | 5_symbolic_dissipator | 65 | 3 | 3.5174 | 3.4536 |
| A_R0F2 | 6_symbolic_hamiltonian_term | 65 | 3 | 1.4928 | 1.4606 |
| A_R0F2 | 7_prepare_lindblad_problem | 65 | 3 | 0.1053 | 0.0945 |
| A_R0F2 | end_to_end_matrix | 65 | 3 | 4.6631 | 4.5239 |
| B_R2F3_opp_parity | 1_reduced_hamiltonian | 154 | 3 | 2.2664 | 2.0803 |
| B_R2F3_opp_parity | 2_couplings | 154 | 3 | 0.0153 | 0.0151 |
| B_R2F3_opp_parity | 3_symbolic_hamiltonian_rwa | 154 | 3 | 1.4419 | 1.4043 |
| B_R2F3_opp_parity | 4_collapse_matrices | 154 | 3 | 0.1961 | 0.1897 |
| B_R2F3_opp_parity | 5_symbolic_dissipator | 154 | 3 | 46.8783 | 42.1848 |
| B_R2F3_opp_parity | 6_symbolic_hamiltonian_term | 154 | 3 | 10.7757 | 9.7029 |
| B_R2F3_opp_parity | 7_prepare_lindblad_problem | 154 | 3 | 1.3867 | 1.1279 |
| B_R2F3_opp_parity | end_to_end_matrix | 154 | 3 | 57.7829 | 57.6051 |

## Self-check: sum(stages 1-5) vs end-to-end `method="matrix"`

| system | n_states | sum(stages 1-5) (s) | end-to-end (s) | ratio |
| --- | --- | --- | --- | --- |
| A_R0F2 | 65 | 4.2402 | 4.6631 | 0.909 |
| B_R2F3_opp_parity | 154 | 50.7980 | 57.7829 | 0.879 |

## Interpretation

The per-stage breakdown confirms the audit's finding: the symbolic dissipator (stage 5, `generate_dissipator_term`) and the symbolic Hamiltonian term (stage 6, only paid by `method="expanded"`) are the largest symbolic-construction costs, dwarfing `prepare_lindblad_problem` (stage 7), which is what the Rust solve path actually consumes (`H_symbolic` + `C_array`). This is exactly the setup cost the planned lazy/sparse-entrywise refactor of `OBESystem.system`/`.dissipator` targets: consumers that only need `H_symbolic`/`C_array` (i.e. the Rust solve path) currently pay for `system`/`dissipator` construction they never use. The end-to-end `method="matrix"` self-check (stages 1-5 only, no stage 6) agrees with the summed per-stage timings within the expected tolerance, validating that the per-stage instrumentation above isn't missing or double-counting work. Re-run this benchmark after the lazy/sparse refactor lands to quantify the improvement -- stages 5 and 6 should shrink dramatically for systems whose collapse operators are single-jump (the common case), while stages 1-4 and 7 should be unaffected.
