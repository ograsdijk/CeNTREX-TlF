# Prepared Hamiltonian assembly and coefficient sharing: library integration

Implemented and validated 2026-10-17. The generic Rust decomposed Hamiltonian path
now prepares static upper-triangle values and per-coefficient (offset, multiplier)
updates once. Upper-triangle RHS kernels, including D+S+U, reuse those arrays.
This removes repeated triangle clearing, dense-to-upper copying loops and checked
index computation. It does not generate native arithmetic for individual molecules.

Repeated scalar coefficient instruction subtrees also share a prepared DAG.
The DAG is immutable; each solver workspace owns evaluation scratch. Parameters
and time are read again at each RHS call. Complex arithmetic, conjugation, scalar
helpers and comparisons retain their original operation order. Expressions with
no sharing, tuple results or unsupported temporary references use the original
evaluator. Dense assembly and expanded-sparse coefficient evaluation also reuse
the DAG when applicable. No additional physical cutoff or density reduction is used.

These changes are automatic for decomposed Rust plans. D+S+U selection and the
library's existing execution-mode defaults are unchanged. Nonzero-only decay and
active switching patterns remain experiments and are not part of this integration.

They have now been checked in a
[paired integrated comparison](INTEGRATED_DECAY_PATTERNS_PAIRED_REPORT.md): RC has
small individual gains and a 2.6% net combined reduction versus an unmodified
control. A special combined interaction is not established; the optical fixtures
do not show a reliable gain. These additions are not enabled in the library.

### Independent switches

Both optimizations default to enabled. They can be disabled independently when
preparing the problem, keeping the same decomposed representation and D+S+U mode:

```python
prepared = prepare_lindblad_problem(
    system, params, backend="rust", hamiltonian_representation="decomposed",
    prepared_hamiltonian_assembly=False,
    shared_coefficient_dag=False,
)
```

With both disabled, the original upper-triangle assembly and coefficient
evaluation paths are used. Disabling assembly also skips building the cached
upper/update arrays; disabling sharing skips DAG construction. Options travel in
the Hamiltonian payload and are honored by batch/grid solves. Entrywise and
Python evaluation do not use these optimizations. Changing settings requires
preparing another problem, rather than mutating an already-built native plan.

## Matched integrated performance

Five alternating before/after observations per system after warmup, one CPU
thread and one BLAS thread, same Rust Tsit5 and D+S+U kernels, 10 us trajectories:

| System | Before (s) | Integrated (s) | Median runtime reduction |
|---|---:|---:|---:|
| RC | 3.415239 | 3.126473 | 8.46% |
| R(2) | 0.280659 | 0.261766 | 6.73% |
| R(0) | 0.086732 | 0.079261 | 8.61% |
| Q(1) | 0.162410 | 0.146949 | 9.52% |

This is a new matched comparison in Python 3.11.13, not a direct comparison with
the earlier prototype's absolute timings. Both native wrappers were freshly
compiled in the same environment. The baseline uses plan.rs and eval.rs from
commit `4eea25f8a5ac6818501372f977eabe18fdd9495c`; both wrappers use the same
current RHS/solver, including pre-existing uncommitted work. Source hashes and
all observations are in `prepared_hamiltonian_integration.json`.

These are the existing numerical fixtures: E=[0,0,200] V/cm, B=[0,0,0.01] G,
optical EOM at 1.6 MHz, beta=3.8. RC has switched microwaves with exact 250 ns
segment boundaries and all four configurations; EOM retains absolute time. These
are not the zero-electric-field production apparatus model or full-duration
transits. Candidate atol/rtol are 1e-11/1e-9, compared with a before-version
reference at 1e-13/1e-11. No parallel-throughput or GPU speedup is claimed.

## Correctness and local build

- All four before/after full-density comparisons pass: 1% relative error for
  reference magnitudes >=1e-6, absolute error <=1e-6 below, including coherences.
- RHS checks compare random positive Hermitian densities and initial states at
  four times/configurations and Rabi multipliers 0, 0.3, 1 and 2, with changed
  detunings. Normalized RHS differences are recorded in the raw JSON.
- 131 Python backend checks pass, including new complex drive/static offset,
  segmented/nonsegmented update, repeated time and multiple execution-mode tests.
  All four independent optimization-switch combinations also pass full-density
  parallel batch tests with parameter overrides and population grid tests.
  Existing backend checks cover static caches, parameter overrides and batches.
- 64 Rust tests pass, with one pre-existing ignored test. New tests check shared
  complex expressions, time/slot changes, phase-modulation helpers and fallbacks.
- The local CPython 3.11 extension was rebuilt and its hash matches the release
  DLL. Python validation imported that rebuilt extension from the working tree.

## Files and reproduction

Implementation: `rust/src/lindblad/plan.rs` and `eval.rs`.
Regression tests: `tests/lindblad/test_prepared_hamiltonian.py` and the Rust DAG
unit tests. Matched timing driver:

```
.venv/Scripts/python.exe benchmarks/confirm_prepared_hamiltonian_integration.py
```

The driver compiles two reusable benchmark extensions, never one per system.
It freezes the baseline plan/evaluator revision and shares the current RHS/solver.
Earlier factorial prototype evidence is in
[GENERIC_PREPARED_RHS_REPORT.md](GENERIC_PREPARED_RHS_REPORT.md).
