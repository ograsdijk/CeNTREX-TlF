# Generic prepared-data CPU RHS experiments

**Follow-up:** [Matched tests on the integrated pipeline](INTEGRATED_DECAY_PATTERNS_PAIRED_REPORT.md)
replace the tentative interpretation of decay/pattern RHS timings below. RC shows
small individual gains and a 2.6% net combined reduction versus an unpatched
control; a special interaction is not established.

Completed 2026-10-16. One reusable Rust binary serves RC, R(2), R(0) and Q(1);
no system-specific native arithmetic or recompilation between models is used.
Production library code is unchanged. This tests the four remaining generic
opportunities after the [generated-RHS experiments](SEPARATELY_GENERATED_RHS_REPORT.md).

## Trajectory results

Matched 10 us Tsit5 solves, five observations per candidate, randomized candidate
order in each repeat after warmup. Medians in seconds. The best candidate is
selected from the trajectory-tested finalists, not solely from RHS microtimings.

| System | Library D+S+U | Adapter control | Best candidate | Reduction vs library | Candidate |
|---|---:|---:|---:|---:|---|
| rc | 3.255132 | 3.286595 | 2.897665 | 10.98% | D+S+U_SoA:prepared_H,nonzero_decay,active_pattern |
| r2 | 0.273197 | 0.281747 | 0.247692 | 9.34% | D+S+U_SoA:prepared_H,shared_coefficient_DAG |
| r0 | 0.088558 | 0.094941 | 0.080729 | 8.84% | D+S+U_SoA:prepared_H,shared_coefficient_DAG,active_pattern |
| q1 | 0.157971 | 0.163022 | 0.143871 | 8.93% | D+S+U_SoA:prepared_H,shared_coefficient_DAG |

The adapter control implements the same assembly/decay/commutator arithmetic
with a separate coefficient buffer and generic benchmark dispatch. Differences
from the library baseline expose harness/layout overhead; do not attribute every
control-relative improvement to a production speedup. Compare against the actual
library column. Small differences among equivalent/no-op combinations are noise.

Prepared Hamiltonian assembly alone reduces solve time by 7.5-9.2% across
these four fixtures and is the clearest generic change to pursue. D+S+U SoA
remains the winning commutator family on each. For RC, adding nonzero decay
and active gate patterns raises the measured reduction to about 11%; the
full four-change combination gives 10.6%. The other optical fixtures have
no gate configurations to specialize and no repeated coefficient DAG nodes,
so tiny differences between those enabled/disabled combinations are not evidence
of useful CSE or switching gains. Enabling every change is not uniformly best.
Some RC samples have large timing outliers; all raw observations are retained.
A sensible integration order is prepared assembly first, then validate the
additional RC combination in the library and a production-context fixture.

## Individual changes and full combination with D+S+U

| System | Variant | Solve seconds | Reduction vs library |
|---|---|---:|---:|
| rc | adapter control | 3.286595 | -0.97% |
| rc | D+S+U_SoA:prepared_H | 3.009745 | 7.54% |
| rc | D+S+U_SoA:nonzero_decay | 3.373771 | -3.64% |
| rc | D+S+U_SoA:shared_coefficient_DAG | 3.309995 | -1.69% |
| rc | D+S+U_SoA:active_pattern | 3.258638 | -0.11% |
| rc | D+S+U_SoA:prepared_H,nonzero_decay,shared_coefficient_DAG,active_pattern | 2.910057 | 10.60% |
| r2 | adapter control | 0.281747 | -3.13% |
| r2 | D+S+U_SoA:prepared_H | 0.248015 | 9.22% |
| r2 | D+S+U_SoA:nonzero_decay | 0.285981 | -4.68% |
| r2 | D+S+U_SoA:shared_coefficient_DAG | 0.280234 | -2.58% |
| r2 | D+S+U_SoA:active_pattern | 0.281039 | -2.87% |
| r2 | D+S+U_SoA:prepared_H,nonzero_decay,shared_coefficient_DAG,active_pattern | 0.251979 | 7.77% |
| r0 | adapter control | 0.094941 | -7.21% |
| r0 | D+S+U_SoA:prepared_H | 0.081361 | 8.13% |
| r0 | D+S+U_SoA:nonzero_decay | 0.095510 | -7.85% |
| r0 | D+S+U_SoA:shared_coefficient_DAG | 0.095218 | -7.52% |
| r0 | D+S+U_SoA:active_pattern | 0.094254 | -6.43% |
| r0 | D+S+U_SoA:prepared_H,nonzero_decay,shared_coefficient_DAG,active_pattern | 0.083748 | 5.43% |
| q1 | adapter control | 0.163022 | -3.20% |
| q1 | D+S+U_SoA:prepared_H | 0.143924 | 8.89% |
| q1 | D+S+U_SoA:nonzero_decay | 0.162993 | -3.18% |
| q1 | D+S+U_SoA:shared_coefficient_DAG | 0.162583 | -2.92% |
| q1 | D+S+U_SoA:active_pattern | 0.163394 | -3.43% |
| q1 | D+S+U_SoA:prepared_H,nonzero_decay,shared_coefficient_DAG,active_pattern | 0.145116 | 8.14% |

## Experimental matrix and implementation

Each system has all 16 on/off combinations of these four changes crossed with
four commutator variants: D+S+U with real/imaginary scratch (SoA), S+U SoA,
D+S+U complex scratch, and S+U complex scratch. This gives 64 RHS candidates
plus the unchanged library D+S+U reference. It is not a scan over every solver
or all execution modes in the library.

1. **prepared_H:** copy a prepared packed static triangle and apply cached
   (upper offset, numeric multiplier) lists for each runtime coefficient.
   Replaces repeated triangle clearing, dense-to-upper traversal and checked indices.
2. **nonzero_decay:** precompute nonzero coherence damping and population transfer
   operations; skip exactly zero damping. No extra physical cutoff is introduced.
3. **shared_coefficient_DAG:** intern identical instruction subtrees across scalar
   coefficient expressions and evaluate shared nodes once. The existing compound
   parameter graph remains evaluated once per RHS; this is structural reuse, not
   algebraic SymPy CSE or native expression compilation.
4. **active_pattern:** prepare sparsity patterns for externally frozen binary gate
   configurations; select at segment boundaries. Remove coefficients only when
   zero propagation from a disabled gate proves them identically zero. Never
   infer a permanent zero from an instantaneous EOM/drive value. RC prepares
   16 combinations of its four PA/PY slots, of which four are physically used.

The kernels use runtime plan dimensions, arrays and coefficients. The prototype
requires decomposed scalar expressions and externally supplied binary gate states;
its adapter recognizes the RC fixture's gate names. A library integration would
need general gate registration/fallback, static-H caching and all supported
expression types. Native compilation is shared, but preparation is per system.

## Preparation and expression reuse

| System | Prepared data creation (ms) | Original coefficient instructions | Shared DAG nodes |
|---|---:|---:|---:|
| rc | 6.198 | 18 | 16 |
| r2 | 0.384 | 4 | 4 |
| r0 | 0.132 | 4 | 4 |
| q1 | 0.187 | 4 | 4 |

One-time benchmark extension compilation: 5.974 s
with cached Cargo dependencies. Preparation above is one observation and includes
all candidate data structures; it is not a carefully repeated preparation benchmark.
The existing compound graph has already captured most coefficient reuse here.

## Screening, accuracy and limits

The RHS screen uses five samples of 1000 repetitions at four EOM times, with
the initial gates fixed for timing. Correctness checks cover all four gate states.
Trajectory finalists include the control, all four individual changes and the
full combination on D+S+U, plus the three fastest RHS combinations in each
commutator family. Trajectories cycle through all four RC states, including
boundary selection overhead. Not every one of the 64 combinations gets a
trajectory benchmark; the raw JSON records the exact shortlist.

Same Ryzen 7 9800X3D, one CPU thread and one BLAS thread as the earlier tests.
Same performance fixtures, E=[0,0,200] V/cm and B=[0,0,0.01] G; these are not
the zero-electric-field production RC apparatus model. Optical EOM is 1.6 MHz
with beta=3.8. Switching is frozen at exact 250 ns segment midpoints; EOM retains
absolute time. Candidate atol/rtol 1e-11/1e-9; strict reference 1e-13/1e-11.

All trajectory candidates pass the full complex-density entry criterion:
1% relative error at reference magnitude >=1e-6, absolute error <=1e-6 below.
RHS sum checks use random positive Hermitian densities, four EOM times and
four RC configurations, normalized error <1e-10. Additional runtime-change
checks cover all 64 candidates on all four systems at optical/microwave Rabi
multipliers 0, 0.3 and 2, with detunings -3.7, 0 and +5.1 MHz. The same binary
is reused. No long apparatus transit, parallel scan throughput or independent
GPU comparison is claimed. Winners need production-context validation before promotion.

## Reproduce and artifacts

```
.venv/Scripts/python.exe benchmarks/bench_generic_prepared_rhs.py
.venv/Scripts/python.exe benchmarks/check_generic_prepared_rhs.py
.venv/Scripts/python.exe benchmarks/report_generic_prepared_rhs.py
```

Use PYTHONIOENCODING=utf-8. Helpers: `generic_prepared_rhs.rs` and
`generic_coefficient_dag.rs`. Raw samples: `*_generic_prepared.json`.
Runtime checks: `generic_prepared_runtime_checks.json`. Hashes:
`generic_prepared_source_manifest.json`. These sources and results are outside `.cache`.
