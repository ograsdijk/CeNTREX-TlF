# Paired decay and active-pattern tests on integrated D+S+U

Completed 2026-10-17. This replaces the tentative interpretation of the earlier
separate RHS timings. Prepared Hamiltonian assembly and the shared coefficient
DAG are enabled in every run. No production code or defaults were changed.

## Incremental solve-time changes

Positive percentages mean less time; negative percentages mean more time.
Values are median within-round reductions versus the current pipeline in the
same benchmark build. Brackets show descriptive 95% paired bootstrap intervals.

| System | Nonzero decay alone | Active patterns alone | Both | Both vs unpatched production control |
|---|---:|---:|---:|---:|
| rc | +2.06% [+0.96, +2.30] | +0.61% [+0.16, +0.95] | +3.22% [+2.87, +3.44] | +2.64% [+2.19, +3.26] |
| r2 | -2.92% [-4.53, -1.43] | -0.14% [-1.53, +2.99] | -3.75% [-6.59, -1.11] | -3.79% [-5.81, -2.38] |
| r0 | -1.23% [-2.93, +4.28] | -2.18% [-6.49, +3.75] | -0.96% [-6.41, +2.03] | +0.83% [-2.75, +4.26] |
| q1 | +0.46% [-0.88, +3.37] | +0.32% [-2.24, +3.93] | -0.23% [-4.59, +2.81] | -1.36% [-3.68, -0.21] |

R(2), R(0) and Q(1) have no polarization switching here: their active pattern
has the same sparsity as the union. Differences there measure incidental
selection/layout overhead, not the removal of physical interactions.
The unpatched-control comparison matters: a gain versus the hooked baseline
is not automatically the same net gain over the installed algorithm.

The paired RC results support small individual improvements: about 2.1% for
decay and 0.6% for patterns, with 3.2% for both within the hooked pipeline.
The net combined reduction versus the unmodified production control is 2.6%
(descriptive interval 2.2-3.3%). The earlier statement that neither addition
helps individually was too strong. A special combined interaction is not
established by these samples. R(2) regresses; R(0)/Q(1) show no reliable
incremental benefit. This supports an optional RC combination, not a new
generic default. Long apparatus transits remain untested here.

## RC interaction and marginal effects

- decay_with_patterns: +2.60% [+2.05, +2.88]
- patterns_with_decay: +1.35% [+0.73, +1.71]

RC log interaction `log(T_both*T_current/(T_decay*T_patterns))`: -0.006490, interval [-0.018399, +0.001177].
Negative values favor both together beyond multiplicative independence; an
interval spanning zero does not demonstrate such an interaction. Medians of
paired ratios need not exactly equal ratios of separate medians.

## Absolute solve-time medians and hook overhead

| System | Current, hooked (s) | Decay (s) | Patterns (s) | Both (s) | Unpatched (s) | Hook overhead vs unpatched |
|---|---:|---:|---:|---:|---:|---:|
| rc | 3.021878 | 2.959968 | 3.003541 | 2.929927 | 3.003801 | +0.17% |
| r2 | 0.265059 | 0.270072 | 0.260548 | 0.270949 | 0.264330 | +0.51% |
| r0 | 0.081060 | 0.082169 | 0.083090 | 0.081502 | 0.082174 | +1.33% |
| q1 | 0.153256 | 0.150459 | 0.150683 | 0.151439 | 0.150214 | +1.04% |

## Method and scope

The baseline and candidates call the current real production RHS, parameter
evaluation, prepared assembly, shared coefficient DAG and Tsit5 source. A
benchmark-only snapshot adds two hooks: select a prepared gate-conditioned
sparse pattern, and replace the structured decay loop by nonzero damping/transfer
lists. It does not use the earlier generic assembly adapter. All four candidates
share one compiled binary and dispatch; a second binary contains the unmodified
production RHS as a separate control. Sources and snapshot hashes are in the JSON.

Each system has 12 paired rounds, each including all four configurations and
the unpatched control. The four-configuration order uses randomized/relabelled
four-round Williams blocks; the control is inserted at a cycling position.
Preparation/build time is excluded. Warmup precedes sampling. One CPU thread
and one BLAS thread are used; no unrelated benchmark runs concurrently.
Intervals resample whole paired rounds (20,000 bootstrap draws, fixed seed),
not individual timings. With only 12 rounds these are descriptive intervals;
they do not cover compiler-layout changes, other workloads or systematic effects.

All are the existing short performance fixtures: 10 us trajectories, E=[0,0,200]
V/cm, B=[0,0,0.01] G, optical EOM at 1.6 MHz with beta=3.8. RC uses 1 MHz
microwave polarization switching with quarter-period offset, all four states,
exact 250 ns segment boundaries and continuously evaluated EOM phase. Pattern
selection overhead at segment boundaries is included. This is not the zero-E
apparatus model, a full transit or a parallel production scan.

Candidate atol/rtol: 1e-11/1e-9. Strict unpatched reference: 1e-13/1e-11.
Every trajectory checks complex density entries: 1% relative error at reference
magnitude >=1e-6, absolute error <=1e-6 below, including coherences. Trace error
is below 1e-9. RHS checks cover random positive Hermitian and initial densities,
four EOM times/all RC gate states, Rabi multipliers 0, 0.3, 1 and 2 and changed
detunings. Raw JSON retains errors, RHS/accepted/rejected counts, execution
orders, every observation, and source hashes.

Active patterns remove coefficients only when gate-zero propagation proves
them identically zero throughout the segment; instantaneous EOM zeros do not
qualify. Decay lists remove exactly zero operations, with no new physical cutoff.
The `counts` array records union NNZ, damping entries, transfer entries, then
NNZ for each binary gate configuration in native parameter-slot order.

## Reproduce

```
.venv/Scripts/python.exe benchmarks/check_rc_integrated_decay_patterns.py
.venv/Scripts/python.exe benchmarks/report_integrated_decay_patterns.py
```

Sources and results live outside `.cache`; only compiled extensions/snapshots
live there. The native libraries are compiled once and reused across systems.
