# FMA, AVX-512 and vector finalization on fused D+S+U

Vectorizing finalization gives the most consistent improvement. The best observed combination differs by system: all three for RC, FMA plus AVX2 finalization for R(2)/Q(1), and finalization alone for R(0) (FMA is nearly tied). AVX-512 is not a universally beneficial setting. No production kernels/defaults changed.

CPU: **AMD Ryzen 7 9800X3D 8-Core Processor**. Runtime checks confirmed AVX2, FMA and AVX512F support. All improvements below are incremental to the previous winning **AVX2 + fused decay + direct packed output** implementation, freshly compiled separately as the true control. They must not be read as comparisons with the older generic solver.

## Best observed combinations

| System | Combination | Previous best (s) | Candidate (s) | Paired reduction | Descriptive 95% interval |
|---|---|---:|---:|---:|---:|
| rc | all_three | 2.030930 | 1.738299 | 14.21% | [10.67, 15.70]% |
| r2 | fma_finalize | 0.224162 | 0.185691 | 17.20% | [16.18, 17.72]% |
| r0 | finalize | 0.072880 | 0.064973 | 11.44% | [9.88, 19.32]% |
| q1 | fma_finalize | 0.136709 | 0.118464 | 13.23% | [11.38, 19.66]% |

For a common combination without AVX-512, FMA plus AVX2 finalization reduced runtime by rc: 11.53%, r2: 17.20%, r0: 11.02%, q1: 13.23%. The small difference between finalization alone and FMA plus finalization for R(0) should not be interpreted as a firm winner.

## All combinations

| System | Option | Median solve (s) | Reduction vs previous winner | Descriptive 95% interval |
|---|---|---:|---:|---:|
| rc | previous_best | 2.030930 | +0.00% | [+0.00, +0.00]% |
| rc | hooked_control | 2.027749 | +0.17% | [-3.27, +2.00]% |
| rc | fma | 1.956708 | +1.89% | [-0.55, +6.06]% |
| rc | avx512 | 1.973727 | +2.10% | [+0.24, +4.13]% |
| rc | fma_avx512 | 1.944749 | +3.89% | [+0.69, +5.72]% |
| rc | finalize | 1.848849 | +8.37% | [+5.36, +9.65]% |
| rc | fma_finalize | 1.773702 | +11.53% | [+9.76, +14.45]% |
| rc | avx512_finalize | 1.763575 | +11.74% | [+9.76, +14.69]% |
| rc | all_three | 1.738299 | +14.21% | [+10.67, +15.70]% |
| r2 | previous_best | 0.224162 | +0.00% | [+0.00, +0.00]% |
| r2 | hooked_control | 0.221767 | +1.04% | [-0.90, +4.54]% |
| r2 | fma | 0.217180 | +3.80% | [+0.51, +5.70]% |
| r2 | avx512 | 0.224809 | +0.36% | [-3.10, +0.61]% |
| r2 | fma_avx512 | 0.226185 | -1.31% | [-2.88, +0.48]% |
| r2 | finalize | 0.189970 | +14.78% | [+13.52, +16.53]% |
| r2 | fma_finalize | 0.185691 | +17.20% | [+16.18, +17.72]% |
| r2 | avx512_finalize | 0.198207 | +12.02% | [+9.20, +12.81]% |
| r2 | all_three | 0.197535 | +11.62% | [+9.99, +12.76]% |
| r0 | previous_best | 0.072880 | +0.00% | [+0.00, +0.00]% |
| r0 | hooked_control | 0.073240 | +0.43% | [-4.37, +6.76]% |
| r0 | fma | 0.071872 | +2.78% | [-0.49, +8.53]% |
| r0 | avx512 | 0.077341 | -4.59% | [-8.38, +2.71]% |
| r0 | fma_avx512 | 0.077202 | -4.88% | [-12.82, +2.55]% |
| r0 | finalize | 0.064973 | +11.44% | [+9.88, +19.32]% |
| r0 | fma_finalize | 0.065081 | +11.02% | [+9.48, +20.48]% |
| r0 | avx512_finalize | 0.069010 | +5.55% | [+4.39, +12.78]% |
| r0 | all_three | 0.069323 | +5.45% | [-1.56, +14.03]% |
| q1 | previous_best | 0.136709 | +0.00% | [+0.00, +0.00]% |
| q1 | hooked_control | 0.131569 | +4.59% | [-1.20, +7.83]% |
| q1 | fma | 0.134087 | +2.19% | [-2.67, +9.57]% |
| q1 | avx512 | 0.139188 | -2.86% | [-5.62, +2.41]% |
| q1 | fma_avx512 | 0.139699 | -2.50% | [-4.69, +4.36]% |
| q1 | finalize | 0.121849 | +11.86% | [+5.59, +17.76]% |
| q1 | fma_finalize | 0.118464 | +13.23% | [+11.38, +19.66]% |
| q1 | avx512_finalize | 0.126812 | +7.03% | [+5.39, +12.85]% |
| q1 | all_three | 0.124915 | +8.47% | [+6.16, +13.42]% |

## Implementation

- FMA fuses the multiply-add/subtract inside complex Hamiltonian products. Accumulation across Hamiltonian couplings remains in its original order, but product rounding changes. Scalar tails remain available.
- AVX-512 processes eight complex density entries per sparse-coupling row update, with 512-bit real and imaginary vectors. AVX2/FMA handles residual entries. Required ISA support is checked before use.
- Finalization uses AVX2 for four off-diagonal entries at once: transposed scratch reads use gathers, diagonal Hamiltonian values are prepared in a compact array, and phase evolution, decay damping and interleaved packed stores are vectorized. Diagonal population-transfer sums and short tails remain scalar. The finalization width remains AVX2 even when the commutator uses AVX-512.
- All options retain the sparse CSR/CSC traversal, Hamiltonian diagonal split, full density coordinates, and exact decay channels. No approximation, coefficient cutoff, state reduction or timestep change is introduced.

The hooked control is the larger experimental kernel with all three additional options disabled. Comparing candidates with the separate previous-winner binary includes dispatch/layout overhead rather than hiding it. Eight randomized interleaved full-solve rounds were run per system: 288 timed trajectories total, plus warmups and strict references. Intervals resample whole paired rounds (20,000 draws) and are descriptive for this small sample. The best row is selected from the tested candidates, so its interval is not selection-adjusted or a cross-CPU guarantee.

## Accuracy and scope

All 288 timed trajectories pass full complex-density checks, including coherences, against strict previous-winner Tsit5 references. Candidate atol/rtol=1e-11/1e-9, reference=1e-13/1e-11. Criterion: 1% entry-relative error above reference magnitude 1e-6, absolute error <=1e-6 below, trace error <1e-9. Arbitrary positive full density matrices are also checked across four EOM times/all microwave configurations, drive multipliers 0/0.3/1/2, and changed detunings.

| System | Max normalized RHS error | Max relative density error (fraction) | Max small-entry absolute error | Max trace error | RHS counts across options |
|---|---:|---:|---:|---:|---|
| rc | 8.42e-19 | 0.000845 | 7.74e-11 | 1.78e-15 | [83495] |
| r2 | 4.48e-20 | 3.71e-07 | 6.8e-14 | 1.33e-15 | [38531] |
| r0 | 7.95e-20 | 1.3e-08 | 6.47e-15 | 3.11e-15 | [68813] |
| q1 | 5.69e-20 | 1.98e-07 | 1.93e-13 | 8.88e-16 | [54353] |

Physical payloads are identical to the preceding fusion benchmark, reused from saved fixtures and individually hashed in the raw data. These are 10 us E=200 V/cm, B=0.01 G performance fixtures with optical EOM beta=3.8 at 1.6 MHz. RC uses uniform X,J=2 initial population and exact 250 ns segmentation for the four microwave configurations; R(2), R(0), Q(1) use uniform modeled X populations. They are not full zero-E apparatus transits. Preparation/build costs are excluded from timing.

Reproduction: `../../check_extended_simd_dsu.py`; kernels: `../../extended_simd_dsu.rs` layered on the prior fused D+S+U implementation. `../../report_extended_simd_dsu.py` creates this report. `results.json` contains raw round orders, timings, density checks, step/RHS counts, fixture/native hashes, CPU features and summaries. All work was implemented/run directly, without agents.
