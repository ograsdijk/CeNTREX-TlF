# Additional CPU OBE optimization checks

Newest follow-up: [FMA, AVX-512 and vectorized finalization](extended_simd_dsu/REPORT.md)
improves on the fused AVX2 winner by another 14.21% for RC, 17.20% for R(2),
11.44% for R(0), and 13.23% for Q(1), using the best tested combination per
system on the Ryzen 9800X3D. FMA plus AVX2 finalization is a common beneficial
choice (11.02–17.20%); AVX-512 is useful for RC but hurts smaller systems here.
All full-density checks pass; these remain benchmark prototypes.

Latest follow-up: [fusing decay and packed output with SIMD](fused_dsu/REPORT.md)
adds about 4% runtime reduction beyond the original SIMD kernel for RC and R(2),
and 11% for R(0), in direct eight-round comparisons. Q(1)'s additional gain is
uncertain. The separate diagonal/decay arithmetic combination adds no consistent
further benefit. These are benchmark prototypes, not integrated defaults.

Follow-up: [preserving D+S+U while adding SIMD](block_simd_dsu/REPORT.md) subsequently
reduced RC runtime by 21.18% and R(2) by 11.21% in eight paired rounds. That kernel
retains the diagonal split and existing sparse traversal; it differs from the
slower full-H AVX2 prototype below. The new blockwise D+S+U variant did not help.

Luna high prepared and ran the comparisons; the orchestrator reviewed the harnesses, corrected the final reduction benchmark and completed it after the delegated agents reached their usage limit. No production library defaults or kernels changed.

| Suggestion | RC result | Other systems / conclusion |
|---|---|---|
| Repeated-time Hamiltonian cache | 0.37% runtime reduction vs unpatched current RHS | R(2) 1.15% slower; marginal and not a generic win |
| Tested AVX2 commutator | 31.37% slower | R(2) 29.29% slower; specific full-H row kernel, not a verdict on all SIMD |
| Shared timestep and H across three initial densities | 4.72% slower than cached independent serial solves | R(2) no reliable difference; existing three-worker batch faster |
| Reduce ODE coordinates, retaining full D+S+U RHS | 8.54% slower | R(2) no reliable gain |
| Reduce expanded-sparse RHS output rows and unreachable inputs | 513% slower than D+S+U | R(2) 88% slower than D+S+U, though faster than full expanded sparse |
| Adaptive Tsit5 instead of Dopri5 | 12.17% runtime reduction | 16–17% on R(2), R(0), Q(1); previous optimized benchmark already uses Tsit5 |
| Fixed RK4 0.5 ns | ~25% runtime reduction, but fails 1% density-entry criterion at +0.5 MHz | Not accepted across tested RC points |
| Fixed RK4 0.4 ns | 6.24–6.89% runtime reduction; all tested checkpoints pass | Worst RC relative entry error 0.495%; these small timesteps slow R(2) |

The commutator accounts for about 91% of profiled core RHS time in RC and 81% in R(2). Exact support closure proves 11.02% and 56.75% of density coordinates removable for the tested RC and R(2) initial populations. Those counts alone do not establish faster execution: scatter/gather costs, sparse indexing and the full coefficient assembly remain. A blockwise D+S+U implementation was not tested.

These are short 10 microsecond E=200 V/cm, B=0.01 G EOM performance fixtures, not the zero-E apparatus or full transit. Most adaptive comparisons use atol=1e-11, rtol=1e-9; references use 1e-13/1e-11. Checks include complex coherences: 1% entry-relative error for reference magnitude >=1e-6, absolute 1e-6 below, plus trace. The fixed-step follow-up checks four times within each trajectory and an additional RC detuning. It does not validate a timestep throughout a production scan.

Detailed methods, raw repetitions, accuracy checks and runnable scripts:

- [Stage reuse, AVX2 and profiling](../rust_stage_reuse_simd_results/STAGE_REUSE_SIMD_REPORT.md)
- [Shared mixtures and support closure](shared_mixtures_sectors/REPORT.md)
- [Reduced-coordinate prototypes](density_reduction/REPORT.md)
- [Adaptive CPU solver comparison](CURRENT_CPU_SOLVERS_REPORT.md)
- [Fixed RK4 checkpoint/detuning follow-up](FIXED_RK4_FOLLOWUP_REPORT.md)
