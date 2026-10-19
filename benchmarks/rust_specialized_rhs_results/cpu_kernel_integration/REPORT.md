# CPU kernel library integration validation

The package now supports `cpu_kernel="auto"` at preparation, independently of the execution mode. On the test machine it resolves to `avx2_fma`. AVX-512 stays explicit. The original implementation remains selectable. All optimized variants fuse structured decay and direct packed output, and vectorize finalization.

CPU: AMD Ryzen 7 9800X3D 8-Core Processor. Eight randomized interleaved rounds per system; all candidates share one freshly compiled generic current-source plan/RHS/Tsit5 wrapper. The `cpu_kernel` payload field drives the actual package dispatch, rather than benchmark-only hook flags. References use the original CPU kernel at atol=1e-13, rtol=1e-11; candidates use 1e-11/1e-9.

| System | Original median (s) | Auto median (s) | Auto resolved | Paired runtime reduction | Descriptive 95% interval |
|---|---:|---:|---|---:|---:|
| rc | 2.623515 | 1.669366 | avx2_fma | 36.47% | [35.66, 38.09]% |
| r2 | 0.263846 | 0.183728 | avx2_fma | 29.21% | [28.33, 31.48]% |
| r0 | 0.079634 | 0.063986 | avx2_fma | 19.92% | [14.75, 21.21]% |
| q1 | 0.150753 | 0.114700 | avx2_fma | 25.84% | [22.30, 27.46]% |

## Explicit selections

| System | Requested kernel | Resolved | Median (s) | Paired reduction vs original |
|---|---|---|---:|---:|
| rc | original | original | 2.623515 | +0.00% |
| rc | auto | avx2_fma | 1.669366 | +36.47% |
| rc | avx2 | avx2 | 1.732311 | +34.34% |
| rc | avx2_fma | avx2_fma | 1.667800 | +36.65% |
| rc | avx512 | avx512 | 1.617346 | +38.52% |
| rc | avx512_fma | avx512_fma | 1.625900 | +37.68% |
| r2 | original | original | 0.263846 | +0.00% |
| r2 | auto | avx2_fma | 0.183728 | +29.21% |
| r2 | avx2 | avx2 | 0.192655 | +29.50% |
| r2 | avx2_fma | avx2_fma | 0.186623 | +30.08% |
| r2 | avx512 | avx512 | 0.191800 | +27.54% |
| r2 | avx512_fma | avx512_fma | 0.186080 | +30.23% |
| r0 | original | original | 0.079634 | +0.00% |
| r0 | auto | avx2_fma | 0.063986 | +19.92% |
| r0 | avx2 | avx2 | 0.068865 | +14.20% |
| r0 | avx2_fma | avx2_fma | 0.064121 | +20.44% |
| r0 | avx512 | avx512 | 0.066124 | +18.78% |
| r0 | avx512_fma | avx512_fma | 0.064779 | +18.49% |
| q1 | original | original | 0.150753 | +0.00% |
| q1 | auto | avx2_fma | 0.114700 | +25.84% |
| q1 | avx2 | avx2 | 0.113966 | +24.99% |
| q1 | avx2_fma | avx2_fma | 0.111700 | +25.47% |
| q1 | avx512 | avx512 | 0.116266 | +23.90% |
| q1 | avx512_fma | avx512_fma | 0.113043 | +24.82% |

## Validation

- 222 Python checks passed across CPU selection, prepared-H optimizations, public aliases and the existing Rust backend suite.
- 66 Rust checks passed; one pre-existing ignored check remains ignored. Synthetic feature tests prove auto falls back safely and never selects AVX-512; unsupported explicit choices fail.
- New tests exercise full complex RHSs at dimensions 1, 2, 4, 5, 8, 9 and 17, including vector tails, both Hamiltonian representations, dephasing and branching decays, static/time-dependent cache invalidation, matrix/packed API interleaving, parameter overrides, photon/rate/population/selected/full/mixed-integral output, parallel batches/grids, switched drives and terminal events.
- Every timed trajectory and warmup passed full complex density checks: 1% entry-relative error above reference magnitude 1e-6, absolute 1e-6 below, trace error <1e-9. Fixed-time random-density checks also compare each integrated variant against original.
- Release extension rebuilt and copied into the project virtualenv editable package. Python lint, Rust formatting and changed-file whitespace checks passed.

| System | Max normalized RHS error | Max relative density error (fraction) | Max small-entry absolute error | Max trace error | RHS counts across kernels |
|---|---:|---:|---:|---:|---|
| rc | 3.71e-19 | 0.000845 | 7.74e-11 | 1.78e-15 | [83495] |
| r2 | 1.28e-19 | 3.71e-07 | 6.8e-14 | 1.33e-15 | [38531] |
| r0 | 6.64e-20 | 1.3e-08 | 6.47e-15 | 3.11e-15 | [68813] |
| q1 | 7.96e-20 | 1.98e-07 | 1.93e-13 | 8.88e-16 | [54353] |

These timings use the established 10 us E=200 V/cm, B=0.01 G EOM performance fixtures with 250 ns segmentation, not full zero-electric-field apparatus transits. RC begins with uniform X,J=2 population; optical fixtures use uniform modeled X populations. Intervals resample whole paired rounds (20,000 draws) and describe this small sample on this CPU; performance on other CPU models is not established.

The public packed Rust D+S+U path is optimized; other execution modes, the default expanded_sparse mode, Python, CUDA, and matrix/split RHS computations retain their paths. The requested option travels through payloads and resolves on the receiving machine. The native plan and Python prepared object expose requested/resolved selections. The original assembly/DAG opt-out flags remain independent.

Reproduction: `../../validate_cpu_kernel_integration.py`; report generator: `../../report_cpu_kernel_integration.py`. `results.json` records every order, timing, density check, step/RHS count and current native source hash. The API tests are `tests/lindblad/test_cpu_kernel.py`. No system-specific native code generation is required in the installed library.
