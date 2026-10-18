# Tsit5 repeated-time reuse and AVX2 commutator trials

This benchmark tested the repeated-time Hamiltonian-reuse and SIMD/commutator suggestions on the current prepared-H + shared coefficient DAG + D+S+U path. It uses benchmark-only native snapshots; no production solver or defaults changed.

The fixtures are 10 µs RC (70 states) and R(2) (41 states), with E = 200 V/cm and B = 0.01 G. They are short performance fixtures, not full apparatus trajectories. Each mode ran 12 randomized interleaved rounds on one CPU thread. The paired intervals below are descriptive 95% bootstrap intervals over whole rounds, not formal significance claims.

## Timings

Reductions are calculated from paired round ratios; positive means faster. ‘Hooked control’ uses the same benchmark wrapper and added option checks with both experimental options disabled. The true control is the unmodified current RHS source, so the table also shows wrapper/hook overhead.

| System | Current unmodified median | Hooked control vs current | Time cache vs current | AVX2 vs current | Both vs current |
|---|---:|---:|---:|---:|---:|
| rc | 3.00878 s | -0.24% [-0.32, -0.08] | +0.37% [+0.13, +0.62] | -31.37% [-31.64, -30.94] | -30.92% [-31.38, -30.14] |
| r2 | 0.25197 s | -1.80% [-2.43, -1.22] | -1.15% [-1.51, -0.47] | -29.29% [-29.87, -28.66] | -28.66% [-29.19, -27.84] |

Time reuse compared with hooked control reduced RC time by 0.65% (95% interval 0.45–0.79%) and R(2) by 0.44% (0.21–1.15%). The current-RHS comparison is the relevant net result: RC improved by 0.37% (0.13–0.62%), while R(2) was 1.15% slower (0.47–1.51%). The hook/control overhead is itself about 0.24% for RC and 1.80% for R(2), which is why both comparisons are shown.

The AVX2 row kernel was 31.37% slower on RC and 29.29% slower on R(2); combining it with time reuse remained about 30.92% and 28.66% slower. It is a result for this specific AVX2 full-Hρ loop and memory order, not a general verdict on SIMD. It is not a candidate for integration.

## What the profile says

The native profiler times unpack/pack outside its core `total_seconds` field. Within that core, the current hooked-control profile split was:

| System | Parameter evaluation | Hamiltonian fill | Commutator | Dissipator |
|---|---:|---:|---:|---:|
| rc | 0.36% | 3.64% | 90.81% | 4.63% |
| r2 | 1.39% | 4.12% | 81.40% | 10.23% |

The profiled trajectory wall time also gives a rough bound on work outside those RHS scopes. Subtracting core RHS time plus separately measured unpack and pack leaves about 0.279 s (9.1% of profiled wall) for RC and 0.041 s (15.4%) for R(2). This residual includes Tsit5 stage construction/error control, solver/output and fixture wrapper work, and profiler overhead; it does not isolate ODE error control by itself.


The commutator dominates both fixtures, especially RC. The profiler therefore points toward a better sparse commutator kernel, but this direct AVX2 rewrite regressed substantially. The next kernel idea should change the sparse traversal or reuse layout, then be compared against the current D+S+U implementation.

Tsit5 made repeated exact-time calls in both systems: RC had 16,565 extra calls at repeated times out of 99,191 RHS calls (16.70%); R(2) had 6,455 out of 38,531 (16.75%). The repeated stages have different density matrices, so only parameter/Hamiltonian work can be reused. Profiling showed the cache reduced Hamiltonian-fill time, but this work is a small part of the whole RHS, which explains the small net RC gain.

## Correctness

Every trajectory was checked against a strict reference (`atol=1e-13`, `rtol=1e-11`) using the complete complex density matrix. Candidate runs used `atol=1e-11`, `rtol=1e-9`; the agreed entrywise criterion was 1% relative error for entries above 1e-6 magnitude and 1e-6 absolute error below that. All five modes passed on both systems. The largest reported relative errors were 2.14e-4 for RC and 3.72e-7 for R(2); maximum small-entry absolute errors were 2.24e-11 and 6.81e-14. Trace errors stayed below 3.4e-15.

Random-Hermitian RHS comparisons differed by at most 1.95e-16 normalized on RC and 1.85e-16 on R(2). A same-time RC gate override after a cached call matched a fresh RHS at 0 normalized error, confirming that parameter changes invalidate the cache. R(2) has no switching gates in this fixture.

All modes made the same number of RHS calls, accepted steps, and rejected steps within each system. Thus the timing differences come from the RHS work rather than solver step-count changes.

## Reproduction and artifacts

Run `python benchmarks/check_stage_reuse_simd.py --run` to rebuild isolated wrappers and repeat the experiment. `--build-only` compiles the isolated modules without running trajectories. The AVX2 kernel helper is `benchmarks/stage_reuse_simd_kernel.rs`; raw round timings, density checks, stage-time counts, and source hashes are in `stage_reuse_simd.json`.

The generated extensions and build logs live under `.cache/benchmarks/stage_reuse_simd_v1/`; those cached build artifacts are not the record of results. The JSON and this report are the durable artifacts.
