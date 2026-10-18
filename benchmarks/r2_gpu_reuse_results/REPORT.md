# R(2) GPU ODE and static eigensystem reuse: completed

Hardware: RTX 5070 Ti; Ryzen 9800X3D (8 physical cores). FP64 throughout. No CPU ODE benchmarks were rerun. All files are benchmark artifacts; no production solver or scan configuration was changed.

## One static eigensystem, many populations and velocities

The earlier GPU dense tests reached batches of 8 different matrices and 20 initial populations per matrix; they refactored every task. The new test factors one archived R(2) F4 generator on the CPU once, uploads its real eigenbasis and initial modal coefficients, then evaluates all 20 resolved X J=2 initial projectors at many propagation times. Velocities span 50–250 m/s and set t = 0.020 m / v in a uniform static field. All other generator parameters are fixed. Arbitrary diagonal mixtures of these 20 populations follow by linear combination.

The 41-level model has 1444 active real packed coordinates; its three sink populations are recovered through the integrated feed exactly. Output is the complete lossless packed density (1681 real values per case), including all coherences and sinks, returned to host memory. Zero sink coherences follow exact reachability checks. CUDA reconstructs the full packed density in this prototype. The existing public CUDA projection path does not yet provide this implementation.

Timings include modal function evaluation, full reconstruction, allocation and GPU-to-CPU transfer. They exclude the shared eigendecomposition, initial solves, one-time factor upload/context startup, disk writing and diagnostics. Each timing is the median of three repetitions. Both CPU and GPU reuse exponential/integral functions across initial populations. CPU batch sizes 128/512/2048/8192 and BLAS thread counts 1/8 were checked; GPU uses the same batch-size sweep. The table compares the best measured CPU and GPU configurations.

| Initial populations × velocities | Cases | CPU seconds | GPU seconds | Speedup |
|---|---:|---:|---:|---:|
| 20 × 1 | 20 | 0.00091 | 0.00165 | 0.55× |
| 20 × 21 | 420 | 0.01926 | 0.00578 | 3.33× |
| 20 × 101 | 2,020 | 0.09742 | 0.01972 | 4.94× |
| 20 × 501 | 10,020 | 0.47548 | 0.11437 | 4.16× |

One-time CPU factor/initial preparation was 0.616 s. Initial CUDA factor upload, including context startup in this run, was 0.109 s. These costs matter for a cold, small job; the speedups above concern reused evaluation. At 20 outputs, the CPU wins.

Every output coordinate was compared with the CPU result: maximum absolute difference 3.61e-16. Complex-entry, trace and positivity diagnostics were also evaluated on up to 60 distributed outputs per row. This is far below the 1% criterion. The initial v1 timings were superseded by v2 after improving CPU reuse of modal functions; quote v2 only.

## R(2) ODE against the existing CPU benchmark

This uses the frozen 40-level cross-system recipe: 8 detuning/power points, all 40 one-hot model populations, 1 µs evolution, and full final density. GPU batches are grouped by optical Rabi rate (200, 80 and 40 trajectories). The CPU number is the existing eight-worker D+S+U result, not a newly run CPU benchmark.

| Method | Seconds | Result |
|---|---:|---|
| Archived CPU D+S+U, 8 workers | 4.6279 | Strict archived tolerances |
| GPU swap_const_exact, 0.5 ns | 1.7593 | Pass |
| GPU swap_const_exact_packed, 0.5 ns | 1.7777 | Pass |
| GPU swap_const_exact_packed_direct, 0.5 ns | 1.8920 | Pass |

Best passing GPU time: 1.7593 s, 2.63× faster than the archived CPU run. Worst complex-entry relative error for |ρij| ≥ 1e-6 was 0.1803%; maximum error below the cutoff was 2.43e-09. Trace and positivity diagnostics pass. A 1 ns scan failed at 10.75% worst relative error and is retained in `matched_cross_v1/`; its faster timing is excluded.

The physical configuration matches the archived CPU fixture. The currently installed native binary has a different hash from the historical CPU binary; this is a comparison to archived CPU performance, not a fresh same-build timing. All 320 output densities were validated against the saved strict CPU reference. Setup/compilation/capture, diagnostics and disk compression are excluded from GPU evolution/output timings, as are historical CPU fixture startup costs.

## Smaller-system EOM and storage checks

The same R(2) model was tested with β=3.8, 1.6 MHz optical phase modulation, entry at absolute t=125 ns, and 1 µs evolution. Sixty distinct cases (detuning −20/0/+20 MHz × 20 initial projectors) are repeated to fill batch 512. This is an ODE test. The saved strict CPU main-state result is checked separately; the full batch is compared with a 0.5 ns GPU reference.

| Variant | Step (ns) | Seconds | Accuracy |
|---|---:|---:|---|
| swap_const_exact | 0.5 | 2.6248 | Pass |
| base | 1 | 1.4055 | Pass |
| swap_const_exact | 1 | 1.3115 | Pass |
| swap_const_exact_packed | 1 | 1.1380 | Pass |
| swap_const_exact_packed_direct | 1 | 1.1448 | Pass |
| swap_const_exact_packed | 2 | 0.5717 | Pass |
| swap_const_exact_packed | 4 | 0.2912 | Fail |

At the same 1 ns step, packed/staged storage plus the stage/tail changes cuts runtime by 19.0%. Packed/direct is effectively tied here; staged is slightly faster. The 2 ns step passes with worst complex-entry error 0.0593% across the batch; 4 ns fails at 13.81%. This EOM confirmation covers 1 µs, not a full experimental transit.

Static nominal R(2) checks at 400 µs also pass for fixed Tsit5 at 1/2/4 ns over all 20 ground projectors. The integrating-factor prototype passes at 4 ns over 1 µs but reaches 1.0313% error at 20 µs, so it is not selected. These static nominal checks do not license a 4 ns step for the wider detuning or EOM scans.

## Ordinary complex density-matrix outputs

A follow-up returns actual complex128 41×41 matrices on the host, including GPU unpacking and the larger full-complex transfer. The CPU uses the same vectorized reconstruction. This preserves the reuse gain with ordinary matrix outputs, not just lossless packed storage. Factorization and one-time upload remain excluded.

| Initial populations × velocities | CPU seconds | GPU seconds | Speedup |
|---|---:|---:|---:|
| 20 × 1 | 0.00100 | 0.00192 | 0.52× |
| 20 × 21 | 0.02957 | 0.00835 | 3.54× |
| 20 × 101 | 0.11984 | 0.02573 | 4.66× |
| 20 × 501 | 0.58225 | 0.15316 | 3.80× |

Maximum difference over every returned complex entry: 3.61e-16. Raw results and source provenance are in `dense_complex_v1/`.

## Artifacts and scope

- `v2_20261012/dense_results.json`: final dense batch sweep; factors stored alongside it.
- `matched_cross_final/`: passing GPU scan, candidate/reference densities, source snapshots and hashes.
- `matched_cross_v1/`: rejected 1 ns wider scan.
- `matched_cross_v1/ode_sweep_results.json`: static step/storage/IF screening.
- `long_confirmation_v1/`: longer static checks through 400 µs.
- `eom_v1/`: EOM candidates, fine references, CPU reference checks and source snapshots.
- `v1_20261012/ode_results.json`: original 41-level smaller-system storage/IF pilot.

The strongest dense reuse gain comes from keeping factors resident and reconstructing many outputs together. It does not mean GPU eigendecomposition itself is faster. Library integration and full-transit EOM validation remain separate work. Core results were complete before the inherited 16:37:10 UTC cutoff. The complex-output follow-up completed on the later keep-going turn. The deadline guard then refused to launch the planned longer EOM jobs; those were not run.
