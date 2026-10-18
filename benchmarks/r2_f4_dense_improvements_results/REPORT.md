# Additional dense-propagator improvements

Same three pure-X 10 mW F4 parameter points; all 20 initial X J2 populations; 1401 photon times.

Exact reduction checks: sink columns have zero feedback; excluded coherences have exactly zero feed from retained variables and zero initial values. The 38-state driven block has 1444 packed variables; all three compact sink populations are recovered by analytic integration of their feed. No dipole pruning or secular approximation is introduced.

The active block is one strongly connected component, so graph-based block splitting offers no further reduction here.

The real-eigenbasis variant replaces complex eigenvector LU with real LU and evaluates only one member of each conjugate pair. Both variants retain full density reconstruction, including sink populations and zero-filled excluded coherences for the validated initial conditions.

## One-worker timings

Two repetitions for method timings, one BLAS thread. Worker throughput uses 32 repeated tasks cycling over the same three matrices, not new detunings; two repetitions. Pool startup/input loading excluded.

| Method | Median seconds per point | Speedup versus full complex |
|---|---:|---:|
| full_complex | 1.091 | 1.00x |
| active_complex | 0.782 | 1.40x |
| active_real | 0.682 | 1.60x |

## Parallel throughput

Same three matrices repeated to supply enough independent tasks; no caching of decompositions across tasks. One BLAS thread per worker. Warm worker pools; process startup and input loading excluded.

| Workers | Median 32-point seconds | Points per second |
|---:|---:|---:|
| 1 | 21.667 | 1.48 |
| 4 | 5.934 | 5.39 |
| 8 | 3.751 | 8.53 |
| 16 | 4.194 | 7.63 |

Best measured configuration: 8 workers. Arithmetic projection for the 5406 parameter points in the original scan: 10.6 minutes of warm compute, excluding setup, extraction, output, and varying physics. This is not a measured full-scan runtime.

## Accuracy and scope

All 20 initial states passed. Maximum difference versus full packed density reconstruction at eight prompt/late times: 5.08e-09. Maximum photon difference over all 1401 times: 1.1e-08. Trace, density-matrix positivity, and cumulative-photon monotonicity checks passed.

Removing sink coherences is valid for initial states with zero such coherence. Arbitrary initial coherences involving sinks need the full path. Retained states are the existing compact 41-level model; this does not reconstruct individual members of compact spectator manifolds.

Only pure X and 10 mW tested. Two timing repetitions, three parameter matrices; wider polarization, power, transition, and conditioning coverage remains necessary before integration as a solver. No changes to centrex_tlf.

Detailed timing stages and numerical checks: [results.json](results.json). Reproduce with `.venv/Scripts/python.exe benchmarks/bench_r2_f4_dense_improvements.py`.
