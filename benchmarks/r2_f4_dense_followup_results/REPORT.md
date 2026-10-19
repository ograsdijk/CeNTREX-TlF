# Dense propagator: eigen routines, matrix buffers, and physics coverage

Eigen routines and matrix assembly at one existing point, then eight selected parameter points covering 0/3/5% Z and 1/10 mW. No full scan, no new ODE trajectories.

## Eigen routines

One +6.8 MHz pure-X 10 mW point; 20 initial states, 1401 samples. Two repetitions in opposite method orders, one BLAS thread. Includes required copies, eigensystem, real LU, and photon projection; excludes diagnostics.

| Method | Median seconds | Speedup versus SciPy |
|---|---:|---:|
| scipy | 0.675 | 1.000x |
| numpy | 0.720 | 0.938x |
| overwrite_fortran | 0.676 | 0.999x |
| balanced | 0.701 | 0.963x |
| lapack_native | 0.666 | 1.014x |
| lapack_reuse | 0.668 | 1.011x |

Fastest in this small sample: lapack_native. Close differences are not evidence of a robust scan-wide win.

Native LAPACK DGEEV returns a real eigenbasis directly, avoiding complex eigenvector assembly. Reuse variant preallocates its Fortran input buffer. Explicit balancing uses a diagonal similarity transform and restores the eigenvectors to the original coordinates.

## Matrix construction

100 assemblies per repetition, three repetitions. Input Jacobian values/topology already available; this measures dense allocation and filling only. Sparse topology preparation/extraction is excluded.

| Method | Median milliseconds |
|---|---:|
| full_then_slice | 8.095 |
| direct_reduced | 1.888 |
| reuse_reduced | 0.355 |
| reuse_scatter | 0.367 |

## Power and polarization coverage

| Z power | Power (mW) | Detuning (MHz) | Maximum cached photon error | Maximum full-density error |
|---:|---:|---:|---:|---:|
| 0% | 1 | 6.8 | 3.83e-11 | 1e-09 |
| 0% | 1 | 28.2 | 2.44e-11 | 3.76e-10 |
| 3% | 10 | 6.8 | 1.18e-10 | 1.15e-09 |
| 3% | 10 | 28.2 | 8.25e-11 | 7.85e-09 |
| 5% | 10 | 6.8 | 1.14e-10 | 5.28e-09 |
| 5% | 10 | 28.2 | 7.9e-11 | 5.32e-09 |
| 3% | 1 | 28.2 | 2.38e-11 | 9.56e-10 |
| 5% | 1 | 28.2 | 2.38e-11 | 1.39e-09 |

All eight selected points passed the same photon accuracy gate (1e-7 + 1e-5 times reference magnitude), full-density comparison, trace, positivity and cumulative-count monotonicity checks. Cached native-ODE references cover four initial states; full dense references and physical checks cover all 20 initial J2 projectors.

Scope: supplements prior 10 mW pure-X checks; does not validate other transitions, time-dependent drives, arbitrary coherent sink preparation, or ill-conditioned eigensystems. No full scan and no centrex_tlf solver changes.

Timing and validation details: [results.json](results.json). Reproduce with `.venv/Scripts/python.exe benchmarks/bench_r2_f4_dense_followup.py`.
