# Dense propagator: threading and all-ground-J2 populations

Pure X, 10 mW, existing three detunings, all 20 individually resolved X J=2 population projectors, 1401 cumulative photon samples through 350 us.

All 20 X J=2 F1/F/mF states are retained individually. Three other ground manifolds remain compact decay sinks; this does not resolve every state of every rotational J. The responses describe initially populated dressed eigenstates, not arbitrary coherent bare-state preparation.

## BLAS threads within one parameter point

Two repetitions, opposite thread-count orders, +6.8 MHz. All 20 populations. Dense eig, LU, initial-state projection, and all photon samples included; diagnostics and common Liouvillian extraction excluded.

| BLAS threads | Median seconds | Speedup versus one |
|---:|---:|---:|
| 1 | 1.084 | 1.00x |
| 2 | 0.917 | 1.18x |
| 4 | 0.784 | 1.38x |
| 8 | 0.786 | 1.38x |
| 16 | 0.903 | 1.20x |

## Independent parameter workers

Same three distinct detunings, one BLAS thread per process, two repetitions. Startup and input loading excluded; dispatch/collection included. Only three points, so this does not test saturation of eight physical cores or sixteen logical threads.

| Workers | Median three-point seconds | Speedup versus one worker |
|---:|---:|---:|
| 1 | 3.255 | 1.00x |
| 2 | 2.264 | 1.44x |
| 3 | 1.149 | 2.83x |

## Population count

| Initial populations | Median one-thread seconds |
|---:|---:|
| 4 | 1.077 |
| 20 | 1.095 |

## Accuracy

All 20 photon curves passed monotonicity, reality, and trace checks at the three points. The four previously simulated states match the saved library to at most 1.65e-09 photons. An independently integrated weighted mixture of all 20 states differs by at most 7.76e-10 photons over all 1401 timestamps.

State labels, exact settings, timing repetitions and diagnostics: [results.json](results.json). Each point has an NPZ with all 20 separate photon responses. Initial density matrices are projectors onto the field-dressed states; mixtures can be reconstructed with any nonnegative normalized population weights.

No full scan or production backend changes. Reproduce with `.venv/Scripts/python.exe benchmarks/bench_r2_f4_dense_threading.py`.
