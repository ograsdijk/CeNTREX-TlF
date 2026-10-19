# Four-population dense-propagator benchmark

Three detunings, pure X, 10 mW, four independent starting states, 1401 times through 350 us.

Reuses the existing exact-propagator approach. At each fixed parameter point, extract the real packed Liouvillian, eigendecompose once, solve for all four initial projectors together, and analytically integrate the photon observable at every saved time. No shared adaptive timestep controller and no physics approximation.

One timing per point. Dense uses one BLAS thread; independent reference uses four Rayon threads. Diagnostics excluded from dense total; common system build excluded from both.

| UV detuning (MHz) | Dense total (s) | Four-thread independent ODE (s) | Wall speedup | Max photon error | Accuracy gate |
|---:|---:|---:|---:|---:|---|
| 6.8 | 1.091 | 6.340 | 5.81x | 1.65e-09 | PASS |
| 28.2 | 1.053 | 6.381 | 6.06x | 8.48e-11 | PASS |
| 80 | 1.093 | 10.791 | 9.87x | 3.32e-10 | PASS |

Three-point total: dense 3.237 s; independent ODE 23.512 s; ratio 7.26x.

Accuracy gate: every cumulative photon value within 1e-7 + 1e-5 times the reference magnitude; trace error and imaginary photon residual below 1e-7; photon increments at least -1e-7. References are the current native solver-integral path, not post-solve trapezoids.

Limits: one timing per point, one polarization and power. This tests feasibility, not full-scan throughput. Production scans can run eight or sixteen independent detunings concurrently; dense parallelism and memory use need separate consideration. Dense factorization scales steeply with retained state count. No production backend changes were made.

Details and stages: [results.json](results.json). All four photon traces and the completed-scan references are saved in the point NPZs.

Reproduce: `.venv/Scripts/python.exe benchmarks/bench_r2_f4_population_propagator.py`.
