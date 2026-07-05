# In-Solver Integral Output Benchmark

## Setup

- Model: two-level Lindblad system with one decay channel.
- Scan: 101 detuning points from -2 to 2.
- Time span: 0 to 4; post-solve grid has 401 save points.
- Repeats per method: 7.
- Threads: Rayon default.
- Integral weights: excited-state population times Gamma = 0.3.

## Timing

| Method | mean (s) | stdev (s) | min (s) | max (s) | repeats |
| --- | ---: | ---: | ---: | ---: | ---: |
| in-solver final, saveat=None | 0.002137 | 0.000219 | 0.001901 | 0.002478 | 7 |
| post-solve populations + np.trapezoid | 0.002850 | 0.000495 | 0.002284 | 0.003585 | 7 |
| in-solver cumulative trace | 0.002665 | 0.000127 | 0.002485 | 0.002835 | 7 |

The final scalar in-solver path is 1.33x faster than saving populations and integrating in Python.
The cumulative in-solver trace path is 1.07x faster than saving all populations and integrating in Python.

## Value Differences

All differences are in expected emitted photons per molecule.

| Comparison | max abs diff | RMS diff | mean signed diff |
| --- | ---: | ---: | ---: |
| post-solve trapezoid - in-solver final | 7.617766e-05 | 4.598143e-05 | 3.473277e-05 |
| in-solver trace final - post-solve trapezoid | 2.220446e-16 | 9.462886e-17 | 2.047317e-17 |
| in-solver trace final - in-solver final | 7.617766e-05 | 4.598143e-05 | 3.473277e-05 |

## Notes

- `output="photon_integral", output_when="final", saveat=None` integrates on accepted solver steps and returns one scalar per trajectory.
- Post-solve integration saves all populations on `saveat`, extracts the excited-state population, and integrates with `np.trapezoid`.
- Cumulative trace uses the new in-solver trace mode at the same `saveat` points as the post-solve path.
- Raw timing repeats: `integral_output_timing_repeats.csv`.
- Raw value comparison: `integral_output_value_comparison.csv`.
