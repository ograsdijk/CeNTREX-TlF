# Dense solver library integration validation

Public solver APIs checked against saved F4 responses at ten selected power/polarization/detuning combinations. Twenty independent initial states, 1401 photon samples; full density checks at four times.

Maximum saved-reference photon error: 3.77e-15. Maximum CPU/CUDA difference: 4.22e-15. Trace and density positivity checks passed.

Small 32-point unique-parameter grid, repeated twice in opposite order. Timings include parameter binding, analytic generator extraction, reduction, conditioning/residual checks, fresh eight-worker process startup, communication, transfers, projection and result collation. Common OBE construction and report writing are excluded. No ODE scan rerun.

Each CPU worker now reuses one native plan/evaluator/workspace, replacing parameter overrides with cache invalidation. Worker readiness is synchronized to measure startup separately. Steady-state timing includes extraction, reduction, decomposition, communication, projection, transfers and collation; it excludes worker startup and teardown.

| Evolution | Total seconds | Total points/s | Worker startup seconds | Steady seconds | Steady points/s |
|---|---:|---:|---:|---:|---:|
| cpu | 12.549 | 2.55 | 8.383 | 3.927 | 8.15 |
| cuda | 12.898 | 2.48 | 8.476 | 3.607 | 8.87 |

Before this fix, the same public 32-point API measured approximately 10.00 s CPU-only and 10.25 s CPU/GPU including startup. Prior warm preassembled-matrix benchmarks measured 8.53 and 9.17 points/s respectively, with parameter binding/generator extraction excluded; those remain a narrower timing boundary.

## Default startup scheduling

Without startup profiling, ready workers begin solving while other workers initialize. The following cold-call timings use this default scheduling (two repetitions), including startup and teardown.

| Evolution | Total seconds | Points/s |
|---|---:|---:|
| cpu | 10.089 | 3.17 |
| cuda | 9.828 | 3.26 |

System: AMD Ryzen 7 9800X3D, eight physical cores / sixteen logical processors, 64 GB RAM class; RTX 5070 Ti. One BLAS thread per CPU worker. Software/driver/BLAS details: [system_config.json](system_config.json). Detailed numerical and timing data: [results.json](results.json).
