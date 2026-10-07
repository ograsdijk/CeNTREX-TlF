# Dense solver library integration validation

Public solver APIs checked against saved F4 responses at ten selected power/polarization/detuning combinations. Twenty independent initial states, 1401 photon samples; full density checks at four times.

Maximum saved-reference photon error: 3.77e-15. Maximum CPU/CUDA difference: 4.22e-15. Trace and density positivity checks passed.

Small 32-point unique-parameter grid, repeated twice in opposite order. Timings include parameter binding, analytic generator extraction, reduction, conditioning/residual checks, fresh eight-worker process startup, communication, transfers, projection and result collation. Common OBE construction and report writing are excluded. No ODE scan rerun.

| Evolution | Median seconds | Points/s |
|---|---:|---:|
| cpu | 10.000 | 3.20 |
| cuda | 10.250 | 3.12 |

System: AMD Ryzen 7 9800X3D, eight physical cores / sixteen logical processors, 64 GB RAM class; RTX 5070 Ti. One BLAS thread per CPU worker. Software/driver/BLAS details: [system_config.json](system_config.json). Detailed numerical and timing data: [results.json](results.json).
