# RC GPU sparse suite

Status: complete; stage: finished.
Jobs recorded: 132; failures: 2; timeouts: 2.

Production RC continues independently. Float64/complex128 throughout.
CPU references/baselines use the production Python 3.13 Rust binary.
Any 12-thread CPU timings below are concurrent with production and are contended;
they are not an uncontended speedup measurement. Library CuPy methods use eager
launching because this version rejects cuBLAS/cuSPARSE graph capture; custom methods use graphs.

## Integrated GPU measurements

| Method | Batch | Interval (us) | dt (ns) | Median evolution (s) | Traj/s | Max relative entry error above 1e-6 | Qualified |
|---|---:|---:|---:|---:|---:|---:|---|
| staged_soa_minor | 128 | 1 | 1.95 | 0.3649 | 350.8 | 6.259% | False |
| staged_soa_minor | 512 | 1 | 1.95 | 1.563 | 327.6 | 6.259% | False |
| staged_soa_minor | 1024 | 1 | 1.95 | 3.292 | 311.1 | 6.259% | False |
| staged_soa_minor | 2048 | 1 | 1.95 | 6.699 | 305.7 | 6.259% | False |
| staged_soa_minor | 4096 | 1 | 1.95 | 15.05 | 272.2 | 6.259% | False |
| staged_soa_minor | 128 | 1 | 0.977 | 0.721 | 177.5 | 0.3948% | True |
| staged_soa_minor | 512 | 1 | 0.977 | 3.084 | 166 | 0.3948% | True |
| staged_soa_minor | 1024 | 1 | 0.977 | 6.488 | 157.8 | 0.3948% | True |
| staged_soa_minor | 2048 | 1 | 0.977 | 13.21 | 155 | 0.3948% | True |
| staged_soa_minor | 4096 | 1 | 0.977 | 29.65 | 138.1 | 0.3948% | True |
| staged_soa_minor | 128 | 1 | 0.488 | 1.436 | 89.11 | 0.02477% | True |
| staged_soa_minor | 512 | 1 | 0.488 | 6.136 | 83.44 | 0.02477% | True |
| staged_soa_minor | 1024 | 1 | 0.488 | 12.9 | 79.36 | 0.02477% | True |
| staged_soa_minor | 2048 | 1 | 0.488 | 26.24 | 78.06 | 0.02477% | True |
| staged_soa_minor | 4096 | 1 | 0.488 | 58.84 | 69.61 | 0.02477% | True |
| staged_soa_major | 128 | 1 | 1.95 | 0.4038 | 317 | 6.259% | False |
| staged_soa_major | 512 | 1 | 1.95 | 1.654 | 309.5 | 6.259% | False |
| staged_soa_major | 1024 | 1 | 1.95 | 3.34 | 306.6 | 6.259% | False |
| staged_soa_major | 2048 | 1 | 1.95 | 6.665 | 307.3 | 6.259% | False |
| staged_soa_major | 4096 | 1 | 1.95 | 13.32 | 307.6 | 6.259% | False |
| staged_soa_major | 128 | 1 | 0.977 | 0.7949 | 161 | 0.3948% | True |
| staged_soa_major | 512 | 1 | 0.977 | 3.259 | 157.1 | 0.3948% | True |
| staged_soa_major | 1024 | 1 | 0.977 | 6.58 | 155.6 | 0.3948% | True |
| staged_soa_major | 2048 | 1 | 0.977 | 13.13 | 156 | 0.3948% | True |
| staged_soa_major | 4096 | 1 | 0.977 | 26.23 | 156.2 | 0.3948% | True |
| staged_soa_major | 128 | 1 | 0.488 | 1.577 | 81.18 | 0.02477% | True |
| staged_soa_major | 512 | 1 | 0.488 | 6.467 | 79.17 | 0.02477% | True |
| staged_soa_major | 1024 | 1 | 0.488 | 13.06 | 78.41 | 0.02477% | True |
| staged_soa_major | 2048 | 1 | 0.488 | 26.06 | 78.6 | 0.02477% | True |
| staged_soa_major | 4096 | 1 | 0.488 | 52.06 | 78.67 | 0.02477% | True |
| fused_aos_minor | 128 | 1 | 1.95 | 0.5796 | 220.9 | 6.259% | False |
| fused_aos_minor | 512 | 1 | 1.95 | 2.271 | 225.4 | 6.259% | False |
| fused_aos_minor | 1024 | 1 | 1.95 | 4.506 | 227.2 | 6.259% | False |
| fused_aos_minor | 2048 | 1 | 1.95 | 9.02 | 227 | 6.259% | False |
| fused_aos_minor | 4096 | 1 | 1.95 | 18.12 | 226 | 6.259% | False |
| fused_aos_minor | 128 | 1 | 0.977 | 1.142 | 112.1 | 0.3948% | True |
| fused_aos_minor | 512 | 1 | 0.977 | 4.473 | 114.5 | 0.3948% | True |
| fused_aos_minor | 1024 | 1 | 0.977 | 8.877 | 115.3 | 0.3948% | True |
| fused_aos_minor | 2048 | 1 | 0.977 | 17.77 | 115.3 | 0.3948% | True |
| fused_aos_minor | 4096 | 1 | 0.977 | 35.63 | 115 | 0.3948% | True |
| fused_aos_minor | 128 | 1 | 0.488 | 2.266 | 56.48 | 0.02477% | True |
| fused_aos_minor | 512 | 1 | 0.488 | 8.877 | 57.68 | 0.02477% | True |
| fused_aos_minor | 1024 | 1 | 0.488 | 17.62 | 58.12 | 0.02477% | True |
| fused_aos_minor | 2048 | 1 | 0.488 | 35.26 | 58.08 | 0.02477% | True |
| fused_aos_minor | 4096 | 1 | 0.488 | 70.72 | 57.92 | 0.02477% | True |
| staged_soa_minor | 512 | 5 | 0.977 | 15.27 | 33.52 | 0.6916% | True |
| staged_soa_minor | 1024 | 5 | 0.977 | 32.14 | 31.87 | 0.6916% | True |
| staged_soa_minor | 2048 | 5 | 0.977 | 65.45 | 31.29 | 0.6916% | True |
| staged_soa_minor | 4096 | 5 | 0.977 | 146.9 | 27.89 | 0.6916% | True |
| staged_soa_minor | 512 | 5 | 0.488 | 30.54 | 16.76 | 0.04467% | True |
| staged_soa_minor | 1024 | 5 | 0.488 | 64.23 | 15.94 | 0.04467% | True |
| staged_soa_minor | 2048 | 5 | 0.488 | 130.5 | 15.69 | 0.04467% | True |
| staged_soa_major | 512 | 5 | 0.977 | 16.14 | 31.72 | 0.6916% | True |
| staged_soa_major | 1024 | 5 | 0.977 | 32.6 | 31.41 | 0.6916% | True |
| staged_soa_major | 2048 | 5 | 0.977 | 65.04 | 31.49 | 0.6916% | True |
| staged_soa_major | 4096 | 5 | 0.977 | 130 | 31.51 | 0.6916% | True |
| staged_soa_major | 512 | 5 | 0.488 | 32.19 | 15.91 | 0.04467% | True |
| staged_soa_major | 1024 | 5 | 0.488 | 65 | 15.75 | 0.04467% | True |
| staged_soa_major | 2048 | 5 | 0.488 | 129.7 | 15.79 | 0.04467% | True |
| staged_soa_minor | 128 | 10 | 0.977 | 7.128 | 17.96 | 0.05021% | True |
| staged_soa_minor | 512 | 10 | 0.977 | 30.49 | 16.79 | 0.05021% | True |
| staged_soa_minor | 1024 | 10 | 0.977 | 64.13 | 15.97 | 0.05021% | True |
| staged_soa_major | 128 | 10 | 0.977 | 7.834 | 16.34 | 0.05021% | True |
| staged_soa_major | 512 | 10 | 0.977 | 32.14 | 15.93 | 0.05021% | True |
| staged_soa_major | 1024 | 10 | 0.977 | 64.89 | 15.78 | 0.05021% | True |

## CPU measurements

| Threads | Count | Interval (us) | atol | Wall times (s) |
|---:|---:|---:|---:|---|
| 1 | 483 | 1 | 1e-12 | [108.5147] |
| 12 | 128 | 1 | 1e-10 | [4.1328, 3.8735, 4.0094] |
| 12 | 512 | 1 | 1e-10 | [15.5665, 14.7697, 14.8009] |
| 12 | 1024 | 1 | 1e-10 | [30.0067, 28.7222, 28.4268] |
| 1 | 483 | 5 | 1e-12 | [530.3544] |
| 1 | 12 | 10 | 1e-12 | [26.754] |
| 1 | 1 | 293.3 | 1e-12 | [38.5359] |
| 1 | 1 | 880 | 1e-12 | [77.2364] |
| 1 | 1 | 176 | 1e-12 | [26.5176] |

## Full-transit checks

- staged_soa_minor, v=150.0 m/s, dt=0.9766 ns: 30 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.001961% above cutoff.
- staged_soa_minor, v=150.0 m/s, dt=0.4883 ns: 59.1 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.001933% above cutoff.
  Difference from coarser GPU timestep across all nine cases/endpoints: 0.2066% above cutoff; this is convergence evidence, not a stricter Rust reference for all cases.
- staged_soa_major, v=150.0 m/s, dt=0.9766 ns: 32.7 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.001961% above cutoff.
- staged_soa_major, v=150.0 m/s, dt=0.4883 ns: 64.4 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.001933% above cutoff.
  Difference from coarser GPU timestep across all nine cases/endpoints: 0.2066% above cutoff; this is convergence evidence, not a stricter Rust reference for all cases.
- staged_soa_minor, v=50.0 m/s, dt=0.9766 ns: 87.6 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 2.936e-07% above cutoff.
- staged_soa_minor, v=50.0 m/s, dt=0.4883 ns: 175 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 2.8e-07% above cutoff.
  Difference from coarser GPU timestep across all nine cases/endpoints: 0.2713% above cutoff; this is convergence evidence, not a stricter Rust reference for all cases.
- staged_soa_major, v=50.0 m/s, dt=0.9766 ns: 95.7 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 2.936e-07% above cutoff.
- staged_soa_major, v=50.0 m/s, dt=0.4883 ns: 191 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 2.8e-07% above cutoff.
  Difference from coarser GPU timestep across all nine cases/endpoints: 0.2713% above cutoff; this is convergence evidence, not a stricter Rust reference for all cases.
- staged_soa_minor, v=250.0 m/s, dt=0.9766 ns: 18.5 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.002016% above cutoff.
- staged_soa_minor, v=250.0 m/s, dt=0.4883 ns: 35.9 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.001793% above cutoff.
  Difference from coarser GPU timestep across all nine cases/endpoints: 0.09208% above cutoff; this is convergence evidence, not a stricter Rust reference for all cases.
- staged_soa_major, v=250.0 m/s, dt=0.9766 ns: 20.2 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.002016% above cutoff.
- staged_soa_major, v=250.0 m/s, dt=0.4883 ns: 39.3 s including 201 population samples and both density endpoints.
  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error 0.001793% above cutoff.
  Difference from coarser GPU timestep across all nine cases/endpoints: 0.09208% above cutoff; this is convergence evidence, not a stricter Rust reference for all cases.

All numerical arrays, per-trajectory diagnostics, failed-method logs, setup and memory measurements are retained alongside results.json.
A short-interval pass does not qualify a production backend or a full scan. No production backend change is made.
