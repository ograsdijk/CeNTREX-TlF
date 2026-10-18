# CUDA prepared Hamiltonian assembly and structural coefficient sharing

Device: NVIDIA GeForce RTX 5070 Ti. Benchmark-only source transformations; production unchanged.

Current library fixed Tsit5, default waveform precomputation, packed FP64 density, tiled and direct RHS, batches 512/1024. RC uses the archived current-apparatus fixture at 150 m/s near the field center (146-147 us, 4.4642857 ns steps); R(2) uses its archived EOM fixture (0.125-1.125 us, 2 ns steps). All density coordinates are evolved. Distinct cases are repeated only to fill batches.

Six randomized interleaved rounds per configuration after compilation/capture/warmup. Timing covers graph evolution and transfer of the final internal packed density. Setup, compilation, graph capture, common-waveform precomputation and disk writes are excluded. The public solver is run first for each candidate and validated against archived strict references; timed graph replays are also compared with the current baseline.

Assembly precomputes exact nonzero coefficient contribution lists for each stored Hamiltonian entry and separates the static coefficient-1 term. No magnitude threshold. DAG sharing interns structurally identical scalar nodes across coefficients/compounds before the existing CUDA source renderer. Tests include both changes individually and combined.

Positive percentages mean reduced runtime. Brackets are descriptive 95% paired bootstrap intervals (10,000 resamples of six whole rounds), not hardware-portability guarantees.

| System | Batch | RHS | Baseline (s) | Assembly reduction | DAG reduction | Both reduction |
|---|---:|---|---:|---:|---:|---:|
| RC | 512 | tiled | 0.973168 | +0.66% [-0.48, +3.30] | -0.35% [-3.75, +4.64] | +0.87% [-2.39, +2.11] |
| RC | 512 | direct | 0.988344 | +0.95% [+0.75, +1.72] | +0.06% [-6.55, +0.20] | +0.97% [-5.40, +1.31] |
| RC | 1024 | tiled | 1.982834 | +0.72% [-1.31, +1.46] | -0.10% [-2.24, +3.03] | +2.55% [-2.98, +5.07] |
| RC | 1024 | direct | 1.934745 | +0.65% [-6.40, +6.28] | -0.06% [-5.75, +5.51] | +0.83% [-0.52, +4.55] |
| R2_EOM | 512 | tiled | 0.571304 | +0.63% [+0.43, +1.11] | +0.14% [-0.50, +0.43] | +0.98% [+0.80, +1.33] |
| R2_EOM | 512 | direct | 0.591315 | +0.30% [-0.32, +0.45] | -0.12% [-0.43, +0.18] | +0.65% [+0.42, +0.73] |
| R2_EOM | 1024 | tiled | 1.107323 | +0.67% [+0.35, +5.61] | -0.15% [-6.49, +9.56] | +0.26% [-4.85, +5.36] |
| R2_EOM | 1024 | direct | 1.247185 | -1.67% [-8.81, +0.28] | -0.49% [-5.20, +1.80] | -1.53% [-9.86, +0.36] |

## Validation and scope

- R2_EOM: all archived-reference entry/coherence, near-zero, trace and positivity checks passed. Maximum internal-coordinate difference from the current kernel: 0. Generated coefficient temporaries: {'base': 1, 'assembly': 1, 'dag': 1, 'assembly_dag': 1}.
- RC: all archived-reference entry/coherence, near-zero, trace and positivity checks passed. Maximum internal-coordinate difference from the current kernel: 0. Generated coefficient temporaries: {'base': 17, 'assembly': 17, 'dag': 16, 'assembly_dag': 16}.

The entry criterion is 1% relative above reference magnitude 1e-6, absolute 1e-6 below; trace error <1e-6 and minimum density eigenvalue >=-1e-7. These are short-interval performance tests, not a new validation of a full apparatus grid or full-transit speedup.

Common waveforms already run through precomputation, so eliminating scalar operations in those expressions chiefly affects precomputation, outside the evolution timings. This prototype does not benchmark precompute=False or claim structural sharing has no benefit for every possible parameter graph. The tested RC graph has only one redundant rendered temporary removed; the small scope limits potential gains.

## Conclusion

Prepared assembly gives roughly 0.6-1.0% paired median reductions for RC, and mostly sub-1% changes for R(2), with one R(2) configuration slower. Most intervals overlap zero. Broader DAG sharing has no consistent evolution benefit; it removes one rendered temporary in RC and none in R(2). Combining the changes does not deliver a robust gain across configurations. Keep the production CUDA implementation unchanged based on these measurements. There is no evidence here for a gain comparable to the recent CPU changes.

No CPU ODE timings rerun. Source transformations and generated CUDA files are retained in this directory; raw paired timings, setup statistics and complete density diagnostics: [results.json](results.json). Reproduce with the benchmark GPU environment: `benchmarks/.venv-gpu-bench/Scripts/python.exe benchmarks/check_cuda_prepared_assembly.py`, then run `benchmarks/report_cuda_prepared_assembly.py`.
