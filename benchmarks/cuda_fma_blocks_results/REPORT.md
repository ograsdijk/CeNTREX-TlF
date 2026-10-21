# FMA kernel: block-size retuning and hardware profile

Benchmark-only experiment, RTX 5070 Ti, 2026-10-21 UTC. The explicit FP64 FMA
specialized kernel retains stage fusion and diagonal reuse. Package source and
production defaults remain unchanged.

## Block-size result: keep 128 / 64 threads

The screen compares 64, 96, 128 and 256 threads on actual last full 1,024-row
and final partial 548-row production batches at 150 m/s. Three random-order
4 us trials per configuration are followed by six balanced adjacent 12 us pairs
against the strongest alternative. Every result is checked against the installed
solver (`rtol=1e-10`, `atol=1e-12`), with unchanged FP64 2 ns Tsit5 and waveforms.

| Batch | Current block | Alternative | Current median | Alternative median | Median paired reduction |
|---|---:|---:|---:|---:|---:|
| 1,024 | **128** | 96 | 2.6413 s | 2.6029 s | 0.86% |
| 548 | **64** | 96 | 1.6246 s | 1.6051 s | 1.72% |

The 96-thread alternative wins only **three of six pairs** for each shape.
Full-batch paired changes range from a 2.04% regression to a 4.05% improvement;
partial-batch changes range from a 9.05% regression to a 6.55% improvement.
These results do not establish a reliable additional gain. Separate medians and
median paired ratios differ because of time variation. The recommendation
requires all confirmation pairs to favor a challenger before replacing the
existing setting; neither challenger meets that rule.

Screen medians:

| Threads | Full 1,024 rows | Partial 548 rows |
|---|---:|---:|
| 64 | 0.9377 s | 0.5501 s |
| 96 | 0.9310 s | 0.5635 s |
| 128 | 0.9197 s | 0.5775 s |
| 256 | 0.9500 s | 0.6236 s |

All timed densities agree with the installed solver within 1.12e-15. All 93
package/native source files checked by provenance hashes remain unchanged.
This is a short-interval launch-configuration comparison, not a new full-flight
or timestep-convergence study. The FMA kernel's earlier full-flight and dark-tail
checks remain documented in the [FMA report](../cuda_fma_results/REPORT.md).

### Experimental 96-thread support

The public options API allows only 64/128/256/512 threads. The initial run
`20261021T034238258433Z` stopped on that allowlist before timing 96 threads.
The driver now uses a benchmark-only options subclass to allow the explicit
96-thread experiment while preserving validation of every other option and
leaving package files untouched. CUDA launches use generic ceil division and
96 is three whole warps. Density equivalence is checked for this configuration.
No package API extension is recommended from these results.

## FMA hardware profile

Nsight Compute 2026.3.1 collected one warmed stage-1 fused derivative launch
for each recommended shape, with 41 replay passes per launch.

| Metric | Full 1,024 rows / 128 threads | Partial 548 rows / 64 threads |
|---|---:|---:|
| Duration | 58.27 us | 33.54 us |
| FP64 pipeline utilization | **83.78%** | **81.56%** |
| DRAM utilization | 46.08% | 4.55% |
| L2 throughput utilization | 46.96% | 46.41% |
| L2 hit rate | 98.70% | 98.23% |
| Achieved occupancy | 73.10% | 65.34% |
| Theoretical occupancy | 83.33% | 83.33% |
| Registers/thread | 48 | 48 |
| Branch efficiency | 100% | 100% |
| Local/shared spills | 0 / 0 | 0 / 0 |

**FP64 instruction execution remains the main measured constraint** in both
launches. Memory utilization has risen in the full batch, but remains below
FP64 utilization. The large difference in DRAM utilization between shapes
should not be interpreted as an intrinsic scaling law: these are separate
warmed launches collected without controlling cache contents.

The dominant stall label remains `tex_throttle`, consistent with the decoupled
FP64 instruction path discussed in the [baseline profile](../cuda_kernel_groups_results/NSIGHT_PROFILE.md).
This kernel does not perform texture lookups. The profile does not identify
branch divergence or spilling as problems, and occupancy is improved from the
previous specialized kernel's 66.74% achieved / 75% theoretical full-batch values.

### Counter confirmation of the FMA change

For the same full-batch stage-1 operation, compared with the previous specialized
kernel (without explicit accumulation FMA):

| Executed opcode | Previous specialized | Explicit FMA |
|---|---:|---:|
| DFMA | 324,576 | 580,512 |
| DMUL | 288,000 | 32,064 |
| DADD | 279,648 | 23,712 |
| Total FP64 arithmetic instructions | 892,224 | 636,288 |

Total executed FP64 arithmetic instructions fall **28.69%**; separate DMUL/DADD
instructions fall **90.17%**. Counting each FMA as two arithmetic operations gives
identical totals (1,216,800) for both kernels: the gain comes from executing the
same arithmetic with fewer instructions. These opcode counts are Nsight SASS
instruction counts, not source-line counts.

The profiled full-batch duration falls from 85.47 us to 58.27 us (31.82%),
consistent with the earlier six-pair 30.09% solver runtime reduction. These
profiled durations are single-launch diagnostics, not a new paired performance
study. Clocks were approximately 3.00 GHz in the previous profile and 2.98 GHz
here. The maintained block settings do not add a proven runtime improvement
beyond the earlier FMA optimization.

Collection used `--set full --launch-count 2 --profile-from-start off`,
`--cache-control none --clock-control none`. Nsight warns that uncontrolled
caches/clocks can introduce variation. Both launches use the same FMA source
hash and 257 active density coordinates. This is not a full-solver trace.
All 93 package/native files in the profile's provenance audit are unchanged.

- [Nsight report](ncu_20261021T0347439472959Z/fma_derivatives.ncu-repz)
- [Counter summary and opcode instances](ncu_20261021T0347439472959Z/counter_summary.json)
- [All raw metrics](ncu_20261021T0347439472959Z/metrics.csv)
- [Full Nsight analysis](ncu_20261021T0347439472959Z/details.txt)
- [Profiler console](ncu_20261021T0347439472959Z/console.txt)
- [Profile source/configuration snapshot](20261021T034746285426Z/results.json)

## Artifacts and reproduction

- [Tuning driver](../bench_cuda_fma_blocks.py)
- [Tuning results](20261021T034433971062Z/results.json)
- [Profiling script](../profile_cuda_fma_tuned_ncu.ps1)
- [Counter extractor](../summarize_cuda_ncu.py)

Each run retains exact helper/driver snapshots, generated CUDA, source hashes,
physical configuration, raw solver statistics, initial-state and parameter
rows, and per-initial-state reference/candidate density arrays.

From CeNTREX-TlF with the SPB environment:

```powershell
$env:PYTHONIOENCODING='utf-8'
../CeNTREX-SPB-testing/.venv/Scripts/python.exe benchmarks/bench_cuda_fma_blocks.py
benchmarks/profile_cuda_fma_tuned_ncu.ps1 -TuningResults benchmarks/cuda_fma_blocks_results/20261021T034433971062Z/results.json
```
