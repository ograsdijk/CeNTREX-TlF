# Nsight Compute: cleanup-1 specialized derivative

Captured with Nsight Compute 2026.3.1 on the RTX 5070 Ti after the user enabled
counter access. This profiles one warmed stage-1 fused FP64 derivative launch:
1,024 trajectories, 257 density coordinates, 128 threads/block, grid (8,257).
The one-group wrapper retains the original coordinate order and expressions.

## Measurements

| Metric | Value |
|---|---:|
| Kernel duration | 85.47 us |
| FP64 pipeline utilization | 79.50% |
| DRAM utilization | 21.53% |
| L2 throughput utilization | 33.12% |
| L2 hit rate | 98.19% |
| Achieved occupancy | 66.74% |
| Theoretical occupancy | 75.00% |
| Registers/thread | 56 |
| Local/shared spilling requests | 0 / 0 |
| Branch efficiency | 100% |
| Executed FP64 FMA instructions | 324,576 |
| Executed non-fused FP64 instructions | 567,648 |

The strongest measured constraint is FP64 instruction execution. The profile
does not support prioritizing DRAM-bandwidth tuning, divergence fixes or spill
reduction for this launch. Occupancy and wave tails can still affect performance;
they are not proved irrelevant by this single profile.

Nsight reports substantial `tex_throttle` stalls. This is not evidence that the
simulation uses texture lookups: on architectures with decoupled math units,
the TEX pipeline can also forward FP64 instructions. See NVIDIA's
[pipeline and stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

Nsight also identifies opportunities to replace separate FP64 multiplies/adds
with FMA instructions. Its estimated speedups are heuristic, not benchmark
results, and are not added together here.

## Guided experiment

The existing generated complex accumulation has the form:

```cpp
ar += r*u - w*v;
ai += r*v + w*u;
```

The benchmark prototype explicitly accumulates each product with FP64 FMA:

```cpp
ar = fma(r, u, ar);
ar = fma(-w, v, ar);
ai = fma(r, v, ai);
ai = fma(w, u, ai);
```

This retains FP64 states, coefficients, diagonal evolution, decay and RK stages.
It changes floating-point rounding and association, so it must pass same-grid
full-density checks. No fast-math flag, precision reduction, support cutoff,
timestep change or package implementation change is included.

## Guided experiment outcome

The [explicit FP64 FMA benchmark](../cuda_fma_results/REPORT.md) passed full-flight
and dark-tail density checks within 3.56e-15 and reduced runtime by a median
30.09% for full batches and 31.04% for partial batches against the tuned
specialized baseline. All six pairs favor FMA for each shape. Register use
fell from 56 to 48/thread. These are measured benchmark results; they are
separate from Nsight's heuristic estimates.

## Provenance and limits

- [Nsight binary report](ncu_20261021T0330186779547Z/derivative.ncu-repz)
- [All raw metrics](ncu_20261021T0330186779547Z/metrics.csv)
- [Full analysis](ncu_20261021T0330186779547Z/details.txt)
- [Console log](ncu_20261021T0330186779547Z/console.txt)
- [Target source/configuration](20261021T033021184926Z/results.json)
- [Rerun script](../profile_cuda_specialized_ncu.ps1)

Collection used 41 replay passes, `--set full`, `--profile-from-start off`,
`--launch-count 1`, `--cache-control none` and `--clock-control none`. The profiler
warns that uncontrolled caches/clocks can introduce variation. The reported
85.47 us agrees closely with the earlier repeated-graph stage-1 median of
85.66 us, but this is one launch/stage/shape, not a complete solver trace or
a precision-convergence study. All 93 package/native files in the target audit
remain unchanged.
