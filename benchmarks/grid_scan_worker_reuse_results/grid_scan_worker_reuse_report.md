# Grid Scan Worker Reuse Benchmark

This benchmark compares the optimized `grid_scan` path against the current generic `parameter_scan` batch path with an equivalent flattened two-parameter grid.

The optimized grid path reuses one Rust RHS workspace and one output collector per worker, and writes fixed-size grid outputs directly into preallocated result storage. The generic parameter-scan path still constructs per-trajectory RHS/output objects and collates results afterward, so it is a practical proxy for the old grid behavior.

## Setup

- Model: two-level Lindblad system with one decay channel.
- Grid: `20 x 20 = 400` trajectories.
- Repeats per method: `21`.
- Solvers: Rust `dopri5`, `expanded_sparse` execution.
- Cases: final populations, final photon integral with `saveat=None`, and selected saveat trace.
- Mean, median, and 10% trimmed mean are reported because short threaded runs can have scheduler outliers.

## Results

| case                  | threads | trajectory_count | optimized_mean_seconds | optimized_median_seconds | optimized_trimmed_mean_seconds | optimized_stdev_seconds | generic_mean_seconds | generic_median_seconds | generic_trimmed_mean_seconds | generic_stdev_seconds | speedup | median_speedup | trimmed_mean_speedup | optimized_trajectories_per_second | generic_trajectories_per_second |
| --------------------- | ------- | ---------------- | ---------------------- | ------------------------ | ------------------------------ | ----------------------- | -------------------- | ---------------------- | ---------------------------- | --------------------- | ------- | -------------- | -------------------- | --------------------------------- | ------------------------------- |
| final_populations     | 1       | 400              | 0.00974206             | 0.0096424                | 0.00968147                     | 0.000261199             | 0.0103988            | 0.0103507              | 0.0103778                    | 0.000135143           | 1.06741 | 1.07346        | 1.07192              | 41059.1                           | 38466                           |
| final_populations     | 4       | 400              | 0.00338442             | 0.0032492                | 0.00332531                     | 0.000321703             | 0.00379609           | 0.003748               | 0.00378843                   | 0.000422616           | 1.12164 | 1.15351        | 1.13927              | 118189                            | 105372                          |
| final_photon_integral | 1       | 400              | 0.00996435             | 0.0098919                | 0.00992026                     | 0.000255449             | 0.0105741            | 0.0105563              | 0.0105513                    | 0.000162384           | 1.06119 | 1.06717        | 1.06361              | 40143.1                           | 37828.3                         |
| final_photon_integral | 4       | 400              | 0.0044943              | 0.0038735                | 0.00389378                     | 0.00240531              | 0.00859602           | 0.0035835              | 0.00361506                   | 0.0227793             | 1.91265 | 0.925132       | 0.92842              | 89001.6                           | 46533.1                         |
| saveat_selected       | 1       | 400              | 0.0115022              | 0.0113494                | 0.0114027                      | 0.000460279             | 0.0131988            | 0.0121103              | 0.0128719                    | 0.00201806            | 1.1475  | 1.06704        | 1.12884              | 34775.9                           | 30305.7                         |
| saveat_selected       | 4       | 400              | 0.00498564             | 0.0047946                | 0.00480317                     | 0.000850088             | 0.00512876           | 0.0050131              | 0.00509364                   | 0.000388073           | 1.02871 | 1.04557        | 1.06047              | 80230.4                           | 77991.5                         |

## Validation

| case                  | threads | max_abs_value_difference |
| --------------------- | ------- | ------------------------ |
| final_populations     | 1       | 0                        |
| final_populations     | 4       | 0                        |
| final_photon_integral | 1       | 0                        |
| final_photon_integral | 4       | 0                        |
| saveat_selected       | 1       | 0                        |
| saveat_selected       | 4       | 0                        |

## Summary

- Best observed trimmed-mean speedup: `1.14x` for `final_populations` with `4` threads.
- Smallest observed trimmed-mean speedup: `0.928x` for `final_photon_integral` with `4` threads.
- These numbers mainly measure grid orchestration overhead. Larger molecular OBE systems are still dominated by RHS evaluations, so the relative speedup there should be smaller, but memory traffic and allocation pressure should still improve.

- Raw timings: `grid_scan_worker_reuse_timings.csv`.
- Summary CSV: `grid_scan_worker_reuse_summary.csv`.
