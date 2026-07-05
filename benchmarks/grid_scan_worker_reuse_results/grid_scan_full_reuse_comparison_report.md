# Grid Scan Full Reuse Comparison

This report compares the saved RHS-only grid-worker baseline against the current implementation, where each grid worker reuses both its RHS workspace and output collector.

The baseline file is `grid_scan_rhs_only_baseline_summary.csv`. It was produced before adding reusable output collectors. The current file is `grid_scan_worker_reuse_summary.csv`.

## Output Reuse Delta

| case                  | threads | rhs_only_mean_seconds | full_reuse_mean_seconds | mean_speedup_from_output_reuse | full_reuse_median_seconds | full_reuse_trimmed_mean_seconds |
| --------------------- | ------- | --------------------- | ----------------------- | ------------------------------ | ------------------------- | ------------------------------- |
| final_populations     | 1       | 0.011698              | 0.00974206              | 1.20077                        | 0.0096424                 | 0.00968147                      |
| final_populations     | 4       | 0.00396436            | 0.00338442              | 1.17135                        | 0.0032492                 | 0.00332531                      |
| final_photon_integral | 1       | 0.011964              | 0.00996435              | 1.20068                        | 0.0098919                 | 0.00992026                      |
| final_photon_integral | 4       | 0.00421607            | 0.0044943               | 0.938092                       | 0.0038735                 | 0.00389378                      |
| saveat_selected       | 1       | 0.0138186             | 0.0115022               | 1.20139                        | 0.0113494                 | 0.0114027                       |
| saveat_selected       | 4       | 0.00447358            | 0.00498564              | 0.897292                       | 0.0047946                 | 0.00480317                      |

## Interpretation

- Best mean speedup from output reuse alone: `1.2x` for `saveat_selected` with `1` threads.
- Worst mean speedup from output reuse alone: `0.897x` for `saveat_selected` with `4` threads.
- These are small, allocation-level changes. The expected benefit is largest when trajectories are cheap and output allocation/collation is a visible fraction of runtime.
- For RHS-dominated molecular OBE solves, this should mainly reduce allocation pressure rather than produce a large wall-time change.

## Validation

- The benchmark still compares `grid_scan` values against equivalent flattened `parameter_scan` values with exact `np.testing.assert_allclose` checks.
