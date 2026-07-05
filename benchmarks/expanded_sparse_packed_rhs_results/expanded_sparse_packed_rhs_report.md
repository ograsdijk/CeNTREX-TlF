# Expanded-Sparse Packed RHS Kernel Benchmark

This report benchmarks experimental packed `expanded_sparse` RHS kernels. The default `expanded_sparse` implementation is the baseline; experimental modes are hidden and benchmark-only.

## Systems
| model | n_states | packed_len | H_nnz | C_ops | C_nnz | prep_seconds_not_timed |
| --- | --- | --- | --- | --- | --- | --- |
| two_level | 0 | 4 | 3 | 1 | 1 | 0.0069518 |
| r2_retained_opposite_parity | 38 | 1444 | 189 | 168 | 168 | 12.6221 |

## RHS-Only Timing
| model | variant | trimmed_mean_seconds_per_call | rhs_calls_per_second | rust_commutator_seconds_per_call |
| --- | --- | --- | --- | --- |
| two_level | baseline | 4.26778e-06 | 234314 | 2.42117e-07 |
| two_level | split_coefficients | 4.1927e-06 | 238510 | 2.3514e-07 |
| two_level | split_inputs | 4.54423e-06 | 220059 | 2.86947e-07 |
| two_level | precomputed_inputs | 4.34359e-06 | 230224 | 3.57047e-07 |
| r2_retained_opposite_parity | baseline | 0.000105777 | 9453.85 | 9.34198e-05 |
| r2_retained_opposite_parity | split_coefficients | 0.000105219 | 9503.96 | 9.44818e-05 |
| r2_retained_opposite_parity | split_inputs | 8.4893e-05 | 11779.5 | 7.56863e-05 |
| r2_retained_opposite_parity | precomputed_inputs | 0.000169286 | 5907.17 | 0.000159186 |

## Single-Trajectory Solve Timing
| model | variant | trimmed_mean_seconds | accepted_steps | rhs_calls |
| --- | --- | --- | --- | --- |
| two_level | baseline | 0.00014124 | 10 | 61 |
| two_level | split_coefficients | 0.00010056 | 10 | 61 |
| two_level | split_inputs | 5.772e-05 | 10 | 61 |
| two_level | precomputed_inputs | 6.052e-05 | 10 | 61 |
| r2_retained_opposite_parity | baseline | 18.6181 | 28651 | 171907 |
| r2_retained_opposite_parity | split_coefficients | 19.4237 | 28651 | 171907 |
| r2_retained_opposite_parity | split_inputs | 16.604 | 28651 | 171907 |
| r2_retained_opposite_parity | precomputed_inputs | 30.4306 | 28651 | 171907 |

## Grid Timing
| model | variant | trajectory_count | threads | trimmed_mean_seconds | trajectories_per_second | rhs_calls |
| --- | --- | --- | --- | --- | --- | --- |
| two_level | baseline | 25 | 4 | 0.0008126 | 30765.4 | 1537 |
| two_level | split_coefficients | 25 | 4 | 0.000757467 | 33004.8 | 1537 |
| two_level | split_inputs | 25 | 4 | 0.000811033 | 30824.9 | 1537 |
| two_level | precomputed_inputs | 25 | 4 | 0.000832733 | 30021.6 | 1537 |
| r2_retained_opposite_parity | baseline | 9 | 4 | 73.1375 | 0.123056 | 1976397 |
| r2_retained_opposite_parity | split_coefficients | 9 | 4 | 65.5588 | 0.137281 | 1976397 |
| r2_retained_opposite_parity | split_inputs | 9 | 4 | 56.7991 | 0.158453 | 1976397 |
| r2_retained_opposite_parity | precomputed_inputs | 9 | 4 | 102.541 | 0.0877696 | 1976397 |

## Validation
| model | variant | check | max_abs_diff | max_rel_diff | value | baseline_value | peak_axis_shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_level | baseline | rhs_vector | 0 | 0 | nan | nan | nan |
| two_level | split_coefficients | rhs_vector | 0 | 0 | nan | nan | nan |
| two_level | split_inputs | rhs_vector | 0 | 0 | nan | nan | nan |
| two_level | precomputed_inputs | rhs_vector | 0 | 0 | nan | nan | nan |
| two_level | baseline | single_photon_integral | 0 | 0 | 0.00106813 | 0.00106813 | nan |
| two_level | split_coefficients | single_photon_integral | 0 | 0 | 0.00106813 | 0.00106813 | nan |
| two_level | split_inputs | single_photon_integral | 1.17616e-14 | 1.10115e-11 | 0.00106813 | 0.00106813 | nan |
| two_level | precomputed_inputs | single_photon_integral | 1.17616e-14 | 1.10115e-11 | 0.00106813 | 0.00106813 | nan |
| two_level | baseline | normalized_scan | 0 | 0 | nan | nan | 0 |
| two_level | split_coefficients | normalized_scan | 0 | 0 | nan | nan | 0 |
| two_level | split_inputs | normalized_scan | 2.03487e-11 | 1.10115e-11 | nan | nan | 0 |
| two_level | precomputed_inputs | normalized_scan | 2.03487e-11 | 1.10115e-11 | nan | nan | 0 |
| r2_retained_opposite_parity | baseline | rhs_vector | 0 | 0 | nan | nan | nan |
| r2_retained_opposite_parity | split_coefficients | rhs_vector | 0 | 0 | nan | nan | nan |
| r2_retained_opposite_parity | split_inputs | rhs_vector | 0 | 0 | nan | nan | nan |
| r2_retained_opposite_parity | precomputed_inputs | rhs_vector | 0 | 0 | nan | nan | nan |
| r2_retained_opposite_parity | baseline | single_photon_integral | 0 | 0 | 1.21286 | 1.21286 | nan |
| r2_retained_opposite_parity | split_coefficients | single_photon_integral | 0 | 0 | 1.21286 | 1.21286 | nan |
| r2_retained_opposite_parity | split_inputs | single_photon_integral | 1.9984e-14 | 1.64768e-14 | 1.21286 | 1.21286 | nan |
| r2_retained_opposite_parity | precomputed_inputs | single_photon_integral | 1.9984e-14 | 1.64768e-14 | 1.21286 | 1.21286 | nan |
| r2_retained_opposite_parity | baseline | normalized_scan | 0 | 0 | nan | nan | 0 |
| r2_retained_opposite_parity | split_coefficients | normalized_scan | 0 | 0 | nan | nan | 0 |
| r2_retained_opposite_parity | split_inputs | normalized_scan | 5.71765e-15 | 1.64768e-14 | nan | nan | 0 |
| r2_retained_opposite_parity | precomputed_inputs | normalized_scan | 5.71765e-15 | 1.64768e-14 | nan | nan | 0 |

## Recommendation Signals
| model | variant | speedup_vs_baseline_rhs | speedup_vs_baseline_single_solve | speedup_vs_baseline_grid | risk_complexity | worth_implementing_signal |
| --- | --- | --- | --- | --- | --- | --- |
| two_level | baseline | 1 | 1 | 1 | none | weak |
| two_level | split_coefficients | 1.01791 | 1.40453 | 1.07279 | low | weak |
| two_level | split_inputs | 0.939164 | 2.44699 | 1.00193 | medium | weak |
| two_level | precomputed_inputs | 0.982545 | 2.33377 | 0.975823 | medium | weak |
| r2_retained_opposite_parity | baseline | 1 | 1 | 1 | none | weak |
| r2_retained_opposite_parity | split_coefficients | 1.0053 | 0.958523 | 1.1156 | low | weak |
| r2_retained_opposite_parity | split_inputs | 1.246 | 1.1213 | 1.28765 | medium | candidate |
| r2_retained_opposite_parity | precomputed_inputs | 0.624843 | 0.61182 | 0.71325 | medium | weak |

Raw CSV files are written next to this report.