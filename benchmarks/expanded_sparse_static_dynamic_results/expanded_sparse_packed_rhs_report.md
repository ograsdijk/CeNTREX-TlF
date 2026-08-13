# Expanded-Sparse Packed RHS Kernel Benchmark

This report benchmarks the optimized static/dynamic `expanded_sparse` RHS kernel against retained legacy-packed and current split-input benchmark controls.

## Systems
| model | n_states | packed_len | H_nnz | C_ops | C_nnz | prep_seconds_not_timed |
| --- | --- | --- | --- | --- | --- | --- |
| two_level | 0 | 4 | 3 | 1 | 1 | 0.0042461 |
| r2_retained_opposite_parity | 38 | 1444 | 189 | 168 | 168 | 8.71251 |

## RHS-Only Timing
| model | variant | trimmed_mean_seconds_per_call | rhs_calls_per_second | rust_commutator_seconds_per_call |
| --- | --- | --- | --- | --- |
| two_level | legacy_packed | 7.796e-07 | 1.28271e+06 | 4.34725e-08 |
| two_level | current_split_inputs | 7.11192e-07 | 1.40609e+06 | 4.324e-08 |
| two_level | static_dynamic | 5.92592e-07 | 1.6875e+06 | 5.124e-08 |
| r2_retained_opposite_parity | legacy_packed | 1.14168e-05 | 87590.2 | 9.41225e-06 |
| r2_retained_opposite_parity | current_split_inputs | 9.88e-06 | 101215 | 9.62125e-06 |
| r2_retained_opposite_parity | static_dynamic | 8.7078e-06 | 114840 | 7.51388e-06 |

## Single-Trajectory Solve Timing
| model | variant | trimmed_mean_seconds | accepted_steps | rhs_calls |
| --- | --- | --- | --- | --- |
| two_level | legacy_packed | 3.382e-05 | 10 | 61 |
| two_level | current_split_inputs | 2.812e-05 | 10 | 61 |
| two_level | static_dynamic | 3.154e-05 | 10 | 61 |
| r2_retained_opposite_parity | legacy_packed | 1.49438 | 28651 | 171907 |
| r2_retained_opposite_parity | current_split_inputs | 1.54584 | 28651 | 171907 |
| r2_retained_opposite_parity | static_dynamic | 1.52177 | 28651 | 171907 |

## Grid Timing
| model | variant | trajectory_count | threads | trimmed_mean_seconds | trajectories_per_second | rhs_calls |
| --- | --- | --- | --- | --- | --- | --- |
| two_level | legacy_packed | 25 | 4 | 0.000709267 | 35247.7 | 1537 |
| two_level | current_split_inputs | 25 | 4 | 0.000527 | 47438.3 | 1537 |
| two_level | static_dynamic | 25 | 4 | 0.000715233 | 34953.6 | 1537 |
| r2_retained_opposite_parity | legacy_packed | 9 | 4 | 6.50366 | 1.38384 | 1976397 |
| r2_retained_opposite_parity | current_split_inputs | 9 | 4 | 6.3305 | 1.42169 | 1976397 |
| r2_retained_opposite_parity | static_dynamic | 9 | 4 | 5.96746 | 1.50818 | 1976397 |

## Validation
| model | variant | check | max_abs_diff | max_rel_diff | value | baseline_value | peak_axis_shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_level | legacy_packed | rhs_vector | 0 | 0 | nan | nan | nan |
| two_level | current_split_inputs | rhs_vector | 0 | 0 | nan | nan | nan |
| two_level | static_dynamic | rhs_vector | 0 | 0 | nan | nan | nan |
| two_level | legacy_packed | single_photon_integral | 1.17616e-14 | 1.10115e-11 | 0.00106813 | 0.00106813 | nan |
| two_level | current_split_inputs | single_photon_integral | 0 | 0 | 0.00106813 | 0.00106813 | nan |
| two_level | static_dynamic | single_photon_integral | 0 | 0 | 0.00106813 | 0.00106813 | nan |
| two_level | legacy_packed | normalized_scan | 5.86556e-10 | 5.75545e-10 | nan | nan | 0 |
| two_level | current_split_inputs | normalized_scan | 5.6842e-10 | 5.6842e-10 | nan | nan | 0 |
| two_level | static_dynamic | normalized_scan | 0 | 0 | nan | nan | 0 |
| r2_retained_opposite_parity | legacy_packed | rhs_vector | 0 | 0 | nan | nan | nan |
| r2_retained_opposite_parity | current_split_inputs | rhs_vector | 0 | 0 | nan | nan | nan |
| r2_retained_opposite_parity | static_dynamic | rhs_vector | 0 | 0 | nan | nan | nan |
| r2_retained_opposite_parity | legacy_packed | single_photon_integral | 1.9984e-14 | 1.64768e-14 | 1.21286 | 1.21286 | nan |
| r2_retained_opposite_parity | current_split_inputs | single_photon_integral | 0 | 0 | 1.21286 | 1.21286 | nan |
| r2_retained_opposite_parity | static_dynamic | single_photon_integral | 0 | 0 | 1.21286 | 1.21286 | nan |
| r2_retained_opposite_parity | legacy_packed | normalized_scan | 5.88418e-15 | 1.64768e-14 | nan | nan | 0 |
| r2_retained_opposite_parity | current_split_inputs | normalized_scan | 3.66374e-15 | 3.66152e-15 | nan | nan | 0 |
| r2_retained_opposite_parity | static_dynamic | normalized_scan | 0 | 0 | nan | nan | 0 |

## Recommendation Signals
| model | variant | speedup_vs_baseline_rhs | speedup_vs_baseline_single_solve | speedup_vs_baseline_grid | risk_complexity | worth_implementing_signal |
| --- | --- | --- | --- | --- | --- | --- |
| two_level | legacy_packed | 1 | 1 | 1 | none | weak |
| two_level | current_split_inputs | 1.09619 | 1.2027 | 1.34586 | medium | weak |
| two_level | static_dynamic | 1.31558 | 1.07229 | 0.991658 | medium | candidate |
| r2_retained_opposite_parity | legacy_packed | 1 | 1 | 1 | none | weak |
| r2_retained_opposite_parity | current_split_inputs | 1.15555 | 0.96671 | 1.02735 | medium | candidate |
| r2_retained_opposite_parity | static_dynamic | 1.3111 | 0.982001 | 1.08985 | medium | candidate |

Raw CSV files are written next to this report.