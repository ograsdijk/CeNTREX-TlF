# Static/Dynamic Kernel Full-Scan Results

These files contain four counterbalanced runs of the legacy packed, previous
split-input, and fully partitioned static/dynamic kernels.

- Compact q1 and r2 plans use the partitioned kernel in production and show
  mean paired speedups of 3.8% and 6.3% over the previous kernel.
- Noncompact q1 and r2 runs exposed 11.8% and 8.4% regressions in the fully
  partitioned candidate. Those results are intentionally retained here.
- Production `expanded_sparse` therefore uses the partitioned layout through
  40 states and the previous split-input layout for larger plans. The latter
  is the exact same Rust function as the `current_split_inputs` benchmark
  control, rather than a second reimplementation.
- [2026-07-12] A follow-up checked whether the 40-state cutoff could be
  replaced with a gate on the partitioned layout's term count / byte
  footprint (measured across these same four systems plus the two-level and
  38-state r2 systems from `bench_expanded_sparse_packed_rhs_kernels.py`).
  It does not separate cleanly: noncompact q1 (a loser, term count 3,812)
  has a smaller partitioned footprint than 38-state r2 (a winner, term count
  6,813), because noncompact retention inflates `upper_len` without adding
  proportionally many coupling terms. The 40-state gate was kept; see
  `IMPLEMENTATION_AUDIT.md` item 7 for the full measurement.

Peak detunings are unchanged. Maximum first-comparison photon differences are
4.25e-10 for noncompact q1 and 8.77e-15 for compact r2.
