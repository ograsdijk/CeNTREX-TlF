# Scan Speedup Candidates: Exact Propagator, Thread Scaling, Batch Worker Reuse

Produced 2026-07-13 by `benchmarks/bench_exact_propagator.py`,
`benchmarks/bench_scan_threads.py`, and `benchmarks/bench_batch_worker_reuse.py`.
System for Parts A/B is the r2-in-static-E-field system built identically to
`examples/lindblad/r2_peak_ratio_vs_z_polarization.ipynb` (via
`benchmarks/diagnose_step_size.py`: R2_F1_7o2_F3, Ez = 171.6 V/cm,
`retain_opposite_parity_levels=True`, `qn_compact=True`, 60 mW, T = 108.7 us,
38 states, packed-real dim 1444). Context: per
`step_size_diagnostics_results/step_size_diagnostics_report.md`, every dopri5
trajectory of this system costs ~24k oscillation-limited steps (~1.1 s),
independent of tolerance — and at fixed scan parameters its Liouvillian is
time-independent. Machine: 16 logical cores.

## Part A — exact-propagator feasibility (VERDICT: works, numerically clean)

The packed-real Liouvillian L (1444 x 1444, ~13.7k nnz) was extracted via the
exact-Jacobian probe (`create_lindblad_rhs_evaluator_py` +
`jacobian_packed_sparse_py`). Verified at both detunings:

- **Time-independence**: jacobian at t = 0 and t = 3.7e-5 s is bit-identical.
- **Linearity/parity**: `L @ packed_rho0` equals `rhs_packed_py(packed_rho0, 0)`
  to rtol 1e-12 (the system is purely linear — no affine part).
- Packed layout confirmed diagonal-first: populations are packed entries 0..37.

Eigendecomposition and analytic propagation
x(t) = V (exp(w t) * (V^-1 x0)), photon integral
Gamma * sum_exc Re[V (phi(w) * c)] with phi(w) = (exp(wT)-1)/w
(series fallback for |wT| ~ 0), against a dopri5 reference at
reltol 1e-9 / abstol 1e-11 (801 saveat points):

| quantity | detuning 0 MHz | detuning 25 MHz |
| --- | --- | --- |
| eig residual \|\|LV-Vw\|\|/\|\|L\|\| | 5.5e-15 | 5.9e-15 |
| cond(V) | 14.9 | 16.2 |
| populations max abs diff vs ref (801 pts) | 5.7e-11 | 7.1e-11 |
| max \|Im population\| (should be 0) | 1.5e-17 | 9.6e-18 |
| photon integral (analytic) | 1.00806440 | 0.89428221 |
| photon integral (ref, reltol 1e-9) | 1.00806459 | 0.89428229 |
| photon integral rel diff | 1.9e-7 | 9.2e-8 |

Despite the formally non-normal L with GHz-scale spectator diagonals,
**cond(V) is ~15** — the eigenbasis is essentially as good as orthogonal, and
the feared ill-conditioning does not occur. The 1e-7 photon-integral
difference is at the accumulation level of the *reference* (trapezoid
integral over adaptive steps), not evidence of analytic error; populations
agree to ~6e-11.

Wall times (per scan grid point):

| cost | detuning 0 MHz | detuning 25 MHz |
| --- | --- | --- |
| jacobian extract + densify | 0.083 s | 0.013 s |
| `np.linalg.eig(L)` | 0.53 s | 0.62 s |
| `lu_factor(V)` | 0.67 s | 0.44 s |
| **setup total (once per grid point)** | **1.28 s** | **1.07 s** |
| per initial state: c = V^-1 x0 (lu_solve) | 2.8 ms | 1.3 ms |
| per initial state: photon integral (final only) | 0.2 ms | 0.2 ms |
| per initial state: 801-point trajectory (exp + matmul) | 56 ms | 72 ms |
| dopri5 per trajectory, reltol 1e-7 (photon_integral/final, median of 3) | 1.132 s | 1.125 s |
| `scipy.linalg.expm(L*dt_save)` (once) | 0.93 s | 0.88 s |
| expm route: 800 repeated matvecs | 93 ms | 90 ms |

(cond(V) computation, 1.0-1.2 s, is diagnostic-only and not needed in
production. The expm route matches the reference to 2.6-3.4e-11 as well.)

**Verdict table:**

| scenario | exact propagator | dopri5 stepping | gain |
| --- | --- | --- | --- |
| 1 initial state, photon integral, per grid point | ~1.1-1.3 s (setup-dominated) | ~1.13 s | ~1x (break-even) |
| each *additional* initial state (same grid point) | ~1.5-3 ms | ~1.13 s | **~400-750x** |
| each additional trajectory *with* 801 saveat points | ~60-75 ms | ~1.2 s+ | **~15-20x** |
| marginal saveat point (vectorized exp+matmul) | ~70-90 us | free-ish while stepping | n/a |

Interpretation for the actual scan shape (~10 polarization fractions x ~120
detunings, one rho0): each (detuning, polarization) pair is a distinct L, so
the plain scan sees only break-even. The exact propagator pays off when
anything is scanned *per fixed L*: multiple initial states (velocity/state
distributions), time-resolved output, or tighter accuracy (it is exact —
dopri5 at reltol 1e-9 costs 1.2-1.4 s and is *less* accurate). Note also the
setup is eig-dominated at n=38 (dim 1444, ~1 s); it scales as O(n^6) with
state count, so this approach degrades quickly for larger systems, and the
1444-dim eig/LU here is single-call LAPACK — the 1.1-1.3 s setup could
likely be trimmed (e.g. `scipy.linalg.lu_factor` warm, threaded BLAS) if it
mattered. Accuracy does NOT kill the approach — it is clean.

## Part B — thread scaling on the real scan shape

`lindblad.grid_scan`, single prepared r2 system, 120 detunings (-5..30 MHz)
as a scan slot, `output="photon_integral"`, weights = Gamma on the 14 B-state
indices, `output_when="final"`, `dense_output=False`, reltol 1e-7 /
abstol 1e-9, dopri5/`expanded_sparse`. 2 repeats, median. 16 logical cores.
Scan results verified bit-identical across all thread counts.

| threads | wall (median) | runs | traj/s | speedup vs 1 | parallel efficiency |
| --- | --- | --- | --- | --- | --- |
| 1 | 156.7 s | 135.3, 178.2 | 0.77 | 1.00x | 100% |
| 2 | 83.5 s | 82.3, 84.8 | 1.44 | 1.88x | 94% |
| 4 | 60.4 s | 66.3, 54.5 | 1.99 | 2.59x | 65% |
| 8 | 30.5 s | 30.6, 30.5 | 3.93 | 5.13x | 64% |
| None (all 16) | 17.1 s | 16.9, 17.2 | 7.03 | 9.19x | 57% |

**Verdict: threads are the single biggest lever available today.** The
120-detuning scan drops from ~2.6 min to 17 s with `threads=None` (default
`parallel=True` already does this). Scaling is near-ideal to 2 threads, then
settles at ~57-65% efficiency — consistent with a 16-logical/8-physical-core
machine (SMT sharing) plus memory-bandwidth contention; the threads=1 and
threads=4 runs show 10-30% run-to-run spread from background load, the 8/None
runs are stable. A full 10 x 120 peak-ratio scan extrapolates to ~2.8 min
wall on this machine instead of ~26 min serial.

## Part C — batch worker reuse in `solve_batch_ode` (implemented)

`rust/src/lindblad/ode_batch.rs`: `solve_batch_ode` used to construct a fresh
`LindbladRhs` (with a new `RhsWorkspace`) and a fresh output accumulator per
trajectory, while `solve_grid_ode_direct` reuses per-thread `GridWorker`s via
rayon `map_init`. The same pattern is now applied to `solve_batch_ode`:
`GridWorker` gained `new_with_event(...)` (stop_event set once at worker
construction, equivalent to the previous per-trajectory clone) and
`solve_batch(...)` (sets the trajectory's scalar parameter overrides — same
slots each trajectory, so a full overwrite — resets the output, solves, and
snapshots). Serial path uses one worker; parallel paths use `map_init`.
`snapshot()` is value-identical to `finish()` for every output type.

Rebuilt with `maturin develop --release` into the repo `.venv` (python 3.11);
the `.pyd` was not file-locked. **`pytest tests -q`: 295 passed, 1 skipped**
(expected counts). Note the rebuild necessarily also compiled other
uncommitted in-progress working-tree changes (`rhs.rs`, `plan.rs`, ...) from
a concurrent workstream; the full suite passing covers both.

Throughput on 1200 cheap trajectories (2-level system from
`tests/lindblad/test_rust_backend.py`'s fixture pattern, 40 Omega x 30 delta,
t_span (0, 0.5), reltol 1e-8 / abstol 1e-10, photon_integral/final, median of
9 runs; `parameter_scan` exercises `solve_batch_ode`, `grid_scan` the
already-reusing direct path; outputs bit-identical between the two APIs):

| API | threads | before (median) | after (median) | change |
| --- | --- | --- | --- | --- |
| parameter_scan (batch path) | 1 | 14.26 ms (84k traj/s) | 10.40 ms (115k traj/s) | **1.37x faster** |
| parameter_scan (batch path) | None (16) | 1.61 ms (744k traj/s) | 1.36 ms (882k traj/s) | **1.18x faster** |
| grid_scan (reference) | 1 | 10.34 ms | 10.30 ms | unchanged |
| grid_scan (reference) | None (16) | 1.46 ms | 1.45 ms | unchanged |

**Verdict: the batch path's ~38% worker-churn penalty vs grid_scan is
eliminated** — `parameter_scan` now matches `grid_scan` throughput on cheap
trajectories (10.40 vs 10.30 ms serial). On expensive trajectories (like the
r2 system, ~1.1 s each) the per-trajectory allocation was already negligible,
so Parts A/B numbers are unaffected.

## Overall recommendation

1. **Use threads (Part B) unconditionally** — free ~9x on this machine for
   the real scan, already the default.
2. **Exact propagator (Part A) is validated and accurate** for this
   time-independent-L system class. Worth building as a scan backend
   primarily when scans reuse an L across many initial states or need dense
   time output; for the plain one-rho0 detuning scan it is break-even per
   grid point (~1.1-1.3 s setup vs ~1.13 s stepping) — and it composes with
   Part B (eigs of different grid points parallelize embarrassingly).
3. **Batch worker reuse (Part C) is done** — uncommitted, tests green,
   `parameter_scan` now on par with `grid_scan`.

Raw numbers: `exact_propagator.csv`, `scan_thread_scaling.csv`,
`batch_worker_reuse.csv` in this folder.
