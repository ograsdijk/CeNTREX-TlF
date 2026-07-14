# Implementation Audit

This file replaces the older audit and roadmap notes:

- `CODEBASE_AUDIT.md`
- `IMPROVEMENTS.md`
- `RUST_HAMILTONIAN_REVIEW.md`
- `rust_obe_rhs_solver_roadmap.md`

No file named `RUST_HAMILTONIAN_OVERVIEW.md` was present in the workspace during
this audit. `RUST_HAMILTONIAN_REVIEW.md` was used as the matching Rust
Hamiltonian audit document.

## Current Test Snapshot

Command run:

```powershell
uv run pytest tests\states tests\lindblad -q
```

Result:

- 125 passed
- 0 failed
- 1 skipped

The old partial-state hashing failures from `CODEBASE_AUDIT.md` did not recur,
and the Rust Lindblad API/wrapper failures found during the audit have been
fixed.

The fixed Rust Lindblad issues were:

- `centrex_tlf/lindblad/batch.py` calls Rust batch/grid functions with stale
  positional arguments after `integral_weights` was added to the PyO3
  signatures.
- Rust solver stats expose `solver`, `accepted_steps`, `rejected_steps`, and
  `rhs_calls`, while tests expect `function_evaluations`.
- Native Rust solvers return stats names such as `dopri5` and `tsit5`, and
  batch/grid stats use the same canonical solver names.
- Some final-output reduced result shapes are now `(1, n)` while tests expect
  `(n,)`.

Additional command run:

```powershell
uv run pytest tests\hamiltonian\test_edge_cases.py tests\lindblad\test_rust_backend.py -q
```

Result:

- 44 passed
- 0 failed

Broader command run:

```powershell
uv run pytest tests\hamiltonian tests\lindblad -q
```

Result:

- 134 passed
- 0 failed
- 1 skipped

Reduced-Hamiltonian fixture command run:

```powershell
uv run pytest tests\hamiltonian\test_reduced_hamiltonian.py -q
```

Result:

- 4 passed
- 0 failed

Rust-vs-Python Hamiltonian command run:

```powershell
uv run pytest tests\hamiltonian\test_rust_vs_python.py -q
```

Result:

- 42 passed
- 0 failed

Rust unit command run:

```powershell
uv run cargo test --manifest-path rust\Cargo.toml -q
```

Result:

- 57 passed
- 0 failed

`cargo test --manifest-path rust\Cargo.toml -q` still does not run standalone
because PyO3 cannot find a Python 3 interpreter outside the `uv` environment.

## Implemented

### General Codebase Audit Items

- Partial-state hashing now supports optional values. Helpers such as
  `_optional_int_hash_value()` and `_optional_half_int_hash_value()` handle
  `None` in `centrex_tlf/states/states.py`.
- `generate_coupling_field(..., pol_vecs=None)` now normalizes `None` to an
  empty list before indexing in `centrex_tlf/couplings/coupling_matrix.py`.
- `state_string_custom()` no longer uses `eval()` in
  `centrex_tlf/states/states.py`.
- `matrix_to_states()` now checks `QN[0]` for both coupled and uncoupled bases,
  and rejects an empty basis explicitly.

### Rust Lindblad / OBE Backend

The roadmap's core Rust backend has been implemented and extended beyond the
original static-only scope:

- Python `PackedHermitianLayout` and Rust packed Hermitian layout.
- Python `StaticLindbladPlan` preparation in `centrex_tlf/lindblad/plan_static.py`.
- Rust packed RHS and JVP entry points.
- Reference, structured, `structured_upper`, and `expanded_sparse` execution
  modes.
- Packed scipy BDF/Radau path using Rust RHS/Jacobian probing.
- DOPRI5 and Tsit5 Rust explicit solvers.
- Generic `rust/src/ode/` module with `OdeRhs`, `OdeOutput`, DOPRI5, Tsit5,
  dense output, save plans, and output extraction.
- Batch and grid solve plumbing for Lindblad trajectories.
- Weighted-integral output support.
- Time-independent Hamiltonian caching in the Rust RHS workspace.
- Sparse-H commutator for `structured_upper`.
- Precomputed dense `C dagger C` for reference dissipator mode.
- IR helper extensions including `gaussian_1d`, `pchip_interp`, and
  `square_wave`.
- Precomputed PCHIP tables and RHS-side PCHIP interval hints.
- RuntimeExpression `.evaluate()` and `.evaluate_array()` support.
- RWA manifold rotation propagation via `_build_manifold_indices()`.
- Instruction opcode discriminant safety with `#[repr(u8)]` and tests.

### Effective Hamiltonian Backend

- Production effective-Hamiltonian code has been extracted into
  `centrex_tlf/effective_hamiltonian/`.
- Rust effective Lindblad solver exists under `rust/src/effective_lindblad/`.
- Effective solver uses the generic ODE module.
- Time-dependent field/coordinate path is supported with operator-grid bounds
  checking.
- Effective batch solve, parameter scan, and grid scan APIs are present.
- Sparse split-real superoperators are used instead of dense operators.
- `operator_interpolation="linear"` and `"pchip"` are supported.
- Constant Rabi-rate and detuning detection are implemented.

### Rust Hamiltonian / Couplings Review Items

- Shared Python-to-Rust state parsing helpers were extracted in `rust/src/lib.rs`.
- Rust B-state `mu_p()` now reads `constants.gl` rather than hardcoding `gL`.
- `BConstants::default()` exists.
- Rust spherical tensor `J - 1` branches are guarded with `psi.j >= 1`.
- Independent Hamiltonian term computation is parallelized with Rayon.
- Rust coupling matrix generation now accepts precomputed state indices instead
  of doing Python-level `rich_compare` scans.
- Rust coupling matrix storage is flat row-major `Vec<Complex64>`.
- Dead dependencies `wigner-3nj-symbols` and `libm` are no longer present in
  `rust/Cargo.toml`.
- Reduced-Hamiltonian fixture comparisons now pass. The omega-basis comparison
  uses sorted eigenvalues because the omega path is intentionally not reduced
  and degenerate partner ordering can vary; parity and total comparisons use a
  tolerance that covers tiny diagonal roundoff at large absolute energy scale.
- Rust state arithmetic still stores terms in `Vec`, but subtraction no longer
  allocates a negated intermediate state, addition reserves incoming capacity,
  and scalar multiplication preallocates output and exits early for zero
  scalars.
- `generate_transform_matrix_py()` now caches coupled-to-uncoupled expansions
  once per basis state before building the inner-product matrix.
- `h_mat_elems_generic()` now builds one lookup map per applied ket and uses
  direct basis-state lookup instead of scanning each term list inside the
  matrix assembly loop.
- B-state Stark and Zeeman components now share `d_p()` and `mu_p()` spherical
  component evaluations across x/y/z construction.
- B-state Stark/Zeeman helper comments now use one explicit sign convention and
  ASCII `mu` notation to avoid comment/code ambiguity.
- Uncoupled omega/parity expansion is now centralized in
  `states.expand_uncoupled_parity_to_omega_components()`.
  `UncoupledBasisState.transform_to_omega_basis()` and the uncoupled electric
  dipole matrix-element path both use this helper, so the signed-Omega phase
  convention lives in one place.
- Effective-Hamiltonian field-grid preparation now records adjacent operator
  variation diagnostics and warns when the fixed-basis operator grid has a large
  relative jump between neighboring field points. The diagnostic covers
  `h_internal`, `h_opt`, `h_det`, the dissipator superoperator, and the
  jump-rate operator.

## Partially Implemented

- Repeated state-index lookup was improved on the Rust coupling path, but
  Python fallback paths still use `list.index()` in places such as
  `centrex_tlf/couplings/coupling_matrix.py`,
  `centrex_tlf/couplings/collapse.py`,
  `centrex_tlf/lindblad/generate_hamiltonian.py`, and
  `centrex_tlf/hamiltonian/utils.py`.
- The OBE roadmap's static packed RHS/JVP and explicit solver goals are done,
  but the implemented backend now also supports time-dependent parameter
  graphs. Native Rust stiff solving is still not implemented.
- The Hamiltonian lowering `"auto"` heuristic exists, but its cost model is
  still not documented or benchmark-justified.
- The Rust Hamiltonian code now has some Rust-side unit tests, but coverage is
  still mostly integration-level through Python.

## Still Outstanding

### General Codebase Items

- `centrex_tlf/hamiltonian/B_uncoupled.py` still has placeholder `HZx()` and
  `HZy()` implementations that return the input state instead of raising or
  implementing the Zeeman terms.
- `centrex_tlf/states/states.py` is still a large mixed-concern module covering
  basis models, hashing, algebra, formatting, and transforms.

### OBE / Effective Solver Items

- Analytical structured Jacobian for OBE is not implemented. The packed scipy
  path still probes the RHS basis vectors.
- SIMD acceleration for the sparse commutator is not implemented.
- BLAS loading remains Windows/scipy-OpenBLAS oriented in `rust/src/lindblad/blas.rs`.
- Native Rust stiff solver support is not implemented; stiff support is still
  through scipy BDF/Radau.
- Optional preassembled packed Liouvillian mode from the original roadmap is
  not implemented as a first-class execution mode.

### Rust Hamiltonian / Couplings Items

- Exact floating-point equality is still used for zero-amplitude filtering.
- Dead or low-value functions such as `j4`, `j6`, `h_c3a`, `h_c3b`, and
  `h_c3c` are still present.
- The new `h_mat_elems_generic()` lookup maps should be benchmarked against the
  old linear scan for very small bases, although the project normally operates
  at 64 or more states where the map-based path should be favored.

## Performance Review (2026-07-11)

Full review of where time is spent across setup, solve, and scan paths.
Measured on Windows, 65-state R(0) F1'=3/2 F'=2 system unless noted; r2
numbers from `benchmarks/expanded_sparse_packed_rhs_results/`.

### Measured baseline

| Quantity | Value |
| --- | --- |
| `generate_OBE_system_transitions` (method="matrix") | 4.4 s |
| — symbolic dissipator (`generate_dissipator_term`) | 3.5 s (3.1 s is the sympy anticommutator matmul) |
| — reduced Hamiltonian generation | 0.64 s (0.56 s = 2x `generate_reduced_B_hamiltonian`) |
| `generate_hamiltonian_term` (-i[H,rho], method="expanded" adds this) | 1.4 s |
| `prepare_lindblad_problem` (Rust plan) | 0.12–0.20 s |
| Rust solve, dopri5/expanded_sparse, 10 us span, 201 saveat | 4.97 ms |
| Same via scipy_rk45 / scipy_bdf / scipy_radau | 55 / 144 / 241 ms |
| python_rk45 reference | 2.37 s |
| r2 (38-state, time-dependent H): RHS cost | 105 us/call (~93 us in the sparse gather kernel) |
| r2 single solve | 18.6 s, 28 651 accepted steps, 171 907 RHS calls |

`benchmarks/bench_setup.py` (added 2026-07-12) confirms the scaling on a
154-state R(2) opposite-parity system: end-to-end setup 62.5 s, of which the
symbolic dissipator is 51.1 s and the symbolic Hamiltonian term 10.7 s —
~99% of the build is the symbolic construction that the lazy/sparse change
removes from the Rust path. `tests/lindblad/test_symbolic_system_equivalence.py`
pins the current construction numerically at the DEFAULT Jmax build (the
wider J range is required for correct rotational mixing in the dressed
states — do not restrict Jmax to the driven manifolds). Undriven spectator
manifolds keep ~2.6e11 rad/s absolute diagonal energies, so the comparison
uses a magnitude-aware atol (~eps * max|H| * max|rho|) instead of a fixed
absolute tolerance; construction bugs sit many orders above that floor.

Setup vs. solve is ~900:1 for time-independent systems, so notebook and
effective-model workflows are setup-bound. For time-dependent oscillatory
systems (r2), solve time = step count x per-call RHS cost dominates.

### Confirmed setup bottlenecks

1. **[IMPLEMENTED 2026-07-12] Symbolic dissipator/system built eagerly but
   unused by the Rust path.**
   Landed: `OBESystem.system`/`.dissipator` are now lazy cached properties
   (same constructor signature; setters kept), built on first access by
   sparse entrywise constructors in `generate_system_of_equations.py`
   (`generate_hamiltonian_term` is O(nnz(H)*n); `generate_dissipator_term`
   iterates collapse-operator nonzeros — also fixing the old `fast=True`
   path's silent truncation to the first nonzero for multi-entry operators —
   and builds the anticommutator from the numeric `C dagger C` nonzeros).
   `method=` in `generate_OBE_system*` is a validated, documented no-op.
   Measured (bench_setup, median of 3): end-to-end build 65-state
   4.66 s -> 0.77 s (6x); 154-state r2-style 57.8 s -> 4.43 s (13x).
   On-demand symbolic build when requested: dissipator 3.5 s -> 0.08 s /
   46.9 s -> 2.3 s; Hamiltonian term 1.5 s -> 0.84 s / 10.8 s -> 6.3 s.
   Full test suite 289 passed, 1 skipped;
   `test_symbolic_system_equivalence.py` pins numeric agreement. Baseline
   timings preserved in `benchmarks/setup_path_results_pre_lazy_baseline/`.
   Original finding follows.
   `generate_dissipator_term` computes the anticommutator as a dense
   numpy-complex x sympy matmul (O(n^3) symbolic ops, 3.1 s at n=65) even
   though `C dagger C` is exactly diagonal for single-jump operators, and
   nothing in `centrex_tlf.lindblad` reads `OBESystem.system` or
   `.dissipator` — `prepare_lindblad_problem` consumes only `H_symbolic` and
   `C_array`. The consumers are the Julia extension
   (`CeNTREX-TlF-julia-extension`, `lindblad_julia/utils_setup.py`) and
   visualization.
   **Plan:** make `system`/`dissipator` lazy cached properties backed by
   sparse entrywise builders (diagonal `C dagger C` -> elementwise
   `-(gamma_i+gamma_j)/2 * rho_ij`; commutator built from H nonzero
   structure, with a general sparse fallback for non-single-jump operators).
   Deprecate `method=` in `generate_OBE_system*` to a no-op; the Julia
   extension keeps its own `method` argument, which selects generated Julia
   code (matrix/her2k vs expanded scalar lines — matrix is much faster in
   Julia) and is unaffected by how the sympy matrices are assembled. Guard
   with an equivalence test (old vs new construction, numeric substitution);
   expect one-time churn in generated Julia text (CSE temp numbering).
2. **B-state Hamiltonian built and diagonalized twice** per optical
   transition (`generate_reduced_hamiltonian_transitions` discovery pass +
   `generate_total_reduced_hamiltonian`). Confirmed 2 calls, 0.56 s. Pass the
   first result through (`H_func_B` or the `ReducedHamiltonian`).
3. **`lower_hamiltonian_upper_triangle` builds both representations**
   (entrywise and decomposed) regardless of the requested one; only "auto"
   needs both. Small today (0.07 s) but is most of prepare time and grows
   with expression complexity.
4. **Effective-model field-grid prep** rebuilds the full compact OBE system
   per field point, serially (`effective_hamiltonian/preparation.py`).
   Linear x per-point seconds. Fix 1 cuts each point ~3–5x; then hoist
   field-independent work and process-parallelize the point loop.

### Confirmed solve bottlenecks

5. **No native Rust stiff solver.** scipy_bdf/radau pay per-RHS-call FFI +
   Python stepping: 29–49x slower than native dopri5 on the same system.
   scipy_rk45 at 11x shows most of that is boundary overhead. A native
   ESDIRK/Rosenbrock/BDF in `rust/src/ode/` (reusing the exact sparse
   Jacobian machinery) also unlocks stiff batch/grid scans.
6. **Oscillatory time-dependent systems are step-count bound.**
   [DIAGNOSED 2026-07-13] Measured on the notebook-identical r2-in-E-field
   system (Ez=171.6 V/cm, 38 states): accepted steps are FLAT across reltol
   1e-5..1e-9 (~24k steps / 108.7 us, dt=4.52 ns) — oscillation-limited, not
   accuracy-limited. The limiter is the driven B J=3 manifold's internal
   spread (73.6 MHz: Stark-split opposite-parity + hyperfine) and X-B
   coupling detunings (<=50.4 MHz); GHz-scale spectator manifolds carry zero
   coherence and are harmless. Secular approximation NOT viable (the 25 MHz
   opposite-parity peak is the observable).
   [MEASURED NEGATIVE 2026-07-13] The per-level co-rotating frame was
   prototyped (`apply_per_level_rotating_frame`, opt-in, validated —
   populations/photon scans frame-invariant to <=9e-5) and is a NET
   SLOWDOWN: 2.4x slower at detuning 0 (steps only 1.23x fewer, per-call
   cost ~3x from trig coefficient evaluation) and 6x slower at 25 MHz
   (steps DOUBLE — the coefficient phases beat against the remaining
   symbolic detuning diagonal). A generic explicit RK must still resolve
   the relocated oscillations; the earlier ~3-10x estimate was wrong.
   The exponential/Lawson 10-25x estimate is consequently UNCONFIRMED —
   it avoids exposing the oscillation to the stepper, but must be
   feasibility-checked (note: at fixed parameters this system's L is
   time-independent, so exact-propagator approaches — eig/expm of the
   1444-dim Liouvillian per grid point — are the natural alternative to
   compare against ~1.1 s/trajectory stepping before building anything).
   Full analysis:
   `benchmarks/step_size_diagnostics_results/step_size_diagnostics_report.md`.
   [SCAN-LEVEL RESULTS 2026-07-13, see
   `benchmarks/scan_speedup_results/scan_speedup_summary.md`]: single-system
   polarization-symbol 2D scans validated (2.22x end-to-end, build overhead
   11.6x lower — adopt); exact-propagator (eig of the time-independent
   Liouvillian) numerically clean but break-even for final-only single-rho0
   scans (wins 400-750x per extra initial state, 15-20x for saveat
   trajectories — niche tool); threads=None gives 9.19x on 16 cores; batch
   worker reuse landed in `ode_batch.rs` (parameter_scan now matches
   grid_scan).
   Original finding follows: r2 needs
   ~29k accepted steps per 200 us trajectory (vs ~300 for the RWA-clean R(0)
   system) because residual fast phases are not removed by the manifold
   rotation. Options: per-dressed-level co-rotating frame (extend
   `_build_manifold_indices`) or an interaction-picture/exponential
   integrator. Largest single lever for r2-class problems; physics-sensitive,
   validate against current r2 notebooks.
7. **[IMPLEMENTED 2026-07-12] Expanded-sparse gather kernel.** The Rust plan
   now propagates time dependencies through compound parameter slots and
   partitions packed RHS terms into trajectory-static and dynamic streams.
   Static terms are grouped by output/input, identical inputs are combined,
   and coefficients are embedded in the packed term layout; scan overrides
   invalidate and rebuild that cache while only dynamic term values are
   refreshed per RHS call. Same-build benchmark controls retain both the old
   unpack/complex/pack kernel and the previous split-input kernel.

   Repeated release measurements are in
   `benchmarks/expanded_sparse_static_dynamic_results/` and
   `benchmarks/q1_r2_rhs_kernel_static_dynamic_results/`. On the targeted
   38-state r2 system, the optimized kernel measured 8.39 us/RHS vs 8.97 us
   for the previous kernel (7 repeats), 1.497 s vs 1.623 s for a trajectory
   (4 repeats), and 5.612 s vs 6.173 s for the 9-point grid (4 repeats).
   Four full compact-scan repeats improved q1 by 3.8% and r2 by 6.3% versus
   the previous kernel, with unchanged peaks and differences <=2.3e-10.

   The first all-size implementation was also measured rather than assumed:
   it regressed on retained noncompact q1 (23.26 s vs 20.80 s) and r2
   (309.57 s vs 285.58 s), because four partition streams lose cache locality
   across large packed upper triangles. Production `expanded_sparse` is
   therefore hybrid: the partitioned layout is selected through 40 states
   (covering the measured 38-state target), while larger systems retain the
   previous split-input kernel. Repeated large-plan RHS checks confirm the
   hybrid path matches that control within run noise. The historical absolute
   105 us/18.6 s numbers were not reproduced by the fresh release build
   (same-build legacy control: 9.30 us/1.694 s), although accepted-step and
   RHS-call counts match; comparisons above therefore use counterbalanced
   controls from the same extension build. Explicit SIMD was not needed.

   **[INVESTIGATED 2026-07-12] Replacing the n_states=40 proxy with a
   plan-statistic gate.** The 40-state cutoff is a proxy calibrated on 4
   systems, so a follow-up measured whether gating directly on the
   partitioned layout's working-set size (total term count across the four
   partition streams, and an estimated byte footprint including the pointer
   arrays and the `StaticPackedTerm` de-duplication lists) would separate
   winners from losers more principled. Measured at the point the workspace
   is built in `rhs.rs` (`build_partitioned_expanded_packed_inputs`), across
   the calibration systems:

   | system | n_states | upper_len | partitioned term count | byte footprint |
   | --- | --- | --- | --- | --- |
   | two_level (winner) | 2 | 3 | 9 | 784 |
   | r2 n=38 (winner) | 38 | 741 | 6,813 | 517,160 |
   | q1 compact (winner) | 17 | 153 | 592 | 48,032 |
   | r2 compact (winner) | 38 | 741 | 6,813 | 517,160 |
   | q1 noncompact (loser) | 66 | 2,211 | 3,812 | 350,432 |
   | r2 noncompact (loser) | 154 | 11,935 | 36,023 | 2,997,048 |

   Neither statistic separates cleanly: the noncompact-q1 loser (term count
   3,812; byte footprint 350,432) is smaller on both measures than the
   38-state r2 winner (term count 6,813; byte footprint 517,160), because
   noncompact retention adds many weakly-coupled states that inflate
   `upper_len` without adding proportionally many nonzero coupling terms —
   term density is decoupled from system size. Per the stated stopping rule,
   this overlap means the n_states gate was kept rather than forced onto a
   statistic that doesn't actually separate the two classes; `n_states` (via
   `upper_len`, which is monotonic in it) remains the cleanest available
   proxy for the cache-locality mechanism. `PARTITIONED_PACKED_MAX_STATES`
   in `rust/src/lindblad/rhs.rs` is unchanged at 40.

### Smaller confirmed items

- `rhs.rs` takes ~6 `Instant::now()` timestamps per RHS call even with
  profiling disabled (~3–4% for 2-level systems; negligible for n >= 30).
- `solve_batch_ode` builds a fresh `LindbladRhs` workspace per trajectory;
  the grid path already reuses per-thread workers via `map_init` — apply the
  same pattern to batch/parameter_scan.
- `State.__eq__` is O(k^2) with `np.allclose` per amplitude pair and is used
  via `list.index()` scans in `reduced_basis_hamiltonian`,
  `coupling_matrix.py:197`, and the RWA transform; replace with id/hash maps
  and scalar tolerance compares.
- `benchmarks/benchmark_obe.py` crashes in its diagnostics stage
  (`KeyError: 'non_rhs_seconds'` — stats key drift vs current Rust stats).
- cProfile inflates sympy-heavy setup ~4x (17.8 s profiled vs 4.4 s wall);
  prefer wall-clock sub-timings for setup benchmarks.
- No setup-path benchmark exists; the largest confirmed cost had no
  coverage. Add one (per stage: reduced H, couplings, collapse, symbolic H,
  lowering) with stored baselines.

### Not worth doing

- Porting the symbolic setup (RWA transform, IR lowering) to Rust — the fix
  is doing less symbolic work; expressions must remain sympy for the API and
  Julia codegen.
- Porting one-time O(n^3) numerics (diagonalization, transforms) — already
  LAPACK via NumPy at n <= few hundred.
- Micro-porting state bookkeeping to Rust — Python-side hash maps suffice.
- The FFI boundary is healthy: one PyO3 call per solve/batch/grid, single
  NumPy buffers back; only the deliberate SciPy fallback crosses per call.

## Suggested Next Order

1. [DONE 2026-07-12] Fix `benchmark_obe.py` stats keys and add the setup-path
   benchmark (`benchmarks/bench_setup.py`) so the items below are measured
   before/after. Note found along the way: the "Non-RHS ms" diagnostics
   column was never real — Rust stats expose no per-phase timing, and
   `_solver_stats_dict` backfills `rhs_seconds`/`total_seconds` from the same
   wall clock; wire `RhsProfileStats` through when working on the RHS kernel.
2. [DONE 2026-07-12] Lazy `system`/`dissipator` with sparse entrywise
   builders; `method=` deprecated to a no-op (Julia extension unaffected).
   Measured: 65-state build 4.66 s -> 0.77 s; 154-state 57.8 s -> 4.43 s.
3. [DONE 2026-07-13] Union-first B-state build: the per-transition
   parity-basis discovery builds in `generate_reduced_hamiltonian_transitions`
   are replaced by ONE Ω-basis build over the union J range of all optical
   transitions; per-transition dressed excited states are identified from it
   by parity-basis largest-overlap matching (same machinery as the total
   build), and `generate_total_reduced_hamiltonian` reuses the identical
   build via the new `B_hamiltonian_omega` parameter instead of rebuilding.
   Which states are included is unchanged (opposite-parity B levels retained
   for Stark mixing; benchmark systems produce identical n_states 65/154;
   reduced-Hamiltonian fixtures pass, full suite 292 passed / 1 skipped).
   Measured stage 1: 0.38 -> 0.16 s (65-state), 2.06 -> 0.75 s (154-state);
   end-to-end build 0.62 -> 0.47 s / 3.65 -> 2.81 s. Savings scale with the
   number of optical transitions. Earlier sub-items follow:
   Lower only the requested Hamiltonian
   representation (ir.py builds a single plan unless representation="auto")
   and id-keyed state index maps with silent equality fallback
   (reduced_basis_hamiltonian, coupling_matrix pre-transform indices, RWA
   transform). Measured: prepare_lindblad_problem 0.11 -> 0.05 s (65-state),
   1.50 -> 0.93 s (154-state); end-to-end build 0.73 -> 0.62 s / 4.11 ->
   3.65 s; 290 tests pass. DEFERRED: single B-state build — not a pure
   dedup, the discovery pass builds in the parity basis while
   generate_total_reduced_hamiltonian uses the Ω basis (design choice
   needed; must not change which states are included, since opposite-parity
   B levels are required for Stark mixing in electric fields).
4. Effective-model field-grid prep: hoist invariants, then
   process-parallelize the per-point loop.
5. Trivial Rust wins: guard `Instant::now()` behind profiling; batch worker
   reuse via `map_init`.
6. [DONE 2026-07-12] Expanded-sparse static/dynamic term split and hybrid
   packed layout, with repeated compact/noncompact benchmarks.
7. Step-count reduction for oscillatory systems (per-level co-rotating
   frame / interaction picture) — highest payoff, needs physics validation.
8. Native Rust stiff solver — largest effort; benchmarks will show how much
   stiff workload remains by then.
9. Prior outstanding items, unchanged: replace or raise in
   `B_uncoupled.HZx/HZy`; benchmark and document the Hamiltonian lowering
   `"auto"` heuristic; benchmark the Rust Hamiltonian assembly path at
   representative basis sizes (64+ states).
