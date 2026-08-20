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

Last run 2026-08-20, full suite, both backends:

```powershell
uv run pytest -q
cargo test --release            # from rust/
```

Result:

- Python: 390 passed, 0 failed, 1 skipped (73.1 s)
- Rust: 61 passed, 0 failed, 1 ignored

The one ignored Rust test is the `h_mat_elems` lookup-vs-linear-scan
benchmark, run explicitly with
`cargo test --release bench_h_mat_elems -- --ignored --nocapture`.

Earlier snapshots in this file quote smaller counts because they were scoped
to `tests\states tests\lindblad` (125 passed) or predate later work; the
number above is the whole suite. The old partial-state hashing failures from
`CODEBASE_AUDIT.md` have not recurred, and the Rust Lindblad API/wrapper
failures found during the audit are fixed.

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
- `B_uncoupled.HZx()` and `HZy()` no longer return the input state unchanged
  behind a `# TODO`. They raise `NotImplementedError` pointing at the
  coupled-basis `B_coupled.HZx`/`HZy`, which is what the Hamiltonian
  generators actually call. The uncoupled-basis transverse Zeeman terms are
  still unimplemented — the change removes a silent-wrong-answer path, it does
  not add the physics. `HZz` was and remains implemented.

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

- [DONE 2026-08-19] Repeated state-index lookup. Of the four files listed,
  three (`couplings/collapse.py:87-93`,
  `lindblad/generate_hamiltonian.py:152-159`,
  `hamiltonian/utils.py:99-106`) already carried the id-keyed map and retained
  `list.index()` only as a rare equality fallback for states that are equal to
  but not identical with a basis entry. The one genuinely unoptimized site,
  `couplings/coupling_matrix.py:99-104`, now uses the same pattern.
  Measured: building the two index mappings drops from 0.33 ms to 0.012 ms
  (65-state, 28x) and 1.24 ms to 0.026 ms (154-state, 47x), confirming the
  O(n^2) scaling. **The absolute saving is negligible** — ~1 ms against a
  ~2.8 s build — so this is a consistency and scaling fix, not a measured
  speedup. Results identical before and after.
- The OBE roadmap's static packed RHS/JVP and explicit solver goals are done,
  but the implemented backend now also supports time-dependent parameter
  graphs. Native Rust stiff solving is still not implemented.
- [DONE 2026-08-19] The Hamiltonian lowering `"auto"` heuristic is now
  benchmarked and documented — see
  `benchmarks/hamiltonian_representation_results/report.md` and
  `benchmarks/bench_hamiltonian_representation.py`. Outcome: the `0.15` and
  `+1` constants in `ir.py:434-441` are inconsequential and cannot usefully be
  calibrated, because the two branches are separated by a factor of 537
  (65-state) to 1182 (154-state); the constants would have to move three
  orders of magnitude to flip a case. The choice `auto` makes is also the
  empirically correct one: decomposed lowers ~20% faster, prepares faster, and
  is the only representation that unlocks `expanded_sparse`, the fastest solve
  path on both systems. Kept as-is per the same stopping rule used for
  `PARTITIONED_PACKED_MAX_STATES`. Context limiting the stakes: `"auto"` is
  not the default (`plan_static.py:120` is `"decomposed"`), nothing in-package
  passes it, and an entrywise plus `expanded_sparse` request errors clearly at
  `rhs.rs:1256` rather than silently degrading.
- The Rust Hamiltonian code now has some Rust-side unit tests, but coverage is
  still mostly integration-level through Python.

## Still Outstanding

### General Codebase Items

- `centrex_tlf/states/states.py` is still a large mixed-concern module covering
  basis models, hashing, algebra, formatting, and transforms.

### OBE / Effective Solver Items

- [DONE 2026-08-20] Analytical structured Jacobian for OBE. The packed scipy
  path used to recover the Jacobian by probing `packed_len()` basis vectors,
  one full RHS evaluation each (`build_packed_jacobian_sparse`,
  `rust/src/lindblad/rhs.rs`). Note the probe was already **exact**, not a
  finite difference -- the Lindblad RHS is linear in rho and the packed
  Hermitian encoding is real-linear -- so this was never an accuracy item,
  only an O(n^4) cost one, and the old wording ("still probes the RHS basis
  vectors") read misleadingly.
  The superoperator turned out to be assembled already: `ExpandedSparseRhsPlan`
  (`plan.rs:650`) is a CSR-style sparse Liouvillian whose term indices are
  upper-triangle indices, `upper_to_packed` maps those to packed real slots,
  and the dissipator is folded into the same terms (its arm in
  `rhs_from_workspace_rho` is empty for the expanded modes). So every term is
  a Jacobian entry and the matrix transcribes in O(nnz).
  `build_packed_jacobian_analytic` does that; `jacobian_packed_sparse_py`
  grew a `method` argument (`"auto"` -- default, transcribe when the plan has
  an expanded form, else probe; `"analytic"`; `"probe"`). The probe is kept as
  the reference the analytic path is tested against, and as the only option
  for an entrywise plan.
  Measured, at the drive-on operating point:

  | system | mode | probe | analytic | speedup | nnz |
  | --- | --- | ---: | ---: | ---: | ---: |
  | A (n=65, dim 4225) | expanded_sparse | 21.8 ms | 0.164 ms | 133x | 5,634 |
  | A | structured | 285.7 ms | 0.148 ms | 1933x | 5,634 |
  | B (n=154, dim 23716) | expanded_sparse | 821.8 ms | 1.118 ms | 735x | 44,508 |
  | B | structured | 4305.0 ms | 1.166 ms | 3691x | 44,508 |

  Agreement is **bitwise**, not approximate: `max|J_analytic - J_probe|` is
  exactly 0.0 against `||J||_inf ~ 1e12`, with identical nnz, in all four
  cells. Both sum the same floating-point products in the same order, so
  anything nonzero would mean the transcription reassociated something. The
  key ratio the item turned on: the probe ran 4,225 full RHS evaluations to
  recover an average of 1.3 nonzeros per column (0.03% dense; 0.008% for B).
  End-to-end on the stiff path, system A over 1e-5 s: `scipy_bdf` with
  `jacobian="exact"` goes **31.1 ms -> 7.4 ms** (median of 3 runs, stable to
  0.3 ms), i.e. the build was 68% of the whole solve and is now ~2%. Context
  bounding the value: only the scipy stiff fallback ever forms a Jacobian
  (`dopri5`/`tsit5` are explicit), and it is cached per solve for a
  time-independent plan -- measured, BDF rebuilt it exactly once even for a
  time-dependent plan at spans of 1e-5, 1e-4 and 1e-3 s, since it only
  refreshes on Newton convergence failure. The case that actually pays is a
  stiff *scan*, where the build is per grid point and never amortized:
  811 ms x 380 points was ~5 minutes of pure assembly on system B.
  Not done, and deliberately: `build_split_jacobian_sparse` (`rhs.rs`) is
  untouched. It probes into the 2n^2 split layout, so it produces a matrix 4x
  the size with 2x the dimension for the same probe count (A: 19,080 nnz at
  dim 8,450 vs 5,634 at 4,225), does not enforce Hermiticity, and is used by
  nothing in `centrex_tlf/` -- only two `benchmarks/profile_bdf*` scripts and
  one test. It should be deleted or demoted to a test-only reference rather
  than given an analytic path of its own.
- SIMD acceleration for the sparse commutator is not implemented.
- BLAS loading remains Windows/scipy-OpenBLAS oriented in `rust/src/lindblad/blas.rs`.
- Native Rust stiff solver support is not implemented; stiff support is still
  through scipy BDF/Radau.
- Optional preassembled packed Liouvillian mode from the original roadmap is
  not implemented as a first-class execution mode.
- [DONE 2026-08-20] One `hamiltonian_valid` flag guarded three *disjoint*
  Hamiltonian caches in `RhsWorkspace` (`rust/src/lindblad/rhs.rs`). The
  complex-matrix path fills `expanded_term_values`, the packed paths fill
  `expanded_term_values_re`/`_im`, and the partitioned packed path
  additionally resolves static coefficients inside
  `partitioned_expanded_packed_inputs` -- but all of them set and tested the
  same boolean. Whichever flavour ran first marked the cache valid; the next
  flavour then skipped filling *its own* cache and read it empty.
  Reproduced in both directions on one evaluator: packed-then-split raised
  `expanded RHS expected 2808 cached term values, got 0`, and
  matrix-then-packed raised `... 2808 cached split term values, got 0 and 0`.
  A fresh evaluator calling either path first worked, which is what made it
  look like "the split Jacobian is broken under expanded_sparse" rather than
  a cache-invalidation bug. Only reachable for time-independent plans -- a
  time-dependent plan refills unconditionally -- and not reachable from
  `centrex_tlf/` at all, since `solve.py` and `batch.py` only ever use the
  packed path. Latent, but it would have bitten the first person to write a
  complex-matrix reference check.
  The partitioned variant was the nastier case: it would have skipped
  `refresh_partitioned_static_terms` and evaluated with static coefficients
  still at 0.0, i.e. silently wrong output rather than an error.
  Fixed by replacing the boolean with `hamiltonian_valid_for:
  Option<HamiltonianCache>` naming which cache holds values, so a flavour
  change forces a refill. Regression test:
  `test_hamiltonian_cache_is_not_shared_between_rhs_flavours`.
- [DONE 2026-08-20] The entrywise + `expanded_sparse` incompatibility now fails
  at the solve entry point instead of at the first RHS call.
  `lower_expanded_sparse_rhs` returns `None` for a non-decomposed plan, and
  `rhs.rs:1256` was the first thing to notice — after integration had started,
  and inside a parallel batch or grid scan that surfaced as a worker error far
  from its cause. `PreparedLindbladProblem.check_execution_mode` now raises a
  `ValueError` naming the fix, called from `solve_lindblad`,
  `solve_lindblad_batch` and `grid_scan` (`parameter_scan` and
  `initial_condition_scan` delegate to the batch path). It keys on the shared
  `expanded_sparse` substring so all six `ExecutionMode::is_expanded_sparse_like`
  variants are covered, and is skipped for `backend="python"`, whose reference
  path maps every non-`"reference"` mode onto the structured RHS and so is
  unaffected by a missing plan. The Rust-side check stays as the backstop.

### ODE Helper Parity Items

Absorbed from `ODE_HELPER_FUNCTION_COMPARISON.md`, which was otherwise fully
implemented and has been removed. The high-level `RuntimeExpression`
constructors it asked for all exist in `centrex_tlf/lindblad/parameters.py` and
are covered by `tests/lindblad/test_parameter_helper_wrappers.py`.

- [DONE] `square_wave` and `sawtooth_wave` are now verified against the Julia
  definitions, and the check found a real bug. `sawtooth_wave` was offset by
  exactly half a period in BOTH backends: Julia's `Waveforms.sawtoothwave` is
  zero-centred (`rem2pi(x, RoundNearest)/pi`, range (-pi, pi]), and the port
  replaced that with floor-based `% 1.0` / `rem_euclid(1.0)` while keeping
  Julia's `- pi`, applying the half-period shift twice. Consequence: `phase=0`
  started halfway up the ramp and each period's discontinuity fell mid-period.
  Fixed by dropping the `- pi` in `helper_functions.py` and
  `rust/src/lindblad/eval.rs`, both carrying a comment so it is not
  "restored". `square_wave` was found correct — it differs from Julia only at
  the exact switching points (`mod2pi` of 0 or pi), which is a measure-zero
  convention difference.
  The bug survived because the suite only ever compared Python against Rust,
  and both were wrong identically. `tests/lindblad/test_parameter_helper_wrappers.py`
  now pins both waveforms against transcribed Julia references across many
  periods and both signs of `t`; the old implementation fails that test on
  401 of 401 sampled points. Additionally confirmed against the REAL Julia
  through `juliacall` + Waveforms.jl: sawtooth agrees to 1.2e-15, square
  exactly.
- [DONE 2026-08-20] `variable_on_off_duty_invT` shares a helper ID with
  `variable_on_off_duty`. Confirmed intentional and now documented rather than
  changed: the Julia backend spells this gate `variable_on_off_duty_invT`
  (`julia_common.jl`, the only spelling on that side), Python and Rust spell it
  `variable_on_off_duty`, and both names are exported so an expression written
  against either vocabulary lowers unchanged. There is one numeric definition,
  not two — the Python `_invT` is a one-line delegation and Rust has only the
  short name.
  A real latent bug turned up next to it: `HELPER_FUNCTION_NAMES` was a plain
  inversion of `HELPER_FUNCTION_IDS`, so for a shared ID whichever name was
  declared *last* won — `variable_on_off_duty_invT`, silently swappable by
  reordering the dict. Aliases are now declared explicitly in
  `HELPER_FUNCTION_ALIASES` and skipped when inverting, so the canonical name
  wins deterministically. Harmless today (both names resolve to the same
  callable in `ir._apply_helper`), but it was one dict reorder away from
  mattering.
  Docstrings added on both Python sites plus `README_OBE_SOLVER.md`; tests in
  `tests/lindblad/test_parameter_helper_wrappers.py` pin the gate against a
  transcribed Julia `mod1` reference (including that `mod1` returns (0, 1], so
  an exact 0 maps to 1.0), pin the alias to the canonical implementation, and
  pin that aliases are the only id collisions.

### Rust Hamiltonian / Couplings Items

- [WONTFIX, measured] Exact floating-point equality is used for
  zero-amplitude filtering (`rust/src/states.rs:58,102,118,137,247`,
  `b_coupled.rs`) — deliberately, and Python agrees (`states/states.py:972`,
  `amp != 0`). Exact filtering only ever *keeps* a dust term; a tolerance
  would *discard* real amplitude, so it is the conservative direction, and an
  opt-in tolerance pruner already exists (`State.remove_small_components(tol)`,
  `states.py:1221`).
  Measured 2026-08-19: in the final built states there is **no dust at all** —
  0 terms below 1e-12 relative out of 150 terms (65-state R(0)) and 474 terms
  (154-state r2); the smallest relative amplitude actually present is ~1e-3.
  In intermediate operator arithmetic (7 X-state operators over a 64-state
  uncoupled basis) dust exists but is negligible: 4 terms out of 858, smallest
  relative amplitude 1.6e-16, i.e. machine-epsilon cancellation residue that
  does not propagate to the built states. Note the clean 13-order-of-magnitude
  separation between physics (~1e-3) and dust (~1e-16) if anyone revisits
  this; there is currently no measured reason to.
- [WONTFIX] `j4`, `j6`, `h_c3a`, `h_c3b` and `h_c3c` were previously listed
  here as dead code. That was wrong for most of them, and the rest are kept
  deliberately as quantum-operator building blocks. Verified call graph:
  - `j4` is **live in both backends** — `rust/src/x_uncoupled.rs:77`
    (`h_rot = B·J² - D·J⁴`) and `general_uncoupled.py:50` /
    `B_coupled.py:41`. It has been called since the 0.2.4 `D_rot` work.
  - `J6` is **live in Python** — the B-state sextic `H_const` term in
    `B_coupled.py:42` and `B_coupled_Omega/rotational.py:69`. Rust's `j6` is
    uncalled only because `rust/src/b_coupled.rs:331-335` inlines the same
    arithmetic as local scalars instead of calling the operator function.
  - `h_c3a`/`h_c3b`/`h_c3c` are the only genuinely uncalled ones. They are
    three independent algebraic forms of the same tensor spin-spin c3 term and
    therefore cross-check each other. Note `h_c3c`'s identity is valid only
    against `h_rot_rigid` (B·J²), not the distorted `h_rot`.
  Do not propose removing any of these on unused-symbol grounds.
- [WONTFIX, measured 2026-08-20] The `h_mat_elems_generic()` lookup maps were
  benchmarked against the old linear scan — see
  `benchmarks/h_mat_elems_lookup_results/report.md` and
  `generate_hamiltonian::tests::bench_h_mat_elems_lookup_vs_linear_scan`
  (`cargo test --release bench_h_mat_elems -- --ignored --nocapture`).
  **The premise was wrong in both directions.** The maps are not merely
  unhelpful at small bases: the linear scan is faster at *every* size
  measured, X and B, from 4 to 320 states, by 3x to 20x. There is no
  crossover in the range this project uses.
  Variability: 7 interleaved trials per cell, 3 independent process runs.
  Individual timings are noisy — per-cell relative spread reaches 91-99% —
  so no single microsecond figure is good to better than ~2x. The *ratio*
  is not, because both implementations are timed against each other inside
  the same trial and the common-mode noise cancels: the worst single trial
  out of ~340 had the scan at 0.41 of the map (a 2.4x win), and per-cell
  median ratios reproduce run-to-run within +-0.02. The bench asserts
  `worst_ratio < 1.0`, i.e. it fails if any individual trial ever goes the
  other way, rather than merely checking separated medians.
  Cause: terms-per-applied-state `k` is set by the operator's selection rules,
  not by basis size, and stays at 1-12 (X) / 1-7.6 (B). So the map pays `n`
  allocations plus `n^2/2` SipHash lookups to avoid `n^2/2 * k` cheap
  integer-struct comparisons. The `scan/map` ratio does climb with `n`
  (0.06 -> 0.35 in X), so a crossover exists near `n ~ 1000` — far past the
  64-320 states these builders see.
  Kept as-is anyway: at the OBE default `Jmax_X = 4` (n = 100) the seven X
  operators cost ~0.8 ms with the maps, run under `rayon`, against a ~2.8 s
  build. Same stopping rule as `PARTITIONED_PACKED_MAX_STATES` and the
  `representation="auto"` cost model. If the assembly path ever goes hot the
  right fix is not the scan but dropping the `n^2` structure: one
  `basis -> index` map over `qn` built ONCE, scattering each applied state's
  `k` terms, i.e. `O(n*k)`. `h_mat_elems_from_applied_linear_scan` stays in
  the file under `cfg(test)` with an equivalence test so the comparison is
  reproducible.

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
   [PROFILED 2026-08-20] Measured on Q1_F1_3o2_F2, Z polarization, 20-state
   compact system: **0.63 s per patch**, of which
   `generate_OBE_system_transitions` is **97%** - the lambdify of
   `H_symbolic` is 2% and `generate_transition_selectors` is ~0%, so hoisting
   those out of the loop is worth 2% and is not the fix. Inside the OBE build
   (cProfile fractions; absolute times inflated ~3x by the profiler):
   55% `generate_total_symbolic_hamiltonian`, almost all of it
   `symbolic_hamiltonian_to_rotating_frame` doing 790 sympy `subs` over the
   expression tree; 27% `generate_reduced_hamiltonian_transitions`, nearly all
   of it 1041 `generate_ED_ME_mixed_state` calls dominated by
   `transform_to_omega_basis`; 16% `collapse_matrices`, mostly
   `calculate_br`. This 0.63 s is already post-Fix-1 (lazy dissipator); that
   speedup is spent.
   [CORRECTED 2026-08-20] The line above originally recommended caching the
   bare dipole matrix elements. **That was wrong**: `ED_ME_coupled` already
   carries `@lru_cache(maxsize=1e6)`, as do `ED_ME_uncoupled`,
   `_ED_ME_uncoupled_omega` and `angular_part`. There was no missing bare-ME
   cache. The real cost was that `generate_ED_ME_mixed_state` re-transforms
   its *mixed* arguments to the Omega basis on every call while its callers
   invoke it from nested loops where one argument is loop-invariant - an
   `O(n_ground*n_excited) -> O(n_excited)` hoist, not a caching problem.
   [DONE 2026-08-20] Fixed at the two live sites: the `minimum_coupling`
   discovery loop in `hamiltonian/reduced_hamiltonian.py` (which also rebuilt
   `1 * gs` per inner iteration) and `calculate_br` in
   `couplings/branching.py` (which rebuilt
   `excited_state.remove_small_components(tol)` per ground state). A shared
   `to_omega_basis` helper in `matrix_elements_electric_dipole.py` mirrors the
   callee's guard so the hoist stays behaviour-preserving.
   `couplings/coupling_matrix.py:121,147` look like the obvious hot loops but
   are **fallback-only** - `generate_coupling_matrix` dispatches to the Rust
   `generate_coupling_matrix_py` whenever `HAS_RUST` - and were left alone.
   Measured: Omega transforms per build 1231 -> 21 (105-state), 4733 -> 57
   (154-state), 1051 -> 21 (20-state compact), i.e. 98-99% removed; build time
   0.784 -> 0.633 s, 3.372 -> 2.683 s, 0.610 -> 0.502 s, a **1.22-1.26x**
   speedup on **every** OBE build, not just the field grid, since
   `generate_reduced_hamiltonian_transitions` is on every build path.
   Verified bitwise, not to tolerance: 10080 ME triples equal under `==`, and
   `H_int` / `C_array` / `QN` / `main_coupling` for three systems identical
   under `np.array_equal` against a `git stash` baseline. See
   `benchmarks/dipole_me_hoist_results/report.md`.
   The 55% symbolic block is *structurally* field-independent
   (sparsity pattern, symbol placement) but bakes the field-dependent energies
   in as literal coefficients, so reusing it means doing the rotating-frame
   transform once with symbolic diagonal energies and substituting numerics
   per point - a real refactor along the same "lower once, bind many" lines as
   `prepare_lindblad_problem`, not a hoist.
   Parallelisation must use **processes**, not threads: the loop is sympy in
   pure Python and GIL-bound. The points are independent (the loop only
   appends), and the ordering assumption that makes that safe was checked -
   `system.QN` label order is identical at 100 and 150 V/cm. Cost to verify
   before committing: pickling `OBESystem` (sympy matrices) back across the
   process boundary.
   [DONE 2026-08-20] The duplicate-build half is fixed.
   `prepare_lindblad_safe_compact_interpolated_model` called
   `prepare_interpolated_effective_model` (which builds every patch) and then
   **rebuilt every patch a second time**, only to read
   `_compact_transition_frequency` off each `system`. The cause was
   structural: `InterpolatedEffectivePatch` carries `bundle` but not
   `system`, so the caller could not reach what had already been built. The
   frequency is now computed in the existing per-point loop where the system
   is in scope (an index lookup plus a diagonal read, negligible) and carried
   on `PreparedInterpolatedEffectiveHamiltonianModel.patch_transition_frequencies`;
   a `None` fallback preserves the old rebuild for hand-constructed models.
   Measured: the removed loop cost 1.98 s of a 5.73 s total at 3 field points
   and 3.85 s of 10.09 s at 6, i.e. **1.5-1.6x** - not the naive 2x, because
   the base model also does alignment, embedding and union-layout work that
   was never duplicated. The ratio approaches 2x as the per-point build grows
   to dominate, so this is worth more on the 154-state systems than on the
   20-state one measured here.
   Sizing for the rest: at 0.63 s/point a 10-point grid is 6 s, which is not
   painful. It matters at scale - the 154-state r2 build is 4.43 s, so a
   20-point grid is ~90 s serial. Do the bare-ME cache before the process
   pool.
   Coverage note: `prepare_interpolated_effective_model` had **no test
   exercising it** - `tests/effective_hamiltonian/test_grid_diagnostics.py`
   builds synthetic 2x2 bundles and covers only the diagnostics helper.
   `tests/effective_hamiltonian/test_interpolated_preparation.py` now pins the
   prep path (grid layout, index-set disjointness, operator shapes and
   Hermiticity, non-negative decay rates, base/safe model agreement) and pins
   the patch transition frequencies against an independent from-scratch
   rebuild, which is the specific guard for the change above. It pins
   invariants and equivalences rather than golden numbers, and runs at
   B = 1e-3 G rather than the 1e-5 placeholder so the +-mF eigenvectors are
   properly determined and the test does not drift with the BLAS build.

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
   [MEASURED NEGATIVE 2026-08-20] The exponential/Lawson route (b) was then
   tested at its cost model rather than built, via a Krylov spike
   (`benchmarks/bench_krylov_exponential.py`, report in
   `benchmarks/krylov_exponential_results/`). Any exponential integrator must
   apply `exp(L0*h)` to a vector; Krylov (`expm_multiply`) is the variant that
   preserves the sparsity the current 6.3 us/matvec RHS depends on, but its
   Krylov dimension scales with `||L0*h||`. Measured on the same 38-state r2
   system at 25 MHz detuning (packed dim 1444, nnz 13,690): matvecs per
   application scale 170x for a 200x increase in h -- linear -- so the
   **projected total work to cover T is FLAT in step size**: 1223.6, 1067.3,
   1005.7, 1016.5, 1025.1, 1041.2, 973.0 and 904.6 s at h = 5 ns ... 1 us,
   i.e. +-15% around ~1000 s while h moves 200x, against a dopri5 baseline of
   0.955 s. **~1000x slower, not 10-25x faster.** Accuracy is not the issue
   (100 Krylov steps match a reltol-1e-10 dopri5 reference to 1.9e-12); the
   O(omega*T) floor is.
   The size of the loss has a specific and reusable cause:
   `||L||_1 = max |Im lambda| = 1.173e12 rad/s = 187 GHz`, which is exactly the
   spectator coherence X J=5 (+120,164 MHz) against X J=1 (-66,523 MHz) --
   sum 186,687 MHz vs 186,682 measured. **dopri5 is immune to the spectator
   manifolds because they carry zero coherence and their phases multiply
   zeros; a Krylov exponential method is not, because the Krylov dimension
   keys off the norm of the whole operator rather than the modes the solution
   occupies.** So the GHz offsets that are free for the explicit stepper are
   catastrophic for the exponential one. Projecting the ideal fix (spectators
   reduced to pure population sinks, `||L||` down to the 73.6 MHz active
   scale) gives ~0.4 s, i.e. **~2.4x** -- which independently reproduces the
   analytic estimate from the active scale, and is the ceiling for this route
   even after work that does not exist yet.
   Consequently the earlier 10-25x estimate for (b) is now MEASURED WRONG, not
   merely unconfirmed. The only remaining mechanism whose cost does not scale
   with omega*T is Magnus with Filon/Levin oscillatory quadrature (integrating
   the oscillatory coefficient integrals analytically instead of sampling
   them) -- research-grade, with order-reduction and resonance risk, for a
   payoff now bounded by the same ~2x argument. Not recommended.
   Scoping note that limits this whole item: for time-INDEPENDENT L the
   exact propagator already solves the problem and is much better than Krylov
   here (dense eig setup 0.72 s, then 0.34 ms per application, accurate to
   2.65e-10) -- at dim 1444 with nnz 13,690 the operator is far too small for
   sparsity to pay. But `exp(Lt)` requires L constant, so it does NOT cover
   polarization switching, Gaussian-beam transit, multipass or phase
   modulation, which is precisely the class item 7 existed to serve. Nothing
   now covers that class better than dopri5.
   Perspective: `parallel=True` already delivers 9.19x on the real scan shape
   and is the default, so item 7 was competing for a lever smaller than the
   one already in use.
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

- [DONE] `rhs.rs` took ~6 `Instant::now()` timestamps per RHS call even with
  profiling disabled (~3–4% for 2-level systems; negligible for n >= 30). The
  clock is now read only when profiling is on, via `profile_timer()` in
  `rust/src/lindblad/rhs.rs`.
- [DONE] `solve_batch_ode` built a fresh `LindbladRhs` workspace per trajectory
  while the grid path reused per-thread workers via `map_init`. Batch and
  parameter_scan now use the same `map_init` pattern in
  `rust/src/lindblad/ode_batch.rs`.
- `State.__eq__` is O(k^2) with `np.allclose` per amplitude pair and is used
  via `list.index()` scans in `reduced_basis_hamiltonian`,
  `coupling_matrix.py:197`, and the RWA transform; replace with id/hash maps
  and scalar tolerance compares.
- [DONE 2026-07-12] `benchmarks/benchmark_obe.py` crashed in its diagnostics
  stage (`KeyError: 'non_rhs_seconds'` — stats key drift vs current Rust
  stats). It now derives `non_rhs_seconds` from `total_seconds - rhs_seconds`.
  Note that the resulting column is a wall-clock difference, not a real
  per-phase measurement — Rust stats expose no phase timing.
- cProfile inflates sympy-heavy setup ~4x (17.8 s profiled vs 4.4 s wall);
  prefer wall-clock sub-timings for setup benchmarks.
- [DONE 2026-07-12] No setup-path benchmark existed, so the largest confirmed
  cost had no coverage. `benchmarks/bench_setup.py` now covers it, with stored
  baselines in `benchmarks/setup_path_results_pre_lazy_baseline/`.

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
4. [PARTIALLY DONE 2026-08-20] Effective-model field-grid prep: hoist
   invariants, then process-parallelize the per-point loop. The duplicate
   patch rebuild in `prepare_lindblad_safe_compact_interpolated_model` is
   removed (measured 1.5-1.6x on that entry point) and the prep path now has
   pinning tests, and the dipole-ME hoist is done (1.22-1.26x on every OBE
   build; the "bare-ME cache" this list previously called for was based on a
   wrong reading - those MEs are already `lru_cache`d, see the corrected
   bottleneck entry). Remaining: process-parallelize the loop (threads cannot
   help - it is GIL-bound sympy). The 55% symbolic rotating-frame block needs a
   "transform once with symbolic energies, substitute per point" refactor and
   should come last. See "Confirmed setup bottlenecks" item 4 for the
   profile.
5. [DONE] Trivial Rust wins: `Instant::now()` is guarded behind profiling
   (`profile_timer()` in `rust/src/lindblad/rhs.rs`) and batch/parameter_scan
   reuse per-thread workers via `map_init` in `rust/src/lindblad/ode_batch.rs`.
6. [DONE 2026-07-12] Expanded-sparse static/dynamic term split and hybrid
   packed layout, with repeated compact/noncompact benchmarks.
7. [WONTFIX, measured 2026-08-20] Step-count reduction for oscillatory
   systems (per-level co-rotating frame / interaction picture). Both routes
   are now measured and both fail; see the expanded entry under "Confirmed
   solve bottlenecks" item 6 and
   `benchmarks/krylov_exponential_results/report.md`. Short version: the
   cost floor is O(omega*T) matvecs for any method that samples the
   oscillation, so step size buys nothing, and `||L||` is dominated 2500x by
   spectator manifolds that the explicit stepper gets for free. Measured
   ~1000x SLOWER than dopri5; the ceiling after ideal fixes is ~2.4x.
8. Native Rust stiff solver — largest effort; benchmarks will show how much
   stiff workload remains by then.
9. [DONE 2026-08-20] Prior outstanding items: benchmark the Rust Hamiltonian
   assembly path at representative basis sizes (64+ states). Done as the
   `h_mat_elems_generic()` lookup-map measurement — X and B operators from 4
   to 320 states, 7 interleaved trials per cell over 3 independent process
   runs, reported in `benchmarks/h_mat_elems_lookup_results/report.md` and
   reproducible via
   `cargo test --release bench_h_mat_elems -- --ignored --nocapture`.
   Headline: assembly is not hot at the sizes this project uses (~0.8 ms for
   the seven X operators at the OBE default `Jmax_X = 4`, under `rayon`,
   against a ~2.8 s build), and the lookup maps the audit flagged are in fact
   slower than a linear scan at every size measured. See the Rust
   Hamiltonian / Couplings Items section for the full result and for the
   `O(n*k)` restructuring to reach for if this path ever does go hot.
   (The Hamiltonian lowering `"auto"` heuristic is done — see the Partially
   Implemented section.)
