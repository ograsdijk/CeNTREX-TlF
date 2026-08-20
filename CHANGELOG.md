# Changelog

## 0.2.5

### Fixed

- **`sawtooth_wave` was offset by exactly half a period, in both backends.** Julia's
  `Waveforms.sawtoothwave` is zero-centred (`rem2pi(x, RoundNearest)/π`, range (−π, π]);
  the Python and Rust ports replaced it with a floor-based `% 1.0` / `rem_euclid(1.0)`
  while keeping Julia's `− π`, applying the half-period shift twice. Consequence:
  `phase=0` started halfway up the ramp and each period's discontinuity fell mid-period.
  Anything driven by `sawtooth_wave` changes numerically. The bug survived because the
  test suite only compared Python against Rust, and both were wrong identically; both
  waveforms are now pinned against transcribed Julia references across many periods and
  both signs of `t`. `square_wave` was checked the same way and is correct.
- A single validity flag in the Rust `RhsWorkspace` guarded three disjoint Hamiltonian
  caches, so interleaving the complex-matrix RHS (`rhs_matrix_py`, `rhs_split_py`,
  `jacobian_split_sparse_py`) with the packed RHS on one evaluator made the second one
  read an empty cache and raise `expanded RHS expected N cached term values, got 0`.
  Both orders were affected, and only for time-independent plans. Not reachable through
  `solve_lindblad`/`grid_scan`, which only use the packed path. The flag now records
  *which* cache is valid.
- `HELPER_FUNCTION_NAMES` was a plain inversion of `HELPER_FUNCTION_IDS`, so for an ID
  shared by two names whichever was declared last won. Aliases are now declared in the
  new `HELPER_FUNCTION_ALIASES` and skipped when inverting, so the canonical name wins
  deterministically. No behaviour change today — both names resolved to the same callable.

### Changed

- Requesting `execution_mode="expanded_sparse"` (or any `experimental_expanded_sparse_*`
  variant) against a problem prepared with `hamiltonian_representation="entrywise"` now
  raises a `ValueError` from `solve_lindblad`, `solve_lindblad_batch` and `grid_scan`
  instead of failing at the first RHS call, mid-solve. Same incompatibility, reported at
  setup with a message naming the fix. `backend="python"` is unaffected: its reference
  path maps every non-`"reference"` mode onto the structured RHS.
- The exact Jacobian used by the SciPy stiff solvers (`solver="scipy_bdf"` /
  `"scipy_radau"` with `jacobian="exact"`) is now transcribed directly from the sparse
  Liouvillian instead of being recovered by probing one basis vector at a time. Results
  are bitwise identical; the build is 133-3691x faster (154-state system: 822 ms -> 1.1 ms),
  taking a stiff solve of the 65-state system from 31.1 ms to 7.4 ms. Problems prepared
  with `hamiltonian_representation="entrywise"` keep the probe automatically.
  `jacobian_packed_sparse_py` gained a `method` argument (`"auto"`, `"analytic"`,
  `"probe"`) for anyone who needs to pin one path.
- `hamiltonian.B_uncoupled.HZx` and `HZy` raise `NotImplementedError` instead of silently
  returning the input state unchanged. Use the coupled-basis `B_coupled.HZx`/`HZy`.

### Performance

- Setup-path loops no longer redo loop-invariant work. `generate_ED_ME_mixed_state`
  transforms its mixed (field-dressed) arguments to the Omega basis on every call, and its
  callers invoke it from nested loops in which one argument is loop-invariant; the
  transform is now hoisted at both live call sites (`hamiltonian.reduced_hamiltonian`'s
  `minimum_coupling` discovery loop and `couplings.calculate_br`) via the new
  `hamiltonian.to_omega_basis`. Omega-basis transforms per build drop 1231 -> 21,
  4733 -> 57 and 1051 -> 21 across three representative systems, taking full OBE builds
  from 0.784 to 0.633 s, 3.372 to 2.683 s and 0.610 to 0.502 s (**1.22-1.26x**, on the path
  of *every* OBE build). Results are bitwise identical, not merely close. Note this is not
  a caching change: `ED_ME_coupled` and friends were already `lru_cache`d.
- `prepare_lindblad_safe_compact_interpolated_model` rebuilt every patch a second time
  purely to recover its transition frequency, after `prepare_interpolated_effective_model`
  had already built them. The frequency is now carried on the prepared model as
  `patch_transition_frequencies`, computed during preparation. **1.5-1.6x** on that entry
  point, scaling with the number of field points.
- `_generate_coupling_matrix_python` indexed `QN` with `QN.index()` per state — a linear
  scan whose every comparison is an `O(k^2)` `State.__eq__` with a per-amplitude
  `np.allclose`. Now an identity-keyed map with an equality-scan fallback. This is the
  Rust fallback path; `generate_coupling_matrix` uses the extension when available.

### Removed

- The duplicate, unweighted definition of `utils.population.generate_uniform_population_state_indices`.
  The module defined it twice and listed it twice in `__all__`; the second definition
  shadowed the first, so the surviving one — which takes `weights=` and handles NumPy
  arrays — is the one that was already in use. Defensive
  `inspect.signature(...)` guards in notebooks are no longer needed.

## 0.2.4

Molecular and fundamental constants are now sourced explicitly and derived rather than
hard-coded, and `utils.plotting` gains a field-dressed X→B level diagram for a single
optical transition. The X rotational energies change as a result: `B_rot` moves by
24.15 kHz and X gains a quartic centrifugal-distortion term, so Hamiltonians cached before
this release are stale.

### Compatibility notes

- **X rotational energies changed; cached or pickled X Hamiltonians are stale.** `B_rot`
  and `D_rot` now derive from the NIST Dunham coefficients rather than from a
  rigid-rotor `B_e - α_e/2`: `B0_X = Y01 + Y11/2 + Y21/4` ≈ 6.667355 GHz (a 24.15 kHz
  move) and `D0_X = -Y02` ≈ 5.84 kHz, with X now carrying the quartic term
  `-D_rot·[J(J+1)]²`. The net level shift is +24.9 kHz in J=1, −65.3 kHz in J=2 and
  −551.2 kHz in J=3 — enough to move a line position, not enough to look broken.
  Anything rebuilt from a `.pkl` or from an `@lru_cache` populated before this release
  will silently disagree with a fresh build; regenerate rather than reuse. Both constants
  stay derived from `Y01_X`/`Y11_X`/`Y21_X`/`Y02_X` — do not hard-code them separately.
- `collapse_matrices()` takes `decay_rate=` (the population decay rate Γ = 1/τ, s⁻¹)
  instead of `gamma=`. The old keyword still works and emits a `DeprecationWarning`;
  passing both raises.

### Added

- `utils.plotting.plot_transition_level_diagram` and
  `utils.plotting.calculate_transition_level_structure`: a field-dressed X→B diagram for
  one optical transition. Every level is drawn as a bar segmented by its zero-field parent
  character — hyperfine `(F1, F)` parents in X, the two Λ-doublet parity parents in B —
  so Stark mixing is visible directly. Levels are matched to parents by adiabatic tracking
  from zero field, so the labels stay correct above the fields where one-shot matching
  breaks. `E` (V/cm) and `B` (Gauss) are both along z, so mF stays good and the
  calculation runs per mF block. Cross-checked against an independent hand-rolled
  calculation in `tests/utils/test_level_diagram.py`.
- `tests/test_constants.py::test_rust_constants_match_scipy_derived_python`, which pins the
  frozen literals in `rust/src/constants.rs` to the scipy-derived Python values. A CODATA
  revision in a newer SciPy now fails loudly instead of silently desynchronising the two
  backends.

### Changed

- Fundamental constants come from `scipy.constants` instead of being inlined, and Γ is
  derived from the measured B³Π₁(v'=0) lifetime `B_LIFETIME` = 99(9) ns
  (Γ = 1/τ ≈ 1.0101e7 s⁻¹, Γ/(2π) ≈ 1.608 MHz) rather than hard-coded.
- `rust/src/constants.rs` mirrors the same derivations, and the X→B transition dipole is a
  named `ED_XTB` constant rather than three inlined literals in `eval.rs`.
- The golden Hamiltonian pickles under `tests/hamiltonian/` were regenerated to match the
  new X rotational model.

## 0.2.3

State identification now warns about assignments that are genuinely ambiguous instead of
about state labels that are merely impure, and `reorder_evecs` orders eigenvectors with an
optimal assignment rather than a greedy per-eigenvector argmax. Automatic main-state
selection no longer prefers a weak mF = 0 pair over a much stronger one.

### Compatibility notes

- `find_exact_states_indices` and `find_exact_states` no longer warn on low overlap by
  default: `overlap_threshold` now defaults to `None`. Pass `overlap_threshold=0.5` to
  restore the previous behaviour. The old check measured label purity, which ordinary
  Stark mixing drives below 0.5 (every X state above roughly 400 V/cm, every B state by
  100 V/cm when both Lambda-doublet partners are retained) while the returned assignment
  stays exact — verified against adiabatic continuation from zero field.
- Returned indices are unchanged; only the warning behaviour and the eigenvector ordering
  in strongly mixed cases differ.

### Added

- `margin_threshold` (default 0.02) on `find_exact_states_indices` and
  `find_exact_states`. It is the last parameter of both signatures, so existing
  positional calls keep their meaning. It warns when the gap between the best and second-best overlap is
  small or negative, which is the condition under which a label can land on the wrong
  eigenvector. The warning names the competing eigenstate and its overlap. The default is
  chosen so the warning stays silent wherever single-shot matching is actually correct, in
  X to 10 kV/cm and in B to 200 V/cm; a larger threshold produces false alarms on ordinary
  builds (a J′=2 F₁=3/2 F=2 target at 171.6 V/cm flags at 0.12). It is a sufficient, not a
  necessary, signal: above ~10 kV/cm in X or ~500 V/cm in B with both parities retained,
  single-shot matching is unreliable regardless of the margin and states should be tracked
  adiabatically.

### Fixed

- `reorder_evecs` matches eigenvectors to the reference with `linear_sum_assignment`
  instead of `argsort(argmax(...))`. The greedy version had no uniqueness guarantee: once
  two eigenvectors claimed the same reference column it returned a silently arbitrary
  ordering. Output is identical wherever the greedy result was well defined; in X, J=0-3
  at 20 kV/cm the total overlap with the reference improves from 26.20 to 44.56. The
  change costs about 0.5 ms per OBE build. When `V_ref` has fewer columns than `V_in` the
  unmatched eigenvectors are appended in their original order, so the output still holds
  every eigenvector as it did before.
- The field-mixed fallback in `select_main_states_indices_coupling` preferred an mF = 0
  ground state unconditionally, so it could return a pair far weaker than the strongest
  available — contradicting its own docstring. `main_coupling` divides the whole coupling
  matrix, so that silently scales up every Rabi rate for workflows that set `Ω` directly,
  and the weak pair can sit above `weak_main_fraction` and escape the warning. Measured on
  the repository's own fallback test case (X J=2 F₁=5/2 F=3 to B J=1 F₁=3/2 F=1 at
  200 V/cm), the selected pair was 20% of the strongest, a 5x inflation. The preference now
  applies only while the mF = 0 pair stays within `mF0_preference_fraction` (default 0.5)
  of the strongest coupling. Pass `mF0_preference_fraction=0.0` for the previous behaviour.
  Pass 1, which handles every bare-allowed case, is unchanged.
- The weak-`main_coupling` warning compared `|ME_main|`, evaluated for `pol_main`, against
  the largest element of *any* polarization's coupling matrix, and derived its "has been
  pruned" claim from that same cross-polarization threshold. It now uses the matrix built
  with `pol_main` when one is present. Its closing sentence also claimed the requested
  power would map to a larger Rabi rate than intended, which is wrong for the
  `power_to_rabi_*` helpers, where the normalization cancels.

## 0.2.2

Transition validity during OBE setup is now decided by the mixed-state dipole matrix
element rather than by applying the E1 selection rules to bare state labels. In an
electric or magnetic field the eigenstates are superpositions, so nominally forbidden
pairs can be genuinely driveable; previously such a system could not be built at all.

### Compatibility notes

- `check_transitions_allowed` is no longer called during OBE setup and is deprecated.
  It applied the selection rules to `state.largest`, which cannot see field mixing. The
  rules are still consulted, but only to explain why a matrix element vanished.
- A transition is rejected only when the mixed-state matrix element between the
  field-dressed main states is zero. At zero field P, F and mF remain good quantum
  numbers, so rule-violating elements vanish identically and the numeric test reproduces
  the previous rule-based behaviour exactly.
- Results for existing builds are unchanged. The canonical `P2_F1_3o2_F1` system at
  200 V/cm keeps its level count and `main_coupling` to twelve significant digits, and
  automatic main-state selection returns the same pair as before at every field.

### Added

- `generate_coupling_field` warns when `main_coupling` falls below `weak_main_fraction`
  (default 1e-2) of the strongest element in the coupling matrix. `main_coupling`
  normalizes the whole matrix, so a weakly coupled main pair silently inflates every
  Rabi rate. The warning also reports when the main element has been pruned outright.
- `select_main_states_indices_coupling` selects the main pair from mixed-state matrix
  elements. It prefers bare-allowed pairs, falling back to the strongest field-mixed
  coupling only when no bare-allowed pair exists.

## 0.2.1

This release substantially reduces Lindblad setup time and improves Rust batch and
grid throughput while preserving numerical results for existing population-based
workflows.

### Compatibility notes

- `OBESystem` is now a regular class rather than a dataclass. Code using
  `dataclasses.fields`, `dataclasses.asdict`, or generated dataclass equality must
  migrate to direct attribute access.
- `OBESystem.system` and `OBESystem.dissipator` are now lazy cached properties.
  Accessing either property materializes the corresponding symbolic matrix.
- The `method` argument accepted by the OBE setup helpers is deprecated and no
  longer controls symbolic construction. Explicitly passing `"expanded"` or
  `"matrix"` emits `DeprecationWarning`; omit the argument instead.
- Multi-entry collapse operators now use the complete Lindblad jump term. Results
  can change where the previous `fast=True` path incorrectly used only the first
  nonzero entry.
- `apply_per_level_rotating_frame` is opt-in. Populations and population-weighted
  integrals are frame invariant, while off-diagonal coherences are returned in the
  rotated frame.

### Performance

- Build B-state Hamiltonians once and reuse them across transition discovery and
  final reduced-system construction.
- Build symbolic systems and dissipators only when consumers request them.
- Reuse Rust batch workers and optimize packed sparse RHS evaluation.
