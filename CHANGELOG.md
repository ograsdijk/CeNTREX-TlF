# Changelog

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
