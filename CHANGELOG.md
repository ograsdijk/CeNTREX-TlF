# Changelog

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
