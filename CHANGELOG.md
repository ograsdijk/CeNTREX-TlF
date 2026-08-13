# Changelog

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
