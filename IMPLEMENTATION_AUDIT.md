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

## Suggested Next Order

1. Replace or raise in `B_uncoupled.HZx/HZy`.
2. Benchmark and document the Hamiltonian lowering `"auto"` heuristic before
   treating it as stable.
3. Benchmark the new Rust Hamiltonian assembly path at representative basis
   sizes, especially the 64-state and larger cases used in normal workflows.
