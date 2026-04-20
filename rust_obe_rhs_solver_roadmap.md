# Rust OBE RHS and Solver Roadmap for `centrex-tlf`

## Purpose

This document is a **detailed implementation roadmap** for adding a **Rust-backed matrix-method Lindblad backend** to `centrex-tlf`, focused on the **static-field / fixed-basis** use case first.

The intended consumer is both:

- a human developer deciding on architecture and staging
- Codex or another coding agent that can read this plan together with the actual source tree and produce a more refined implementation plan

This is **not** intended to be the final design spec. It is a staged roadmap that should be refined after inspecting the current source code in detail and after early benchmarks.

---

## Executive summary

### Main recommendation

Build a new **Rust static-field Lindblad backend** around the following principles:

1. **Keep the current physics construction flow** in Python.
2. **Limit the Rust v1 path to the matrix method**.
3. **Assume static fields only** for the first implementation.
4. Use a **packed Hermitian real-valued state representation** for the propagated density matrix.
5. Implement **RHS** and **JVP** first.
6. Make an **explicit adaptive RK solver** the default static solver path.
7. Add an **optional stiff solver path** on top of the same kernels.
8. Defer dynamic fields, interpolated field-dependent operator assembly, scans, callbacks, and full Jacobian assembly.

### Why this is the right v1

This matches the dominant actual use case:

- fixed basis
- static fields
- mostly non-stiff explicit integration in practice
- occasional need for stiff support
- desire to exploit Hermiticity and structured dissipators
- no desire for runtime compilation

### What should *not* be done in v1

Do **not** start with:

- dynamic fields
- runtime Rust code generation or compilation
- the expanded symbolic backend
- a fully redesigned front-end physics API
- explicit full Jacobian as the main target
- full Julia runtime feature parity

---

## Current repo context

The current repos already provide the right broad separation between **physics construction** and **runtime execution**.

### Relevant pieces in `centrex-tlf`

These are the main files/modules that matter for the new backend:

- `centrex_tlf/lindblad/utils_setup.py`
  - builds `OBESystem`
  - generates `H_symbolic`, `C_array`, `system`, and related Lindblad ingredients
- `centrex_tlf/lindblad/generate_system_of_equations.py`
  - symbolic density matrix generation
  - Hamiltonian term generation
  - dissipator term generation
- `centrex_tlf/lindblad/...`
  - other utility functions used for compacting states, decay handling, etc.
- `rust/src/lib.rs`
  - existing Rust/PyO3 entry points for Hamiltonian and coupling-related kernels
- `pyproject.toml`
  - already uses `maturin` + `pyo3`

### Relevant pieces in `CeNTREX-TlF-julia-extension`

These provide the current reference runtime/backend design:

- `centrex_tlf_julia_extension/lindblad_julia/utils_setup.py`
  - constructs the Julia-side OBE system
  - has the split between `expanded` and `matrix` methods
- `centrex_tlf_julia_extension/lindblad_julia/utils_solver.py`
  - problem setup
  - solve helpers
  - parameter scans
  - output helpers
- `centrex_tlf_julia_extension/lindblad_julia/utils_julia_matrix.py`
  - matrix-method Lindblad wrappers and functors
- `centrex_tlf_julia_extension/lindblad_julia/utils_julia_matrix_assemble.py`
  - Hermitian-aware Hamiltonian and dissipator assembly code generation
- `centrex_tlf_julia_extension/lindblad_julia/julia_common.jl`
  - Hermitian commutator helper and common modulation/helper functions
- `centrex_tlf_julia_extension/lindblad_julia/ode_parameters.py`
  - current parameter handling and compound variable logic

### Implication

The new Rust backend should **reuse the current Python-side problem construction** as much as possible and should only replace the **Lindblad execution layer**.

---

## Scope for v1

### In scope

- static fields only
- fixed basis only
- matrix method only
- packed Hermitian state representation
- Rust RHS kernel
- Rust JVP kernel
- explicit adaptive solver path
- optional stiff solver path
- Python-side plan preparation for static problems
- validation against current Python/Julia behavior

### Out of scope

- dynamic fields
- time-dependent interpolation of operator data
- basis tracking / instantaneous basis updates
- expanded symbolic backend in Rust
- parameter scans / ensemble solves
- callback/output-function parity with Julia
- preconditioners in first pass
- full direct Jacobian assembly as first-class API
- steady-state solver specialization
- matrix exponential / Krylov exponential static propagators

---

## Design goals

1. **No runtime compilation** after install.
2. **Static-field hot path should be very cheap**.
3. **Exploit Hermiticity throughout**.
4. **Exploit dissipator structure throughout**.
5. **Preserve the current front-end construction flow** as much as possible.
6. **Make the first backend useful before it is perfect**.
7. **Build around kernels that remain useful for stiff solvers later**.
8. **Make dynamic fields an additive later step, not a prerequisite**.

---

## Architectural overview

### Big picture

The recommended architecture has three layers.

### Layer 1: physics construction (existing Python path)

Use existing code to construct the static problem:

- states
- reduced Hamiltonian
- couplings
- compacted state sets if needed
- `H_symbolic`
- `C_array`
- `OBESystem`

This layer remains in Python.

### Layer 2: static plan preparation (new Python layer)

Convert `OBESystem` into a backend-neutral **static Lindblad plan**.

This layer should:

- inspect and validate the problem
- choose a packed state layout
- lower static Hamiltonian data
- lower static dissipator/collapse data
- optionally choose/precompute a full packed Liouvillian
- package all of this into a Rust-friendly payload

### Layer 3: runtime kernels + solver (new Rust layer)

The Rust backend consumes the static plan and provides:

- `rhs(y)`
- `jvp(v)`
- explicit solve
- optional stiff solve

The runtime hot path should avoid all symbolic work.

---

## Static-field mathematical model

For static fields, fixed basis, and fixed couplings/collapse structure, the system is a **linear time-independent ODE** in the propagated state.

At the density-matrix level:

- `dρ/dt = L ρ`

At the packed real state level:

- `dy/dt = A y`

where `A` is the representation of the static Liouvillian in the packed Hermitian basis.

This means:

- the RHS is linear
- the JVP is the same operator applied to another vector
- the Jacobian is constant
- the static case is much simpler than the general time-dependent one

### Important implementation consequence

For static fields, there are two viable execution styles:

#### Style A: operator-free structured kernels
Apply:

- commutator kernel
- dissipator kernel

on each RHS call, using preprocessed static operator data.

#### Style B: preassembled packed Liouvillian
Build the full packed-basis operator once:

- `rhs(y) = A y`
- `jvp(v) = A v`

This may be excellent for small/moderate systems.

### Recommendation

Implement **both**, but:

- start with structured kernels and reference dense implementations
- add optional preassembled Liouvillian mode later in the same milestone
- benchmark and choose heuristics afterward

---

## State representation

### Recommended representation

Use a **packed Hermitian real state vector**.

For an `n x n` Hermitian density matrix:

- store all `n` diagonal entries as real scalars
- for each upper-triangle element `(i, j)` with `i < j`, store:
  - `Re(ρ[i, j])`
  - `Im(ρ[i, j])`

This gives a real vector of length `n^2`.

### Why this representation

- exploits Hermiticity
- avoids redundant conjugate storage
- avoids a `2 n^2` real representation
- matches standard real ODE solver interfaces well
- makes JVP and Jacobian interpretation straightforward

### Required helpers

The backend needs reliable:

- `pack_density_matrix(full_complex_matrix) -> packed_real_vec`
- `unpack_density_matrix(packed_real_vec) -> full_complex_matrix`

These should exist both in Python reference code and in Rust.

### Layout object

A single layout object should define:

- state dimension `n`
- packed length `n^2`
- diagonal index mapping
- upper-triangle index mapping
- reverse mapping for testing/debugging

---

## Python-side static plan layer

### Goal

Create a plan-preparation layer that is completely static and backend-oriented.

### Recommended new modules

Suggested Python files/modules to add:

- `centrex_tlf/lindblad/plan_static.py`
- `centrex_tlf/lindblad/state_layout.py`
- `centrex_tlf/lindblad/backends/base.py`
- `centrex_tlf/lindblad/backends/rust.py`
- `centrex_tlf/lindblad/solve.py`

Optional helper files:

- `centrex_tlf/lindblad/reference_dense.py`
- `centrex_tlf/lindblad/reference_packed.py`

### Core Python objects

#### `PackedHermitianLayout`
Responsibility:

- build and expose packed index maps
- validate dimensions
- provide helper conversions for tests

#### `StaticHamiltonianPlan`
Responsibility:

- hold static Hamiltonian data for Rust
- possible forms:
  - upper-triangle dense data
  - structured terms grouped by type

#### `StaticDissipatorPlan`
Responsibility:

- hold static collapse or dissipator structure
- allow either:
  - raw collapse operators
  - preprocessed sparse update structure
  - both, if useful for reference vs fast modes

#### `StaticLindbladPlan`
Responsibility:

- main backend-facing object
- contains:
  - layout
  - Hamiltonian plan
  - dissipator plan
  - optional static Liouvillian
  - mode flags / metadata

### Required Python entry points

#### `prepare_static_lindblad_plan(obe_system, backend="rust", ...)`
Tasks:

- validate static-field assumptions
- validate matrix-method assumptions
- create packed layout
- extract static operator information from `OBESystem`
- lower into Rust-friendly payload
- possibly choose reference/fast/preassembled modes

#### `solve_lindblad(...)`
Tasks:

- convenience wrapper
- build plan if necessary
- select solver backend
- call explicit or stiff solver
- return user-facing result object

### Important note

For this static-only roadmap, do **not** redesign `odeParameters` yet. If needed, simply consume the numerical values already substituted into the static problem or the static operator data already implicit in the generated `OBESystem`.

---

## Rust-side module roadmap

### Suggested Rust module tree

Add a new Lindblad subtree under `rust/src`:

- `lindblad/mod.rs`
- `lindblad/layout.rs`
- `lindblad/plan.rs`
- `lindblad/pack.rs`
- `lindblad/reference_rhs.rs`
- `lindblad/commutator.rs`
- `lindblad/dissipator.rs`
- `lindblad/rhs.rs`
- `lindblad/jvp.rs`
- `lindblad/liouvillian.rs`
- `lindblad/solver_explicit.rs`
- `lindblad/solver_stiff.rs`
- `lindblad/python_api.rs`

### Suggested main Rust types

#### `PackedHermitianLayout`
Fields:

- `n: usize`
- diagonal index tables
- upper-triangle index tables
- packed length

#### `StaticHamiltonianPlan`
Possible fields:

- upper-triangle dense complex data
- static term lists if structured assembly is preferred
- optional split between diagonal and off-diagonal parts

#### `StaticDissipatorPlan`
Possible fields:

- raw collapse operators for reference path
- sparse structure for fast path
- precomputed `C†C`-related structure

#### `StaticLindbladPlan`
Fields:

- layout
- Hamiltonian plan
- dissipator plan
- optional preassembled packed Liouvillian
- selected execution mode

#### `ExecutionMode`
Enum suggestions:

- `ReferenceDense`
- `StructuredKernels`
- `PreassembledLiouvillian`

---

## Phase-by-phase implementation roadmap

# Phase 0: architecture freeze and design note

## Goal

Agree on interfaces before writing kernels.

## Tasks

1. Write a short design note in the repo.
2. Freeze v1 scope.
3. Freeze the static-only matrix-method assumption.
4. Freeze packed Hermitian state layout.
5. Freeze the distinction between:
   - plan preparation
   - runtime kernel execution
   - solver layer

## Deliverables

- short design markdown in repo
- list of v1 non-goals

---

# Phase 1: Python plan preparation layer

## Goal

Get a backend-neutral static plan object in place.

## Tasks

1. Add `PackedHermitianLayout` in Python.
2. Add `StaticHamiltonianPlan` in Python.
3. Add `StaticDissipatorPlan` in Python.
4. Add `StaticLindbladPlan` in Python.
5. Implement `prepare_static_lindblad_plan(...)`.
6. Add a placeholder Rust backend adapter.
7. Add simple serialization/packing for Rust calls.

## Key design choice

At this stage, the Python side should still use current `generate_OBE_system(...)` and existing Lindblad generation machinery. The new code should only **extract and lower** static operator information.

## Deliverables

- `plan_static.py`
- `state_layout.py`
- plan object construction tests

---

# Phase 2: Rust reference implementation

## Goal

Get a correct Rust RHS/JVP path before optimizing anything.

## Tasks

1. Implement packed layout in Rust.
2. Implement pack/unpack functions.
3. Implement reference dense commutator.
4. Implement reference dense dissipator.
5. Implement reference RHS using:
   - unpack packed state
   - reconstruct full Hermitian matrix
   - apply dense commutator
   - apply dense dissipator
   - repack result
6. Implement JVP reference by applying the same operator to an arbitrary packed vector.
7. Expose these functions to Python through PyO3.

## Notes

The goal here is correctness and testability, not speed.

## Deliverables

- Rust reference kernels
- Python bindings
- comparison tests against Python reference implementation

---

# Phase 3: fast packed commutator kernel

## Goal

Replace the dense/reference commutator path with a Hermitian-aware fast path.

## Tasks

1. Decide first data representation for static Hamiltonian in Rust:
   - dense upper triangle
   - structured term lists
2. Implement a packed or upper-triangle-aware commutator kernel.
3. Also implement an alternative scratch full-matrix path for benchmarking.
4. Keep both behind one interface.
5. Benchmark both on representative static systems.

## Recommendation

Do **not** assume upfront that the BLAS-style full-matrix path will always win. For moderate state sizes, a specialized packed upper-triangle kernel may be better.

## Deliverables

- `commutator.rs`
- benchmark comparing direct packed vs scratch full-matrix path

---

# Phase 4: fast packed dissipator kernel

## Goal

Exploit dissipator structure in the static case.

## Tasks

1. Decide on static dissipator representation:
   - raw static collapse operators only
   - preprocessed sparse updates
   - both
2. Add preprocessing on the Python side or Rust side to lower the static collapse structure.
3. Implement packed Hermitian dissipator application.
4. Compare against the dense/reference dissipator path.
5. Validate trace preservation and correct symmetry behavior.

## Important recommendation

The dense generic dissipator should remain only as the reference/debug path. The production path should use structural information.

## Deliverables

- `dissipator.rs`
- static dissipator preprocessing path
- validation tests

---

# Phase 5: fast static RHS and JVP

## Goal

Expose the fast static operator path as the primary kernel interface.

## Tasks

1. Implement `rhs_static_fast(plan, y) -> dy`.
2. Implement `jvp_static_fast(plan, v) -> out`.
3. Wire execution mode selection:
   - reference dense
   - structured kernels
4. Add performance counters / debug hooks if useful.
5. Benchmark against Julia matrix-method results for small systems.

## JVP note

For the static linear case, the JVP is the same operator action on another vector. So the same core operator application should be reusable.

## Deliverables

- `rhs.rs`
- `jvp.rs`
- Python-callable fast kernels

---

# Phase 6: optional preassembled static Liouvillian

## Goal

Add a second static execution mode that may be faster for smaller systems.

## Tasks

1. Implement assembly of the full packed-basis Liouvillian matrix.
2. Add a mode where:
   - `rhs(y) = A y`
   - `jvp(v) = A v`
3. Compare this against structured kernels on representative system sizes.
4. Add plan metadata to indicate whether a preassembled operator is present.
5. Add heuristics for when this mode should be preferred.

## Recommendation

This should be optional and benchmark-driven, not mandatory.

## Deliverables

- `liouvillian.rs`
- operator assembly tests
- performance comparison results

---

# Phase 7: explicit solver backend

## Goal

Provide the normal workflow first.

## Context

The dominant current static-field workflow uses RK45/Tsit5-like explicit adaptive methods. Therefore the Rust backend should first expose an explicit adaptive solver path that feels like the current normal workflow.

## Tasks

1. Define a minimal solver config object.
2. Implement an explicit adaptive solver backend in Rust.
3. Expose config options:
   - `abstol`
   - `reltol`
   - `dt` or initial step if needed
   - `saveat`
   - `save_start`
4. Keep the returned result object minimal and useful.
5. Match current user expectations where practical.

## Notes

Do not try to clone all Julia solver config/runtime semantics immediately.

## Deliverables

- `solver_explicit.rs`
- Python `solve_lindblad(..., solver="explicit")`
- tests comparing against known explicit runs

---

# Phase 8: stiff solver backend

## Goal

Add optional stiff support using the same kernels.

## Kernel requirements

Use:

- `rhs`
- `jvp`

Do not require full dense Jacobian first.

## Tasks

1. Define a stiff solver config.
2. Implement the first stiff backend around the same static plan.
3. Support matrix-free linearization through JVP.
4. Add validation against known stiff or semi-stiff cases.
5. Benchmark explicit vs stiff path on representative cases.

## Recommendation

Treat the stiff path as optional and additive, not the main v1 runtime.

## Deliverables

- `solver_stiff.rs`
- Python `solve_lindblad(..., solver="stiff")`
- benchmark report

---

# Phase 9: validation against current code paths

## Goal

Establish trust before heavy optimization.

## Test categories

### Structural tests

- pack/unpack roundtrip
- layout index correctness
- Hermiticity reconstruction

### Kernel correctness tests

- Rust reference RHS vs Python dense reference RHS
- fast packed RHS vs Rust reference RHS
- fast packed dissipator vs dense reference dissipator
- JVP vs operator application on basis vectors

### Cross-backend tests

- Rust static RHS vs current Julia matrix backend on small systems
- Rust explicit solver vs current Julia Tsit5-like results on small systems

### Physics sanity checks

- trace preservation
- real populations
- pure decay limits
- simple few-level coherent dynamics

## Deliverables

- unit tests
- integration tests
- backend comparison fixtures

---

# Phase 10: benchmarking and heuristics

## Goal

Choose good defaults based on measured behavior.

## Benchmarks to run

1. reference dense RHS vs packed fast RHS
2. structured commutator vs scratch full-matrix commutator
3. structured dissipator vs dense dissipator
4. structured kernels vs preassembled Liouvillian
5. Rust explicit solver vs current Julia workflow
6. explicit solver vs stiff solver on borderline-stiff static problems

## Output

A small heuristic for internal mode selection, for example:

- use preassembled Liouvillian below some size threshold
- use structured kernels above that threshold
- recommend stiff solver only when explicit runtime/step count crosses a threshold

## Deliverables

- benchmark scripts
- benchmark results markdown
- internal heuristics

---

## Explicit list of deferred work

The following should be deferred until after the static backend is working well.

### Dynamic-field features

- field interpolation
- time-dependent operator assembly
- redesigned parameter/coefficient graph for dynamic use

### Advanced runtime features

- parameter scans / ensemble solves
- callbacks
- output functions / quadratures
- progress bars
- distributed execution

### Advanced numerical features

- preconditioners
- full dense Jacobian as a primary interface
- steady-state solvers
- matrix exponential propagators
- specialized Krylov exponential methods

### API refactors

- full `odeParameters` replacement
- major `TransitionSelector` redesign

---

## Recommended implementation order in practice

If this work is handed to Codex or another coding agent, I would recommend the following working order:

1. Inspect current Python Lindblad construction path in detail.
2. Implement Python `PackedHermitianLayout`.
3. Implement Python `StaticLindbladPlan`.
4. Implement Rust pack/unpack.
5. Implement Rust reference RHS/JVP.
6. Add cross-check tests against Python reference code.
7. Implement fast static commutator kernel.
8. Implement fast static dissipator kernel.
9. Implement fast RHS/JVP.
10. Add optional preassembled Liouvillian mode.
11. Implement explicit adaptive solver.
12. Implement optional stiff solver.
13. Add benchmarks and heuristics.

---

## Suggested questions for Codex to resolve from the source tree

When Codex reads this plan together with the source tree, it should answer at least these questions before locking in implementation details:

1. Which exact objects in `OBESystem` are easiest to lower directly for a static matrix backend?
2. Is it better to lower `H_symbolic` to static dense upper-triangle values on the Python side or in Rust?
3. What is the best static representation of the collapse operators for fast dissipator application?
4. Which parts of the current Julia matrix assembly logic can be mirrored directly without source generation?
5. For representative state sizes in this project, is a preassembled packed Liouvillian competitive with structured kernels?
6. Is there already enough Rust infrastructure in the main crate to house the Lindblad backend cleanly, or should the new modules be further isolated?
7. Which result/config objects from the Julia extension are most worth mirroring in the Python-facing Rust solver API?

---

## Final recommendation

For the first serious Rust backend in `centrex-tlf`, implement a **static-field, matrix-only, packed-Hermitian Lindblad runtime** with:

- fast static RHS
- fast static JVP
- explicit adaptive solver as the default
- optional stiff solver on the same plan

and keep everything else out of scope until this path is validated and benchmarked.
