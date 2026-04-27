# Lindblad OBE & Effective Hamiltonian — Improvement Plan

## Original Benchmark Baseline (R(0) F1'=3/2 F'=2, 65 states, 10µs)

| Solver         | Exec Mode          | Backend | Median (ms) |
|----------------|--------------------|---------|-------------|
| dopri5         | structured         | rust    | 117         |
| dopri5         | structured_blas    | rust    | 124         |
| dopri5         | structured_upper   | rust    | 344         |
| dopri5         | reference          | rust    | 58,892      |
| scipy RK45     | structured         | rust    | 221         |
| scipy BDF      | structured         | rust    | 3,359       |
| scipy Radau    | structured         | rust    | 5,037       |
| scipy RK45     | structured         | python  | 4,517       |

## Current Benchmark Baseline (after all improvements)

| Solver         | Exec Mode          | Backend | Median (ms) | vs Original Baseline |
|----------------|--------------------|---------|-------------|----------------------|
| dopri5         | structured         | rust    | 110         | ~same                |
| dopri5         | structured_upper   | rust    | **40**      | **8.6x**             |
| dopri5         | reference          | rust    | 41,594      | 1.4x                 |
| scipy RK45     | structured         | rust    | 229         | ~same                |
| scipy BDF      | structured         | rust    | 1,193       | **2.8x**             |
| scipy BDF      | structured_upper   | rust    | **436**     | **7.7x**             |
| scipy Radau    | structured         | rust    | 1,470       | **3.4x**             |
| scipy Radau    | structured_upper   | rust    | **667**     | **7.6x**             |
| scipy RK45     | structured         | python  | 4,591       | ~same                |

All solvers agree to within ~1e-8 in final populations.

### Per-RHS-call performance (500 calls, n=65)

| Mode              | µs/call | Bottleneck                |
|-------------------|---------|---------------------------|
| structured        | 192     | commutator (97%, BLAS)    |
| structured_upper  | **31**  | commutator (83%, sparse)  |
| reference         | 116,823 | dissipator (99.7%)        |

---

## Completed

### Removed duplicate custom DOPRI5 solver

Deleted `solver.rs` (custom DOPRI5 implementation). The `ode_solvers` crate path
(`solver_ode.rs`) was the sole Rust solver. `solver="explicit"` is an alias
for `solver="dopri5"` in Python. Removed the `save_everystep` parameter.

### Removed execution mode aliases

`"structured_blas"`, `"structured_jumps"`, `"reference_dense"`, and `"dense"`
aliases removed. The three execution modes are now:
- `"reference"` — naive dense commutator + dense dissipator
- `"structured"` — dense commutator (BLAS when available) + structured dissipator
- `"structured_upper"` — sparse-H commutator + structured dissipator (upper-tri)

### Precomputed C†C in dense dissipator

Added `dense_cdagger_c: Vec<Complex64>` to `PreparedLindbladPlan`, computed once
during `parse_plan_payload`. `add_dense_dissipator` now reads precomputed C†C
instead of recomputing per call.

**Result:** reference mode dissipator 158ms → 120ms per 500 calls (1.3x).

### Sparse-H commutator for `structured_upper`

Replaced the branch-per-element `add_commutator_upper` with a sparse commutator
that exploits the Hamiltonian's sparsity pattern:

1. **`HermitianSparsePattern`** — built at preparation time from the Hamiltonian
   plan's structural nonzero pattern. Stores CSR (upper triangle) and CSC
   (transpose lookup for Hermitian lower-triangle access).

2. **`fill_sparse_h_values`** — extracts evaluated H values into compact sparse
   array each step (O(nnz), negligible).

3. **`add_commutator_sparse_upper`** — iterates only over nonzero H entries in the
   inner k-loop. For the R(0) system (nnz=69, n=65), this reduces from 65
   multiply-accumulates per (i,j,k) to ~1-2.

**Result:** `structured_upper` commutator 486ms → 13ms per 500 calls (**37x**).
Full solve 344ms → 44ms (**7.8x**), now 2.6x faster than `structured` with BLAS.

### Packed-state BDF/Radau solver

Replaced the split-real scipy path (state size 2n²) with a packed-state path
(state size n²):

1. **`build_packed_jacobian_sparse`** — probes the packed RHS directly, producing
   an n²×n² Jacobian instead of 2n²×2n².

2. **`_solve_scipy_with_rust_packed_rhs`** — calls `rhs_packed_py` and
   `jacobian_packed_sparse_py`, returns `LindbladResult` directly (no
   split-real ↔ complex conversion overhead).

The Jacobian is 4x smaller, so LU factorization (the dominant cost in implicit
solvers) scales as (n²)³ vs (2n²)³ = 8x less work.

**Result:** BDF 3,359ms → 1,225ms (**2.7x**) with `structured`, 421ms (**8.0x**)
with `structured_upper`. Radau 5,037ms → 1,490ms (**3.4x**) with `structured`,
700ms (**7.2x**) with `structured_upper`.

### Time-independent Hamiltonian caching

Added `is_time_dependent: bool` to `PreparedLindbladPlan`, computed during
`parse_plan_payload` by scanning all compiled expressions for the `Time` opcode.
Added `hamiltonian_valid` and `h_sparse_valid` flags to `RhsWorkspace`.

When the problem is time-independent and the Hamiltonian has already been
evaluated, subsequent RHS calls skip:
- Parameter graph evaluation
- Hamiltonian matrix fill
- Sparse H value extraction (for `structured_upper`)

**Result:** `structured_upper` per-call: parameter_eval 1.3% → 0.1%,
hamiltonian_fill 3.6% → 0.2%. Full solve 44ms → 40ms (**10%**).

### IR expression system extensions

Added `gaussian_1d(x, center, sigma)` (helper ID 15) and
`pchip_interp(x, grid, values)` (helper ID 16) to the Lindblad IR expression
system. Both work end-to-end through the Rust bytecode evaluator.

- `gaussian_1d`: evaluates `exp(-(x-center)²/(2σ²))` as a single helper call
- `pchip_interp`: monotone piecewise cubic Hermite interpolation with full
  coefficient computation in Rust. Uses the same tuple-argument mechanism as
  `linear_interp`.
- `pchip_tabulated()` convenience function in `parameters.py`

### Polarization switching uses `square_wave` helper

Changed `generate_lindblad_parameters` to use `square_wave(t, ω, φ)` instead of
the 3-step `sin(ω*t+φ) > 0` chain for two-polarization microwave transitions.

**Result:** 24% faster for time-dependent polarization switching (fewer compound
parameter evaluations and one fewer `sin` call per step).

### RWA manifold rotation fix

Fixed `generate_unitary_transformation_matrix` in `generate_hamiltonian.py` to
propagate rotation frequencies within `(electronic_state, J)` manifolds.

Previously, only states directly coupled by the laser received rotation
frequencies; uncoupled states (e.g., mF≠0 for Z polarization, or F1=3/2 for
Q1 F1'=1/2 transitions) remained in the lab frame with GHz-scale absolute
energies in `H_symbolic`.

Added `_build_manifold_indices(QN)` which groups states by `(electronic_state, J)`.
After the equation solver determines rotation frequencies for directly coupled
states, all members of the same manifold inherit the same frequency.

**Result:** Q1 F1=1/2 F=0 max `h_internal` diagonal drops from 14,784 MHz to
0.11 MHz. Hyperfine splittings within manifolds preserved. OBE solver backward
compatible (R(0) benchmark unchanged).

### Effective Hamiltonian package

Extracted production-path effective Hamiltonian code from the monolithic
`examples/effective_hamiltonian_runtime.py` (6146 lines) into
`centrex_tlf/effective_hamiltonian/` (12 modules, ~2400 lines):

- `_utility.py`, `_superoperators.py`, `_decay.py`, `_alignment.py`,
  `_embedding.py` — internal helpers
- `operator_bundle.py` — `OperatorBundle` dataclass + transforms
- `compact_reference.py` — `build_compact_reference_decomposed_bundle`
- `models.py` — patch dataclasses + 3 prepared model classes
- `preparation.py` — `prepare_lindblad_safe_compact_interpolated_model`,
  `prepare_instantaneous_interpolated_effective_model`
- `solve.py`, `initial_state.py`, `observables.py`

Supports any TlF P/Q/R optical transition. `transition` and
`optical_polarization` are required parameters (no defaults).

### Rust effective Lindblad solver

Added `rust/src/effective_lindblad/` module for solving the effective Hamiltonian
Lindblad equation in Rust:

- `EffectiveLindbladPlan`: precomputed split-real Liouvillian superoperators at
  field grid points with interval differences for fast linear interpolation.
- RHS: parameter graph evaluation → field grid interpolation → dense matvec.
- Uses the generic `ode::dopri5` solver (63 lines vs 256 in the old standalone).
- `rust_plan.py`: Python preparation layer that builds superoperators from the
  `PreparedLindbladSafeCompactInterpolatedHamiltonianModel`, converts to
  split-real, and serializes to Rust.

**Result:** static Q1 solve: 253ms Rust vs 391ms scipy (1.5x speedup). Final
point accuracy 7.6e-10 vs ultra-tight reference. The system has MHz-scale
Liouvillian eigenvalues after the RWA manifold fix.

### Unified generic ODE module

Replaced 6 Lindblad-specific solver files (~2650 lines) with a generic
`rust/src/ode/` module (~1100 lines):

- `OdeRhs` trait for pluggable right-hand-side evaluation
- `OdeOutput` trait with `FullOutput`, `PopulationsOutput`, `SelectedOutput`,
  `FinalOnlyOutput` — inline output extraction during stepping, no
  post-processing needed
- Generic DOPRI5 and Tsit5 solvers with proper dense output, PI step control,
  stiffness detection, automatic initial step size
- Shared `Controller`, `hinit`, `SavePlan` infrastructure

Both the OBE Lindblad solver and the effective Hamiltonian solver use the same
stepping code via `OdeRhs` implementations (`LindbladRhs`, `EffectiveLindbladRhs`).

Consolidated Python API: 3 Rust endpoints (`solve_lindblad_ode_py`,
`solve_lindblad_batch_ode_py`, `solve_lindblad_grid_ode_py`) replace 12 old
endpoints. `python_api.rs` reduced from 1199 to 629 lines.

Removed `ode_solvers` crate dependency.

**Net result:** -2430 lines of Rust code.

### diffsol BDF solver (removed)

The diffsol BDF solver was added and tested but removed during the ODE module
unification. It was ~4x slower than scipy BDF for the R(0) system due to faer's
sparse LU being less optimized than scipy's SuperLU. The `diffsol` and `faer`
crate dependencies were removed.

If needed in the future, a BDF solver can be added to the `ode/` module using
the `OdeRhs` trait, without external crate dependencies.

### Time-dependent effective Hamiltonian solver

Validated the Rust effective Lindblad solver with time-dependent fields: molecule
flying through a spatially varying electric field with a Gaussian laser beam.
Population agreement 8.9e-7 at matched tolerances (reltol=1e-6).

Added bounds checking: the solver errors if the field coordinate goes outside the
operator grid, preventing silent use of wrong operators.

**Result:** 80ms Rust vs 4.5s Python scipy RK45 — **56x speedup**. The system
is non-stiff after the RWA manifold fix (stiffness ratio 2-5.5), so explicit
DOPRI5 is optimal. BDF is 3x slower than RK45 for this problem.

### Effective Hamiltonian batch support

Added batch solve with per-trajectory parameter overrides using Rayon:
- `parameter_scan(plan, rho0, t_span, parameter_slots=["v"], parameter_batch=velocities)`
- `grid_scan(plan, rho0, t_span, scan={"v": velocities, "omega0": rabi_rates})`
- `EffectiveLindbladBatchResult` dataclass matching the OBE `LindbladBatchResult` pattern
- Serial and parallel dispatch with optional thread count
- Populations and full output modes, saveat and final-only

**Result:** 13.2x speedup vs Python (10 trajectories, parallel).

### Sparse superoperators for effective Hamiltonian

Replaced dense split-real superoperators with CSR sparse format. The Liouvillian
superoperator density for Q1 is 0.1-10%:
- `L_opt`: 0.1-0.3% (240-600 nonzeros out of 202,500)
- `L_det`: 0.1-0.2%
- `L_combined`: 0.2-10.1% (varies with field)

The sparse matvec does 500-20,000 multiply-adds instead of 202,500 (dense).
Dense storage and code path removed entirely.

**Result:** 80ms sparse vs ~1700ms dense — **23x faster**. vs Python: **56x**.

### Precomputed PCHIP coefficients

Added `PchipTable` to `ParameterGraph` with precomputed monotone cubic Hermite
coefficients and interval caching (`pchip_hints` in workspace, persisted across
RHS calls). Python precomputes tables in `_extract_pchip_tables` during
`lower_parameter_graph` and passes them to Rust in the plan payload.

Negligible speedup for single solves (PCHIP cost dwarfed by matvec), but the
infrastructure enables O(1) evaluation in batch mode where the same table is
traversed monotonically by each trajectory.

### RuntimeExpression Python evaluation

Added `.evaluate(**overrides)` and `.evaluate_array(variable, values, **overrides)`
methods to `RuntimeExpression` for debugging, plotting, and interactive use:

```python
z = linear(Time(), offset=z0, slope=v)
z.evaluate(t=25e-6)              # single point
Omega.evaluate_array("t", t_arr) # vectorized
```

Handles scalar parameters, tuple parameters (PCHIP grid/values via
`smp.Tuple` substitution), and helper functions (`pchip_interp`, `gaussian_1d`)
via `lambdify` with the helper function module dict. Also added `__repr__`.

### Instruction op ID safety

Added `#[repr(u8)]` to Rust `InstructionOp` enum, ensuring discriminant values
are fixed at the declared integers. Added 3 Rust tests (value verification,
roundtrip, unknown value rejection) and 1 Python test verifying all 19 opcodes
match between Python `ir.py` and Rust `eval.rs`.

### Output mode parity + weighted integral reduction

Unified output modes across all four solver paths (OBE single, OBE batch,
effective single, effective batch). Added `WeightedIntegralOutput` to
`ode/output.rs` for computing `∫ Σ w_i * y_i dt` inline during stepping using
trapezoidal rule. Covers photon integral (`w = J_ii`) and excited population
integral (`w = 1` for excited states).

All solver endpoints now accept `integral_weights` parameter for
`output="weighted_integral"`.

### Configurable operator grid interpolation

Added `operator_interpolation` parameter to `prepare_effective_lindblad_rust_plan`
(default `"linear"`, optional `"pchip"`). Sparse operator values are stored as
PCHIP cubic coefficients per entry per interval. Linear mode uses `c2=c3=0`
(exact linear interpolation), preserving trace. PCHIP mode uses full monotone
Hermite coefficients for higher accuracy near avoided crossings but may introduce
small trace errors (~0.01%).

### Constant Ω/δ detection

Added `constant_rabi` and `constant_detuning` fields to `EffectiveLindbladPlan`.
During plan construction, `detect_constant_slot` checks if the Rabi rate or
detuning expressions depend on time (directly or through compound dependencies).
If constant, the scalar value is cached and reused every RHS call without
re-evaluating the parameter graph for those slots.

---

## Remaining Items

### 1. Analytical structured Jacobian (OBE)

**Priority: Low** (the effective Hamiltonian solver doesn't use BDF; OBE BDF
uses scipy which handles its own Jacobian via the packed probing path)

`build_packed_jacobian_sparse` still probes with n² basis vectors. An analytical
Jacobian using the Liouvillian superoperator structure would reduce cost from
O(n² × RHS) to O(nnz_H × n + n) for the structured case. Main benefit is for
time-dependent OBE problems where the Jacobian must be rebuilt per step.

**Estimated effort:** ~4-6 hours.

### 2. SIMD for sparse commutator (OBE)

**Priority: Low**

The sparse commutator inner loop does 1-2 complex MACs per output element.
Vectorize the outer loop by processing multiple (i,j) pairs with AVX2,
treating `H[i,k] * ρ[k,j]` as a scalar × vector sweep across j.

**Estimated speedup:** ~1.5-2x on the commutator (~10µs/call saved).
**Estimated effort:** ~4-6 hours (unsafe code, feature detection, fallback).

### 3. Auto-selection heuristic for Hamiltonian lowering

**Priority: Low**

The `"auto"` threshold in `ir.py` has no documented derivation. Benchmark with
multiple system sizes to validate or update the crossover.

**Estimated effort:** ~2 hours.

### 4. BLAS portability

**Priority: Deferred**

`blas.rs` uses `LoadLibraryW` to find scipy's OpenBLAS DLL at runtime
(Windows-only). Future: build with included BLAS for cross-platform support.

### 5. Adaptive coordinate-grid diagnostics

**Priority: Low**

Warn if operator variation between adjacent grid points exceeds a threshold,
suggesting the user add more field points in that region.

**Estimated effort:** ~2 hours.
