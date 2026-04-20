# Lindblad OBE Implementation — Improvement Plan

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
| bdf (diffsol)  | structured_upper   | rust    | 1,729       | 2.0x                 |
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
(`solver_ode.rs`) is now the sole Rust solver. `solver="explicit"` is an alias
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

### diffsol BDF solver (Rust-native stiff solver)

Added `solver="bdf"` using the `diffsol` crate with `FaerSparseLU` for sparse
LU factorization. Entirely in Rust — no Python in the solve loop.

- Uses `OdeBuilder` with `rhs_implicit` closures wrapping `rhs_packed_into`
- Jacobian-vector product `J·v = rhs(v)` (exact, since the system is linear)
- `use_coloring(true)` for NaN-based sparsity detection + graph coloring
- `FaerSparseMat<f64>` matrix type with `FaerSparseLU<f64>` linear solver

**Result:** 1,450ms with `structured_upper`. Slower than scipy BDF (421ms)
because faer's sparse LU is less optimized than scipy's SuperLU for this
Jacobian sparsity pattern. Correct results (agreement to ~1e-8).

**Limitation:** diffsol 0.12 seals the `OdeEquationsRef` trait, preventing
custom `OdeEquations` implementations. This blocks `jacobian_inplace` —
the only way to customize the Jacobian is through the `OdeBuilder` closure API.
A future diffsol version may expose this, or a fork could be used.

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

---

## Remaining Items

### 1. Instruction op ID coupling between Python and Rust

**Priority: Medium**

`InstructionOp` IntEnum values in `ir.py` must match the integer dispatch in
Rust's `eval.rs` by convention. A reorder in either language silently breaks the
other.

**Plan:**
- Add `#[repr(u8)]` to Rust `InstructionOp` enum and explicitly assign values.
- Add a cross-language unit test for opcode mapping.

**Estimated effort:** ~1 hour.

### 2. Analytical structured Jacobian

**Priority: Medium** (less urgent now that packed Jacobian reduced BDF cost)

`build_packed_jacobian_sparse` still probes with n² basis vectors. An analytical
Jacobian using the Liouvillian superoperator structure would reduce cost from
O(n² × RHS) to O(nnz_H × n + n) for the structured case. Main benefit is for
time-dependent problems where the Jacobian must be rebuilt per step.

**Estimated effort:** ~4-6 hours.

### 3. Improve diffsol BDF performance

**Priority: Low-Medium**

The diffsol BDF is ~4x slower than scipy BDF for this system. Possible paths:
- Wait for diffsol to unseal `OdeEquationsRef` and implement `jacobian_inplace`
  to skip coloring-based Jacobian probing entirely.
- Fork diffsol to add the custom Jacobian path.
- Investigate faer sparse LU performance vs SuperLU — may improve with faer
  updates.
- Try ESDIRK34 (`diffsol::Tableau::esdirk34`) as an alternative implicit method.

### 4. SIMD for sparse commutator

**Priority: Low**

The sparse commutator inner loop does 1-2 complex MACs per output element.
Vectorize the outer loop by processing multiple (i,j) pairs with AVX2,
treating `H[i,k] * ρ[k,j]` as a scalar × vector sweep across j.

**Estimated speedup:** ~1.5-2x on the commutator (~10µs/call saved).
**Estimated effort:** ~4-6 hours (unsafe code, feature detection, fallback).

### 5. Auto-selection heuristic for Hamiltonian lowering

**Priority: Low**

The `"auto"` threshold in `ir.py` has no documented derivation. Benchmark with
multiple system sizes to validate or update the crossover.

**Estimated effort:** ~2 hours.

### 6. BLAS portability

**Priority: Deferred**

`blas.rs` uses `LoadLibraryW` to find scipy's OpenBLAS DLL at runtime
(Windows-only). Future: build with included BLAS for cross-platform support.
