# Rust Hamiltonian / Couplings Code Review

Review of the Rust code responsible for generating Hamiltonians (X and B state),
electric dipole couplings, basis transforms, and supporting infrastructure
(Wigner symbols, quantum operators, state types). The Lindblad/ODE solver code
is out of scope and reviewed separately in `IMPROVEMENTS.md`.

Each issue includes a comparison with the Python implementation to determine
whether the Rust code is a faithful port or introduces new problems.

## Files in Scope

| File | Lines | Purpose |
|------|-------|---------|
| `src/lib.rs` | 431 | PyO3 Python bindings |
| `src/states.rs` | 401 | Quantum state types, arithmetic, basis transforms |
| `src/constants.rs` | 78 | Physical constants, `XConstants`, `BConstants` |
| `src/quantum_operators.rs` | 224 | Angular momentum operators (J, I1, I2), composition |
| `src/wigner.rs` | 282 | Wigner 3j, 6j, Clebsch-Gordan (pure Rust) |
| `src/x_uncoupled.rs` | 212 | X state Hamiltonian terms (uncoupled basis) |
| `src/b_coupled.rs` | 486 | B state Hamiltonian terms (coupled Omega basis) |
| `src/generate_hamiltonian.rs` | 140 | Matrix element computation, Hamiltonian assembly |
| `src/coupling.rs` | 258 | Electric dipole coupling matrix |

---

## 1. Bugs / Correctness Issues

### 1.1 `mu_p` hardcodes `gL = 1.0` — present in both Python and Rust

**Rust:** `b_coupled.rs:392` — `let g_l = 1.0;`
**Python:** `B_coupled_Omega/zeeman.py:72` — `gL = 1.0`

Both Python and Rust hardcode `gL = 1.0` in the coupled-Omega `mu_p` operator.
The `BConstants` dataclass defines `gL: float = 1` and `gS: float = 2`, but
neither the Python nor Rust coupled-Omega Zeeman code reads them.

The **Python uncoupled-basis** B state Zeeman (`B_uncoupled.py:376-393`) *does*
use `coefficients.gL` and `coefficients.gS`, with both orbital and spin
contributions via separate 3j symbol terms. The coupled-Omega code omits the
spin term entirely.

**Two sub-issues:**

1. **`gL` should be read from constants, not hardcoded.** Even though the
   physical value is 1, using `constants.gL` (Python) / `constants.gl` (Rust)
   would allow sensitivity studies and is trivially fixed in both languages.
   This is a minor improvement, not a physics error — `gL = 1` is the correct
   orbital g-factor.

2. **`gS` / spin Zeeman term is omitted.** This is a deliberate physics
   approximation for ²Π₁/₂: the spin contribution is suppressed because the
   spin-orbit interaction mixes L and S projections, and the net effect of `gS`
   on the ²Π₁/₂ Zeeman splitting is small. Adding the spin term in the coupled
   basis would require additional operator structure (extra 3j symbols for the
   spin part, similar to `B_uncoupled.py:386-394`), not just replacing a
   constant. The Python docstring documents this: "Spin contribution neglected
   (g_S terms small for ²Π₁/₂)".

**Verdict:** The `gL` hardcoding is a minor issue in both languages (should
read from constants for flexibility). The `gS` omission is a documented physics
approximation, valid for TlF B state work but worth noting if the code is ever
extended to other electronic states.

**Severity:** Low-Medium (read `gL` from constants in both Python and Rust)

### 1.2 Spherical tensor operators produce `NaN` and `J=-1` states when `J=0` — matches Python

**Rust:** `x_uncoupled.rs:126-196` (`r10`, `r1m`, `r1p`)
**Python:** `X_uncoupled.py:386-524` (`R10`, `R1m`, `R1p`)

When `J=0`, both implementations compute `sqrt(negative_number)` for the `J-1`
branch and create states with `J=-1`:

- `R10`: denominator `8*J^2 - 2 = -2` when `J=0`, so `sqrt(0 / -2) = sqrt(0) = 0` for `amp1` (numerator is also 0 since `mJ=0`). For `amp2`, the denominator `6 + 8*J*(J+2) = 6` gives a valid result. So for `J=0, mJ=0`, `R10` does produce `amp1 = 0` — the `NaN` issue only arises for `R1m`/`R1p` where the numerator can be nonzero.
- `R1m`: `amp1 = sqrt((0+0)*(0+0-1) / -1)` = `sqrt(0)` = `0` when `J=0,mJ=0`.
- `R1p`: same as `R1m` by symmetry.

In practice, for `J=0` the only valid `mJ` value is `0`, and the numerators
`(J ± mJ) * (...)` always produce zero, making `amp1 = 0` despite the
negative denominator. The `J=-1` ket is created but with zero amplitude, and
both implementations filter zero-amplitude terms.

However: in Rust, the zero-amplitude filtering uses exact `!= 0.0` comparison
(see issue 1.4), so the `(0.0, J=-1_ket)` term might not be filtered if
floating-point rounding produces a tiny nonzero residual. In Python,
`numpy.sqrt(0 / -2)` returns `nan` (0/negative = 0.0 in float, `sqrt(0.0) = 0.0`),
so Python produces `amp1 = 0.0` and filters it out.

**Verdict:** In normal use (`J=0, mJ=0`) both produce zero amplitudes for the
`J-1` branch, so this is not causing incorrect results. But it's fragile — a
guard `if psi.j >= 1` on the `J-1` branch would be cleaner and avoid any edge
case with floating-point residuals.

**Severity:** Low (no practical impact with current quantum number ranges, but
worth hardening)

### 1.3 Rust `h_ff` uses `h_c3_alt` — matches Python `Hff_alt`

**Rust:** `x_uncoupled.rs:81-87` — `h_ff` calls `h_c3_alt`
**Python:** `generate_hamiltonian.py:185` — `_generate_uncoupled_hamiltonian_X_python` calls `Hff_alt`

Both the Rust and Python Hamiltonian generators use the alternative `c3`
formulation (`Hc3_alt` / `h_c3_alt`), not `Hc3a + Hc3b + Hc3c`. The Rust
function is somewhat confusingly named `h_ff` (matching the Python `Hff` naming)
but uses the `_alt` implementation. The `h_c3a`, `h_c3b`, `h_c3c` functions
exist in Rust but are dead code (confirmed by compiler warnings).

**Verdict:** Consistent. The dead `h_c3a/b/c` functions could be removed to
reduce confusion.

### 1.4 Floating-point equality for zero-amplitude filtering

**Rust:** `states.rs:48,56,81,104,220,250` — `amp.re != 0.0 || amp.im != 0.0`
**Python:** `states.py:978` — `amp != 0`

Both use exact equality to filter zero amplitudes. In Python, `0.0 != 0` is
`False` so exact-zero amplitudes are filtered; same in Rust. Neither handles
near-zero floating-point residuals from arithmetic. This is consistent behavior,
not a Rust regression.

**Severity:** Low (consistent with Python, but neither handles near-zero well)

### 1.5 Division-by-zero risk in `h_c3a`, `h_c3b`, `h_c3c` — dead code, matches Python

**Rust:** `x_uncoupled.rs:37,43,49`
**Python:** `X_uncoupled.py:119,139,159`

Both divide by constant products (`c1*c2`, `c4*B_rot`). Since these functions
are dead code in both Rust and the active Python path (both use the `_alt`
formulation), this is moot.

**Severity:** None (dead code)

---

## 2. Performance Issues

### 2.1 O(N x G) Python-level index lookup in coupling matrix — **MEDIUM**

**File:** `lib.rs:383-393`

`find_index` does a Python-level `rich_compare` loop over all `qn` for every
ground and excited state. For N total states with G ground and E excited states,
this is O((G+E) x N) Python interop calls.

The Python side (`couplings/coupling_matrix.py:41-116`) does the same O(N)
search per state using `list.index()`, but in pure Python this is expected.
In Rust, the Python interop overhead (GIL acquisition, object marshalling)
makes this particularly expensive.

**Fix:** Pass indices directly from Python, or build a Python-side dict before
calling into Rust.

### 2.2 Repeated `transform_to_uncoupled` in transform matrix — **MEDIUM**

**File:** `lib.rs:282-286` via `states.rs:380-397`

`inner_product` between coupled states calls `transform_to_uncoupled()` for
every (i,j) pair, recomputing Clebsch-Gordan coefficients N x M times.

The Python side avoids this via `@lru_cache` on many operator functions. The
Rust code has no caching mechanism.

**Fix:** Precompute the uncoupled expansions once before the matrix loop.

### 2.3 Linear scan for inner product in `h_mat_elems` — **LOW-MEDIUM**

**File:** `generate_hamiltonian.rs:59-63`

The inner product `<a|H|b>` scans all terms of `H|b>` linearly to find a
matching basis state. For operators that produce many terms (e.g., `h_ff` which
sums 5+ sub-operators), building a `HashMap` from each precomputed `h_applied[j]`
would give O(1) lookup.

The Python side uses `State.__matmul__` (`@` operator) which does a similar
linear scan — so this is consistent, but the Rust code could do better.

### 2.4 Heavy allocation churn in state arithmetic — **MEDIUM**

**File:** `states.rs:70-167`

Every `+`, `-`, `*` on `UncoupledState`/`CoupledState` allocates a new
`HashMap`, inserts all terms, filters, and collects into a new `Vec`. Operator
compositions like `h_ff` chain 5+ additions, each allocating intermediates.

The Python side (`states.py:995-1027`) uses a list-based approach with a dict
for lookups, preserving insertion order. It does not create `HashMap`
equivalents on every operation, making the Python approach slightly more
efficient per operation (though Python is slower overall).

**Options:**
- Use `Vec`-based accumulation with a sort-and-merge at the end
- Keep terms as a `HashMap` internally and only convert to `Vec` for output
- Use an arena allocator for short-lived intermediates

### 2.5 Parallelize independent Hamiltonian term computation — **MEDIUM**

**File:** `generate_hamiltonian.rs:106-139`

The 7 X-state and 12 B-state Hamiltonian terms are computed sequentially but are
completely independent. `rayon` is already a dependency. Use `rayon::scope` to
compute all terms in parallel.

The Python side also computes these sequentially (but benefits from `lru_cache`
on individual operator calls).

### 2.6 `d_p` / `mu_p` recomputation in Stark/Zeeman terms — **LOW**

**File:** `b_coupled.rs:371-386, 469-484`

`h_sx` and `h_sy` both call `d_p(psi, 1, ...)` and `d_p(psi, -1, ...)`.
Similarly, `h_zx`/`h_zy`/`h_zz` all call `mu_p`.

The Python side avoids this with `@lru_cache(maxsize=int(1e6))` on both `mu_p`
and all `HZx/HZy/HZz/HSx/HSy/HSz` functions. The Rust code has no caching.

**Fix:** Cache `d_p`/`mu_p` results per `(psi, p)`, or restructure to compute
all Stark/Zeeman components together.

### 2.7 Coupling matrix uses `Vec<Vec<Complex64>>` — **LOW**

**File:** `coupling.rs:235`

A jagged `Vec<Vec<Complex64>>` is not cache-friendly and requires an extra copy
in the PyO3 layer (`PyArray2::from_vec2`). Use a flat `Vec<Complex64>` with
row-major indexing, consistent with the Hamiltonian matrices.

---

## 3. Dead / Unused Code

### 3.1 `wigner-3nj-symbols` and `libm` crates were unused — **DONE**

~~**File:** `Cargo.toml:15-16`~~

Both `wigner-3nj-symbols = "0.2.0"` and `libm = "0.2"` were listed as
dependencies but never imported anywhere in the codebase (confirmed by grep
and `cargo check` warnings). **Removed in this review.**

### 3.2 Dead functions flagged by compiler

The following functions are dead code (confirmed by `cargo check` warnings):
- `quantum_operators.rs`: `j4`, `j6`, `apply_op`
- `x_uncoupled.rs`: `h_c3a`, `h_c3b`, `h_c3c`
- `constants.rs`: `A0` constant, `XConstants::d` field, `BConstants::gl`,
  `BConstants::gs`, `BConstants::gamma` fields

The `gl`/`gs`/`gamma` fields are populated from Python but never read by Rust
code. They exist only for completeness of the `BConstants` struct mirroring the
Python dataclass. This is fine but worth documenting.

---

## 4. Code Quality / Clarity

### 4.1 `com` function name — matches Python

**Rust:** `quantum_operators.rs:202`
**Python:** `X_uncoupled.py` uses `com(Hc1, Hc2, psi, coefficients)` identically

The name `com` is used in both Python and Rust. Despite suggesting "commutator",
it computes operator composition A(B(psi)). Both codebases use the same name,
so renaming in Rust alone would create a naming inconsistency. If renamed, both
should change.

### 4.2 Duplicated Python state-parsing boilerplate — **MEDIUM**

**File:** `lib.rs` (5 separate locations)

The Python-to-Rust state conversion code is copy-pasted across
`generate_uncoupled_hamiltonian_X_py`, `generate_coupled_hamiltonian_B_py`,
`generate_transform_matrix_py`, and `generate_coupling_matrix_py`. Each instance
manually extracts `J`, `F`, `mF`, `I1`, `I2`, `F1`, `Omega`, `P`,
`electronic_state` with identical logic.

**Fix:** Extract shared helper functions:
```rust
fn parse_uncoupled_state(s: &Bound<'_, PyAny>) -> PyResult<UncoupledBasisState> { ... }
fn parse_coupled_basis_state(s: &Bound<'_, PyAny>) -> PyResult<CoupledBasisState> { ... }
fn parse_coupled_state(s: &Bound<'_, PyAny>) -> PyResult<CoupledState> { ... }
```

### 4.3 `BConstants` lacks `Default` impl

**File:** `constants.rs:49-78`

`XConstants` has a `Default` impl with physical values but `BConstants` does not.
The Python `BConstants` dataclass has defaults for all fields.

### 4.4 Comment/code sign convention mismatch in B state

**File:** `b_coupled.rs:370,377`

Comment at line 370 says `H_Sx = -(d_{-1} - d_{+1})/sqrt(2)` while the code
computes `(d_p(+1) - d_p(-1)) / sqrt(2)`. These are algebraically identical.
The Python code (`zeeman.py:168`) writes `-(mu_p(-1) - mu_p(+1)) / sqrt(2)`
which matches the comment form. The Rust algebraically simplified but didn't
update the comment.

### 4.5 `ED_ME_coupled` in Rust omits `mu_E` scaling — matches Python

Compared the full `ed_me_coupled` function in `coupling.rs:60-187` against the
Python `ED_ME_coupled` in `matrix_elements_electric_dipole.py:290-381`. The Rust
version does not multiply by `mu_E` (the transition dipole moment), matching the
Python version which also returns the dimensionless matrix element. The `mu_E`
scaling is handled at the `d_p` operator level for the B-state Stark effect, not
in the coupling matrix element calculation.

---

## 5. Suggested Priority Order

| # | Issue | Severity | Effort | Section |
|---|-------|----------|--------|---------|
| 1 | Extract state-parsing helpers | Medium | 1-2 hr | 4.2 |
| 2 | Parallelize Hamiltonian terms with rayon | Medium | 1-2 hr | 2.5 |
| 3 | Reduce allocation churn in state arithmetic | Medium | 2-4 hr | 2.4 |
| 4 | Cache `transform_to_uncoupled` | Medium | 1 hr | 2.2 |
| 5 | O(N x G) Python index lookup | Medium | 1 hr | 2.1 |
| 6 | Add caching for `d_p`/`mu_p` (Rust has no `lru_cache`) | Medium | 1-2 hr | 2.6 |
| 7 | Guard `J-1` branch in spherical tensors | Low | 15 min | 1.2 |
| 8 | Use flat Vec for coupling matrix | Low | 30 min | 2.7 |
| 9 | Remove dead functions (`h_c3a/b/c`, `j4`, `j6`, etc.) | Low | 15 min | 3.2 |
| 10 | Add `BConstants::default()` | Low | 15 min | 4.3 |
| 11 | Fix comments/formatting | Low | 15 min | 4.4 |

---

## 6. Testing Notes

There are no Rust-side unit tests (`rust/tests/` is empty). All testing is done
from the Python side, which calls into the Rust functions via PyO3. This provides
integration-level coverage: if the full Hamiltonian eigenvalues match between
Python and Rust backends, individual operator correctness is implicitly verified.

Adding Rust-side unit tests would catch regressions faster (no Python roundtrip
needed) and test edge cases like `J=0` operators in isolation. Suggested minimum:
- Wigner 3j/6j against known tabulated values
- Clebsch-Gordan coefficient orthogonality
- Individual Hamiltonian terms for small J (e.g., J=0,1)
- Coupled-to-uncoupled basis transform round-trip

---

## Completed

- **Removed dead dependencies** `wigner-3nj-symbols` and `libm` from
  `Cargo.toml`. Neither was imported anywhere in the codebase. Build verified
  with `cargo check`.
