# Hoisting loop-invariant Omega-basis transforms: ~1.25x off every OBE build

Run: `uv run python benchmarks/bench_dipole_me_hoist.py`
(Windows, release extension build, 2026-08-20.)

## The claim that was wrong

The audit's item-4 profile attributed 27% of a compact OBE build to
`generate_ED_ME_mixed_state` and recommended **"cache the field-independent bare
dipole matrix elements shared across grid points"**.

That recommendation was wrong. `ED_ME_coupled` — the actual bare-basis matrix
element — already carries `@lru_cache(maxsize=1e6)`, as do `ED_ME_uncoupled`,
`_ED_ME_uncoupled_omega` and `angular_part`. There was no missing bare-ME cache
to add.

The real cost was that `generate_ED_ME_mixed_state` transforms its *mixed*
(field-dressed) arguments to the Omega basis on **every call**, and the callers
invoke it from nested loops in which one argument is loop-invariant.
`State.transform_to_omega_basis` rebuilds a state by repeated `+=`, which is why
the original profile showed 24,950 `CoupledBasisState.__init__` and 24,514
`State.__add__` calls underneath it.

So this was an `O(n_ground * n_excited) -> O(n_excited)` hoist, not a caching
problem: no new cache, no hashability work, no tolerance argument.

## Scope

Only two call sites were live. `couplings/coupling_matrix.py:121,147` look like
the obvious hot loops but are **fallback-only** — `generate_coupling_matrix`
dispatches to the Rust `generate_coupling_matrix_py` whenever `HAS_RUST`, which
is the normal case. They were deliberately left alone.

- **Site A**, `hamiltonian/reduced_hamiltonian.py` — the `minimum_coupling`
  discovery loop, where essentially all the measured dipole time went. Three
  invariants were recomputed per pair: the excited state's Omega transform, and
  `1 * gs`, which built a fresh `CoupledState` on every inner iteration.
- **Site B**, `couplings/branching.py` — `calculate_br` rebuilt
  `excited_state.remove_small_components(tol)` once per ground state, and its
  Omega transform then happened again per call inside the callee.

## Measured redundancy

Counted by wrapping `generate_ED_ME_mixed_state` and
`State.transform_to_omega_basis`, keying distinct states on `id()`.

| case | n | transforms before | distinct | worst state | transforms after |
| --- | ---: | ---: | ---: | ---: | ---: |
| A `R0_F1_3o2_F2` | 105 | 1 231 | 510 | 144x | **21** |
| B `R2_F1_7o2_F3` | 154 | 4 733 | 1 988 | 196x | **57** |
| C `Q1_F1_3o2_F2` compact | 20 | 1 051 | 330 | 144x | **21** |

The `distinct` column undercounts the redundancy, and the gap is instructive:
`id()`-keying treats a freshly built but *equal* object as distinct, which is
exactly what `1 * gs` produces on every inner iteration. So `total - distinct`
(58-69%) was only a **lower bound** on removable work; the actual reduction is
**98-99%**.

The `worst state repeated 144x / 196x` figure is the direct signature of the
predicted pattern: a single excited state transformed once per ground state.

## Timings

Median of 3 (after one warm-up rep to populate the `lru_cache`s), full
`generate_OBE_system_transitions` build at Ez = 171.6 V/cm:

| case | before | after | speedup |
| --- | ---: | ---: | ---: |
| A `R0_F1_3o2_F2` (105 states) | 0.784 s | 0.633 s | **1.24x** |
| B `R2_F1_7o2_F3` (154 states) | 3.372 s | 2.683 s | **1.26x** |
| C `Q1_F1_3o2_F2` compact (20 states) | 0.610 s | 0.502 s | **1.22x** |

Baseline captured by `git stash`-ing exactly the three changed files, so before
and after ran on the same machine, same interpreter, same extension build.

Because `generate_reduced_hamiltonian_transitions` is on the path of **every**
OBE build, this is a package-wide setup win rather than the field-grid-only one
item 4 implied.

## Correctness

The risk here is not tolerance, it is identity — the hoist applies the same
deterministic transform to the same objects, so results must be *exactly* equal.

1. **Property check**: 10 080 (ground, excited, polarization) triples compared
   `ME(gs, es)` against `ME(gs, to_omega_basis(es))` with `==`, not `np.isclose`.
   **0 bitwise mismatches.**
2. **End-to-end check**: `H_int`, `C_array`, `QN` labels and `main_coupling` for
   all three systems, captured before (via `git stash`) and after.
   **All 12 arrays bitwise identical** (`np.array_equal`), including
   `B_C` at shape (640, 154, 154).
3. Full suite: **390 passed, 1 skipped**.

The one change that alters an intermediate value is the `break` added to the
discovery loop: `gs` was previously appended once per excited state it couples
to, and is now appended once. That list is consumed only as
`np.unique([s.J for s in nonzero_coupling])`, so duplicates cannot affect the
result — and check 2 confirms it empirically rather than by argument alone. It
also accounts for the drop in ME calls (A 720 -> 569, B 2 744 -> 1 939).

## Caveats

- The `distinct` counts are `id()`-based and therefore a lower bound on
  redundancy, as described above. They were used only to decide whether to
  proceed, not to size the win.
- Timings are single-machine medians of 3; the run-to-run spread was ~2%, well
  below the 22-26% effect.
- `coupling_matrix.py` was left unchanged by design. If the Rust extension is
  ever unavailable, those two Python fallback loops still carry the original
  O(n^2) transform behaviour and would be the next place to apply
  `to_omega_basis`.
