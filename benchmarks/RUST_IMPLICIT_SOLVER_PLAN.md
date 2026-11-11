# Plan: an implicit (stiff) Lindblad solver in Rust

Status: plan only, nothing implemented. Written 2026-11-10.

## Outcome: not recommended for the intended workload

The target use is multipass Gaussian beam profiles with sinusoidal polarization
switching. Measured on that problem (recipe system with X and Z polarization
components, 5 Gaussian passes, 1 MHz sinusoidal switching, 108.7 µs), an implicit
solver does not beat the explicit one. `dopri5`-class explicit integration needs about
250 000-280 000 RHS calls, about 0.6 s at 2.3 µs each.

SciPy Radau on the same problem, with the sparse time-dependent Jacobian. "Estimated
native" prices SciPy's own counts at the measured costs (real + complex factorization
3.0 ms, real + complex solve 0.1 ms, RHS 2.3 µs):

| Detuning, Rabi rate | `rtol` | Steps | Factorizations | Population error | Estimated native |
|---|---|---|---|---|---|
| 0 MHz, 0.11 MHz | 1e-4 | 1 581 | 138 | 3e-6 | 0.8 s |
| 0 MHz, 0.11 MHz | 1e-5 | 3 022 | 150 | 1.4e-7 | 1.2 s |
| 0 MHz, 1.14 MHz | 1e-4 | 2 050 | 1 648 | 9e-8 | 5.6 s |
| 100 MHz, 0.11 MHz | 1e-5 | 1 958 | 584 | 1.8e-7 | 2.3 s |
| 100 MHz, 1.14 MHz | 1e-5 | 4 556 | 2 584 | 2.5e-7 | 9.1 s |

Like for like against the current solver, measured in the package with the same
runtime expressions (`gaussian` passes on the Rabi rate, `sine` on the two
polarization amplitudes), one trajectory on one core:

| Detuning, Rabi rate | `dopri5` measured | `dopri5` error | Radau estimated | Radau if factorizations were free | Radau error |
|---|---|---|---|---|---|
| 0 MHz, 0.11 MHz | 0.87 s | 1e-11 | 1.2 s | 0.33 s | 1.4e-7 |
| 0 MHz, 1.14 MHz | 0.87 s | 1e-11 | 5.6 s | 0.23 s | 9e-8 |
| 100 MHz, 0.11 MHz | 0.78 s | 3e-12 | 2.3 s | 0.22 s | 1.8e-7 |
| 100 MHz, 1.14 MHz | 0.78 s | 1.5e-10 | 9.1 s | 0.50 s | 2.5e-7 |

`dopri5` takes about 46 000 steps (41 000 at 100 MHz) whatever the tolerance; at
`reltol=1e-3` it is still accurate to 5e-7 or better and no faster. `tsit5` is within
10% of it. The "free factorizations" column is steps × (one real and one complex solve
plus three RHS calls) and is a bound, not an achievable number: it is 1.6-3.8× faster
than `dopri5`, at four orders of magnitude worse accuracy.

The second-order implicit midpoint is worse than Radau: 1e-4 population error at 10 ns
steps, with 180-1 900 factorizations.

The reason is that the 1 MHz switching and the stronger drive move the dynamics of
interest closer to the fast frequencies, so there is less of a gap for an implicit
method to step over, and the matrix changes on every step, so factorizations cannot be
reused for long.

The rest of this document is kept as a record of the measurements. The static-drive
conclusion (use `dense_eig`) and the smooth-envelope result (2.5-5× for a slowly
varying single Gaussian) still stand, but they do not justify the project on their
own.

## Why

On the AGENTS.md recipe system (P2 F1'=3/2 F'=1, 29 levels, 200 V/cm, Rabi rate
2π × 0.114 MHz, 108.7 µs) the explicit solvers are limited by stability, not accuracy:

| `dopri5`, one trajectory | steps | mean step | rejected | time |
|---|---|---|---|---|
| detuning 0 MHz, `reltol=1e-7` | 46 380 | 2.34 ns | 0 | 0.61 s |
| detuning 0 MHz, `reltol=1e-4` | 45 855 | 2.37 ns | 0 | 0.70 s |
| detuning 150 MHz, `reltol=1e-7` | 60 428 | 1.80 ns | 0 | 0.82 s |

The populations change on a microsecond scale, but the step is pinned to ~2 ns by
coherences oscillating at the detuning plus the B-state splittings (50, 68, 118 MHz).
With the drive off the same trajectory takes 6 steps.

A-stable implicit methods do not have this limit. Measured on the same problem
(errors are against `dopri5` at `reltol=1e-10`):

| Method | Steps | Time | Max population error |
|---|---|---|---|
| `dopri5` (Rust), `reltol=1e-7` | 46 380 | 0.61 s | reference |
| SciPy Radau, `rtol=1e-4` (generic Python) | n/a (58 LU, 7 294 RHS) | 0.50 s | 1e-8 |
| SciPy Radau, `rtol=1e-6` | n/a (56 LU, 29 435 RHS) | 1.77 s | 2e-11 |
| Implicit midpoint, fixed 20 ns, Python loop + SuperLU | 5 435 | 0.23 s | 3e-8 |
| Implicit midpoint, fixed 50 ns | 2 174 | 0.089 s | 1.4e-7 |
| Implicit midpoint, fixed 200 ns | 543 | 0.022 s | 9e-7 |

`scipy_bdf` took 50-60 s: the higher-order BDF formulas are unstable near the imaginary
axis, which is where these eigenvalues are. BDF is not a candidate.

So a second-order method in a Python loop is already 7× faster than the Rust explicit
solver at 1e-7 population accuracy. Coherence errors are larger (6e-5 at 50 ns): the
fast coherences are stepped over, not resolved. That is the trade being made, and it
has to be visible to the user.

## Relation to `dense_eig`: static drives are already covered

Added after the first draft, and it changes what this solver is for.

`solver="dense_eig"` already handles the static-drive case, and much faster than
anything above. Same recipe system, detuning scan, photon integral:

| | 32 points | 380 points |
|---|---|---|
| `dopri5`, `parallel=True` | 3.0 s (94 ms/pt) | 29.9 s (79 ms/pt) |
| `dense_eig`, `parallel=False` | 0.50 s (15.5 ms/pt) | 5.95 s (15.7 ms/pt) |
| `dense_eig`, `parallel=True` | 10.3 s | 12.7 s |

- `dense_eig` on one core is 5-6× faster than `dopri5` on 16 logical cores, and agrees
  with it to 1.4e-11.
- At the time of this measurement `parallel=True` added about 10 s of process-pool
  start-up to a `dense_eig` scan, after which points cost about 7 ms each. That start-up
  has since been fixed (workers now start concurrently, about 1.5 s): the 380-point scan
  takes 2.2 s in parallel and the 32-point one 1.6 s. (An earlier measurement in this session reported "328 ms/pt" for
  `dense_eig`; that was this start-up cost on a 32-point scan, not the solver.)
- Extra initial states are almost free: one parameter point with 1, 23 and 230 initial
  states took 0.02, 0.02 and 0.03 s. An implicit ODE solver needs a trajectory per
  initial state and cannot compete there.

So for static drives the implicit solver would be slower than what exists (the
prototype's best case is 22 ms per trajectory against 15.5 ms, and that is before
adaptive stepping). Its value is in what `dense_eig` rejects:

- time-dependent drives: polarization switching, Gaussian or multipass beam profiles,
  phase modulation;
- terminal events;
- sampled quadrature;
- generators with singular or ill-conditioned eigenbases;
- systems large enough that the dense eigendecomposition stops being cheap. It was not
  measured beyond 29 levels here.

That makes Phase 4 (time-dependent drives) the point of the project, not an add-on.
Every number in the prototype table is for a static drive, where one factorization
serves the whole trajectory. With a time-dependent drive the matrix changes and must
be refactored (1.3 ms, about 35 solves) whenever it has moved too far, so the gain
there is unmeasured and will be smaller.

## Gate result: a time-dependent drive (measured)

Recipe system with a Gaussian Rabi envelope (σ = T/6), so `J(t) = J0 + Ω(t) J1`.
Reference: DOP853 at `rtol=1e-11`. All implicit numbers are Python loops over SuperLU;
`dopri5` in Rust takes about 0.6 s per trajectory on this system.

Fully implicit midpoint, factorization reused and refreshed only when the iteration
needs more than 3 passes:

| Step | Steps | Factorizations | Solves per step | Time | Population error (0 / 100 MHz) |
|---|---|---|---|---|---|
| 50 ns | 2 174 | 27 / 20 | 2.9 | 0.36 s | 7e-8 / 5e-8 |
| 100 ns | 1 087 | 45 / 29 | 2.9 | 0.22 s | 4e-7 / 1.5e-8 |
| 200 ns | 543 | 66 / 41 | 2.9 | 0.16 s | 1.7e-6 / 4e-9 |
| 500 ns | 217 | 67 / 47 | 2.7 | 0.11 s | 1e-5 / 2e-9 |

So a time-dependent drive costs about 3 solves per step and a few dozen
factorizations per trajectory, and the gain over `dopri5` is roughly 2.5-5× at 1e-6 to
1e-7, not the 7-25× of the static case. Refactoring on every step instead takes 3.0 s:
reuse of the factorization is essential, not an optimization.

**IMEX is ruled out.** ARS(2,2,2), implicit in `J0` and explicit in `Ω(t) J1`, needs
only one real factorization and no iteration, which made it the obvious cheaper
candidate. It is not accurate enough off resonance: on the static problem at 100 MHz
detuning the photon number is wrong by 3e-3 relative at 50 ns steps and still 7e-4 at
20 ns, where implicit midpoint is at the 1e-7 level. The far-detuned coherence follows
the drive quasi-statically, and that balance is only captured when the coupling is
inside the implicit solve.

The gate passes, but narrowly: the project buys a factor of a few on time-dependent
drives, on this system. Decide on that basis.

## Cheaper alternative for piecewise-constant drives (unmeasured)

Polarization switching is a square wave, so the generator is constant on each
half-period. Those segments can be propagated exactly with the existing dense
machinery (`prepare_dense_lindblad_propagator`: one decomposition per distinct
generator, reused for every segment and every initial state), with no ODE solver at
all. `prepare_periodic_lindblad_propagator` already does a version of this, but only
on the CUDA path. A CPU driver that chains dense propagators over segments is
Python-level work on top of what exists and should be tried before the Rust solver if
switching problems are the main use. It does not help smooth envelopes.

## What makes this problem cheap for an implicit method

- The equations are linear: `dy/dt = J(t) y`, with `y` the packed real density matrix
  (dimension n², 841 for 29 levels).
- `J` is sparse: 6 132 nonzeros, 7.3 per row.
- With a static drive `J` is constant over a trajectory, so one factorization serves
  every step of the same size.
- The sparsity pattern is the same for every scan point of a prepared problem, so the
  symbolic analysis can be done once per plan and only the numeric factorization
  repeated.

Measured with SciPy's SuperLU on the 841 × 841 matrix `I - hγJ`: real LU 1.3 ms (fill
39 k nonzeros), one solve 37 µs; complex LU 1.7 ms, one solve 63 µs. One explicit RHS
in Rust is 2.3 µs, so a solve costs about as much as 16 RHS calls and a factorization
about 550.

## Method

**Primary: Radau IIA, 3 stages, order 5** (the RADAU5 scheme).

- A-stable and L-stable, so it is stable on the imaginary axis where BDF is not; SciPy's
  implementation of the same scheme already works on this problem.
- Per step size it needs one real and one complex factorization of size n². For a
  linear problem with the exact Jacobian the simplified Newton iteration converges in
  one pass: one real solve, one complex solve and three RHS evaluations per step.
- It has an embedded error estimate (adaptive steps) and a collocation polynomial
  (dense output for `saveat` and event root finding).

**Fallback if complex sparse LU is a problem: a stiffly accurate ESDIRK4(3) scheme.**
Real arithmetic only, one factorization, five solves per step.

**Also expose a fixed-step implicit midpoint** (`fixed_implicit_midpoint`), matching the
existing `fixed_rk2` / `fixed_rk4` family. It is the measured prototype above, the
simplest thing to validate the linear algebra against, and it is symmetric (no
numerical damping), which makes it a useful cross-check on Radau's L-stable damping.

## Decision needed before coding: the sparse linear solver

`rust/Cargo.toml` currently depends only on `pyo3`, `numpy`, `num-complex` and `rayon`.
The solver needs a sparse LU for real and complex matrices with a reusable symbolic
factorization. Two options:

1. **Add `faer`** (pure Rust; sparse LU for `f64` and `c64`, symbolic/numeric split).
   Recommended. Costs: a large dependency, longer builds, bigger wheels.
2. **Write a small sparse LU in-tree** (left-looking, pivot order fixed from a first
   factorization). No dependency, but several hundred lines of numerical code to own
   and test, and likely slower.

Phase 0 below settles this with measurements instead of opinion.

## Where it plugs in

- `rust/src/ode/mod.rs`: `OdeRhs` has `eval`, `dim`, `error_control_dim`, `event_value`.
  Add a sub-trait for linear problems, e.g.
  `LinearOdeRhs: OdeRhs { fn jacobian_pattern(&self) -> &CscPattern; fn jacobian_values(&mut self, t, &mut [f64]); fn is_time_dependent(&self) -> bool; }`.
- `rust/src/lindblad/rhs.rs`: `build_packed_jacobian_analytic` already transcribes the
  expanded sparse plan into (row, col, value) triples in O(nnz), but it allocates and
  sorts through a `BTreeMap` on every call. Add a prepared "Jacobian plan": the CSC
  pattern plus, for each expanded term, its slot and sign, so refilling values at a new
  `t` or new parameters is a flat loop.
- `rust/src/lindblad/ode_impl.rs`: implement `LinearOdeRhs` for `LindbladRhs`.
- `rust/src/ode/radau.rs` (new) and an implicit branch in `rust/src/ode/fixed.rs`.
- `rust/src/ode/batch.rs`: `OdeSolver::Radau5`, `OdeSolver::FixedImplicitMidpoint`,
  `from_str`, and dispatch in `solve_single`. The photon integral is currently done by
  augmenting the state (`IntegralAugmentedRhs`); for the implicit solvers compute it
  from the stage values instead (`h Σ b_i w·Y_i`), so the linear system keeps its size
  and pattern.
- `rust/src/lindblad/ode_batch.rs`, `python_api.rs`: thread the new solver through
  single, batch and grid entry points; keep one symbolic factorization per worker.
- `centrex_tlf/lindblad/solve.py` (solver lists near lines 672 and 734) and `batch.py`
  (`rust_solvers` near lines 290 and 588): accept the new names.
- Outputs go through the existing `OdeOutput` trait unchanged.

## Phases

**Gate — time-dependent prototype.** Done; see "Gate result" above. It passes with a
gain of roughly 2.5-5× for a second-order method.

**Phase 0 — linear algebra choice (small).**
Export `I - hγJ` for the 29-level recipe and the 67-level rotational-cooling system
(dimension 4 489). Benchmark `faer` sparse LU, real and complex: symbolic time, numeric
refactor time, solve time, fill. Compare with the SuperLU numbers above. Decide
dependency versus in-tree. Exit: a table and a decision.

**Phase 1 — Jacobian plan.**
Prepared CSC pattern and flat value refill. Tests: equals `build_packed_jacobian_sparse`
(probing) on the four systems used this session, at several times and parameter
overrides; `J·y` equals the RHS for random `y`.

**Phase 2 — static-drive solvers, single trajectory.**
`fixed_implicit_midpoint` first, then `radau5` with step control, dense output,
`saveat`, the photon integral from stage values, and `PopulationThreshold` events.
Reject time-dependent plans with a clear error for now. Tests: populations, coherences
and photon integrals against `dopri5` at tight tolerance; trace stays 1; results
independent of `saveat`.

**Phase 3 — batch and grid scans, Python plumbing.**
Per-point numeric refactorization with the shared symbolic analysis; `parallel=True`.
Benchmarks against `dopri5` on narrow and wide detuning scans and on 26, 29, 31 and
67-level systems, reporting steps, factorizations and time per point.

**Phase 4 — time-dependent drives.**
Freeze `J` at the start of a step, iterate the simplified Newton with true residuals,
and refactor when convergence slows (the RADAU5 policy). Force step boundaries at
discontinuities (polarization switching already declares them for the CUDA path).
Gaussian envelopes and phase modulation need their own accuracy checks.

**Phase 5 — documentation.**
`README_OBE_SOLVER.md` solver table, AGENTS.md performance notes, changelog. State
plainly which accuracy the tolerances control.

## Risks and open questions

- **Accuracy semantics.** Large steps do not resolve the fast coherences; Radau damps
  them, implicit midpoint aliases them. Populations and photon counts were right to
  1e-7 or better in the prototype, but anything that depends on a fast coherence
  (Ramsey-type phases, short pulses) needs the explicit solver or tight tolerances.
  This should be documented and, where possible, detected.
- **Fill-in at larger sizes.** Fill was 6× at dimension 841. It is unmeasured at 4 489
  and beyond; if it grows badly the gain shrinks.
- **Gain depends on the problem.** The win comes from the gap between the fastest
  frequency and the dynamics of interest. A resonant, strongly driven system with no
  far-detuned levels has little to gain.
- **Complex factorization.** Radau needs it; if the chosen library handles it poorly,
  fall back to ESDIRK.
- **Build and wheels.** A new dependency must build on the manylinux and Windows
  release jobs in `.github/workflows/release.yml`.
- **Not the default.** `dopri5` stays the default until the new solver has been
  validated across the example notebooks.

## Expected gain

From the prototype, for static drives on the recipe system: roughly 5-25× per
trajectory over `dopri5` at 1e-7 to 1e-6 population accuracy, depending on the step the
tolerance allows. That is an extrapolation from a second-order fixed-step method in
Python; the Radau numbers do not exist until Phase 2.

For static drives that gain is moot, because `dense_eig` is faster still (see above).
The case that matters is a time-dependent drive. The gate measurement above puts it at
roughly 2.5-5× over `dopri5` for implicit midpoint with a reused factorization. Radau's
fifth order should allow larger steps for the same accuracy, but each step costs a
complex solve as well, so how much it adds is not known until Phase 2.
