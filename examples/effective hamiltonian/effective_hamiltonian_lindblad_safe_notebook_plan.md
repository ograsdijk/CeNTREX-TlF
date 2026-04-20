# Fixed-Basis Compact-Derived Lindblad-Safe Plan

This document is the implementation plan for a new effective-Hamiltonian path
and a new notebook to test it.

The goal is to replace the current unstable dissipator construction with a
Lindblad-safe interpolation method while keeping the runtime basis fixed and
avoiding moving-basis gauge terms.

This plan is intended for **time-dependent fields**. Static-field handling is
already available and is not the target here.

## Goal

Build a new reduced model with these properties:

- exact compact reduced patches are used as the offline reference data
- runtime uses one fixed reduced basis
- `h_internal`, `h_opt`, and `h_det` are interpolated with operator-specific
  structural constraints
- decay is **not** interpolated as a raw superoperator
- instead, decay is interpolated through excited-subspace decay kernels by
  sector
- the reconstructed runtime dissipator remains Lindblad-safe

The first test case is:

- `transitions.Q1_F1_1o2_F0`

and the first notebook should stress the previously problematic low-field region.

## Why This New Path Is Needed

The current population-sink path projects the full compact dissipator onto the
coherent block and adds sink feed separately. That construction is not guaranteed
to preserve complete positivity, and it has already shown:

- small positive-real Liouvillian eigenvalues
- instability near the first sink-onset threshold
- worsening agreement as sink decay becomes more important

The fix is to interpolate and reconstruct the dissipator from **sector-resolved
decay kernels** rather than from the finished dissipator superoperator.

The first notebook run with the new fixed-basis compact-derived path also showed
several concrete follow-up issues that the implementation and notebook must
address:

- `h_opt` interpolation is essentially exact
- the large reported full-matrix `h_internal` error is misleading; the active
  coherent block is essentially exact and the mismatch is in sink-block
  bookkeeping
- `h_det` interpolation is currently wrong for the tested `Q1` construction:
  the exact compact reference has nonzero ground-ground structure, while the new
  path currently forces `h_det` onto the excited block
- the static notebook comparison used inconsistent `t_span` and `t_eval`
- the oscillating-field run is still very slow because the generic solver
  rebuilds the full bundle on every RHS call

## High-Level Design

The new method has four main pieces:

1. exact compact reduced patches at selected field points
2. fixed-basis alignment of the Hamiltonian-like operators
3. sector-kernel interpolation for decay
4. Lindblad reconstruction of the reduced dissipator

## Runtime Basis Choice

This path uses a **fixed reduced runtime basis**.

That means:

- no instantaneous basis at runtime
- no basis-motion term
- no patchwise runtime basis switching

Instead:

1. choose a master patch
2. align all compact patch operators into that basis
3. interpolate within that one basis during time evolution

This is the “fixed-basis compact-derived interpolation” method.

## Common Rotating Frame

The fixed-basis compact-derived time-dependent model must not re-zero the laser
detuning independently at every field patch.

Instead, all compact patches must be shifted into one **common rotating frame**
before alignment and interpolation.

### Why this is needed

The compact `qn_compact` OBE construction is naturally field-local: at each
field patch it provides a model with `detuning = 0` defined relative to that
patch's own resonant transition frequency.

That is correct for a static-field compact solve, but not for a
time-dependent-field propagation in one common frame.

For propagation, the user detuning must mean:

- laser detuning relative to one fixed reference frequency used for the whole
  run

Then, as the field changes, the Stark shift should move the `g-e` transition
energy inside `h_internal`, so the laser naturally goes off resonance.

### Patch frequency extraction

For each compact patch field `E_k`:

1. build the exact compact decomposed bundle at `E_k`
2. use its internal Hamiltonian at zero drive and zero user detuning
3. identify the addressed ground and excited states in that compact basis
4. compute the local compact transition frequency
   - `omega_patch(E_k) = E_excited(E_k) - E_ground(E_k)`

This should be computed directly from the same compact internal Hamiltonian used
to construct the patch.

### Common-frame correction

Choose one fixed propagation-frame reference:

- `omega_frame = omega_patch(E_master)`

Then correct each patch before alignment/interpolation:

- `h_internal_corrected(E_k) = h_internal(E_k) + [omega_patch(E_k) - omega_frame] * h_det(E_k)`

`h_opt`, `h_det`, and the dissipator are left unchanged.

After this correction:

- `detuning = 0` means on resonance at the master field
- field-dependent Stark shifts move the transition away from that fixed laser
  reference through `h_internal`

This correction must be applied before patch alignment and interpolation.

## Patch Construction

For each selected field patch `E_k`:

1. build the exact compact reduced model at `E_k`
2. extract:
   - `h_internal(E_k)`
   - `h_opt(E_k)`
   - `h_det(E_k)`
   - compact decay data
3. extract decay kernels on the excited subspace by sector

The exact compact model is the same object already used as the reference in the
existing notebooks.

## Basis Alignment

All patch operators must be rotated into one common reduced basis before
interpolation.

### Alignment procedure

1. choose a master patch field
2. align neighboring patches outward from the master patch
3. use overlap-based unitary alignment between neighboring patches
4. accumulate the transforms back into the common runtime basis

### Gauge smoothing

Alignment should not only maximize overlap. It should also make the basis vary as
smoothly as possible from patch to patch.

This means:

- remove arbitrary phase flips
- minimize patch-to-patch basis change relative to the previous aligned patch
- in nearly degenerate subspaces, align the whole subspace smoothly rather than
  rigidly tracking one vector at a time

This is effectively a parallel-transport-style gauge choice on the patch grid.

Good gauge smoothing is a major part of improving interpolation quality.

## Hamiltonian-Like Operator Interpolation

The operator interpolation rules should be different for different operators.

### `h_internal`

For the fixed-basis compact-derived method:

- interpolate only between neighboring patches
- preserve Hermiticity
- preserve internal block structure
- keep `g-e = 0`
- keep `e-g = 0`
- interpolate `g-g` and `e-e` blocks separately

Recommended default:

1. interpolate `g-g` block locally between neighboring patches
2. interpolate `e-e` block locally between neighboring patches
3. reconstruct each block as Hermitian
4. force cross internal blocks to zero

No global polynomial or spline fits should be used for the full matrix.

### `h_opt`

`h_opt` should preserve optical block structure.

Recommended rule:

1. interpolate only the `g-e` block
2. reconstruct `e-g = (g-e)^\dagger`
3. force `g-g = 0`
4. force `e-e = 0`

This is stronger and better conditioned than interpolating a full dense matrix.

### `h_det`

`h_det` should preserve its known structure.

Do not hardcode the detuning structure generically.

Instead:

1. inspect the aligned exact compact patch bundles
2. identify the actual nonzero block pattern used by that compact construction
3. preserve that pattern explicitly during interpolation

Only if the compact construction really gives an excited-only detuning operator
should the implementation use the excited-only rule:

1. interpolate only the excited-excited block
2. force all other blocks to zero

If the chosen formulation gives a different exact block pattern, preserve that
pattern explicitly.

## Decay Interpolation: Core Change

This is the central change in the new method.

### What is interpolated

Do **not** interpolate:

- the raw dissipator superoperator
- the full Liouvillian
- a projected coherent-block dissipator

Instead, interpolate only **sector-resolved decay kernels on the excited
subspace**.

For each decay sector `s`, define:

- `K_s(E_k)`

where `K_s` acts only on the excited manifold.

These sector kernels contain:

- diagonal partial decay rates
- off-diagonal excited-state coherence decay structure into that sector

### Fixed union of decay sectors

The decay-space bookkeeping must remain fixed over the full field range.

So:

1. define the union of all decay sectors that appear anywhere in the field range
2. for every patch field, store a kernel for every sector in that union
3. if a sector is absent at a patch, assign it the zero kernel there

Example for `Q1`:

- return to active `X, J=1`
- sink `X, J=0`
- sink `X, J=2`

Then:

- below first onset:
  - `K_{J=0} = 0`
  - `K_{J=2} = 0`
- after first onset:
  - `K_{J=0} > 0`
  - `K_{J=2} = 0`
- after second onset:
  - `K_{J=0} > 0`
  - `K_{J=2} > 0`

The sector set does not change. Only the kernels do.

### Physical constraints on each sector kernel

Each `K_s(E)` must be:

- Hermitian
- positive semidefinite

and the sector sum must satisfy:

- `sum_s K_s(E) = K_total(E)`

on the excited subspace.

This is the matrix generalization of:

- nonnegative branching fractions
- branching fractions summing to one

for the single-excited-state case.

### Interpolation rule for decay kernels

Use **local convex interpolation** only.

For a field between `E_k` and `E_{k+1}``:

- `K_s(E) = w_k(E) K_s(E_k) + w_{k+1}(E) K_s(E_{k+1})`

with:

- `w_k(E) >= 0`
- `w_{k+1}(E) >= 0`
- `w_k(E) + w_{k+1}(E) = 1`

This guarantees:

- Hermitian kernels remain Hermitian
- PSD kernels remain PSD
- the sector sum rule is preserved under the same weights

This is the key reason local convex interpolation is preferred over splines or
global polynomial fits.

### Reconstructing the dissipator

After interpolation:

1. obtain interpolated `K_s(E)` for all sectors
2. reconstruct the reduced dissipator from those kernels
3. use the same sector kernels for:
   - active return channels
   - sink channels

This reconstruction should be the only source of the runtime dissipator.

There should be:

- no raw dissipator projection
- no separate ad hoc coherent-block dissipator fix
- no superoperator interpolation

## Sink Representation

The sink representation should remain a separate modeling choice.

Two options should be supported later:

1. compact spectator sink states inside the reduced Hilbert space
2. external sink populations

The important point is that both must be driven by the **same interpolated
sector kernels**.

The sink representation should not change the decay interpolation method.

## New Runtime Path

Add a new, separate implementation in
`examples/effective hamiltonian/effective_hamiltonian_runtime.py`.

Do not modify existing:

- SW paths
- previous interpolated atlas paths
- instantaneous-basis paths
- population-sink path

Suggested new public entry points:

- `prepare_lindblad_safe_compact_interpolated_model(...)`
- `solve_lindblad_safe_compact_interpolated_model(...)`
- `solve_static_lindblad_safe_compact_interpolated_model(...)`

These names can change, but the new path should be clearly separate.

## Time-Dependent RK45 Performance Plan

For the time-dependent path, the current generic solver pattern is too
expensive because every RHS call rebuilds a full `OperatorBundle`, including:

- operator interpolation
- target-kernel interpolation
- `c_array` reconstruction
- dissipator assembly from jumps

For static-field solves this is acceptable. For oscillating-field `RK45` solves
it creates large overhead on top of the genuine stiffness from the fixed-basis
Hamiltonian.

### Recommended fast RHS path

Keep `RK45`, but avoid rebuilding the bundle on every RHS evaluation.

Instead:

1. precompute the patch-endpoint operator pieces once
2. precompute their superoperator forms once
3. use only local interpolation inside the RHS

### Precompute by interval

For each neighboring field interval `[E_k, E_{k+1}]`, precompute and store:

- endpoint `h_internal` blocks
- endpoint `h_opt` blocks
- endpoint `h_det` blocks
- endpoint dissipator superoperators

and preferably also the interval differences:

- `dH_int = H_int(E_{k+1}) - H_int(E_k)`
- `dH_opt = H_opt(E_{k+1}) - H_opt(E_k)`
- `dH_det = H_det(E_{k+1}) - H_det(E_k)`
- `dD = D(E_{k+1}) - D(E_k)`

so the RHS can evaluate each object as:

- `A(E) = A_k + w(E) * dA_k`

using only the active neighboring interval.

### Why the dissipator can be precomputed this way

For the Lindblad-safe kernel path:

- the sector kernels are interpolated with local convex weights
- the map from sector kernels to the dissipator superoperator is linear

Therefore the final dissipator superoperator on an interval can also be
interpolated locally from the interval endpoints without reconstructing
`c_array` at every RHS call.

This keeps the runtime model consistent with the kernel interpolation, while
removing the most expensive per-step bookkeeping.

### Fast RHS target form

The time-dependent RHS should act directly on the vectorized density matrix:

- `drho_vec/dt = L(E(t), Omega(t), delta(t)) @ rho_vec`

with

- `L = L_int(E) + 0.5 * Omega(t) * L_opt(E) + delta(t) * L_det(E) + D(E)`

where each term is obtained from the precomputed local interval data.

### Scope of this optimization

This optimization should be implemented only for the new fixed-basis
compact-derived Lindblad-safe path.

Do not change the generic bundle-based solver used by the other existing model
paths.

### Solver benchmarking plan

After the common-frame correction and notebook fixes are in place:

1. benchmark the corrected one-period oscillating-field run with the current
   optimized `RK45` path
2. record:
   - wall time
   - `nfev`
   - photons over one period
3. if needed later, compare against stiff solvers such as `BDF` or `Radau`
   using the same precomputed local-Liouvillian pieces

### Notebook/runtime fixes required before judging the method

The new notebook should explicitly implement the following fixes before the
results are treated as decisive:

1. common rotating-frame fix
   - shift all compact patches into one common propagation frame using the
     compact internal-Hamiltonian transition frequency at each patch

2. `h_det` fix
   - make the runtime `h_det` construction match the exact compact block
     pattern, not the current excited-only assumption

3. `h_internal` diagnostics fix
   - report active-block-only `h_internal` error as the primary coherent
     Hamiltonian diagnostic
   - keep the full-matrix error only as a secondary sink-structure diagnostic

4. Static time-window fix
   - ensure `t_span` and `t_eval` match in the notebook static comparisons
   - ensure photon integrals are computed over the same interval actually used
     in the solve

5. One-period oscillation runs
   - for `100 kHz`, use one full period (`10 us`) as the standard oscillating
     comparison window

6. Liouvillian diagnostic interpretation
   - keep the `max Re eig(L)` diagnostic
   - also compare it to the overall Liouvillian scale, so tiny positive values
     at the level of numerical eigensolver noise are not overinterpreted

## Notebook Plan

Create a new notebook:

- `examples/effective hamiltonian/q1_fixed_basis_compact_interpolated_lindblad_safe.ipynb`

### Notebook scope

This notebook should test only the new fixed-basis compact-derived Lindblad-safe
path.

Do not compare against:

- SW
- instantaneous basis
- patchwise instantaneous basis

unless the new method still fails.

### Notebook sections

#### 1. Setup

Define:

- `TRANSITION = transitions.Q1_F1_1o2_F0`
- patch fields over the test range
- static comparison fields
- oscillating-field waveform

#### 2. Patch diagnostics

Show:

- patch fields
- coherent reduced dimensions
- decay-sector list
- excited-subspace kernel shapes

#### 3. Static operator checks

At selected fields:

- `0`
- `7`
- `7.5`
- `8`
- `10`
- `20`
- `50 V/cm`

compare the new interpolated model against the exact compact reference.

Diagnostics:

- active-block `h_internal` errors
- full-matrix `h_internal` errors
- `h_opt` block errors
- `h_det` block errors
- decay-kernel errors by sector
- explicit block norms for `h_internal` and `h_det`

#### 4. Lindblad-safety checks

At the same fields, verify:

- each interpolated `K_s(E)` is Hermitian
- each interpolated `K_s(E)` is PSD
- `sum_s K_s(E)` matches the interpolated total excited decay kernel
- maximum real Liouvillian eigenvalue is not positive

This is the main diagnostic that should replace the previous unstable behavior.

#### 5. Static dynamics checks

Run short static simulations at:

- `0`
- `7`
- `7.5`
- `8`
- `10 V/cm`

Compare:

- photon number
- scattering signal vs time
- population conservation

Use one consistent static comparison interval everywhere in the notebook. Do not
mix a longer `t_span` with a shorter `t_eval` grid for photon integration.

#### 6. Oscillating-field test

Use:

- `0 -> 50 V/cm`
- `100 kHz`
- one full oscillation period (`10 us`)

Compare:

- cumulative photons
- scattering signal
- whether the old low-field instability is gone
- runtime with the generic bundle-per-RHS path versus the precomputed local
  Liouvillian-piece RHS path

#### 7. Summary section

Summarize:

- whether the positive-real Liouvillian problem is gone
- whether the threshold-region blow-up is gone
- how static agreement compares to the old population-sink notebook

## Validation Criteria

The new path counts as successful if it achieves all of the following:

1. no positive-real Liouvillian eigenvalues at tested fields
2. no negative or exploding photon counts
3. total population conservation
4. stable oscillating-field propagation through the low-field threshold region
5. better agreement with exact compact reference than the previous
   population-sink path
6. materially lower RK45 wall time for the oscillating-field case by avoiding
   bundle reconstruction on every RHS call

## Priority Order

Implementation should proceed in this order:

1. exact patch kernel extraction
2. fixed union of decay sectors
3. local convex kernel interpolation
4. Lindblad reconstruction of the reduced dissipator
5. operator interpolation for `h_internal`, `h_opt`, `h_det`
6. precomputed local-Liouvillian RHS path for time-dependent RK45 solves
7. notebook validation on `Q1`

The most important change is step 3 plus step 4. After that, the next priority
for time-dependent use is step 6, because the generic bundle-per-RHS solve path
adds unnecessary runtime overhead on top of the genuine fixed-basis dynamics.
