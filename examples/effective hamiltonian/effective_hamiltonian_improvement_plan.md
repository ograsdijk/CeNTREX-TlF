# Effective Hamiltonian Improvement Plan

This note captures the current diagnosis and the recommended next steps for the
effective-Hamiltonian / reduced-Liouvillian work in
`examples/effective hamiltonian/effective_hamiltonian_runtime.py`.

## Current Diagnosis

The main problem is no longer the reduced Hamiltonian itself. The unstable
behavior is coming from how the reduced dissipator is being constructed once we
try to:

- keep a fixed reduced state space across field,
- replace external manifolds by sink channels,
- and interpolate between fields.

In particular, the current population-sink path builds the reduced open-system
model by:

1. taking the full compact dissipator superoperator,
2. projecting it onto the coherent block,
3. and adding sink feed separately.

That projected coherent dissipator is not guaranteed to remain a valid Lindblad
generator. The observed small positive-real Liouvillian eigenvalues are a direct
symptom of that.

For the new fixed-basis compact-derived notebook path, the first full notebook
run refined that diagnosis:

- `h_opt` interpolation is already essentially exact
- the large reported full-matrix `h_internal` error is mostly a sink-block
  bookkeeping mismatch; the active coherent block is essentially exact
- `h_det` is currently being interpolated with the wrong structural assumption
  in the new runtime path
- the static notebook comparison currently contains a `t_span` / `t_eval`
  mismatch that needs to be fixed before using those photon integrals as a
  benchmark
- the current compact-derived interpolation is still field-locally detuned;
  it must be shifted into one common propagation frame so the Stark shift moves
  the transition energy relative to a fixed laser reference
- the oscillating-field runtime is dominated not only by the fixed-basis
  off-diagonal dynamics, but also by unnecessary bundle reconstruction on every
  RK45 RHS call

## What Must Be Preserved

For a general transition with multiple excited hyperfine levels, preserving only
the diagonal branching-ratio sum rule is not enough.

What must be preserved is the decay-kernel structure on the excited subspace.
For each decay sector `s`, define a decay kernel

- `K_s(E)`

on the excited manifold. For a valid reduced dissipator, each `K_s(E)` must be:

- Hermitian
- positive semidefinite (PSD)

and the sector sum must reproduce the total excited-state linewidth operator:

- `sum_s K_s(E) = K_total(E)`

This is the matrix generalization of:

- nonnegative branching ratios
- branching fractions summing to `1`

for the single-excited-state case.

## Recommended Direction

The recommended path is to stop interpolating arbitrary dissipator
superoperators and instead interpolate only Lindblad-safe decay data.

The general strategy is:

1. Build the exact compact reduced model at each field patch.
2. Extract decay kernels by sector on the excited subspace.
3. Interpolate those kernels with an interpolator that preserves:
   - Hermiticity
   - PSD
   - the sector sum rule
4. Reconstruct the reduced dissipator from the interpolated kernels.

At the same time, for the fixed-basis compact-derived time-dependent path:

5. shift every compact patch into one common rotating frame before alignment and
   interpolation

This ensures that the user detuning is defined relative to one fixed reference
frequency for the whole propagation, rather than being re-zeroed at every field.

## Common Propagation Frame

The compact `qn_compact` OBE construction is naturally field-local. At each
field it provides a reduced model with zero detuning defined relative to that
field's own resonant transition.

That is not the right convention for a time-dependent propagation in a single
fixed frame.

The desired convention is:

- choose one fixed laser reference frequency for the whole run
- let the field-dependent Stark shift move the transition energy through
  `h_internal`
- interpret the runtime `detuning` parameter relative to that one fixed frame

### Recommended implementation

For each compact patch `E_k`:

1. build the compact decomposed bundle at `E_k`
2. extract the compact internal Hamiltonian at zero drive and zero user
   detuning
3. identify the addressed ground and excited states in that compact basis
4. compute the local compact transition frequency
   - `omega_patch(E_k) = E_excited(E_k) - E_ground(E_k)`

Choose a master/reference field and define:

- `omega_frame = omega_patch(E_master)`

Then correct each patch before alignment/interpolation:

- `h_internal_corrected(E_k) = h_internal(E_k) + [omega_patch(E_k) - omega_frame] * h_det(E_k)`

`h_opt`, `h_det`, and the dissipator are unchanged.

After this correction:

- `detuning = 0` means on resonance at the master field
- as the electric field changes, the transition naturally moves off resonance
  through `h_internal`

## Sector Structure

For each transition, the decay sectors should be separated into:

- active return sectors
- external sink sectors

Examples:

- `R0`: active lower manifold plus compacted sink sectors
- `Q1`: active `X, J=1` manifold plus sink sectors such as `X, J=0` and `X, J=2`

The reduced runtime model should then be built from:

- coherent Hamiltonian on the active manifold,
- decay kernels for return into the active manifold,
- decay kernels or feed channels for external sinks.

Whether external sinks are represented as:

- compact Hilbert-space spectator states, or
- scalar sink populations

should be a separate modeling choice. The decay data itself should come from the
same kernel-based construction in either case.

## Handling Changing Decay Space

This is the key point for the low-field `Q1` issue.

The compact model may expose different decay destinations at different fields.
For example:

- below some threshold, a sink sector may be absent,
- above threshold, that sector may appear with a small but nonzero decay kernel.

The recommended way to handle this is **not** to let the runtime decay-sector
bookkeeping change with field. Instead:

1. Define the **union of decay sectors** over the whole field range of interest.
2. For every field patch, extract decay kernels for **all** sectors in that union.
3. If a sector is absent at a given field, assign that sector the zero kernel:
   - `K_s(E_k) = 0`
4. Interpolate those sector kernels across field.

So the decay-space bookkeeping is fixed, even when the raw compact model starts
showing a new sink only above a threshold.

### Example

For `Q1`, the union sector set might be:

- return to active `X, J=1`
- sink `X, J=0`
- sink `X, J=2`

Then:

- below the first onset:
  - `K_{J=0}(E) = 0`
  - `K_{J=2}(E) = 0`
- between the first and second onset:
  - `K_{J=0}(E) > 0`
  - `K_{J=2}(E) = 0`
- above the second onset:
  - `K_{J=0}(E) > 0`
  - `K_{J=2}(E) > 0`

The decay-sector set stays fixed the whole time. Only the kernels vary, and they
are allowed to pass continuously through zero.

This is different from trying to force a fixed coherent Hilbert-space union of
all compact states. The proposal here is only to fix the **decay-sector**
bookkeeping, not to insist that every compact sink state must appear as a
coherent basis state at every field.

## Interpolation Constraints

The interpolation should be designed so that the result is automatically
physical, not repaired afterward if possible.

### Minimal safe option

Use local convex interpolation between neighboring patch fields:

- `K_s(E) = w0(E) K_s(E0) + w1(E) K_s(E1)`

with:

- `w0(E) >= 0`
- `w1(E) >= 0`
- `w0(E) + w1(E) = 1`

This guarantees:

- Hermitian kernels remain Hermitian
- PSD kernels remain PSD
- the sector-sum rule is preserved under the same weights

This is the safest general interpolation rule.

### Hermitian parameterization

If entrywise interpolation is still desired, only the independent Hermitian
entries should be interpolated:

- diagonal entries
- one triangle of off-diagonal entries

and the opposite triangle should be reconstructed by conjugation. This enforces
Hermiticity, but by itself it does **not** guarantee PSD.

So the recommended default remains:

- convex interpolation directly on the full sector kernels

## Hamiltonian and Operator Interpolation

The dissipator is not the only place where structure should be preserved during
interpolation. The operator interpolation rules should depend on the operator and
on the reduced-model construction being used.

### `h_internal`

`h_internal` is method-dependent and should **not** be given one universal
interpolation rule here.

Its best representation depends on the effective-model construction, for
example:

- fixed-basis Schrieffer-Wolff / local-reference reduction,
- fixed-basis compact-derived interpolation,
- instantaneous-basis interpolation,
- or patchwise instantaneous propagation.

In all cases the interpolated `h_internal` should remain Hermitian, but the
useful extra structure depends on the basis choice:

- in a fixed basis, off-diagonal couplings may be physically essential and
  should not be artificially suppressed,
- in an instantaneous basis, `h_internal` may be diagonal or nearly diagonal
  inside each patch,
- in a patchwise method, the relevant structure may be tied to the patch basis
  itself rather than a global interpolant.

So the detailed interpolation constraints for `h_internal` should be specified
per method, not globally in this note.

For the **fixed-basis compact-derived interpolation** method, the recommended
rule is:

1. Build exact compact reduced patches at selected fields.
2. Rotate each patch into one common fixed reduced basis.
3. Interpolate `h_internal` in that fixed basis using only the neighboring patch
   fields.

The interpolation constraints for this method are:

- preserve Hermiticity,
- preserve the internal block structure,
- keep ground-excited internal blocks exactly zero,
- and interpolate the `g-g` and `e-e` blocks separately.

So for a field between `E_k` and `E_{k+1}`, the default should be local
piecewise interpolation:

- use only the aligned patches at `E_k` and `E_{k+1}`,
- do not use global polynomial or spline fits across the whole field range.

This local-interpolation rule is important. The intended default is:

- interpolation should be determined only by neighboring patch fields,
- distant patch points should not influence the value at the current field,
- and no global fit should be allowed to create overshoot near thresholds or
  rapidly changing regions.

In this method, the recommended block structure for `h_internal` is:

- `g-g`: interpolated Hermitian block
- `e-e`: interpolated Hermitian block
- `g-e = 0`
- `e-g = 0`

If sink channels are represented as external populations rather than Hilbert-space
states, then they do not appear in `h_internal` at all. If sink Hilbert-space
spectator states are retained, any known zero sink blocks should be kept exactly
zero during interpolation.

### Gauge smoothing during patch alignment

The quality of the `h_internal` interpolation depends strongly on how the exact
compact patches are aligned into the fixed reduced basis before interpolation.

The recommended alignment should not only maximize overlap at each patch, but
also choose the patch-to-patch gauge so the aligned basis varies as smoothly as
possible with field.

This means:

- remove arbitrary phase flips from one patch to the next,
- choose the unitary patch alignment that minimizes basis change relative to the
  previous aligned patch,
- and, in nearly degenerate subspaces, align the whole subspace smoothly rather
  than tracking individual eigenvectors too rigidly.

This is effectively a parallel-transport-style gauge choice on the patch grid.

Good gauge smoothing often improves interpolation quality as much as changing the
interpolator itself, because it removes artificial jagged behavior from the
matrix elements before interpolation is even attempted.

### `h_det`

`h_det` should not be treated as a generic dense Hermitian matrix if more
structure is known.

The important correction from the new notebook results is that the structure
must be inferred from the aligned exact compact patches for the method being
used. It must not be hardcoded from a generic expectation.

If the construction implies that `h_det`:

- is diagonal, or
- acts only on the excited manifold, or
- has fixed zero blocks,

then those constraints should be imposed directly during interpolation.

In particular, if only the excited manifold carries detuning, then the default
interpolation rule should be:

- interpolate only the excited-excited block,
- keep ground-ground and ground-excited blocks exactly zero.

But if the aligned exact compact patches show a different nonzero block pattern,
that exact pattern should be preserved instead.

### `h_opt`

`h_opt` should preserve the optical block structure.

The physically relevant part is the ground-excited coupling block. In the usual
RWA-style construction:

- ground-ground block should remain zero,
- excited-excited block should remain zero,
- only the ground-excited block should be interpolated,
- the excited-ground block should be reconstructed as the conjugate transpose.

So the recommended interpolation rule is:

1. interpolate the `g-e` block only,
2. set `e-g = (g-e)^\dagger`,
3. force `g-g = 0`,
4. force `e-e = 0`.

This is stronger and better conditioned than treating `h_opt` as a generic full
Hermitian matrix.

### Additional structure to exploit

Where applicable, interpolation should also preserve:

- exact sparsity / selection-rule zeros,
- fixed block structure,
- known parity or field-reversal symmetry,
- and any fixed gauge/phase convention chosen for the reduced basis.

These constraints should be enforced operator-by-operator rather than using one
generic dense-matrix interpolation rule for everything.

## Why This Fixes the Current Failure Mode

The current unstable behavior appears because the coherent-block dissipator was
obtained by raw superoperator projection. That can leave the GKSL cone and
produce:

- positive-real Liouvillian eigenvalues,
- unphysical oscillations,
- and explosive photon-number behavior.

Interpolating PSD sector kernels avoids this by construction.

For the single-excited-state case, this reduces to interpolating:

- nonnegative partial decay rates

with the total-rate sum rule enforced. For multi-excited-state transitions, the
same logic is applied at the matrix-kernel level.

## Implementation Plan

### Phase 1: Exact patch diagnostics

Add helpers to extract from each exact compact patch:

- active excited-subspace decay kernel(s)
- sink-sector decay kernel(s)
- total excited-state decay kernel

and verify numerically that:

- each sector kernel is Hermitian
- each sector kernel is PSD
- sector sums reproduce the full compact total kernel

### Phase 2: Kernel-based interpolated dissipator

Add a new path, separate from the current population-sink path, that stores for
each field patch:

- coherent `h_internal`
- coherent `h_opt`
- coherent `h_det`
- decay kernels by sector on the excited subspace, using a fixed union of decay
  sectors over the whole field range

Runtime evaluation should:

1. interpolate the Hamiltonian pieces as before,
2. interpolate the sector kernels with convex weights,
3. reconstruct the reduced dissipator from those kernels.

### Phase 3: Sink representation choice

Support two sink treatments:

1. compact spectator sink states inside the reduced Hilbert space
2. external sink populations

Both should use the same interpolated sector kernels. The only difference should
be how sink channels are represented in the propagated state.

### Phase 4: Validation

Validation should compare the new kernel-based path against the exact compact
reference at static fields and across the low-field threshold regions. The main
checks are:

- no positive-real Liouvillian eigenvalues
- no negative photon numbers
- total population conservation
- stable behavior through sink-onset thresholds
- improved agreement with compact-reference photon counts

The validation notebook should also explicitly include:

- active-block-only `h_internal` error as the primary coherent-Hamiltonian
  metric
- full-matrix `h_internal` error as a secondary sink-structure metric
- explicit `h_det` block diagnostics
- a consistent static comparison time window
- one-period oscillating-field runs as the default time-dependent benchmark

### Phase 5: RK45 performance for time-dependent fields

Keep `RK45` for now, but avoid the current generic bundle-per-RHS runtime path
for the new fixed-basis compact-derived model.

For this path, the time-dependent solver should:

1. precompute endpoint Hamiltonian pieces and dissipator superoperators for each
   neighboring field interval,
2. precompute interval differences,
3. evaluate the Liouvillian inside the RHS by local interpolation only,
4. act directly on the vectorized density matrix without rebuilding a full
   `OperatorBundle` and `c_array` on every RHS call.

This optimization is consistent with the local convex decay-kernel
interpolation, because the map from sector kernels to the dissipator
superoperator is linear.

The goal is to remove unnecessary Python and reconstruction overhead while
keeping the same physical model and the same `RK45` integrator.

This performance phase should be implemented only after the following notebook
issues are fixed:

1. common propagation-frame correction for compact patches
2. correct `h_det` structure in the runtime model
3. correct `h_internal` diagnostics in the notebook
4. correct static comparison time window
5. one-period oscillating benchmark setup

## Priority

The next change should not be another basis/interpolation trick. The first
priority is to replace the current projected dissipator construction with a
Lindblad-safe kernel-based construction.

After that, the next practical priority for time-dependent use is to remove the
bundle-rebuild overhead from the `RK45` RHS for the new fixed-basis path.

For the current notebook path specifically, the immediate implementation order
should be:

1. shift compact patches into one common propagation frame
2. fix `h_det` interpolation to match the aligned exact compact block pattern
3. fix notebook diagnostics so active-block `h_internal` is the primary metric
4. fix the static `t_span` / `t_eval` mismatch
5. switch the oscillating comparison to one full period
6. then add the fast local-Liouvillian RHS path for RK45

Together, those are the minimum changes most likely to remove the present
instability and make the oscillating-field simulations usable while remaining
general across transitions with multiple excited hyperfine levels.
