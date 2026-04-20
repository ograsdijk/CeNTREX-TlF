# Local-Reference Effective Model Plan

## Goal

Extend the useful field range of the reduced effective-Hamiltonian model by using a small set of exact local baselines instead of a single global reference basis.

The intended structure is:

- exact compact `7+3` reduced model at each chosen reference field `E_ref`
- perturbative corrections only for `ΔE = E - E_ref`
- multiple local patches covering the full operating field range

## Why

The current single-reference implementation is exact at its chosen `E_ref`, but its accuracy degrades as the applied field moves away from that reference.

This degradation is not primarily from the dark-state nullspace disappearing. It comes from the reduced excited-state subspace and its branching drifting away from the exact compact dressed model as the field moves too far from the reference.

## Recommended Method

1. Choose a small set of reference fields `E_ref^(k)`.
2. At each reference field:
   - build the exact compact dressed `7+3` model
   - use that as the local baseline reduced model
   - build perturbative corrections in `ΔE`
3. During runtime:
   - select the nearest reference patch
   - evaluate the local corrected reduced model in that patch

This gives:

- exact agreement at every reference field
- better local accuracy around each reference
- a scalable way to cover a wide field range without full diagonalization at every ODE step

## Basis Handling Between Patches

Each patch has its own reduced basis, so basis alignment must be handled explicitly.

For neighboring patches:

1. Represent both reduced kept-state bases in the same parent-space basis.
2. Compute the overlap matrix:
   - `M = V_k^† V_{k+1}`
3. Compute the best unitary gauge alignment from an SVD:
   - `M = U Σ W^†`
   - `R = U W^†`
4. Rotate one patch into the other patch's gauge before comparing, switching, or interpolating:
   - `H -> R^† H R`
   - `h_opt -> R^† h_opt R`
   - `ρ -> R^† ρ R`
   - dissipator superoperators must be rotated in Liouville space

## Runtime Strategy

### Option A: Nearest-Patch Selection

- easiest and most robust
- use the nearest `E_ref`
- no interpolation required
- when switching patches during propagation, rotate the state/density matrix into the new patch basis

### Option B: Patch Interpolation

- align neighboring patch bases first
- interpolate aligned reduced operators
- for the dissipative part, interpolate reduced decay kernels or the dissipator superoperator
- do not interpolate raw jump operators directly

## Implementation Steps

1. Keep the current exact-local baseline construction.
2. Add support for building a list of local models at reference fields:
   - e.g. `100`, `150`, `200`, `250 V/cm`
3. Add overlap-based gauge alignment between neighboring local models.
4. Add a runtime patch selector:
   - initially nearest-patch only
5. Add optional patch switching for time-dependent propagation:
   - transform `ρ` with the patch-to-patch unitary when crossing regions
6. If needed later, add smooth patch blending after basis alignment.

## Validation

For each patch:

- verify exact agreement at `E = E_ref`
- scan above and below `E_ref`
- compare to the compact dressed reference:
  - sector populations
  - side excited-state subspace overlap
  - side excited-state decay kernels
  - dark-state diagnostics

## Practical Recommendation

Start with:

- exact local baselines
- nearest-patch selection
- no interpolation

Only add interpolation if patch boundaries become visible in the dynamics.
