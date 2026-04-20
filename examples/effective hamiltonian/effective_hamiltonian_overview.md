# Effective Hamiltonian Overview

This document explains the effective-Hamiltonian model implemented in
`examples/effective hamiltonian/effective_hamiltonian_runtime.py`.

The focus here is the model as it exists now:

- fixed reference basis
- full parent-space cached operator blocks
- compact propagated `7+3` model
- exact compact local baseline at `E_ref`
- perturbative correction away from that reference

It also explains what is done once at model-preparation time and what is still
evaluated as a function of the current field during runtime.

## Goal

The goal is to build a reduced driven-dissipative model that:

- propagates only a small coherent subspace `P`
- represents the rest of the parent Hilbert space as an eliminated subspace `Q`
- avoids full parent-space diagonalization during runtime
- remains exact at a chosen reference field `E_ref`
- stays accurate for moderate deviations away from that reference

This is **not** an instantaneous-eigenbasis propagation scheme. The propagated
basis is fixed once the model has been prepared.

## Spaces

The implementation uses three levels of description.

### 1. Parent space

This is the full fixed-basis space used for perturbative reduction.

For the current TlF example it contains:

- `64` X states
- `120` B states from the `J = 1..3` construction space

so the parent-space size is `184`.

### 2. Kept coherent space `P`

The kept coherent subspace contains `7` states:

- all `X, J = 0` states (`4`)
- the `B, J = 1, F1 = 1/2, F = 1, P = -` triplet (`3`)

These are the coherently propagated states.

### 3. Sink states

The final reduced propagated model also includes `3` explicit sink states:

- `X, J = 1` sink
- `X, J = 2` sink
- `X, J = 3` sink

So the final propagated reduced model has `10` states total:

- `7` coherent states
- `3` sinks

## Fixed basis

The model is always built in a fixed basis.

That basis is chosen by diagonalizing the **full parent internal Hamiltonian**
once at the chosen reference field:

- `E = E_ref`
- `B = B_ref`

The result is a single fixed basis used for:

- the parent internal Hamiltonian terms
- the optical operator
- the detuning operator
- the collapse operators

This diagonalization happens only once when the model is prepared. It is not
repeated during runtime.

## What is done at preparation time

When `prepare_effective_hamiltonian_model(...)` is called, the following is
done once:

1. Build the full parent X and B Hamiltonian terms.
2. Diagonalize the full parent internal Hamiltonian at `E_ref`, `B_ref` to
   define the fixed basis.
3. Transform all parent-space operators into that fixed basis:
   - internal Hamiltonian terms
   - optical operator
   - detuning operator
   - collapse operators
4. Define the kept coherent states `P` and eliminated states `Q`.
5. Cache the `PP / PQ / QP / QQ` blocks of all relevant operators.
6. If the compact local-reference mode is applicable, also build:
   - the exact compact dressed `7+3` model at `E_ref`
   - the perturbative reduced model at `E_ref`

After this step, runtime no longer needs any full parent-space diagonalization.

## Exact call flow

This section describes exactly what happens in the two main entry points.

### `prepare_effective_hamiltonian_model(...)`

When `prepare_effective_hamiltonian_model(...)` is called, the following objects
are computed and stored once:

1. The full parent X and B state lists.
2. The fixed reference basis from full parent-space diagonalization at
   `E_ref`, `B_ref`.
3. The parent-space operators in that fixed basis:
   - `H0`
   - `HSx`, `HSy`, `HSz`
   - `HZx`, `HZy`, `HZz`
   - bare optical operator
   - bare detuning operator
   - bare collapse operators
4. The kept and eliminated index sets:
   - `p_indices`
   - `q_indices`
5. The kept excited and kept ground subsets inside `P`.
6. The sink index groups inside `Q`.
7. The cached blocks of every relevant parent-space operator:
   - `PP`
   - `PQ`
   - `QP`
   - `QQ`

If the compact local-reference mode applies, it also computes and stores:

8. The exact compact dressed `7+3` bundle at `E_ref`.
9. The perturbative reduced bundle at `E_ref`.

These are stored in the prepared model object and reused later.

### `effective_bundle(E, B)`

When `effective_bundle(E, B)` is called, the current implementation does the
following:

1. Evaluate the perturbative reduced bundle at the current field `E, B`.
2. If no local exact baseline is available, return that perturbative bundle.
3. If a local exact baseline exists:
   - take the exact compact reference bundle at `E_ref`
   - subtract the perturbative bundle evaluated at `E_ref`
   - add the perturbative bundle evaluated at the current field
4. Return the resulting reduced bundle.

So in compact form:

`effective_bundle(E) = compact_exact(E_ref) + [ perturbative(E) - perturbative(E_ref) ]`

This is why the returned reduced model is exact at `E = E_ref`.

### `_effective_bundle_perturbative(E, B)`

The actual perturbative runtime construction is inside
`_effective_bundle_perturbative(E, B)`.

That function performs these steps every time it is called:

1. Assemble the current internal Hamiltonian blocks from cached blocks:
   - `H_PP(E)`
   - `V_PQ(E)`
   - `V_QP(E)`
   - `H_QQ(E)`
2. Assemble the full parent Hamiltonian only from those already-assembled
   blocks when needed for diagnostics/output.
3. Compute the current reference transition frequency from the kept block.
4. Solve the block-SW Sylvester equation for the current field to get `S(E)`.
5. Build the second-order coherent Hamiltonian correction.
6. Build the current truncated dressed-state transform from `S(E)`.
7. Use those dressed states to compute:
   - reduced optical operator
   - reduced detuning operator
   - reduced decay amplitudes
8. Build the reduced dissipator superoperator from the dressed decay structure.
9. Embed the coherent `7`-state model plus the `3` sink states into the final
   `10 x 10` reduced bundle.

So what is field-dependent at runtime is:

- the assembled Hamiltonian blocks
- the SW generator
- the dressed states
- the reduced optical operator
- the reduced decay kernels / dissipator

What is **not** recomputed at runtime is:

- the fixed parent basis
- the transformed parent operators
- the cached `PP / PQ / QP / QQ` operator blocks
- the exact compact baseline at `E_ref`
- the perturbative reduced bundle at `E_ref`

## What is evaluated at runtime

When `effective_bundle(E, B)` is called:

1. The current field-dependent operator blocks are assembled from cached blocks.
2. The perturbative reduction is evaluated at the current field.
3. Optical and decay structures are evaluated at the current field using the
   current perturbative dressed-state construction.
4. If a local exact baseline exists, the final bundle is formed as:

   `exact compact baseline at E_ref`
   `+ [perturbative(E) - perturbative(E_ref)]`

So the model is still a function of the current field, but the large
parent-space work has already been moved into preparation time.

## Internal Hamiltonian structure

In the fixed reference basis, the parent internal Hamiltonian is represented as

`H_int(E, B) = H0 + Ex HSx + Ey HSy + Ez HSz + dBx HZx + dBy HZy + dBz HZz`

with

- `dB = B - B_ref`

The code caches the `PP / PQ / QP / QQ` blocks of:

- `H0`
- `HSx`, `HSy`, `HSz`
- `HZx`, `HZy`, `HZz`
- optical operator
- detuning operator
- collapse operators

Runtime only assembles field-dependent combinations of these cached blocks.

## Meaning of the `PP / PQ / QP / QQ` blocks

Once the parent space is split into `P` and `Q`, any parent-space operator can
be written in block form:

```text
O =
[ O_PP  O_PQ ]
[ O_QP  O_QQ ]
```

The blocks mean:

- `O_PP`: the part of the operator acting entirely within the kept space `P`
- `O_QQ`: the part acting entirely within the eliminated space `Q`
- `O_PQ`: the part coupling `Q` into `P`
- `O_QP`: the part coupling `P` into `Q`

For the internal Hamiltonian this becomes:

- `H_PP(E)`: exact coherent Hamiltonian kept explicitly
- `V_PQ(E)`: coupling from kept states into omitted states
- `V_QP(E)`: Hermitian-conjugate coupling back
- `H_QQ(E)`: Hamiltonian of the eliminated space

The whole perturbative reduction is about using `V_PQ`, `V_QP`, and `H_QQ` to
compute how the omitted states shift and dress the effective dynamics inside
`P`.

## What `P` and `Q` mean mathematically

The letters `P` and `Q` are being used in two related ways:

- as names for the kept and eliminated subspaces
- as the corresponding projection operators onto those subspaces

If the full parent Hilbert space is `H_parent`, then we split it as

`H_parent = H_P ⊕ H_Q`

where:

- `H_P` is the kept coherent subspace
- `H_Q` is the eliminated subspace

The corresponding projectors satisfy

- `P^2 = P`
- `Q^2 = Q`
- `P + Q = I`
- `PQ = QP = 0`

In the actual code, once the basis has been ordered so that the kept states come
first and the eliminated states come second, these projectors are represented
implicitly by index lists:

- `p_indices`
- `q_indices`

That is why the code can extract blocks by slicing arrays instead of explicitly
building large dense projector matrices.

## How the blocks are constructed in code

Suppose `O` is any parent-space operator in the fixed reference basis, written
as a full `N x N` matrix.

Then the mathematical blocks are

- `O_PP = P O P`
- `O_PQ = P O Q`
- `O_QP = Q O P`
- `O_QQ = Q O Q`

In matrix form, after the basis is ordered as `[P, Q]`, this becomes

```text
O =
[ O_PP  O_PQ ]
[ O_QP  O_QQ ]
```

In the implementation, those blocks are created by taking the corresponding
rows and columns of the full parent-space operator:

- `O_PP = O[np.ix_(p_indices, p_indices)]`
- `O_PQ = O[np.ix_(p_indices, q_indices)]`
- `O_QP = O[np.ix_(q_indices, p_indices)]`
- `O_QQ = O[np.ix_(q_indices, q_indices)]`

This is done once at preparation time for all cached operators:

- internal Hamiltonian terms
- optical operator
- detuning operator
- collapse operators

At runtime, the field-dependent blocks are assembled by combining those cached
blocks with the current field components.

## Perturbative reduction of the coherent Hamiltonian

At runtime the current field-dependent internal blocks are assembled:

- `H_PP(E)`
- `V_PQ(E)`
- `V_QP(E)`
- `H_QQ(E)`

Then the code solves the block Schrieffer-Wolff Sylvester equation

`H_PP X - X H_QQ = V_PQ`

to obtain the field-dependent off-diagonal dressing map.

## Why the Schrieffer-Wolff equation looks like this

The full parent-space internal Hamiltonian is written as

```text
H =
[ H_PP  V_PQ ]
[ V_QP  H_QQ ]
```

where:

- `H_PP` is the part we want to keep exactly
- `H_QQ` is the omitted-space Hamiltonian
- `V_PQ`, `V_QP` couple the kept and omitted subspaces

The aim of Schrieffer-Wolff reduction is to find a basis transformation that
removes the off-diagonal `P-Q` coupling order by order, so that the effective
Hamiltonian is block-diagonal up to the desired perturbative order.

The transformation is written in terms of an anti-Hermitian generator `S`:

`H_tilde = e^(-S) H e^(S)`

with `S` chosen to be block-off-diagonal:

```text
S =
[  0    S_PQ ]
[ S_QP   0   ]
```

and for Hermitian problems

`S_QP = -S_PQ^†`

If we expand to first order in `S`, the transformed Hamiltonian is

`H_tilde = H + [H, S] + ...`

To remove the first-order block-off-diagonal part, we require the `P-Q` block of

`H + [H, S]`

to vanish.

Keeping only first-order terms in the off-diagonal coupling gives

`V_PQ + H_PP S_PQ - S_PQ H_QQ = 0`

which is equivalent to

`H_PP S_PQ - S_PQ H_QQ = -V_PQ`

Depending on sign convention for `S`, this is the same Sylvester problem as the
one solved in code. In the implementation the sign is absorbed into the chosen
definition of `S_PQ`, so the solve is written as

`H_PP X - X H_QQ = V_PQ`

and then the anti-Hermitian generator is assembled from that `X`.

The important point is:

- the solve uses the **full kept block** `H_PP`
- the **full eliminated block** `H_QQ`
- not just their diagonals

That is why this is a block-SW construction rather than a scalar-denominator
isolated-level perturbation theory.

## Why this is called a Sylvester equation

A Sylvester equation is a matrix equation of the form

`A X + X B = C`

The block-SW equation can be written in this form by identifying

- `A = H_PP`
- `B = -H_QQ`
- `C = V_PQ`

so it can be solved efficiently with standard linear-algebra routines without
diagonalizing the full parent Hamiltonian at runtime.

From this, the anti-Hermitian block generator `S(E)` is built:

```text
S(E) =
[   0    S_PQ(E) ]
[ S_QP(E)   0    ]
```

with `S_QP(E) = -S_PQ(E)^†`.

The reduced coherent Hamiltonian on `P` is then

`H_eff(E) = H_PP(E) + 1/2 [ S_PQ(E) V_QP(E) - V_PQ(E) S_QP(E) ]`

## Derivation of the second-order effective Hamiltonian

Starting from

`H_tilde = e^(-S) H e^(S)`

the Baker-Campbell-Hausdorff expansion gives

`H_tilde = H + [H,S] + 1/2 [[H,S],S] + ...`

The Schrieffer-Wolff generator is chosen so that the block-off-diagonal part is
removed to first order. Once that is done, the block-diagonal part of
`H_tilde` contains the second-order virtual effect of `Q` on `P`.

Projecting onto the kept block gives

`H_eff = P H_tilde P`

and, after using the first-order cancellation condition, the kept block becomes

`H_eff = H_PP + 1/2 ( S_PQ V_QP - V_PQ S_QP )`

This is exactly the formula used in the code.

Interpretation:

- `H_PP` is the exact kept-space Hamiltonian
- the second term is the virtual correction from excursions into `Q`

So the effective Hamiltonian contains:

- the exact coherent dynamics already present inside `P`
- plus the second-order shift and renormalization induced by the eliminated
  states

That correction can modify:

- diagonal energies
- off-diagonal couplings inside `P`

which is why the reduction can change not only level positions but also mixing
and coupling structure inside the kept manifold.

## What is perturbative and what is exact inside `H_eff`

Inside the current construction:

- everything in `H_PP(E)` is exact
- only the effect of leaving `P`, evolving in `Q`, and returning to `P` is
  treated perturbatively

So this is a **block perturbation theory around a kept subspace**, not a
perturbation theory of isolated bare levels.

This means:

- `H_PP(E)` is kept exactly
- the omitted space only enters through its second-order virtual effect on `P`

## Optical and decay construction

The optical and dissipative parts are also functions of the current field.

They are **not** taken once at preparation time and then frozen forever.

Instead, at each `effective_bundle(E, B)` call, the code:

1. builds the current generator `S(E)`
2. builds a truncated dressed-state transform from `S(E)`
3. uses those dressed states to evaluate matrix elements of the **bare**
   optical and collapse operators

So the optical and decay pieces are re-evaluated at each field from the cached
parent-space operators plus the current perturbative dressing.

No full parent-space diagonalization is done at runtime.

### Dressed-state transform used currently

The current implementation uses a truncated state transform derived from
`S(E)`.

Currently:

- second order is used for the general kept/sink basis
- third order is used for the kept excited columns

This was introduced because the excited side doublet was more sensitive than
the rest of the reduced basis.

### Optical coupling

The reduced optical operator is built from matrix elements of the **bare**
optical operator between:

- dressed kept ground states
- dressed kept excited states

So the optical operator is field-dependent because the dressed states are
field-dependent.

### Decay / dissipator

The reduced decay structure is built from matrix elements of the **bare**
collapse operators between:

- dressed kept excited states
- dressed kept ground states
- dressed omitted X states

The omitted X states are grouped into the compact sink sectors:

- `J = 1`
- `J = 2`
- `J = 3`

From these dressed decay amplitudes the code builds:

- a recycling kernel back into the kept `J = 0` block
- sink-sector decay kernels
- an explicit reduced dissipator superoperator on the final `10`-state model

So the optical and decay pieces are both evaluated at the current field using
the current perturbative dressed-state construction.

## Local-reference exact baseline

The most important later improvement was the local-reference construction.

### Why it was needed

A model centered at `0 V/cm` was only accurate over a small field range. The
main failure was not the dark-state nullspace itself, but drift of the reduced
excited-state subspace and its decay branching as the field moved too far from
the zero-field reference.

### What the current code does

For a chosen reference field `E_ref`, the model builds:

1. the exact compact dressed `7+3` model at `E_ref`
2. the perturbative reduced model at `E_ref`

Then, at runtime for a current field `E`, it builds:

3. the perturbative reduced model at `E`

Finally the returned effective model is

`exact compact baseline at E_ref`
`+ [perturbative(E) - perturbative(E_ref)]`

This has an important consequence:

- when `E = E_ref`, the perturbative difference vanishes
- so the reduced model is **exactly equal** to the compact dressed `7+3` model
  at that reference field

Away from `E_ref`, only the deviation from the reference is treated
perturbatively.

### What is exact and what is approximate

At `E = E_ref`:

- `h_internal`
- `h_opt`
- `h_det`
- the reduced dissipator
- decay kernels
- branching

all match the compact dressed reduced model exactly.

For `E != E_ref`:

- the baseline remains the exact compact model at `E_ref`
- the change away from that field is approximated by the perturbative parent
  reduction

So this is a **local exact baseline** plus **local perturbative correction**
scheme.

## Why the model is still local, not global

Even with the exact local baseline, the model is still only locally accurate.

It becomes less accurate as the field moves too far from `E_ref`, because the
reduced excited-state subspace and decay structure drift away from the exact
compact dressed model faster than the local perturbative correction can track
them.

This is why the recommended extension path is not one global model over a huge
field range, but multiple local patches.

## Diagnostics

Several diagnostics were added to understand where the reduced model succeeds or
fails.

Important helpers include:

- `summarize_static_comparison(...)`
- `compact_reference_diagnostics(...)`
- `excited_state_composition_diagnostics(...)`
- `side_excited_subspace_diagnostics(...)`
- `scan_local_reference_window(...)`

The most useful quantities are:

- dark-state overlap
- side excited-subspace singular values
- side projector error norm
- sector-resolved decay-kernel errors
- short-time sector-population mismatch

## Practical use

### Build a local model around an operating field

Example:

```python
model = ehr.prepare_effective_hamiltonian_model(
    reference_electric_field=(0.0, 0.0, 200.0),
    space_generation_electric_field=(0.0, 0.0, 200.0),
)
```

This makes the reduced model exact at `200 V/cm`.

### Scan the useful local range

Example:

```python
scan = ehr.scan_local_reference_window(
    reference_electric_field=200.0,
    scan_fields=[150.0, 175.0, 200.0, 225.0, 250.0],
)
```

This returns, for each scanned field:

- side excited-subspace mismatch
- relative side decay-kernel errors
- short-time sector-population mismatch versus the compact dressed reference

## Recommended extension path

The recommended next step is not one global perturbative model over a very wide
field range.

Instead, use several local reference patches:

- choose a small set of `E_ref` values
- build an exact local baseline at each one
- use perturbative corrections only for `ΔE = E - E_ref` inside each patch

This is described in:

- `examples/effective hamiltonian/local_reference_patch_plan.md`

## Summary

The current effective-Hamiltonian model is:

- a fixed-basis reduced model
- built from a full parent space and a compact propagated `7+3` model
- using block-SW reduction for the coherent internal Hamiltonian
- using current-field dressed states for optical and decay construction
- and, most importantly, using an **exact compact local baseline at `E_ref`**
  plus perturbative corrections away from that reference

That local-reference version is the one that should be used going forward.
