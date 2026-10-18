# RC CPU RHS review: Julia generation versus current Rust

Reviewed 2026-10-16. Static source review and existing benchmark inspection only:
no ODE benchmark, compilation experiment or solver change was run.

## Finding

The old Julia approach contains optimizations worth recovering as an optional
prepared-system specialization. The current Rust implementation already exploits
Hermiticity, sparse Hamiltonians, structured decay and preallocated workspaces;
these are not missing. The remaining distinction is generated native arithmetic
for a particular system versus generic kernels traversing its prepared data.
A CPU speedup is plausible, not demonstrated by this review or by comparing
runs on different hardware/tolerances.

## Julia paths verified in the sibling extension

`centrex_tlf_julia_extension/lindblad_julia/generate_julia_code.py`:
`system_of_equations_to_lines` runs SymPy CSE on the expanded symbolic system,
rewrites lower-triangle references using conjugates, and emits diagonal/upper
triangle assignments. `generate_preamble` binds typed parameters and compound
variables inside `@inbounds`.

`utils_setup.py:generate_OBE_system_julia` provides two distinct paths:

- `expanded`: generated system-specific ODE equations with CSE.
- `matrix`: generated Hamiltonian and dissipator functions, with a generic
  `liouvillian_commutator_her2k!` operation. This path is not a fully expanded
  specialized commutator.

`utils_julia_matrix_assemble.py:sympy_matrix_to_julia_fill_hermitian` applies CSE
only to the selected triangle, prunes unused temporaries, inlines single-use
expressions and emits nonzero assignments. `utils_julia_matrix.py` uses typed
functors to connect generated functions to the ODE RHS.

The old notebook's exact selected path still matters; the extension supports
both. Generated Julia code does not by itself establish the original tolerance,
state count, RHS evaluation count or hardware-normalized throughput.

## Current implementation and opportunities

1. **Prepared-system native RHS specialization (highest priority to benchmark).**
   `rust/src/lindblad/rhs.rs` has optimized generic commutators but still traverses
   sparse indices and scratch rows. The D+S+U SoA path reconstructs a full dense
   rho buffer, accumulates separate real/imaginary scratch products, then packs
   the derivative. An optional generated kernel could bake in sparsity, numeric
   coupling coefficients, index offsets and zero structure, retaining runtime
   drive/detuning slots. Generate real packed-Hermitian arithmetic, with the
   structured decay fused where useful. Preserve all density coordinates.
   Compare small tiled/unrolled kernels against fully expanded expressions:
   huge straight-line functions and global CSE may increase instruction-cache
   pressure, register spills and compile time. Faster is not guaranteed.

2. **Specialize the four microwave polarization configurations.**
   Existing kernels test runtime-zero Hamiltonian entries and use a union
   sparsity pattern. Prebuild active patterns or generated kernels for the four
   configurations. Select the configuration at a switching boundary and reuse
   it within that smooth segment. Gaussian/EOM/optical Doppler coefficients
   remain time-dependent. The library already clips affine switch boundaries;
   specialize rather than repeat an existing clipping implementation. Switching
   state selection must be correct at RK stages and endpoint boundaries.

3. **Hoist static assembly and decay work out of RHS calls.**
   `plan.rs:fill_upper_into` clears and reconstructs the full static triangle
   for a time-dependent decomposed Hamiltonian. Cache its packed representation
   and precompute offsets at preparation rather than repeatedly validate indices.
   `add_structured_dissipator_upper_dense_rho` visits every pair even when both
   decay rates are zero; prepare only nonzero damping and transfer entries.
   Measure these small changes before introducing generated-code dependencies.
   Static-Hamiltonian caching already exists; this recommendation concerns
   static contributions inside a time-dependent Hamiltonian, not caching the
   full time-dependent matrix incorrectly.

4. **Compile runtime expressions / cross-coefficient CSE (secondary priority).**
   `eval.rs:eval_scalar_expression_into` executes an instruction stream, rather
   than native code specialized to each expression. The entrywise Python IR
   does perform CSE, while `_lower_hamiltonian_upper_triangle_decomposed`
   groups identical symbolic coefficients but compiles each coefficient
   separately. A shared coefficient DAG or compiled waveform function can
   remove repeated subexpressions that survive compound-parameter factoring.
   Many compound parameters are already evaluated once in the parameter graph;
   do not count that existing reuse as a new optimization.

## Existing profiling evidence

`rc_propagation_results/rc_kernel_matrix_v1/nominal/profile.json` records an
older sparse/dense single-unpack variant at about34.6 us/RHS call for the direct
initial state. Its instrumented component times relative to native total were
approximately86% commutator,9% Hamiltonian fill,5% unpack,4% dissipator and2%
pack; scalar parameter evaluation was0.48%. Timers include nested work and
instrumentation, so these percentages overlap and are not an additive budget.
This is an archived related variant, not a new profile of the latest D+S+U
kernel or a multipass production trajectory. It nevertheless argues against
assuming scalar-expression compilation alone will yield a large overall gain.

The expanded-sparse implementation in the same archive was slower, illustrating
that flattening equations into sparse contribution streams is not equivalent
to generating an efficient native RHS. `rhs.rs` also documents cache-locality
limitations of per-output streams. Preserve the D+S+U baseline.

## Recommended matched experiment

Use the same freshly built full70-state RC system, same H and collapse operators,
waveforms, initial states, switching boundaries and output for:
current Rust D+S+U, current expanded_sparse, Julia expanded, Julia matrix, and
one generated real packed RHS prototype. Reuse preparation across scan points.

Start with direct RHS comparisons at arbitrary Hermitian states, several EOM
phases and all four switch configurations. Then compare full trajectories with
matching error against a strict reference, recording RHS counts, accepted and
rejected steps, solve-only time, generation/compilation time and peak memory.
Run one-core and matched six-/twelve-worker throughput on the same machine with
one BLAS thread per worker. Equal solver tolerances alone do not guarantee equal
physical accuracy across different layouts (Julia complex matrix versus Rust
real packed state); use the agreed entrywise density check as well.

This separates cheaper RHS evaluations from changes in adaptive step counts,
and makes compilation amortization explicit. No speedup magnitude is promised.

## Completed Rust-only benchmarks

[Initial generated/expanded tests](rust_specialized_rhs_results/REPORT.md) and
[separately generated H/D follow-up](rust_specialized_rhs_results/SPLIT_FUNCTIONS_REPORT.md)
are complete. The latter retains the existing D+S+U commutator and shows8.7%
less RC solve time and10.4% less Q(1) solve time in paired10 us tests. These are
benchmark-only prototypes, not library integration or full production timings.

The previously missing [full separately generated H RHS and D RHS test](rust_specialized_rhs_results/SEPARATELY_GENERATED_RHS_REPORT.md)
is also complete. It applies independent CSE and Hermitian simplifications and
does not call the generic commutator. It is 5.47 times slower for RC, 7.87% slower
for R(2), 12.36% faster for R(0), and 3.83% faster for Q(1) in the tested short
trajectories. All tested complex densities and isolated contribution checks pass.

## Generic prepared-data follow-up

[Generic prepared-data factorial experiments](rust_specialized_rhs_results/GENERIC_PREPARED_RHS_REPORT.md)
test all 16 combinations of prepared Hamiltonian assembly, nonzero-only decay,
shared coefficient DAGs and active binary switching patterns across four sparse
commutator variants. One reusable compiled binary serves all four systems.
Prepared assembly alone reduces matched 10 us solve times by 7.5-9.2%; the best
RC combination reduces time by 11%. D+S+U SoA remains the winning family. All
tested complex density and runtime-parameter-change checks pass. These remain
benchmark prototypes; production integration and apparatus-context validation
are separate work. The report records adapter overhead and noisy timing samples.

Prepared Hamiltonian assembly and shared scalar coefficient DAGs have since been
[integrated into the library](rust_specialized_rhs_results/PREPARED_HAMILTONIAN_INTEGRATION.md).
Matched before/after tests of the integrated path give 6.7-9.5% reductions across
these four fixtures; all density checks and backend regressions pass. The extra
RC-specific decay/switching experiments are not enabled by this integration.
