# Static dense construction and reconstruction check

Measured on the Ryzen 7 9800X3D using one BLAS thread. The initial prototype
measurements below predate integration; their raw samples are frozen in
[results.json](results.json). The changes are now integrated; see the final
section. Run `benchmarks/validate_dense_reconstruction_integration.py` with the
project virtualenv to validate/time the current library against those samples.
The original `check_dense_construction_reconstruction.py` harness compares the
installed library with the prototype; rerunning it after integration no longer
measures the old library baseline and overwrites its results file.

## Generator construction

The library already constructs the packed real Liouvillian analytically from
the decomposed sparse Hamiltonian/RHS terms, without probing every basis vector.
It does not call the recently optimized D+S+U ODE RHS kernel for this operation.

A freshly built R(2) F'=4 model, pure X polarization, 10 mW, three detunings:

| Detuning (MHz) | Warm parameter binding + analytic extraction + CSR (ms) |
|---:|---:|
| 6.8 | 1.047 |
| 28.2 | 1.131 |
| 80.0 | 1.096 |

Nine measurements per point; 57,622 nonzero generator entries. Native plan and
workspace creation and common OBE construction are excluded, matching their
reuse in a dense worker. The saved same-size matrices took 695–734 ms to reduce,
factor and validate. Eliminating extraction entirely would save only about
0.15% of extraction-plus-factorization here. Further assembly SIMD work has
little end-to-end headroom for this example.

## CPU reconstruction

Three existing saved 41-level R(2) F'=4 matrices, 1,444 active packed coordinates,
three compact sink populations. These are the historical performance fixtures;
no apparatus parameters or cached physical results were updated.

Each case includes the 20 individual ground J=2 population projectors, an
initially occupied sink, and a PSD initial state with an imaginary coherence
between supported ground states. Times span 0–350 us. Five measurements per
method/case in randomized interleaved order, after correctness/warmup calls.
Factorization is reused and excluded from reconstruction timings.

Prototype changes:

- Solve for initial modal coefficients once per call rather than again for each
  sink projection/time chunk; compute sink feed using the existing feed basis
  and integrated modes. Chunk width remains 32 times.
- Unpack the full batch using NumPy triangular indexing, replacing the Python
  loops over initial states, sample times and density entries.

| Saved point | Initials × times | Current complex density (ms) | Prototype (ms) | Speedup |
|---:|---:|---:|---:|---:|
| 0 | 22 × 2 | 32.48 | 6.18 | 5.25× |
| 1 | 22 × 2 | 32.86 | 5.84 | 5.63× |
| 2 | 22 × 2 | 33.54 | 6.64 | 5.05× |
| 0 | 22 × 65 | 912.78 | 121.50 | 7.51× |
| 1 | 22 × 65 | 895.86 | 108.08 | 8.29× |
| 2 | 22 × 65 | 898.44 | 106.13 | 8.47× |

These are reconstruction speedups, not eigendecomposition speedups. Including
one factorization reduces the estimated whole-call gain to about 3–4% for two
times, or about 49% for 65 times, for these cases. These estimates add separately
measured stage medians and are not direct public whole-call benchmarks.

Most of the saving is unpacking. For 65 times, current packed reconstruction
takes 87–90 ms versus 83–89 ms for the prototype; unpacking alone drops from
813–839 ms to 21–29 ms. For two times, packed reconstruction drops from 8–9 ms
to 5–6 ms. Population/photon outputs and callers already consuming packed full
output do not incur the complex-unpacking cost and would not receive the large
speedup in the table.

All complex density entries match the current implementation to a maximum
absolute difference of 8.9e-16, including sink populations and coherences.
No thresholding, density-coordinate removal, solver tolerance change or
approximation was introduced. This is numerical equivalence to the existing
solver, not an independent validation of the saved physical model.

## CUDA scope

Source inspection shows the CUDA full-density path already assembles complex
densities with device tensor indexing and caches its basis on the device; it
does not use the slow Python unpack loops. It also solves initial coefficients
once per call. The two CPU fixes therefore do not transfer directly to that
path. No new CUDA measurements: PyTorch is absent from the current project
virtualenv. The existing CUDA implementation and dependencies were unchanged.

Recommendation: integrate the CPU reconstruction changes, with coverage for
empty time/initial batches, sink-only propagators, arbitrary supported complex
initial states, and reduced/unreduced generators. Do not pursue generator
construction optimizations based on these measurements alone.

## Integrated library validation

Both CPU changes are now automatic in the library. The vectorized conversion
is shared through `PackedHermitianLayout.unpack_batch`, including ordinary
`LindbladResult.density_matrices()` calls. Temporary complex upper triangles
are bounded to chunks of at most 2,048 matrices, further reduced for large
state spaces. Existing invalid-empty-initial-batch behavior is preserved;
empty sample grids return correctly shaped arrays.

Current library medians over five warm measurements per case:

| Saved point | Initials × times | Integrated complex density (ms) | Speedup vs saved original median |
|---:|---:|---:|---:|
| 0 | 22 × 2 | 5.24 | 6.20× |
| 1 | 22 × 2 | 5.32 | 6.17× |
| 2 | 22 × 2 | 5.58 | 6.01× |
| 0 | 22 × 65 | 104.56 | 8.73× |
| 1 | 22 × 65 | 104.70 | 8.56× |
| 2 | 22 × 65 | 105.19 | 8.54× |

The comparisons use the earlier original-library medians rather than a new
interleaved before/after run. Current library outputs match the validated
prototype; raw timings/errors are in [library_validation.json](library_validation.json).

157 tests passed, one optional CUDA test skipped (PyTorch unavailable): dense
APIs, shared batch unpacking, and Rust-backend regression tests. New independent
matrix-exponential comparisons cover reduced/unreduced systems, complex sink
coherences, initial sink populations, time-chunk boundaries, and a conversion
batch crossing 2,048 matrices. Zero generators cover sink-only propagation,
one-level models, and empty time grids. A coefficient-solve count check confirms
one solve per full reconstruction call. No native rebuild is required.
