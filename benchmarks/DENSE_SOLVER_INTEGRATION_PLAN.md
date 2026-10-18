# Proposed dense Lindblad solver integration

Planning only. No changes to the main library have been made or authorized by this plan.

## Public API

Add `solver="dense_eig"` to `solve_lindblad`, `solve_lindblad_batch`,
`initial_condition_scan`, `parameter_scan`, and `grid_scan`. Keep the existing
default solver unchanged. Add `evolution_device="cpu"` (default) or `"cuda"`;
both choices perform eigendecomposition and initial-state solves on the CPU.
The CUDA option only evaluates modal evolution/observable projections using
optional PyTorch. Do not include GPU eigendecomposition or autograd in this work.

Use existing result classes, state ordering, parameter-slot conventions and
output shapes. In batch mode group identical parameter rows and solve their
initial states as columns of one decomposition. Grid scans must support several
independent initial states per parameter point, with explicit, documented
parameter-point/initial-state ordering and reshape metadata. An explicit
`rho0_batch` grid argument is preferable to overloading the existing matrix input.

Expose a reusable `DenseLindbladPropagator` through a preparation function so
callers can retain a factorization, change initial populations and sample times,
and reconstruct the full density matrix without decomposing again. Parameters
are an immutable snapshot; changed Hamiltonian/decay parameters need a new
propagator. Do not silently reuse a factorization after parameter mutation.

## Numerical implementation

1. Require a time-independent prepared model. Check the entire runtime parameter
   graph and Hamiltonian expressions, including indirect time dependencies.
   Reject modulation, time-dependent envelopes and terminal events explicitly.
   Continue to use the Rust prepared plan for analytic packed Liouvillian
   extraction; this is a CPU solver with Rust model lowering, not a new
   preparation backend.
2. Extract the exact packed real Liouvillian with the existing analytic Jacobian.
   Use the validated real eigenbasis and real LU solve, with one representative
   from each complex-conjugate pair. Evaluate `exp(lambda * dt)` and the stable
   `expm1(lambda * dt) / lambda` integral, including zero modes. Use elapsed time
   from the supplied initial time, not absolute time.
3. Generalize the benchmark's sink reduction. Remove only structurally zero,
   unreachable coherences and zero-feedback sink variables when the initial
   conditions permit it. Reconstruct sink populations analytically, including
   any initial sink population. No numerical magnitude pruning. If reduction
   cannot be proved valid, retain the full packed generator. For a reusable
   reduced factorization, detect and reject unsupported later initial coherences
   or provide an explicit full-factorization path.
4. Preserve full packed density output, populations, selected complex coherences,
   weighted rates and analytic cumulative integrals. Keep integral values
   independent of the requested sample grid. Define sampled-quadrature behavior
   explicitly rather than silently changing the existing API. Initially reject
   unsupported event/option combinations with actionable messages.
5. Check factorization residuals and eigenbasis conditioning. Defective or
   numerically singular eigenbases must fail clearly with an ODE-solver suggestion;
   do not return plausible but incorrect trajectories or silently change methods.
   Explain that ODE timestep/tolerance options do not control spectral accuracy.

## Parallelism and optional CUDA evolution

Use CPU processes across distinct parameter points, with one BLAS thread per
worker and all initial states sharing that point's decomposition. Reuse prepared
payloads/workspaces within workers; do not pass unpicklable Rust handles between
processes. Preserve scan ordering after asynchronous completion. Support serial
execution and explicit worker counts through the existing parallel/threads API.
Use physical-core discovery for an automatic count where available, with a
documented conservative fallback; explicitly use eight workers on this system.

For the optional hybrid path, CPU workers send only eigenvalues and projected
coefficients to a single GPU consumer. Evaluate ready points in bounded batches
(initial benchmark choice: eight), padding differing mode counts with zero
weights. Include backpressure and a memory bound rather than queuing the whole
scan on CUDA. Keep eigenvectors on the CPU for photon-only output. Full density
reconstruction may use larger projections or the CPU; document the chosen path.
Do not run threaded CUDA eigensolvers.

Import PyTorch only when CUDA evolution is requested. Validate installation,
CUDA availability and the requested device before launching workers. Add an
optional dependency extra; base installation must work without PyTorch. Choose
how physical-core discovery and BLAS thread limiting are packaged explicitly
(psutil/threadpoolctl are currently present locally but not declared dependencies).
No differentiability claims: this remains a forward NumPy/SciPy/PyTorch solver.

## Validation and delivery sequence

1. Implement and test the CPU propagator and reconstruction independently.
2. Wire single, batch and grid APIs; test parameter grouping, multiple initial
   states, ordering, shapes, metadata and worker error propagation.
3. Add optional GPU evolution using the same CPU factors. Test missing dependency
   and unavailable CUDA handling; run numerical CPU/CUDA comparisons locally.
4. Compare small driven/decaying models with independent `scipy.linalg.expm` and
   existing ODE results, including coherent initial states, repeated/zero
   eigenvalues, a defective generator, nonzero start times, dark sinks and an
   initially populated sink. Check trace, Hermiticity, positivity and integrals.
5. Recheck the saved F4 matrices and selected existing power/polarization cases,
   including density reconstruction and photons. Repeat a small throughput test
   through the library API, including parameter binding/extraction, communication
   and output overhead. No full scan is needed for integration validation.
6. Update solver documentation, public exports and AGENTS.md with examples,
   eligibility limits, CPU worker settings and the optional GPU dependency.
   Record hardware, software, BLAS, thread/process settings and benchmark scope.

Measured anchor: eight-worker CPU throughput 7.98 points/s versus 9.17 points/s
for CPU decomposition plus batch-eight CUDA photon evaluation in the paired
benchmark. This is about 15% more throughput, based on two repetitions of three
saved matrices. Preserve the exact active-block reduction and shared-population
decomposition; these are much larger gains than optional GPU projection.
