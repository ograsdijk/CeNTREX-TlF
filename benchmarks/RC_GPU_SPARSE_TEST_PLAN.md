# RC GPU sparse-method comparison

Additional full-complex-output check: one eigensystem reused across the same
20 initial populations × velocity grid, now returning ordinary complex128
41×41 matrices. GPU evaluation/unpacking/transfer took 0.153 s for 10,020
matrices versus 0.582 s CPU (3.80x). The 420/2020-output workloads gave about
3.5x/4.7x. See `r2_gpu_reuse_results/dense_complex_v1/`. Core tests below were
complete before the inherited cutoff; this follow-up completed on the later
keep-going turn. The deadline guard then refused the planned full-transit
EOM extension, so EOM reference coverage remains 1 us. No further GPU runs.

R(2) and cached static-eigensystem follow-up completed 2026-10-12. Results:
`r2_gpu_reuse_results/REPORT.md`. One CPU eigendecomposition was reused across
20 initial J=2 populations and 1/21/101/501 forward velocities (20–10,020 full
packed-density outputs). With modal functions shared across initial populations
on both devices and CPU BLAS/batch sizes tuned, reused GPU evaluation was about
3.3–4.9x faster for the larger workloads; 10,020 outputs took 0.114 s GPU versus
0.476 s CPU. These include all density coherences, exact sink recovery and host
transfer, excluding one-time factorization/upload. Small 20-output jobs favor CPU.
This is a benchmark prototype of GPU full-density reconstruction, not a public
solver change. Velocities change only duration in the uniform static test.

Against the frozen 8-point, 40-initial-state R(2) CPU workload, GPU fixed Tsit5
at 0.5 ns took about 1.76 s versus the archived 8-worker CPU D+S+U time 4.63 s
(about 2.6x); worst complex-entry error was 0.1803%. The 1 ns wider scan failed
at 10.75% and is excluded. The installed native binary differs from the archived
CPU build; physical configuration and all saved density references were checked.
At batch 512 with R(2) EOM phase modulation, packed/staged storage and graph/stage
changes saved about 19% at the same 1 ns step. A 2 ns step passed the 60-case
1 us screen; 4 ns failed. Static nominal checks passed through 400 us. The IF
prototype slightly failed the 1% criterion at 20 us and is not retained. No CPU
ODE reruns or production changes; these core jobs finished before the inherited cap.

V3 authorized 2026-10-12: (1) exact even-step tail graphs, (2) diagonal-only
GPU population-history buffers with full-density transfers only at endpoints,
(3) packed upper-triangle Hermitian density storage with staged and direct RHS
variants, and (4) local-step integrating-factor Tsit5 for the Hamiltonian
diagonal plus decay loss, retaining sparse coupling and population transfers.
Completed in `rc_gpu_integrator_results/v3_20261012_135724/`. Retained the existing
16:37:10 UTC deadline from v2; no CPU ODE reruns and no production changes.
Correctness probes include arbitrary coherent PSD density matrices against
the established full FP64 RHS. Promising methods require the 483-case
near-peak check and full endpoint/population-history checks at 50/150/250 m/s.
Packed storage reconstructs the full density, including all coherences.

V3 outcome (21 jobs completed): retain exact tail graphs, GPU population
sampling, and FP64 packed Hermitian storage with direct sparse RHS. At
150 m/s and batch 128, full-transit timings were: previous v2 settings
79.263 s; exact tails 70.305 s; GPU sampling alone 78.346 s; both 69.206 s;
both plus packed/direct 59.479/59.459/59.564 s (median 59.479 s). This is
24.96% less runtime than the matched v3 baseline. Packed/staged gave
59.456 s at batch 128 (indistinguishable) but was slower at batch 512,
so retain direct RHS. At batch 512 over 5 us: full 5.680 s,
packed/staged 4.612 s, packed/direct 4.446 s. Live array allocations after
the short run fell from 562.4 to 244.7 MB; these are not total/peak GPU VRAM.

The diagonal integrating-factor prototype passed at 4.464 ns but took
6.798 s for the short test and 79.780 s for the full transit, so reject it
on performance. Steps 8/12/16/24 ns failed the entrywise criterion.
The selected packed method passed the 483-case near-peak check (0.9251%
worst entry error), 1024-trajectory confirmation, and full endpoints at
50/150/250 m/s (0.2560/0.3434/0.3157%). Arbitrary-coherence PSD RHS errors
were below 6e-17 relative to the established full FP64 RHS. Final densities
differ from the v3 baseline by at most 4.1e-16, and population histories
by 1.3e-14; see `output_equivalence.json`. Small batches can be slower with
packed/direct; the speed recommendation concerns large batches. Full-transit
reference coverage remains nine distinct cases per velocity. No new CPU
ODE timings or production changes. See the v3 `REPORT.md` and source snapshots.

Follow-up v2 authorized 2026-10-12: finer Tsit5 steps (3.90625, 4.16667,
4.46429, 5 ns), predetermined synchronized schedules, fixed-step density/
FSAL buffer swapping and constant-coefficient stage kernels. Completed in
`rc_gpu_integrator_results/v2_20261012_133710/`, budget origin 13:37:10 UTC,
hard deadline 16:37:10 UTC including preparation. Single GPU worker; archived
CPU and finer GPU references only. Test full transits with 201 population
samples and both density endpoints. The schedule follows the microwave
envelope; the current GPU fixture's UV amplitude remains on throughout.

V2 outcome: retain base step 250/56 ns = 4.464285714 ns and the gentle
synchronized schedule (1.2x step for |(t-center)/sigma| > 0.8), plus the
combined buffer swapping/constant-stage implementation. Full transit at
150 m/s, batch 128, took 79.642/79.710/79.775 s, median 79.710 s, versus
97.492 s for the original fixed Tsit5 settings (18.24% less runtime).
At the tuned step, swapping and constant stages individually gave no gain;
together they saved about 4% in the short test and 2.4% in the full transit.
They produced bitwise-identical saved densities at the same timestep.

The 483-case near-peak check had worst entry error 0.9251%; full endpoint
checks at 50/150/250 m/s had errors 0.2560/0.3434/0.3157% versus the archived
finer GPU reference, with strict CPU subsets also passing. All errors quoted
are complex-entry relative errors for reference magnitudes >=1e-6. Near-zero
absolute errors, population histories and physical diagnostics are recorded
separately. The full-transit cases remain nine distinct detuning/J-mixture
combinations, not the entire grid. Reject fixed 5 ns (1.709%) and the strong
1.5x schedule (1.0198%). No production backend/configuration change was made.

Status: original suite completed. The user authorized the GPU integrator follow-up
and all five optimization experiments on 2026-10-12; this follow-up completed.
Output: `rc_gpu_integrator_results/v1_20261012_052500/`.
Fixed Tsit5 won on the production fixture at 3.90625 ns, with FP64 split
real/imaginary state, the staged sparse RHS, and 16-step CUDA graphs. At 512
trajectories over 5 us, median evolution was 6.863 s versus RK4's 15.770 s
(2.30x). Shared adaptive Tsit5 took 9.165 s. None of the five optimization
experiments cleared the 3% retention threshold; mixed precision was slower.

Full transit at 150 m/s, batch 128, including 201 population samples and both
density endpoints: Tsit5 97.593 s versus RK4 207.848 s (2.13x). RK4 at
1.953125 ns failed the endpoint target; RK4 at 0.9765625 ns passed. Tsit5
at 3.90625 ns passed at 50, 150 and 250 m/s for nine distinct cases
(-20/0/+20 MHz, each J mixture), compared to archived finer GPU outputs;
the strict CPU subset at zero detuning/J=2 also passed. This is not full-transit
validation of every production scan point. Historical 200 V/cm fields have a
different timestep requirement; Tsit5 at 0.9765625 ns was slower than the
earlier RK4 result there. No new CPU ODE benchmarks were run.
Detailed timings, complex-entry diagnostics, source hashes and saved densities:
`rc_gpu_integrator_results/v1_20261012_052500/REPORT.md`.
Follow-up budget origin 2026-10-12 05:25 UTC; hard deadline 08:25 UTC, including preparation.
Original suite was authorized on 2026-10-11 with a three-hour hard cap.
Budget origin: 2026-10-11 23:55:49 UTC; deadline: 2026-10-12 02:55:49 UTC.
Production RC must continue without changing its sources, backend, recipe or caches.

## What D+S+U actually does

The production CPU mode is `structured_sparse_dense_diag_soa_single_unpack`.
The completed CPU RC comparisons selected it as the fastest forward finalist.

- D: remove the Hamiltonian diagonal from the sparse multiply and apply
  `-i*(H_ii-H_jj)*rho_ij` directly.
- S: use separate real/imaginary scratch arrays for the sparse-Hamiltonian,
  dense-density multiply. The CPU density input itself is still complex.
- U: avoid a redundant conversion/packing of the density on each RHS call.
- Compute `A=H_off*rho` once, then reconstruct the commutator with
  `-i*(A-A_dagger)`, using Hermiticity. Use structured loss and population
  transfer rather than a general collapse-matrix or dense Liouvillian multiply.

Relevant implementation: `rust/src/lindblad/rhs.rs`, particularly
`add_commutator_sparse_dense_upper_soa` and `uses_single_unpack`.
Current evidence: `benchmarks/RC_PROPAGATION_PLAN.md`.

GPU real/imaginary separation is a candidate, not an assumed improvement.
Extra real GEMM/SpMM launches can offset the benefit. A fused kernel with
coalesced accesses is a separate candidate from four library real multiplies.
GPU state can stay expanded and resident across all stages; literal CPU
unpacking is unnecessary. Benchmark packed storage separately only if justified.

## Candidate methods

| Method | Purpose |
|---|---|
| Existing complex dense, two-sided commutator | Reproduce the first screen |
| Complex dense, one-product Hermitian reconstruction | Isolate the benefit of avoiding the second multiply |
| Real/imaginary dense, one-product reconstruction | Isolate splitting from sparsity |
| Complex sparse off-diagonal + direct diagonal | Test sparsity with library SpMM |
| Real/imaginary sparse off-diagonal + direct diagonal | Test the split/library approach |
| Fused custom sparse real/imaginary kernel | GPU adaptation of D+S+U, including structured decay and resident state |

For sparse methods, use the prepared support without numerical thresholding or
discarding small couplings. Test both trajectory-major and trajectory-contiguous
layouts on the fused candidate. Avoid atomics where each thread can own an output
entry. Compare staged `H*rho` plus reconstruction against direct output-entry
accumulation if implementation time permits. Do not assume packing the upper
triangle helps on GPU: it reduces storage but adds indexing/reconstruction work.

CuPy 14.2.0 and CUDA component wheels were installed in the isolated benchmark
environment with user authorization. Custom NVRTC kernels now compile and all
11 candidate variants passed eight derivative checks, including arbitrary
coherences. Triton is unnecessary for this suite. CuPy kernel cache lives in
`benchmarks/.cupy_cache`, avoiding the restricted home-directory cache.
CuPy library cuBLAS/cuSPARSE graph capture is unsupported in this version; those
candidates are explicitly labeled eager. Custom candidates use CUDA graphs.
Both direct fused RHS and staged sparse-product/reconstruction variants are tested,
with trajectory-major/minor and complex-interleaved/separate-array layouts.

## Staged suite, after authorization to start

Use a three-hour hard wall maximum including preparation and compilation. A watchdog must stop only benchmark children, never production.
Checkpoint each measurement. Stop starting jobs whose estimated completion
would exceed the budget. Prune only performance failures within matched tests;
retain accuracy failures and their diagnostics in the report.

1. **RHS checks and screen (roughly 15 minutes).** Compare all implemented
   candidates on identical Hermitian test matrices and physical states at
   representative times/configurations. Include arbitrary coherences, sink
   populations and boundary-adjacent times. Check complex derivatives against
   Rust or a direct NumPy Lindblad calculation before timing. Batch sizes
   512, 1,024, 2,048 and 4,096, as emphasized by the user.
   Warm libraries/compilers separately. Use at least three timed repetitions.
2. **Integrated screen (roughly 20 minutes).** Best three GPU kernels, same
   integrator, time grid and complex128/float64 precision; compare 1 us and
   5 us intervals near the microwave peak. Five microseconds covers the joint
   switching/EOM period. Use multiple detunings and all three J mixtures,
   with a deterministic parameter pattern spanning the actual 161-detuning grid.
   Include the original 12-case pattern for continuity. Compare 128, 512,
   1,024, 2,048 and 4,096 rows. Sweep steps around 2, 1 and 0.5 ns; rank only
   accuracy-qualified results. Check errors per trajectory, not batch-averaged.
3. **Longer finalists (remaining budget).** Best one or two methods, 10–50 us
   first. Attempt selected full-transit cases at 50, 150 and 250 m/s only when
   the measured estimate fits the remaining budget. Reproduce population save
   locations, drive-exit full density and post-decay full density. Use existing
   production checkpoints for agreement checks where available, identifying
   their atol=1e-10 provenance; these are not stricter reference solutions.
   A full-transit accuracy claim needs tighter-reference or timestep-convergence
   evidence, including both endpoint matrices. Stop with a clearly scoped short-
   interval result if that evidence cannot fit the budget.

The known 2 ns fixed-RK4 setting failed the 1% short-interval density-entry gate;
roughly 1 ns passed that particular test. Neither result establishes a timestep
for full-transit production. See the next planned integrator comparison below;
adaptive error control must be per trajectory or conservatively per group.

## Next comparison: GPU Tsit5 versus RK4

Added at the user's request on 2026-10-12, subsequently authorized to run with
the five optimizations below. Prioritize this comparison before extensive RK4-specific optimization.
RK4 was selected for prototype simplicity, not demonstrated superiority to
Tsit5 on GPU. Keep the winning staged sparse D+S+U-style RHS and FP64 precision
fixed so the comparison measures integration strategy.

Compare three GPU integrators:

1. Fixed-step RK4, retaining the measured baseline and failed-accuracy settings.
2. Fixed-step Tsit5 with its actual tableau and valid stage reuse, sweeping
   timesteps independently of RK4. Do not assume its extra stages are offset
   by larger stable, accurate timesteps; measure total RHS calls and wall time.
3. Adaptive Tsit5 with one shared timestep per batch. Compute a normalized
   embedded-error estimate for each trajectory and accept a step only if every
   trajectory passes; use the maximum per-trajectory estimate to control the
   next timestep. Never average errors across trajectories. Start with the
   production rtol=1e-9/atol=1e-10 as a candidate, not an accuracy guarantee.

Clip steps exactly at polarization switches and required output times, retain
absolute EOM phase, and invalidate cached stages across drive discontinuities.
For shared adaptive stepping, record rejected work and whether a few difficult
trajectories limit the batch. Compare grouping by forward velocity and detuning
regime if that limitation is material, including regrouping overhead.

Use existing archived CPU RC fixtures and references first (1, 10 and 200 us,
plus the exact 20-state J=2 batch), then production-field cases with multiple
detunings, J mixtures and forward velocities. Preserve the same initial
matrices, fields, amplitudes, phases, output products and spans within each
comparison. Include batches 128, 512 and 1,024; retain the exact smaller
archived batch for its measured parallel CPU comparison. Large batches must
include heterogeneous trajectories as well as independent nominal replicas.

Rank complete integration times at matched full-density accuracy, using the
existing 1% complex-entry criterion above 1e-6 and reporting near-zero absolute
errors separately. Check both density endpoints and the population history for
production cases. A timestep passing a 200-us final-density check does not
qualify that timestep at earlier times or for a different workload. Report
accepted/rejected steps, RHS calls, per-trajectory error, memory and setup /
capture / transfer costs. Reuse archived CPU timing and reference artifacts;
identify any ideal-thread-scaling estimates separately from measured parallel
CPU timings. Further CPU ODE runs require a demonstrated gap in those artifacts.

Completed evidence:
[original sparse suite](rc_gpu_sparse_results/v1_20261011_235549/REPORT.md) and
[matched archived CPU fixtures](rc_gpu_matched_cpu_results/REPORT.md).

### Five optimization experiments after selecting the integrator

Test separately at fixed numerical settings, retaining only improvements that
pass the density criterion, then measure their combination:

1. Fuse RHS reconstruction with construction of the next integration stage,
   avoiding a separate derivative-read/stage-write kernel. Use distinct input
   and output stage buffers so population-transfer reads remain race free.
2. Reuse the Hamiltonian at repeated stage times: RK4's two midpoint stages,
   or Tsit5's two final-time stages. Respect the frozen switch configuration.
3. Specialize/unroll sparse multiplication for the fixture's exact coupling
   structure. Record kernel register/local-memory attributes and RHS agreement.
4. Test FP32 state/stage storage with FP64 accumulation as an explicit mixed-
   precision experiment; reject it on accuracy failures. Baseline remains FP64.
5. Test larger CUDA graphs (16, 64, 256 steps), including capture cost and
   endpoint padding. No assumed gain from increasing graph or batch size.

User prioritizes fixed synchronized steps; shared-step adaptive Tsit5 remains a
bounded comparison to measure the cost of batch-wide error control and stragglers.
Use one GPU worker at a time. A separate PowerShell monitor may be closed safely.

## CPU comparison and resource accounting

Use native Rust D+S+U with 12 threads as the throughput baseline on the SAME
parameter rows, initial matrices, spans and output products. Also retain a
one-thread reference for numerical checks and CPU work accounting.

Production already uses 12 Rust threads. Simultaneous benchmark CPU timings are
therefore contended, even with low process priority. Keep strict references on
one CPU thread. Permit only brief 12-thread CPU timing samples and record the
concurrent production workload; do not report those as an uncontended CPU/GPU
speedup. Obtain a clean matched-throughput comparison once production finishes,
unless the user explicitly authorizes another resource arrangement. Do not pause
or throttle production implicitly.

Report setup/compilation, transfers, RHS time, complete ODE evolution, output
conversion and total wall time separately. Include trajectories/second at fixed
physical duration and accuracy, GPU memory, accepted/rejected steps where
applicable, and concurrent production progress. Compare GPU and CPU at matched
full-density accuracy, not merely nominal local tolerances or equal timesteps.

Keep Hamiltonian components, state and stages on GPU. Charge preparation and
reconstruction costs; no hidden float32/TF32 or complex64 substitution.
Use CUDA graphs or fused stepping where supported, but check compilation/capture
cost against the actual finite scan workload.

## Accuracy and artifacts

Entrywise target: 1% complex relative error for reference magnitudes >=1e-6,
including coherences. Also report absolute error over all entries and near-zero
entries separately; do not treat the cutoff as an automatic absolute-error pass.
Record trace, Hermiticity, positivity and relative Frobenius diagnostics.
Retain the uncompacted 70-state model and exact switching segmentation.

Read the production context and recipe only. All benchmark code/results belong
under `benchmarks/`, in a new versioned output directory, never in the SPB
production checkpoint tree or simulation store. Save source/context hashes,
software/device versions, parameter tables, timing repetitions, complex reference
and candidate outputs, failed-method diagnostics, progress, and a final report.
No production backend change is authorized by this plan.

GPU memory-layout guidance:
https://docs.nvidia.com/nsight-compute/ComputeTriage/
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
