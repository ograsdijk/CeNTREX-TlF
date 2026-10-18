# Further CUDA experiments

GPU: NVIDIA RTX 5070 Ti. Benchmark-only prototypes; production solver unchanged.

Tests cover precomputed packed-density gather indices, a shared-column tiled direct RHS, FP32 product/RHS arithmetic with FP64 state and RK updates, trajectory-local persistent Tsit5, and fixed-shape buffer/graph reuse across changed detunings.

The screen uses 512 trajectories and four randomized interleaved rounds; confirmation uses 1,024 and six rounds. RC spans 146-147 us near the field center with polarization switching, dt=4.4642857 ns. R(2) includes an EOM, spans 0.125-1.125 us, dt=2 ns. Archived distinct initial-state/detuning cases are repeated to fill batches. No CPU ODEs rerun.

Evolution timings exclude setup/capture/precomputation and include final internal-density download. Every candidate runs through the public solver for archived-reference checks, and timed graph replays are checked again. State/stage storage remains FP64 even for FP32 arithmetic. The hybrid tiles one density column in shared memory, evaluates both sides of the commutator, applies decay and writes packed output directly.

## Large-batch confirmation

Positive percentages indicate reduced runtime vs the same RHS baseline. Parentheses give median seconds. Gather variants use their corresponding tiled/direct baseline; hybrid and FP32 variants use direct. Neither comparison changes dt or physics.

| System | Method | Median seconds | Paired runtime reduction | Passed density checks |
|---|---|---:|---:|---|
| RC | base_tiled | 1.914056 | +0.00% | True |
| RC | base_direct | 1.927742 | +0.00% | True |
| RC | gather_tiled | 1.935316 | -0.81% | True |
| RC | gather_direct | 1.967815 | -0.94% | True |
| RC | hybrid32 | 2.025528 | -5.53% | True |
| RC | gather_hybrid32 | 2.081382 | -7.97% | True |
| RC | fp32_product_direct | 2.294871 | -18.93% | True |
| R2_EOM | base_tiled | 1.104692 | +0.00% | True |
| R2_EOM | base_direct | 1.223610 | +0.00% | True |
| R2_EOM | gather_tiled | 1.133497 | -3.86% | True |
| R2_EOM | gather_direct | 1.241664 | +0.80% | True |
| R2_EOM | hybrid32 | 1.216926 | +1.84% | True |
| R2_EOM | gather_hybrid32 | 1.226914 | +0.75% | True |
| R2_EOM | fp32_product_direct | 1.477384 | -19.99% | True |

### Precomputed indices plus conjugation signs

A separate four-round 1,024-trajectory check precomputes both packed indices and conjugation signs. Comparisons are paired against the corresponding RHS.

| System | Method | Median seconds | Paired runtime reduction | Passed |
|---|---|---:|---:|---|
| RC | gather_sign_tiled | 1.905523 | -1.36% | True |
| RC | gather_sign_direct | 2.091317 | -10.88% | True |
| R2_EOM | gather_sign_tiled | 1.108687 | -2.72% | True |
| R2_EOM | gather_sign_direct | 1.326592 | -11.09% | True |

Full raw rounds and diagnostics: [screen.json](screen.json), [confirm1024.json](confirm1024.json), [gather_sign1024.json](gather_sign1024.json). The original 512-screen fp32_product_direct label used the same full-RHS lowering as fp32_rhs_direct; do not treat that duplicate as an independent product-only result. The product-only/direct lowering was separated before 1,024-trajectory confirmation. Generated CUDA sources are retained beside these JSON files.

## Persistent integration

One block owns one trajectory, keeping the base/current packed density and Hamiltonian in shared memory, using global FP64 RK stage arrays, block barriers between stages, FSAL across steps/chunks, and the original clipped time grid. Launches advance only 2 or 16 steps, avoiding an unbounded long-lived kernel. Full density output only; photon integration was not added to this rejected prototype.

| System | Batch | Product arithmetic | Baseline direct (s) | Best persistent (s) | Shared memory/block |
|---|---:|---|---:|---:|---:|
| R2_EOM | 1024 | fp32_products | 1.195492 | 2.253879 | 32016 bytes |
| R2_EOM | 1024 | fp64 | 1.195883 | 1.880224 | 32016 bytes |
| RC | 512 | fp64 | 0.931357 | 1.675422 | 91328 bytes |
| RC | 1024 | fp32_products | 1.913350 | 3.742653 | 91328 bytes |
| RC | 1024 | fp64 | 1.884992 | 3.119386 | 91328 bytes |

Persistent comparisons use three warm repetitions per launch configuration. Kernel attributes and full complex errors are in the per-system JSON files. Large shared-memory allocations constrain occupancy; attributing the slowdown to a specific hardware bottleneck would require profiler counters.

## Buffer/graph reuse

Reuses buffers, captured graphs, static model and unchanged initial states for fixed-shape/support/varying-pattern/time-grid batches. Every call replaces scalar parameters, resets state/clock/error and regenerates common waveform tables. Detuning offsets +0.125, -0.25 and +0.5 MHz are compared with fresh public calls. Fresh-packed calls use prepacked initial states to separate packing from reuse. Three interleaved rounds per detuning setting; values agree to 1e-12 absolute and 1e-11 relative. Timings include parameter uploads and final public-packed density output. These are short-call wall times, not full-transit speedups.

| System | Batch | Duration (us) | Fresh matrix input (s) | Fresh packed input (s) | Reused (s) |
|---|---:|---:|---:|---:|---:|
| RC | 1024 | 1 | 3.679460 | 1.981616 | 1.924187 |
| R2_EOM | 1024 | 1 | 1.660672 | 1.130030 | 1.104302 |
| RC | 1024 | 10 | not measured | 18.886601 | 18.801684 |
| R2_EOM | 1024 | 10 | not measured | 11.135219 | 11.286528 |
| RC | 512 | 1 | 1.774049 | not measured | 0.884168 |
| R2_EOM | 512 | 1 | 0.867280 | not measured | 0.570046 |

### RC and R(2), 10 us reuse comparison

1,024 trajectories, identical prepacked inputs, current default tiled RHS, fixed Tsit5. The original short-fixture start times and dt are retained, extending the endpoint to 10 us after entry. Three changed-detuning batches (+0.125/-0.25/+0.5 MHz offsets), two randomized interleaved rounds each: six fresh/reused pairs per system. Full packed densities are compared at each offset, atol=1e-12 and rtol=1e-11. This isolates setup reuse; it is not a new timestep-accuracy or full-transit benchmark.

| System | Fresh packed median (s) | Reused median (s) | Paired median runtime reduction | Descriptive 95% interval | Max packed difference |
|---|---:|---:|---:|---:|---:|
| RC | 18.886601 | 18.801684 | +0.65% | [-4.42, +1.44]% | 0 |
| R2_EOM | 11.135219 | 11.286528 | -2.26% | [-3.50, +0.98]% | 0 |

Positive percentages mean less runtime. Intervals resample the six paired rounds and describe this small sample on this GPU. Raw data: [session_reuse_b1024_10us.json](session_reuse_b1024_10us.json). Reproduce with `benchmarks/check_cuda_session_reuse.py --batch 1024 --duration-us 10 --repeats 2 --methods fresh_packed reused` in the benchmark GPU environment.

The RC/R(2) matrix-input timings above predate vectorized packing and CUDA grid prepacking in the library. That CPU overhead is now addressed by `layout.pack_batch` and by packing distinct CUDA-grid initial states before parameter expansion. Explicit packed snapshots remain reusable across calls.

## R(0) reuse after the packing update

18-level R(0) F'=2 fixture, E=200 V/cm, B=0.01 G, optical Rabi rate Gamma. Uniform J=0 population and a coherent J=0 initial state; three detunings (-20/0/+20 MHz), repeated to fill 1,024 trajectories. EOM off (beta=0) or beta=3.8 at 1.6 MHz. dt=1 ns; 1 us and 10 us intervals beginning at 125 ns. Large rotational offsets of uncoupled compact spectators are excluded from the initial step-size estimate. Six randomized interleaved rounds per case, with changed detunings each round. This checks reuse equality, not independent R(0) step convergence. All reused packed entries match fresh solves exactly.

| EOM beta | Duration (us) | Fresh packed (s) | Reused (s) | Per-round runtime reduction range |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0.225000 | 0.208515 | +5.23% to +9.70% |
| 0 | 10 | 2.170331 | 2.194850 | -3.13% to +3.62% |
| 3.8 | 1 | 0.224007 | 0.211654 | +1.58% to +8.09% |
| 3.8 | 10 | 2.178272 | 2.165113 | -1.42% to +3.63% |

Reuse saves roughly 6-7% in the short R(0) calls; longer calls are effectively tied and include slower rounds. Reuse is not uniformly faster, including within a single system. Raw samples: [r0_reuse.json](r0_reuse.json). Reproduce with `benchmarks/check_r0_cuda_reuse.py` in the benchmark GPU environment.

## Integrated packing

`PackedHermitianLayout.pack_batch` packs matrix batches with bounded NumPy indexing. Batch solvers, dense preparation and matrix-snapshot conversion use it automatically; CUDA grids validate/pack the distinct initial matrix batch before replicating packed states over parameters. Repeated external calls can reuse an explicitly packed owned snapshot. No identity-based cache of mutable input arrays or graph-reuse API was introduced.

1,024 random complex Hermitian matrices at each representative model size; one scalar reference timing, three vectorized timings, exact equality of every packed coordinate. Timings below measure packing only, not ODE evolution.

| Size | States | Scalar pack (s) | Vectorized median (s) |
|---|---:|---:|---:|
| RC | 70 | 1.783690 | 0.045024 |
| R2 | 40 | 0.906483 | 0.010576 |
| R0 | 18 | 0.178153 | 0.001382 |

Raw data: [batch_packing_integration.json](batch_packing_integration.json); reproduce with `benchmarks/validate_batch_packing.py`. Tests verify scalar equivalence, complex signs, strided/empty inputs, chunk boundaries and owned snapshots; CUDA grid tests verify packing once before parameter expansion and validation of matrix inputs before conversion.

## Full-transit mixed-arithmetic accuracy

150 m/s; light-field exit and after 25 excited-state lifetimes with all optical/MW drives disabled. RC: nine distinct detuning/J-mixture cases. R(2): 60 distinct detuning/initial-state cases with an EOM. References are the archived finer-step full trajectories. This does not validate an entire apparatus grid.

| System | Variant | Worst relative entry/coherence error | Trace error | Passed |
|---|---|---:|---:|---|
| RC | base_tiled | 0.20051% | 1.37e-14 | True |
| RC | fp32_product_tiled | 0.29771% | 2.83e-07 | True |
| RC | fp32_product_direct | 0.24727% | 3.09e-07 | True |
| RC | fp32_rhs_direct | 0.24055% | 3.31e-07 | True |
| R2_EOM | base_tiled | 2.3401e-05% | 2.25e-14 | True |
| R2_EOM | fp32_product_tiled | 0.015335% | 3.23e-08 | True |
| R2_EOM | fp32_product_direct | 0.012412% | 3.21e-08 | True |
| R2_EOM | fp32_rhs_direct | 0.01097% | 9.72e-08 | True |

Acceptance: each complex entry with reference magnitude >=1e-6 must have <=1% relative error; smaller entries <=1e-6 absolute. Trace error <1e-6; minimum density eigenvalue >=-1e-7. Complete near-zero and positivity diagnostics are retained in [mixed_full.json](mixed_full.json). No tolerances or state-space pruning were changed to pass a candidate.

## Findings

None of these kernel prototypes improves evolution consistently over the current default on the tested RC/R(2) systems. Early 512-trajectory gather gains in RC did not survive 1,024-trajectory confirmation; adding precomputed conjugation signs also loses. The tiled-direct hybrid, including its gather combination, does not beat the best existing RHS consistently. Mixed sparse/RHS arithmetic costs roughly 20% more runtime at large batch sizes despite passing the stated full-transit accuracy checks at 150 m/s. Persistent FP64 integration is about 1.6 times the baseline runtime at 1,024 trajectories; its FP32-product combination is about 1.9-2.0 times baseline. Keep production evolution kernels unchanged.

For repeated 1 us calls at 1,024 trajectories, graph/buffer reuse reduces wall time by approximately 2.9% (RC) and 2.3% (R2) compared with fresh calls using packed initial inputs. Most apparent setup saving against matrix inputs comes from avoiding repeated CPU packing: the already-supported packed input reduces these short fresh-call wall times by approximately 46% and 32%, respectively. That existing API choice should be used before adding a reusable-session API. These percentages do not predict full-transit speedups, where evolution dominates. The subsequent 10 us RC/R(2) comparison finds no reliable reuse gain either: both descriptive paired intervals include zero, and R(2) has a slower reused median. All endpoint packed densities match exactly. This supports leaving graph/buffer reuse out of the production API for these workloads. Any session implementation would need explicit invalidation for changed shapes, support/varying patterns, time grids and discontinuities; this prototype only reuses compatible detuning batches.

## Reproduction

Use `benchmarks/.venv-gpu-bench/Scripts/python.exe` for:

- `benchmarks/check_cuda_remaining.py` (512 screen); add `--batch 1024 --rounds 6 --name confirm1024 --variants base_tiled base_direct gather_tiled gather_direct hybrid32 gather_hybrid32 fp32_product_direct` for confirmation.
- `benchmarks/check_cuda_persistent.py` (use `--threads 256 --chunks 16 --batch 1024 --rounds 3` for large-batch confirmation; add `--arithmetic fp32_products` for the combination).
- `benchmarks/check_cuda_session_reuse.py --batch 1024`.
- `benchmarks/check_cuda_mixed_full.py`.
- `benchmarks/report_cuda_remaining.py` regenerates this report.
