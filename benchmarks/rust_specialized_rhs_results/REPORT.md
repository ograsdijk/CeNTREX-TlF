# Rust-only specialized Lindblad RHS benchmark

**Generic follow-up:** [Prepared-data optimizations and combinations](GENERIC_PREPARED_RHS_REPORT.md)
use one compiled binary across systems and reduce the best matched solve times
by approximately 9-11%. These are separate from the system-specific generators below.

**Full generated H RHS + D RHS follow-up:**
[The missing separate-equation experiment](SEPARATELY_GENERATED_RHS_REPORT.md)
is now complete. It replaces the generic commutator and is 5.47 times slower for
RC, while reducing R(0) time by 12.36%. This is distinct from the hybrid below.

Completed2026-10-16 on AMD Ryzen7 9800X3D (8 physical /16 logical cores), **one CPU thread**, BLAS1. Rust1.98.0, release opt-level3, fat LTO, codegen-units1, no fastmath. No Julia benchmark was run.

**Follow-up:** [Separately generated Hamiltonian and dissipator functions](SPLIT_FUNCTIONS_REPORT.md)
improve RC by8.7% and Q(1) by10.4% in paired10 us tests. The tables below
cover the earlier expanded/row prototypes, not that later matrix-style variant.

## Result

The current D+S+U kernel remains fastest for RC, R(2), and Q(1) among these prototypes. Global real packed RHS CSE improves the small R(0) model by approximately11% in trajectory solves. These results do not demonstrate that all possible generated kernels are slower; they test specific generators and layouts.

## Matched 1 us full-density trajectories

Each cell is the median of three solve-only observations, milliseconds. All backends use the exact same current Rust Tsit5 source, real packed state, tolerances, initial density and exact switching segments.

| System | States | D+S+U | Generated rows | Generated packed | Packed + compiled coefficients | Packed + global CSE |
|---|---:|---:|---:|---:|---:|---:|
| RC | 70 | 609.400 | 923.966 | 5839.521 | 5833.574 | 4991.860 |
| R2 | 41 | 48.269 | 54.029 | 119.432 | 116.463 | 125.804 |
| R0 | 18 | 17.973 | 21.743 | 20.819 | 19.904 | 15.980 |
| Q1 | 26 | 32.633 | 38.708 | 48.488 | 47.866 | 48.396 |

D+S+U was retimed in the CSE experiment; its numbers above come from that paired experiment. The earlier native expanded_sparse baseline was also tested; per-case JSON files retain all observations. Identical RHS counts across methods within a model support attributing these solve differences to kernel cost rather than adaptive step selection. These short times must not be extrapolated linearly to200 us: adaptive step counts and transients evolve.

## Direct RHS timings

Median of five native-loop samples,200 repetitions of four EOM times per sample; no Python dispatch or allocation per call. Times in microseconds. Correctness was checked at arbitrary Hermitian positive density states and the initial state, all four RC polarization configurations and four EOM times. Timed loops use one fixed gate configuration; trajectories exercise all four.

| System | D+S+U | Packed global CSE | CSE relative to D+S+U |
|---|---:|---:|---:|
| RC | 32.411 | 263.744 | 8.138x runtime |
| R2 | 6.069 | 16.297 | 2.685x runtime |
| R0 | 1.026 | 0.910 | 0.887x runtime |
| Q1 | 2.477 | 3.899 | 1.574x runtime |

## Longer trajectories and accuracy

The screened10 us runs confirm the R(0) CSE improvement: **86.813 ms D+S+U versus77.366 ms CSE**, an approximately10.9% wall-time reduction. R(2) and Q(1) longer tests retained D+S+U and the competitive row prototype; D+S+U remained faster. Candidates whose RHS was more than1.5x baseline were screened out of10 us tests after completed1 us accuracy/timing measurements. RC has completed1 us measurements for every prototype; its initial long-stage benchmark was stopped after those results to avoid spending time on clearly slower kernels. No200 us or parallel-throughput benchmark was performed.

Candidate trajectory tolerances: rtol1e-9, atol1e-11. Strict same-solver baseline references: rtol1e-11, atol1e-13. All saved trajectory tests pass1% entrywise complex-density relative error where |rho_ref|>=1e-6, absolute1e-6 below. Typical observed relative differences are much smaller. All saved trace checks pass1e-9. Generated RHS normalized max differences from D+S+U are approximately machine precision. Full density, including coherences, is evolved and checked; no clipping or renormalization.

## Models and exact matching

These are numerical benchmark fixtures, **not production apparatus parameters**. E=[0,0,200] V/cm, B=[0,0,0.01] G for each model, using the freshly built current molecular constants. RC is the existing uncompacted70-state P(2) F1prime3/2 Fprime1 + J1↔2 + J2↔3 fixture with independent switched MW polarizations,1 MHz each and quarter-period offset. The pure initial state is X J2 F1=5/2 F2 mF0. Every trajectory begins at125 ns. Switches are frozen to midpoint values on exact250 ns segments for **every** method; EOM uses absolute/global time. Segment restarts are included equally in native solve times.

Optical systems: R(2) F1prime7/2 Fprime4 (41 states), R(0) F1prime3/2 Fprime2 (18 states), Q(1) F1prime3/2 Fprime2 (26 states). These optical fixtures use qn_compact=True, retain opposite excited parity and include a coherent1.6 MHz EOM with beta3.8. RC uses qn_compact=False. Optical initial populations are uniform over modeled X states. The smaller fixtures are not interchangeable with uncompacted production systems; compare methods within each row.

## Prototypes and limits

- Generated rows: bake in the numerical coupling graph, offsets and coefficients, retain contiguous vector row loops, fuse only nonzero damping/transfers. This prototype still processes zero-valued inactive edges and is not four-configuration specialization.
- Generated packed: emit fixed packed-coordinate derivative expressions in32-output tiles, retaining generic coefficient evaluation.
- Compiled coefficients: compile the waveform/parameter DAG into Rust rather than interpreting it; scan values remain runtime parameters. The benchmark emitter only supports the scalar expressions needed by these fixtures.
- Global CSE: build real packed derivative equations, apply SymPy CSE across all outputs, generate Rust with shared temporary arrays and fixed-size state references. Temporaries are evaluated in128-entry tiles and outputs in64-entry tiles. This is full equation CSE but not a monolithic function or the older Julia code layout.

Lower scalar operation count did not imply a faster kernel. Code footprint, temporary traffic, indirect packed-state access, loss of vector-friendly loops and inactive-edge work are plausible explanations; no hardware-counter profiling was done to distinguish them. The results do not rule out four-configuration specialization, different tiling/CSE schedules, SIMD generators or smaller selective specializations. No public solver/backend/default changed.

## Generation and compilation costs

| System | Global CSE generation (s) | Rust compile (s) | Temporaries | Scalar operation count before / after |
|---|---:|---:|---:|---:|
| RC | 198.19 | 90.40 | 4620 | 1056520 / 787745 |
| R2 | 16.80 | 11.67 | 865 | 86087 / 66262 |
| R0 | 1.60 | 5.71 | 135 | 8150 / 6437 |
| Q1 | 5.38 | 7.44 | 241 | 30098 / 23006 |

R(0) global CSE costs about7.3 s to generate and compile, versus9.45 ms saved per10 us trajectory: roughly770 such trajectories to amortize that measured setup cost. Compilation observations include build-system work and may reuse previously built dependencies. Native artifact caching is possible but is not a general library feature here.

## Reproduction and artifacts

From the library repository root, with the existing Cargo/Python environment and one BLAS thread:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.venv/Scripts/python.exe benchmarks/bench_rust_specialized_rhs.py
.venv/Scripts/python.exe benchmarks/bench_rust_cse_rhs.py
```

The first command compiles five isolated native variants and runs the screen. The second adds global real CSE. Both reuse Rust numerical source via module includes rather than copying the solver. The installed extension is not overwritten. Generation and binaries live in ignored `.cache/benchmarks/rust_specialized_rhs_v1/`; wrapper source is [specialized_rhs_native.rs](../specialized_rhs_native.rs).

Small results in this directory belong in Git: `rc.json`, `r2.json`, `r0.json`, `q1.json` and the corresponding `_cse.json` files. They retain timing repeats, error checks, accepted/rejected steps, RHS calls and generated-source hashes. [environment.json](environment.json) identifies source versions and hardware. Generated artifacts are not included in Git and must be rebuilt.

Recommended next step: keep D+S+U for RC/R(2)/Q(1); consider the R(0) result a small optional specialization candidate, pending broader fields/powers/detunings and longer-duration tests. Do not replace the generic solver based on these fixtures alone.
