# Separate generated Hamiltonian and dissipator: Rust follow-up

**Scope clarification:** this report concerns generated Hamiltonian **assembly**
plus the generic D+S+U commutator. The requested independently generated full
Hamiltonian RHS and dissipator are now tested in the
[separate-equation report](SEPARATELY_GENERATED_RHS_REPORT.md); that candidate is
5.47 times slower for RC. The improvement below belongs only to this hybrid.

Completed2026-10-16. This follow-up tests the separately generated matrix path discussed after the initial expanded-RHS benchmark. It **improves RC performance**, unlike the earlier fully expanded CSE prototype.

## Actual separation

The RHS calls these independent components:

1. A generated scalar coefficient function (runtime drive values and true EOM time retained).
2. `generated_hamiltonian`: fill the Hermitian upper triangle from its static array and generated dynamic entries, with CSE applied to those entries.
3. The existing **unmodified D+S+U SoA commutator**, using the generated H.
4. `generated_dissipator`: separately generated nonzero population-transfer and coherence-damping expressions, with its own independent CSE; adds D(rho) to the Hamiltonian derivative.

The dissipator reads the original input density, not the Hamiltonian derivative. All density coordinates are evolved. The Hamiltonian is not expanded into commutator equations in this variant; it is generated as a matrix, as in the Julia matrix-style path. This is therefore a different experiment from the earlier global CSE of the complete derivative.

## Confirmed 10 us trajectory timings

Five paired observations per method, alternating execution order after warmup. Medians below; one core, same Ryzen7 9800X3D and current Rust Tsit5 source as the first report.

| System | States | Baseline D+S+U (s) | Separate generated H/D (s) | Wall-time reduction |
|---|---:|---:|---:|---:|
| RC | 70 | 3.251019 | 2.968411 | 8.69% |
| R2 | 41 | 0.280135 | 0.269252 | 3.88% |
| R0 | 18 | 0.090352 | 0.086865 | 3.86% |
| Q1 | 26 | 0.169303 | 0.151649 | 10.43% |

RC and Q(1) show clear approximately9–10% improvements in this sample. R(2) is a smaller approximately4% improvement. R(0) is noisy: separate-path observations range0.081–0.113 s, so its3.9% median improvement should not be treated as a firm result. The earlier packed global-CSE R(0) path is a different candidate and gave approximately11% in its own paired-method experiment. No statistical confidence interval or long-duration production benchmark is claimed here.

Longer native-loop RHS samples (15 observations per method,2000 repetitions x4 EOM times each) reduce the very-short timing noise seen in the initial split screen:

| System | D+S+U RHS (us) | Separate H/D RHS (us) |
|---|---:|---:|
| RC | 32.471 | 30.015 |
| R2 | 6.120 | 5.853 |
| R0 | 1.070 | 0.954 |
| Q1 | 2.583 | 2.292 |

## Correctness and model scope

All variants pass the same entrywise full-density checks against strict references at1 us and10 us:1% relative above magnitude1e-6, absolute1e-6 below. The same accepted/rejected steps and RHS counts were observed within each matched trajectory comparison. Paired-path endpoint packed differences are2.46e-14 for RC and at most1.95e-15 in the optical cases. Initial random-Hermitian and pure-state RHS tests span four EOM times and all four RC switch configurations. Trace checks pass1e-9. No clipping, renormalization or change to the equations was used.

The fixtures are the same numerical systems as [the initial report](REPORT.md): E=[0,0,200] V/cm, B=[0,0,0.01] G,1.6 MHz EOM beta3.8, RC70 states uncompacted; R(2)41, R(0)18, Q(1)26 states with spectator compaction. They are numerical performance fixtures, not the current or old-paper apparatus simulation recipes. All methods use identical250 ns RC switching segments, gates frozen at each midpoint and EOM evaluated at absolute time. Solve timing excludes model preparation and compilation, includes the equal segment restarts, and is single-core. No Julia code was run.

## Generated functions and setup costs

| System | H CSE temporaries | D CSE temporaries | Dynamic H entries | Active dissipator packed entries | Generation (s) | Compile (s) |
|---|---:|---:|---:|---:|---:|---:|
| RC | 6 | 0 | 448 | 858 | 1.008 | 7.776 |
| R2 | 2 | 0 | 100 | 1175 | 0.555 | 8.623 |
| R0 | 2 | 0 | 20 | 268 | 0.110 | 5.593 |
| Q1 | 2 | 0 | 56 | 436 | 0.214 | 6.023 |

Zero dissipator CSE temporaries means the pass found no shared nontrivial expressions for that generated numeric decay model, not that the dissipator was omitted. Nonzero damping and transfers are emitted directly; ground-ground zero damping operations are omitted. The static H array is reused and dynamic assembly has fixed indices. Runtime coefficient evaluation is compiled for the expressions used in these fixtures; the emitter is not a general library backend.

For RC the measured setup is approximately8.8 s, versus0.283 s saved per10 us solve in the paired confirmation: around31 such trajectories to amortize that setup observation. This is not an extrapolation to full-length trajectories. Compilation may reuse existing Cargo dependencies.

## What the result establishes

Separately generating H assembly and decay while retaining a vector-friendly sparse commutator is more promising for RC than expanding the entire master equation into straight-line scalar expressions. This comparison changes assembly, static-index specialization, nonzero decay emission and coefficient evaluation together; no isolated component ablation or hardware-counter profile was run, so individual contributions are not established. A separately CSE-generated expanded Hamiltonian commutator was not measured in this follow-up.

The implementation is benchmark-only and does not change installed binaries, defaults or production results. Before library integration, test longer trajectories, other fields/powers/detunings and waveforms, and cached generation/loading. Preserve the generic fallback and prepare-once runtime scans.

## Reproduction and persistent artifacts

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.venv/Scripts/python.exe benchmarks/bench_rust_split_rhs.py
.venv/Scripts/python.exe benchmarks/confirm_rust_split_rhs.py
```

The first command generates/compiles isolated Rust benchmark extensions and checks all four systems. The second reuses those binaries for paired confirmation. The adapter includes an exact snapshot of the current generic RHS source plus a benchmark-only helper that calls its private commutator functions; it does not modify the production source.

Persistent small data: `rc_split.json`, `r2_split.json`, `r0_split.json`, `q1_split.json` and corresponding `_split_paired.json` files contain repeats, step statistics, accuracy and source hashes. Large generated source/binaries stay under ignored `.cache/benchmarks/rust_specialized_rhs_v1/split_*/`. This report and the JSON files are outside cache for Git.
