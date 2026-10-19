# Solver improvements: evidence and reproduction

This branch contains the package implementation, its self-contained regression
tests, benchmark source, and curated small result/provenance files. Optional
CUDA density reduction and periodic maps remain opt-in. CF4/Magnus remains a
benchmark prototype; no default solver or execution-mode switch selects it.

## Measured speedups and what the baseline means

CPU: Ryzen 7 9800X3D. GPU: RTX 5070 Ti. These are comparisons within each
specified workload, not universal speed guarantees or a single cumulative
before/after benchmark.

| CPU model | Original packed D+S+U median | Current auto median | Ratio of medians |
|---|---:|---:|---:|
| RC | 2.623515 s | 1.669366 s | 1.57x |
| R(2) | 0.263846 s | 0.183728 s | 1.44x |
| R(0) | 0.079634 s | 0.063986 s | 1.24x |
| Q(1) | 0.150753 s | 0.114700 s | 1.31x |

The CPU comparison uses eight randomized interleaved rounds of 10 us EOM
trajectories on the same generic, freshly built dispatch. Hamiltonian assembly
and coefficient-DAG preparation are already shared by both sides. It measures
the newer SIMD/FMA/decay/output kernels, not all gains since the original
expanded-sparse implementation. The default execution mode remains unchanged;
these kernel gains apply when packed D+S+U is selected. See
[CPU integration](rust_specialized_rhs_results/cpu_kernel_integration/REPORT.md)
and [generic preparation](rust_specialized_rhs_results/GENERIC_PREPARED_RHS_REPORT.md).

GPU comparisons:

- A later matched rerun now compares the earlier full-storage GPU Tsit5 with
  the current library on identical 1024-trajectory, 10 us workloads: RC
  25.837 -> 16.569 s (1.56x, reduction enabled), R(2) 14.641 -> 8.138 s
  (1.80x, CK54 without reduction). Same-step default Tsit5 gains are 1.40x
  and 1.41x respectively. All full complex-density checks pass. See the
  [matched before/after report](gpu_cumulative_results/REPORT.md) for scope.
- Older full-transit RC output/storage/graph improvements: 79.263 -> 59.479 s
  (1.33x), at batch 128. [Report](rc_gpu_integrator_results/v3_20261012_135724/REPORT.md).
- Latest exact RC density reduction: 18.503 -> 16.323 s (1.13x) for 1024
  trajectories over 10 us, integration plus state download. The public API's
  1 us end-to-end confirmation, including support construction, saves about 9%.
  R(2) reduction was slower. [Algorithm report](cuda_algorithm_results/REPORT.md).
- R(2) CF4/Magnus prototype: 8.433 -> 6.823 s versus CK54 (1.24x), for 1024
  trajectories over 10 us, with full-density validation. This is **not installed
  as a library solver** in this branch.
- Periodic maps: extremely fast reuse at one fixed periodic parameter key, but
  construction and the number of shared initial states/durations determine the
  gain. The 1024-density test including construction gave about 2x for RC and
  5x for R(2); the small distinct-state batches do not amortize construction.
  See the algorithm report and the public-API validation JSON files alongside it.

Do not multiply these ratios: durations, models, batch sizes and timing scopes
differ. The [historical CPU/GPU comparison](rc_gpu_matched_cpu_results/REPORT.md)
explicitly distinguishes actual measured CPU batches from ideal-thread-scaling
estimates, and identifies cases that fail the later density-accuracy criterion.
There is no single measured ratio from the repository's original CPU defaults
to every currently optimized CPU/GPU path.

## Self-contained package validation

Build/install the project with its Rust extension, then run:

```powershell
python -m pytest tests/lindblad -q
```

CuPy/PyTorch-dependent tests skip without their optional extras/hardware. With
the GPU extras installed, the same tests cover independent complex matrix
exponentials and segmented integration, coherences, all CUDA ODE output modes,
reachability changes, periodicity rejection, entry phases and fractional cycles.
CPU tests cover runtime ISA selection/fallback and packed/split/matrix interfaces.

For newly generated physical fixtures, use `bench_rc_propagation.py` (its `setup`
and `smoke` subcommands are small; baseline runs are explicit) and
`cross_system_fixtures.build_fixture`. See `RC_PROPAGATION_PLAN.md` and
`RC_CROSS_SYSTEM_BENCHMARK_PLAN.md`. These builders reproduce the stated
benchmark recipes, not every historical environment or frozen artifact hash.

## Archived fixture requirements

The experiment drivers are also retained for provenance. Some deliberately load
archived Hamiltonians, strict references, or the sibling `CeNTREX-SPB-testing`
prepared production context rather than rebuild expensive models. Their local
paths in raw JSON are historical provenance, not portable download links.
Large density-history `.npz`/HDF5 files, native binaries, virtual environments,
logs and compiler/test caches are not included in this publication.

Consequently, those archived-data drivers are **not all runnable from a bare
clone**. To repeat them exactly, supply the named artifacts and verify the
recorded hashes/configurations. To run new comparisons, regenerate inputs with
the physical fixture builders above and record a new manifest/reference; do not
claim it is the identical archived test. Production RC fixture generation lives
in the sibling experiment repository; its prepared context is required by
`rc_gpu_sparse_methods.load`. The package regression suite has no dependency on
that sibling cache.

Benchmark-only generated Rust/CUDA sources are retained where small. They are
not compiled or selected by package installation. Failed candidates and accuracy
failures remain marked in the reports; their timings are not qualified speedups.
