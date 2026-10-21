# Explicit FP64 FMA in the generic CUDA kernels

Benchmark-only experiment on an NVIDIA GeForce RTX 5070 Ti, 2026-10-21 UTC,
using CuPy 14.2.0 and NumPy 2.2.6 for timings.
Package implementation and defaults are unchanged.

## Result

Explicit FMA reduced matched solver wall time in all **48 of 48 pairs**.
Six warm pairs were measured per path and batch, with three in each order,
shuffled using a fixed seed. Each percentage is the median of paired runtime
reductions; the two times are separate medians.

| Generic path | Batch | Baseline (s) | FMA (s) | Runtime reduction |
|---|---:|---:|---:|---:|
| Direct RHS | 1,024 | 0.9684 | 0.7649 | 21.13% |
| Shared-component RHS | 1,024 | 0.7526 | 0.5834 | 22.52% |
| Ordinary sparse product | 1,024 | 0.9739 | 0.7902 | 18.88% |
| Tiled sparse product | 1,024 | 0.9038 | 0.7138 | 21.09% |
| Direct RHS | 748 | 0.7427 | 0.5904 | 20.44% |
| Shared-component RHS | 748 | 0.5856 | 0.4588 | 21.70% |
| Ordinary sparse product | 748 | 0.6947 | 0.5618 | 19.17% |
| Tiled sparse product | 748 | 0.6555 | 0.5117 | 21.92% |

The worst individual pair still reduced time by 18.73%.
These results support integrating explicit FMA into all four generic paths.
They do not establish performance on other GPUs or every model.

## Workload and controls

The current sibling `CeNTREX-SPB-testing` cleanup-1 engine builds a 26-level
system. Inputs are the actual last full 1,024-row batch and final 748-row batch
of its scan at 150 m/s. The previous specialized benchmark's 548-row final batch
is not the current grid's final batch.

Each solve evolves a 2 us interval centered on the middle pass-boundary
Doppler switch, also crossing polarization switches, with the existing 2 ns
Tsit5 timestep. Full density output is returned. All tests force
`rhs_specialization="off"`, use 256-thread blocks, 16-step graphs, and the
same coefficients, matrix support, physical settings, and output within each
baseline/FMA pair. Timing includes setup, graph capture, evolution and output;
CUDA compilation is warmed before measured pairs. Timings are short-interval
solver wall times, not isolated derivative-kernel timings or full-flight times.

Direct/product/tiled disable Hamiltonian sharing and use all 351 upper density
pairs. The shared-component path enables sharing and exact density reduction,
retaining 257 pairs. The product path sets `tile_size=0`; tiled uses 32.
Comparisons between different rows therefore include these structural choices;
the FMA gain is measured within each row.

Only complex-product accumulation is replaced by four explicit double-precision
`fma` calls, on both sides of the commutator. No fast math, FP32 conversion,
term pruning, stage fusion or timestep change is introduced. The benchmark
patches source strings in memory and restores them after each solve.

## Numerical checks

Every warmup and timed solve is checked against the same generic direct density
reference with `rtol=1e-10`, `atol=1e-12`. The maximum density difference is
**7.77e-16**, and maximum trace error is **3.11e-15**.

Additional three-level validation covers two independent shared drive components,
complex edges, a continuously varying diagonal, coherent initial states, a zero
drive trajectory and odd graph tails. All four paths pass with both Tsit5 and
CK54; the largest density difference across these checks is **2.22e-16**.

Full-flight confirmation covers **96 trajectories**, comprising eight physical
settings and all 12 initial states, over the current engine's complete
**400.694 us** flight at 150 m/s. Both baseline and FMA pass for all four paths,
with four density snapshots per trajectory. Across all paths the maximum
density difference from the generic direct baseline is **4.88e-15**, and maximum
trace error is **3.86e-14**. The FMA direct difference is **3.44e-15**. These are
validation runs, not warmed paired full-flight performance measurements.
These checks establish agreement on the same grid, not convergence of the 2 ns
timestep. No new package-level CUDA execution defaults have been promoted.

## Artifacts and reproduction

- [Complete timings, paired ranges, diagnostics and source hashes](20261021T152051773324Z/results.json)
- [Timing driver](../bench_cuda_generic_fma.py)
- [Full-flight validation driver](../validate_cuda_generic_fma.py)
- [Complete full-flight checks and density snapshots](../cuda_generic_fma_validation_results/20261021T152251757351Z/results.json)

The timestamped timing directory also contains input matrices, parameter rows,
selection indices, warmup density outputs, generic source snippets and a driver
snapshot. Cleanup engine/scan source snapshots and their hashes were also saved
after the timing run. The source snippets include a representative 29-level tiled expansion;
the actual benchmark uses the engine's 26 levels. Six CUDA-related package source
files were hashed before/after timing and all are unchanged. The initial attempt
using the repository environment passed the small-model checks but could not
import the scan runner because `rich` was absent; its failed record is retained
in `20261021T152014946196Z`. The completed timing run uses the sibling scan
environment, which imports the editable package from this repository.

```powershell
$env:PYTHONIOENCODING='utf-8'
../CeNTREX-SPB-testing/.venv/Scripts/python.exe benchmarks/bench_cuda_generic_fma.py
./.venv/Scripts/python.exe benchmarks/validate_cuda_generic_fma.py
```

The release PR includes the linked successful JSON summaries and current drivers.
Raw density/input NPZ files, snapshots and the failed preliminary-run directory
remain local artifacts and are not part of the release commit.
