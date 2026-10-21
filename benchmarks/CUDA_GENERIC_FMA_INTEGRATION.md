# Generic CUDA FP64 FMA integration

The four generic CUDA commutator paths now accumulate complex products with
explicit FP64 `fma`: `direct_rhs`, `component_rhs`, `product`, and `product_tile`.
The source is in `centrex_tlf/lindblad/_cuda_kernels.py`. Both fixed Tsit5 and
CK54 use these defaults, including `rhs_specialization="off"` and generic
fallbacks from automatic specialization. No new solver option is introduced.

The change is exactly the six complex accumulation sites tested by the
[prototype benchmark](cuda_generic_fma_results/REPORT.md). Each complex
multiply-accumulate uses four sequential FMAs, preserving the measured order.
No precision reduction, fast-math flag, support pruning, launch retuning,
stage fusion, physical parameter or timestep change is included. CPU and
specialized CUDA kernels are unchanged.

## Benchmark controls

`bench_cuda_generic_fma.py` and `validate_cuda_generic_fma.py` now run the
installed package without arithmetic patches for their `fma` variant. The
`baseline` variant temporarily uses the independent, frozen pre-FMA source in
`_cuda_generic_baseline.py`. Source assertions verify that the installed kernel
is the expected arithmetic-only transformation, preventing an accidental
FMA-versus-FMA comparison. New artifact directories include the control source
and driver snapshots. Historical prototype artifacts are unchanged.

## Integrated validation

The release-branch CUDA test file passes with **69 passed, 5 skipped**, including
seven specialized-dispatch/option tests in addition to the generic checks. The five skips are
optional PyTorch static CUDA tests: that installation reports CUDA unavailable.
The CuPy CUDA ODE tests run on the RTX 5070 Ti. New tests compare complex density
matrices against independent SciPy Liouvillian exponentiation for all four
generic paths with both solvers, including full/reduced direct and component
paths, coherent initial matrices, varied drive strengths and zero drive.
Existing tests cover multiple shared components, per-trajectory remainders,
output integrals, switching and chunking. Ruff checks pass for the changed
Python files.

The test environment uses a workspace-local `CUPY_CACHE_DIR`; its default
location is not writable under the workspace policy. This is an execution
environment setting, not a package behavior change.

## Integrated paired timings

The new [integrated timing run](cuda_generic_fma_results/20261021T155314751110Z/results.json)
uses the current 26-level cleanup-1 system, the actual final 1,024/748-row
batches at 150 m/s, a 2 us interval crossing switches, 2 ns Tsit5 steps,
256-thread blocks, and six warm alternating pairs per path and batch. These
are solver wall times including setup and output, not isolated kernel timings.
Each percentage is the median of paired reductions; times are separate medians.

| Path | Batch | Original (s) | Integrated FMA (s) | Runtime reduction |
|---|---:|---:|---:|---:|
| Direct RHS | 1,024 | 0.9702 | 0.7645 | 21.19% |
| Shared-component RHS | 1,024 | 0.7497 | 0.5807 | 22.52% |
| Ordinary product | 1,024 | 0.9738 | 0.7910 | 18.84% |
| Tiled product | 1,024 | 0.9032 | 0.7137 | 20.94% |
| Direct RHS | 748 | 0.7446 | 0.5899 | 20.91% |
| Shared-component RHS | 748 | 0.5845 | 0.4610 | 21.00% |
| Ordinary product | 748 | 0.6944 | 0.5605 | 19.33% |
| Tiled product | 748 | 0.6536 | 0.5117 | 21.71% |

Every one of 48 timing pairs favors FMA. Sharing/density-reduction choices match
the prototype: the component path retains 257 pairs, the other paths retain
all 351 pairs. Comparisons within each row isolate the FMA change. The runs
also pass the multi-drive validation for both Tsit5 and CK54. Package source
hashes remain unchanged during the integrated timing run.
The maximum density difference in warmup/timed cleanup solves is **7.77e-16**,
with trace error **3.11e-15**; the multi-drive difference is **2.22e-16**.

The [integrated full-flight confirmation](cuda_generic_fma_validation_results/20261021T155453904132Z/results.json)
passes for both original and installed FMA arithmetic in all four paths:
96 trajectories (eight physical settings and all 12 initial states), four
density snapshots over the complete 400.694 us flight, at the same 2 ns Tsit5
step. The maximum density difference across paths is **4.88e-15**, and maximum
trace error is **3.86e-14**. The installed FMA direct path differs by **3.44e-15**.
Full-flight runs check accuracy; they are not paired performance measurements.

Same-grid agreement does not establish timestep convergence, and these short-interval measurements
do not predict a full production scan's wall time or results on other GPUs.

## Reproduction

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CUPY_CACHE_DIR=Join-Path $PWD 'benchmarks/.cupy_cache'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
./.venv/Scripts/python.exe -m pytest tests/lindblad/test_cuda.py -q -p no:cacheprovider
../CeNTREX-SPB-testing/.venv/Scripts/python.exe benchmarks/bench_cuda_generic_fma.py
./.venv/Scripts/python.exe benchmarks/validate_cuda_generic_fma.py
```

The release branch includes JSON timing/validation summaries and benchmark
drivers. Raw NPZ densities, local source snapshots and GPU profiles are retained
in the original workspace and excluded from the PR.
