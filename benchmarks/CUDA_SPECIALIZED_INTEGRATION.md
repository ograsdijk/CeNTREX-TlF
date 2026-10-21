# Integration of the benchmarked CUDA FP64 FMA kernel

The prepared-sparsity single-component FMA implementation is now part of
`centrex_tlf.lindblad.cuda`, with code generation in `_cuda_specialized.py`.
This promotes only the completed benchmark improvements:

- Specialized commutator and structured decay, with explicit FP64 FMA accumulation.
- RHS/next-stage Tsit5 fusion, preserving the original stage coefficients and FSAL resets.
- Diagonal reuse only after proving stage invariance; stage zero retains its
  FSAL behavior and stage one always performs fresh diagonal assembly.
- The measured full/partial launch choices, with an explicit override.

No density repacking, imaginary-diagonal elimination, active-polarization
sparsity, envelope factorization, precision reduction or numerical-step change
is included. The CPU implementation is unchanged. Package code has no imports
from benchmark modules or from the SPB repository.

## Selection and diagnostics

`CudaSolverOptions(rhs_specialization="auto")` is the CUDA default. It enables
specialization for direct single-shared-component Tsit5 chunks without an
integral state, with at least 512 trajectories and 64 prescribed grid steps.
The diagonal must be provably invariant at the compiler's within-step switch
time. Multiple components, per-trajectory H remainders and continuous diagonal
time dependence retain the generic kernel. Source generation is bounded to
32 levels, 528 density pairs, 256 edges, 6,000 complex terms and 1 MB.
These bounds are conservative implementation limits, not physical cutoffs.

`rhs_specialization="off"` selects the generic implementation, which now also
uses explicit FP64 FMA accumulation. It disables specialization, not FMA.
`"on"` skips the batch/step heuristic and rejects structurally unsupported or
oversized cases. Auto records CUDA source compilation failures and falls back;
explicit on propagates the compiler error. CuPy manages binary caching.

Specialized chunks use 128 threads when full and 64 when partial unless
`specialized_block_size` is explicitly supplied. These are workload-derived
heuristics; only the 1,024/548 cleanup shapes were tuned. The generic path
continues to honor `block_size`. Actual launch size and specialization details
appear in chunk solver statistics with `collect_stats=True`.

The current cleanup-1 caller already supplies direct RHS, exact density
reduction, Tsit5 and 1,024-row chunks, so eligible full and 548-row final
chunks select the integrated path without a caller change. Smaller chunks can
use the generic fallback. Initial states, both excited parities, beam geometry,
fields, power normalization, switches, outputs and timestep are unchanged.

## Evidence and limits

The integrated arithmetic and scheduling are ported from:

- `_cuda_sparse_specialization.py`: single-component prepared-support generator.
- `_cuda_fma_accumulation.py`: explicit FMA complex accumulation.
- `_cuda_stage_reuse_prototype.py`: invariant-diagonal proof and fused Tsit5 loop.

Prior prototype evidence:

- [Full-flight/dark-tail density checks and paired FMA timings](cuda_fma_results/REPORT.md):
  maximum density difference 3.55e-15; approximately 30–31% less runtime than
  the tuned specialized non-FMA prototype.
- [Block retuning and FMA hardware profile](cuda_fma_blocks_results/REPORT.md):
  retained 128/64 threads; 48 registers/thread; about 29% fewer executed FP64
  arithmetic instructions than the previous specialized kernel.

Those measurements describe the benchmark prototypes. They are not relabelled
as new measurements of this package integration. The release branch now passes 69 CUDA tests (five optional PyTorch CUDA tests
are skipped). Specialized dispatch is exercised with automatic and forced
selection, full and partial chunks, coherent initial matrices, continuous
drives, FSAL/output boundaries, and comparisons against the generic FMA path.
These tests establish numerical agreement, not a new specialized speedup
measurement. Generic FMA timings and full-flight checks are recorded in
[the generic integration report](CUDA_GENERIC_FMA_INTEGRATION.md).

For future control comparisons use `rhs_specialization="off"`; historical
benchmark drivers that assume the package is generic need that option in their
base options. Original benchmark output directories and source hashes remain
unchanged. New simulation cache identities include the modified CUDA module and
the new generator module; historical cache hashes must retain their original
meaning.

Pending optimization ideas, deferred numerical experiments and completed negative
results are consolidated in [CUDA performance follow-ups](CUDA_PERFORMANCE_FOLLOWUPS.md).

Only report summaries are included in the release branch. Historical raw GPU
profiles, source snapshots and NPZ densities remain local benchmark artifacts.
