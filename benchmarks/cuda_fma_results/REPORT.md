# Explicit FP64 FMA accumulation in the specialized CUDA derivative

Benchmark-only optimization guided by the successful
[Nsight hardware profile](../cuda_kernel_groups_results/NSIGHT_PROFILE.md),
2026-10-21 UTC, RTX 5070 Ti. No package implementation or production defaults
are changed.

## Change and rationale

The measured stage-1 derivative used the FP64 pipeline at 79.5% of sustained
capacity, while DRAM utilization was 21.5%. Nsight counted 324,576 fused and
567,648 non-fused FP64 instructions and suggested reducing separate multiplies
and adds. The new benchmark replaces complex-product accumulation with explicit
double-precision `fma` calls:

```cpp
// Previous generated form:
ar += r*u - w*v;
ai += r*v + w*u;
// New generated form:
ar = fma(r, u, ar);
ar = fma(-w, v, ar);
ai = fma(r, v, ai);
ai = fma(w, u, ai);
```

The same replacement is made for the other side of the commutator. All 4,278
complex terms are retained. FP64 states, matrix values, decay, diagonal terms,
Tsit5 stages, timestep, support and physical configuration remain the same.
No fast-math flag or FP32 conversion is used. Floating-point association and
rounding change, so this requires numerical checks rather than bitwise identity.

The compiled kernel uses **48 registers/thread**, versus 56 for the baseline,
and zero local/shared memory. Generated source grows from 448,133 to 606,419 bytes.
The first validation solve took 18.11 s including cold JIT/setup; warm source
generation/module lookup took about 4 ms. These are not isolated compiler timings.

## Numerical checks

Compared with the installed solver, `rtol=1e-10`, `atol=1e-12`:

| Check | Maximum absolute density difference |
|---|---:|
| Odd 13 ns interval | 4.27e-17 |
| 12 us interval crossing switches, four snapshots | 1.11e-15 |
| Initial coherent-state probe crossing switches | 1.11e-15 |
| Full 377.124 us flight, four snapshots | 3.55e-15 |
| Additional 25-lifetime dark tail | 3.55e-15 |

Full-flight validation covers eight physical settings and all 12 independent
initial states (96 rows), including 80 mW / +100 MHz. Input states, parameters,
per-initial-state reference/candidate density outputs, statistics and source
snapshots are retained. This establishes same-grid equivalence, not convergence
of the unchanged 2 ns timestep. The existing flight endpoint and 17-pass
geometry are preserved.

## Timing design

The comparison baseline is the combined specialized kernel, with diagonal reuse
and stage fusion, using tuned blocks: 128 threads for the full 1,024-row batch,
64 for the final 548-row partial batch. Actual last full/partial production rows
at 150 m/s are used. Each timing evolves a 12 us interval across Doppler and
polarization switches with the same 2 ns Tsit5 method and original waveforms.

Six adjacent pairs alternate the two orders (three of each, shuffled), after
warmups. Every timed result is compared with an installed-solver density
reference. Timing includes solver setup/output overhead; JIT is warmed.
These are matched short-interval timings, not a full production scan.

## Result: repeatable 30–31% additional runtime reduction

| Batch | Tuned specialized baseline | Explicit FMA | Median paired runtime reduction |
|---|---:|---:|---:|
| 1,024 rows, 128 threads | 3.8895 s | 2.7694 s | **30.09%** |
| 548 rows, 64 threads | 2.3978 s | 1.6696 s | **31.04%** |

Times are separate medians; percentage reductions are medians of adjacent-pair
ratios. Every one of the six pairs favors FMA for each batch shape. Full-batch
reductions span 25.75–33.46%, and partial-batch reductions span 27.32–33.10%.
Median paired speedups are 1.431x and 1.450x respectively.

The largest timed density difference from the installed solver is 1.23e-15.
All 93 package/native files in the timing source-integrity audit are unchanged.
The full-flight and dark-tail checks above independently cover the changed
rounding/association beyond the short timing interval.

**Retain this as the next benchmark candidate for package integration.** The gain
is measured against the already tuned combined specialized prototype, not the
stock solver. Do not multiply it by speedups from unmatched historical runs or
extrapolate these seconds directly to a full production scan. The subsequent [block-size and FMA hardware-profile study](../cuda_fma_blocks_results/REPORT.md)
retained 128/64 threads, found no reliable extra block-size gain, and confirmed
that FP64 instruction execution remains dominant after FMA. The initial timings
here retained block sizes for a matched comparison.

## Evidence and reproduction

- [Prototype](../_cuda_fma_accumulation.py)
- [Validation driver](../validate_cuda_fma_accumulation.py)
- [Timing driver](../bench_cuda_fma_accumulation.py)
- [Validation results](../cuda_fma_validation_results/20261021T033359582589Z/results.json)
- [Timing results](20261021T033633570658Z/results.json)

From CeNTREX-TlF using the SPB environment:

```powershell
$env:PYTHONIOENCODING='utf-8'
../CeNTREX-SPB-testing/.venv/Scripts/python.exe benchmarks/validate_cuda_fma_accumulation.py --confirm-only --batch 96
../CeNTREX-SPB-testing/.venv/Scripts/python.exe benchmarks/bench_cuda_fma_accumulation.py
```
