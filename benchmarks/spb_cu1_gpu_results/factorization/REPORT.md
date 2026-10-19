# CU1 GPU shared-waveform and shared-Hamiltonian prototype

Benchmark only: no installed library implementation or default was changed. Prototype modules are isolated under benchmarks/ and substituted only inside the benchmark process.

The existing 26-level CU1 model retains both excited parities, alternating-pass Doppler, 0.5 MHz X/Z switching, 13 prism-ordered passes and the uniform twelve-state J=1 mixture. Fixed Tsit5 at 2 ns, direct RHS, exact density reduction, graph_steps=8, batch_size=1024. Matched variants use precompute_bytes=256 MB (a ceiling, not the actual allocation).

## Short screen: shared waveform and static trajectory factors

1024 simultaneous trajectories over 18-30 us, three interleaved rounds after warmup. Public-call wall time includes setup, integration and final population transfer; physical model construction is excluded.

| Workload | Existing GPU s | Factored waveform s | Speedup |
|---|---:|---:|---:|
| shared_envelope | 3.314 | 3.297 | 1.005x |
| varying_height | 6.205 | 4.079 | 1.521x |
| height_power | 5.652 | 3.754 | 1.505x |

The already-shared envelope is effectively unchanged. Initial varying-height rounds were noisier than later rounds; raw individual timings and CPU loads are retained, not discarded.

## Second screen: shared off-diagonal Hamiltonian

A second independently interleaved screen compares waveform-only factorization with shared Hamiltonian storage. These medians are from a separate timing block; do not multiply independently measured speedup ratios as a matched result.

| Workload | Waveform only s | Waveform + shared H s | Additional speedup |
|---|---:|---:|---:|
| varying_height | 3.984 | 3.527 | 1.129x |
| height_power | 3.853 | 3.588 | 1.074x |

## Full-transit confirmation

1024 distinct varying-height trajectories over the complete 271.739 us transit, one paired measurement at the same settings. This is a measured full scan, not a duration extrapolation.

| Existing GPU s | Combined prototype s | Speedup | Runtime reduction |
|---:|---:|---:|---:|
| 130.918 | 77.105 | 1.698x | 41.1% |

Historical 16-graph-step GPU timings are context only; the matched confirmation above uses eight graph steps for both variants.

## Correctness and memory

- Waveform-only versus unchanged same-step full density: max absolute difference 2.22e-16.
- Shared-H versus waveform-only full density: max absolute difference 3.33e-16.
- Twelve full-transit cases were checked at first/middle/last pass centers and endpoint. Both prototypes also pass the existing strict 0.5 ns reference criterion, including complex coherences, trace, positivity and remaining J=1 population.
- Coefficient evaluation matched across 66 times (centers, overlaps and boundary sides) and nine height/power combinations, including zero power. Additional short density comparisons cover 24 height/power cases.
- Generic tests cover both fixed solvers, two independently scaled drives, different pass heights, a varied temporal frequency, precompute off, insufficient precompute budget, zero drive, an all-zero shared factor and odd graph tails; max absolute packed-density difference 2.22e-16.
- An early prototype read beyond the precomputed table during an idle graph tail. The benchmark-only kernel now skips zero-width idle steps; explicit odd-tail tests pass.
- Off-diagonal H buffers: 2,097,152 → 2,048 bytes. Waveform tables: 60,896,192 → 121,792,384 bytes. The prototype adds subexpression cache columns without packing unused original columns, so overall memory need not decrease. These are buffer sizes, not measured peak device memory.

## Scope and next decision

Sharing uses exact expression structure and batch parameter dependence. Common-height unrotated multipass profiles expose the transverse amplitude factor; other separable coefficient subexpressions are cached generically. Shared H is used only when every nonzero off-diagonal component has the same finite real trajectory factor; distinct drives or nonseparable time dependence retain separate H storage. Diagonal Doppler/detuning and dissipative terms are not scaled.

No static eigendecomposition changes, CPU optimization sweep, velocity convolution or production scan were run. Height/power screening confirms equivalence to the unchanged fixed-step solver; it is not a new full-transit timestep convergence study at each power. No RC/R(2) speedup is claimed from this CU1 benchmark.

The measured gain supports considering library integration later, with a controllable fallback, tighter cache layout and regression coverage. Integration remains a separate user decision.

Evidence: validate.json, validate_shared_h.json, generic_checks.json, screen.json, screen_shared_h.json and confirm_shared_h.json. Per-stage source hashes and settings are retained. Formatting and conservative factor extraction were refined between stages; the raw stage hashes preserve that history.

Reproduce from benchmarks/: run check_cuda_factorization_generic.py; bench_spb_cu1_factorization.py --stage validate; --stage screen; --stage validate --shared-h; --stage screen --shared-h; --stage confirm --shared-h; then report_spb_cu1_factorization.py.
