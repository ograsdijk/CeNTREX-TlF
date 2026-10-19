# Cross-system Lindblad benchmark status

Update: 2026-10-11. All three static parallel-dense stages are complete (6,048/6,048 accuracy-passing comparisons), as are all three EOM screens (231/231) and focused per-output dense scans (189/189). No physical benchmark jobs remain in this queue. The older stage-status sections below are a historical snapshot and undercount completed regime, batch, throughput and long-duration runs. Production API/default changes remain out of scope. The suite definition is in [RC_CROSS_SYSTEM_BENCHMARK_PLAN.md](../RC_CROSS_SYSTEM_BENCHMARK_PLAN.md).

## Completed per-output parallel comparisons

All-model one-hot initial states (40 for R2; 49 for either R3), eight detuning/power points, 1 microsecond evolution, eight workers for both methods. Dense uses one BLAS thread per process. Times are median whole-parameter-scan wall times across three repeats, including output reconstruction and dense result transfer. Native timings are reused from the frozen parallel-dense stage with matching provenance. Startup/build/reference costs are separate. Dense cold estimates add the measured parallel factorization-only scan to the measured requested-output scan; they are additive estimates, not separately timed end-to-end cold runs. Each parameter point needs its own factorization.

### Full final density matrix, including coherences

| System | D+S+U (s) | Dense cold estimate (s) | Dense reused (s) | Cold / reused speedup |
|---|---:|---:|---:|---:|
| R(2) | 4.6279 | 1.0841 | 0.0930 | 4.27x / 49.76x |
| R(3), F1'=7/2 | 14.8122 | 3.7513 | 0.1812 | 3.95x / 81.75x |
| R(3), F1'=9/2 | 8.9256 | 3.7926 | 0.1816 | 2.35x / 49.15x |

### Total emitted photons

| System | D+S+U (s) | Dense cold estimate (s) | Dense reused (s) | Cold / reused speedup |
|---|---:|---:|---:|---:|
| R(2) | 4.4495 | 1.0210 | 0.0299 | 4.36x / 148.81x |
| R(3), F1'=7/2 | 14.3901 | 3.6322 | 0.0621 | 3.96x / 231.72x |
| R(3), F1'=9/2 | 8.5823 | 3.6734 | 0.0624 | 2.34x / 137.54x |

### All final populations

| System | D+S+U (s) | Dense cold estimate (s) | Dense reused (s) | Cold / reused speedup |
|---|---:|---:|---:|---:|
| R(2) | 4.5082 | 1.0689 | 0.0778 | 4.22x / 57.95x |
| R(3), F1'=7/2 | 14.3230 | 3.7279 | 0.1578 | 3.84x / 90.77x |
| R(3), F1'=9/2 | 8.6806 | 3.7671 | 0.1561 | 2.30x / 55.61x |

### Selected and joint outputs: reused dense vs forward

Each cell gives `dense seconds / forward seconds`, with the same eight-worker scan workload above.

| Output | R(2) | R(3), F1'=7/2 | R(3), F1'=9/2 |
|---|---:|---:|---:|
| One selected population | 0.0406 / 4.4902 | 0.0599 / 14.3839 | 0.0643 / 8.6970 |
| Four selected populations | 0.0417 / 4.4703 | 0.0696 / 14.4381 | 0.0740 / 8.6077 |
| Group population sum | 0.0673 / 4.4531 | 0.1300 / 14.3765 | 0.1190 / 8.4951 |
| Full density plus photons | 0.1147 / 4.6047 | 0.2221 / 14.8042 | 0.2283 / 8.9849 |

Sources: [R2 output scans](cross_system_dense_output_scans_v1/focused_v1/r2_f4/result.json), [R3 7/2 output scans](cross_system_dense_output_scans_v1/focused_v1/r3_f1_7o2_f4/result.json), [R3 9/2 output scans](cross_system_dense_output_scans_v1/focused_v1/r3_f1_9o2_f4/result.json). All 189 focused rows passed accuracy. Final-only output was measured; saving full density at many timesteps remains untested here.

## Completed phase-modulated EOM screens

Compact R2/R3 models retain their static-control basis, fields and initial states. Optical phase is exp(i beta sin(omega t)), beta=3.8, frequency=1.6 MHz, absolute entry time=125 ns, duration=1 microsecond. Each fixture passed 77/77 kernel-output cells. Dense static propagation is ineligible. The actual prepared R2 Hamiltonian was Hermitian to roundoff and time-varying. These are single-observation screens, not repeated performance estimates.

| D+S+U speedup over structured | R(2) | R(3), F1'=7/2 | R(3), F1'=9/2 |
|---|---:|---:|---:|
| Full density | 1.019x | 1.291x | 1.338x |
| Photons | 1.112x | 1.288x | 1.330x |
| All populations | 1.021x | 1.289x | 1.329x |

Sources: [R2 EOM](cross_system_eom_screen_v1/suite_v2/r2_f4/screen.json), [R3 7/2 EOM](cross_system_eom_screen_v1/suite_v2/r3_f1_7o2_f4/screen.json), [R3 9/2 EOM](cross_system_eom_screen_v1/suite_v2/r3_f1_9o2_f4/screen.json). The earlier R2 suite_v1 attempt stopped after 44 passing cells because of tuple indexing in selected-population validation; its partial artifacts remain preserved. The fixed runner was validated and all fixtures rerun under suite_v2.

## Parallel parameter scans: output-specific timing

Report photons, full final density (including coherences), all populations, selected populations, group sum, and joint density-plus-photons separately. The previously quoted R2 dense scan times (1.35 s cold and 0.42–0.45 s warm at eight workers) cover **all seven separate output requests combined**. They are not timings for any individual output.

The original frozen parallel driver records whole-scan forward timing per output, but records dense timing per point and whole-scan timing only for all seven outputs together. The completed focused follow-up above measures dense whole-scan time for each output separately; point timings must not be compared directly with forward whole-scan timings.

Completed R2 forward measurements below cover all 40 model initial states, eight detuning/power points and 1 µs evolution; times are medians of three repeats. These are separate output requests. Source: [R2 parallel result](cross_system_parallel_dense_v1/suite_v1/r2_f4/result.json).

| Requested output | D+S+U, 1 worker (s) | D+S+U, 3 workers (s) | D+S+U, 8 workers (s) |
|---|---:|---:|---:|
| Full final density | 28.3384 | 13.7899 | 4.6279 |
| Total emitted photons | 28.1984 | 14.8461 | 4.4495 |
| All populations | 28.2218 | 13.7817 | 4.5082 |
| One selected population | 28.1176 | 14.0410 | 4.4902 |
| Four selected populations, individually | 28.1040 | 13.9164 | 4.4703 |
| Selected group population sum | 28.1165 | 15.2264 | 4.4531 |
| Full density plus photons, together | 28.4337 | 11.0666 | 4.6047 |

The focused manifests record factorization-only setup, reused evaluation, their combined estimated cost, and worker startup separately. Factorizations are reusable at the same detuning/power point; changing either requires a new factorization.

## Completed K=1 nominal screens

All four fixture screens have 77/77 accuracy-passing rows (11 kernels × 7 outputs), with the same native extension SHA256 `1240a38c…eed21`. Screen, reference JSON and NPZ checksums were verified against each manifest. They are single timing observations at 1 µs; the “fastest” values are candidate-selection data only.

| Fixture | States | Screen artifact | Mean fastest-vs-structured wall reduction across 7 outputs* |
|---|---:|---|---:|
| R2-F4 (`r2_f4`) | 40 | [screen.json](cross_system_screen_v1/nominal/r2_f4/screen.json) | 5.8% |
| R3-F4-a (`r3_f1_7o2_f4`) | 49 | [screen.json](cross_system_screen_v1/nominal/r3_f1_7o2_f4/screen.json) | 25.4% |
| R3-F4-b (`r3_f1_9o2_f4`) | 49 | [screen.json](cross_system_screen_v2/nominal/r3_f1_9o2_f4/screen.json) | 27.8% |
| RC | 70 | [screen.json](cross_system_screen_v3/nominal/rc/screen.json) | 22.0% |

*Arithmetic mean of the seven per-output reductions for the fastest accuracy-passing row versus `structured`; one timing per row, so not a repeated-performance estimate. Outputs are full density, photons, all populations, selected 1, selected 4, group sum and full-plus-photon.

## Relative kernel speeds: full final density

Speedup is `structured wall time / kernel wall time`, with `structured = 1.00×`; higher is faster. These are **single-run, nominal 1 µs full-density timings**, including output reconstruction. Small differences remain preliminary. All rows passed the accuracy gates. **Bold rows are the three existing comparison kernels.** The source artifacts are linked in the fixture table above.

| Kernel / variant | R(2) F′=4 | R(3) F₁′=7/2 F′=4 | R(3) F₁′=9/2 F′=4 | RC |
|---|---:|---:|---:|---:|
| **`structured`** | **1.00×** | **1.00×** | **1.00×** | **1.00×** |
| **`expanded_sparse`** | **0.88×** | **1.04×** | **1.07×** | **0.32×** |
| **`structured_upper`** | **0.62×** | **0.79×** | **0.82×** | **0.71×** |
| Base | 0.95× | 1.18× | 1.21× | 1.09× |
| D — direct diagonal | 0.95× | 1.23× | 1.26× | 1.17× |
| S — separate real/imaginary | 0.97× | 1.26× | 1.30× | 1.23× |
| U — unpack once | 0.97× | 1.21× | 1.26× | 1.19× |
| D+S | 1.02× | 1.29× | 1.33× | 1.24× |
| D+U | 1.01× | 1.27× | 1.30× | 1.21× |
| S+U | 1.03× | 1.30× | 1.32× | 1.25× |
| D+S+U | 1.06× | 1.34× | 1.38× | 1.29× |
| Dense eigensystem | Not measured at K=1 | Not measured at K=1 | Not measured at K=1 | Ineligible: time-dependent |
| Scalar adjoint | Unsupported for full density | Unsupported for full density | Unsupported for full density | Unsupported for full density |

Base is `experimental_structured_sparse_dense`: a sparse Hermitian Hamiltonian acting on a dense density matrix. D, S and U add the optimizations named above; combined rows enable the corresponding switches together. R(2)/R(3) use compact models (40/49 states); RC remains uncompacted (70 states). All kernels evolve the full density, including coherences. These numbers apply to full-density output; photon and population products have their own timing rows.

## Relative kernel speeds: total emitted photons

The same nominal 1 µs, K=1 comparison, now requesting photon integral output. Speedup is relative to `structured = 1.00×`; higher is faster. These are single observations, not repeated winner estimates. RC includes the full-density segment handoff required by the current segmented interface. **Bold rows are the existing comparison kernels.**

| Kernel / method | R(2) F′=4 | R(3) F₁′=7/2 F′=4 | R(3) F₁′=9/2 F′=4 | RC |
|---|---:|---:|---:|---:|
| **`structured`** | **1.00×** | **1.00×** | **1.00×** | **1.00×** |
| **`expanded_sparse`** | **0.88×** | **1.04×** | **1.07×** | **0.31×** |
| **`structured_upper`** | **0.63×** | **0.79×** | **0.82×** | **0.72×** |
| Base | 0.96× | 1.18× | 1.23× | 1.15× |
| D | 0.99× | 1.22× | 1.27× | 1.17× |
| S | 0.99× | 1.26× | 1.31× | 1.22× |
| U | 0.98× | 1.22× | 1.26× | 1.19× |
| D+S | 1.03× | 1.28× | 1.32× | 1.25× |
| D+U | 1.02× | 1.26× | 1.28× | 1.22× |
| S+U | 1.01× | 1.30× | 1.35× | 1.25× |
| D+S+U | 1.07× | 1.34× | 1.38× | 1.28× |
| Dense eigensystem | Not measured at K=1 | Not measured at K=1 | Not measured at K=1 | Ineligible: time-dependent |
| Scalar adjoint | Not measured at K=1 | Not measured at K=1 | Not measured at K=1 | Not measured at K=1 |

### Dense and adjoint photon comparison: all initial states

Dense and adjoint are propagation methods, rather than RHS kernel switches. Their completed measurements use a different workload: **one worker, 1 µs, photons for every retained one-hot initial state** (K=40 for R2; K=49 for each R3). The baseline below is **D+S+U forward = 1.00×**, not `structured`. All listed photon rows passed accuracy. Single observations; JIT warm-up and fixture construction are excluded. Dense factorization is shown both included and already reused. Adjoint uses the Python/SciPy/Numba prototype and returns one photon response per initial state, not density matrices.

| Method / cost scope | R(2), K=40 | R(3) F₁′=7/2, K=49 | R(3) F₁′=9/2, K=49 | RC |
|---|---:|---:|---:|---|
| D+S+U forward | 1.00× | 1.00× | 1.00× | Pending |
| Dense, factorization + evaluation | 1.13× | 1.52× | 0.82× | Ineligible |
| Dense, reused factorization | 34.02× | 60.22× | 32.94× | Ineligible |
| Scalar adjoint, warm integration | 1.45× | 2.82× | 2.40× | Pending |

Sources: [R2 batch](cross_system_batch_amortization_v1/nominal1us_retry1/r2_f4/result.json), [R3 F₁′=7/2 batch](cross_system_batch_amortization_v2/stage4_batch_v2_retry1/r3_f1_7o2_f4/result.json), [R3 F₁′=9/2 batch](cross_system_batch_amortization_v2/stage4_batch_v2_retry1/r3_f1_9o2_f4/result.json). These serial batch speedups do not establish gains against an eight-worker forward scan; the matched-budget throughput stage is separate.

### Photon batches: fastest measured worker settings

Compare the fastest accuracy-passing setting among the tested worker budgets 1, 3 and 8 for each method. These are all-model initial-state batches at 1 µs, one timing observation per setting. Native workers are independent Rust trajectory workers; dense workers here are BLAS threads during reused evaluation. Factorization is excluded from the warm dense column. This is best measured latency within a maximum budget of eight workers, not evidence of an optimal worker count outside the tested settings.

| System / initial count | D+S+U, fastest measured | Dense reused, fastest measured | Dense speedup over parallel D+S+U |
|---|---:|---:|---:|
| R(2), K=40 | 153.75 ms (8 workers) | 22.17 ms (3 BLAS threads) | 6.93× |
| R(3) F₁′=7/2, K=49 | 464.63 ms (8 workers) | 40.81 ms (3 BLAS threads) | 11.39× |
| R(3) F₁′=9/2, K=49 | 286.32 ms (8 workers) | 41.06 ms (8 BLAS threads) | 6.97× |
| RC | Pending | Ineligible: time-dependent | Not applicable |

Use this comparison, rather than the serial-forward ratios above, when assessing gains over the tested parallel forward baseline. Differences between dense thread settings are small and require repeats before choosing a thread count. Cold factorization cost and parameter-scan throughput remain separate comparisons.

## Completed parameter screens

The regime stage has 25 complete artifacts so far: 10 R2-F4, 10 R3-F4-a and 5 R3-F4-b regimes. That is 1,925/1,925 completed kernel-output rows passing accuracy, with zero unsupported or failed rows. See the [stage 4 artifact tree](cross_system_regime_screen_v1/stage4_v1/). RC regime rows and the remaining R3-F4-b regimes are not represented in this completed count.

## Completed initial-state batch pilot

The R2-F4 1 µs pilot compares K=4, K=20 driven J=2 states and K=40 model states at worker budgets 1/3/8 across seven outputs. The raw result is [result.json](cross_system_batch_amortization_v1/nominal1us_retry1/r2_f4/result.json): 129 rows passed, 12 failed the positivity gate, and 9 were unsupported. The 12 failures are native full/joint-density rows for K=20/K=40; the worst minimum eigenvalue was about −6.0×10⁻⁷ despite density errors below 10⁻⁶. Preserve these rows as accuracy failures. A tighter retry is being collected; until its final manifest is complete, no corrected batch timing is reported.

## Remaining work

Added: parallel dense parameter-point processes (1/3/8 workers, one BLAS thread each), compared with parallel D+S+U on identical detuning/power points and initial batches. Cover all seven outputs, including full density, with cold factorization and warm reuse reported separately. This test is pending. Existing runs captured available system RAM, not peak method memory; the new comparison will measure sampled aggregate process-tree peak RSS and report its sampling interval for both methods.

Nominal RC/R3 regime coverage, batch retry and other fixtures, parameter-throughput comparisons, and repeated long-duration finalist rows remain incomplete. A source-only long-finalist driver is under review. No cross-system production recommendation is made here. The source and per-cell artifacts remain under `benchmarks/rc_propagation_results/`; use each JSON manifest and checksum before aggregating later stages.
