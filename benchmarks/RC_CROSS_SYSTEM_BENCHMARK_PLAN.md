# Cross-system Lindblad benchmark suite

Status: experimental fixture and screening infrastructure are authorized. The first measured stage is the nominal K=1 cross-system screen; the checker is the sole physical runner, and each stage requires source review before launch. Production API promotion and default changes remain deferred. This suite supersedes the execution order in [the implementation proposal](RC_ACCELERATION_IMPLEMENTATION_PLAN.md). Historical evidence remains in [the benchmark report](rc_propagation_results/README.md).

## Question to answer

Which method is fastest at the required accuracy for a specified system, requested output, number of initial states, parameter scan and total worker budget? Produce recommendations by workload, rather than select one universal winner.

Two independent choices must be crossed: the RHS execution kernel and the native output mode. Benchmark photon-only and selected-population output directly; do not substitute a full-density return followed by extraction. All forward kernels still evolve the density internally. Reduced return data does not imply reduced equations of motion.

## Frozen physical fixtures

| Fixture | Definition | Purpose |
|---|---|---|
| R2-F4 | `R2_F1_7o2_F4`, X J=2 to B J'=3 | Established static optical system; existing R2 workflow provides a reference recipe |
| R3-F4-a | `R3_F1_7o2_F4`, X J=3 to B J'=4 | Different rotational manifold and coupling structure |
| R3-F4-b | `R3_F1_9o2_F4` | Distinct F1' branch; screen separately rather than silently choose a line |
| RC | Existing P(2) optical plus J=1/2/3 microwave fixture | Coherent EOM modulation and switched polarization; established 70-state benchmark |

Use explicit dressed quantum-number labels for main states and initial states; assert uniqueness and a nonzero main coupling. Freeze each fixture's E/B fields, polarization, main-pair Rabi normalization, parity retention, retained states, compaction and basis truncation before collecting comparable timings. Record actual state count, packed dimension, Hamiltonian sparsity and decay structure. Verify R3 basis convergence rather than copy R2 cutoffs. Use a magnetic field that resolves the relevant degeneracies.

Primary configurations use `qn_compact=True` for R2 and both R3 branches. RC must remain `qn_compact=False`, using the established 70-state model; do not add a compact RC variant. A full-density product is the complete density matrix of the frozen model: compact for R2/R3 and uncompacted for RC. Individual-population products must refer to states that remain individually resolved; a compact aggregate can support its total population, not the separate populations or coherences of merged states. If a requested R2/R3 output requires resolving a merged manifold, retain it explicitly where supported or use a separately labeled uncompacted fixture for that output.

R2/R3 initially use static drives, so dense propagation is eligible. RC uses the validated switched/EOM configuration with a recorded nonzero entry phase. Add a small controlled time-dependent R2/R3 or static-RC comparison where needed to distinguish system size from time dependence. Gaussian envelopes are a later targeted control, not a reason to label every system periodic.

The separate phase-modulated R2/R3 screen uses the same compact basis, fields, dressed main labels and initial density as each frozen static fixture. It rebinds only the optical coupling as a true complex envelope, `Ω(t)=Ω₀ exp(i β sin(ω_EOM t))`, with constant amplitude, RC `β` and `f_EOM`, and a nonzero absolute entry phase of 125 ns. This is phase modulation, not intensity/amplitude modulation. The first EOM screen is 3 fixtures × 11 kernels × 7 final-output products = 231 rows, with independent tight Dopri5/Tsit5 full-plus-photon references and the existing per-output/physical gates. Dense-eig is ineligible because the generator is time-dependent. If the phase-frame method is included, reconstruct to the laboratory frame before accuracy checks. EOM detuning/power and long-duration rows are later bounded stages, selected from measured short-stage costs; they do not replace the static screens.

Added EOM control: benchmark phase-modulated versions of all three R2/R3 optical fixtures, preserving their static counterparts and frozen compact basis/fields/main-state normalization. Initially use the RC EOM modulation index and frequency, recording the exact values and entry phase. Apply optical phase modulation with the Hermitian conjugate coupling, not amplitude modulation. Screen all 11 forward kernels and seven output products (231 nominal K=1 cells) against paired tight references, including full-density coherences. Static dense is ineligible; assess scalar adjoints separately. Start with a 1 µs pilot and budget focused detuning/power and longer-duration comparisons from its measured cost before broadening. Report static and EOM cases separately.

Compare methods only on identical physics and basis within a fixture. Compacted and uncompacted variants, if added, are separate fixtures: compaction must preserve every requested resolved output, and exact reduced methods must reconstruct the specified sink populations and density entries.

## Kernel factorial

Use the sparse-Hermitian times dense-density base with three switches:

- D: handle the diagonal Hamiltonian contribution directly.
- S: separate real/imaginary accumulation, the current fastest measured variant.
- U: unpack density only once and use the compatible dense-density dissipator path.

Test all eight cells: base, D, S, U, DS, DU, SU, DSU. Four cells already exist; the pairwise and triple combinations need benchmark-only implementations and independent algebra/cache tests. Do not assume gains add, silently fall back, or redirect an existing experimental name to a different implementation.

Add `structured`, `expanded_sparse` and `structured_upper`: **11 forward kernels** per fixture. Profile all of them, but use complete trajectory timings to select winners. Calibrate both Tsit5 and Dopri5 on each fixture instead of assuming the package default is optimal; cross-check finalists with both solvers.

## Native output matrix

Every forward kernel, including every combination, must receive its own short physical screening row for each primary output:

| Requested product | Native output contract | What is compared |
|---|---|---|
| Full final density | `output="full"`, `output_when="final"` | All populations and coherences; include reconstruction in end-to-end time |
| All final populations | `output="populations"`, final | Entire population vector |
| Selected populations separately | `output="selected"`, diagonal `output_indices`, final | One population and a small vector, e.g. four resolved states, as separate sizes |
| Total emitted photons | `output="photon_integral"`, final, solver-native integral | One weighted cumulative photon count |

Also screen a summed-manifold product and `full_and_integral` as a joint product. For forward methods use the native selected diagonal output and sum only those entries, counting that small final reduction in the requested-product timing. Distinguish this single sum from returning each component. Use quantum labels to choose meaningful states for each fixture; do not assume a J=0 manifold is explicitly resolved in every compact model.

Later test sampled trajectories separately from final-only output. Include `full_and_integral` as a distinct joint-product case where supported, since callers requesting both density and photons should not be charged two forward solves. Never use that joint result as the cost of a scalar-only product without identifying necessary internal state transfer.

For switched RC, preserve exact discontinuities and absolute EOM time for all methods. The current segmented Python harness transfers full density between segments even for a scalar final product. Audit this overhead explicitly. A benchmark driver may keep that carry state inside the solver; until available, report segment handoff costs and the actual native outputs used. Do not hide a full-density-plus-photon duplicate solve or pretend the current public API has an unimplemented discontinuity schedule.

## Other method families and eligibility

| Method | Eligible comparisons |
|---|---|
| Native adaptive forward methods | All primary outputs and fixtures, subject to supported plan types |
| Static `dense_eig` and reusable dense propagator | Truly static fixtures; same initial batch and outputs; factorization, reduction and full-density reconstruction charged explicitly |
| Adjoint | Photon integral, one selected population or summed manifold, and multiple separate populations with one response solve per independent objective |
| Fused initial-state block prototype | Full-density and derived-output experiments; retain Python/SciPy/Numba backend label; prior negative result does not establish native performance |
| Phase-frame transformation | Time-dependent fixtures where defined, with reconstruction to the original frame before validation |
| Sparse/dense splitting | Applicable controls only; previously failed switched-EOM cases remain historical unless a specific change justifies a new test; static/no-EOM cases are distinct models |

Adjoint is not an RHS output switch: one objective yields one scalar response for each initial state. For four separate final populations, charge four adjoints; a total manifold population needs one. Screen objective counts 1, 4 and, where practical, the complete population vector. Record that full-density output is unsupported by the tested scalar adjoint interface rather than manufacture an equivalent full-density timing. Distinguish existing Python prototypes from any new benchmark-only native implementations.

Period-Arnoldi/Krylov and full period maps are outside this suite's active execution scope. Preserve the earlier period-Arnoldi result as an accuracy failure, and the full-map result as construction-cost-gated/unmeasured: they are different outcomes. Do not schedule either again, including parallel construction or amortization experiments, without a new specific reason and a separate decision to revisit them. The report may reference this historical evidence but must not list these methods as pending suite runs.

## Initial states, parameters and parallel work

Use K=1, a fixed small representative mixture/batch, all individually resolved states in the driven ground manifold, and all retained model states where useful. R2 has 20 J=2 basis populations; R3 has 28 J=3 basis populations before any relevant model reduction. Select by labels and verify actual membership. Uncompacted RC retains the documented K=20 subset and K=70 all-model-state study. Validate every returned initial-state result, including zero/near-zero photon responses and a nontrivial weighted mixture.

Cover resonance and near/far detuning on both sides, plus weak/nominal/strong optical power; vary microwave power separately for RC. Center each optical line on its own dressed main pair. Record angular-frequency detuning and Rabi values and power-normalization assumptions. Prefer one-axis sweeps plus selected interaction corners to an unbounded Cartesian grid. Never eliminate a method globally from its resonance timing: eligibility and finalist selection are local to system, output and parameter regime.

Report the parallel axis explicitly: independent initial-state solves, parameter-point workers, BLAS threads, and any shared block computation. Compare complete scans at the **same total worker budget**, initially 1, 3 and 8 where feasible. For example, allocate adjoint workers across parameter points, while the forward competitor may allocate the same workers across points or initial states. Include a competent forward scheduling baseline, not an artificially restricted one. Avoid nested oversubscription and record actual worker/BLAS counts.

## Accuracy and timing rules

### Added parallel dense and memory comparison

For the three static R2/R3 fixtures, benchmark independent parameter-point processes at worker budgets 1, 3 and 8, with one BLAS thread per process. Use the same detuning/power points and driven-manifold/all-model initial batches as the forward/adjoint throughput study. Factor each point once and reuse its factorization across initial states and all seven output products, including full final density. Compare with Rust D+S+U using the same total worker budget. Report cold scan time including factorization, reused-factorization scan time, and per-product evaluation/reconstruction time separately; use three repeats and the existing accuracy gates. RC remains ineligible for the static dense method.

Measure both methods' sampled peak aggregate process-tree RSS (parent plus children), starting RSS, peak increase and sampling interval; record Windows private bytes where available. Existing screens recorded available system memory snapshots, not solver peak memory. Do not reinterpret those snapshots as measured method memory consumption.

- Freeze requested accuracy before selecting performance winners. Keep a strict kernel-isolation tier separate from a practical production tier calibrated to the existing accepted RC accuracy (approximately 1e-6 density absolute and 0.3% populated-state relative error). Do not assume one rtol is equally accurate across solvers, methods or fixtures.
- Use a short tolerance ladder spanning practical loose settings through tight references; select the fastest setting that meets the same output error target. Do not require a full tolerance sweep at every parameter point. Keep failed rows and tighter retries.
- Kernel-equivalence tests must include complex Hermitian densities, lower-triangle Hamiltonian contributions, coherent states, decay transfer, packed/matrix paths and cache switching. New combinations require these checks before physical runs.
- Build independent tight references per frozen fixture/representative parameter regime, with cross-solver or independent-generator checks. Reuse old artifacts only after explicit stable-provenance and checksum verification. Dynamic CPU/memory snapshots are measurements, not model identity.
- Full-density claims require all elements. Population claims require each requested component or sum, absolute errors near zero and relative errors above a documented floor. Photon claims use absolute plus relative integral error. Forward physical checks include finiteness, trace, Hermiticity and positivity; adjoint costates require duality and observable checks, not trace one.
- Separate build/compile/JIT, generator extraction/factorization, per-point preparation, warm integration, output/reconstruction, memory and aggregate scan time. Record cold and reused costs. Tiny timing differences need repeated evidence.
- Use at least three alternating accepted comparisons for a claimed performance winner and report spread. Use current before/after resource sampling with the 50% median-CPU ceiling; modest unrelated activity is allowed, competing heavy simulations pause collection. Do not rewrite historical guard outcomes.

## Staged execution and bounded coverage

1. **Catalog and feasibility.** Freeze fixtures and enumerate case IDs, support/eligibility, output shapes, dense memory estimates and exact discontinuity handling. Implement only benchmark infrastructure and missing experimental combinations required for the suite; defer production APIs/default changes.
2. **Correctness and RHS profiles.** Check all eight factorial cells and existing controls on every fixture. RHS profiles diagnose cost but do not eliminate physical screening rows.
3. **Short native-output screen.** Screen all 11 kernels on all four fixtures for seven K=1 products: full final density, photon integral, full population vector, one selected population, four selected populations, a summed manifold, and joint full-plus-photon output. This is 308 forward-kernel/output cells before dense and adjoint eligibility rows, solver calibration, other methods, and reference/retry work. One observation per screen identifies candidates but cannot establish a small speedup. Build each fixture once per stage and reuse its prepared plan; record setup separately.
4. **Parameter and reuse screens.** Repeat screening in the selected detuning/power regimes and initial-batch workloads. Each method gets a per-regime screen before pruning within that regime. Add eligible dense, adjoint and other-family comparisons. Include a fixed-budget multi-parameter scan, especially for adjoint.
5. **Repeated finalists and long durations.** Repeat competitive methods separately for each output and system; always retain a baseline. Use progressively longer durations, including at least 200 us for RC and realistic R2/R3 durations from their workflows. Test a focused set of nominal, detuned and power corners. Do not run the full 11-kernel factorial at 200 us everywhere merely for completeness. Mark untested long-duration cells explicitly.
6. **Recommendation report, then implementation plan.** Produce per-system/output/initial-batch/parameter-regime recommendations, setup break-even points, accuracy-versus-cost plots and worker-budget throughput plots. Only then revise the production implementation proposal.

Before launching a stage, calculate its explicit cell count and estimate wall time from measured short pilots, including references, resource sampling and repeats. Set time/memory limits and checkpoints per stage. The initial 308-cell count covers only the nominal K=1 forward-kernel/output screen, not the full suite. Advance based on recorded results, not a fixed giant overnight job.

Every catalog row must have a status: planned, passed, accuracy failed, unsupported/not applicable, resource paused, timeout/unmeasured, or deliberately deferred with a reason. Preserve provenance, raw timings, failed attempts and repetitions. No missing row should be interpreted as a loss or silently removed from the comparison.
