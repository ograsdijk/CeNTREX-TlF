# Lindblad kernel and propagation-method promotion tests

Status: planned, not started. This is the follow-up to the completed RC/R2/R3 benchmarks. Its purpose is to establish where D+S+U is a safe default candidate, validate dense and adjoint eligibility, and measure the cost of sampled outputs. It does not change package defaults or authorize benchmark execution by itself. User priority: first integrate the already successful methods as explicit opt-in package capabilities; run this broader suite afterward. **Do not change defaults during that implementation.**

## Estimated effort and order

The additional testing suite is estimated at 1-2 working days: fixture/driver/correctness work 3-6 hours; pilot and structural screens 1-3 hours; sampling/batch/parallel comparisons 3-8 hours; longer checks/report 1-3 hours. These are provisional effort/runtime budgets, not a promised unattended wall-clock finish. Revise after the compact R(0) pilot. Native adjoint implementation is separate from that estimate.

Implementation first: stable opt-in names for the benchmarked forward kernels, package adjoint response support with explicit supported objectives and eligibility, retention of existing static dense/factorization-reuse APIs, focused regression tests and usage documentation. Keep all default solver, execution mode and tolerance values unchanged. The full fixture suite below remains queued future work, not an automatically started benchmark run.

## Decisions this suite should support

1. Can the D+S+U implementation receive a stable public name and become the default Rust forward kernel, with documented eligibility and fallback behavior?
2. When should users explicitly choose static dense propagation or adjoint observable responses instead of forward integration?
3. How do full-density storage, native selected outputs, grouped populations and photon integration affect sampled-trajectory cost?

Keep kernel selection, integration algorithm and tolerances separate. Passing this suite does not automatically change the default solver or tolerances. Do not automatically route scalar requests to adjoint.

## Fixture matrix

Freeze a manifest before timing each fixture: Hamiltonian/collapse arrays and hashes, native extension, parameter graph, state labels/order, retained states, main coupling normalization, units, initial densities, objective weights, absolute time span and discontinuities. Physical fields and normalization are constant within a comparison; only declared runtime parameters vary.

| Fixture | Construction and controls | Purpose |
|---|---|---|
| Small optical | **Compacted R(0), F'=2**; record its full excited F1' label and measured dimension rather than assuming a level count. Static X-polarized drive; add one smooth-envelope control. | Small-system overhead and compaction compatibility. |
| Larger optical | Uncompacted versions of the existing R(2) F1'=7/2 F'=4 and R(3) F1'=7/2 F'=4 fixtures; retain excited opposite parity. | Spectators, weak couplings and larger density matrices. |
| Dense mixed Hamiltonian | A deterministic synthetic complex Hermitian Hamiltonian with dense off-diagonal couplings and valid decay operators; pair with a sparse Hamiltonian at the same dimension, spectral scale and decay scale. Include a physical field-mixed optical control if needed. | Isolates matrix structure from size. A field-mixed model alone does not guarantee a dense Hamiltonian in its working basis. |
| General dissipator | Deterministic 4- and 12-level models with coherent, multi-entry collapse operators, including decay and dephasing; one transfer-only control. | Exercises C rho C-dagger and anticommutator terms beyond source-to-target decay. |
| Simultaneous drives | Existing uncompacted RC basis, **never compact RC**. Static compatible-drive control and a control with genuine residual relative-phase time dependence. Reuse established labels and experimental field settings. | Multiple optical/microwave drives, complex phases and rotating-frame eligibility. |
| Smooth amplitude | Compacted R(0) and one existing R(3) fixture with Gaussian and bounded multipass envelopes; preserve phase and absolute-time conventions. | Smooth time dependence distinct from sinusoidal phase modulation and square-wave switching. |
| Stiff control | Small dissipative model with separated decay/drive scales; fixed physics and a bounded pilot. | Checks when explicit integrators lose to stiff fallbacks; not a demand that one kernel win everywhere. |

Reuse completed compact R2/R3 static and EOM results and uncompacted RC as controls; do not rerun the entire historical matrix. An uncompacted fixture is a separate physical model: compare methods within that model, not its runtime directly against a compact model as if outputs were identical. Compaction cannot recover populations/coherences of merged states.

For physical fixtures use label-based selection and a magnetic field that resolves relevant degeneracies. Use documented angular-frequency units. Reuse the established experimental fields where possible; do not silently substitute the tiny placeholder field in quantitative comparisons.

## Initial-state coverage

- A one-hot population, a fixed population mixture, and batches of all retained one-hot states.
- A normalized coherent pure state spanning driven levels, and a seeded mixed density with complex coherences. Include excited-state support where meaningful.
- Verify every input is finite, Hermitian, positive semidefinite and trace one.
- Dense reduction must cover the complete declared initial support. Use `reduce=False` for general-coherence controls unless reachability reduction is independently verified. Reconstruct the full specified density, including sinks and coherences.
- For adjoint, evaluate both basis-state responses and contractions with the coherent/mixed inputs. Returning only diagonal responses is insufficient for coherent inputs.

## Methods and eligibility

| Method family | Required comparison | Restrictions |
|---|---|---|
| Existing forward kernels | `structured`, `expanded_sparse`, `structured_upper` | Keep explicit existing implementations and names. |
| Experimental forward kernels | Sparse-Hermitian/dense-density base; D, S, U; D+S, D+U, S+U, D+S+U | Full eight-cell factorial; verify actual dispatch and record unsupported plans. |
| Explicit integration | Tsit5 and Dopri5 on the same forward kernel | Calibrate accuracy separately; compare accuracy-passing cost rather than merely identical requested tolerances. |
| Stiff integration | SciPy BDF and Radau on small/stiff controls and a bounded representative physical case | Record actual RHS/Jacobian path. Do not imply every forward kernel is supported by each SciPy path. |
| Independent check | Small-model direct matrix exponential and/or Python reference RHS | Correctness oracle, not a required production performance competitor. |
| Static dense | `dense_eig` and reusable propagator, all supported outputs and initial batches | Static generator only; reject ill-conditioned eigenbases and unsupported features. Charge factorization and reconstruction. |
| Adjoint | Photons; terminal selected populations; group sum; independent-observable batches | Label current Python/SciPy/Numba prototype separately from any future native Rust implementation. General collapse operators require the full adjoint dissipator. No full-density substitution. |

If an optimized kernel does not support a plan, record explicit rejection or documented fallback and the actual kernel used. A silent fallback cannot count as an optimization result. Check whether the proposed default can support ordinary full-matrix prepared Hamiltonians as well as decomposed plans; maintain an explicit compatibility policy.

For smooth/multidrive models, assess whether one rotating frame makes the generator truly static. Dense is eligible only after that transformation is proved and outputs are restored to the declared frame. Do not treat a periodically varying generator as static or reintroduce previously unsuccessful period-map methods into this suite.

## Output and sampling matrix

All kernels continue evolving the full density internally. Benchmark each request separately:

1. Full density, including coherences.
2. Photon integral with declared emission weights.
3. All populations.
4. One selected population.
5. Four selected populations where individually resolved; otherwise a documented smaller subset.
6. One selected-group sum, distinct from returning every group member.
7. Full density plus photon integral in one solve.

First use final-only output. For selected finalists, use identical grids of **51 and 501 samples**, including endpoints; use a denser grid only if these show meaningful storage costs. Measure actual native final/sample output paths. At sampling times compare:

- Native selected/population/photon/joint output.
- Full-density samples followed by the same selection/reduction, including extraction cost.
- A genuinely fused weighted/group output if the library supports one. If it does not, label selection-plus-sum accurately and do not claim in-integrator fusion.

Photon integration must use the solver-native running integral or dense analytic mode integral. A trapezoidal sum of saved populations is a separate accuracy/cost control, not an equivalent photon solver. Count extra integral-state work. Full-plus-photons should not require two independent forward solves.

Measure solver wall time, result reconstruction, in-memory bytes, sampled process-tree RSS/private bytes and optional compressed artifact write time separately. Include extraction in end-to-end requested-output timing. Keep saving benchmark artifacts outside solver timings; if evaluating disk-output cost, report it as a separate metric. Repeat memory tests in fresh processes so retained earlier results do not contaminate peaks.

Dense can reuse one factorization for initial states and sample times at one parameter point. For adjoint, separate terminal observables at different sample times generally require additional backward work; a sampled response strategy must account for that work explicitly. Do not price a full sampled trajectory as one terminal adjoint solve.

## Accuracy and independent validation

Before performance collection, test the forward RHS on random Hermitian inputs against direct complex-matrix Lindblad algebra, including dense Hamiltonians and general collapse operators. For adjoint verify Frobenius duality with independent algebra and the packed-Hermitian metric; off-diagonal entries carry weight two. Compare adjoint objectives to forward results for coherent inputs as well as population mixtures. Do not apply trace-one or positivity requirements to the costate.

Use paired tight Dopri5/Tsit5 references for physical trajectories and an independent direct matrix exponential for small static models. Tighten references if their disagreement exceeds 1e-8 in density or photon count. Record failed references rather than selecting whichever reference favors a candidate.

Retain the cross-system practical gates: density max absolute error <=1e-6; each population absolute error <=1e-6 and relative error <=0.003 where reference population >=1e-6; photon absolute error <=1e-6 and relative error <=1e-5 above the declared near-zero floor. Apply population checks componentwise, including selected outputs. Also record strict density 1e-7/population-relative 0.001 results. Physical trace/Hermiticity error <=1e-9 and minimum density eigenvalue >=-1e-8, at sampled times as well as endpoints. Models are trace-preserving unless a separately specified loss model explicitly changes that expectation.

Start candidate calibration at rtol=1e-7, atol=1e-9; retain failures and retry 1e-8/1e-10 then 1e-9/1e-11 as necessary. These are starting points, not universal recommendations. Eigenmode methods do not use ODE tolerances; validate their numerical conditioning and outputs directly.

Time-dependent and segmented tests must preserve absolute physical time, phase continuity and all known switch boundaries. Reverse schedules correctly for adjoint. Do not reset phase at a segment boundary or apply a propagator from a different frame without the coherence transformation.

## Bounded execution stages

1. **Source and algebra audit.** Freeze fixtures, applicability table and provenance. Run small RHS/adjoint duality and output-layout checks before physical timing. No planning agents are needed.
2. **Small pilot.** Compacted R(0) F'=2, 1 microsecond, one initial state, all 11 kernels with both explicit solvers. Screen all seven final products with Tsit5; use full density and photons for the initial Dopri5 cross-check, expanding if results differ. Verify dense static and adjoint scalar products. Report measured cost and an ETA before expanding.
3. **Structural screen.** Each new fixture: all 11 kernels, nominal short trajectory and full density/photons/all populations. Test coherent inputs independently. Check the remaining output contracts on finalists and representative kernels. Bound the stiff control by the pilot cost.
4. **Accuracy-qualified finalists.** Retain structured, D+S+U and any distinct winner. Compare Tsit5/Dopri5, dense and eligible adjoint on nominal and selected detuning/power points: zero, +/-20 MHz; optical Rabi 0.3/1/3 times Gamma. Use a bounded subset of corners rather than a full Cartesian expansion. Synthetic models use documented dimensionless equivalents.
5. **Sampling and amortization.** Finalists on final/51/501 samples and all seven products; K=1, K=4 where available, driven-state batch and all-model batch. Adjoint objective counts 1, 4 and a larger set (up to all populations) must be charged individually or as an explicitly implemented block method. Include photon-plus-selected requests as a distinct joint workload.
6. **Matched parallel throughput.** Eight parameter points and worker budgets 1/3/8, one BLAS thread per dense/adjoint process. Compare whole-scan time for each output independently, with three alternating accepted repeats. Dense setup-only, warm reuse and startup are separate; label additive cold estimates. Adjoint parallelizes parameter points/objectives under the same total budget as forward; never compare serial adjoint to parallel forward as evidence against the method.
7. **Bounded longer checks.** Selected new fixtures at 10 microseconds, then one representative experimental-length case only where short results or accuracy concerns justify it. Reuse existing long RC/R2/R3 evidence; do not automatically repeat every cell at 109/200 microseconds.
8. **Promotion review.** Summarize eligibility, correctness failures, timing spread and memory. Decide stable names/default compatibility and native adjoint API scope before production edits. Any default change needs normal package regression checks, including batch/grid output ordering and direct API calls without an explicit execution mode.

## Reporting and resource rules

- One physical runner; Luna Medium agents may implement/check drivers when execution is authorized. Root reviews each ready driver promptly and clears the next stage without leaving an idle review queue. Report only stage completion or blockers from workers.
- Retain the established 50% median background-CPU guard, allowing modest user activity. Do not run competing heavy benchmarks. Derive stage ETAs from pilots and revise them when measured costs disagree.
- Freeze code/native hashes before collection. Preserve failed attempts; use new tags for corrected runs. Save numeric arrays without pickle and use short Windows-safe paths.
- Use small per-stage/per-row checkpoints or append-only records. **Do not rewrite cumulative full-row JSON after every row**; that created excessive disk use in the earlier suite.
- Tables are separate for photons, full density, populations and selected/joint outputs. Show existing kernels distinctly, equal-budget forward/dense/adjoint comparisons, cold/warm/setup scope, and unsupported cases with reasons. Keep prototype and native timings separate.
- A near tie within repeat spread is inconclusive. Regressions on small/general models require an explicit compatibility or selection decision; do not declare a universal winner from molecular fixtures alone.

Sampled-output tests and this additional-system suite are planned work. Previously discussed serrodyne segmentation, general piecewise-static propagators and static-plus-time-dependent correction methods are separate future experiments, not additions to the active benchmark queue.
