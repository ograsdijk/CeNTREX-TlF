# RC propagation benchmark results

## Accepted idle reruns and recommendation

The user confirmed the other simulations had finished before these reruns.
Every listed `idle_*` JSON has `timings_accepted: true` and passed its
known-process/timing guard; before-run CPU samples stayed below the suite's
25% gate. The initial idle confirmation was 1.6–2.6% CPU with about 48 GB
available. The revised reference protocol refines
both tolerance and initial step twice, then performs a tight cross-solver
check. Candidate accuracy checks cover all saved full density matrices and the
solver-native photon yield accumulated over the exact switch segments.

Three separately benchmarked outputs are in scope: photon yield; the full
final density matrix including every coherence; and final-state populations
for all levels or selected groups X J=1, X J=2, X J=3, and the B manifold.
Also measure native J=1 population-index subsets from one initial state for
selected-output speed. Candidate acceptance is product-specific: a full-rho
job must reconstruct and validate every matrix element; a photon job must
pass its photon criterion; a population job must pass the returned-population
criterion. A population-only result can be accepted for that product, but it
does not stand in for requested full density output. Use the full-density
reference as truth for all derived outputs.

Existing accepted comparison-suite timings include a full-density solve plus
a separate solver-native photon-integral solve; they are not end-only or
observable-only timings. The new observable runner benchmarks five outputs
in independent runs. Its photon-only run uses native `full_and_integral`
internally at each switch segment to carry rho while retaining the photon
yield; its reported product is still photons only. Full density, all
populations, the J=1 subset and the named single population are measured in
their own runs. A public joint photon-plus-final-rho benchmark could be useful
later, but it does not replace these separate output tests. The first nominal
200 us run completed with five pairwise provisional-reference passes:

| Output product | One-run propagation wall | Returned population count | Pairwise error |
|---|---:|---:|---|
| Photon yield | 50.65 s | — | `3.36e-10` absolute |
| Full final density | 50.91 s | — | `1.11e-8` max element |
| All final populations | 50.29 s | 70 | `2.34e-10` max absolute |
| Native X J=1 subset | 48.32 s | 12 | `8.80e-12` max absolute |
| One named initial-state population | 49.27 s | 1 | `5.24e-14` absolute |

Every product passes its output-specific pairwise tolerance against the
200 us tight-Dopri5 comparison, which is not a multi-refinement reference.
No 200 us multi-refinement ladder is currently planned. The five independently computed outputs agree exactly at
the reported precision (maximum cross-output discrepancy 0). The J=1 subset
contains 12 of 70 states. Combining the initial run and two reverse-order
repeats gives medians/ranges of 50.285 s (49.508–50.416) for all populations,
48.545 s (48.323–48.627) for J=1, and 48.231 s (47.876–49.271) for one named
population. The subset medians are 3.46% and 4.08% lower, respectively, but
this does not establish that native extraction caused the difference: all
three paths carry the full density through the same 799 segments, and RHS
counts were not serialized. No more repeats are planned. This remains one
nominal parameter point and one initial state, not a parameter-grid cost
estimate.

The observable run reports shared build/segmented-plan setup once (9.53 s);
the full five-output process peaked at 597 MB RSS. Each
output's propagation wall includes solver calls and Python density carry;
artifact compression and writing are reported separately. Peak RSS is for the
whole five-output run, not a per-product memory comparison. `timings_accepted`
records the resource guard only; accuracy acceptance is product-specific in
each output's reference comparison. A completed, timing-accepted run without
a compatible reference is not thereby accuracy-validated.

For the 70-state, un-compacted short RC pilots (2–10 us constant-envelope,
plus one 1 us Gaussian case), `tsit5` with
`execution_mode="structured"` is the best tested default. It passes the full
accuracy checks and is about 3.2–3.4 times faster than the same-solver
expanded-sparse path in the nominal, strong, and weak short single-state cases.
The actual RC interaction duration is at least 200 us; none of these short
pilots establishes accuracy or runtime at that duration. The nominal 200 us
structured calibration and tight-reference comparison are available below. The recommendation
applies only to the tested short-pilot model, settings, and entry phases; it
does not establish performance for the actual-duration workload, different
truncations, a production cooling scan, or all envelopes.

The measured speedup uses the existing `structured` execution mode with the
existing Lindblad ODE solvers; this work did not add a new solver or native
RHS kernel.
The reusable full-state batch-output API and benchmark-only drivers make these
comparisons possible. It is a focused representative-case study, not an
exhaustive GPU or kernel-tuning effort.

| Case and entry phase | Expanded-sparse tsit5 | Structured tsit5 | Speedup | Accuracy |
|---|---:|---:|---:|---|
| Weak, 2 us, 0 ns | 5.41 s | 1.69 s | 3.19x | Both pass at `1e-9` |
| Nominal, 10 us, 137 ns | 32.95 s | 10.25 s | 3.21x | Both pass at `1e-9` |
| Strong, 10 us, 411 ns | 48.21 s | 14.37 s | 3.35x | Both pass at `1e-9` |
| Nominal Gaussian, 1 us, 73 ns | 4.24 s | 1.30 s | 3.26x | All four frame/mode candidates pass |

The actual-duration timing calibration is separate from these accepted short
comparisons. A nominal 200 us `tsit5`/structured run took 97.61 s over 800
exact switch segments, including saved full density and a separate native
photon solve. Its physical self-checks passed (trace error `7.1e-15`, minimum
eigenvalue `-4.2e-10`, photon yield `3.503669`), but it is not yet
reference-validated. The tight dopri5 comparison took 221.70 s; pairwise
checks report maximum density error `1.695e-8`, population error `3.716e-10`,
and photon error `3.364e-10`. It remains a pairwise 200 us comparison, not a
multi-refinement reference. The joint 100 ns save grid and switch edges yield 2,401
snapshots. First and last 10 us yield 1.2618 and 0.0005246 photons,
respectively, and mean segment cost falls from about 0.261 s early to 0.0849 s
late. Do not linearly scale short-pilot timings or treat pairwise 200 us
agreement as completed long-duration reference validation.

Structured dopri5 also passes on the nominal 10 us case at 11.48 s and weak
2 us case at 1.89 s. In the weak mode comparison, structured tsit5 has a
maximum saved-density element error of `6.86e-9`; all tested structured
variants also pass trace, Hermiticity, positivity, residual-population, and
photon criteria. The weakest `1e-7` expanded-sparse candidates remain
unacceptable: their weak-case density errors are `8.51e-7` (dopri5) and
`1.21e-6` (tsit5), despite small population and photon errors.

The phase-frame rewrite does not add a useful gain. On nominal 10 us,
phase-structured tsit5 takes 10.29 s versus 10.25 s without the transform; on
strong 10 us it takes 14.45 s versus 14.37 s. For the nominal Gaussian 1 us
case, phase-expanded takes 4.24 s versus 4.24 s, and phase-structured takes
1.31 s versus 1.30 s. Phase-frame preparation adds about 3.6 s. There is no
observed amortization case for the transform yet.

For three independent weak-case initial states at 2 us, structured batch
propagation takes 3.53 s serially, 2.46 s on 3 threads, and 2.25 s on 8 threads.
All per-initial-state accuracy checks pass; the largest saved-density element
error is `1.30e-8`. Eight threads give about 1.57x throughput over serial, not
linear scaling. This is measured at three states only; 10- and 100-state
batch costs have not been measured.

The native RHS profile used 3,200 calls per mode at 16 times. It measured
152.4 us/call for expanded-sparse and 41.0 us/call for structured, with
identical packed RHS results. Expanded-sparse time was mostly Hamiltonian
coefficient filling (53.4%) and commutator work (46.3%); structured time was
mostly commutator work (90.7%). Runtime parameter/coefficient evaluation was
below 1% in both modes, so separate input-term grouping was not pursued; the
profile points to Hamiltonian fill and commutator work as the relevant costs.
An additional profile of a dense synthetic state (selected population mixed
with a normalized complex uniform superposition) measured 147.7 versus
40.7 us/call; the initial-state profile measured 148.0 versus 40.9 us/call.
Runtime parameter evaluation stayed below 1% in all four profiles. The packed
RHS outputs agree between modes to relative `6.3e-16` on the dense state.
These profiles show that the tested existing-mode gain is not an artifact of
diagonal sparsity, while remaining fixed-state probes rather than trajectory
benchmarks.

The accepted idle 1 us sparse split pilot confirms the EOM limitation. The
reduced sparse Strang split with EOM fails the full-density gate
(`3.80e-5` maximum element error) and photon gate (`1.94e-4` relative error),
and its 1.77 s median is slower than the matching structured ODE baseline
(about 1.13 s). The tested 5 ns candidate was both slower and outside
tolerance, so finer steps were not attempted. In the separate no-EOM case, exact reduced
propagation passes (`6.84e-11` density error) and takes 0.886 s versus 1.167 s
for structured ODE. The direct no-EOM plan preparation is 3.49 s; the general
static route needs about 14.37 s for static-plan preparation, four factor
extractions, and common-support preparation. Since common model build cancels
in this comparison, the measured 0.281 s per-solve saving implies roughly 39
serial no-EOM solves to recover the extra static setup. This is a projection
for repeated solves of that fixed static model, not measured batch throughput
or an EOM acceleration result.

The matching idle dense split pilot also fails the switched EOM target at
5 ns (`3.80e-5` density error; `1.77e-6` photon absolute error) and takes
6.80 s, versus about 1.13 s for structured ODE. Its four dense generator
factorizations take about 73.56 s in total after 13.68 s static-plan
preparation; estimated four-factor workspace is 6.15 GB, with 3.07 GB peak
process RSS. The exact dense no-EOM solve passes at 0.450 s, while its
structured ODE reference passes at 1.130 s. This no-EOM check is not evidence
of a switched EOM speedup, and the one-state warm solve times do not establish
an amortized batch benefit.

Cold preparation is about 9.6–9.9 s for these cases. Thus the measured
single-state nominal end-to-end totals are about 42.8 s for expanded-sparse
versus 20.0 s for structured, including common model build/preparation but
excluding disk and process startup. A batch of three states costs about 12.0 s
end-to-end with 8 threads, including its common preparation. For an explicitly
serial, fixed-preparation projection using the nominal 10 us medians, 10 states
would take roughly 339 s expanded-sparse or 112 s structured, and 100 states
roughly 3,305 s or 1,035 s. These arithmetic projections are not measured
batch-throughput results; they exclude parallel scheduling and any cache
effects. They describe only a 10 us pilot workload, not the intended >=200 us
RC trajectories. Measured batch evidence covers three states only; 10- and
100-state end-to-end batch costs and cache/reuse break-even points remain
unmeasured.

### Periodic-map and matrix-free period-action gates

The exact one-period pilot used the canonical 5 us interval, 20 exact
switch-boundary segments, and an 8-column Hermitian packed basis batch spanning
the three X populations, a B population, and real/imaginary XX and XB
coherences. It retains absolute-time EOM modulation and measures both full
density output and a separate native photon-integral solve. The 1e-9 pass took
10.92 s; a sampled 1e-10 refinement took 15.88 s. Their maximum final matrix
element difference was `2.31e-9` and maximum photon-integral difference was
`3.41e-11`. This is a sampled tolerance probe only; `accuracy_status` remains
`accuracy_pending`, and no map-accuracy claim is made.

The full map needs 4,900 basis actions and 192,080,000 raw bytes (about
183 MiB), so storage fits the 8 GiB budget. Using the slower pilot chunk and a
2x safety factor, the estimated 8-column-batched construction takes about
19,471 s (5.41 h), over the 1,800 s gate. The gate rejects full-map
construction. A previous common-support analysis estimated 4,360 retained
coordinates and four common sinks, which would save only about 21% of raw map
storage; that estimate was not revalidated for this canonical period and is
not used to pass the gate. The complete map was not constructed. If revisited,
a map keyed to one entry phase would require an exact prefix from the arbitrary
start phase to the canonical 5 us boundary, repeated full-period applications,
then a remainder solve. Other entry phases need their own phase-keyed maps or
equivalent prefix/suffix treatment; Gaussian envelopes break periodicity and
cannot use this periodic map directly. This was an eight-column one-period
cost probe, not a completed Floquet map, map diagonalization, or
Fourier-expanded Floquet solver. The 30-minute limit is a benchmark budget,
not a claim that a full map is physically impossible. The phase-frame ODE was
tested separately and did not improve runtime; higher-order Magnus and
Fourier-expanded Floquet approaches were not tried. A separate benchmark-only
matrix-free period-Arnoldi pilot is described below; it is not a full map or a
general Floquet-solver evaluation.
The build estimate is about 2.7 h without the safety factor or 5.4 h with it.
Against the 10.25 s **10 us pilot** structured trajectory only, this gives an
optimistic break-even of roughly 950–1,900 phase-aligned 10 us trajectories.
This break-even applies only to the 10 us pilot. It does not use the actual
>=200 us duration: the 200 us nominal calibration currently takes 97.61 s
and has only a pairwise tight-Dopri comparison, not a multi-refinement
reference. Revisit long-duration validation only if needed for the selected
production tolerance, and recalculate amortization against the intended
workload. A full density map is linear
and could be reused across initial densities only while the four generators
stay fixed. Changed detunings, Rabi rates, microwave settings, or entry phases
need new maps or additional prefix/suffix propagation. In a workload spanning
many detuning, Rabi, or field values, most points will not share one map. Photon
yield reuse would also need a per-period integral functional or an augmented
state map; the pilot only timed a separate photon solve. For one fixed initial
state and constant envelope, one longer ODE solve with saved shorter times may
cover multiple durations more cheaply. Actual workload reuse could justify
construction despite the pilot's 30-minute budget, so this gate defers that
decision rather than rejecting maps on amortization grounds.

The matrix-free period-Arnoldi pilot applies the exact segmented 5 us period
action to vectors in a Krylov basis seeded by the nominal initial density. It
used 24 Arnoldi actions plus three linearity probes (27 period actions total),
took 177.85 s of action wall time plus 10.01 s setup.
The signed-coherent-input additivity discrepancy was `1.40e-10`, supporting
linearity of the implemented action to that measured precision. It did not
meet the full-density accuracy/physical gates at any tested Krylov dimension
q=4, 8, 12, 16, or 24. The q=24 endpoint had maximum density/population
absolute error `4.794e-6`, relative population error `8.60e-2` above the
population reporting floor, trace `0.999992594` (error `7.41e-6`), and minimum
eigenvalue `8.25e-13`. Lower dimensions were substantially worse; q=4 and q=8
were unstable. The q=24 action cost alone was about 3.5 times the nominal
200 us full-density output run (50.91 s), while still failing accuracy. This
is a scoped negative result for this initial-state-seeded basis, 200 us
nominal parameter key, solver/tolerance and tested q range; it does not rule
out other Krylov formulations, a reusable full map, Magnus methods, or
Fourier-expanded Floquet approaches. No full map or map diagonalization was
built. The Krylov span is tied to this initial state and fixed generator key;
reuse for other initial densities requires explicit span/projection-error
validation and is not equivalent to a full map that acts on arbitrary rho.
The supplied tight-Dopri reference is still pairwise only, so these candidate
errors are provisional against that reference rather than a completed
four-run-converged score. Given the failed q=24 checks and higher cost, further
dimension increases in this same pilot are not justified as a 200 us speed
route without a materially changed algorithm or cost profile.

An artifact-only population analysis of those same Krylov checkpoints found
that no tested q met the all-population, J=1-group, or named-population
criteria. At q=24, all-population maximum absolute/relative errors were
`4.7945e-6`/`8.5965e-2`; J=1 group errors were `5.1571e-7`/`7.3851e-2`; the
named level's absolute error was `2.997e-8`, but its relative error was
`9.467e-3` and failed the population criterion. q=12 and q=16 also had
negative populations. These are projections from the full-density-action
checkpoints, not timings of optimized population-only propagation; no photon
Krylov result was measured. The analysis artifact is
`idle_nominal_200us_period_krylov_population_products.json`.

A final-time response map for population mixtures remains a deferred proposal. No implementation or physical pilot is currently authorized. Its construction and expected reuse are unmeasured.

The full final density matrix is one required output product;
population-only or observable-only projection cannot satisfy a full-rho
request. Acceptance is specific to each candidate's requested product:
full-rho methods must preserve all elements and coherences, photon methods
must pass photon checks, and population methods must pass checks on returned
populations. Reduced-basis, Krylov, and period-action candidates cannot be
timed against the full-rho baseline unless they reconstruct and validate the
full matrix. The full-rho reference remains the truth source for the separate
products.

## Authorized methods and deferred proposals

The user authorized five methods: diagonal split, SoA/vectorization, single-unpack, fused block, and adjoint, plus an 11-case detuning/power crossover matrix. The 11-case kernel matrix is complete. The staged 1 µs fused-block/adjoint pilot is also complete for nominal, +20 MHz, and 3Γ optical-drive cases; its artifacts and scoped interpretation appear below. The checker was the sole physical runner. Separately,
the completed nominal kernel-matrix case screened eight methods and every
screening method passed accuracy checks; the SoA variant's three full-density
plus separate-photon observations had a median 18.9% lower wall than
`structured`. This result is detailed below and does not imply performance for
other parameter points. Per-parameter-point reuse is
essential: changing detuning, Rabi rates, field, entry phase, duration, or the
envelope schedule requires a new response.

1. **Adjoint propagation for photons and selected population outputs.** For a
   scalar output `y = Tr(O rho(T)) + integral Tr(W(t) rho(t)) dt`, one backward
   solve of `-d(lambda)/dt = L(t)^† lambda + W(t)`, with `lambda(T)=O`, gives
   `y = Tr(lambda(0) rho(0))` for every diagonal one-hot initial state. For
   photon yield, `W = Gamma P_B`; for final population outputs, use the
   corresponding level or group projector. Thus one adjoint solve can cover
   all initial mixtures for one scalar product; four adjoints give the four
   X J=1/J=2/J=3/B group totals, while all 70 individual final populations
   need 70 scalar outputs unless a block implementation is effective. The
   backward solve must follow the original absolute-time EOM phase and exact
   reversed switch-segment schedule. Derive `L^†` in the physical trace inner
   product: the packed Hermitian coordinate metric weights coherence
   coordinates, so a plain transpose of a packed RHS is not automatically the
   correct physical adjoint. Validate the duality identity and outputs against
   forward solves for all one-hot anchors and mixed initial states. This
   produces only the requested scalar(s), not a full final density matrix.
   The benchmark-only prototype and toy complex-H/time-dependent-H/decay
   duality checks are implemented and pass. The physical pilot compares one
   photon adjoint and one X J=1 group-population adjoint against separate
   forward scalar-output solves for three one-hots and their mixture. The pilot
   passed its accuracy gates but was slower than the corresponding forward
   scalar-output runs; see the measured 1 µs results below. No 70x wall-speedup
   claim is supported.

2. **Final-time forward response map for population mixtures (deferred).** No implementation or physical pilot is currently authorized.

3. **Optimized block RHS propagation.** Propagate a block of initial states
   through a shared operator/kernel, which is distinct from launching
   independent parallel batch solves or forcing all trajectories to share one
   adaptive controller. A block implementation may improve cache use or
   sparse matrix-matrix work, but it does not by itself reuse the final
   response across later mixtures. The previous shared-step experiment lost
   to independent Rayon solves, and the RC RHS profile found coefficient
   evaluation below 1% of cost; benefit therefore needs evidence from a
   purpose-built block commutator/RHS kernel, not shared coefficient
   evaluation alone. The benchmark-only Python/SciPy/Numba prototype was
   compared with the Rust parallel baseline at 1 µs. It passed per-state full
   density checks but was substantially slower; the shared controller uses a
   packed-coordinate norm that is not a per-state Frobenius norm. See the
   measured pilot section below.

4. **Late-time slow modes or a periodic steady orbit (deferred; physical runs not authorized).** Constant-envelope RC
   dynamics repeat every 5 us, and the nominal 200 us run spans 40 periods.
   If the period action rapidly contracts transients toward a low-dimensional
   slow/periodic manifold, one could compute that tail and use an augmented
   per-period photon functional. This may help long durations or many initial
   mixtures at one generator key, but it must bound the transient error in
   every final-density element and in the accumulated photon yield. Dark
   manifolds can leave long-lived population modes, so a small late photon
   rate alone does not prove the full density has converged. No spectral-gap,
   tail-error, or speed evidence is available; Gaussian envelopes are not
   periodic.

5. **Restarted or block Krylov period actions (deferred; physical runs not authorized).** A block seed may cover more
   initial populations than the already-tested single-density Arnoldi basis,
   but it may also increase the Krylov dimension and period-action cost. The
   q<=24 single-seed pilot failed accuracy and was slower than direct
   propagation; restarted/block variants remain untested and lower priority
   absent evidence of a low-rank response space. They must reconstruct and
   validate full density for full-rho requests. The existing nominal
   full-density ODE propagation took about 50 s at 200 us; response-map,
   adjoint, block, and slow-mode costs remain unmeasured against that anchor.

### Fused-block and adjoint pilot: 1 µs results

The checker ran three scoped cases: nominal, +20 MHz detuning, and optical drive 3Γ. The fused-block K=3 full-density runs and the photon and X J=1-group adjoints all passed their accuracy and resource gates. The v1 shared-block/photon artifacts use runner SHA256 `400b95f812cdf483defae8f711bbee8de12727485fa8c30d2269287e35eb53f7`; the v1 X J=1 attempt hit a runner bug before producing a measurement. Its corrected v2 stages used fresh references and runner SHA256 `393b6c19e0f5582c78f5279b3d74876f39ad29b14a9d25afdeb295a709552d24`. Reference NPZ and block-output NPZ checksums match their metadata, and each measured method’s provenance matches its corresponding reference. The block maximum density error was `2.74e-9`; maximum adjoint scalar absolute errors were `3.90e-11` for photons and `2.21e-11` for X J=1 population. The original failed J=1 stage is preserved in the v1 folder.

The prototypes use Python/SciPy/Numba with DOP853 (`rtol=1e-9`, `atol=1e-11`, `dt=1e-10 s`); the Rust baselines use Tsit5. Timed prototype rows exclude first-use switch-plan preparation and Numba warmup. Switch-plan preparation took about 13.7–13.8 s per isolated method stage, Numba warmup about 0.7 s; first-use totals including case setup were about 29.9–33.7 s. These setup costs are reported separately from amortized warm method time. Block mixture-linearity checks were timed separately and are not included in the K=3 comparison. Adjoint forward checks time four independent Rust scalar-output solves (three one-hots plus one weighted mixture), matching the outputs the single adjoint returns.

| Case | K=3 block | Rust full-density, 3 threads | Block / Rust | Photon adjoint / forward | X J=1 adjoint / forward |
|---|---:|---:|---:|---:|---:|
| Nominal | 8.675 s | 0.949 s | 9.1× slower | 7.103 / 2.732 s (2.6×) | 6.836 / 2.704 s (2.5×) |
| +20 MHz | 7.546 s | 0.816 s | 9.2× slower | 6.260 / 2.432 s (2.6×) | 5.984 / 2.421 s (2.5×) |
| Optical 3Γ | 9.874 s | 1.124 s | 8.8× slower | 7.920 / 3.316 s (2.4×) | 7.472 / 3.330 s (2.2×) |

At K=3, all three prototype methods were slower than their matching Rust forward baselines at this short duration, even after setup was amortized. This is a Python/SciPy/Numba prototype result, not a test of a native block or adjoint implementation. The later K=20/K=70 nominal study below measures a distinct amortization regime. No 10 µs scale-up was run, and this K=3 pilot does not change package defaults or the current RC recommendation (`structured`).

Artifacts are under [`block_adjoint_pilot`](block_adjoint_pilot/): v1 has the three full case runs; v2 contains the fresh-reference X J=1 reruns. The full provenance, per-state accuracy, solver statistics, resource guards, setup timing, and output checks are recorded in the JSON files.

### All-state block/adjoint amortization: nominal 1 µs

To test the larger-batch crossover directly, the checker ran the nominal 1 µs case at 137 ns entry phase with deterministic K=20 and K=70 one-hot batches. K=20 samples X J=1/2/3 states evenly within sorted quantum-label groups and retains the three earlier pilot anchors; K=70 covers every model basis state, including B and sink states. Strict references were computed once for all 70 inputs and reused for the K=20 subset. All nine stages passed the accuracy and 50% median-CPU gates, and the JSON/NPZ checksums and reference provenance were verified. Every timing below is one observation, not a repeat or a distribution.

| Output and initial batch | Prototype result | Rust Tsit5 structured, 1 / 3 / 8 threads | Interpretation |
|---|---:|---:|---|
| Full final density, K=20 | Shared block 29.308 s | 11.014 / 4.428 / 2.316 s | Block is 12.65× slower than the 8-thread batch |
| Full final density, K=70 | Shared block 108.664 s | 37.499 / 15.433 / 7.048 s | Block is 15.42× slower than the 8-thread batch |
| Photon integral, K=20 | One all-70 adjoint 7.155 s | 11.043 / 4.407 / 2.247 s | Adjoint is 1.54× faster than serial, slower than 3/8-thread batches |
| Photon integral, K=70 | One all-70 adjoint 7.155 s | 37.932 / 14.929 / 7.166 s | Adjoint is 5.30× faster than serial and 2.09× faster than 3 threads; it is 0.14% faster than 8 threads, effectively a tie |
| Final X J=1 population, K=20 | One all-70 adjoint 6.872 s | 10.961 / 4.285 / 2.343 s | Adjoint is 1.60× faster than serial, slower than 3/8-thread batches |
| Final X J=1 population, K=70 | One all-70 adjoint 6.872 s | 37.546 / 18.662 / 6.942 s | Adjoint is 5.46× faster than serial and 2.72× faster than 3 threads; it is 1.0% faster than 8 threads, within single-observation uncertainty |

The adjoint computes all 70 diagonal-input responses in one backward solve, so its single timing is reused for both K=20 and K=70; it does not reconstruct final density matrices. The K=20 and K=70 direct scalar rows time exactly K one-hot solves and exclude mixture checks and validation. Rust’s threads propagate independent initial states in parallel; the adjoint is a single-worker solve. Thus the K=70 rows show the expected amortization opportunity against serial forward work and near parity with eight-thread scalar batches, while the K=20 rows remain slower than parallel direct batches. There are no timing repeats, so the sub-percent comparison with eight threads is unresolved. The block method returns full density but is substantially slower than Rust at both batch sizes.

All 20 K=20 block densities and all 70 K=70 block densities passed per-state checks; maximum density errors were `4.18e-9` and `5.66e-9`, respectively, with maximum populated-population relative errors `7.35e-6` and `2.40e-5`. The photon adjoint’s maximum absolute error over all 70 inputs was `4.15e-11`; the X J=1 adjoint’s was `2.37e-11`. Both had zero recorded diagonal imaginary residual. The prototype uses Python/SciPy/Numba DOP853; its warm outer wall includes per-stage context preparation (about 1.4 ms) and its stats also expose solver-only work. Model builds took about 9.5 s and four-switch-plan preparation about 13.8 s; those first-use costs are listed separately and should not be mixed with amortized timings. Rust baselines are Tsit5 at `rtol=1e-9`, `atol=1e-11`, and `dt=1e-10 s`.

These nominal 1 µs measurements support the user’s amortization hypothesis for scalar adjoints at K=70 relative to serial forward solves, not a general speedup claim: the parallel eight-thread result is effectively tied and has only one timing. They do not establish performance at 10–200 µs, other detunings or powers, or production workloads. Full-density requests still require forward propagation. The prototype is not a native block/adjoint implementation, and package defaults and the current `structured` RC recommendation are unchanged. No 10 µs scale-up was run.

The versioned runner and complete stage artifacts are under [`block_adjoint_amortization_v2/nominal`](block_adjoint_amortization_v2/nominal/nominal/), with the [all-stage supervisor summary](block_adjoint_amortization_v2/nominal/nominal/supervision.json). Runner SHA256 is `361f186387c0593aba14dd89b91bc32f5a6e136383d0dd8926a2ea5077a0452e`; the native extension SHA256 is `41a78c1cbe365d287ccb5f8b8caf7da810a5bd4818ee78a32ffba133f7477706`. The all-70 strict reference is reused from the accepted v1 reference; every stage records the reference JSON/NPZ checksums and exact stable source, model, rate, prototype, batch API, and native provenance. Only the volatile resource snapshot is excluded from the reused-reference comparison. The supervisor summary and stage artifacts show all nine stages accepted.

Artifacts: `idle_nominal_10us_entry137.json`,
`idle_nominal_10us_structured.json`, `idle_nominal_10us_phase_structured.json`,
`idle_strong_10us_entry411.json`, `idle_strong_10us_phase_structured.json`,
`idle_weak_2us_modes.json`, `idle_gaussian_1us_sideband.json`,
`idle_batch_weak_2us_structured.json`, `idle_sparse_weak_1us.json`,
`idle_dense_weak_1us.json`, `idle_rhs_profile_nominal.json`,
`idle_rhs_profile_dense_nominal.json`, `idle_periodic_nominal_5us.json`, and
`idle_nominal_200us_structured.json` (timing calibration only; pairwise
comparison available), `idle_nominal_200us_tight_dopri.json` (pairwise
comparison only; no 200 us multi-refinement ladder currently planned), and
`idle_nominal_200us_outputs.json` with separate photon, full-density,
all-population, J=1 subset, and named-population NPZ outputs,
`idle_nominal_200us_population_repeats.json`, plus
`idle_nominal_200us_period_krylov_progress.json` and its checkpoint density
artifact (a scoped, failed-acceptance matrix-free period-action pilot), and
`idle_nominal_200us_period_krylov_population_products.json` (checkpoint-only
population analysis).

## Earlier pilots (provisional)

**Status: provisional.** These are short feasibility pilots collected before
the user confirmed other `simulate_all.py` jobs had finished; they remain
separate from the accepted `idle_*` measurements above. The harness has a
best-effort process/PID guard, but command-line access can be restricted, so a
clear guard result is not an idle guarantee. Some earlier runs started with
recorded CPU samples below the 25% gate, but that does not establish that the
workstation remained uncontended for the full run.
Completed comparison artifacts use the 70-state, un-compacted model and record
their own source/extension hashes and resource samples in JSON. The measurements
below keep their artifact-specific settings; timing is not pooled across runs.

## Numerical findings

The completed short ODE reference checks in `ode_weak_2us.json`,
`ode_nominal_1us_entry137.json`, `ode_strong_quarterus_entry411.json`, and
`ode_gaussian_1us_sideband_entry73.json` are marked validated by the harness.
The dopri5 refinement and tsit5 cross-check errors are below the configured
reference-validation gates in each. For example, the largest listed density
refinement difference is `6.71e-10` and the largest listed cross-solver
difference is `1.35e-10`; listed photon differences are below `5e-13`.
Those checks tightened tolerances but did not reduce the initial step; they do
not satisfy the revised reference-refinement protocol. Later `idle_*` reruns
reduce both and check two successive refinements before accepting each
reference. The earlier 5 us nominal and 1 us strong artifact files contain
reference progress but no candidates; their cases were subsequently measured
in the accepted idle reruns above.

On the completed cases, `tsit5` at `reltol=1e-9` passes the recorded density,
population, trace, Hermiticity, positivity, and photon checks. The faster,
looser `1e-7` weak-case candidates fail the density-matrix absolute-error gate
(about `8.5e-7` for dopri5 and `1.2e-6` for tsit5), despite small population
errors. Density accuracy therefore changes which solver settings are
acceptable.

The phase-frame transform agrees with the original-frame ODE reference on each
completed case. It is not a consistent speedup: the nominal 1 us pilot is
slightly faster with phase-frame tsit5, the Gaussian 1 us pilot is faster, and
the strong 0.25 us pilot is slower. Transform preparation alone took about
3.5–5.3 s in the recorded runs and is excluded from per-trajectory medians;
the reuse break-even has not been measured.

The switched EOM Strang split does not yet meet the full-density target. The
static artifact scores the final full density matrix (not its entire saved
trajectory). In `static_dense_weak_1us.json`, even the smallest tested
interval (5 ns) has final-matrix maximum element error `1.41e-5` and photon-yield error `1.74e-6`; the larger
10 ns and 20 ns steps are less accurate. The reduced sparse split pilot in
`sparse_weak_1us.json` also fails the density target at every tested interval;
at 1.25 ns it remains at `1.86e-6`. These are failed acceptance attempts, not
evidence that further refinement cannot converge.

The no-EOM four-generator dense split does pass against its corresponding
no-EOM ODE case (`1.63e-11` final-matrix maximum element error). It validates that limited
static propagation path only; it says nothing about the EOM phase kicks.

## Timing evidence (not final recommendations)

The comparison harness times the full-density ODE solve plus a separate
solver-native photon-integral solve. The common model build and phase-frame
preparation are reported separately. These short runs use only one initial
state, two repeats in most cases, and selected entry phases. Their medians are
illustrative, not portable performance claims.

| Case | `tsit5`, original frame | `tsit5`, phase frame | Recorded result |
|---|---:|---:|---|
| Weak, 2 us, entry 0 ns | 8.77 s | 8.26 s | Both pass at `1e-9`; loose `1e-7` fails density accuracy |
| Nominal, 1 us, entry 137 ns | 9.65 s | 9.34 s | Both pass at `1e-9`; only a small pilot difference |
| Strong, 0.25 us, entry 411 ns | 3.40 s | 3.86 s | Both pass at `1e-9`; phase frame is slower here |
| Nominal Gaussian envelope, 1 us, entry 73 ns | 6.49 s | 5.47 s | Both pass at `1e-9`; one short envelope case |

For context, the weak 2 us dopri5 `1e-9` median was 9.67 s. The static dense
factorization artifact reports 19.98–23.68 s per switch-state factorization
(about 90.1 s total across four factors) and a conservative estimate of about
1.54 GB per factor. Its peak process RSS was about 3.12 GB. Those setup costs
must be included in any reuse comparison; no amortized batch break-even has
been measured. The no-EOM split's 0.24 s propagation median is a different
model and must not be compared as an EOM acceleration result.

## Incomplete attempts and limits

- `failed_attempts.json` records an early static-run shape error; that attempt's
  timings were rejected. The batch/time-axis handling was subsequently fixed.
- `ode_nominal_5us_entry137.json` and `ode_strong_1us_entry411.json` stop after
  model/reference progress and contain no candidate rows. Treat them as
  incomplete, not as failed numerical methods or measured comparisons.
- `setup_weak.json` and the smoke JSON files are setup/smoke checks, not timing
  baselines. Their setup path used different thread settings and should not be
  mixed with the comparison-suite medians.
- The accepted batch test covers three initial states and one entry phase.
  Ten- and 100-state batches and broader entry-phase coverage remain
  unmeasured.
- The accepted RHS/coefficient profile is a fixed-state probe; profile data
  over coherent and excited points along accepted trajectories would help
  explain state-dependent cost.
- The one-period cost gate is measured and rejects the full map under the
  configured 30-minute budget. The map itself, phase-prefix/remainder accuracy,
  and reduced-map support were not tested because construction did not pass the
  gate.

## Scope and remaining coverage

The short-pilot benchmark matrix and implementation checks are complete; the
focused test suite passed (264 passed, 2 skipped). The nominal 200 us
structured run, pairwise tight-reference comparison, separate output-product
suite and repeats, and scoped matrix-free period-Arnoldi pilot are complete.
The fixed-parameter final-time response-map idea remains deferred; no physical pilot is authorized. Late-time slow modes and restarted/block Krylov are also deferred and unauthorized for physical runs.
The adjoint/block pilot source also has no physical results yet. The
short nominal 10 us tolerance sweep and three-setting repeat check are
complete with accepted timing guards. The
historical `idle_nominal_10us_entry137.json` reference is validated, but its
native/API provenance no longer matches the current build. The new sweep
therefore built and validated a fresh four-stage reference. Earlier retry
artifacts stopped at the resource guard before yielding valid candidate rows
and are retained as incomplete attempts. The plan documents the sweep and
full-trajectory error/runtime reporting in
[`RC_PROPAGATION_PLAN.md`](../RC_PROPAGATION_PLAN.md). The nominal 200 us
reference remains pairwise only; a long-duration convergence check is a later
step only if needed to validate the selected production tolerance. Wider
initial-state and entry-phase coverage remains an additional
study. A workload-based periodic-map amortization study remains open. This
report does not claim exhaustive coverage of all propagation methods. Do not
extrapolate the measured three-state batch to 10 or 100 states. The periodic
map was not constructed under the pilot budget; whether a higher budget or a
validated reduced map is worthwhile depends on expected reuse. Keep
constant-envelope and Gaussian-envelope conclusions separate. Production cooling scans remain
out of scope.

The default comparison targets are those in `RC_PROPAGATION_PLAN.md`: density
and population absolute error `<= 1e-7`, trace and Hermiticity residual `<=
1e-9`, minimum eigenvalue `>= -1e-8`, residual-population relative error
`<= 1e-3` when the reference population is at least `1e-6`, and photon-yield
error `<= 1e-6 + 1e-5*abs(reference_yield)`. The broader benchmark campaign
remains incomplete, and this short case does not select a general production
method.

## Short nominal 10 us tolerance sweep and representative repeats

The accepted sweep artifact is
[`idle_nominal_10us_tolerance_sweep_retry2.json`](idle_nominal_10us_tolerance_sweep_retry2.json)
with full saved density arrays in the matching `_densities.npz`. It is marked
`complete`, `timings_accepted`, and `reference_validated`; whole-run peak RSS
was 611 MB. Before-run CPU samples were 1.5–2.6%, available memory was 42.9
GB, and no `simulate_all.py` process was detected. These process checks are a
best-effort guard, not proof of a contention-free host. Each candidate uses
exact switch-segment boundaries, absolute EOM phase, 122 identical saved
times, structured mode, and `dt=0.1 ns` as the initial step. The timing is the
outer full-density solve plus separate native photon-integral solve, including
segment handling and full-density carry; it is not a photon-only or
endpoint-only timing.

The repeat artifact is
[`idle_nominal_10us_tolerance_repeats_retry2.json`](idle_nominal_10us_tolerance_repeats_retry2.json)
and is also marked complete with accepted timing guards. Per-row accepted and
rejected step counts, full-density RHS calls, full-density solve wall time,
separate photon solve wall time, and combined solve-call wall time are
preserved in the sweep JSON; step/RHS counters apply only to the full-density
solve.

The current-native reference used structured dopri5 at
`rtol/atol/dt = 1e-10/1e-12/0.1 ns`, `3e-11/3e-13/0.05 ns`, and
`1e-11/1e-13/0.025 ns`, followed by tsit5 at `1e-11/1e-13/0.025 ns`. Those
four calls took 17.05, 21.44, 26.57, and 24.39 s (89.44 s total). The ladder
is validated: successive density differences were `3.34e-10` and
`9.52e-11`, and the tight cross-solver difference was `3.37e-11`. The tight
reference photon yield is 1.33549187458. The older reference remains a
validated historical baseline, not the scoring source for this current-native
grid.

The primary rows couple `abstol=rtol/100`; the additional controls use fixed
`abstol=1e-10`. The table gives maximum all-saved-time errors; the density
endpoint is also shown because the worst error can occur earlier. Wall times
are one-pass values except for the three repeated rows, where the median of
three complete runs is shown. Repeat ranges are DOPRI5 scaled `1e-8`:
7.489–7.538 s; Tsit5 scaled `1e-8`: 6.685–6.741 s; Tsit5 fixed
`1e-7/1e-10`: 6.604–6.635 s. All nine repeated runs passed the joint gates.
Common model preparation took 9.66 s in the sweep and 9.88 s in the repeat
driver; this setup is excluded from propagation times. The four-stage
diagnostic reference cost (89.44 s total) is also excluded from production
candidate timing.
“Joint gates” means the full density, all-population, photon, trace,
Hermiticity, and positivity criteria pass together. Candidate rows failing
the full-density gates may still pass an output-specific photon criterion.

| Solver | rtol | atol | Wall (s) | rho max, all / endpoint | Population max abs | Population max rel (`ref >= 1e-6`) | Photon abs error | Min eigenvalue | Joint gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| dopri5 | 1e-2 | 1e-4 | 3.34 | 1.31e1 / 7.41e0 | 2.94e-1 | 9.52e4 | 7.02e-3 | -3.80e1 | no |
| dopri5 | 1e-3 | 1e-5 | 3.37 | 1.07e-2 / 6.57e-3 | 2.59e-4 | 5.81e1 | 1.24e-5 | -2.57e-2 | no |
| dopri5 | 1e-4 | 1e-6 | 3.39 | 7.57e-4 / 4.84e-4 | 1.85e-5 | 5.11e0 | 1.34e-6 | -9.07e-4 | no |
| dopri5 | 1e-5 | 1e-7 | 3.48 | 1.07e-4 / 6.81e-5 | 2.04e-6 | 7.08e-1 | 5.02e-8 | -1.10e-5 | no |
| dopri5 | 1e-6 | 1e-8 | 3.84 | 6.83e-6 / 4.73e-6 | 1.80e-7 | 3.40e-2 | 4.52e-8 | 0.00e0 | no |
| dopri5 | 1e-7 | 1e-9 | 5.15 | 5.56e-7 / 4.87e-7 | 1.84e-8 | 2.33e-3 | 1.23e-8 | 0.00e0 | no |
| dopri5 | 1e-8 | 1e-10 | 7.53 | 5.07e-8 / 4.26e-8 | 1.91e-9 | 2.21e-4 | 1.59e-9 | 0.00e0 | yes |
| dopri5 | 1e-9 | 1e-11 | 11.18 | 4.80e-9 / 3.89e-9 | 1.91e-10 | 2.23e-5 | 1.81e-10 | 0.00e0 | yes |
| dopri5 | 1e-7 | 1e-10 | 7.59 | 6.69e-8 / 4.47e-8 | 2.07e-9 | 2.37e-4 | 1.72e-9 | 0.00e0 | yes |
| dopri5 | 1e-9 | 1e-10 | 7.47 | 4.95e-8 / 4.25e-8 | 1.89e-9 | 2.22e-4 | 1.58e-9 | 0.00e0 | yes |
| tsit5 | 1e-2 | 1e-4 | 3.29 | 3.49e2 / 2.22e2 | 7.40e0 | 2.78e6 | 5.94e-1 | -1.03e3 | no |
| tsit5 | 1e-3 | 1e-5 | 3.34 | 2.50e-2 / 1.30e-2 | 4.67e-4 | 1.32e2 | 5.38e-6 | -6.31e-2 | no |
| tsit5 | 1e-4 | 1e-6 | 3.32 | 1.38e-3 / 7.89e-4 | 2.99e-5 | 9.65e0 | 2.51e-6 | -2.12e-3 | no |
| tsit5 | 1e-5 | 1e-7 | 3.37 | 1.25e-4 / 1.04e-4 | 3.42e-6 | 8.76e-1 | 4.39e-7 | -2.96e-5 | no |
| tsit5 | 1e-6 | 1e-8 | 3.58 | 1.36e-5 / 6.77e-6 | 2.40e-7 | 7.14e-2 | 5.52e-8 | 0.00e0 | no |
| tsit5 | 1e-7 | 1e-9 | 4.66 | 7.12e-7 / 6.37e-7 | 2.09e-8 | 3.17e-3 | 3.33e-9 | 0.00e0 | no |
| tsit5 | 1e-8 | 1e-10 | 6.72 | 4.78e-8 / 4.26e-8 | 1.34e-9 | 2.20e-4 | 1.41e-11 | 0.00e0 | yes |
| tsit5 | 1e-9 | 1e-11 | 10.21 | 3.22e-9 / 2.84e-9 | 7.74e-11 | 1.44e-5 | 2.37e-11 | 0.00e0 | yes |
| tsit5 | 1e-7 | 1e-10 | 6.61 | 5.38e-8 / 4.46e-8 | 1.48e-9 | 2.40e-4 | 2.20e-12 | 0.00e0 | yes |
| tsit5 | 1e-9 | 1e-10 | 6.78 | 4.72e-8 / 4.24e-8 | 1.33e-9 | 2.18e-4 | 1.41e-11 | 0.00e0 | yes |

The static error-versus-runtime figure is
[`tolerance_sweep_error_vs_runtime.png`](tolerance_sweep_error_vs_runtime.png),
generated by [`plot_rc_tolerance_sweep.py`](../plot_rc_tolerance_sweep.py).
Point annotations show `-log10(rtol)`; color distinguishes solver and
tolerance policy, marker shape distinguishes endpoint density and relative
population errors, and filled versus open points shows whether all joint
gates pass. The photon target is
`1e-6 + 1e-5*abs(reference) = 1.44e-5`. At `rtol=1e-3`, both photon errors
alone are within that target, but the density trajectories are strongly
nonphysical (minimum eigenvalues `-0.026` and `-0.063`); photon accuracy alone
does not make those outputs usable. At `rtol=1e-6`, both photon error and
physical checks pass, while full-density/population accuracy gates still
fail. On the coupled scaled-abstol ladder, the earliest joint
full-density/population/photon pass is `rtol=1e-8` for both solvers; the
fixed-abstol controls at `rtol=1e-7` also pass all gates. Tsit5
`rtol=1e-8`, `atol=1e-10` took 6.72 s at the three-run median and had
`4.78e-8` max all-time density error, `1.34e-9` max population absolute
error, `2.20e-4` relative error for populations at least `1e-6`, and
`1.41e-11` photon error. The `1e-7` scaled-abstol candidates are faster but
miss the residual-population relative gate (`2.33e-3` DOPRI5,
`3.17e-3` Tsit5). This nominal 10 us sweep does not establish suitable
tolerances for 200 us propagation or other parameter points.

For a density-only view of tolerance behavior and execution cost, see
[`tolerance_convergence_and_cost.png`](tolerance_convergence_and_cost.png),
generated by [`plot_rc_tolerance_convergence.py`](../plot_rc_tolerance_convergence.py).
The first panel plots maximum full-density element error over all saved times
against the requested relative tolerance (loose on the left, tight on the
right), with the paired `abstol=rtol/100` sweep lines and separate fixed-atol
controls. This is a **paired rtol/atol tolerance sweep**, not a timestep-order
fit; changing relative tolerance alone is not inferred from the primary
curve. The second panel shows propagation wall time with min–max bars for the
three repeated points (one-pass measurements elsewhere), and the third shows
the density-error/time tradeoff with joint-pass status. Repeated candidate
trajectories reproduced the original density-error values exactly. The
`3.34e-10` reference-refinement difference shown in panel one measures
agreement between two reference levels; it is not a rigorous error bound or
proof of exact truth. The fixed controls demonstrate absolute-tolerance
sensitivity at the same `rtol`: Tsit5 `1e-7/1e-9` has density error
`7.12e-7` and fails the density gate, while `1e-7/1e-10` has `5.38e-8` and
passes all joint gates in the repeated trials.

For this nominal 10 us full-density-plus-photon workload, `tsit5` with
`rtol=1e-8`, `abstol=1e-10`, and initial `dt=0.1 ns` is the conservative
measured setting: it passes all joint gates at a 6.72 s repeated median.
The fixed-atol `rtol=1e-7` candidate is slightly faster in these repeats and
also passes, but this one-case comparison does not establish a broad solver
or long-duration recommendation.

## Structured upper-triangle RHS kernel probe (partial comparison)

The profile artifact
[`idle_nominal_10us_structured_upper_profile.json`](idle_nominal_10us_structured_upper_profile.json)
compares the existing `structured` RHS with the existing `structured_upper` kernel on
the nominal 70-state model. At the initial, mostly diagonal density, native
instrumented time per RHS call was 38.165 us for `structured` and 63.021 us
for `structured_upper`. At a coherent trajectory midpoint it was 43.001 us
and 61.483 us, respectively. These native counters are separate from
end-to-end wall timing including Python/FFI: that measurement was 41.3 us per
call for `structured` and 64.1–65.9 us for `structured_upper`. The midpoint
RHS outputs agree to maximum absolute difference `3.49e-10` and relative
difference `3.41e-16`. On these probes, the existing alternative kernel is slower in both
instrumented native work and wall time; this does not support a kernel-speedup
claim.

The trajectory comparison
[`idle_nominal_10us_structured_upper_comparison.progress.json`](idle_nominal_10us_structured_upper_comparison.progress.json)
and its checkpoint density artifact have `complete=false` and
`timings_accepted=false`; the resource guard stopped the run after four of six
planned trials. All four completed trajectories pass the recorded accuracy
and physical checks. Raw outer solve timings and guard outcomes are:

| Repeat | RHS mode | Wall (s) | Post-run guard | Timing use |
|---:|---|---:|---|---|
| 1 | `structured_upper` | 9.939 | clear | First matched pair only |
| 1 | `structured` | 7.017 | clear | First matched pair only |
| 2 | `structured` | 6.933 | clear | Unpaired observation |
| 2 | `structured_upper` | 10.369 | failed: CPU 28.4%, 30.7%, 24.1% | Contended; excluded |

The first guard-clear pair gives a limited `structured_upper`/`structured`
ratio of 1.4165. The last upper-kernel trial failed its post-run CPU guard;
available memory was 41.9 GB and no known simulation process was detected.
The checkpoint explicitly accepts only that first pair as a limited
observation; no aggregate median or range is accepted. Although all four
trajectories meet the accuracy gates (maximum all-time density error about
`4.777e-8`, residual-population relative error about `2.202e-4`), this partial
comparison does not justify a general runtime claim. Retain `structured` as
the current choice. The route is stopped with no automatic retry planned.
Neither this single nominal case nor the fixed-state RHS profile establishes
behavior for other state counts or density structures.

## Experimental sparse-Hermitian × dense-density RHS profile

The new opt-in `experimental_structured_sparse_dense` mode combines the upper
Hamiltonian representation with one row-wise sparse-Hermitian × dense-density
product, then extracts the packed upper commutator. This profile is a fixed
70-state nominal-case probe. Its initial/random results and coherent midpoint continuation are
[`idle_nominal_10us_sparse_dense_candidate_profile.json`](idle_nominal_10us_sparse_dense_candidate_profile.json)
and
[`idle_nominal_10us_sparse_dense_candidate_coherent_profile.json`](idle_nominal_10us_sparse_dense_candidate_coherent_profile.json).

| Probe | `experimental_structured_sparse_dense`: native / Python-FFI wall (µs/call) | `structured`: native / Python-FFI wall (µs/call) |
|---|---:|---:|
| Initial density | 32.521 / 37.251 | 38.715 / 44.343 |
| Coherent reference trajectory midpoint | 35.417 / 41.871 | 39.009 / 42.094 |
| Random Hermitian density (coherent-profile continuation) | 34.635 / 42.866 | 38.647 / 41.987 |

Native instrumentation and Python/FFI wall measurements are separate timing
boundaries; component fractions are omitted because the instrumented stage
timers overlap. The initial-density arrays matched exactly. At the coherent
midpoint, candidate and structured RHS differed by maximum absolute
`3.49e-10` and relative `3.41e-16`. At the random Hermitian probe, the
maximum absolute difference was `2.98e-8` and the reported relative
difference was `1.45e-18`. The candidate is faster in native timing for all
three probes; its Python/FFI wall timing is lower at the initial and coherent
probes and higher on the random probe. CPU
samples before and after profiling were 9.4%, 13.3%, 7.8% and 22.6%, 14.1%,
8.8%, respectively; both checks passed the new experiment's 50% cutoff. The
artifacts record the current extension SHA256 and source revision. The fresh
current-build reference is
[`idle_nominal_10us_sparse_dense_reference.json`](idle_nominal_10us_sparse_dense_reference.json)
with saved densities in
[`idle_nominal_10us_sparse_dense_reference_densities.npz`](idle_nominal_10us_sparse_dense_reference_densities.npz).
It uses 122 saved times; successive
reference density differences were `3.341e-10` and `9.518e-11`, and the tight
cross-solver difference was `3.367e-11`. Its final photon yield is
`1.33549187458`. Reference construction took four solver calls; its cost is
not included in candidate timings.

The matched trajectory artifact is
[`idle_nominal_10us_sparse_dense_candidate_comparison.json`](idle_nominal_10us_sparse_dense_candidate_comparison.json),
with full saved density arrays in the matching
[`NPZ`](idle_nominal_10us_sparse_dense_candidate_densities.npz). Each row uses
`tsit5`, `rtol=1e-8`, `atol=1e-10`, initial `dt=0.1 ns`, the same 70-state
model and 10 µs constant-envelope trajectory beginning at 137 ns. Timing is
the full-density solve plus a separate native photon-integral solve, including
switch-segment handling and density carry. Common model preparation was
10.05 s and is reported separately. Three alternating pairs completed:

| Pair | `experimental_structured_sparse_dense` (s) | `structured` (s) | Candidate / structured |
|---:|---:|---:|---:|
| 1 | 6.141 | 6.936 | 0.8854 |
| 2 | 5.986 | 6.965 | 0.8593 |
| 3 | 6.040 | 6.822 | 0.8854 |

The candidate median was 6.040 s (sample SD 0.079 s); `structured` was
6.936 s (sample SD 0.076 s). The median of the paired ratios was 0.8854;
the ratio of the mode medians is 0.8708. Thus this run measured 12.92% lower
median wall time, with pairwise reductions of 11.46–14.07%. All six trials
passed the median-of-three 50% CPU resource gate and the numerical gates. Each had 10,811
accepted steps and 64,907 full-density RHS calls. Against the reference, the
candidate's maximum all-time density error was `4.777e-8`, maximum population
absolute error `1.339e-9`, relative error for populations at least `1e-6` was
`2.202e-4`, and photon-yield absolute error was `1.41e-11`. The candidate and
`structured` outputs agreed to maximum density-element difference
`2.289e-14` and photon difference `8.88e-15`.

This is a measured speedup for this opt-in kernel on one 70-state, nominal
10 µs, 137 ns-entry, full-density-plus-photon workload. It does not cover
other output products, state counts, field/drive settings or the 200 µs
workload. Keep the current RC recommendation `structured` and leave package
defaults unchanged pending broader validation across other cases.
The profile and trajectory artifacts use extension SHA256
`2bfbd8d5059bbd3bca325e170fe90af8cfed64704e17fe2a9a66b0b35d0e27ec` and
source revision `4eea25f8a5ac6818501372f977eabe18fdd9495c`.

### Eight-method kernel matrix: nominal SoA result

The completed [nominal case summary](rc_kernel_matrix_v1/nominal/case_summary.json)
screens eight methods against a fresh strict reference at the same 70-state,
10 µs, 137 ns-entry nominal point, with `tsit5`, `rtol=1e-8`, and
`atol=1e-10`. Every screening row passed its recorded full-trajectory
accuracy checks; all 18 screening and repeat rows passed. `structured` and
four new sparse-dense variants have three timing observations each (initial
screen plus two alternating paired repeats). The other screened modes have
one observation and should not be read as repeated timing results.

| Metric | `structured` | `experimental_structured_sparse_dense_soa` |
|---|---:|---:|
| Full-density plus separate photon trajectory median | 6.678 s (6.670–6.678) | 5.417 s (5.404–5.445) |
| RHS native profile mean | 38.30 µs/call | 28.64 µs/call |

The SoA variant's median full-trajectory wall was 18.9% lower at this one
nominal setting. The native RHS profile was about 25.2% lower; these are
different timing boundaries. Across the matrix, maximum all-time density
error versus reference was `4.82e-8` and maximum photon-yield absolute error
was `1.88e-11`; the recorded resource guards passed. The implementation uses
separate real/imaginary scratch arrays for output arithmetic while retaining
interleaved complex input, so this result makes no SIMD claim. It is an
opt-in implementation and does not change the package default or current RC
recommendation (`structured`). The nominal point alone cannot establish a crossover rule or broader speedup; follow-up points are summarized below.


### Additional detuning points with three matched timing observations

The −20, −5, and +5 MHz cases each have a validated current-build reference and three passing full-trajectory timing observations for `structured` and each of the four sparse-dense variants, all at `tsit5`, `rtol=1e-8`, and `atol=1e-10`. In each case the screening run and two paired repeats passed the full accuracy gate; before/after resource guards passed. Reference and trajectory NPZ checksums match their summaries; all sampled reference and candidate resource guards passed. Across these rows the largest all-time density error was `6.25e-8`, maximum populated-state relative population error `2.64e-4`, and maximum photon-yield absolute error `5.85e-10`. All cases use source revision `4eea25f8a5ac6818501372f977eabe18fdd9495c` and Rust extension SHA256 `41a78c1cbe365d287ccb5f8b8caf7da810a5bd4818ee78a32ffba133f7477706`.

Times are medians of the three observations (seconds) for a full-density solve plus a separate photon solve. Ratios are medians of within-repeat candidate/`structured` wall ratios, so they preserve paired timing rather than dividing aggregate medians.

| Detuning | `structured` | Sparse-dense | Diagonal split | SoA | Single-unpack |
|---|---:|---:|---:|---:|---:|
| −20 MHz | 8.537 | 7.375 (0.864×) | 7.232 (0.845×) | 6.847 (0.801×) | 7.227 (0.847×) |
| −5 MHz | 7.136 | 6.215 (0.870×) | 6.050 (0.846×) | 5.813 (0.816×) | 6.053 (0.848×) |
| +5 MHz | 6.936 | 6.023 (0.869×) | 5.868 (0.846×) | 5.615 (0.812×) | 5.866 (0.847×) |

These matched cases show a consistent advantage for the opt-in sparse-dense variants on the tested points, with the SoA version fastest among these five modes. The complete detuning and power matrix is summarized below; the −100 MHz result required tighter tolerance after the original screen failed and has only two accepted observations. Package defaults and the current RC recommendation (`structured`) remain unchanged pending broader workload validation.
The supporting [profile](rc_kernel_matrix_v1/nominal/profile.json),
[reference metadata](rc_kernel_matrix_v1/nominal/reference.json), and
[trajectory arrays](rc_kernel_matrix_v1/nominal/trajectories.npz) retain the
per-method timing, accuracy, and provenance details.

### −100 MHz detuning crossover: latest passing rows

The [−100 MHz case summary](rc_kernel_matrix_v1/detuning_-100MHz/case_summary.json)
is complete. Its fresh reference passed the Dopri5 refinement, tight Tsit5
crosscheck and physical checks: max density differences were `6.41e-10`
between the two Dopri5 refinement levels and `1.92e-10` on the Tsit5
crosscheck; trace residual was `2.44e-15`, with minimum eigenvalue 0. The
final reference photon yield is `0.0051666472185`. The reference and
trajectory NPZ hashes match the case summary. Both use Rust extension SHA256
`41a78c1cbe365d287ccb5f8b8caf7da810a5bd4818ee78a32ffba133f7477706` and
source revision `4eea25f8a5ac6818501372f977eabe18fdd9495c`. All eight methods
were screened at `rtol=1e-8`, `atol=1e-10`, but those first rows missed the
accuracy gate. Each method then has a passing row at `rtol=1e-9`,
`atol=1e-11`; the latest passing row per method is shown below. Ratios use a
`structured` row from the same case and tolerance: the same paired repeat
where one exists, otherwise the matching accuracy-retry row. Thus these are
fair within-case comparisons, not comparisons to nominal-case timings.

| Mode | Latest passing wall | Compared `structured` wall | Ratio | Latest pass / repeated coverage |
|---|---:|---:|---:|---|
| `structured` | 16.974 s | 16.974 s | 1.000× | paired repeat 2; no repeat 3 |
| `structured_upper` | 25.807 s | 17.079 s | 1.511× | accuracy retry only |
| `expanded_sparse` | 58.037 s | 17.079 s | 3.398× | accuracy retry only |
| `experimental_structured_sparse_dense` | 14.777 s | 17.079 s | 0.865× | accuracy retry only |
| `experimental_structured_sparse_dense_diag` | 14.164 s | 16.974 s | 0.834× | paired repeat 2; no repeat 3 |
| `experimental_structured_sparse_dense_soa` | 13.497 s | 16.974 s | 0.795× | paired repeat 2; no repeat 3 |
| `experimental_structured_sparse_dense_single_unpack` | 14.190 s | 16.974 s | 0.836× | paired repeat 2; no repeat 3 |
| `phase_gauge` | 17.031 s | 17.079 s | 0.997× | accuracy retry only |

The four paired-repeat modes did not complete repeat 3 within the case budget;
the other candidate modes have only their accuracy-retry timing. Do not report
accepted multi-repeat medians for this detuning. All reference and trial
resource guards passed, and the final passing rows meet their recorded
full-density, population, photon and physical gates. The initial looser-
tolerance failures remain in the artifacts rather than being overwritten.
The complete 11-case matrix is summarized below. The −100 MHz case is the exception to full three-observation coverage: its initial 1e-8 screen failed accuracy, and two matching rows passed at 1e-9/1e-11. Do not treat that case’s two-row timing as a three-repeat result.


### Completed 11-case detuning and power matrix

The run manifest and all 11 case summaries report completion. Every case has a validated fresh reference; both reference and trajectory NPZ hashes match the corresponding summary, all summaries use Rust extension SHA256 `41a78c1cbe365d287ccb5f8b8caf7da810a5bd4818ee78a32ffba133f7477706` and source revision `4eea25f8a5ac6818501372f977eabe18fdd9495c`. Resource guards passed for every recorded trial. At the ten cases other than −100 MHz, all eight screening methods and the three-observation `structured`/four-variant timing set passed at `rtol=1e-8`, `atol=1e-10`. At −100 MHz, all eight looser-tolerance screens were retained as accuracy failures; all eight methods then passed at `rtol=1e-9`, `atol=1e-11`, but only two accepted observations were available for `structured` and the four new variants.

The table compares the full-density solve plus separate photon solve. Times are medians of the accepted observations; `n` indicates accepted observations per mode. The reduction is calculated from matched within-trial SoA/`structured` wall-time ratios. The −100 MHz figure has only two accepted matched rows and is explicitly partial.

| Case | `rtol` / `atol` | `n` | `structured` median (s) | SoA median (s) | Paired wall reduction |
|---|---|---:|---:|---:|---:|
| Nominal | 1e-8 / 1e-10 | 3 | 6.678 | 5.417 | 18.9% |
| Detuning −100 MHz | 1e-9 / 1e-11 | 2 | 17.027 | 13.535 | 20.5% (partial) |
| Detuning −20 MHz | 1e-8 / 1e-10 | 3 | 8.537 | 6.847 | 19.9% |
| Detuning −5 MHz | 1e-8 / 1e-10 | 3 | 7.136 | 5.813 | 18.5% |
| Detuning +5 MHz | 1e-8 / 1e-10 | 3 | 6.936 | 5.615 | 18.9% |
| Detuning +20 MHz | 1e-8 / 1e-10 | 3 | 7.059 | 5.779 | 18.1% |
| Detuning +100 MHz | 1e-8 / 1e-10 | 3 | 7.243 | 5.893 | 18.7% |
| Optical drive 0.3Γ | 1e-8 / 1e-10 | 3 | 6.128 | 5.018 | 18.1% |
| Optical drive 3Γ | 1e-8 / 1e-10 | 3 | 8.123 | 6.580 | 19.0% |
| Microwave drive 0.3Γ | 1e-8 / 1e-10 | 3 | 5.932 | 4.857 | 18.1% |
| Microwave drive 3Γ | 1e-8 / 1e-10 | 3 | 9.937 | 7.999 | 19.3% |

Across ten cases with three accepted observations, paired SoA wall reductions range from 18.1% to 19.9%. At −100 MHz the available two-row comparison is consistent with that range, but the changed tolerance and incomplete repeats prevent treating it as an equivalent replicate. These results show a repeatable SoA advantage across this specific 70-state, 10 µs, 137 ns-entry matrix. They do not validate the 200 µs workload, other state counts/output products, or production use. Keep package defaults unchanged and retain `structured` as the current RC recommendation pending broader workload validation.
