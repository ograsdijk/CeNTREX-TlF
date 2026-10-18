# Rotational-cooling propagation plan

> **Study-plan authority (2026-10-11):** apparatus, velocity scans and production
> scope are defined in the sibling repository's
> [RC study plan](../../CeNTREX-SPB-testing/reports/rotational_cooling/PLAN.md).
> This file supplies library solver evidence. Its fixed-duration, constant-envelope
> study assumptions and earlier production-grid discussion are superseded by that
> plan. Do not use them to launch or estimate the SPB cooling study.

Updated 2026-10-11. The required product is the **full density matrix, including
all coherences**, for the uncompacted RC model. Use the implemented Rust forward
solver with the D+S+U kernel. Adjoint observable responses do not provide this
product and are outside this RC workflow.

## Current method and settings

- `backend="rust"`, `hamiltonian_representation="decomposed"`.
- `solver="tsit5"`.
- `execution_mode="structured_sparse_dense_diag_soa_single_unpack"`.
- Use `reltol=1e-9`, `abstol=1e-10`, per the user's runtime preference.
  This is a speed/accuracy compromise: the historical nominal comparison gave
  a worst complex-entry relative error of 9.44% above the 1e-6 magnitude cutoff,
  versus 0.649% with `abstol=1e-11`. Thus the selected setting has not passed
  the earlier 1% entrywise target. The nominal measured runtime fell from
  36.71 s to 29.76 s (about 19% less); these are benchmark-field observations,
  not a production-field accuracy or timing guarantee.
- Request `output="full"`; use `output_when="final"` when only the final density
  is needed. Save intermediate matrices only when the analysis needs them.
- Keep `qn_compact=False`. Preserve every modeled level and coherence.
- Prepare once and reuse the prepared model across runtime detuning/Rabi values
  and initial states. Use bounded parallel independent trajectories for scans,
  with a Windows main guard and an explicit total worker budget.

These are explicit RC settings, not changes to package defaults. The stable
kernel name and its experimental compatibility alias execute identical
arithmetic. See [solver usage](../README_OBE_SOLVER.md), the
[implementation record](LINDBLAD_OPT_IN_IMPLEMENTATION_PLAN.md), and the
[current handoff](LINDBLAD_SESSION_HANDOFF.md).

## Completed evidence

The cross-system nominal screen and repeated RC finalist comparisons at 10 and
200 microseconds are complete. The D+S+U kernel was the fastest measured forward
finalist for all seven tested RC output products. Three accepted observations
per method/output were recorded in the nominal long-duration comparisons.

| Full final density | `structured` median | D+S+U median |
|---|---:|---:|
| 10 microseconds | 2.222 s | 1.800 s |
| 200 microseconds | 45.729 s | 36.601 s |

The measured full-density wall reduction is about 19–20%. These are single-initial-
state, constant-envelope, nominal-case timings including final output, not
whole-scan estimates. The historical kernel names in the artifacts begin with
`experimental_structured_sparse_dense`; preserve their provenance.

Sources: [10 microsecond results](rc_propagation_results/cross_system_long_finalists_v1/suite10us_v1/rc/10us/result.json),
[200 microsecond results](rc_propagation_results/cross_system_long_finalists_v1/suite_long_v1/rc/200us/result.json),
and [cross-system report](rc_propagation_results/cross_system_report.md).
The report's lower historical queue/status sections are stale; use completed
JSON manifests for the corresponding stages.

The accepted 200 microsecond full-density timings above used `reltol=1e-9`,
`abstol=1e-11` after tighter retries. The original `1e-7/1e-9` setting failed the
historical positivity gate; `1e-8/1e-10` also missed that gate (minimum eigenvalue
about -3.17e-8 against a -1e-8 cutoff). Preserve those results. The newer
entrywise coherence comparison below independently supports `1e-9/1e-11` for
the user's 1% target at the agreed magnitude cutoff; do not replace measured
errors with a generic strict-gate pass/fail label.

The native forward aliases and adjoint APIs are implemented and correctness-
tested locally (recorded Lindblad suite: 313 passed, 2 skipped). Native adjoint
performance is unmeasured and is irrelevant to this full-density workflow.
Static `dense_eig` cannot propagate the coherent EOM/switched RC generator.

## Production geometry and fields

**RC has no applied electric field: `E = np.zeros(3)` V/cm.** The earlier
200 V/cm field belongs to the numerical benchmark only and must not be used
for production rotational cooling.

Use right-handed laboratory axes **+x east, +y north, +z up**:

- Molecular beam: east to west, along -x.
- Laser arrives from the north: propagation along -y (north to south).
- Magnetic field: unshielded Earth field at Lemont/Argonne, retaining all three
  components rather than replacing it with a small axial placeholder.
- Existing site estimate: `B = [-0.01327, +0.19433, -0.48889]` G, magnitude
  about 0.52627 G. Source: [saved Earth-field record](../reports/r2_f4_earth_field/data/earth_field.json),
  WMM2025 evaluated for 2026-10-06, latitude 41.7183, longitude -87.9817,
  approximate elevation 0.2 km. This is a modeled field, not an in-lab measurement.

The laser polarization must lie in the x-z plane. The historical optical X
polarization is geometrically compatible, but a propagation direction alone
does not specify its polarization. Express all optical and microwave
polarizations in the same ENU coordinates. Rebuild Hamiltonians, dressed
states, couplings, branching and Rabi normalization at the production fields.
Do not reuse matrices or normalization from the E=200 V/cm benchmark. With
the tilted magnetic field, lab-z mF is not an exact state label; preserve a
consistent tracked/identified input-state mapping.

## Physical model

- Build P(2), F1'=3/2, F'=1 plus `MicrowaveTransition(1, 2)` and
  `MicrowaveTransition(2, 3)` using current APIs and the production geometry above.
- Preserve hyperfine, relevant excited parities and all coherences;
  select initial states by their recorded physical labels and state mapping.
- Keep `qn_compact=False`. The historical benchmark has 70 states with
  `Jmax_X=5`, `Jmax_B=3` and opposite excited parity retained. Reuse these
  truncations as a starting point, but verify the actual production retained
  basis and state count after rebuilding at E=0 and the full Earth vector.
- Existing drive configuration: coherent optical EOM at 1.6 MHz with beta=3.8,
  and 1 MHz microwave polarization switching with 50% duty cycle and a
  quarter-period (250 ns) offset. Preserve the recorded phase convention.
- Energies, detunings and Rabi rates are angular frequencies in rad/s; E is V/cm
  and B is Gauss. Convert physical powers using the rebuilt main couplings.
  User-selected laser powers are 150, 250, 350 and 450 mW; microwave powers
  are 300, 500 and 700 mW. Use physical-power conversion for each rebuilt
  coupling; do not substitute the benchmark Gamma controls.
- Intended interaction duration is at least 200 microseconds. The historical
  nominal case was measured at that duration under benchmark fields, not the
  production fields. Do not attribute its timing/accuracy to the corrected model.
- Transverse-velocity distributions and population ratios are applied later.
  Keep constant amplitudes for the present response-grid model; explicit
  spatial envelopes require their own model and checks when introduced.

## Per-initial-state final-density output

For each parameter point, start independently in every designated initial
field-dressed ground state i, with `rho0_i = |i><i|`. Propagate its complete
density matrix and save `rho_final_i` at the interaction endpoint. Initial
population ratios are chosen later; they are not needed to launch these basis
trajectories. Independent trajectories may share preparation and run in parallel.

For any later population mixture with weights p_i at the same parameter point,
compute `rho_final = sum_i p_i * rho_final_i`. This follows from linear Lindblad
evolution and retains the coherences generated from each initial population.
Store one matrix per initial-state label and parameter point, together with the
common output-basis label/index mapping, fields, drives, duration, envelope,
entry phase, switch schedule, tolerances and native/model provenance. Keep input
and output state axes distinct. Do not average or discard the individual results.
The exact input-state inventory must be recorded by labels; the benchmark K=20
subset and K=70 all-model batch are different inventories, not interchangeable.

## Production detuning scan and existing settings

User-selected physical-power axes:

- Laser power: **150, 250, 350, 450 mW** (`[0.15, 0.25, 0.35, 0.45]` W).
- Microwave power: **300, 500, 700 mW** (`[0.3, 0.5, 0.7]` W).
- The two microwave drives vary **independently**: each of J=1<->2 and
  J=2<->3 uses its own 300/500/700 mW axis. Run the full Cartesian product:
  4 laser powers x 3 MW12 powers x 3 MW23 powers x 161 detunings =
  **5,796 parameter points per initial state**. These are per-drive powers,
  not a combined total to divide between the microwave transitions.
- Define optical beam dimensions/profile and microwave field/intensity calibration
  from the apparatus inputs when converting watts to Rabi rates. The per-drive
  Rabi references must use the production model's own main couplings.

The production scan needs a fine detuning grid near the cooling resonance.
The user rejected the sparse benchmark detunings and the +/-100 MHz extent.
Do not use the benchmark stress-test points as the production scan grid.

User-selected detuning grid: **-20 to +20 MHz inclusive, in 0.25 MHz steps**
(161 points per initial state and drive setting). Generate it as
`detuning_MHz = 0.25 * np.arange(-80, 81)` and convert with
`detuning_rad_s = 2*np.pi*1e6*detuning_MHz` at the solver boundary. The grid
choice is settled. Do not use the earlier 0.5 MHz step or +/-10 MHz proposal.
The user's instruction not to launch physical runs remains in force.

The production geometry/fields above supersede benchmark field inputs. The
following values describe benchmark provenance and existing drive settings,
not a license to copy those fields into production. Population ratios remain
free and are not a prerequisite for per-initial-state runs:

- Historical RC benchmark fixture only: E=[0,0,200] V/cm, B=[0,0,0.01] G, constant amplitudes,
  optical EOM 1.6 MHz with beta=3.8, microwave switching 1 MHz with the recorded
  quarter-period offset. The recent 200 microsecond validation uses 125 ns entry
  time and nominal optical/microwave Rabi rates of Gamma each.
- Historical benchmark context only: the [11-case kernel catalog](compare_rc_kernel_matrix.py) specifies detunings
  0, +/-5, +/-20 and +/-100 MHz at nominal strengths, plus separate optical and
  microwave Rabi-rate controls at 0.3Gamma and 3Gamma on resonance. These are
  one-axis controls, not a Cartesian product. Its original duration is
  10 microseconds and entry time 137 ns; do not silently carry those historical
  timing inputs into the newer 200 microsecond/125 ns configuration.
- The [cross-system regime catalog](cross_system_regime_screen.py) additionally
  defines selected detuning/optical and optical/microwave corners. It is a method
  benchmark catalog, not automatically an instruction to run all those cases.

The prior documents specify benchmark points and scan axes, but do not define a
complete input-state inventory. The production detuning range and spacing are
now fixed above. Transverse-velocity distributions will come from separate
simulations later; they are not a prerequisite for the RC response runs. Select the existing
model configuration and the finer production grid chosen by the user when
assembling the manifest; apply the corrected production fields and ENU geometry. Do not
substitute stress-test detunings or require population weights to be fixed. No physical run is launched by this plan update.

## Transverse velocities and Doppler averaging

The transverse-velocity distribution will be obtained from separate simulations
later. Do not request or assume its width before running the RC response grid.
The user selected the wider +/-20 MHz range and finer 0.25 MHz steps precisely
to support this later averaging. Save **unaveraged** full final density matrices
for every initial state and detuning; both population weights and transverse-
velocity weights remain free for post-processing. No velocity distribution is
applied during these response runs.

For the current constant-field, constant-amplitude, fixed-duration model, if
transverse velocity changes only optical Doppler detuning, reuse the detuning
response grid instead of propagating an extra independent velocity axis.
With v_parallel positive along optical propagation, use
`delta_eff = delta_lab - k_optical dot v` (angular frequencies). Equivalently,
if velocity is defined positive toward the laser, the Doppler contribution is
positive. Record the propagation direction and sign convention explicitly.
For the specified laser direction, `k_optical = [0, -k, 0]`, so the optical
Doppler contribution is `+k * v_y`, where v_y is positive northward. The ideal
east-to-west longitudinal beam velocity contributes zero optical Doppler shift
for this perpendicular geometry.
Use the UV optical frequency/wavelength for k, not an IR seed frequency or an
unexamined utility default. `utils.detuning.velocity_to_detuning` returns rad/s
with the toward-laser-positive convention.

For each initial state i, integrate the full endpoint matrix over the velocity
distribution at the Doppler-shifted detuning. Combine populations afterward:
`rho_final = sum_i p_i * integral f(v) * rho_final_i(delta_eff(v)) dv`.
Use the same output-state basis and physical phase convention when combining
complex matrices; transform from any velocity-dependent rotating frame first
where needed. Do not average populations alone and discard coherences.

The 161-point grid specifies the requested detuning response range. During later post-processing, check coverage for every shifted detuning used
in the simulation-derived velocity average. Distinguish molecular-frame response detuning from a requested
laboratory detuning axis. Do not extrapolate, clamp to edge values, zero-fill
missing density matrices, or truncate/renormalize velocity tails silently.
If a full lab-frame +/-20 MHz averaged spectrum requires response values
outside the grid, propose explicit padding or a supported smaller averaging
window before adding physical runs; the user has not authorized a broader grid.
Later, check interpolation/velocity-quadrature convergence within the 1%
accuracy budget; solver tolerance alone does not validate the averaging procedure.
These are post-processing checks, not missing inputs for the initial RC runs.

If transverse motion changes beam intensity/path, entry phase, fields or
interaction duration, a single detuning convolution is insufficient: propagate
the affected velocity/path cases explicitly. If microwave Doppler shifts are
included, use each microwave's own propagation vector and update its detuning;
reusing a one-dimensional optical-detuning response is then conditional on
those microwave shifts being negligible or already accounted for.

## Switching and propagation

Each microwave independently switches between its two polarization states with
a **1 MHz square-wave cycle frequency** (1 microsecond period, 50% duty cycle).
Their relative phase is **pi/2**, equivalent to a 250 ns offset. This interleaves
the two switches and produces four equal 250 ns joint configurations per cycle:

| Time modulo 1 microsecond | MW 1 selector | MW 2 selector |
|---|---:|---:|
| 0 to 250 ns | 1 | 1 |
| 250 to 500 ns | 1 | 0 |
| 500 to 750 ns | 0 | 0 |
| 750 to 1000 ns | 0 | 1 |

Selectors 0/1 name the two polarization choices for each microwave, not on/off
microwave power. This ordering uses `sin(2*pi*1e6*t)` and
`sin(2*pi*1e6*t + pi/2)`, matching the existing `switch_values` implementation.
Individual square-wave periods are 1 microsecond; the combined configuration
changes every 250 ns. Absolute entry time may start inside any of these quarters.

End and restart forward propagation at every known microwave switching boundary,
carrying the complete complex density matrix between segments. Freeze each
polarization amplitude to its correct segment value; do not rely on a square
wave's endpoint convention. Preserve absolute physical time throughout so the
EOM phase is continuous and correct at nonzero entry times.

The existing segmented forward driver in `benchmarks/bench_rc_propagation.py`
provides the reference pattern. `dt` is an initial adaptive step, not a maximum
step or a request to land on switching edges. Keep the full-density segment
handoff in both correctness comparisons and timing boundaries.

## Completed benchmark-field tolerance validation

**Provenance:** completed checks below used E=[0,0,200] V/cm and
B=[0,0,0.01] G. Preserve that distinction from the production geometry.
**User direction: no more validation/benchmark runs.** Reuse the existing
solver evidence and user-selected `1e-9/1e-10` tolerances; do not add another
production-field spot-check as a prerequisite. Production remains unlaunched
until the user explicitly instructs starting it.

The [elementwise tolerance report](rc_propagation_results/coherence_tolerance_check_v2/REPORT.md)
checks nominal, detuning -20/+20 MHz, optical 3Gamma and microwave 3Gamma at
200 microseconds. All five pass with `reltol=1e-9`, `abstol=1e-11` at the
42 stored tight-reference times per case, including the endpoint. The earlier
accuracy target was at most 1% complex entrywise relative error for **every density-matrix entry,
including coherences**, with reference magnitude at least 1e-6. The user accepted
that magnitude cutoff. Report absolute errors below it; the largest measured
value is 1.38e-8. This slightly exceeds the suggested companion 1e-8 absolute
bound, so that extra bound is not passed universally. Population or matrix-norm
agreement does not substitute for the entrywise criterion.

Worst relative errors are 0.649% nominal, 0.118% at -20 MHz, 0.101% at +20 MHz,
0.405% optical 3Gamma, and 0.273% microwave 3Gamma. Tight Tsit5 and Dopri5
references agree within 0.041% on entries above the cutoff. Candidates retain
all 802 switch-boundary/end-point matrices, but only 42 reference snapshots
were stored; do not claim relative validation at every intervening time.

The initial `1e-8/1e-10` setting reached 9.44% coherence error above the cutoff.
The nominal `1e-9/1e-10` control remained at 9.44%. The accepted nominal
`1e-9/1e-11` run took 36.71 s versus 29.76 s for that control (single observations,
about 23% more time), and remained faster than the strict reference. Preserve
original results and physical diagnostics in the
[initial report](rc_propagation_results/full_density_tolerance_check_v1/REPORT.md).
No clipping or renormalization was applied.

## Remaining work for the simulation

1. Assemble the run manifest from the existing settings/catalog above and the
   per-initial-state contract. Record the exact input-state labels and parameter
   points. Save full final matrices individually; mixture ratios remain free.
   Use current APIs and fresh Hamiltonians.
2. Reuse completed checks and the selected solver/tolerances. Do not launch
   additional tolerance ladders, cross-solver checks or field-validation runs;
   the user explicitly rejected further validation.
3. Use the recorded laser powers 150/250/350/450 mW and microwave powers
   300/500/700 mW independently for each microwave drive. Preserve per-state
   full-density output at all 5,796 parameter combinations.
4. Estimate scan cost from measured representative full-density trajectories,
   including segment handoffs and requested saving. Separate model/preparation
   time, propagation, output storage and peak memory. Match worker budgets and
   avoid competing heavy simulation jobs when collecting timings; record
   resource conditions (established median background CPU ceiling: 50%).
5. Run production scans only when separately requested. This update selects the
   forward workflow and does not launch benchmarks or production simulations.

## Historical investigations and deferred methods

The [earlier RC results](rc_propagation_results/README.md) retain the short
coupled-tolerance sweep, 11-case SoA detuning/power matrix, output-specific
comparisons, and failed or inconclusive prototypes. Those results are historical
and must not be overwritten or attributed to the newly built extension.

The earlier [acceleration proposal](RC_ACCELERATION_IMPLEMENTATION_PLAN.md)
and [cross-system suite plan](RC_CROSS_SYSTEM_BENCHMARK_PLAN.md) contain deferred
implementation language superseded by the implementation record and completed
manifests. They are background investigations, not the current RC solver choice.

Do not requeue scalar adjoints, shared-block prototypes, phase-frame/split
propagation, explicit period maps, Arnoldi/Krylov pilots or late-time slow-mode
methods for this workflow. Saving forward results per initial state for later
mixture reweighting is the required output contract, not a deferred solver method. The period-Arnoldi pilot failed
its tested full-density gates and was slower than direct propagation. Other
routes remain separate investigations requiring a concrete reason and scope;
none may replace full density with scalar or population-only output.

## Artifacts and delivery

- Driver/reference pattern: `benchmarks/bench_rc_propagation.py` and the
  cross-system RC fixture in `benchmarks/cross_system_fixtures.py`.
- Completed evidence: `benchmarks/rc_propagation_results/README.md`,
  `cross_system_report.md`, and the linked machine-readable results.
- Future simulation outputs must preserve state-label mappings, model/runtime
  parameters, entry phase and switch schedule, solver/tolerances, saved times,
  source/native hashes, and timing boundaries. Store complete requested density
  matrices; derive populations and coherences from those matrices.
- Preserve failed trials and historical artifacts. No commits, publishing,
  package-default changes or new physical runs are part of this plan update.
