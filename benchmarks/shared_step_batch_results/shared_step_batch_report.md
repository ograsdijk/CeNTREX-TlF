# Shared-step Lindblad batching experiment

## Executive summary

The experimental solver propagates complete packed density matrices for a parameter × initial-condition batch, shares one adaptive DOPRI5 controller, and reuses one expanded-sparse structural plan. It agrees with independent solves to the expected integration accuracy. On this machine and these physical examples it **does not beat the existing Rayon-parallel solver**. The shared RHS is faster for some time-dependent coefficient and population-basis cases, but the extra sparse traversal and steps imposed on easy trajectories outweigh that gain in full solves. Keep independent stepping as the default. A production shared mode is not recommended from these measurements; a separate coefficient/RHS optimization may merit investigation for time-dependent fields.

This is a benchmark-only implementation. The production batch/grid API and default independent stepping remain available and unchanged. The only generic integrator change is an optional error-norm/maximum-step hook used by the experimental path; the old path takes the original default.

The prototype and benchmark scripts are preserved on branch `benchmark/shared-step-lindblad-batching`, commit `cd86232`. This `main` copy contains the report and its linked raw results only. Check out that experiment commit to inspect or rerun the implementation; the prototype is not a supported solver API.

## Implementation tested

`solve_lindblad`, `solve_lindblad_batch`, `initial_condition_scan`, `parameter_scan`, and `grid_scan` prepare one system, then invoke Rust `solve_batch_ode` for separate trajectories. Rust reuses workers and Rayon distributes independent DOPRI5/Tsit5 solves. Each worker has its own packed Hermitian state, coefficient workspace, adaptive controller, sparse traversal, saveat interpolation, and solver-native integral state. Fixed-step solvers use separate integration paths. The packed state has `N²` real entries: `N` populations and all Hermitian coherence components. `expanded_sparse` compiles common output/input term topology, while the decomposed Hamiltonian, `basis_terms`, dynamic coefficient expressions, and parameter graph supply runtime coefficients. Structural lowering is shared, but coefficient evaluation and expanded-sparse RHS calls recur for each trajectory and RK stage. A parameter point changes runtime coefficient values; a different initial projector changes only the packed starting state. Parameters that change state count, field-dressed coupling topology, collapse operators, polarization-built coupling matrices, or prepared OBE structure require a separate plan. A velocity entering `z=v*t` and a spatial envelope is coefficient-only and can share the plan.

The experiment accepts contiguous `[parameter][initial][packed_state]` storage and flattens parameter grids into the first axis. At each RK stage, it evaluates the parameter graph/decomposed coefficients once per parameter point and traverses the common expanded-sparse plan across that point's initial states. Each trajectory retains its full density matrix. It computes the normal normalized embedded error separately for every trajectory and controls the batch by their maximum; it records the controlling trajectory and attempted timesteps. Returned `saveat` values use RK interpolation and do not force integration steps. Photon/weighted integrals are appended to each trajectory's ODE state and integrated at RK stages, not summed from returned samples. A maximum-step cap is used only for the narrow-pulse velocity study, in both shared and valid independent references.

Concretely, `rust/src/lindblad/shared_step_experiment.rs` wraps one `solve_dopri5` call around the whole packed batch. `ExperimentalBatchRhs` in `rust/src/lindblad/rhs.rs` owns one `RhsWorkspace` per parameter point; each workspace has its own runtime overrides and coefficient cache, while every point refers to the same prepared sparse plan. The first initial state at each point uses the existing packed expanded-sparse path to refresh coefficients and sparse term values. Remaining initial states at that point reuse those values while traversing the compiled terms. `rust/src/ode/dopri5.rs` asks the experimental RHS for an optional batch error norm and maximum step; ordinary RHS types return `None` and retain their original behavior. The experimental PyO3 entry point in `rust/src/lindblad/python_api.rs` accepts an array shaped `[P,I,N²]` and parameter values shaped `[P,K]`. The Python benchmark flattens independent references to `P×I` rows and passes the same prepared plan, tolerances, outputs, and parameter values to both paths.

For every attempted RK step, the implementation evaluates all stage RHS values, computes one normalized embedded error over the `N²` physical components of **each** trajectory, selects the largest trajectory error, and accepts only if that maximum passes. The trajectory controlling the maximum is counted. Thus trajectories sharing a parameter point also share coefficient evaluation, but they can still need different adaptive steps because their initial populations produce different dynamics. For example, in the one-parameter Q(1) run with 12 projectors, independent capped solves accumulated 10,642 accepted steps (about 887 each); the shared run took 1,652 steps for *all 12*, or 19,824 projector-steps. Its 1.60× faster RHS did not offset the 1.86× increase in projector-step work.

Population projectors are selected from physically relevant ground states. Linearity reconstructs arbitrary initial population mixtures at every saved time and for weighted photon integrals. Explicit coherent density matrices remain supported. The prototype does not build a complete `N²` initial Hermitian basis. It is single-threaded internally; the fair parallel comparator is the existing four-thread Rayon path. A separate cached state-major RHS layout and a SciPy CSR block-diagonal comparison were also measured.

## Environment and reproducibility

AMD Ryzen 7 9800X3D, 8 physical / 16 logical cores; Windows build 26200; Python 3.11.13, NumPy 2.3.1, SciPy 1.16.0; rustc 1.98.0. Rust extension built with `maturin develop --release`; existing Rayon runs used four threads. Solve timings exclude system preparation and compilation. Each case warms the independent and shared paths before timing. R(0) has two repeats in its main matrix; the slower retained-parity solver cases have one. RHS microbenchmarks have two or more repeats. The raw CSVs retain individual observations, including accepted/rejected steps and RHS calls. Timing variation, especially sub-millisecond RHS measurements, limits close rankings. [Environment](benchmark_environment.json), [system metadata](system_r2_full.json), [raw solver timings](solver_timings.csv), [RHS timings](rhs_microbench.csv), [accuracy](accuracy_full.csv), and [straggler data](straggler_stats.csv) provide exact inputs and outputs.

Commands used, with the project virtualenv and release extension:

```powershell
.venv\Scripts\python.exe -m maturin develop --release
.venv\Scripts\python.exe benchmarks/bench_shared_step_batch.py --system r0 --counts 1 4 8 16 32 64 128 --initials 1 4 --kinds rabi_narrow rabi_broad detuning rabi_straggler detuning_straggler --repeats 2
.venv\Scripts\python.exe benchmarks/bench_shared_step_batch.py --system r2_compact --counts 4 --initials 1 --kinds rabi_broad detuning rabi_detuning --repeats 1
.venv\Scripts\python.exe benchmarks/bench_shared_step_batch.py --system r2_full --counts 1 4 --initials 1 --kinds rabi_narrow --repeats 1
.venv\Scripts\python.exe benchmarks/bench_shared_step_batch.py --system q1_velocity_envelope --counts 1 4 8 16 32 64 128 --initials 1 4 8 12 --kinds velocity_dynamic velocity_straggler_dynamic --repeats 1
.venv\Scripts\python.exe benchmarks/bench_shared_step_batch.py --system q1_velocity_envelope --counts 4 16 64 --initials 1 4 --kinds rabi_velocity_dynamic --repeats 1
.venv\Scripts\python.exe benchmarks/bench_shared_step_batch.py --system r2_full --counts 1 4 8 16 32 64 128 --initials 1 4 --kinds rabi_narrow --layout-only --repeats 2
.venv\Scripts\python.exe benchmarks/bench_shared_step_block_diag.py --system r2_compact
.venv\Scripts\python.exe benchmarks/bench_shared_step_coefficient_profile.py
.venv\Scripts\python.exe benchmarks/validate_shared_step_batch.py --system r2_compact --kind rabi_narrow --count 4 --initials 4
.venv\Scripts\python.exe benchmarks/summarize_shared_step_batch.py
```

Additional raw cases are represented in the CSVs. The full 128-point solver matrix on 154-state R(2) would require much longer than the selected full-system checks; solver conclusions for that size are limited to one and four trajectories plus RHS-only scaling. The complete 1–128 parameter-size RHS matrix and a broad R(0) solver matrix were run. Archived `preliminary_*`, `uncapped_*`, and `prevalidated_*` CSVs are exploratory and excluded from the summary. They include an early R(0) dark-state-only choice and an uncapped narrow-pulse velocity baseline; use the main CSVs for conclusions.

## Benchmark systems and physical settings

Fields are `[x,y,z]`, E in V/cm and B in G. Rabi and detuning are angular frequencies. All realistic solves use DOPRI5, expanded-sparse RHS, decomposed Hamiltonian, `reltol=1e-7`, `abstol=1e-9`, photon-integral final output unless a saveat validation is explicitly named. Photon weights are the excited-state indices times Γ. Rabi values below are main-coupling normalizations; scan ranges are specified below.

| System | Transition | N / packed | Opposite parity | E; B | Polarization | Baseline Ω; envelope; interval; `dt` | Relevant population basis |
|---|---|---:|---|---|---|---|---:|
| Two-level microbenchmark | Existing synthetic two-level | 2 / 4 | No | 0; 0 | synthetic | Existing `bench_fixed_step_solvers.py` model | 1 tested |
| R(0) | `R0_F1_3o2_F2` | 65 / 4,225 | No | [0,0,0]; [0,0,0] | Z | 9.802 Mrad/s; flat; 0–10 µs; 0.1 ns | 4 |
| Q(1), compact retained | `Q(1) F1'=1/2 F'=0` | 17 / 289 | Yes | [0,0,200]; [0,0,1e-5] | X | 1.609 Mrad/s; spatial Gaussian for dynamic-velocity adaptation; 0–50 µs; 0.1 ns | 12 |
| R(2), compact retained | `R(2) F1'=7/2 F'=3` | 38 / 1,444 | Yes | [0,0,200]; [0,0,1e-5] | X | 1.926 Mrad/s; flat 2 cm / 184 m/s; 0–108.70 µs; 2 ns | 20 |
| R(2), full retained | same | 154 / 23,716 | Yes | [0,0,200]; [0,0,1e-5] | X | same | 20 |

R(0) reuses the repository's `benchmark_obe.py` physical setup. Q(1)/R(2) use the retained-opposite-parity notebook transition, selectors, nonzero electric field, magnetic field, X polarization, power-derived main Rabi rate, and excited-population photon weights. R(2) uses the example's full 2 cm interaction interval. The Q(1) forward-velocity experiment adapts the repository's time-dependent beam geometry to the full OBE: `z=-5 mm+v*t`, a Gaussian drive centered at 2 mm with 0.2 mm spatial sigma, and `v=0.7–1.3 × 184 m/s` (a ±3σ range for the repository's 10% forward-velocity spread). It is a *different envelope* from the flat-top Q(1) notebook, and is labeled accordingly. It uses `maximum_step=0.2 µs`; all ordinary examples have no maximum-step cap. The full metadata JSON files record exact values and basis indices.

Rabi scans use 0.8–1.2 Ω (narrow) and 0–4 Ω (broad). Detuning uses ±30 MHz for R(0) and ±80 MHz for retained systems, multiplied by `2π×10⁶`. The combined Rabi × detuning grids use 0.5–1.5 Ω and those detuning ranges. `rabi_velocity_dynamic` combines 0.5–1.5 Ω with the Q(1) forward-velocity range. An earlier transverse-velocity label is **Doppler detuning equivalence**, because with the flat envelope transverse velocity only changes optical detuning; it is not independent evidence for velocity batching. The nontrivial velocity results in this report are the Q(1) dynamic-envelope cases.

## RHS-only results

The table gives median `independent RHS time / batched RHS time`; greater than one favors direct batching. These calls include coefficient evaluation and sparse application. The largest direct RHS difference was 1.19e-7 in derivative units, from a changed summation order; the integration accuracy comparisons below are the decisive numerical check. [RHS scaling plot](rhs_speedup.png) and [raw data](rhs_microbench.csv) show all parameter counts 1, 4, 8, 16, 32, 64, 128.

| System; initial projectors | 1 parameter | 16 parameters | 64 parameters | 128 parameters |
|---|---:|---:|---:|---:|
| R(0); 1 | 0.99× | 1.04× | 1.55× | 1.10× |
| R(0); 4 | 0.55× | 0.62× | 0.99× | 0.96× |
| R(2) compact; 1 | 0.96× | 0.96× | 0.94× | 0.99× |
| R(2) compact; 8 | 0.62× | 0.64× | 1.08× | 0.99× |
| R(2) full; 4 | 0.66× | 0.84× | 0.94× | 0.86× |
| Q(1) dynamic velocity; 4 | 1.23× | 1.35× | 1.33× | 1.33× |
| Q(1) dynamic velocity; 12 | 1.60× | 1.63× | 1.84× | 2.68× |

The dynamic case benefits from evaluating time-dependent coefficients once per parameter point for many projectors. Constant-coefficient expanded-sparse cases already have efficient independent worker caches; direct batching often loses. The two-level case sometimes reached ~1.1× at moderate counts, but is only a controller/kernel microbenchmark and is not evidence for a realistic speedup.

The existing Rust RHS profiler separates parameter-graph evaluation, Hamiltonian coefficient filling, and sparse commutator traversal for a representative single trajectory over 200 times. Parameter evaluation plus coefficient filling was ~47% of total RHS time for the Q(1) dynamic spatial field, versus ~1.0% for R(0), ~0.9% for compact R(2), and ~0.1% for full R(2). These are phase measurements of the independent RHS, not a direct decomposition of the batched kernel, but explain why sharing coefficients helps the dynamic case and barely matters for constant-coefficient systems. [Coefficient profile](coefficient_profile.csv) has raw seconds and call counts; the RHS microbenchmarks above measure the whole batched kernel separately.

The cached state-major layout (`[state][trajectory]`) was compared with trajectory-major storage. For R(0), state-major / trajectory-major RHS speed was 0.48× with one projector and 1.09× with four at 16 points, falling to 0.46× and 0.89× at 128. For full R(2), the corresponding numbers were 0.54×/1.20× and 0.41×/0.61×. Thus the candidate sometimes helps a small multi-projector batch, but the prototype's trajectory-major layout wins overall and avoids transposition cost. [Layout data](rhs_layout_microbench.csv) retain conversion times.

An explicit SciPy CSR block diagonal of repeated same-parameter Liouvillians was 1.3–1.7× faster *than repeated SciPy CSR matvec calls* on measured R(0)/compact R(2) initial batches, but duplicates matrix memory linearly: compact R(2) used 170 kB for one copy and 1.36 MB for eight. It requires assembly and is not a like-for-like win over the Rust expanded-sparse kernel, especially when coefficients change with parameter or time. [Block-diagonal data](block_diag_microbench.csv) include assembly, CSR matrix–matrix multiplication, and memory. No giant matrix was integrated with the adaptive controller.

## Full solver results

Speedups below are `independent / shared`; greater than one would favor shared. The serial and four-thread Rayon baselines both use the *existing* independent production solver for ordinary scans. [R(0) scaling](solver_scaling_r0.png), [Q(1) velocity scaling](solver_scaling_q1_velocity_envelope.png), [retained R(2) scaling](solver_scaling_r2_compact.png), and [raw timings](solver_timings.csv) show throughput and step counts.

| Case | Parameters × projectors | Shared seconds | Speedup vs independent serial | Speedup vs Rayon |
|---|---:|---:|---:|---:|
| R(0), narrow Rabi | 128×1 | 0.86 | 0.45× | 0.20× |
| R(0), broad Rabi | 128×1 | 2.00 | 0.40× | 0.17× |
| R(0), detuning | 128×1 | 9.98 | 0.34× | 0.12× |
| R(0), narrow Rabi + basis | 128×4 | 4.96 | 0.24× | 0.11× |
| R(0), detuning + basis | 128×4 | 66.60 | 0.15× | 0.05× |
| R(0), Rabi × detuning | 64×1 | 4.21 | 0.44× | 0.18× |
| Retained R(2), compact, narrow Rabi | 16×1 | 21.32 | 0.88× | 0.30× |
| Retained R(2), compact, narrow Rabi + basis | 4×4 | 26.42 | 0.68× | 0.23× |
| Retained R(2), compact, broad Rabi | 4×1 | 5.00 | 0.67× | 0.32× |
| Retained R(2), compact, detuning | 4×1 | 10.16 | 0.61× | 0.33× |
| Retained R(2), full, narrow Rabi | 1×1 | 12.80 | 0.91× | 0.95× |
| Retained R(2), full, narrow Rabi | 4×1 | 45.42 | 0.98× | 0.36× |

R(0) population-basis-only, parameter-only, and parameter × basis cases all favor independent solves. The independent Rayon path gains from assigning complete trajectories to workers; the prototype has no internal parallelism. R(2) compact shows a smaller single-thread gap on narrow Rabi but still no win, and four-thread Rayon remains substantially faster. The 154-state case has only one- and four-trajectory full-solver checks, so no large-system solver crossover is claimed. The measured crossover for these full-solver workloads is **beyond the tested range or absent**; R(0) was measured through 128 parameter points and four projectors, Q(1) through 128 and 12, and RHS-only retained R(2) through 128 and eight.

The simplest requested case was **Rabi rate only, one identical initial state per point, constant laser envelope**. Even there, R(0) at 128 points took 0.86 s shared versus 0.39 s independent serial and 0.18 s Rayon; full retained R(2) at four points took 45.42 s versus 44.63 s and 16.55 s. For R(0), the shared controller took 84 steps versus about 74 per independent trajectory, so timestep divergence was modest. With constant coefficients, however, the existing independent RHS cache leaves little coefficient work to share, and the full batched RK/controller work still costs more. No larger full R(2) solve was measured, so the result does not establish an absolute impossibility of a crossover.

### Dynamic forward velocity and combined scans

For the 0.2 mm spatial pulse, the uncapped production adaptive solver can jump entirely over the pulse. At 207.7 and 223.4 m/s it used only seven steps and produced essentially zero photons, while capped independent solves gave ~0.2975 and ~0.2855 photons. The shared batch happened to see those pulses because other velocities caused intermediate steps. This is a pulse-resolution issue, not evidence of greater accuracy from a shared controller. [Velocity diagnostic](velocity_diagnostic.csv) and [uncapped accuracy](accuracy_unbounded_velocity.csv) retain the comparison. Valid velocity timing ratios below therefore use independent *single-trajectory capped* solves from the experimental entry point. Four-thread capped parallel data are also recorded for selected counts; uncapped production Rayon timings remain in the raw CSV but are not an accuracy-valid comparator for this narrow pulse.

| Q(1) case | Parameters × projectors | Shared seconds | Speedup vs capped independent serial | Speedup vs capped independent four-thread |
|---|---:|---:|---:|---:|
| Forward velocity, full ±3σ spread | 128×1 | 2.62 | 0.87× | 0.33× |
| Forward velocity + basis | 128×4 | 7.96 | 0.71× | 0.24× |
| Forward velocity + 12 projectors | 128×12 | 20.00 | 0.72× | not measured |
| Rabi × forward velocity | 64×1 | 1.19 | 0.91× | 0.33× |
| Rabi × forward velocity + basis | 64×4 | 3.83 | 0.66× | 0.24× |

The Rabi × detuning and Rabi × forward-velocity grids are flattened into general parameter rows with no parameter-specific nested integrator. Detuning × basis is represented by the R(0) measurements above. Doppler-only transverse velocity in a flat field belongs to the detuning category. No field amplitude/polarization scan that requires rebuilding the coupling matrix was forced into the shared sparse plan.

## Accuracy and output semantics

The independent adaptive solver is the reference for ordinary cases. The Q(1) dynamic-envelope case uses capped independent singleton solves as explained above. Full packed saveat tests used seven returned times, including the final time; integrals were also compared at all seven times. [Accuracy CSV](accuracy_full.csv) contains individual population and coherence maxima.

| Validation | Maximum population difference | Maximum coherence difference | Weighted-integral difference | Population-mixture reconstruction |
|---|---:|---:|---:|---:|
| Two-level | 2.73e-9 | 1.93e-9 | 6.46e-10 | 1.53e-9 |
| R(0), broad Rabi, 4×4 | 8.84e-8 | ≤8.96e-8 | 2.65e-8 | 9.58e-8 |
| Q(1) dynamic velocity, 4×4 | 4.79e-11 | 8.44e-8 | 5.32e-12 | 1.07e-7 |
| R(2) compact retained, 4×4 | 3.96e-11 | 4.12e-7 | 1.15e-12 | 4.50e-7 |
| R(2) full retained, 1×1 | 0 | 0 | 0 | not tested |

Two different explicit R(0) density matrices with initial coherences, batched across four Rabi points, also agreed in the full packed state to 8.96e-8. A broad Rabi range includes zero drive. Multiple identical parameter points and final-only modes were covered by the smoke/solver cases. Because the whole density matrix is propagated, the reported coherence differences reflect numerical integration, not a reduced physical model. The larger absolute packed differences in R(2) are within the expected accumulation from the specified tolerances over ~28,500 accepted steps. A linear combination of basis responses reconstructs saved density matrices; the same linearity applies to final populations and weighted photon responses. `saveat` remains a request for output samples, never a fixed integration grid.

## Shared-step stragglers

The controller records the maximum per-trajectory error at every attempted step. In the R(0) one-outlier Rabi case, the 10× Rabi trajectory controlled **100%** of attempts. At 64 points, the batch took 686 accepted steps, versus about 74 steps per ordinary independent trajectory inferred from the aggregate independent count; each easy trajectory therefore did roughly 9.3× the stage work it needed alone. The one-outlier detuning case was more severe: the far-detuned trajectory controlled **100%**, with 3,843 common steps versus ~74 for an ordinary trajectory, about **52×** easy-trajectory step work. Those batches took 2.23 and 12.27 s shared versus 0.22 and 0.35 s independent serial. [Attempted-step histories](steps_r0_rabi_straggler_p64_i1_r0.npz) and [controller frequencies](straggler_stats.csv) are saved.

For the Q(1) forward-velocity tail case, the fast-velocity outlier controlled ~55.4% of attempted steps and two trajectories controlled at least one step; pulse times spread the fine-step region across the interval. With the common 0.2 µs cap, 128 velocities took ~1,655 accepted shared steps, while valid independent solves averaged ~1,654. The cap dominates accepted-step count in this diagnostic, so the velocity timing loss here mostly reflects RHS/controller cost across the full batch. Without a cap, the production baseline sometimes misses the pulse, preventing a meaningful uncapped straggler ratio. The Rabi and detuning outlier cases show the shared-controller penalty much more clearly.

## Recommendation

| Workload | Recommendation from these measurements |
|---|---|
| Small batches | Keep independent adaptive solves. Setup and controller overhead dominate. |
| Large Rabi or detuning scans | Keep Rayon independent solves; no full-solver crossover through 128 parameter points on R(0). Broad detuning and isolated hard points strongly penalize shared steps. |
| Forward velocity with spatial fields | Resolve narrow pulses explicitly in either method; then keep independent stepping. The time-dependent batched RHS itself can gain 1.3–2.7× when many population projectors share a velocity. |
| Population-basis scans | Use basis reconstruction conceptually for repeated initial-population fits, but propagate basis responses with independent adaptive solves for these systems. |
| Parameter × population basis | The tested shared full solver loses despite coefficient reuse. Independent basis solves followed by linear reconstruction are the practical choice. |
| Medium R(0) and compact retained R(2) | Keep the existing four-thread Rayon architecture. |
| Full 154-state retained R(2) | One- and four-trajectory solver checks plus RHS-only scaling are insufficient to claim a large-batch crossover. Do not enable a shared production mode based on it. |

A future opt-in `batch_stepping="shared"` and `initial_mode="population_basis"` API could be considered only after a materially faster kernel or a workload with genuinely shared timestep histories is demonstrated. This task stops at the benchmark prototype, raw results, and report. It does not change `rho0`, `rho0_batch`, or existing `solve_lindblad`/scan callers.
