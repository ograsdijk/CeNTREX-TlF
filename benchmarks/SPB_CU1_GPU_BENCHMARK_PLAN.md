# CU1 GPU benchmark under concurrent CPU load

Authorized GPU-only run. The user's other CPU simulations remain untouched;
the notebook-method CPU baseline and independent CPU density reference are deferred.

## Fixed system

- R(1), F1'=5/2, F'=2, E=[0,0,150] V/cm, B=[0,0,0.001] G.
- Opposite excited parity retained: 26 levels (12 J=1, ten excited, four sinks).
- Equal initial population over all twelve J=1 states, no initial coherences.
- 80 mW input, 13 prism-ordered passes, 10% loss per pass; summed intensities.
- Intensity Gaussian sigma: 0.75 mm along the molecular beam, 7.5 mm vertically.
- Pass centers 3.5..45.5 mm in 3.5 mm steps; height normalization preserves power.
- 184 m/s, 50 mm trajectory, 271.739 us; 0.5 MHz X/Z switching; no phase EOM.
- Doppler sign alternates at spatial midpoints; all discontinuities clipped.
- Freeze the reference scripts' explicit optical carrier, 1.103e15 Hz, and dipole
  2.6675506e-30 C m. This overrides the newer package default intentionally.

The builder follows `scripts/cleanup_spb_doppler.py` / `cu1_3cm_scan.py` from
`yyhshdu1/CeNTREX_state_cleanup_new`, commit
`117071b8be3a3f1b85a09b0046a9cd16891e9530`, while retaining the user-requested
uniform twelve-state mixture rather than the scripts' eight-state SPB mixture.
No velocity convolution/ensemble weighting is needed to measure solver speed;
each trajectory is integrated independently and its parameters are recorded.

## Stages

1. Smoke test GPU lowering and density conservation on a short driven interval.
2. Full-transit Tsit5 references at 1 and 0.5 ns for twelve edge/central cases.
   Check full complex density at the first, middle, last pass and the endpoint.
3. Screen Tsit5 and CK54 at 2, 4, 8 ns; refine CK54 to 1/0.75 ns if necessary.
   Require <=1% relative density-entry
   error for reference magnitude >=1e-6; <=1e-6 absolute below the floor, trace
   <=1e-6 and minimum eigenvalue >=-1e-7. Also check remaining J=1 population
   with allowance max(1% of reference remaining population,1e-6).
4. Screen tiled/direct RHS, common-waveform precomputation and exact reduction
   at batch 1024. Check kernel variants against same-step tiled full densities.
5. Confirm selected configurations on 60 full-transit cases: detuning
   17/19/20.2/22 MHz, vx=-1/-0.5/0/0.5/1 m/s and height=0/4.25/8.5 mm.
6. Retain current tiled Tsit5 as a GPU control and the fastest screened option
   for each pattern; avoid repeating clearly slower candidates over the full grid.
   Full-transit final-population timings use three randomized rounds at batch
   1024 and one batch-512 scaling pilot, over identical grids of 1024 distinct
   parameter combinations. Two patterns distinguish shared-envelope scans from
   varying-height scans. A single scaling pilot is not a repeated timing claim.

GPU references establish convergence, not independent CPU verification. Report
that limitation until the CPU reference can run without competing simulations.
The original notebook method for the later CPU baseline is current-library
Dopri5 / expanded_sparse / decomposed, dt=2 ns, rtol=1e-7, atol=1e-9, Rust
parallelism; it is not the archived 0.2.5 native binary.

## Timing scope

CUDA events bracket graph-based integration. Total public-call wall time includes
setup/capture, integration, output sampling and transfer; internal setup and host
graph-submission times are also retained. CUDA event spans can include idle gaps
if CPU submission is starved, so they are not pure sums of kernel execution times.
CPU load is recorded per call. Warmup/screening results and repeated full-transit
results are labeled separately. Only the benchmark process has below-normal
priority and single-threaded BLAS; no other process is altered.

Driver: `bench_spb_cu1_gpu.py --stage fixture|accuracy|refine|tune|confirm|performance|report`.
`run_spb_cu1_gpu_followup.py` queues confirmation, performance and report generation;
`pipeline.json` and `run.log` record its current stage and progress.
Small JSON results go in `spb_cu1_gpu_results/`; density references remain in the
ignored `.cache/benchmarks/spb_cu1_gpu` directory. No production solver changes.

## Potential follow-up: factor the transverse Gaussian envelope

Concrete test plan: [SPB_CU1_GPU_FACTORIZATION_PLAN.md](SPB_CU1_GPU_FACTORIZATION_PLAN.md).

The current `varying_height` benchmark varies molecular trajectory position `y0`,
not the laser beam width: sigma_y stays fixed at 7.5 mm. All thirteen passes
have y-center zero and the same vertical width. The Rabi envelope therefore
factorizes exactly:

```
Omega(t, y0) = Omega(t, 0) * exp(-y0**2 / (4*sigma_y**2))
```

If sigma_y is also varied at fixed input power, relative to the on-axis reference
waveform at sigma_ref, the constant multiplier becomes

```
sqrt(sigma_ref/sigma_y) * exp(-y0**2 / (4*sigma_y**2))
```

This assumes unchanged longitudinal profiles, pass powers and input power, with
identical vertical centers and widths across passes. Pass-specific vertical
centers/widths generally prevent this simple factorization.

The current CUDA path classifies the full optical coefficient as variable when
y0 varies. It then evaluates the full thirteen-Gaussian sum per trajectory/RK
stage and selects separate off-diagonal Hamiltonian storage. Possible follow-up:
precompute the common on-axis waveform, compute one static attenuation factor
per trajectory, and apply it within optical Hamiltonian assembly or the RHS.
Preserve separate diagonal detuning/Doppler terms and polarization switches.
Partial coefficient sharing could generalize this beyond this particular model.

Compare against both the current mixed-height batch and grouping by the four
distinct y0 values; grouping may lose throughput through smaller batches and
additional graph submissions. Validate complex densities at in-field checkpoints
and final output against the unchanged unfactored model. Measure waveform
evaluation, Hamiltonian assembly, RHS memory traffic, setup, CUDA event time and
full-call wall time separately. The earlier approximately 50-second full-transit
difference was an extrapolation from the short screen, not a profiled attribution
to Hamiltonian assembly alone.

This is a documented candidate, not an implemented optimization. Preserve the
running benchmark unchanged as its baseline.
