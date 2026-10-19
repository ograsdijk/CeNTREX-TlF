# CUDA library integration validation

Implemented in the public single, batch and scan APIs. CPU defaults remain unchanged.

## CUDA ODEs

FP64 real/imaginary packed density, sparse Hamiltonian products, structured decay, tiled kernels, CUDA graphs, optional common waveform precomputation, shared off-diagonal Hamiltonians, and prescribed wing-step schedules. Both fixed-step Tsit5 and low-storage CK54 are opt-in. Photon integrals use the RK stages. Additional density snapshots can be saved separately from population/photon histories.

Accuracy criterion: each complex density entry with reference magnitude >=1e-6 has <=1% relative error; smaller entries have <=1e-6 absolute error. Trace error <=1e-6 and minimum eigenvalue >=-1e-7. All rows below pass.

| Model | Solver | Cases | Step (ns) | Worst entry/coherence relative error | Near-zero max absolute error |
| --- | --- | ---: | ---: | ---: | ---: |
| RC | fixed_tsit5 | 483 | 4.46429 | 0.925104% | 6.59e-09 |
| R2 EOM | fixed_tsit5 | 60 | 2 | 0.0594397% | 4.56e-10 |
| R2 EOM | fixed_ck54 | 60 | 1.95 | 0.973733% | 8.03e-09 |

RC short validation covers all 161 detunings and three initial J mixtures over 146–151 microseconds, using the saved strict CPU reference. R(2) covers 60 EOM trajectories against the saved 0.5 ns reference. These checks reuse archived results; no CPU ODE benchmarks were rerun. The parameter lowering is general and has no hard-coded RC slot indices, geometry, switches, or optical waveform.

## Full RC trajectories

Nine cases per velocity: three initial J mixtures and UV detunings -20, 0, +20 MHz. Gaussian microwave envelopes and 1 MHz polarization switching with quarter-period offset. Population history has 201 samples, density saved at field exit and after 25 excited-state lifetimes of dark evolution. Reference: saved, converged 512-substep RK4 trajectories from the earlier RC benchmark.

| Forward speed (m/s) | Worst complex relative error | Near-zero max absolute error | Population-history max absolute difference |
| ---: | ---: | ---: | ---: |
| 150 | 0.343422% | 5.39e-12 | 7.84e-07 |
| 50 | 0.25602% | 4.01e-10 | 1.09e-06 |
| 250 | 0.31572% | 3.32e-10 | 5.01e-07 |

Tsit5 uses 4.4642857 ns nominal steps and 20% longer steps outside center ±0.8 sigma. Switching, requested outputs and schedule boundaries clip steps. These values are validated for the saved fixtures, not universal defaults.

## Static reconstruction

Archived R(2) 40-level eigensystem, 20 initial states × 21 times (420 full complex density matrices): maximum difference 1.05e-15 from the saved-factor reference evaluation. No eigendecomposition was repeated for this check. Full packed/complex reconstruction now runs on CUDA; reusable propagators retain their factor tensors on-device and expose clear_cuda_cache(). Initial-state linear solves and eigendecomposition remain on CPU.

## Automated checks

- GPU environment: 41 tests passed across test_cuda.py and test_dense.py.
- CPU environment: 314 tests passed, 17 skipped across tests/lindblad (before adding two further CUDA input-validation tests and three optional static GPU cases; these additional tests passed in the GPU environment).
- CPU-only import does not load CuPy or PyTorch.
- Ruff and git diff whitespace checks passed.

Coverage includes independent complex Liouvillian exponentials, segmented DOP853 switching checks, all 16 scalar helpers, arbitrary initial coherences, scanned phase/frequency boundaries, selected lower-triangle entries, multiple chunks, zero-span solves, analytic photon decay, prescribed schedules, dark sinks, empty sample arrays, bounded Cartesian reconstruction, and persistent CPU-worker sessions with GPU reconstruction.

Windows CuPy/PyTorch CUDA DLLs must load in compatible order. The CUDA ODE entry point imports installed PyTorch before CuPy; when initializing these packages externally, import PyTorch first.

Reproduction and raw diagnostics:

- validate_cuda_library.py / cuda_library_validation.json
- validate_cuda_library_full_rc.py / cuda_library_full_rc_validation.json
- validate_cuda_library_static.py / cuda_library_static_validation.json

Validation scripts read archived benchmark fixtures (and the RC prepared context in the sibling testing repository). They write only local benchmark diagnostics and do not change or launch production scans.

## Further GPU algorithm experiments (2026-10-18)

[Full report and raw data](cuda_algorithm_results/REPORT.md) cover exact density-coordinate reduction,
matrix-free CF4/Magnus exponential actions, and complete periodic propagation maps. The structured
reduction prototype reduces RC runtime by about 12%; CF4/Taylor8 reduces R(2) runtime by about 19%
relative to CK54. Both promising candidates pass full-transit complex-density checks against the
existing strict references, including dark decay. Periodic-map construction is unfavorable for the
few distinct initial mixtures at each parameter point. CF4 remains a benchmark prototype.
Exact coordinate reduction and periodic cycle maps are now opt-in library APIs:
`CudaSolverOptions(density_reduction=True)` and
`prepare_periodic_lindblad_propagator(...)`. Existing defaults are unchanged.
See the usage and limitations in `README_OBE_SOLVER.md`.

Integration validation: `validate_cuda_periodic_reduction.py` and
`cuda_algorithm_results/production_reduction.json` / `production_periodic.json`.
The public reduction API retains numerical agreement with the original solver;
its 1024-trajectory 1 us end-to-end timing includes support construction and shows
about 9% less runtime. The periodic API is checked at three full-length forward
velocities (150/184/250 m/s), with nonzero entry phase and fractional cycles,
against direct full-density CUDA evolution. The RC validation explicitly removes
the two Gaussian envelope factors in a copied archived IR to make a genuinely
periodic model; the original Gaussian recipe is unchanged.
