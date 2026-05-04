# RF Ramsey simulator - Phase A + V3b/V4/V5/V7 perf report

**Goal**: benchmark the Jmax=6 RF Ramsey scan on this PC with a smaller
segmented-grid fine timestep, `dt_fine = 0.05 us` (50 ns), and update the
recommended fast path.

**Result**: V3b fixes the V3a static-projection fidelity problem. A follow-up
`basis_dt` convergence sweep improves the recommended scan setting to
**V3b K=24, basis_dt=10 us, 8 workers**: the 41-point omega_rf scan runs in
**4.49 min**, with single-trajectory fidelity **0.999864** and five-point
full-basis anchor max survival error **1.90e-05**.

## Reference configuration

| Parameter | Value |
|---|---|
| Jmax | 6 (basis size 196) |
| dt_fine (segmented grid) | 0.05 us |
| n_steps (per trajectory) | 138,335 |
| Trajectory | z = -2 -> +2 m at v_z = 184 m/s |
| DC E-field | symmetric plateau 2 -> 30 -> 2 kV/cm, ramps at z = +/-1.5 m |
| RF coils | magnetic, in x, at z = +/-1.25 m, B = 0.113 G (calibrated pi/2) |
| RF carrier | 119.64 kHz (Tl spin-flip resonance), phi1 = 0, phi2 = +pi/2 |
| Initial state | adiabatic ancestor of \|J=1, mJ=-1, m1=-1/2, m2=-1/2> at 30 kV/cm |
| Cached reference elapsed | **44.47 min / trajectory** |
| Reference survival | 0.027722 |

The reference `Psi_final` lives in
[`benchmarks/_cache/reference_jmax6_dt50ns.npz`](benchmarks/_cache/reference_jmax6_dt50ns.npz).

## Variants benchmarked

Each variant runs the same trajectory; the parallel variants additionally run
a 41-point omega_rf scan over +/-2 fringe spacings (+/-147 Hz) around the
resonance.

| Variant | Trajectory | Speedup vs V1 | Fidelity | norm^2 | max population error | 41-pt scan | Scan speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v1** baseline (full N=196) | 44.47 min | 1.00x | 1.000000 | 1.0000 | 0.00e+00 | ~30.4 h (est.) | 1.00x |
| v3a_K16 (truncated, K=16) | 10.5 s | **254.79x** | 0.401140 | 0.6607 | 5.78e-01 | - | - |
| v3a_K24 (truncated, K=24) | 31.9 s | **83.66x** | 0.916004 | 0.9631 | 1.06e-01 | - | - |
| v3a_K32 (truncated, K=32) | 60.2 s | **44.30x** | 0.735672 | 0.9631 | 1.18e-01 | - | - |
| v3a_K48 (truncated, K=48) | 80.8 s | **33.03x** | 0.997835 | 0.9992 | 5.73e-03 | - | - |
| **v3a_K24_par8** (rec. fast) | 31.5 s | **84.7x** | 0.916004 | 0.9631 | 1.06e-01 | **3.89 min** | **~469x** |
| **v3a_K48_par8** (rec. accurate) | 79.1 s | **33.7x** | 0.997835 | 0.9992 | 5.73e-03 | **9.10 min** | **~200x** |
| v3b_K24 (tracked, K=24) | 43.6 s | **61.2x** | 0.999505 | 0.9998 | 2.17e-03 | - | - |
| v3b_K32 (tracked, K=32) | 72.6 s | **36.7x** | 0.996949 | 0.9999 | 1.44e-03 | - | - |
| v3b_K48 (tracked, K=48) | 90.6 s | **29.4x** | 0.980180 | 1.0000 | 1.31e-03 | - | - |
| **v3b_K24_par8** (rec. accurate) | 43.9 s | **60.8x** | 0.999505 | 0.9998 | 2.17e-03 | **4.17 min** | **~437x** |
| **v3b_K24_basis10_par8** (rec. converged scan) | 1.36 min | **32.7x** | 0.999864 | 1.0000 | 6.40e-04 | **4.49 min** | **~406x** |
| v3b_K48_basis1 (high-accuracy single trajectory) | 8.74 min | **5.09x** | 0.999999 | 1.0000 | 1.54e-05 | 16.84 min | **~108x** |
| v4_krylov (full basis) | timed out >60 min | slower than V1 | - | - | - | not run | - |
| v5_numba (full basis) | 40.20 min | **1.11x** | 1.000000 | 1.0000 | 4.55e-12 | not run | - |
| v7_cupy (full basis GPU) | 14.24 min | **3.13x** | 1.000000 | 1.0000 | 4.94e-07 | not run; ~9.7 h projected | - |

Notes:
- **Fidelity** = squared phase-insensitive overlap with the truth `Psi_final`.
  1.0 means a perfect match up to global phase.
- **norm^2** is the norm of the test final state. Values below 1 indicate
  amplitude leaked outside the truncated K-dimensional subspace.
- **max population error** is the largest single-component bare-basis
  population error.
- The "Scan speedup" compares against the estimated sequential full-basis
  baseline: `41 * 44.47 min = 30.4 h`. I did not run V1's full 41-point scan.

## Variant details

### V1 - Baseline (full propagator, segmented grid, sequential)

The current production code uses `propagate_midpoint` on the full N=196 basis,
with a segmented adaptive `t_grid` and single-shot exact unitaries across the
constant-H stretches. With `dt_fine = 0.05 us`, the segmented grid has 138,335
steps and the full reference trajectory took 44.47 min on this PC.

### V2 - Process-parallel scan (cloudpickle + spawn)

`scan.run_scan(..., n_workers=N)` spawns worker processes via
`multiprocessing.spawn`, serializing the `RamseyRFConfig` once with
cloudpickle. Each worker pins BLAS to 1 thread so multiple workers can use
separate cores. V2 has no single-trajectory speedup, but it is what makes the
41-point scan practical when combined with the truncated propagator.

### V3a - Static truncated propagator

V3a pre-projects the Hamiltonian onto a K-dimensional subspace built once from
`H_high` on the 30 kV/cm plateau, selecting eigenvectors with largest summed
overlap to the bare J=1 sublevels. The decomposed truncated propagator
[`propagator_truncated.propagate_midpoint_truncated_decomposed`](ramsey_rf/propagator_truncated.py)
pre-projects the 7 Hamiltonian components once, then each timestep rebuilds
the KxK Hamiltonian as a weighted sum before `eigh(K)`.

#### Convergence vs K

| K | Norm preserved | Fidelity |
|---:|---:|---:|
| 16 | 66.1% | 0.401 |
| 24 | 96.3% | 0.916 |
| 32 | 96.3% | 0.736 |
| 48 | 99.9% | 0.998 |

The K=32 fidelity dip remains present at 50 ns. K=48 is still the accurate
static truncation choice; K=24 is useful only for fast survey scans.

### V3b - Adiabatic-tracked truncated propagator

V3b tracks the relevant K-dimensional dressed subspace along the trajectory.
The implementation uses the high-field J=1-overlap subspace as the physical
seed, tracks it down to the low-field entrance, then follows the DC-field
dressed basis on a coarse `basis_dt = 50 us` grid. During propagation, the
coefficient vector is transferred between neighboring tracked bases and the
decomposed Hamiltonian components are reprojected only when the active tracked
basis changes.

This removes the V3a K=32 pathology and makes a smaller subspace viable:

| K | Norm preserved | Fidelity | max population error |
|---:|---:|---:|---:|
| 24 | 99.98% | 0.999505 | 2.17e-03 |
| 32 | 99.99% | 0.996949 | 1.44e-03 |
| 48 | 100.00% | 0.980180 | 1.31e-03 |

The initial K=48 tracked result had low population error but worse full-state
fidelity, consistent with phase error from the coarse moving-basis
approximation. The `basis_dt` convergence sweep confirms that this is mostly
a basis-tracking resolution issue:

| K | basis_dt | Trajectory | Fidelity | max population error | survival error |
|---:|---:|---:|---:|---:|---:|
| 24 | 50 us | 0.75 min | 0.999505 | 2.17e-03 | 1.02e-05 |
| 24 | 10 us | 1.36 min | 0.999864 | 6.40e-04 | 3.23e-07 |
| 32 | 1 us | 8.35 min | 0.999788 | 1.39e-04 | 1.34e-04 |
| 48 | 10 us | 2.09 min | 0.999962 | 1.23e-04 | 5.58e-05 |
| 48 | 1 us | 8.74 min | 0.999999 | 1.54e-05 | 5.45e-08 |

Five exact V7/CuPy full-basis anchors were also built at
`F_RES + [-2, -1, 0, +1, +2] * 73.6 Hz`. Scan confirmation against those
anchors gave:

| Setting | 41-pt scan | Max anchor survival error | RMS anchor survival error |
|---|---:|---:|---:|
| K=24, basis_dt=10 us | 4.49 min | 1.90e-05 | 1.27e-05 |
| K=24, basis_dt=50 us | 3.51 min | 1.79e-04 | 8.89e-05 |
| K=32, basis_dt=1 us | 14.84 min | 5.80e-04 | 3.17e-04 |
| K=48, basis_dt=1 us | 16.84 min | 2.97e-06 | 1.51e-06 |

For production scans, K=24 with `basis_dt=10 us` is the best current
speed/accuracy point. K=48 with `basis_dt=1 us` is the best reduced-basis
agreement with the full trajectory, but its 41-point scan misses the 15 min
scan-practicality target.

### V4 - Sparse Krylov hybrid

V4 is implemented as
[`propagator_krylov.propagate_midpoint_krylov_hybrid`](ramsey_rf/propagator_krylov.py):
short active steps use sparse `scipy.sparse.linalg.expm_multiply`, while long
inactive steps stay on the exact dense `eigh` path. This is the intended
hybrid from the original plan, but it is not viable for this Hamiltonian at
50 ns timesteps.

The full `v4_krylov` trajectory did not finish within a 60 min timeout. A
prefix benchmark on this PC gave:

| Prefix steps | Elapsed | Per step | Krylov steps | Dense eigensolves |
|---:|---:|---:|---:|---:|
| 10 | 26.7 s | 2.67 s | 9 | 1 |
| 50 | 145.7 s | 2.91 s | 49 | 1 |
| 100 | 294.9 s | 2.95 s | 99 | 1 |

At ~2.95 s per fine Krylov step, the 138,335-step trajectory projects to
well over 100 hours. The likely cause is the large spectral width of the
full molecular Hamiltonian: even at 50 ns, `expm_multiply` must resolve fast
field-free phases. V4 should not be used as a full-basis replacement here
without moving to an interaction-picture formulation or combining it with a
reduced basis.

### V5 - Numba rotate/apply kernel

V5 is implemented as
[`propagator_numba.propagate_midpoint_numba`](ramsey_rf/propagator_numba.py).
It keeps the same dense full-basis SciPy `eigh` as V1, but replaces the
per-step eigenbasis rotation
`V.conj().T @ Psi -> phase multiply -> V @ tmp` with a Numba-compiled loop.
Numba is optional; the benchmark was run with `uv run --with numba`.

The result is accurate but only modestly faster:

| Variant | Trajectory | Speedup vs V1 | Fidelity | max population error |
|---|---:|---:|---:|---:|
| v1 baseline | 44.47 min | 1.00x | 1.000000 | 0.00e+00 |
| v5_numba | 40.20 min | 1.11x | 1.000000 | 4.55e-12 |

This confirms the original expectation: at N=196 and 138,335 steps, dense
eigensolve time dominates. V5 is a small full-basis improvement, but it does
not change the production recommendation relative to V3b.

### V7 - CuPy GPU dense eigensolve

V7 is implemented as
[`propagator_cupy.propagate_midpoint_cupy`](ramsey_rf/propagator_cupy.py).
It keeps the same full-basis midpoint algorithm as V1: assemble dense `H(t)`,
diagonalize with `cupy.linalg.eigh`, rotate/apply on the GPU, and copy
`Psi_final` back to NumPy only at the end. CuPy is optional and is not added
to the project dependencies or `uv.lock`.

The CUDA workstation test used:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Driver CUDA | 13.1 (`nvidia-smi`), CuPy reports driver 13010 |
| CuPy package | `cupy-cuda13x[ctk]` via `uv run --with` |
| CuPy version | 14.0.1 |
| CUDA runtime | 13000 linked to CuPy, 13010 locally installed |
| Device compute capability | 12.0 |

The first CuPy smoke test with bare `cupy-cuda13x` failed because the
optional CUDA component DLLs (`nvrtc`/`nvJitLink`) were not present. Running
with `cupy-cuda13x[ctk]` fixed that, but on Windows with `uv --with` the
NVIDIA component wheels live in a transient overlay path. The V7 module adds
those `nvidia/cu13/bin/x86_64` DLL directories before importing CuPy so
`cupy.show_config()` and a 32x32 Hermitian `cupy.linalg.eigh` smoke test pass.

Measured result:

| Variant | Trajectory | Speedup vs V1 | Speedup vs V5 | Fidelity | norm^2 | max population error |
|---|---:|---:|---:|---:|---:|---:|
| v7_cupy | 14.24 min | 3.13x | 2.82x | 1.000000 | 1.000000 | 4.94e-07 |

The 41-point sequential GPU scan was not run. The plan allowed the scan if
single-trajectory V7 beat V5 by at least 2x, which it did, but one full-basis
V7 trajectory already takes 14.24 min, while the complete V3b K=24 41-point
CPU scan takes 4.17 min. A literal sequential V7 scan would therefore project
to `41 * 14.24 min = 9.7 h`, so it is dominated for scan work.

## Recommendation

For the 41-point omega_rf scan at `dt_fine = 0.05 us`:

> Use **V3b K=24, `basis_dt=10 us`, `n_workers=8`**: fidelity 0.999864,
> max population error 6.40e-04, five-anchor max survival error 1.90e-05,
> and **4.49 min** instead of the estimated 30.4 h full-basis sequential scan.

The old static fallback, **`v3a_K48_par8`**, remains usable but is now slower
and less accurate for the tested trajectory: 9.10 min at fidelity 0.997835.

The previous V3b default, **K=24, `basis_dt=50 us`**, remains a good fast
fallback at 3.51-4.17 min per scan, but the 10 us setting is the better
production recommendation because it improves both state fidelity and
full-basis scan-anchor agreement at only a small scan-time cost.

For full-basis truth/reference runs on the CUDA workstation, **`v7_cupy`** is
now the fastest exact path tested: 14.24 min vs 40.20 min for V5 and 44.47 min
for V1. It should not replace V3b for scans unless a future batched GPU scan
shares Hamiltonian work across frequencies.

## What's not in Phase A

| ID | Variant | Status | Expected gain |
|---|---|---|---|
| V6 | complex64 reduction | not yet | 1.5-2x per eigh, with accuracy cost to quantify |
| V7b | Batched GPU scan | not yet | Only worth revisiting if the scan can run points concurrently or reuse Hamiltonian assembly |

Given the 4.49 min V3b K=24, `basis_dt=10 us` scan at 50 ns, the cleanest
next investment is promoting the tracked propagator behind a simulator config
flag with an explicit `basis_dt` control.

## Reproducing these numbers

PowerShell on this PC needed UTF-8 console output for the benchmark script's
Unicode prints:

```powershell
$env:PYTHONIOENCODING='utf-8'

# 1. Build the 50 ns cached reference (44.47 min on this run)
uv run python examples\ramsey_rf\benchmarks\reference.py --force

# 2. Run V1 + V3a single-trajectory variants
uv run python examples\ramsey_rf\benchmarks\perf_bench.py `
    --variants v1 v2 v3a_K16 v3a_K24 v3a_K32 v3a_K48 `
    --report examples\ramsey_rf\benchmarks\_cache\perf_50ns_single.md

# 3. Run the parallel-scan headline numbers
uv run python examples\ramsey_rf\benchmarks\perf_bench.py `
    --variants v3b_K24_par8 --scan --n-workers 8 `
    --report examples\ramsey_rf\benchmarks\_cache\perf_50ns_v3b_k24_scan.md
uv run python examples\ramsey_rf\benchmarks\perf_bench.py `
    --variants v3a_K48_par8 --scan --n-workers 8 `
    --report examples\ramsey_rf\benchmarks\_cache\perf_50ns_k48_scan.md
uv run python examples\ramsey_rf\benchmarks\perf_bench.py `
    --variants v3a_K24_par8 --scan --n-workers 8 `
    --report examples\ramsey_rf\benchmarks\_cache\perf_50ns_k24_scan.md

# 4. Run V5 with optional Numba
uv run --with numba python examples\ramsey_rf\benchmarks\perf_bench.py `
    --variants v5_numba `
    --report examples\ramsey_rf\benchmarks\_cache\perf_50ns_v5_numba.md

# 5. Run V7 with optional CuPy on the CUDA workstation
uv run --with 'cupy-cuda13x[ctk]' python examples\ramsey_rf\benchmarks\perf_bench.py `
    --variants v7_cupy `
    --report examples\ramsey_rf\benchmarks\_cache\perf_50ns_v7_cupy.md

# 6. Run V3b basis_dt convergence and scan-anchor confirmation
uv run python examples\ramsey_rf\benchmarks\v3b_basis_dt_sweep.py `
    --k 24 32 48 `
    --basis-dt-us 100 50 25 10 5 2.5 1 `
    --report examples\ramsey_rf\benchmarks\_cache\v3b_basis_dt_convergence.md `
    --csv examples\ramsey_rf\benchmarks\_cache\v3b_basis_dt_convergence.csv
uv run --with 'cupy-cuda13x[ctk]' python examples\ramsey_rf\benchmarks\v3b_basis_dt_sweep.py `
    --build-v7-anchors `
    --anchor-cache examples\ramsey_rf\benchmarks\_cache\v7_scan_anchors_50ns.npz
uv run python examples\ramsey_rf\benchmarks\v3b_basis_dt_sweep.py `
    --scan-selected --n-workers 8 `
    --csv examples\ramsey_rf\benchmarks\_cache\v3b_basis_dt_convergence.csv `
    --anchor-cache examples\ramsey_rf\benchmarks\_cache\v7_scan_anchors_50ns.npz
```

CPU hardware: this PC, 8+ CPU cores. V7 hardware: CUDA workstation with RTX
5070 Ti, driver CUDA 13.1.
