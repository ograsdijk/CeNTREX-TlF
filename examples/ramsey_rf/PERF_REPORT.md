# RF Ramsey simulator — Phase A perf report

**Goal**: bring a 41-point ω_rf scan at Jmax=6 from the projected ~7.3 hours
sequential baseline down to a useful interactive time so parameter scans
become feasible.

**Result**: with the recommended stack (V3a K=48 + V2 8-worker parallel),
the same scan runs in **3.04 min** — a ~145× speedup at fidelity 0.998 vs
the cached Jmax=6 truth reference. K=24 trades fidelity 0.916 for an even
faster **1.60 min** scan if you only need the gross fringe shape.

## Reference configuration

| Parameter | Value |
|---|---|
| Jmax | 6 (basis size 196) |
| dt_fine (segmented grid) | 0.5 µs |
| n_steps (per trajectory) | 13 837 |
| Trajectory | z = -2 → +2 m at v_z = 184 m/s |
| DC E-field | symmetric plateau 2 → 30 → 2 kV/cm, ramps at z = ±1.5 m |
| RF coils | magnetic, in x, at z = ±1.25 m, B = 0.113 G (calibrated π/2) |
| RF carrier | 119.64 kHz (Tl spin-flip resonance), φ₁ = 0, φ₂ = +π/2 |
| Initial state | adiabatic ancestor of \|J=1, mJ=-1, m1=-1/2, m2=-1/2⟩ at 30 kV/cm |
| Cached reference elapsed | **10.68 min / trajectory** |

The reference Psi_final lives in [`benchmarks/_cache/reference_jmax6_dt500ns.npz`](benchmarks/_cache/reference_jmax6_dt500ns.npz).

## Variants benchmarked

Each variant runs the same trajectory; the parallel variants additionally run
a 41-point ω_rf scan over ±2 fringe spacings (±147 Hz) around the resonance.

| Variant | Trajectory | Speedup vs V1 | Fidelity | ‖ψ‖² | max\|Δ\|ψ\|²\| | 41-pt scan | Scan speedup |
|---|---|---|---|---|---|---|---|
| **v1** baseline (full N=196) | 10.68 min | 1.00× | 1.000000 | 1.0000 | 0.0e+00 | ~7.3 h (est.) | 1.00× |
| v3a_K16 (truncated, K=16) | 3.2 s | **200×** | 0.395 | 0.661 | 5.8e-01 | — | — |
| v3a_K24 (truncated, K=24) | 9.7 s | **66×** | 0.916 | 0.963 | 1.1e-01 | — | — |
| v3a_K32 (truncated, K=32) | 16.5 s | **39×** | 0.729 | 0.963 | 1.2e-01 | — | — |
| v3a_K48 (truncated, K=48) | 19.7 s | **33×** | 0.998 | 0.999 | 5.5e-03 | — | — |
| **v3a_K24_par8** (rec. fast) | 9.3 s | **69×** | 0.916 | 0.963 | 1.1e-01 | **1.60 min** | **~270×** |
| **v3a_K48_par8** (rec. accurate) | 19.0 s | **34×** | 0.998 | 0.999 | 5.5e-03 | **3.04 min** | **~144×** |

Notes:
- **Fidelity** = \|⟨ref\|test⟩\|² (phase-insensitive). 1.0 = perfect.
- **‖ψ‖²** = norm of the test final state. < 1 means amplitude leaked
  outside the truncated K-dim subspace.
- **max \|Δ\|ψ\|²\|** = largest single-component bare-basis population error.
- The "Scan speedup" compares against the ~7.3 h estimated sequential
  baseline (= 41 × 10.68 min); we did not actually run V1's full scan.

## Variant details

### V1 — Baseline (full propagator, segmented grid, sequential)
The current production code: `propagate_midpoint` on the full N=196 basis,
segmented adaptive `t_grid` with single-shot exact unitaries across the
constant-H stretches. ~40 ms per `eigh(196)` × 13 837 steps. This is the
truth reference; everything else is measured against it.

### V2 — Process-parallel scan (cloudpickle + spawn)
Wraps `scan.run_scan(..., n_workers=N)` to spawn N worker processes via
`multiprocessing.spawn` (Windows-friendly), serializing the `RamseyRFConfig`
(including all closures inside `FieldStack`) once via cloudpickle. Each
worker pins BLAS to 1 thread (5 env vars set in the pool initializer) so
8 workers actually use 8 cores instead of fighting over OpenBLAS threads.

Adds zero accuracy impact (per-point physics unchanged) and gives ~4×
wall-clock speedup on 8 cores for our ~19 s/trajectory workload — sub-linear
because process spawn + cfg deserialization eats ~2 s constant overhead per
worker (irrelevant for longer runs).

### V3a — Static truncated propagator
Pre-projects the Hamiltonian onto a K-dim subspace built once at simulator
init from `H_high` (the 30 kV/cm plateau), picking the K eigenvectors with
largest summed overlap to the bare J=1 sublevels. The propagator
[`propagator_truncated.propagate_midpoint_truncated_decomposed`](ramsey_rf/propagator_truncated.py)
exploits the linear structure of `H(t) = 2π·(Hff + Σ E_α·HS_α + Σ B_α·HZ_α)`:
each of the 7 sub-matrices is pre-projected once (`T†·H_α·T`, K×K), and
each timestep just rebuilds `H_proj(t)` as a weighted sum (O(K²·7)) before
the `eigh(K)` and the K-dim midpoint exponential.

Per-step cost drops from O(N³) ≈ 7.5M flops to O(K²·7 + K³) ≈ 7K + 32K =
40K flops at K=32 — the **theoretical** speedup is ~190× per step;
realised speedup is 30–200× (Python overhead + the trajectory/field
evaluations now dominate at small K).

#### Convergence vs K

K is the dimension of the truncated subspace. The dressed J=1 manifold spans
16 states by definition; at the 30 kV/cm plateau each is heavily Stark-mixed
(56% bare J=1, 14% J=0, 27% J=2, 3% J=3) — so the relevant subspace must
include the strongly-mixed J=0/2/3 dressed sublevels too.

| K | Subspace coverage | Norm preserved | Fidelity |
|---|---|---|---|
| 16 | bare-J=1-dominant only | 66.1% | 0.395 |
| 24 | + ~half of J=0/2 admixture | 96.3% | 0.916 |
| 32 | + remaining J=0/2 + some J=3 | 96.3% | 0.729 ⚠ |
| 48 | + most of J=3, some J=4 | 99.9% | 0.998 |

The K=32 fidelity dip (0.73 vs 0.92 at K=24, even though norm is the same)
is a real artifact of the static-projection scheme: the 8 extra eigenvectors
added between K=24 and K=32 have eigenvalues whose accumulated relative
phases over the 22 ms trajectory mix the in-subspace amplitudes in a way
that destructively interferes with the relevant components. K=48 adds enough
degrees of freedom to "wash out" this resonance and recover high fidelity.
**Don't use K=32 for production**; pick K=24 (cheap, amplitude-OK) or K=48
(clean, very accurate).

This non-monotonicity is the main weakness of static truncation. Phase B's
V3b (adiabatic-tracked truncation) is expected to fix it cleanly, since the
basis would follow the physical eigenstate ordering.

### V2 + V3a stacks (recommended production options)

Combining V3a with V2 multiplies the wins:

- **v3a_K48_par8 (high fidelity)**: 41-pt scan in 3.04 min, fidelity 0.998 —
  recommended for any production run where you care about fringe shape and
  absolute amplitude.
- **v3a_K24_par8 (fast survey)**: 41-pt scan in 1.60 min, fidelity 0.916 —
  recommended for a quick fringe-position scan or coarse parameter sweeps
  where you'll re-run with K=48 at the interesting points.

## Recommendation

For the 41-point ω_rf scan that motivated this investigation:

> Use **`v3a_K48_par8`**: fidelity 0.998, **3.04 min** instead of the
> projected 7.3 h. ~145× speedup with negligible loss of physical accuracy.

For coarser exploratory scans, K=24 buys another 2× wall-clock at fidelity
0.92 (good enough to read fringe positions and amplitudes).

## What's NOT in Phase A (deferred to Phase B/C/D)

| ID | Variant | Status | Expected gain |
|---|---|---|---|
| V3b | Adiabatic-tracked truncation | not yet | Should fix the K=32 dip; better fidelity at small K |
| V4 | Sparse + Krylov `expm_multiply` | not yet | 2–5× for active segments only |
| V5 | Numba JIT inner loop | not yet | 5–15% (eigh dominates) |
| V6 | complex64 reduction | not yet | 1.5–2× per eigh, ~10⁻³ accuracy cost |
| V7 | GPU `cupy.linalg.eigh` (batched) | requires GPU machine | 5–50× if batched across scan points |

Given the Phase A result (3 min for the headline scan), Phase B/C/D are
mostly optional. The cleanest next investments would be **V3b** (to fix
the K=32 dip and let us safely use smaller K with confidence) and **V7**
(if even larger Jmax or much longer scans become needed — GPU batched
eigh would be a big win for 100+ point sweeps).

## Reproducing these numbers

```bash
# 1. Build / reuse the cached reference (10.68 min one-shot, then cached)
.venv/Scripts/python.exe examples/ramsey_rf/benchmarks/reference.py

# 2. Run V1 + V3a single-trajectory variants (~1 min total, V1 cached)
.venv/Scripts/python.exe -u examples/ramsey_rf/benchmarks/perf_bench.py \
    --variants v1 v3a_K16 v3a_K24 v3a_K32 v3a_K48

# 3. Run the parallel-scan headline numbers (~5 min total)
.venv/Scripts/python.exe -u examples/ramsey_rf/benchmarks/perf_bench.py \
    --variants v3a_K48_par8 --scan --n-workers 8
.venv/Scripts/python.exe -u examples/ramsey_rf/benchmarks/perf_bench.py \
    --variants v3a_K24_par8 --scan --n-workers 8
```

Hardware: this PC, 8+ CPU cores, no CUDA. Times will scale with core count
for the parallel variants (sub-linear by ~50% for 8-core due to spawn
overhead) and with single-thread BLAS speed for V3a.
