# Krylov exponential integrator: measured, and it is ~1000x SLOWER

Run: `uv run python benchmarks/bench_krylov_exponential.py`
(Windows, release build, 2026-08-20.)

Decision gate for audit item 7 ("step-count reduction for oscillatory systems",
estimated 10-25x). System is the r2-in-static-E-field system built identically
to `diagnose_step_size.py` / `examples/lindblad/r2_peak_ratio_vs_z_polarization.ipynb`:
R2_F1_7o2_F3, Ez = 171.6 V/cm, `retain_opposite_parity_levels=True`,
`qn_compact=True`, 60 mW, T = 108.7 us, 38 states, packed-real dim 1444,
nnz 13,690. Detuning 25 MHz (the opposite-parity peak — the observable of
interest, and the case where the per-level rotating frame did worst).

## The claim under test

Any exponential/Lawson integrator must apply `exp(L0*h)` to a vector. Krylov
(`expm_multiply`) is the route that preserves sparsity, but its Krylov
dimension scales with `||L0*h||`. If that scaling is linear, the work to cover
a fixed span `T` is **independent of the step size** — larger steps just cost
proportionally more each — and both the exponential method and explicit RK
cost O(omega*T) matvecs. The decisive measurement is therefore the *projected
total* to cover T as a function of h.

## Result: the total is flat

Matvecs counted exactly via a counting `LinearOperator`, not inferred from
timings. Single sparse matvec: 6.32 us. dopri5 baseline at reltol 1e-7:
**0.955 s**.

| h [ns] | \|\|L\*h\|\| | matvecs / application | t_apply [ms] | steps to cover T | **projected total [s]** | vs dopri5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 5 865 | 3 738 | 56.3 | 21 739 | **1 223.6** | 0.001x |
| 10 | 11 730 | 7 290 | 98.2 | 10 870 | **1 067.3** | 0.001x |
| 25 | 29 324 | 17 958 | 231.3 | 4 348 | **1 005.7** | 0.001x |
| 50 | 58 648 | 35 730 | 467.6 | 2 174 | **1 016.5** | 0.001x |
| 100 | 117 296 | 71 274 | 943.1 | 1 087 | **1 025.1** | 0.001x |
| 250 | 293 240 | 177 906 | 2 394.7 | 435 | **1 041.2** | 0.001x |
| 500 | 586 479 | 341 790 | 4 475.6 | 217 | **973.0** | 0.001x |
| 1000 | 1 172 958 | 637 995 | 8 322.5 | 109 | **904.6** | 0.001x |

**Step size buys nothing.** Over a 200x range in `h` the projected total moves
by +-15% around ~1000 s, with no trend that would reward larger steps. Matvecs
per application scale as 170x for a 200x increase in h — linear to within the
measurement, exactly as predicted. This is the O(omega\*T) floor, measured
directly.

Accuracy is not the problem: 100 Krylov steps at h = 1e-8 over a 1 us sub-span
match a reltol-1e-10 dopri5 reference to **1.9e-12** in populations. The method
is correct and useless.

## Why it is 1000x rather than the ~2x predicted from the active scale

The back-of-envelope estimate (~1-2x) used the *active* frequency scale — the
73.6 MHz B J=3 manifold spread that limits dopri5. The measured
`||L||_1 = max |Im lambda| = 1.173e12 rad/s = 186 682 MHz`, i.e. **187 GHz**,
2500x larger.

That number is exactly the spectator manifolds. From the step-size report's
manifold table, X J=5 sits at +120 164 MHz and X J=1 at -66 523 MHz;
120 164 + 66 523 = 186 687 MHz, matching the measured 186 682 to 5 parts in
1e5. The dominant eigenvalue is the coherence between two spectator manifolds.

**This is the crux.** dopri5 is unaffected by those manifolds because they
carry zero coherence — their phases multiply zeros, which is precisely why the
step-size report concluded the GHz offsets are harmless. A Krylov exponential
method has no such immunity: the Krylov dimension keys off the norm of the
whole operator, not off which modes the solution actually occupies. **The
spectator manifolds are free for the explicit stepper and catastrophic for the
exponential one.**

Projecting the ideal fix — strip the spectators to pure population sinks so
`||L||` drops to the ~73.6 MHz active scale — gives 1000 s / 2500 = ~0.4 s
against dopri5's 0.955 s, i.e. **~2.4x**. That lands on the analytic estimate,
and it is the ceiling for this route even after work that does not currently
exist.

## The dense route, for comparison

| | cost |
| --- | ---: |
| `np.linalg.eig(L)` (1444 dim) | 0.54 s |
| `np.linalg.inv(V)` | 0.17 s |
| **setup total** | **0.72 s** |
| per application, any t | 0.34 ms |
| accuracy vs dopri5 reltol 1e-7 | 2.65e-10 |

The dense exact propagator remains what the earlier study found: setup-dominated
and roughly break-even for a single trajectory (0.72 s vs 0.955 s), then
essentially free per additional initial state or saveat point. It is 3-4 orders
of magnitude better than Krylov here, because at dim 1444 with nnz 13,690 the
matrix is far too small and too dense-spectrumed for sparsity to pay.

Note the analytic Jacobian makes L extraction free — **0.71 ms**, versus the
13-83 ms the probe cost — but that was never the bottleneck: setup is eig and
inv. Extraction went from ~6% of setup to ~0.1%.

## Conclusion

Item 7's exponential/Lawson route is **not viable via Krylov**, and the failure
is structural rather than an implementation detail:

1. Cost is flat in step size — the O(omega\*T) floor is real and measured.
2. It is set by `||L||`, which the spectator manifolds dominate by 2500x, and
   those are exactly the modes the explicit stepper gets for free.
3. Even removing them entirely projects to ~2.4x, not 10-25x.

The only route left whose cost does *not* scale with omega\*T is Magnus with
Filon/Levin oscillatory quadrature, which integrates the oscillatory coefficient
integrals analytically instead of sampling them. That is research-grade work
with real order-reduction/resonance risk, for a payoff now bounded by the same
~2x argument once the spectator issue is handled.

For time-independent L the answer is already available and better: diagonalize
the Liouvillian (the exact propagator). Its limitation is precisely the one that
motivates item 7 — it does not survive time-dependent parameters (polarization
switching, Gaussian beam transit, multipass, phase modulation).

Recommendation: close item 7's exponential-integrator route as WONTFIX on
measurement, per the same stopping rule used for `PARTITIONED_PACKED_MAX_STATES`,
the `representation="auto"` cost model and the `h_mat_elems` lookup maps. Keep
the exact propagator as a niche scan backend for fixed-L work. Note that threads
already deliver 9.19x on the real scan shape and are the default.

## Caveats

- `solver_stats` came back empty (`{}`) from this harness, so accepted-step
  counts are not re-reported here; they are in
  `step_size_diagnostics_results/`. Same stats-key drift noted elsewhere in the
  audit. The dopri5 wall time (0.955 s) is ~17% below the 1.157 s recorded in
  the step-size report for the same configuration — different machine load and
  a newer extension build; the comparison above is internally consistent since
  every number comes from this one run.
- The projected totals assume a fixed step `h` and no rejected steps. That is
  generous to the exponential method, which makes the negative result stronger.
- Timings use the plain sparse matrix (scipy then takes the exact 1-norm);
  matvec counts come from a second pass through a counting `LinearOperator`,
  which uses `onenormest` instead. Counts and timings are therefore from
  separate passes by construction.

Raw numbers: `results.json`, `krylov_cost_vs_step.csv` in this folder.
