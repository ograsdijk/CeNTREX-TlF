# Step-Size Diagnostics: r2 in a Static Electric Field (plan step 7 pre-work)

Produced by `benchmarks/diagnose_step_size.py` (2026-07-13). System is built
identically to `examples/lindblad/r2_peak_ratio_vs_z_polarization.ipynb`:
R2_F1_7o2_F3, Ez = 171.6 V/cm, B = 1e-5 G, `retain_opposite_parity_levels=True`,
`qn_compact=True`, X polarization, 60 mW rectangular beam (2x2 cm),
T = 108.70 us, dopri5 / `expanded_sparse`, abstol = 1e-9.

38 states (compact). Rabi = 2pi x 0.316 MHz, Gamma = 2pi x 1.560 MHz.

## 1. Frequency content of the RWA Hamiltonian (rabi at 60 mW, detuning 0)

Residual diagonal energies per (electronic, J) manifold, MHz (/2pi):

| manifold | states | min | max | spread |
| --- | --- | --- | --- | --- |
| B J=3 (driven, opposite parity retained) | 14 | -24.34 | +49.28 | **73.6** |
| X J=2 (driven) | 20 | -1.17 | +0.02 | 1.19 |
| X J=3 (spectator, decay target) | 1 | +151.3 | +151.3 | 0 |
| X J=1 (spectator) | 1 | -66 523 | | 0 |
| X J=4 (spectator) | 1 | +53 490 | | 0 |
| X J=5 (spectator) | 1 | +120 164 | | 0 |

Couplings: 76 nonzero, max |Omega_ij| = 2pi x 0.243 MHz, median 0.043 MHz.
Coupling detunings |H_ii - H_jj|: max 50.4 MHz, median 15.8 MHz.
Spectral radius rho(H) = 2pi x 120 GHz (spectator manifolds);
rho(H - diag) = 2pi x 0.374 MHz.

## 2. Observed stepping (accepted steps over 108.7 us)

| detuning | reltol | accepted | rejected | RHS calls | mean dt | wall |
| --- | --- | --- | --- | --- | --- | --- |
| 0 MHz | 1e-5 | 23 983 | 3 | 143 917 | 4.53 ns | 1.11 s |
| 0 MHz | 1e-7 | 24 031 | 0 | 144 187 | 4.52 ns | 1.12 s |
| 0 MHz | 1e-9 | 24 037 | 0 | 144 223 | 4.52 ns | 1.22 s |
| 25 MHz | 1e-5 | 23 983 | 4 | 143 923 | 4.53 ns | 1.22 s |
| 25 MHz | 1e-7 | 24 038 | 0 | 144 229 | 4.52 ns | 1.17 s |
| 25 MHz | 1e-9 | 24 046 | 0 | 144 277 | 4.52 ns | 1.16 s |

## 3. Interpretation

**The step size is oscillation-limited, not accuracy-limited.** Accepted
steps are flat over four orders of magnitude in reltol (23 983 -> 24 046,
+0.3%) with essentially zero rejections. An accuracy-limited RK5(4) would
change steps by ~10^(4/5) ~ 6.3x over that tolerance range.

**The limiting frequency is the driven-manifold structure, not the huge
spectator offsets.** The formal upper bound from removing the whole static
diagonal (rho(H)/max(rho(H-diag), Gamma) ~ 77 000x) is meaningless: the
GHz-scale spectator manifolds (X J=1,4,5) carry *zero coherence* — they are
populated only by decay (diagonal transfer) and never develop off-diagonal
elements, so their phases multiply zeros and do not constrain the solver.
The fastest *active* phases are coherences within/against the B J=3 manifold:
its internal spread is 73.6 MHz — the Stark-split opposite-parity doublet
plus hyperfine structure created by the 171.6 V/cm field — and X-B coupling
detunings reach 50.4 MHz. The observed dt = 4.52 ns corresponds to a
stability/resolution limit at ~2pi x (75-115) MHz, consistent with those
scales (dopri5 imaginary-axis extent ~3.3: 3.3/(2pi x 75 MHz) = 7.0 ns,
with controller safety below that).

Conclusion: **the static E field itself sets the step size.** This class of
simulation gets slower exactly when the physics of interest (parity mixing,
opposite-parity peak at +25 MHz) is turned on. Detuning value and tolerance
are irrelevant to cost; every trajectory in a scan pays ~24k steps / ~144k
RHS calls / ~1.1 s.

## 4. What removing the static phases can buy

Target scales after analytic removal of the static diagonal: Gamma =
1.56 MHz, Rabi <= 0.24 MHz, X J=2 spread 1.2 MHz — but coupling
*coefficients* then oscillate at their detunings (<= 50.4 MHz) in the
rotated frame, with small amplitude (Omega/detuning ~ 0.005-0.2).

- (a) **Per-dressed-level co-rotating frame** (extend the manifold rotation
  in `lindblad/generate_hamiltonian.py` to rotate each level at its own
  static energy; H becomes explicitly time-dependent with exp(i*Delta*t)
  coefficient phases). Explicit RK then steps at the *accuracy* limit of a
  small-amplitude 50 MHz driving term rather than the stability limit of a
  75-115 MHz state phase; estimated gain **~3-10x**. Costs: time-dependent
  coefficient evaluation per RHS call (loses the time-independent H cache
  for this system class — it is already time-dependent-equivalent in effect),
  more work per call. Moderate implementation risk, all in the symbolic
  RWA layer.
- (b) **Exponential/Lawson (interaction-picture) integrator in
  `rust/src/ode/`**: propagate the full static linear part (diagonal +
  decay) exactly between steps; the integrator resolves only the slow
  envelope (Gamma, Rabi). Estimated gain **~10-25x** (dt ~ 20-100 ns).
  Larger engineering effort; needs Magnus/Filon-style handling of the
  oscillatory coupling coefficients to reach the upper end.
- (c) **Secular approximation** (drop far-off-resonant couplings): NOT
  viable here — the 25 MHz opposite-parity response is the measured
  observable.

A pragmatic sequence: prototype (a) first (cheap to try, validated against
the notebook peak ratios and photon integrals), and use its measured gain to
decide whether (b) is worth building.

## 5. Side observations

- `find_exact_states` warns about ~0.48 overlaps for two approximate states
  during the build: at 171.6 V/cm the parity states are near-maximally mixed
  (1/sqrt(2) ~ 0.71 amplitude -> 0.48 probability), so the bare-parity
  labels are genuinely ambiguous. Pre-existing behavior, expected at strong
  mixing, but worth knowing that state *labels* in this regime are
  half-arbitrary.
- Scan-level impact: a typical peak-ratio scan (~10 polarization fractions x
  ~120 detunings) costs ~1200 trajectories x 1.1 s / parallelism. A 10x
  step-count reduction converts multi-minute scans to tens of seconds.

Raw numbers: `hamiltonian_metrics.csv`, `solve_step_counts.csv` in this
folder.

## 6. Per-level co-rotating frame prototype (measured)

Implemented as an opt-in helper,
`centrex_tlf.lindblad.generate_hamiltonian.apply_per_level_rotating_frame(obe_system)
-> OBESystem`: for each state `i` it extracts the *numeric* residual part
`E_i` of `H[i,i]` (symbolic detuning terms are left in place by zeroing all
free symbols before evaluating), and applies the diagonal unitary `T =
diag(exp(-i*E_i*t))`, giving `H'[i,i] = H[i,i] - E_i` and, for `i != j`,
`H'[i,j] = H[i,j] * exp(i*(E_i-E_j)*t)`. The Lindblad dissipator is exactly
invariant for single-jump collapse operators, so `C_array` is reused
unchanged. Validated (`tests/lindblad/test_per_level_rotating_frame.py`,
3 tests, all passing):

- A 2-level toy with a static 30 MHz diagonal splitting plus a symbolic
  detuning: populations at 200 saveat points (dopri5, `expanded_sparse`,
  reltol=1e-8, abstol=1e-10) agree between frames to atol 1e-6.
- The actual r2 system at detuning 0 and 25 MHz: final populations
  (reltol=1e-7, abstol=1e-9) agree to atol 1e-5.

So the transform is physically correct. **Measured on
`benchmarks/bench_per_level_frame.py`, it is a net slowdown, not a
speedup**, for this system:

| frame | detuning | reltol | accepted | rejected | RHS calls | mean dt | wall (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0 MHz | 1e-5 | 23 983 | 3 | 143 917 | 4.53 ns | 1.118 s |
| original | 0 MHz | 1e-7 | 24 031 | 0 | 144 187 | 4.52 ns | 1.152 s |
| original | 0 MHz | 1e-9 | 24 037 | 0 | 144 223 | 4.52 ns | 1.156 s |
| rotated | 0 MHz | 1e-5 | 16 293 | 0 | 97 759 | 6.67 ns | 2.433 s |
| rotated | 0 MHz | 1e-7 | 19 485 | 0 | 116 911 | 5.58 ns | 2.814 s |
| rotated | 0 MHz | 1e-9 | 19 564 | 0 | 117 385 | 5.56 ns | 2.864 s |
| original | 25 MHz | 1e-5 | 23 983 | 4 | 143 923 | 4.53 ns | 1.185 s |
| original | 25 MHz | 1e-7 | 24 038 | 0 | 144 229 | 4.52 ns | 1.157 s |
| original | 25 MHz | 1e-9 | 24 046 | 0 | 144 277 | 4.52 ns | 1.120 s |
| rotated | 25 MHz | 1e-5 | 33 945 | 0 | 203 671 | 3.20 ns | 4.980 s |
| rotated | 25 MHz | 1e-7 | 47 837 | 0 | 287 023 | 2.27 ns | 6.972 s |
| rotated | 25 MHz | 1e-9 | 49 036 | 0 | 294 217 | 2.22 ns | 7.266 s |

(wall times at reltol=1e-7 are the median of 3 repeats; other reltols are
single runs, consistent with section 2's methodology.)

**Step count**: at detuning 0 MHz the rotated frame *does* take fewer
accepted steps than the original (24 031 -> 19 485 at reltol=1e-7, 1.23x
fewer) -- some of the report's prediction holds. But at detuning 25 MHz
(the opposite-parity peak, the actual observable of interest) the rotated
frame takes *almost twice as many* steps (24 038 -> 47 837, 0.50x, i.e. 2x
worse), not fewer. The per-level rotation moves the B J=3 manifold's 73.6
MHz internal spread from the Hamiltonian *diagonal* (state phase) onto the
*off-diagonal coupling coefficients* (oscillating at up to that same 73.6
MHz), and dopri5's step-size controller is limited by the fastest complex
eigenvalue/derivative content of the RHS regardless of whether that content
sits on the diagonal or the off-diagonal -- so the fundamental oscillation
that was setting dt = 4.52 ns is still present, just relocated. At detuning
25 MHz the coupling into the near-resonant opposite-parity states apparently
makes this worse, not better.

**Tolerance scaling**: the rotated frame's accepted-step count *does* now
grow mildly with tighter reltol -- 16 293 -> 19 564 (1.20x) from reltol 1e-5
to 1e-9 at detuning 0 MHz, and 33 945 -> 49 036 (1.44x) at detuning 25 MHz --
so it is not perfectly flat/oscillation-limited like the original frame
(which stays within +/-0.3% across the same range). But both ratios are far
below the ~6.3x an accuracy-limited RK5(4) would show over four decades of
reltol (reltol^(-1/5) = (1e4)^(1/5) ~= 6.3), so the rotated frame is only
*partially* accuracy-limited -- it is still substantially constrained by
something oscillation/stability-like, just not as rigidly as before.

**Per-RHS-call cost**: the predicted cost side of the trade-off dominates.
Dividing wall time by RHS calls at reltol=1e-7: original ~= 8.0e-6 s/call at
both detunings; rotated ~= 2.41-2.43e-5 s/call at both detunings -- a
consistent ~3.0x per-call overhead from evaluating the now-time-dependent
coupling coefficients (sin/cos per RHS call) instead of static numbers, on
top of the `decomposed` IR now carrying one symbolic coefficient per unique
`(E_i - E_j)` frequency rather than a handful of static Rabi/polarization
terms. Combined with the *higher* step count at detuning 25 MHz, net wall
time is 2.4x slower at detuning 0 MHz (1.152 s -> 2.814 s) and 6.0x slower
at detuning 25 MHz (1.157 s -> 6.972 s).

**Photon-integral scan equivalence** (-5..30 MHz, 1 MHz steps, `output=
"photon_integral"`, weights = Gamma on the B-manifold indices, both frames):
curves agree closely (max abs diff 8.05e-5, max rel diff 8.95e-5 ~=
0.009%), and the argmax detuning matches exactly (0.0 MHz in both frames) --
confirming the observable is frame-invariant as expected, well within
numerical-accumulation noise. But the full 36-point scan took 52.7 s in the
original frame and 224.6 s in the rotated frame -- **4.3x slower**, driven
by the same per-call-cost and (at several detunings near the opposite-parity
resonance) step-count effects seen in the table above.

**Net speedup at the notebook's reltol=1e-7, detuning=0 MHz: 0.41x (i.e.
2.4x SLOWER)**, and worse (0.17x, 6.0x slower) at the physically interesting
25 MHz opposite-parity detuning. This misses the "expected ~3-10x" estimate
by a wide margin, in the wrong direction.

**Verdict on the negative result**: keeping the helper is still worthwhile
as an opt-in, validated utility (`apply_per_level_rotating_frame` is correct
and may help other systems where couplings within a manifold are weak or
absent), but **it is not applied to the r2-in-static-E-field benchmark
system** -- it makes that system slower, not faster, and this negative
result is recorded here rather than glossed over.

**Verdict on the exponential/Lawson integrator route (b), estimated
10-25x**: this experiment weakens confidence in that estimate too, but does
not necessarily invalidate the approach. The per-level rotation failed
because it turned a diagonal (state-phase) fast oscillation into an
equally-fast *explicit coupling-coefficient* oscillation that a generic
explicit RK stepper still has to resolve step-by-step, at higher per-call
cost besides -- it only *relocated* the problem instead of *removing* it
from the stepper's view. A true exponential/Lawson integrator would instead
propagate the full linear part (diagonal *and* the near-resonant B J=3
intra-manifold couplings, plus decay) exactly between steps via a matrix
exponential or Magnus expansion, so the stepper would only need to resolve
the slowly varying envelope (Rabi ~0.24 MHz, decay Gamma ~1.56 MHz) -- a
fundamentally different mechanism from the one just tested here, and
plausibly still capable of the larger gain. That said, given that even an
*exact* analytic removal of the diagonal made this system worse, not
better, the 10-25x estimate for (b) should be treated as unconfirmed and
optimistic until it is checked with a small prototype (e.g. a proper
matrix-exponential propagator on the B J=3 sub-block) rather than committed
to as a multi-week engineering investment on the strength of the earlier
spectral-radius argument alone.

Raw numbers: `per_level_frame_bench.csv` in this folder.
