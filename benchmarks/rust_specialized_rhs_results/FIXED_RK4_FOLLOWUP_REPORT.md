# Fixed RK4 checkpoint and detuning follow-up

The earlier 0.5 ns RK4 result does not pass the agreed density-error criterion at every tested RC operating point. At +0.5 MHz optical detuning its maximum relative density-entry error is 1.205%, exceeding 1%. Reducing the timestep to 0.4 ns passes the tested cases, with a smaller 6–7% RC runtime reduction versus adaptive Tsit5.

| Fixture | RK4 step | Runtime reduction vs Tsit5 | Maximum relative density error | All checkpoints pass? |
|---|---:|---:|---:|---|
| RC nominal | 0.5 ns | 25.31% | 0.940% | Yes |
| RC +0.5 MHz | 0.5 ns | 25.90% | 1.205% | **No** |
| RC nominal | 0.4 ns | 6.24% | 0.386% | Yes |
| RC +0.5 MHz | 0.4 ns | 6.89% | 0.495% | Yes |
| R(2) nominal | 0.5 ns | −86.06% | 0.0000140% | Yes |
| R(2) nominal | 0.4 ns | −132.44% | 0.00000567% | Yes |

Negative runtime reductions denote slowdowns. These R(2) timesteps were selected for RC; this is not an optimized fixed-step search for R(2). Maximum small-entry absolute errors were below 1.2e-9 for every candidate, including the failed RC relative-error case.

Each scenario has 12 interleaved rounds of current-tolerance Tsit5, RK4 0.5 ns, and RK4 0.4 ns. All six method orders occur twice. Runtime reductions are medians of within-round ratios. Full complex density snapshots are taken at 2.5, 5, 7.5 and 10 microseconds after the initial time of 125 ns, within one continuous trajectory segmented at the existing 250 ns boundaries. References use Tsit5 atol=1e-13, rtol=1e-11; the adaptive candidate uses atol=1e-11, rtol=1e-9. Fixed-step accuracy is governed by dt. The entrywise check is 1% relative above reference magnitude 1e-6, absolute 1e-6 below, with trace error <=1e-9.

These are the short E=200 V/cm, B=0.01 G performance fixtures with optical EOM at 1.6 MHz and beta=3.8. RC has the existing four microwave polarization configurations. They are not the zero-electric-field apparatus model or full transit. Passing these checks does not validate 0.4 ns throughout a production scan.

Reproducer: `../check_fixed_rk4_followup.py`, using the isolated checkpoint wrapper in `../check_current_cpu_solvers.py`. Raw per-run timings, checkpoint errors, paired bootstrap intervals, fixture metadata and wrapper hash: `fixed_rk4_followup.json`. No production solver changes were made.
