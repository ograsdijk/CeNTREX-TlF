# Current Rust CPU OBE solver comparison

Compared adaptive Dopri5 and Tsit5 on the current prepared-Hamiltonian/scalar-DAG D+S+U path. Both solvers ran in one dynamic-dimension extension using the same live Rust RHS, plan, evaluator, and ODE sources. Trajectories used the existing 10 us RC/R(2)/R(0)/Q(1) performance fixtures, segmented at 250 ns intervals; RC includes switched microwave gates, and the EOM phase uses absolute time. The common field settings are E=[0,0,200] V/cm and B=[0,0,0.01] G; the optical drive uses 1.6 MHz EOM modulation with beta=3.8. RC includes P(2) plus the 1-2 and 2-3 microwave drives, with 70 states; the optical fixtures have 41 (R2), 18 (R0), and 26 (Q1) states. The RC initial condition is the existing X J=2, F1=5/2, F=2, mF=0 fixture; the optical-only systems start uniformly over their modeled X levels. These are short solver fixtures, not full molecular-beam transit calculations.

Candidate tolerances were `atol=1e-11`, `rtol=1e-9`; the adaptive reference was Tsit5 at `atol=1e-13`, `rtol=1e-11`. Each candidate was checked against the full complex reference density: 1% maximum relative error for entries above magnitude 1e-6, 1e-6 maximum absolute error below that cutoff, and trace error at most 1e-9. One Rust/BLAS thread was used. There were 12 interleaved pairs per system with solver order alternating exactly six times each way. Confidence intervals are descriptive bootstrap 95% intervals over the 12 paired runtime ratios (20,000 resamples), not formal guarantees.

| System | Levels | Dopri5 median (s) | Tsit5 median (s) | Tsit5 reduction vs Dopri5 | RHS calls (Dopri5 / Tsit5) | Max relative error fraction, D / T | Max small-entry absolute error, D / T |
|---|---:|---:|---:|---:|---:|---:|---:|
| RC | 70 | 3.4070 | 2.9884 | 12.17% [11.92, 12.36] | 112889 / 99191 | 2.83e-04 / 2.14e-04 | 1.55e-11 / 2.25e-11 |
| R2 | 41 | 0.2997 | 0.2503 | 16.43% [16.17, 16.77] | 46241 / 38531 | 3.65e-07 / 3.72e-07 | 7.03e-14 / 6.81e-14 |
| R0 | 18 | 0.0923 | 0.0768 | 16.84% [15.50, 17.34] | 82565 / 68813 | 1.91e-08 / 1.30e-08 | 9.64e-15 / 6.50e-15 |
| Q1 | 26 | 0.1737 | 0.1447 | 16.87% [16.50, 17.28] | 65093 / 54353 | 1.10e-07 / 1.98e-07 | 9.10e-14 / 1.93e-13 |

The table reports relative-error fractions (for example, 2.14e-04 means 0.0214%). The strict Tsit5 reference used 282,077 / 114,857 / 206,027 / 162,155 RHS evaluations for RC / R(2) / R(0) / Q(1). No adaptive candidate rejected steps at this tolerance. Tsit5 used fewer RHS calls and was faster by about 12–17% in all four fixtures; every paired interval excludes zero.

The fixed-step screen found that fixed Dopri5, RK4, and RK2 all produced non-finite density at `dt=5 ns` on RC. At `dt=0.5 ns`, fixed RK4 passed all 12 paired accuracy checks against the same strict Tsit5 reference. Its median was 2.2637 s versus 2.9995 s for Tsit5, a 24.55% reduction [24.38, 24.68] in paired runtime. The worst of the 12 runs had 0.9396% relative error above 1e-6, 5.52e-10 absolute error below the cutoff, and trace error 4.44e-16. Fixed-step `atol` and `rtol` do not control this result; the step size does.

This establishes fixed RK4 at 0.5 ns as a promising RC option under this short fixture, with paired repeats. It is close to the 1% error limit and has only been checked at the final 10 µs endpoint. It does not establish a general fixed-step recommendation or performance on R(2); intermediate density checkpoints, a nearby dt comparison, another operating point, and R(2) validation remain useful before selecting it for production. The not-yet-run [follow-up harness](../check_fixed_rk4_followup.py) performs one segmented trajectory per candidate with save-at checkpoints at 2.5, 5, 7.5, and 10 us, tests dt=0.4 and 0.5 ns, and covers nominal RC, RC at +0.5 MHz detuning, and nominal R(2).

All recorded runs were sequential on one host after the exclusive timing slot began. The generic extension was compiled before timed trajectory calls; model preparation and reference solves were also outside candidate timing. There was no concurrent solver or compilation workload during measurements.

Raw paired measurements and per-run accuracy: `current_cpu_solvers_paired.json`. Fixed-step screen and paired finalist: `current_cpu_solvers_fixed_screen.json`. Reproducer: `../check_current_cpu_solvers.py`.
