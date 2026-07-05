# Fixed-Step Solver Benchmark

This benchmark investigates fixed-step Rust RK solvers for coarse OBE scans. The fixed-step solvers use `dt` as a maximum step and shorten steps only to land on `saveat` points or `t1`.

## Setup

- Model: two-level Lindblad system with one decay channel.
- Single trajectory: final photon integral at zero detuning.
- Grid scan: 201 detuning points from `-2` to `2`, final photon integral.
- Repeats per timing case: `15`.
- Reference: adaptive Rust `dopri5` with `dt=1e-3`, `reltol=1e-8`, `abstol=1e-10`.

## Runtime

| case                  | mode   | threads | trimmed_mean_seconds | speedup_vs_adaptive | accepted_steps | rhs_calls |
| --------------------- | ------ | ------- | -------------------- | ------------------- | -------------- | --------- |
| adaptive_dopri5       | single | 1       | 4.66154e-05          | 1                   | 15             | 91        |
| adaptive_dopri5       | grid   | 1       | 0.00851422           | 1                   | 3845           | 23271     |
| adaptive_dopri5       | grid   | 4       | 0.00301188           | 1                   | 3845           | 23271     |
| fixed_dopri5_dt5e-2   | single | 1       | 5.24692e-05          | 0.888433            | 16             | 96        |
| fixed_dopri5_dt5e-2   | grid   | 1       | 0.00630495           | 1.3504              | 3216           | 19296     |
| fixed_dopri5_dt5e-2   | grid   | 4       | 0.00218582           | 1.37791             | 3216           | 19296     |
| fixed_dopri5_dt2p5e-2 | single | 1       | 8.71231e-05          | 0.535052            | 32             | 192       |
| fixed_dopri5_dt2p5e-2 | grid   | 1       | 0.0123737            | 0.688092            | 6432           | 38592     |
| fixed_dopri5_dt2p5e-2 | grid   | 4       | 0.00376988           | 0.798932            | 6432           | 38592     |
| fixed_dopri5_dt1e-3   | single | 1       | 0.00149035           | 0.0312782           | 800            | 4800      |
| fixed_dopri5_dt1e-3   | grid   | 1       | 0.362712             | 0.0234738           | 160800         | 964800    |
| fixed_dopri5_dt1e-3   | grid   | 4       | 0.0991171            | 0.0303871           | 160800         | 964800    |
| fixed_rk4_dt5e-2      | single | 1       | 3.32231e-05          | 1.4031              | 16             | 64        |
| fixed_rk4_dt5e-2      | grid   | 1       | 0.0056199            | 1.51501             | 3216           | 12864     |
| fixed_rk4_dt5e-2      | grid   | 4       | 0.00205359           | 1.46664             | 3216           | 12864     |
| fixed_rk4_dt2p5e-2    | single | 1       | 7.85077e-05          | 0.593768            | 32             | 128       |
| fixed_rk4_dt2p5e-2    | grid   | 1       | 0.0101969            | 0.834985            | 6432           | 25728     |
| fixed_rk4_dt2p5e-2    | grid   | 4       | 0.00344928           | 0.873191            | 6432           | 25728     |
| fixed_rk4_dt1e-3      | single | 1       | 0.00135915           | 0.0342976           | 800            | 3200      |
| fixed_rk4_dt1e-3      | grid   | 1       | 0.241279             | 0.0352879           | 160800         | 643200    |
| fixed_rk4_dt1e-3      | grid   | 4       | 0.0649972            | 0.0463386           | 160800         | 643200    |
| fixed_rk4_dt2e-3      | single | 1       | 0.000662685          | 0.0703432           | 400            | 1600      |
| fixed_rk4_dt2e-3      | grid   | 1       | 0.122392             | 0.0695653           | 80400          | 321600    |
| fixed_rk4_dt2e-3      | grid   | 4       | 0.0331224            | 0.0909317           | 80400          | 321600    |
| fixed_rk2_dt1e-2      | single | 1       | 6.69154e-05          | 0.696632            | 80             | 160       |
| fixed_rk2_dt1e-2      | grid   | 1       | 0.0134659            | 0.63228             | 16080          | 32160     |
| fixed_rk2_dt1e-2      | grid   | 4       | 0.00406063           | 0.741726            | 16080          | 32160     |
| fixed_rk2_dt5e-4      | single | 1       | 0.00128773           | 0.0361996           | 1601           | 3202      |
| fixed_rk2_dt5e-4      | grid   | 1       | 0.23674              | 0.0359644           | 321801         | 643602    |
| fixed_rk2_dt5e-4      | grid   | 4       | 0.0703103            | 0.0428369           | 321801         | 643602    |

## Validation Against Adaptive Reference

| case                  | mode   | threads | photon_integral_abs_error | photon_integral_rel_error | peak_shift | max_normalized_line_error |
| --------------------- | ------ | ------- | ------------------------- | ------------------------- | ---------- | ------------------------- |
| fixed_dopri5_dt5e-2   | single | 1       | 8.22454e-06               | 0.00111632                | nan        | nan                       |
| fixed_dopri5_dt5e-2   | grid   | 1       | 8.22454e-06               | 0.00111632                | 0          | 0.00158156                |
| fixed_dopri5_dt5e-2   | grid   | 4       | 8.22454e-06               | 0.00111632                | 0          | 0.00158156                |
| fixed_dopri5_dt2p5e-2 | single | 1       | 1.75834e-05               | 0.0023866                 | nan        | nan                       |
| fixed_dopri5_dt2p5e-2 | grid   | 1       | 1.75834e-05               | 0.0023866                 | 0          | 0.00192765                |
| fixed_dopri5_dt2p5e-2 | grid   | 4       | 1.75834e-05               | 0.0023866                 | 0          | 0.00192765                |
| fixed_dopri5_dt1e-3   | single | 1       | 2.06979e-05               | 0.00280935                | nan        | nan                       |
| fixed_dopri5_dt1e-3   | grid   | 1       | 2.06979e-05               | 0.00280935                | 0          | 0.00204512                |
| fixed_dopri5_dt1e-3   | grid   | 4       | 2.06979e-05               | 0.00280935                | 0          | 0.00204512                |
| fixed_rk4_dt5e-2      | single | 1       | 8.22537e-06               | 0.00111644                | nan        | nan                       |
| fixed_rk4_dt5e-2      | grid   | 1       | 8.22537e-06               | 0.00111644                | 0          | 0.00157991                |
| fixed_rk4_dt5e-2      | grid   | 4       | 8.22537e-06               | 0.00111644                | 0          | 0.00157991                |
| fixed_rk4_dt2p5e-2    | single | 1       | 1.75834e-05               | 0.00238661                | nan        | nan                       |
| fixed_rk4_dt2p5e-2    | grid   | 1       | 1.75834e-05               | 0.00238661                | 0          | 0.00192754                |
| fixed_rk4_dt2p5e-2    | grid   | 4       | 1.75834e-05               | 0.00238661                | 0          | 0.00192754                |
| fixed_rk4_dt1e-3      | single | 1       | 2.06979e-05               | 0.00280935                | nan        | nan                       |
| fixed_rk4_dt1e-3      | grid   | 1       | 2.06979e-05               | 0.00280935                | 0          | 0.00204512                |
| fixed_rk4_dt1e-3      | grid   | 4       | 2.06979e-05               | 0.00280935                | 0          | 0.00204512                |
| fixed_rk4_dt2e-3      | single | 1       | 2.06829e-05               | 0.00280731                | nan        | nan                       |
| fixed_rk4_dt2e-3      | grid   | 1       | 2.06829e-05               | 0.00280731                | 0          | 0.00204455                |
| fixed_rk4_dt2e-3      | grid   | 4       | 2.06829e-05               | 0.00280731                | 0          | 0.00204455                |
| fixed_rk2_dt1e-2      | single | 1       | 1.9886e-05                | 0.00269914                | nan        | nan                       |
| fixed_rk2_dt1e-2      | grid   | 1       | 1.9886e-05                | 0.00269914                | 0          | 0.00211229                |
| fixed_rk2_dt1e-2      | grid   | 4       | 1.9886e-05                | 0.00269914                | 0          | 0.00211229                |
| fixed_rk2_dt5e-4      | single | 1       | 2.07009e-05               | 0.00280975                | nan        | nan                       |
| fixed_rk2_dt5e-4      | grid   | 1       | 2.07009e-05               | 0.00280975                | 0          | 0.00204548                |
| fixed_rk2_dt5e-4      | grid   | 4       | 2.07009e-05               | 0.00280975                | 0          | 0.00204548                |

## Interpretation

- Best fixed-step speedup: `1.52x` for `fixed_rk4_dt5e-2` in `grid` mode with `1` thread(s).
- Worst fixed-step speedup: `0.0235x` for `fixed_dopri5_dt1e-3` in `grid` mode with `1` thread(s).
- `fixed_dopri5` and `fixed_rk4` are useful candidates when `dt` is coarse enough to keep the accepted-step count comparable to adaptive `dopri5`. In this benchmark, `dt=0.05` gives the speedup.
- Small fixed steps are counterproductive: `dt=1e-3` and `dt=2e-3` perform hundreds of fixed steps where adaptive `dopri5` takes only tens of steps.
- `fixed_rk2` is included as a lower-stage comparison, but it did not win here at the tested validation settings.
- Photon-count errors are measured against the current adaptive solver baseline, including its accepted-step quadrature for final integrals.
- For real molecular OBE scans, the acceptance criterion should be peak shifts below the requested MHz tolerance and photon-count/line-shape errors acceptable for the fit or coarse scan stage.

- Raw timings: `fixed_step_solver_timings.csv`.
- Validation CSV: `fixed_step_solver_validation.csv`.
