# Fixed Tsit5 follow-up: result

Selected: FP64 staged sparse RHS, base step **4.464285714 ns**, buffer swapping and constant-coefficient stages, with a synchronized **5.357142857 ns** step where `abs((t-center)/sigma) > 0.8`. Steps still clip at microwave polarization boundaries and output times. The schedule uses the microwave envelope; UV remains on. There are no error-driven accept/reject decisions.

Full-transit batch 128 at 150 m/s, including 201 population samples and both density outputs: **97.492 s -> 79.710 s**, **18.24% less runtime**. Selected repeats: 79.642, 79.710, 79.775 s. Disk compression is outside these timings; setup/capture is reported separately.

Finer step tuning supplies most of the gain. Buffer swapping or constant stages alone did not help; combined, they improved the short test by about 4% and the full transit by 2.4%. Saved matrices match the unmodified solver bit-for-bit at equal timesteps (`bookkeeping_equivalence.json`). The gentle schedule then saves a further 5.6% of full-transit runtime.

Validation: 483 distinct detuning/J-mixture cases over the near-peak 5 us interval, worst complex-entry error **0.9251%**. Full endpoints at 50/150/250 m/s for nine distinct cases each: **0.2560/0.3434/0.3157%**, compared to archived finer GPU outputs; strict archived CPU subsets also pass. Relative errors apply at reference magnitudes >=1e-6; near-zero absolute errors and population-history differences are reported separately. Full-grid full-transit validation remains necessary before production adoption.

Rejected: fixed 5 ns (1.709% error) and the strong 1.5x schedule (1.0198%). Larger-batch confirmation passed at 1024 trajectories. No CPU ODE benchmarks were rerun; production files and settings were unchanged. All 18 jobs completed within the three-hour cap.

Variant names: `swap` removes density/FSAL copies through buffer permutations; `const` uses constant-coefficient stage kernels; `schedule_gentle/strong` multiply the base step by 1.2/1.5 outside the central microwave-envelope region. `source_snapshot/` retains the final source files; the manifest records the schedule-launch correction before job 11.

---

# GPU integrator selection and optimizations

Stage: complete.
Winner: {'integrator': 'tsit5', 'dt_ns': 4.464285714285714, 'variant': 'swap_const_schedule_gentle'}. Retained optimization switches: ['swap_const_schedule_gentle'].

Full complex-entry target: 1% above magnitude 1e-6; near-zero absolute errors are reported separately. No CPU ODEs or production changes. Production fixture uses 483 distinct detuning/J-mixture cases repeated only to fill larger batches. Historical fixture uses independent replicas of the archived nominal trajectory.

| Fixture / us | Integrator | Variant | Batch | dt initial/fixed ns | Chunk | Seconds | Entry error % | Pass |
|---|---|---|---:|---:|---:|---:|---:|---|
| production / 5 | tsit5 | base | 512 | 3.906 | 16 | 6.92 | 0.4483 | True |
| production / 5 | tsit5 | base | 512 | 4.167 | 16 | 6.513 | 0.6363 | True |
| production / 5 | tsit5 | base | 512 | 4.464 | 16 | 6.092 | 0.9251 | True |
| production / 5 | tsit5 | base | 512 | 5 | 16 | 5.505 | 1.709 | False |
| production / 5 | tsit5 | swap | 512 | 4.464 | 16 | 6.109 | 0.9251 | True |
| production / 5 | tsit5 | const | 512 | 4.464 | 16 | 6.096 | 0.9251 | True |
| production / 5 | tsit5 | swap_const | 512 | 4.464 | 16 | 5.846 | 0.9251 | True |
| production / 293.3333333333333 | tsit5 | base | 128 | 3.906 | 16 | 97.49 | 0.1 | True |
| production / 293.3333333333333 | tsit5 | base | 128 | 4.464 | 16 | 86.44 | 0.2 | True |
| production / 293.3333333333333 | tsit5 | swap_const | 128 | 4.464 | 16 | 84.39 | 0.2 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle | 128 | 4.464 | 16 | 79.64 | 0.3434 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_strong | 128 | 4.464 | 16 | 73.19 | 1.02 | False |
| production / 879.9999999999999 | tsit5 | swap_const_schedule_gentle | 9 | 4.464 | 16 | 28.06 | 0.256 | True |
| production / 176.0 | tsit5 | swap_const_schedule_gentle | 9 | 4.464 | 16 | 6.747 | 0.3157 | True |
| production / 5 | tsit5 | swap_const_schedule_gentle | 128 | 4.464 | 16 | 1.294 | 0.2751 | True |
| production / 5 | tsit5 | swap_const_schedule_gentle | 1024 | 4.464 | 16 | 12.27 | 0.9251 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle | 128 | 4.464 | 16 | 79.71 | 0.3434 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle | 128 | 4.464 | 16 | 79.77 | 0.3434 | True |

## Full-transit validation

Includes 201 population samples, density at field exit, and density after 25 lifetimes with drives off. Nine distinct cases are detunings -20/0/+20 MHz times J=1/2/3 mixtures; larger batches repeat these cases. All endpoint entries are compared to archived RK4 at 0.48828125 ns. An archived strict CPU reference separately checks J=2 at zero detuning. This does not validate every detuning and velocity over its full transit.

| Method / variant | v m/s | Batch | Base step ns | Total s | Endpoint error % vs finer GPU | Strict CPU subset % | Population max absolute difference | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tsit5 / base | 150 | 128 | 3.90625 | 97.49 | 0.1 | 0.0018647 | 3.994e-07 | True |
| tsit5 / base | 150 | 128 | 4.46429 | 86.44 | 0.20001 | 0.0018068 | 7.308e-07 | True |
| tsit5 / swap_const | 150 | 128 | 4.46429 | 84.39 | 0.20001 | 0.0018068 | 7.308e-07 | True |
| tsit5 / swap_const_schedule_gentle | 150 | 128 | 4.46429 | 79.64 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_strong | 150 | 128 | 4.46429 | 73.19 | 1.0198 | 0.0017976 | 9.449e-07 | False |
| tsit5 / swap_const_schedule_gentle | 50 | 9 | 4.46429 | 28.06 | 0.25603 | 3.1262e-07 | 1.09e-06 | True |
| tsit5 / swap_const_schedule_gentle | 250 | 9 | 4.46429 | 6.747 | 0.31571 | 0.0019519 | 5.012e-07 | True |
| tsit5 / swap_const_schedule_gentle | 150 | 128 | 4.46429 | 79.71 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_gentle | 150 | 128 | 4.46429 | 79.77 | 0.34343 | 0.0018049 | 7.837e-07 | True |

Adaptive error control: maximum of per-trajectory RMS complex embedded errors. Accepted/rejected steps, setup/capture/transfer cost, FP32 storage diagnostics and full numeric outputs are in individual JSON/NPZ files. Short-interval fixed/adaptive accuracy is checked against the same archived strict CPU references; a local tolerance alone is not a pass. Full-transit reference scope is stated separately above.

The previous approximately 2.56x CPU throughput estimate used fixed RK4 at 0.9765625 ns and 128 replicas of the historical 200 us fixture. It divided an archived single-CPU runtime by ideal 12-way scaling; it was not a measured 12-worker speedup. Historical and production fields differ, so the production Tsit5/RK4 gain must not be multiplied into that estimate. No CPU ODE timings were rerun in this suite.
