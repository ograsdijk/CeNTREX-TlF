# Four GPU follow-up experiments: completed

Selected: **fixed Tsit5, FP64 packed upper-triangle density storage, direct sparse RHS, exact even-step graph tails, and GPU population sampling**. Retain the v2 synchronized schedule: 4.464285714 ns centrally and 5.357142857 ns where `abs((t-center)/sigma)>0.8`. No adaptive acceptance/rejection. All coherences are retained, and full 70x70 matrices are reconstructed for both density outputs.

Full transit, 150 m/s, 128 trajectories, 201 population samples plus field-exit and post-decay density: **79.263 s -> 59.479 s**, **24.96% less runtime**. Selected repeats: 59.479, 59.459, 59.564 s. Setup/compilation/capture and disk compression are outside these evolution/output timings and are recorded separately where applicable.

| Experiment | Full-transit seconds | Outcome |
|---|---:|---|
| Matched previous settings | 79.263 | Baseline |
| 1. Shorter graph tails | 70.305 | Retain; removes idle steps around 201 output times |
| 2. GPU population sampling alone | 78.346 | Small gain; diagonal history stays on GPU |
| 1 + 2 | 69.206 | Retain both |
| 1 + 2 + 3. Packed/direct density | 59.479 median | Best for the tested large batches |
| Packed/staged cross-check | 59.456 | Tied at batch 128; slower than direct at batch 512 |
| 4. Diagonal integrating-factor Tsit5 | 79.780 | Passing step slower; larger steps fail accuracy |

Packed storage at batch 512 over 5 us: full 5.680 s, packed/staged 4.612 s, packed/direct 4.446 s. Live array allocations after that short run: 562.4 MB -> 244.7 MB, excluding CUDA context, peak allocations and the full-transit population buffer. Packed storage retains both components of each upper-triangle entry, with zero imaginary padding on the diagonal. This is a memory representation change, not state-space truncation or reduced density output.

The integrating-factor prototype evolves `L0_ij=-i*(Hii-Hjj)-(rate_i+rate_j)/2` analytically within each step, and uses six Tsit5 RHS stages for the sparse commutator and population transfers in local transformed coordinates. At 4.464 ns it passed the short-entry criterion (0.799%) but was slower; 8/12/16/24 ns failed (18.75/154.43/681.70/9188.92%). The passing full transit had trace error 2.71e-8. These results concern this prototype; they do not rule out all exponential methods.

Validation: arbitrary coherent PSD RHS probes agree with the established full FP64 RHS to <6e-17 relative for packed layouts. The selected method passes all 483 detuning/J-mixture cases over the near-peak 5 us interval (worst complex-entry error **0.9251%**), 1024-trajectory confirmation, and full endpoints at 50/150/250 m/s (**0.2560/0.3434/0.3157%** vs archived finer GPU results). Archived strict CPU subsets also pass. Relative errors apply for reference magnitudes >=1e-6; near-zero absolute errors and population-history differences are separately reported. The selected final densities differ from the matched v3 baseline by <=4.1e-16, and population histories by <=1.3e-14 (`output_equivalence.json`).

Limits: full-transit validation covers nine distinct detuning/J-mixture cases per velocity, repeated to fill larger batches; it does not cover the entire production grid. Small batches can be slower with packed/direct. Full-grid full-transit validation is still required before production adoption. No new CPU ODE benchmarks were run, and production settings/files were unchanged. All 21 jobs completed within the existing 16:37:10 UTC deadline.

Variant names: `exact` uses even-step tail graphs with at most one idle tail step; `devicepop` gathers 201 population diagonals on-device and downloads that buffer once; `packed` stores the upper triangle; `packed_direct` evaluates the two sparse row products directly for each independent output, avoiding full product scratch arrays. GPU sample times and dark endpoints are checked. Generated CUDA sources are saved per job; final source snapshots and hashes are retained below.

---

# GPU integrator selection and optimizations

Stage: complete.
Winner: {'integrator': 'tsit5', 'dt_ns': 4.464285714285714, 'variant': 'swap_const_schedule_gentle_exact_devicepop_packed_direct'}. Retained optimization switches: ['swap_const_schedule_gentle_exact_devicepop_packed_direct'].

Full complex-entry target: 1% above magnitude 1e-6; near-zero absolute errors are reported separately. No CPU ODEs or production changes. Production fixture uses 483 distinct detuning/J-mixture cases repeated only to fill larger batches. Historical fixture uses independent replicas of the archived nominal trajectory.

| Fixture / us | Integrator | Variant | Batch | dt initial/fixed ns | Chunk | Seconds | Entry error % | Pass |
|---|---|---|---:|---:|---:|---:|---:|---|
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle | 128 | 4.464 | 16 | 79.26 | 0.3434 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle_exact | 128 | 4.464 | 16 | 70.31 | 0.3434 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle_devicepop | 128 | 4.464 | 16 | 78.35 | 0.3434 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle_exact_devicepop | 128 | 4.464 | 16 | 69.21 | 0.3434 | True |
| production / 5 | tsit5 | swap_const_schedule_gentle_exact_devicepop | 512 | 4.464 | 16 | 5.68 | 0.9251 | True |
| production / 5 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed | 512 | 4.464 | 16 | 4.612 | 0.9251 | True |
| production / 5 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 512 | 4.464 | 16 | 4.446 | 0.9251 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 128 | 4.464 | 16 | 59.48 | 0.3434 | True |
| production / 5 | if_tsit5 | swap_const_schedule_gentle_exact_devicepop | 512 | 4.464 | 16 | 6.798 | 0.799 | True |
| production / 5 | if_tsit5 | swap_const_schedule_gentle_exact_devicepop | 512 | 8 | 16 | 3.87 | 18.75 | False |
| production / 5 | if_tsit5 | swap_const_schedule_gentle_exact_devicepop | 512 | 12 | 16 | 2.544 | 154.4 | False |
| production / 5 | if_tsit5 | swap_const_schedule_gentle_exact_devicepop | 512 | 16 | 16 | 1.941 | 681.7 | False |
| production / 5 | if_tsit5 | swap_const_schedule_gentle_exact_devicepop | 512 | 24 | 16 | 1.326 | 9189 | False |
| production / 293.3333333333333 | if_tsit5 | swap_const_schedule_gentle_exact_devicepop | 128 | 4.464 | 16 | 79.78 | 0.35 | True |
| production / 879.9999999999999 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 9 | 4.464 | 16 | 33.31 | 0.256 | True |
| production / 176.0 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 9 | 4.464 | 16 | 6.753 | 0.3157 | True |
| production / 5 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 128 | 4.464 | 16 | 1.074 | 0.2751 | True |
| production / 5 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 1024 | 4.464 | 16 | 9.015 | 0.9251 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 128 | 4.464 | 16 | 59.46 | 0.3434 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed_direct | 128 | 4.464 | 16 | 59.56 | 0.3434 | True |
| production / 293.3333333333333 | tsit5 | swap_const_schedule_gentle_exact_devicepop_packed | 128 | 4.464 | 16 | 59.46 | 0.3434 | True |

## Full-transit validation

Includes 201 population samples, density at field exit, and density after 25 lifetimes with drives off. Nine distinct cases are detunings -20/0/+20 MHz times J=1/2/3 mixtures; larger batches repeat these cases. All endpoint entries are compared to archived RK4 at 0.48828125 ns. An archived strict CPU reference separately checks J=2 at zero detuning. This does not validate every detuning and velocity over its full transit.

| Method / variant | v m/s | Batch | Base step ns | Total s | Endpoint error % vs finer GPU | Strict CPU subset % | Population max absolute difference | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tsit5 / swap_const_schedule_gentle | 150 | 128 | 4.46429 | 79.26 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_gentle_exact | 150 | 128 | 4.46429 | 70.31 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_gentle_devicepop | 150 | 128 | 4.46429 | 78.35 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_gentle_exact_devicepop | 150 | 128 | 4.46429 | 69.21 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_gentle_exact_devicepop_packed_direct | 150 | 128 | 4.46429 | 59.48 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| if_tsit5 / swap_const_schedule_gentle_exact_devicepop | 150 | 128 | 4.46429 | 79.78 | 0.34996 | 0.0018099 | 1.057e-05 | True |
| tsit5 / swap_const_schedule_gentle_exact_devicepop_packed_direct | 50 | 9 | 4.46429 | 33.31 | 0.25603 | 3.1262e-07 | 1.09e-06 | True |
| tsit5 / swap_const_schedule_gentle_exact_devicepop_packed_direct | 250 | 9 | 4.46429 | 6.753 | 0.31571 | 0.0019519 | 5.012e-07 | True |
| tsit5 / swap_const_schedule_gentle_exact_devicepop_packed_direct | 150 | 128 | 4.46429 | 59.46 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_gentle_exact_devicepop_packed_direct | 150 | 128 | 4.46429 | 59.56 | 0.34343 | 0.0018049 | 7.837e-07 | True |
| tsit5 / swap_const_schedule_gentle_exact_devicepop_packed | 150 | 128 | 4.46429 | 59.46 | 0.34343 | 0.0018049 | 7.837e-07 | True |

Adaptive error control: maximum of per-trajectory RMS complex embedded errors. Accepted/rejected steps, setup/capture/transfer cost, FP32 storage diagnostics and full numeric outputs are in individual JSON/NPZ files. Short-interval fixed/adaptive accuracy is checked against the same archived strict CPU references; a local tolerance alone is not a pass. Full-transit reference scope is stated separately above.

The previous approximately 2.56x CPU throughput estimate used fixed RK4 at 0.9765625 ns and 128 replicas of the historical 200 us fixture. It divided an archived single-CPU runtime by ideal 12-way scaling; it was not a measured 12-worker speedup. Historical and production fields differ, so the production Tsit5/RK4 gain must not be multiplied into that estimate. No CPU ODE timings were rerun in this suite.
