# GPU integrator selection and optimizations

Stage: complete.
Winner: {'integrator': 'tsit5', 'dt_ns': 3.90625, 'atol': 1e-10, 'rtol': 1e-09}. Retained optimization switches: ['base', 'graph_steps=16'].

Full complex-entry target: 1% above magnitude 1e-6; near-zero absolute errors are reported separately. No CPU ODEs or production changes. Production fixture uses 483 distinct detuning/J-mixture cases repeated only to fill larger batches. Historical fixture uses independent replicas of the archived nominal trajectory.

| Fixture / us | Integrator | Variant | Batch | dt initial/fixed ns | Chunk | Seconds | Entry error % | Pass |
|---|---|---|---:|---:|---:|---:|---:|---|
| production / 1 | rk4 | base | 512 | 0.4883 | 16 | 6.347 | 0.02477 | True |
| production / 1 | rk4 | base | 512 | 0.9766 | 16 | 3.223 | 0.3948 | True |
| production / 1 | rk4 | base | 512 | 1.953 | 16 | 1.659 | 6.259 | False |
| production / 1 | tsit5 | base | 512 | 0.9766 | 16 | 5.51 | 0.000323 | True |
| production / 1 | tsit5 | base | 512 | 1.953 | 16 | 2.847 | 0.006496 | True |
| production / 1 | tsit5 | base | 512 | 3.906 | 16 | 1.515 | 0.3795 | True |
| production / 1 | adaptive_tsit5 | base | 512 | 0.9766 | 16 | 2.9 | 0.02341 | True |
| production / 1 | adaptive_tsit5 | base | 512 | 0.9766 | 16 | 2.094 | 0.3459 | True |
| production / 5 | rk4 | base | 512 | 0.9766 | 16 | 15.77 | 0.6916 | True |
| production / 5 | tsit5 | base | 512 | 3.906 | 16 | 6.863 | 0.4483 | True |
| production / 5 | adaptive_tsit5 | base | 512 | 0.9766 | 16 | 9.165 | 0.3459 | True |
| production / 5 | tsit5 | fused | 512 | 3.906 | 16 | 6.762 | 0.4483 | True |
| production / 5 | tsit5 | reuse | 512 | 3.906 | 16 | 6.854 | 0.4483 | True |
| production / 5 | tsit5 | special | 512 | 3.906 | 16 | 6.864 | 0.4483 | True |
| production / 5 | tsit5 | mixed | 512 | 3.906 | 16 | 7.129 | 0.568 | True |
| production / 5 | tsit5 | base | 512 | 3.906 | 64 | 7.405 | 0.4483 | True |
| production / 5 | tsit5 | base | 512 | 3.906 | 256 | 9.55 | 0.4483 | True |
| production / 5 | tsit5 | base | 512 | 3.906 | 16 | 6.86 | 0.4483 | True |
| production / 5 | tsit5 | base | 128 | 3.906 | 16 | 1.489 | 0.1237 | True |
| production / 5 | tsit5 | base | 1024 | 3.906 | 16 | 14.36 | 0.4483 | True |
| production / 5 | tsit5 | base | 2048 | 3.906 | 16 | 29.2 | 0.4483 | True |
| historical / 1 | tsit5 | base | 128 | 0.9766 | 16 | 1.262 | 0.146 | True |
| historical / 10 | tsit5 | base | 128 | 0.9766 | 16 | 12.28 | 0.1885 | True |
| historical / 200 | tsit5 | base | 128 | 0.9766 | 16 | 245.2 | 0.0004462 | True |
| production / 293.3333333333333 | tsit5 | base | 128 | 3.906 | 16 | 97.59 | 0.1 | True |
| production / 293.3333333333333 | rk4 | base | 128 | 1.953 | 16 | 107 | 3.469 | False |
| production / 293.3333333333333 | rk4 | base | 128 | 0.9766 | 16 | 207.8 | 0.2065 | True |
| production / 879.9999999999999 | tsit5 | base | 9 | 3.906 | 16 | 36.11 | 0.1275 | True |
| production / 176.0 | tsit5 | base | 9 | 3.906 | 16 | 8.635 | 0.04603 | True |

## Full-transit validation

Includes 201 population samples, density at field exit, and density after 25 lifetimes with drives off. Nine distinct cases are detunings -20/0/+20 MHz times J=1/2/3 mixtures; larger batches repeat these cases. All endpoint entries are compared to archived RK4 at 0.48828125 ns. An archived strict CPU reference separately checks J=2 at zero detuning. This does not validate every detuning and velocity over its full transit.

| Method | v m/s | Batch | Step ns | Total s | Endpoint error % vs finer GPU | Strict CPU subset % | Population max absolute difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| tsit5 | 150 | 128 | 3.90625 | 97.59 | 0.1 | 0.0018647 | 3.994e-07 |
| rk4 | 150 | 128 | 1.95312 | 107 | 3.4692 | 0.0022386 | 5.42e-06 |
| rk4 | 150 | 128 | 0.976562 | 207.8 | 0.2065 | 0.0019611 | 2.783e-07 |
| tsit5 | 50 | 9 | 3.90625 | 36.11 | 0.1275 | 2.8743e-07 | 5.793e-07 |
| tsit5 | 250 | 9 | 3.90625 | 8.635 | 0.046029 | 0.0016137 | 2.546e-07 |

Adaptive error control: maximum of per-trajectory RMS complex embedded errors. Accepted/rejected steps, setup/capture/transfer cost, FP32 storage diagnostics and full numeric outputs are in individual JSON/NPZ files. Short-interval fixed/adaptive accuracy is checked against the same archived strict CPU references; a local tolerance alone is not a pass. Full-transit reference scope is stated separately above.

The previous approximately 2.56x CPU throughput estimate used fixed RK4 at 0.9765625 ns and 128 replicas of the historical 200 us fixture. It divided an archived single-CPU runtime by ideal 12-way scaling; it was not a measured 12-worker speedup. Historical and production fields differ, so the production Tsit5/RK4 gain must not be multiplied into that estimate. No CPU ODE timings were rerun in this suite.
