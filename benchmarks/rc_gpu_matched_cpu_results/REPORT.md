# Matched historical RC CPU/GPU comparison

Suite status: complete.

The archived nominal CPU cases are reproduced on GPU: 70 states, E=[0,0,200] V/cm, B=[0,0,0.01] G, pure X J=2 F1=5/2 F=2 mF=0 initial state, constant optical and microwave Rabi rates each equal to Gamma, coherent EOM 1.6 MHz beta=3.8, 1 MHz microwave polarization switching with 250 ns offset, absolute entry time 125 ns. These are historical numerical fixtures, not production experimental fields.

GPU: fixed-step FP64 RK4 and staged sparse diagonal/real-imaginary implementation. CPU: archived native Rust D+S+U / adaptive Tsit5 full-final-density timings. No CPU ODEs were rerun. Each GPU batch contains independently integrated replicas of the corresponding single CPU benchmark case; no trajectory results are reused. GPU RHS checked against the original CPU basis at all four switching configurations, including arbitrary coherences. No density clipping or renormalization.

## Measured timings and accuracy

| Duration (us) | Batch | dt (ns) | GPU median (s) | Repeats | CPU single trajectory (s) | Serial CPU throughput ratio | Ratio assuming ideal 12-thread CPU scaling | GPU worst complex-entry error | CPU archived error | Both within 1%? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 128 | 0.4883 | 1.532 | 3 | 0.3017 | 25.22x | 2.10x | 0.5726% | 0.1677% | Yes |
| 1 | 1024 | 0.4883 | 13.83 | 3 | 0.3017 | 22.34x | 1.86x | 0.5726% | 0.1677% | Yes |
| 1 | 128 | 0.9766 | 0.7693 | 1 | 0.3017 | 50.20x | 4.18x | 6.145% | 0.1677% | No |
| 10 | 128 | 0.4883 | 15.28 | 3 | 1.8 | 15.08x | 1.26x | 0.8562% | 4.781% | No |
| 10 | 512 | 0.4883 | 65.71 | 3 | 1.8 | 14.03x | 1.17x | 0.8562% | 4.781% | No |
| 200 | 128 | 0.4883 | 306.1 | 1 | 36.6 | 15.30x | 1.28x | 0.0008653% | 0.00557% | Yes |
| 200 | 128 | 0.9766 | 152.5 | 1 | 36.6 | 30.72x | 2.56x | 0.003471% | 0.00557% | Yes |

Errors compare full complex entries, including phase and coherences, where the archived reference magnitude is at least 1e-6. GPU is checked against both archived strict Tsit5 and Dopri5 references; the larger error is tabulated. CPU error is recomputed from the archived accepted output against its strict Tsit5 reference. Below-cutoff absolute errors and trace, Hermiticity and eigenvalue diagnostics are in each JSON.

## Interpretation limits

The 12-thread column is an arithmetic comparison assuming ideal linear scaling; it is not a measured 12-thread CPU benchmark. The GPU batch times are measured. Historical CPU timings exclude model setup; GPU timing excludes compilation, capture and final host transfer, with full density retained on device. Current GPU runs can have production CPU activity in parallel; the archived CPU benchmark had low background activity. The rebuilt fixture and historical native binary/source provenance are recorded, and direct archived-density agreement is checked rather than assumed.

The archived 10-us CPU timing used rtol=1e-7, atol=1e-9 and fails the subsequently adopted 1% complex-entry target. Its timing is included for provenance, not an accuracy-matched speedup claim. The 200-us CPU accepted timing used rtol=1e-9, atol=1e-11; 1-us CPU used rtol=1e-7, atol=1e-9. The long GPU cases are single timing observations, while the listed short cases have three repetitions.

The coarse 0.977-ns GPU pilot is preserved as a failed-accuracy result. It must not be used as a qualified speedup. These fixtures have one nominal parameter point; they do not measure production full-grid throughput.

## Sources

- 1 us CPU: [screen.json](C:/Users/ogras/Documents/GitHub/CeNTREX-TlF/benchmarks/rc_propagation_results/cross_system_screen_v3/nominal/rc/screen.json); references and accepted output arrays alongside that archived suite.
- 10 us CPU: [result.json](C:/Users/ogras/Documents/GitHub/CeNTREX-TlF/benchmarks/rc_propagation_results/cross_system_long_finalists_v1/suite10us_v1/rc/10us/result.json); references and accepted output arrays alongside that archived suite.
- 200 us CPU: [result.json](C:/Users/ogras/Documents/GitHub/CeNTREX-TlF/benchmarks/rc_propagation_results/cross_system_long_finalists_v1/suite_long_v1/rc/200us/result.json); references and accepted output arrays alongside that archived suite.

GPU raw timings, density outputs and per-case diagnostics are alongside this report.

## Exact archived parallel initial-state batches

These compare the same initial-state list, nominal parameters, duration and full-density output against measured archived eight-worker CPU timings. CPU timing is a single observation; GPU timing is a median of three.

| Initial states | GPU dt (ns) | GPU seconds | Archived 8-worker CPU seconds | Measured speed ratio | GPU entry error | CPU entry error | Both within 1%? |
|---|---:|---:|---:|---:|---:|---:|---|
| K20_driven_X_J2 | 0.2441 | 0.6936 | 1.089 | 1.57x | 0.1315% | 0.8687% | Yes |
| K20_driven_X_J2 | 0.4883 | 0.3481 | 1.089 | 3.13x | 2.061% | 0.8687% | No |

The saved batch reference was validated by the original CPU suite against Tsit5 and Dopri5; one reference array is archived per initial-state batch. Individual JSONs link the CPU timing source and list exact initial indices.
