# Matched GPU before/after comparison

RTX 5070 Ti, FP64, 1,024 independent trajectories over 10 us, full complex final
density output. Three randomized interleaved timing rounds after warmup. All
variants pass the same full-density accuracy checks. No CPU timings were rerun.

| System | Earlier full-storage Tsit5 | Current default Tsit5 | Current best installed option | Best speedup | Runtime saved |
|---|---:|---:|---:|---:|---:|
| RC, 70 levels | 25.837 s | 18.433 s | 16.569 s, Tsit5 + exact reduction | **1.56x** | **35.9%** |
| R(2) EOM, 40 levels | 14.641 s | 10.382 s | 8.138 s, CK54 without reduction | **1.80x** | **44.4%** |

At the same Tsit5 base step, the current default is 1.40x faster for RC and
1.41x faster for R(2). RC exact reduction saves a further 10.1% relative to the
current default. R(2) exact reduction takes 10.855 s, 4.6% slower than its default;
leave it off there. R(2) CK54 saves 21.6% relative to current Tsit5.

## Baseline and timing scope

The baseline is `rc_gpu_integrators.IntegratorGPU`, `integrator="tsit5"`,
`variant="base"`: the earlier full-matrix split-real storage implementation,
with sparse Hamiltonian products, structured decay and CUDA graphs already in
place. It precedes packed upper-triangle storage, the newer generic library
kernels, waveform precomputation and exact density-coordinate reduction.
This is not a comparison to the very first dense GPU prototype or to RK4.

Current candidates use the publication-validated wheel built from implementation
commit `836fe22`, native SHA256
`42410756074cb150046256f90702d2e529494cce24282bc762c603b5426bb659`.
The actual public batch solve constructs each current integrator; subsequent
graph replays isolate warm evolution/output throughput from repeated compilation.
No production defaults or physical recipes were changed.

Each timing includes reset, integration, device-to-host transfer, and
reconstruction of full complex density matrices. Compilation, graph capture,
fixture loading, accuracy checks and disk I/O are outside these timings. Raw
`setup` fields have different scopes: the original field is construction/capture,
whereas current fields are the complete first public solve including evolution.
Do not compare those fields as setup-only speedups; the current solver's
`chunks[].setup_seconds` supplies its internal setup measurement.

Tsit5 uses a 4.464285714 ns base step for RC and 2 ns for R(2); CK54 uses 1.95 ns
for R(2). Each implementation retains its own boundary clipping. The old engine
clips quarter-microsecond boundaries even for R(2), which has no switched drive.
Consequently the base-step comparison is not a claim of identical step grids.

## Inputs and accuracy

RC uses the archived 250 mW UV / 500 mW per microwave, zero-electric-field,
Earth-magnetic-field model, EOM beta 3.8 at 1.6 MHz and offset 1 MHz microwave
polarization switching. The interval starts at 146 us, around the Gaussian
microwave profile center for 150 m/s. There are 483 distinct combinations:
161 detunings from -20 to +20 MHz at 0.25 MHz spacing and three independent
uniform-within-J mixtures. These are repeated only to fill the 1,024 batch slots;
every slot is independently integrated.

R(2) uses the archived 40-level F'=4 fixture, EOM beta 3.8 at 1.6 MHz, entry time
125 ns, 20 initial states and detunings -20/0/+20 MHz: 60 distinct combinations,
repeated to fill 1,024 slots.

References integrate every distinct combination with Tsit5 at half and quarter
the candidate base step. The quarter-step result is the accuracy reference;
half/quarter differences check convergence independently of old/new agreement.

| System / candidate | Worst relative density-entry error | Largest absolute error below cutoff |
|---|---:|---:|
| RC original | 0.41005% | 4.34e-9 |
| RC current default / reduced | 0.41004% | 4.34e-9 |
| R(2) original / current Tsit5 | <=0.000079% | <=4.72e-13 |
| R(2) current CK54 | 0.005544% | 5.37e-11 |

Relative errors include complex coherences where reference magnitude >=1e-6;
below that cutoff the absolute allowance is 1e-6. Trace errors are below 1.2e-14
and minimum eigenvalues pass the -1e-7 allowance. Half/quarter reference differences
are <=0.00776% for RC and <=0.00000132% for R(2). Same-step current Tsit5 outputs
differ from the original by at most 2.0e-12 (RC) and 1.43e-10 (R(2)) in absolute
complex-entry magnitude.

These are measured short-interval, large-batch throughput gains. They do not
establish the gain for full molecular transits, intermediate output histories,
smaller batches, or an entire production parameter grid. Periodic maps and the
CF4/Magnus prototype are excluded from this comparison.

## Reproduction

```powershell
benchmarks/.venv-gpu-bench/Scripts/python.exe benchmarks/check_gpu_cumulative.py
```

The driver uses `.cache/publication/installed` when present, otherwise the normal
package environment. It requires the archived RC prepared context and R(2)
inputs described in [the evidence index](../PUBLICATION_INDEX.md). Raw medians,
individual rounds, accuracy diagnostics and solver metadata are in
[results.json](results.json). [provenance.json](provenance.json) records input and
implementation hashes. The large original inputs remain local.
