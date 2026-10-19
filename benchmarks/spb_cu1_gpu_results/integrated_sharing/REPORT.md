# Integrated CUDA coefficient and Hamiltonian sharing

Package implementation in lindblad/_cuda_lowering.py, _cuda_sharing.py, _cuda_kernels.py and cuda.py. Both sharing controls now default to auto. CPU and static dense-eigen methods are unchanged.

## Measured performance

1024 simultaneous trajectories, fixed Tsit5 at 2 ns, direct RHS with exact density reduction, graph_steps=8 and the default 128 MB time-table budget. Three interleaved rounds after warmup for short screens. Public-call time includes setup/capture, integration and final-population transfer; model construction is excluded. CPU load, CUDA event spans and setup statistics are retained in each raw JSON file.

| Workload | Both off, s | Coefficient auto / H off, s | Both auto, s | Off / auto |
|---|---:|---:|---:|---:|
| CU1 varying heights, 12 us | 5.547 | 3.653 | 3.284 | 1.689x |
| CU1 shared envelope, 12 us | 3.266 | 3.230 | 3.232 | 1.011x |
| RC independent height/powers, 1 us | 4.664 | 4.615 | 4.634 | 1.006x |
| RC fixed powers, 1 us | 4.169 | 4.170 | 4.360 | 0.956x |

Forcing RC components took 5.129 s (10.0% slower than off). The additional per-edge FP64 scaling outweighed saved assembly. Auto now declines multi-component/mixed storage when the density dimension is large relative to the coefficient count (n > 3K); hamiltonian_sharing='on' bypasses this arithmetic gate for other-device benchmarks. It still respects structural/real-factor eligibility, the four-component bound and storage checks.

RC coefficient-only gains are small and should not be described as a demonstrated substantial RC speedup. Fixed-power RC had timing variation despite equivalent already-shared H code. Raw samples were retained; the small median differences do not establish an algorithmic regression.

### Full CU1 confirmation

One paired complete 271.739 us scan with 1024 distinct varying-height trajectories; matching numerical and output settings. All final population entries also agree within rtol=1e-10, atol=1e-12.

| Both off, s | Both auto, s | Speedup | Runtime reduction |
|---:|---:|---:|---:|
| 127.561 | 74.660 | 1.709x | 41.5% |

## Memory and correctness

CU1 full-scan time tables: 60,896,192 bytes with both off, 60,896,192 with auto. Off-diagonal H buffers: 2,097,152 to 2,048 bytes. Packed live cache columns avoid the prototype's doubled time table. These are buffer sizes, not peak total device-memory measurements.

62 CUDA/periodic regression tests passed with auto as the default, including independent components with overlapping support, a nonseparable remainder, complex polarization, complex parameter fallback, real-helper validation, zero drives, odd graph tails, bounded/precompute-off caches, chunking, density snapshots and photon/selected output. The focused non-GPU cache-layout/option tests are included. Ruff checks pass for changed Python files.

| Fixture | Worst same-step complex-density absolute difference across tested modes |
|---|---:|
| cu1 | 2.22e-16 |
| rc | 4.16e-16 |
| r2 | 0 |

CU1 also passes the existing independent/converged strict-reference criterion at first/middle/last pass centers and exit. RC checks twelve representative trajectories through 10 us with four density checkpoints. These are optimization-equivalence checks, not a new tolerance/timestep search.

## Fixture scope and limitations

CU1 uses the previously frozen 26-level recipe, uniform twelve-state J=1 mixture, alternating-pass Doppler and X/Z switching. No recipe or optical-carrier default changed.

RC uses the existing cross-system numerical benchmark fixture, including its declared 200 V/cm electric field and 0.01 G axial magnetic field. This is not the present experimental zero-electric-field RC production recipe. The varying case adds a static 7.5 mm Gaussian vertical attenuation to UV and independent relative power factors for UV and each microwave, converted to Rabi factors by square root. It keeps the existing time profiles/EOM and microwave switching; no production scan was launched.

The fresh transverse-field R(2) CPU fixture has ~2.38e-5 rad/s imaginary diagonal roundoff, which the existing CUDA real-diagonal check rejects even with sharing off. The benchmark follows the earlier public GPU validation's real-diagonal projection, checks that correction against the matrix floating-point scale, and records its size in r2_shared_accuracy.json. The package's diagonal validation was not relaxed.

Complex scalar overrides use the old paths to preserve conjugation and real-helper checks. Time-dependent nonseparable factors use a per-trajectory remainder when worthwhile or the complete old H path. Auto is a conservative dependence/storage/arithmetic heuristic, not a device-specific timing tuner or a guarantee of fastest execution.

## Reproduction and evidence

Driver: benchmarks/bench_cuda_sharing_integration.py --fixture cu1|rc|r2 --stage accuracy|timing|full [--varying] [--force-components]. Full stage is CU1 varying-height only. RC timing includes explicit forced components to expose the regression auto avoids. Raw results carry package source hashes and chunk diagnostics. The initial RC component timings remain in rc_varying_timing_initial_components.json; final timings record the revised auto policy.

Tests: benchmarks/.venv-gpu-bench/Scripts/python.exe -m pytest tests/lindblad/test_cuda.py tests/lindblad/test_periodic.py -q -p no:cacheprovider (CUPY_CACHE_DIR set to benchmarks/.cupy_cache). Report: benchmarks/report_cuda_sharing_integration.py.

The package implementation and acceptance evidence are included in the 0.3.0 release preparation.
