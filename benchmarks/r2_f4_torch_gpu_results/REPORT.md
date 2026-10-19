# PyTorch GPU dense forward benchmark

Three saved reduced 1444-dimensional Liouvillians, pure X 10 mW; all 20 populations; 1401 photon samples. Batch-eight cycles over the same three inputs; no new scan points.

GPU: NVIDIA GeForce RTX 5070 Ti; PyTorch 2.14.1+cu130, CUDA build 13.0. Double precision; autograd disabled.

GPU path: general non-Hermitian eigendecomposition, complex solve for all initial-state columns, analytic cumulative photon projection. CPU comparison uses the optimized reduced real eigenbasis. CUDA is synchronized around each stage. Input stacking/transfers and result download are included; common Liouvillian construction, validation, and file saving excluded.

Full-size GPU first-call warm-up: 0.917 s (separate from steady-state timings).

| Configuration | Median seconds | Parameter points/s |
|---|---:|---:|
| Optimized CPU, one thread, one point | 0.696 | 1.44 |
| PyTorch GPU, batch 1 | 0.561 | 1.78 |
| PyTorch GPU, batch 3 | 1.709 | 1.75 |
| PyTorch GPU, batch 8 | 4.474 | 1.79 |
| Previously measured CPU, eight workers, 32 points | 3.751 | 8.53 |

The eight-worker CPU row is from the previous benchmark, not a simultaneous rerun. It is the relevant scan-throughput reference; beating one CPU thread alone would not establish a scan speedup.

## Accuracy

Maximum single-point photon difference: 6.23e-11. All single-point gates passed: True. Batched maximum scaled photon error: 0.000622 (pass below 1).

Peak PyTorch CUDA allocation: 1.80 GB; reserved: 3.05 GB. Library workspace may have additional allocation.

No differentiability implementation, no production changes, and no full scan. Detailed stage timings and settings: [results.json](results.json).

Reproduce using the isolated benchmark environment: `benchmarks/.venv-gpu-bench/Scripts/python.exe benchmarks/bench_r2_f4_torch_gpu.py`.

## System configuration

Captured after the benchmark on the same system. CPU: AMD Ryzen 7 9800X3D 8-Core Processor; 8 physical cores, 16 logical processors. Reported physical RAM: 61.64 GiB (64 GB installed class). GPU: NVIDIA GeForce RTX 5070 Ti, 16303 MiB reported VRAM; driver 591.86. OS: Windows-10-10.0.26200-SP0. Python 3.11.13, NumPy 2.3.1, SciPy 1.16.0, PyTorch 2.14.1+cu130, CUDA build 13.0. Double precision, autograd disabled. CPU workers use one BLAS thread each. Full BLAS/library details: [system_config.json](system_config.json).
