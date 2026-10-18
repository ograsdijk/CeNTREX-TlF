# CPU/GPU pipeline and CUDA concurrency benchmark

Three saved pure-X, 10 mW matrices (6.8, 28.2, 80 MHz); 20 initial populations and 1401 cumulative-photon times. Every task refactors its matrix; no new OBE scans.

Two repetitions. Warm worker pools and CUDA libraries; startup excluded. Timings include task scheduling, CPU process communication, coefficient/input transfers, and photon result download. Common matrix construction and validation/file writing excluded.

The pipeline performs real-eigenbasis decomposition and initial-state solves in eight single-threaded CPU processes. The main process projects coefficients on CUDA in batches while remaining CPU tasks continue. Stream tests use distinct CUDA streams and host threads; there is no per-task device-wide synchronization.

| Configuration | Median seconds | Points/s |
|---|---:|---:|
| cpu8 (32 points) | 4.010 | 7.98 |
| cpu8_gpu_batch1 (32 points) | 3.662 | 8.74 |
| cpu8_gpu_batch8 (32 points) | 3.490 | 9.17 |
| gpu_streams1 (16 points) | 8.927 | 1.79 |
| gpu_processes2 (16 points) | 9.224 | 1.73 |
| gpu_processes4 (16 points) | 9.254 | 1.73 |

All photon accuracy, imaginary-part, and monotonicity checks passed. Maximum photon difference: 6.23e-11.

This is a small repeated-input throughput benchmark, not a full scan or differentiability test. Details: [results.json](results.json).

Concurrent threaded CUDA-stream eigendecomposition failed during warm-up with CUSOLVER_STATUS_INTERNAL_ERROR. Higher thread counts were not pursued; separate processes were tested to isolate library contexts.

## System configuration

Captured after the benchmark on the same system. CPU: AMD Ryzen 7 9800X3D 8-Core Processor; 8 physical cores, 16 logical processors. Reported physical RAM: 61.64 GiB (64 GB installed class). GPU: NVIDIA GeForce RTX 5070 Ti, 16303 MiB reported VRAM; driver 591.86. OS: Windows-10-10.0.26200-SP0. Python 3.11.13, NumPy 2.3.1, SciPy 1.16.0, PyTorch 2.14.1+cu130, CUDA build 13.0. Double precision, autograd disabled. CPU workers use one BLAS thread each. Full BLAS/library details: [system_config.json](system_config.json).
