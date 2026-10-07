# Dense CPU pool and structure reuse experiments

Small R(2) F'=4 benchmark: pure X polarization, 10 mW, 32 distinct detunings near 6.8, 28.2 and 80 MHz; all 20 ground J=2 initial populations and 1401 cumulative-photon times. Same field-dressed model as the existing scan. No full scan or library changes.

Eight workers, one BLAS thread per worker. Three repeats per mode; cache and baseline order alternates. Every parameter point gets a new eigendecomposition; no response or decomposition caching. Native evaluators are created on their owning thread/process.

| Mode | Median 32-point seconds | Points/s | Range (s) |
|---|---:|---:|---:|
| fresh_process | 10.105 | 3.17 | 10.027-10.357 |
| persistent_process | 3.991 | 8.02 | 3.912-4.011 |
| persistent_process_cached_structure | 3.963 | 8.08 | 3.893-4.005 |
| thread | 19.986 | 1.60 | 19.913-20.008 |
| thread_cached_structure | 19.841 | 1.61 | 19.800-19.911 |

Persistent-process startup/readiness: 8.491 s once. Thread startup/readiness: 0.106 s once. Warm timings include extraction, reduction, conditioning/residual checks, decomposition, photon projection, communication and result collection. Common OBE construction and pool teardown are excluded from warm timings. Fresh-process timing includes startup and teardown and lets ready workers begin immediately; persistent readiness uses a barrier to measure startup separately.

Subsequent calls with persistent processes are 2.53x faster than fresh calls. For three calls, illustrative totals using measured median calls are 30.32 s fresh versus 20.46 s persistent plus its one-time startup (persistent teardown excluded). This is mainly an improvement for repeated calls; a single long scan already amortizes worker startup.

Threads are 5.01x slower than warm processes on this machine. The experiment establishes the timing, not the underlying contention mechanism.

Exact structure/buffer reuse gives 1.007x warm-process throughput. This is within the observed run-to-run range; no reliable additional speedup established. The prototype caches reachability/index maps and reuses the dense generator, sink-feed and real-eigenbasis buffers. It compares the full CSR nonzero pattern and initial-support mask on every point, rebuilding whenever either changes. It retains all production eigenbasis condition/residual checks.

Accuracy: maximum saved-reference photon error 1.9e-08; maximum difference between timing variants 0. Zero-Rabi and zero-detuning pattern changes agree with uncached calculations (process error 0, thread error 0); changed and empty initial-support checks passed. Comparisons cover all 20 populations and all photon times. Existing full-density reconstruction code is unchanged; this experiment does not add new full-density tests.

Recommendation: persistent process pools for repeated scans/fitting calls; retain processes over threads on this machine. Defer structure/buffer caching unless another workload demonstrates a worthwhile gain.

Hardware/software provenance: [system_config.json](system_config.json), copied from the existing benchmark on this same machine; its recorded BLAS default thread counts are not the benchmark setting (one thread here). GPU and Torch fields are hardware/previous-environment information; this benchmark uses the main CPU-only virtualenv and does not import Torch. Raw results: [results.json](results.json).
