# Public dense session validation

Pure-X 10 mW R(2) F'=4: 32 distinct detunings near 6.8, 28.2 and 80 MHz, 20 independent ground J=2 populations, 1401 photon samples. Eight physical-core workers, one BLAS thread each. Three repeated public grid calls use one prepared model and one session. No full scan rerun.

Fresh public call: 10.138 s. One-time session startup: 8.806 s. Median subsequent call: 3.994 s (8.01 points/s). Complete session lifetime including three calls, startup and teardown: 21.006 s.

Maximum difference from the fresh public API: 0. Every call reported reuse and zero new worker startup. Timings include model fingerprint checks, grouping, parameter binding/extraction, reduction, decomposition, projection, communication and result collation; common OBE construction is excluded.

Additional automated tests cover independent matrix-exponential references, changing parameters/initial support/output/times, restoring defaults, model mutation rejection, worker failure, cleanup and CPU/CUDA switching. Optional CUDA validation runs in the isolated GPU environment.

Detailed timings: [results.json](results.json). Hardware/software provenance: [system_config.json](system_config.json); copied base configuration from the same machine, with current run settings added. Torch/GPU metadata describes the separate GPU environment and hardware, not dependencies used in this CPU timing run.
