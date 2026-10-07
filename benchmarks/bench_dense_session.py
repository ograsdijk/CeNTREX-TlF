"""Validate and time the public reusable session on a small F4 grid."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from centrex_tlf import lindblad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reports/r2_f4_earth_field/obe"))
import scan

OUT = Path(__file__).resolve().parent / "dense_session_results"


def main():
    OUT.mkdir(exist_ok=True)
    _, prepared, _, rabis, excited, _, _ = scan.build(0.0)
    with np.load(
        Path(__file__).resolve().parent / "r2_f4_dense_threading_results/point_0_inputs.npz"
    ) as saved:
        packed = saved["packed"].T.copy()
    detunings = np.array([[6.8, 28.2, 80.0][i % 3] + i * 0.0001 for i in range(32)])
    options = dict(
        rho0_batch=packed,
        scan={"detuning": 2 * np.pi * 1e6 * detunings, "rabi": [rabis[1]]},
        solver="dense_eig",
        output="photon_integral",
        output_when="saveat",
        saveat=scan.TIMES,
        integral_weights=[(i, scan.GAMMA) for i in excited],
        parallel=True,
        collect_stats=True,
    )
    report = dict(
        generated_at=datetime.now().astimezone().isoformat(), tasks=32, workers=8, repeated_calls=[]
    )
    start = time.perf_counter()
    baseline = lindblad.grid_scan(prepared, None, (0.0, 350e-6), threads=8, **options)
    report["fresh_call_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    with lindblad.DenseLindbladSession(prepared, threads=8) as session:
        report["session_startup_seconds"] = session.startup_seconds
        for repeat in range(3):
            tick = time.perf_counter()
            answer = lindblad.grid_scan(
                prepared, None, (0.0, 350e-6), dense_session=session, **options
            )
            elapsed = time.perf_counter() - tick
            error = float(abs(answer.values - baseline.values).max())
            assert error < 1e-7 and answer.solver_stats["pool_reused"]
            assert answer.solver_stats["worker_startup_seconds"] == 0.0
            report["repeated_calls"].append(
                dict(
                    repeat=repeat,
                    seconds=elapsed,
                    points_per_second=32 / elapsed,
                    max_error=error,
                    stats=answer.solver_stats,
                )
            )
            print(
                f"Reusable session {repeat}: {elapsed:.3f}s, {32 / elapsed:.2f} points/s, error {error:.3g}",
                flush=True,
            )
    report["session_lifetime_seconds_including_teardown"] = time.perf_counter() - start
    median = float(np.median([r["seconds"] for r in report["repeated_calls"]]))
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    config = json.loads(
        (
            Path(__file__).resolve().parent / "dense_library_integration_results/system_config.json"
        ).read_text()
    )
    config["current_benchmark"] = dict(
        interpreter=".venv/Scripts/python.exe", cpu_only=True, blas_threads_per_worker=1
    )
    (OUT / "system_config.json").write_text(json.dumps(config, indent=2))
    lines = [
        "# Public dense session validation",
        "",
        "Pure-X 10 mW R(2) F'=4: 32 distinct detunings near 6.8, 28.2 and 80 MHz, 20 independent ground J=2 populations, 1401 photon samples. Eight physical-core workers, one BLAS thread each. Three repeated public grid calls use one prepared model and one session. No full scan rerun.",
        "",
        f"Fresh public call: {report['fresh_call_seconds']:.3f} s. One-time session startup: {report['session_startup_seconds']:.3f} s. Median subsequent call: {median:.3f} s ({32 / median:.2f} points/s). Complete session lifetime including three calls, startup and teardown: {report['session_lifetime_seconds_including_teardown']:.3f} s.",
        "",
        f"Maximum difference from the fresh public API: {max(r['max_error'] for r in report['repeated_calls']):.3g}. Every call reported reuse and zero new worker startup. Timings include model fingerprint checks, grouping, parameter binding/extraction, reduction, decomposition, projection, communication and result collation; common OBE construction is excluded.",
        "",
        "Additional automated tests cover independent matrix-exponential references, changing parameters/initial support/output/times, restoring defaults, model mutation rejection, worker failure, cleanup and CPU/CUDA switching. Optional CUDA validation runs in the isolated GPU environment.",
        "",
        "Detailed timings: [results.json](results.json). Hardware/software provenance: [system_config.json](system_config.json); copied base configuration from the same machine, with current run settings added. Torch/GPU metadata describes the separate GPU environment and hardware, not dependencies used in this CPU timing run.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
