"""Small CPU experiments only; no changes to public solver implementation."""

import inspect
import json
import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad import dense

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reports/r2_f4_earth_field/obe"))
import scan

OUT = Path(__file__).resolve().parent / "dense_pool_reuse_results"
LOCAL = threading.local()
LIMIT = None


class StructureCache:
    """Exact pattern/support validation before reusing reduction and buffers."""

    def prepare(self, matrix, layout, packed):
        support = None if packed is None else np.any(packed != 0, axis=0)
        hit = (
            hasattr(self, "indices")
            and np.array_equal(self.indices, matrix.indices)
            and np.array_equal(self.indptr, matrix.indptr)
            and np.array_equal(self.support, support)
        )
        if not hit:
            csc = matrix.tocsc()
            self.sinks = np.array(
                [i for i in range(layout.n) if csc.indptr[i] == csc.indptr[i + 1]], int
            )
            reachable = np.ones(layout.packed_len, bool) if support is None else support.copy()
            frontier = list(np.flatnonzero(reachable))
            while frontier:
                col = frontier.pop()
                for row in csc.indices[csc.indptr[col] : csc.indptr[col + 1]]:
                    if not reachable[row]:
                        reachable[row] = True
                        frontier.append(row)
            reachable[self.sinks] = False
            self.keep = np.flatnonzero(reachable)
            self.excluded = np.setdiff1d(np.arange(layout.packed_len), np.r_[self.keep, self.sinks])
            assert matrix[self.excluded][:, np.r_[self.keep, self.sinks]].nnz == 0
            row = np.repeat(np.arange(layout.packed_len), np.diff(matrix.indptr))
            col = matrix.indices
            mapping = np.full(layout.packed_len, -1, int)
            mapping[self.keep] = np.arange(len(self.keep))
            active = (mapping[row] >= 0) & (mapping[col] >= 0)
            self.active_data = np.flatnonzero(active)
            self.active_row, self.active_col = mapping[row[active]], mapping[col[active]]
            smap = np.full(layout.packed_len, -1, int)
            smap[self.sinks] = np.arange(len(self.sinks))
            feed = (smap[row] >= 0) & (mapping[col] >= 0)
            self.feed_data = np.flatnonzero(feed)
            self.feed_row, self.feed_col = smap[row[feed]], mapping[col[feed]]
            self.A = np.empty((len(self.keep), len(self.keep)), order="F")
            self.feed = np.empty((len(self.sinks), len(self.keep)))
            self.R = np.empty(self.A.shape, order="F")
            self.indices, self.indptr = matrix.indices.copy(), matrix.indptr.copy()
            self.support = None if support is None else support.copy()
        self.A.fill(0)
        self.A[self.active_row, self.active_col] = matrix.data[self.active_data]
        self.feed.fill(0)
        self.feed[self.feed_row, self.feed_col] = matrix.data[self.feed_data]
        return self.sinks, self.keep, self.excluded, self.A


# Preserve the production eig/LU/conditioning/residual code verbatim. Only the
# exact structural reduction, dense fill, R allocation and sink feed are changed.
source = inspect.getsource(dense._factor)
begin = source.index("    csc = matrix.tocsc()")
end = source.index("    if len(keep):", begin)
source = (
    source[:begin]
    + "    sinks, keep, excluded, A = cache.prepare(matrix, layout, packed)\n"
    + source[end:]
)
source = source.replace("def _factor(", "def cached_factor(").replace(
    "    start = time.perf_counter()", "    cache = LOCAL.cache\n    start = time.perf_counter()"
)
source = source.replace('R = np.empty(A.shape, order="F")', "R = cache.R")
source = source.replace("matrix[sinks][:, keep] @ R", "cache.feed @ R")
namespace = dict(vars(dense), LOCAL=LOCAL)
exec(source, namespace)
cached_factor = namespace["cached_factor"]


def initialize(payload, packed, options, times, barrier=None, process=False):
    global LIMIT
    if process:
        LIMIT = threadpool_limits(limits=1)
    # Construct unsendable PyO3 evaluator on its owning thread/process.
    LOCAL.extractor = dense._StaticLiouvillian(payload, "expanded_sparse")
    LOCAL.packed, LOCAL.options, LOCAL.times = packed, options, times
    LOCAL.cache = StructureCache()
    LOCAL.barrier = barrier


def ready():
    LOCAL.barrier.wait(timeout=120)
    return True


def task(item):
    parameters, reuse = item
    tick = time.perf_counter()
    matrix = LOCAL.extractor.matrix(parameters)
    extraction = time.perf_counter() - tick
    prop = (
        cached_factor(matrix, LOCAL.extractor.layout, LOCAL.packed)
        if reuse
        else dense._factor(matrix, LOCAL.extractor.layout, LOCAL.packed)
    )
    tick = time.perf_counter()
    answer = prop.evaluate(LOCAL.packed, LOCAL.times, **LOCAL.options)
    return answer, dict(
        extraction=extraction,
        factor=prop.stats["factorization_seconds"],
        projection=time.perf_counter() - tick,
    )


def run(pool, parameters, reuse, reference):
    tick = time.perf_counter()
    results = list(pool.map(task, [(p, reuse) for p in parameters]))
    elapsed = time.perf_counter() - tick
    values = np.stack([r[0] for r in results])
    error = 0.0 if reference is None else float(abs(values - reference).max())
    assert error < 1e-7, error
    return values, dict(
        seconds=elapsed,
        points_per_second=len(parameters) / elapsed,
        max_error=error,
        stage_medians={k: float(np.median([r[1][k] for r in results])) for k in results[0][1]},
    )


def write_report():
    data = json.loads((OUT / "results.json").read_text())
    modes = [
        "fresh_process",
        "persistent_process",
        "persistent_process_cached_structure",
        "thread",
        "thread_cached_structure",
    ]
    timings = {}
    lines = [
        "# Dense CPU pool and structure reuse experiments",
        "",
        "Small R(2) F'=4 benchmark: pure X polarization, 10 mW, 32 distinct detunings near 6.8, 28.2 and 80 MHz; all 20 ground J=2 initial populations and 1401 cumulative-photon times. Same field-dressed model as the existing scan. No full scan or library changes.",
        "",
        "Eight workers, one BLAS thread per worker. Three repeats per mode; cache and baseline order alternates. Every parameter point gets a new eigendecomposition; no response or decomposition caching. Native evaluators are created on their owning thread/process.",
        "",
        "| Mode | Median 32-point seconds | Points/s | Range (s) |",
        "|---|---:|---:|---:|",
    ]
    for mode in modes:
        rows = [r for r in data["measurements"] if r["mode"] == mode]
        key = "cold_seconds_including_teardown" if mode == "fresh_process" else "seconds"
        values = [r[key] for r in rows]
        timings[mode] = float(np.median(values))
        lines.append(
            f"| {mode} | {timings[mode]:.3f} | {32 / timings[mode]:.2f} | {min(values):.3f}-{max(values):.3f} |"
        )
    startup = data["persistent_process_startup_seconds"]
    lines += [
        "",
        f"Persistent-process startup/readiness: {startup:.3f} s once. Thread startup/readiness: {data['thread_startup_seconds']:.3f} s once. Warm timings include extraction, reduction, conditioning/residual checks, decomposition, photon projection, communication and result collection. Common OBE construction and pool teardown are excluded from warm timings. Fresh-process timing includes startup and teardown and lets ready workers begin immediately; persistent readiness uses a barrier to measure startup separately.",
        "",
        f"Subsequent calls with persistent processes are {timings['fresh_process'] / timings['persistent_process']:.2f}x faster than fresh calls. For three calls, illustrative totals using measured median calls are {3 * timings['fresh_process']:.2f} s fresh versus {startup + 3 * timings['persistent_process']:.2f} s persistent plus its one-time startup (persistent teardown excluded). This is mainly an improvement for repeated calls; a single long scan already amortizes worker startup.",
        "",
        f"Threads are {timings['thread'] / timings['persistent_process']:.2f}x slower than warm processes on this machine. The experiment establishes the timing, not the underlying contention mechanism.",
        "",
        f"Exact structure/buffer reuse gives {timings['persistent_process'] / timings['persistent_process_cached_structure']:.3f}x warm-process throughput. This is within the observed run-to-run range; no reliable additional speedup established. The prototype caches reachability/index maps and reuses the dense generator, sink-feed and real-eigenbasis buffers. It compares the full CSR nonzero pattern and initial-support mask on every point, rebuilding whenever either changes. It retains all production eigenbasis condition/residual checks.",
        "",
        f"Accuracy: maximum saved-reference photon error {data['saved_reference_error']:.3g}; maximum difference between timing variants {max(r['max_error'] for r in data['measurements']):.3g}. Zero-Rabi and zero-detuning pattern changes agree with uncached calculations (process error {data['persistent_process_zero_parameter_check']:.3g}, thread error {data['thread_zero_parameter_check']:.3g}); changed and empty initial-support checks passed. Comparisons cover all 20 populations and all photon times. Existing full-density reconstruction code is unchanged; this experiment does not add new full-density tests.",
        "",
        "Recommendation: persistent process pools for repeated scans/fitting calls; retain processes over threads on this machine. Defer structure/buffer caching unless another workload demonstrates a worthwhile gain.",
        "",
        "Hardware/software provenance: [system_config.json](system_config.json), copied from the existing benchmark on this same machine; its recorded BLAS default thread counts are not the benchmark setting (one thread here). GPU and Torch fields are hardware/previous-environment information; this benchmark uses the main CPU-only virtualenv and does not import Torch. Raw results: [results.json](results.json).",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    _, prepared, _, rabis, excited, _, _ = scan.build(0.0)
    with np.load(
        Path(__file__).resolve().parent / "r2_f4_dense_threading_results/point_0_inputs.npz"
    ) as saved:
        packed = saved["packed"].T.copy()
    options = dict(
        output="photon_integral",
        output_indices=None,
        integral_weights=[(i, scan.GAMMA) for i in excited],
        t0=0.0,
    )
    args = (prepared.to_payload(), packed, options, scan.TIMES)
    parameters = [
        dict(detuning=2 * np.pi * 1e6 * ([6.8, 28.2, 80.0][i % 3] + i * 0.0001), rabi=rabis[1])
        for i in range(32)
    ]
    report = dict(
        generated_at=datetime.now().astimezone().isoformat(),
        workers=8,
        tasks=32,
        initial_states=len(packed),
        samples=len(scan.TIMES),
        measurements=[],
    )
    reference = None
    context = mp.get_context("spawn")
    with threadpool_limits(limits=1):
        # Fresh process pool per API-sized call: default unsynchronized scheduling.
        for repeat in range(3):
            tick = time.perf_counter()
            with ProcessPoolExecutor(
                8, mp_context=context, initializer=initialize, initargs=(*args, None, True)
            ) as pool:
                values, row = run(pool, parameters, False, reference)
            row["cold_seconds_including_teardown"] = time.perf_counter() - tick
            if reference is None:
                reference = values
                with np.load(
                    Path(__file__).resolve().parent
                    / "r2_f4_dense_improvements_results/point_0_responses.npz"
                ) as saved:
                    report["saved_reference_error"] = float(
                        abs(values[0, ..., 0] - saved["active_real_photons"]).max()
                    )
                assert report["saved_reference_error"] < 1e-7
            report["measurements"].append(dict(mode="fresh_process", repeat=repeat, **row))
            print("fresh_process", repeat, row, flush=True)
        for mode in ["persistent_process", "thread"]:
            barrier = context.Barrier(8) if mode == "persistent_process" else threading.Barrier(8)
            tick = time.perf_counter()
            factory = ProcessPoolExecutor if mode == "persistent_process" else ThreadPoolExecutor
            extra = dict(mp_context=context) if mode == "persistent_process" else {}
            with factory(
                8,
                initializer=initialize,
                initargs=(*args, barrier, mode == "persistent_process"),
                **extra,
            ) as pool:
                futures = [pool.submit(ready) for _ in range(8)]
                for f in futures:
                    f.result()
                startup = time.perf_counter() - tick
                report[mode + "_startup_seconds"] = startup
                for repeat in range(3):
                    # Alternate baseline/cache order to reduce thermal/order bias.
                    for reuse in [False, True] if repeat % 2 == 0 else [True, False]:
                        _, row = run(pool, parameters, reuse, reference)
                        name = mode + ("_cached_structure" if reuse else "")
                        report["measurements"].append(dict(mode=name, repeat=repeat, **row))
                        print(name, repeat, row, flush=True)
                # Force structural change and changed support: exact invalidation.
                probes = [
                    dict(parameters[0], rabi=0.0),
                    dict(parameters[0], detuning=0.0),
                    parameters[0],
                ]
                expected, _ = run(pool, probes, False, None)
                _, check = run(pool, probes, True, expected)
                report[mode + "_zero_parameter_check"] = check["max_error"]
        # Explicit changed initial-support cache invalidation, including empty support.
        initialize(*args)
        matrix = LOCAL.extractor.matrix(parameters[0])
        for initial in [packed[:1], np.zeros_like(packed[:1]), packed]:
            a = dense._factor(matrix, prepared.layout, initial).evaluate(
                initial, scan.TIMES, **options
            )
            b = cached_factor(matrix, prepared.layout, initial).evaluate(
                initial, scan.TIMES, **options
            )
            assert float(abs(a - b).max()) < 1e-7
        report["changed_initial_support_check"] = "passed"
    config = json.loads(
        (
            Path(__file__).resolve().parent / "dense_library_integration_results/system_config.json"
        ).read_text()
    )
    (OUT / "system_config.json").write_text(json.dumps(config, indent=2))
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    write_report()


if __name__ == "__main__":
    main()
