"""Saved-matrix benchmarks: CPU/GPU pipeline and explicit CUDA streams."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import argparse
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
import bench_r2_f4_dense_improvements as improved

OUT = Path(__file__).resolve().parent / 'r2_f4_gpu_concurrency_results'
DATA = {}
LIMIT = None
GPU_STREAM = None


def initialize_cpu():
    global LIMIT
    LIMIT = threadpool_limits(limits=1)
    for i in range(3):
        with np.load(improved.OUT / f'point_{i}_worker_inputs.npz') as a:
            DATA[i] = {k: a[k].copy() for k in a.files}


def ready(i):
    time.sleep(.05)
    return i


def initialize_gpu():
    global GPU_STREAM
    import torch
    initialize_cpu()
    torch.set_num_threads(1)
    GPU_STREAM = torch.cuda.Stream()
    stream_work(0, [0], [DATA[i] for i in range(3)], GPU_STREAM)


def gpu_process_work(task):
    return stream_work(0, [task], [DATA[i] for i in range(3)], GPU_STREAM)[0]


def coefficients(a):
    # Same real eigenbasis as the validated CPU prototype, without time projection.
    w, V = la.eig(a['A'], check_finite=False)
    real = np.flatnonzero(w.imag == 0)
    pos = np.flatnonzero(w.imag > 0)
    neg = pos + 1
    np.testing.assert_array_equal(w[neg], w[pos].conj())
    R = np.empty(a['A'].shape, dtype=float, order='F')
    R[:, real] = V[:, real].real
    R[:, pos] = V[:, pos].real
    R[:, neg] = V[:, pos].imag
    C = la.lu_solve(la.lu_factor(R, check_finite=False), a['packed'], check_finite=False)
    obs = improved.previous.prior.scan.GAMMA * R[a['excited']].sum(axis=0)
    wr = (obs[real, None] * C[real]).T.astype(complex)
    wp = ((obs[pos] + 1j * obs[neg])[:, None] * (C[pos] - 1j * C[neg])).T
    # Real modes plus one representative of each conjugate pair.
    return np.concatenate((w[real], w[pos])), np.concatenate((wr, wp), axis=1)


def cpu_work(task, hybrid):
    a = DATA[task % 3]
    if hybrid:
        return task, coefficients(a)
    photons, _, _ = improved.real_factor(a['A'], a['packed'], a['excited'], a['times'])
    return task, photons


def gpu_project(items, times):
    import torch
    with torch.inference_mode():
        # Different matrices can have different counts of real/conjugate modes.
        # Pad with zero-weight modes so heterogeneous parameter points can batch.
        modes = max(len(x[1][0]) for x in items)
        ws = np.zeros((len(items), modes), dtype=complex)
        cs = np.zeros((len(items), items[0][1][1].shape[0], modes), dtype=complex)
        for i, (_, (wi, ci)) in enumerate(items):
            ws[i, :len(wi)] = wi
            cs[i, :, :len(wi)] = ci
        w = torch.as_tensor(ws, device='cuda')
        weights = torch.as_tensor(cs, device='cuda')
        wt = w[..., None] * times
        denom = torch.where(w.abs() > 0, w, torch.ones_like(w))[..., None]
        phi = torch.where(wt.abs() < 1e-7, times * (1 + wt/2 + wt**2/6), torch.expm1(wt)/denom)
        out = (weights @ phi).real.cpu().numpy()
        return [(x[0], out[i]) for i, x in enumerate(items)]


def stream_work(lane, task_ids, arrays, stream):
    import torch
    results = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        for task in task_ids:
            a = arrays[task % 3]
            # Each task uploads and decomposes afresh: no factorization caching.
            A = torch.as_tensor(a['A'], device='cuda', dtype=torch.float64)
            packed = torch.as_tensor(a['packed'], device='cuda', dtype=torch.complex128)
            times = torch.as_tensor(a['times'], device='cuda', dtype=torch.float64)
            w, V = torch.linalg.eig(A)
            C = torch.linalg.solve(V, packed)
            obs = improved.previous.prior.scan.GAMMA * V[a['excited']].sum(dim=0)
            wt = w[:, None]*times
            denom = torch.where(w.abs() > 0, w, torch.ones_like(w))[:, None]
            phi = torch.where(wt.abs() < 1e-7, times*(1+wt/2+wt**2/6), torch.expm1(wt)/denom)
            photons = ((obs[:, None]*C).T @ phi).cpu().numpy()
            stream.synchronize()
            results.append((task, photons))
    return results


def check(values, references):
    errors = [float(np.max(abs(v.real-references[i % 3]))) for i, v in values]
    imaginary = max(float(abs(v.imag).max()) for _, v in values)
    increment = min(float(np.diff(v.real, axis=-1).min()) for _, v in values)
    assert len({i for i, _ in values}) == len(values)
    assert max(errors) < 1e-7 and imaginary < 1e-7 and increment > -1e-7
    return dict(max_photon_error=max(errors), max_imaginary=imaginary,
                min_photon_increment=increment, passed=True)


def main():
    import torch
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    arrays, refs = [], []
    for i in range(3):
        with np.load(improved.OUT / f'point_{i}_worker_inputs.npz') as a:
            arrays.append({k: a[k].copy() for k in a.files})
        with np.load(improved.OUT / f'point_{i}_responses.npz') as a:
            refs.append(a['active_real_photons'].copy())
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    resume = parser.parse_args().resume
    result = (json.loads((OUT/'results.json').read_text(encoding='utf-8')) if resume else
              dict(generated_at=datetime.now().astimezone().isoformat(),
                   gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, rows=[]))
    def record(name, repeat, tasks, wall, values):
        row = dict(configuration=name, repeat=repeat, tasks=tasks, wall_s=wall,
                   points_per_second=tasks/wall, **check(values, refs))
        result['rows'].append(row)
        (OUT/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'{name} repeat {repeat}: {wall:.3f} s, {tasks/wall:.2f} points/s; error {row["max_photon_error"]:.3g}', flush=True)

    with threadpool_limits(limits=1):
        # Startup, input loading, and CUDA warm-up are excluded from steady-state timing.
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'),
                                 initializer=initialize_cpu) as pool:
            list(pool.map(ready, range(32)))
            times = torch.as_tensor(arrays[0]['times'], device='cuda')
            warm = pool.submit(cpu_work, 0, True).result()
            check(gpu_project([warm], times), refs)
            torch.cuda.synchronize()
            for repeat in range(2):
                # Reverse order on the second repetition to limit ordering bias.
                configs = [('cpu8', 0), ('cpu8_gpu_batch1', 1), ('cpu8_gpu_batch8', 8)]
                if repeat: configs.reverse()
                for name, batch in configs:
                    if any(r['configuration']==name and r['repeat']==repeat for r in result['rows']):
                        continue
                    tick = time.perf_counter()
                    futures = [pool.submit(cpu_work, i, bool(batch)) for i in range(32)]
                    values, waiting = [], []
                    for future in as_completed(futures):
                        item = future.result()
                        if not batch:
                            values.append(item)
                        else:
                            waiting.append(item)
                            if len(waiting) == batch:
                                values.extend(gpu_project(waiting, times)); waiting=[]
                    if waiting: values.extend(gpu_project(waiting, times))
                    torch.cuda.synchronize()
                    record(name, repeat, 32, time.perf_counter()-tick, values)

        for lanes in [1, 2, 4]:
            if all(any(r['configuration']==f'gpu_streams{lanes}' and r['repeat']==rep
                       for r in result['rows']) for rep in range(2)):
                continue
            streams = [torch.cuda.Stream() for _ in range(lanes)]
            try:
                with ThreadPoolExecutor(max_workers=lanes) as pool:
                    # Warm every host-thread/stream lane before measurement.
                    warm = [pool.submit(stream_work, i, [i], arrays, streams[i]) for i in range(lanes)]
                    for f in warm: check(f.result(), refs)
                    for repeat in range(2):
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        futures = [pool.submit(stream_work, i, list(range(i, 16, lanes)), arrays, streams[i])
                                   for i in range(lanes)]
                        values = [item for f in futures for item in f.result()]
                        torch.cuda.synchronize()
                        record(f'gpu_streams{lanes}', repeat, 16, time.perf_counter()-tick, values)
            except RuntimeError as exc:
                result.setdefault('failed_configurations', []).append(
                    dict(configuration=f'gpu_streams{lanes}', error=str(exc)))
                (OUT/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(f'GPU stream concurrency failed: {exc}', flush=True)
                break
        # Independent processes isolate library contexts after the thread failure.
        for workers in [2, 4]:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                     initializer=initialize_gpu) as pool:
                list(pool.map(ready, range(workers*4)))
                for repeat in range(2):
                    tick = time.perf_counter()
                    values = list(pool.map(gpu_process_work, range(16)))
                    record(f'gpu_processes{workers}', repeat, 16, time.perf_counter()-tick, values)
    report = ['# CPU/GPU pipeline and CUDA concurrency benchmark', '',
              'Three saved pure-X, 10 mW matrices (6.8, 28.2, 80 MHz); 20 initial populations and 1401 cumulative-photon times. Every task refactors its matrix; no new OBE scans.', '',
              'Two repetitions. Warm worker pools and CUDA libraries; startup excluded. Timings include task scheduling, CPU process communication, coefficient/input transfers, and photon result download. Common matrix construction and validation/file writing excluded.', '',
              'The pipeline performs real-eigenbasis decomposition and initial-state solves in eight single-threaded CPU processes. The main process projects coefficients on CUDA in batches while remaining CPU tasks continue. Stream tests use distinct CUDA streams and host threads; there is no per-task device-wide synchronization.', '',
              '| Configuration | Median seconds | Points/s |', '|---|---:|---:|']
    for name in dict.fromkeys(r['configuration'] for r in result['rows']):
        rows = [r for r in result['rows'] if r['configuration']==name]
        wall = float(np.median([r['wall_s'] for r in rows]))
        report.append(f'| {name} ({rows[0]["tasks"]} points) | {wall:.3f} | {rows[0]["tasks"]/wall:.2f} |')
    report.extend(['', f'All photon accuracy, imaginary-part, and monotonicity checks passed. Maximum photon difference: {max(r["max_photon_error"] for r in result["rows"]):.3g}.', '',
                   'This is a small repeated-input throughput benchmark, not a full scan or differentiability test. Details: [results.json](results.json).'])
    if result.get('failed_configurations'):
        report.extend(['', 'Concurrent threaded CUDA-stream eigendecomposition failed during warm-up with CUSOLVER_STATUS_INTERNAL_ERROR. Higher thread counts were not pursued; separate processes were tested to isolate library contexts.'])
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(f'Finished {OUT/"REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
