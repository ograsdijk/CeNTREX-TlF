"""Run the archived nominal RC CPU workloads on GPU; never run CPU ODEs.

Each batch repeats the CPU benchmark's one initial state and nominal parameters
in independently integrated arrays. There is no propagation/result reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "rc_gpu_matched_cpu_results"
OLD = HERE / "rc_propagation_results"
SOURCES = {
    1: OLD / "cross_system_screen_v3/nominal/rc/screen.json",
    10: OLD / "cross_system_long_finalists_v1/suite10us_v1/rc/10us/result.json",
    200: OLD / "cross_system_long_finalists_v1/suite_long_v1/rc/200us/result.json",
}


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def cpu_record(duration):
    source = SOURCES[duration]
    data = json.loads(source.read_text())
    rows = data["rows"] if duration == 1 else data["raw_rows"]
    rows = [r for r in rows if r.get("product") == "full" and
            r.get("kernel", "").endswith("diag_soa_single_unpack")]
    times = [r["accepted_wall_seconds"] for r in rows]
    return dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                median_s=float(np.median(times)), times_s=times,
                tolerance=rows[0]["accepted_tolerance"],
                native_sha256=data["native_extension_sha256"])


def worker(duration, batch, substeps, repeats, initial_batch=None):
    import cupy as cp
    import rc_gpu_sparse_methods as methods
    payload = pickle.loads((OUT / "fixture.pkl").read_bytes())
    plan, n = payload["plan"], payload["n"]
    coeff = plan["coefficients"]
    expected = [[3], [6], [14, 5], [15, 5], [16, 9], [17, 9], [4, 13], [10]]
    actual = [[i["index"] for i in c["expression"]["instructions"] if i["op"] == 2]
              for c in coeff]
    assert actual == expected, actual
    base = {k: complex(v["re"], v["im"]).real for k, v in
            zip(payload["graph"]["slot_names"], payload["graph"]["base_values"])}
    assert all(base[k] == 0 for k in ("δ0", "δ1", "δ2"))
    basis = []
    for c in coeff:
        b = np.zeros((n, n), complex)
        for term in c["basis_terms"]:
            b[term["i"], term["j"]] += complex(term["re"], term["im"])
        basis.append(b)
    diagonal = np.asarray(plan["static_matrix"])
    assert np.count_nonzero(diagonal - np.diag(np.diag(diagonal))) == 0
    # Existing GPU kernel orders MW1 as (1-PA, PA). Archived CPU plan orders
    # it as (PA, 1-PA), so exchange the two corresponding matrices exactly.
    full = np.asarray([basis[i] + np.triu(basis[i], 1).conj().T for i in [3, 2, 4, 5, 6]])
    support = np.any(full != 0, axis=0)
    np.fill_diagonal(support, False)
    rows, columns = np.nonzero(support)
    tp, tc, tr = [0], [], []
    for entries in payload["transfers"]:
        for entry in entries:
            tc.append(entry["source"])
            tr.append(entry["rate"])
        tp.append(len(tc))
    f = dict(basis=full, diagonal=np.diag(diagonal).real.copy(),
             mask=np.diag(basis[0]).real.copy(), rows=rows.astype(np.int32),
             columns=columns.astype(np.int32),
             ptr=np.r_[0, np.cumsum(support.sum(axis=1))].astype(np.int32),
             tp=np.array(tp, np.int32), tc=np.array(tc, np.int32), tr=np.array(tr),
             rates=np.asarray(payload["rates"]),
             base=dict(mw_peak1=base["Ω1"], mw_peak2=base["Ω2"], PUVZ0=base["PX0"],
                       Ωt0=base["Ωt0"], β0=base["β0"], ωphase0=base["ωphase0"]))
    methods.fixture = lambda context: f
    if initial_batch:
        batch_source = OLD / "cross_system_batch_amortization_v2/stage4_batch_v2_retry1/rc/result.json"
        batch_data = json.loads(batch_source.read_text())
        batch_rows = [r for r in batch_data["rows"] if r.get("product") == "full" and
                      r.get("batch") == initial_batch and r.get("workers")]
        row8 = next(r for r in batch_rows if r["workers"] == 8)
        indices = row8["initial_indices"]
        assert batch == len(indices) and duration == 1
        initials = np.zeros((batch, n, n), complex)
        initials[np.arange(batch), indices, indices] = 1.
        methods.points = lambda count, pattern: (np.zeros(count), np.arange(count))
    else:
        initials = payload["rho0"][None]
        methods.points = lambda count, pattern: (np.zeros(count), np.zeros(count, int))
    # An effectively infinite Gaussian sigma evaluates to exactly 1.0 in FP64,
    # preserving the archived constant-amplitude CPU benchmark.
    context = dict(prepared=SimpleNamespace(layout=SimpleNamespace(n=n)),
                   physics=dict(uv_length_m=1.), sigma_I=1e99,
                   rho0=initials)
    start = payload["configuration"]["entry_phase_s"]
    g = methods.GPU(context, "staged_soa_minor", batch, start, start + duration * 1e-6,
                    substeps, velocity=1.)
    # Check the GPU RHS against the original (unswapped) CPU basis and switch
    # convention, at all four configurations and the nonzero entry clock.
    checks = []
    rng = np.random.default_rng(293)
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = x @ x.conj().T
    rho /= np.trace(rho)
    host = np.repeat(rho[None], batch, axis=0)
    y = g.upload(host)
    for t in [125e-9, 375e-9, 625e-9, 875e-9, 1.125e-6]:
        a = int(np.sin(2*np.pi*1e6*t) >= 0)
        b = int(np.sin(2*np.pi*1e6*t + np.pi/2) >= 0)
        c = [a*base["Ω1"], (1-a)*base["Ω1"], b*base["Ω2"], (1-b)*base["Ω2"],
             base["PX0"]*base["Ωt0"]*np.exp(1j*base["β0"]*np.sin(base["ωphase0"]*t))]
        upper = sum(v*np.triu(basis[k]) for v, k in zip(c, [2,3,4,5,6]))
        h = diagonal + upper + np.triu(upper,1).conj().T
        ref = -1j*(h@rho-rho@h) - .5*(f["rates"][:,None]+f["rates"][None,:])*rho
        for i, entries in enumerate(payload["transfers"]):
            for entry in entries:
                ref[i,i] += entry["rate"]*rho[entry["source"],entry["source"]]
        g.clock.set(np.array([t, 0., g.end, g.dt]))
        g.rhs(y, g.ks[0], 0.)
        got = g.download(g.ks[0])[0]
        err = float(np.linalg.norm(got-ref)/np.linalg.norm(ref))
        assert err < 1e-12, err
        checks.append(dict(t=t, relative_frobenius=err))
    del host, y
    setup = time.perf_counter()
    g.capture()
    capture_s = time.perf_counter() - setup
    actual, times, final_time = g.evolve(repeats)
    assert abs(final_time - g.end) < 1e-14
    if initial_batch:
        z = np.load(batch_source.parent / "references.npz")
        reference = z[initial_batch + "_rho_reference"]
    else:
        z = np.load(SOURCES[duration].parent / "reference.npz")
        reference = z["tsit5_full"]
    ref = np.broadcast_to(reference, actual.shape)
    accuracy = methods.diagnostics(actual, ref)
    independent = (accuracy if initial_batch else
                   methods.diagnostics(actual, np.broadcast_to(z["dopri5_full"], actual.shape)))
    if initial_batch:
        row1 = next(r for r in batch_rows if r["workers"] == 1)
        cpu = dict(source=str(batch_source), median_s=row1["accepted_wall_seconds"]/batch,
                   times_s=[row1["accepted_wall_seconds"]/batch],
                   tolerance=row8["accepted_tolerance"],
                   native_sha256=batch_data["native_extension_sha256"],
                   measured_8thread_batch_s=row8["accepted_wall_seconds"],
                   reference_solver=batch_data["reference_solver"],
                   sha256=hashlib.sha256(batch_source.read_bytes()).hexdigest())
        cpu_file = OLD / "cross_system_batch_amortization_v2" / row8["output_artifact"]
        cz = np.load(cpu_file)
        cpu_output = cz["accepted_output"] if "accepted_output" in cz else cz[cz.files[0]]
        cpu["accuracy"] = methods.diagnostics(cpu_output, ref)
    else:
        cpu = cpu_record(duration)
    median = float(np.median(times))
    report = dict(status="complete", duration_us=duration, batch=batch, repeats=repeats,
                  dt_ns=g.dt*1e9, method=g.method, precision="float64/complex128",
                  configuration=payload["configuration"], metadata=payload["metadata"],
                  initial_state=payload["metadata"]["initial_label"],
                  batch_semantics="Independent replicas of the archived nominal trajectory; no result reuse",
                  cpu=cpu, evolution_s=times, median_s=median, capture_s=capture_s,
                  seconds_per_trajectory=median/batch,
                  speedup_vs_serial_cpu=cpu["median_s"]*batch/median,
                  speedup_vs_ideal_12thread_cpu=cpu["median_s"]*batch/median/12,
                  accuracy_vs_archived_tsit5=accuracy,
                  accuracy_vs_archived_dopri5=independent, rhs_checks=checks,
                  device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                  final_time=final_time, start=start, end=g.end,
                  fixture_sha256=hashlib.sha256((OUT/"fixture.pkl").read_bytes()).hexdigest())
    if initial_batch:
        report.update(initial_batch=initial_batch, initial_indices=indices,
                      initial_state=initial_batch,
                      batch_semantics="Exact archived initial-state batch, one trajectory per listed state",
                      speedup_vs_measured_8thread_cpu=cpu["measured_8thread_batch_s"]/median)
        report["accuracy_vs_archived_reference"] = report.pop("accuracy_vs_archived_tsit5")
        report.pop("accuracy_vs_archived_dopri5")
    stem = f"rc_{duration}us_b{batch}_s{substeps}" + ("_"+initial_batch if initial_batch else "")
    np.savez_compressed(OUT / (stem + ".npz"), rho=actual if initial_batch else actual[0], reference=reference)
    write(OUT / (stem + ".json"), report)
    print(json.dumps({k:report[k] for k in ["duration_us", "batch", "median_s", "speedup_vs_serial_cpu", "speedup_vs_ideal_12thread_cpu"]}), flush=True)
    print("Max entry error percent:", accuracy["max_relative_above_cutoff"]*100, flush=True)


def suite():
    OUT.mkdir(exist_ok=True)
    deadline = time.monotonic() + 25*60
    jobs = [(1,128,512,3), (1,1024,512,3), (10,128,512,3),
            (10,512,512,3), (200,128,512,1), (200,128,1024,1)]
    completed = []
    for duration, batch, steps, repeats in jobs:
        name = f"rc_{duration}us_b{batch}_s{steps}"
        write(OUT/"progress.json", dict(status="running", current=name, completed=completed, total=len(jobs)))
        print(f"[{len(completed)+1}/{len(jobs)}] {name}", flush=True)
        with (OUT/(name+".stdout.log")).open("w") as stdout, (OUT/(name+".stderr.log")).open("w") as stderr:
            child = subprocess.Popen([sys.executable, __file__, "--worker", "--duration", str(duration),
                                      "--batch", str(batch), "--substeps", str(steps), "--repeats", str(repeats)],
                                     stdout=stdout, stderr=stderr)
            try:
                code = child.wait(timeout=max(1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                write(OUT/"progress.json",dict(status="timed_out", current=name, completed=completed))
                return
        if code:
            write(OUT/"progress.json",dict(status="failed", current=name, completed=completed))
            raise RuntimeError(f"{name} failed; inspect its stderr")
        completed.append(name)
        print((OUT/(name+".stdout.log")).read_text(), flush=True)
    write(OUT/"progress.json",dict(status="complete", completed=completed, total=len(jobs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--duration", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--substeps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--initial-batch")
    args = parser.parse_args()
    os.environ.setdefault("CUPY_CACHE_DIR", str(HERE/".cupy_cache"))
    if args.worker:
        if args.duration == 200 and args.substeps == 1024 and not args.initial_batch:
            previous = OUT / "rc_200us_b128_s512.json"
            if previous.exists():
                prior = json.loads(previous.read_text())
                if (prior["accuracy_vs_archived_tsit5"]["passes_1percent_above_cutoff"] and
                        prior["accuracy_vs_archived_dopri5"]["passes_1percent_above_cutoff"]):
                    write(OUT/"skipped_convergence.json", dict(status="skipped",
                          reason="200-us timestep already passes both independent archived strict references"))
                    print("Skipping finer 200-us repetition: both archived-reference accuracy checks passed.")
                    sys.exit(0)
        worker(args.duration, args.batch, args.substeps, args.repeats, args.initial_batch)
    else:
        suite()
