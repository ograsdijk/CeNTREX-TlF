"""Checkpointed sequential benchmark scheduler; watchdog owns only its children."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SPB = ROOT.parent / "CeNTREX-SPB-testing"
METHODS = [
    "dense_two",
    "dense_one",
    "dense_split",
    "sparse_complex",
    "sparse_split",
    "fused_aos_major",
    "fused_aos_minor",
    "fused_soa_major",
    "fused_soa_minor",
    "staged_soa_major",
    "staged_soa_minor",
]


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(11):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 10:
                if path.name == "progress.json":
                    temp.unlink(missing_ok=True)
                    return
                raise
            time.sleep(0.05)


def table(count, pattern):
    if pattern == "grid":
        det = [-20.0 + 0.25 * ((i * 61) % 161) for i in range(161) for _ in range(3)]
    elif pattern == "full":
        det = [x for x in [-20.0, 0.0, 20.0] for _ in range(3)]
    else:
        det = [x for x in [-20.0, -1.0, 0.0, 20.0] for _ in range(3)]
    return [det[i % len(det)] for i in range(count)], [i % 3 + 1 for i in range(count)]


class Suite:
    def __init__(self, args):
        self.out = Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = datetime.fromisoformat(args.deadline).timestamp()
        self.start = time.time()
        self.rows = []
        self.stage = "initializing"
        self.child = None
        self.current = None
        self.failed = 0
        self.timed_out = 0
        self.status = "running"
        self.env = os.environ.copy()
        self.env.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONIOENCODING="utf-8",
            PYTHONUNBUFFERED="1",
            RAYON_NUM_THREADS="12",
            CUPY_CACHE_DIR=str(ROOT / "benchmarks/.cupy_cache"),
        )
        source_names = [
            "run_rc_gpu_sparse_suite.py",
            "rc_gpu_sparse_job.py",
            "rc_gpu_sparse_methods.py",
            "rc_gpu_sparse_cpu.py",
            "rc_gpu_sparse_kernels.cu",
        ]
        sources = {
            str(ROOT / "benchmarks" / name): hashlib.sha256(
                (ROOT / "benchmarks" / name).read_bytes()
            ).hexdigest()
            for name in source_names
        }
        pointer = json.loads(
            (SPB / ".cache/simulations/optical_bloch/native_rc_current.json").read_text()
        )
        context = Path(pointer["progress_path"]).parent / "prepared_context.pkl"
        atomic(
            self.out / "manifest.json",
            {
                "started_utc": datetime.now(UTC).isoformat(),
                "budget_origin_utc": "2026-10-11T23:55:49+00:00",
                "deadline_utc": args.deadline,
                "sources": sources,
                "production_pointer": pointer,
                "production_context": str(context),
                "context_sha256": hashlib.sha256(context.read_bytes()).hexdigest(),
                "gpu_python": sys.executable,
                "cpu_python": str(SPB / ".venv/Scripts/python.exe"),
                "methods": METHODS,
                "rhs_batch_sizes": [512, 1024, 2048, 4096],
                "precision": "float64/complex128",
                "production_untouched": True,
            },
        )

    def remaining(self):
        return self.deadline - time.time()

    def update(self):
        atomic(
            self.out / "progress.json",
            {
                "status": self.status,
                "stage": self.stage,
                "completed_jobs": len(self.rows),
                "failed_jobs": self.failed,
                "timed_out_jobs": self.timed_out,
                "elapsed_s": time.time() - self.start,
                "remaining_budget_s": max(0.0, self.remaining()),
                "current_job": self.current,
                "runner_pid": os.getpid(),
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )

    def kill_child(self):
        if self.child is not None and self.child.poll() is None:
            # Root PID belongs to this Popen; production is never its descendant.
            if os.name == "nt":
                subprocess.run(
                    ["taskkill.exe", "/PID", str(self.child.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                self.child.kill()
            with suppress(subprocess.TimeoutExpired):
                self.child.wait(timeout=10)

    def job(self, kind, timeout=300, **settings):
        if self.remaining() < 20:
            return None
        settings = {"kind": kind, **settings}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        stem = f"{kind}_{settings.get('method', 'rust')}_{key}"
        output = self.out / (stem + ".json")
        if output.exists():
            return json.loads(output.read_text())
        settings["output"] = str(output)
        path = self.out / (stem + ".job.json")
        atomic(path, settings)
        is_cpu = kind == "cpu"
        python = SPB / ".venv/Scripts/python.exe" if is_cpu else Path(sys.executable)
        script = (
            ROOT / "benchmarks" / ("rc_gpu_sparse_cpu.py" if is_cpu else "rc_gpu_sparse_job.py")
        )
        self.current = {
            "kind": kind,
            "method": settings.get("method", "Rust D+S+U"),
            "batch": settings.get("batch", len(settings.get("detunings_MHz", []))),
            "start": settings.get("start"),
            "end": settings.get("end"),
            "substeps": settings.get("substeps"),
            "output": str(output),
        }
        self.update()
        tick = time.monotonic()
        limit = min(timeout, self.remaining() - 5)
        with (
            (self.out / (stem + ".stdout.log")).open("w", encoding="utf-8") as stdout,
            (self.out / (stem + ".stderr.log")).open("w", encoding="utf-8") as stderr,
        ):
            self.child = subprocess.Popen(
                [str(python), "-u", str(script), str(path)],
                cwd=ROOT,
                env=self.env,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            while self.child.poll() is None:
                if time.monotonic() - tick > limit or self.remaining() <= 5:
                    self.kill_child()
                    self.timed_out += 1
                    break
                time.sleep(0.2)
                if int(time.monotonic() - tick) % 3 == 0:
                    self.update()
        elapsed = time.monotonic() - tick
        code = self.child.returncode
        self.child = None
        if code == 0 and output.exists():
            result = json.loads(output.read_text())
            result["watchdog_job_wall_s"] = elapsed
            result["output"] = str(output)
            atomic(output, result)
        else:
            self.failed += 1
            result = {
                "job": settings,
                "status": "failed",
                "exit_code": code,
                "watchdog_job_wall_s": elapsed,
                "stderr": str(self.out / (stem + ".stderr.log")),
                "output": str(output),
            }
            atomic(output, result)
        self.rows.append(result)
        atomic(self.out / "results.json", self.rows)
        print(self.stage, stem, result.get("status", "complete"), f"{elapsed:.1f}s", flush=True)
        self.current = None
        self.update()
        self.report()
        return result if code == 0 else None

    def reference(
        self, duration, pattern="grid", count=483, start=146e-6, velocity=150.0, full=False
    ):
        det, js = table(count, pattern)
        if full:
            det, js = [0.0], [2]
        result = self.job(
            "cpu",
            timeout=1200,
            detunings_MHz=det,
            initial_J=js,
            threads=1,
            repeats=1,
            start=start,
            end=start + duration,
            velocity=velocity,
            full=full,
            reltol=1e-10,
            abstol=1e-12,
            reference=True,
            pattern=pattern,
        )
        return str(Path(result["output"]).with_suffix(".npz")) if result else None

    def report(self):
        lines = [
            "# RC GPU sparse suite",
            "",
            f"Status: {self.status}; stage: {self.stage}.",
            f"Jobs recorded: {len(self.rows)}; failures: {self.failed}; timeouts: {self.timed_out}.",
            "",
            "Production RC continues independently. Float64/complex128 throughout.",
            "CPU references/baselines use the production Python 3.13 Rust binary.",
            "Any 12-thread CPU timings below are concurrent with production and are contended;",
            "they are not an uncontended speedup measurement. Library CuPy methods use eager",
            "launching because this version rejects cuBLAS/cuSPARSE graph capture; custom methods use graphs.",
            "",
            "## Integrated GPU measurements",
            "",
            "| Method | Batch | Interval (us) | dt (ns) | Median evolution (s) | Traj/s | Max relative entry error above 1e-6 | Qualified |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in self.rows:
            j = r["job"]
            if r.get("evolution_s"):
                seconds = median(r["evolution_s"])
                accuracy = r["accuracy"]
                lines.append(
                    f"| {j['method']} | {j['batch']} | {(j['end'] - j['start']) * 1e6:.3g} | {r['dt_s'] * 1e9:.3g} | {seconds:.4g} | {j['batch'] / seconds:.4g} | {100 * accuracy['max_relative_above_cutoff']:.4g}% | {accuracy['passes_1percent_above_cutoff']} |"
                )
        lines += [
            "",
            "## CPU measurements",
            "",
            "| Threads | Count | Interval (us) | atol | Wall times (s) |",
            "|---:|---:|---:|---:|---|",
        ]
        for r in self.rows:
            j = r["job"]
            if j["kind"] == "cpu" and "timings" in r:
                lines.append(
                    f"| {j['threads']} | {len(j['initial_J'])} | {(j['end'] - j['start']) * 1e6:.4g} | {j['abstol']} | {[round(t['wall_s'], 4) for t in r['timings']]} |"
                )
        lines += ["", "## Full-transit checks", ""]
        for r in self.rows:
            j = r["job"]
            if j["kind"] == "full" and r.get("status") == "complete":
                lines.append(
                    f"- {j['method']}, v={j['velocity']} m/s, dt={r['dt_s'] * 1e9:.4g} ns: {r['full_wall_s_including_sampling_and_post_decay']:.3g} s including 201 population samples and both density endpoints."
                )
                if "strict_reference_subset_accuracy" in r:
                    a = r["strict_reference_subset_accuracy"]
                    lines.append(
                        f"  Strict Rust comparison covers J=2, zero detuning, both endpoints: max relative complex-entry error {100 * a['max_relative_above_cutoff']:.4g}% above cutoff."
                    )
                if "timestep_difference" in r:
                    a = r["timestep_difference"]
                    lines.append(
                        f"  Difference from coarser GPU timestep across all nine cases/endpoints: {100 * a['max_relative_above_cutoff']:.4g}% above cutoff; this is convergence evidence, not a stricter Rust reference for all cases."
                    )
        lines += [
            "",
            "All numerical arrays, per-trajectory diagnostics, failed-method logs, setup and memory measurements are retained alongside results.json.",
            "A short-interval pass does not qualify a production backend or a full scan. No production backend change is made.",
            "",
        ]
        (self.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    def execute(self):
        try:
            self.stage = "RHS methods and large batches"
            screens = []
            for method in METHODS:
                for batch in [512, 1024, 2048, 4096]:
                    r = self.job("rhs", timeout=150, method=method, batch=batch)
                    if r:
                        screens.append(r)
                    if self.remaining() < 3600:
                        break
            scores = {}
            for r in screens:
                name = r["job"]["method"]
                score = median([t["wall_s_per_rhs"] for t in r["timings"]]) / r["job"]["batch"]
                scores[name] = min(scores.get(name, float("inf")), score)
            ranked = sorted(scores, key=scores.get)
            finalists = ranked[:3]
            if not finalists:
                raise RuntimeError("No RHS candidate passed validation")
            atomic(
                self.out / "rhs_ranking.json",
                {"scores_seconds_per_trajectory_rhs": scores, "finalists": finalists},
            )
            self.stage = "strict Rust 1 us reference"
            ref1 = self.reference(1e-6)
            if not ref1:
                raise RuntimeError("Strict 1 us reference unavailable")
            self.stage = "1 us GPU integrator and accuracy screen"
            integrated = []
            for method in finalists:
                for substeps in [128, 256, 512]:
                    for batch in [128, 512, 1024, 2048, 4096]:
                        if self.remaining() < 3000:
                            break
                        r = self.job(
                            "ode",
                            timeout=300,
                            method=method,
                            batch=batch,
                            substeps=substeps,
                            start=146e-6,
                            end=147e-6,
                            reference=ref1,
                            pattern="grid",
                            repeats=3,
                        )
                        if r and r["accuracy"]["passes_1percent_above_cutoff"]:
                            integrated.append(r)
            # Explicit short 12-thread CPU comparison: same rows and output, contended.
            self.stage = "brief production-build 12-thread CPU samples"
            for count in [128, 512, 1024]:
                det, js = table(count, "grid")
                self.job(
                    "cpu",
                    timeout=120,
                    detunings_MHz=det,
                    initial_J=js,
                    threads=12,
                    repeats=3,
                    start=146e-6,
                    end=147e-6,
                    reltol=1e-9,
                    abstol=1e-10,
                    reference=False,
                )
            best = {}
            for r in integrated:
                name = r["job"]["method"]
                score = median(r["evolution_s"]) / r["job"]["batch"]
                if name not in best or score < best[name][0]:
                    best[name] = (score, r["job"]["substeps"])
            best_methods = sorted(best, key=lambda name: best[name][0])[:2]
            if not best_methods:
                raise RuntimeError("No integrated method passed the density-entry gate")
            self.stage = "strict Rust joint-period reference"
            ref5 = self.reference(5e-6) if self.remaining() > 3000 else None
            passed5 = []
            if ref5:
                self.stage = "5 us joint switching/EOM period"
                for method in best_methods:
                    for substeps in sorted({best[method][1], 512}):
                        for batch in [512, 1024, 2048, 4096]:
                            if self.remaining() < 2400:
                                break
                            r = self.job(
                                "ode",
                                timeout=600,
                                method=method,
                                batch=batch,
                                substeps=substeps,
                                start=146e-6,
                                end=151e-6,
                                reference=ref5,
                                pattern="grid",
                                repeats=3,
                            )
                            if r and r["accuracy"]["passes_1percent_above_cutoff"]:
                                passed5.append(r)
            if passed5:
                passed5.sort(key=lambda r: median(r["evolution_s"]) / r["job"]["batch"])
                best_methods = list(dict.fromkeys(r["job"]["method"] for r in passed5))[:2]
                selected_steps = {
                    name: min(r["job"]["substeps"] for r in passed5 if r["job"]["method"] == name)
                    for name in best_methods
                }
            else:
                selected_steps = {name: max(512, best[name][1]) for name in best_methods}
            for duration in [10e-6, 50e-6]:
                if self.remaining() < 1800:
                    break
                self.stage = f"strict Rust {duration * 1e6:g} us finalist reference"
                reference = self.reference(duration, pattern="legacy", count=12)
                if reference:
                    self.stage = f"{duration * 1e6:g} us GPU finalists"
                    for method in best_methods:
                        for batch in [128, 512, 1024]:
                            if self.remaining() < 1500:
                                break
                            self.job(
                                "ode",
                                timeout=600,
                                method=method,
                                batch=batch,
                                substeps=selected_steps[method],
                                start=146e-6,
                                end=146e-6 + duration,
                                reference=reference,
                                pattern="legacy",
                                repeats=3,
                            )
            self.stage = "selected full-transit endpoint and population checks"
            for velocity in [150.0, 50.0, 250.0]:
                if self.remaining() < 600:
                    break
                duration = 0.044 / velocity
                reference = self.reference(
                    duration, pattern="full", count=1, start=0.0, velocity=velocity, full=True
                )
                for method in best_methods:
                    if self.remaining() < 300:
                        break
                    r = self.job(
                        "full",
                        timeout=900,
                        method=method,
                        batch=9,
                        substeps=selected_steps[method],
                        start=0.0,
                        end=duration,
                        velocity=velocity,
                        pattern="full",
                        reference=reference,
                    )
                    if r and self.remaining() > 450:
                        self.job(
                            "full",
                            timeout=900,
                            method=method,
                            batch=9,
                            substeps=2 * selected_steps[method],
                            start=0.0,
                            end=duration,
                            velocity=velocity,
                            pattern="full",
                            reference=reference,
                            convergence_reference=str(Path(r["output"]).with_suffix(".npz")),
                        )
            self.status = "complete" if self.remaining() > 5 else "budget_complete"
            self.stage = "finished"
        except BaseException:
            import traceback

            (self.out / "suite_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            self.status = "failed"
            raise
        finally:
            self.kill_child()
            self.current = None
            self.update()
            self.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    suite = Suite(args)
    suite.execute()
