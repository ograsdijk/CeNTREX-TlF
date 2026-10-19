"""Summarize the matched historical RC GPU tests without running simulations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rc_gpu_matched_cpu_results"
accuracy = json.loads((OUT / "archived_cpu_accuracy.json").read_text())
progress = json.loads((OUT / "progress.json").read_text())
rows = [json.loads(p.read_text()) for p in OUT.glob("rc_*us_b*_s*.json")]
batch_rows = [r for r in rows if r.get("initial_batch")]
rows = [r for r in rows if not r.get("initial_batch")]
rows.sort(key=lambda r: (r["duration_us"], r["dt_ns"], r["batch"]))
lines = [
    "# Matched historical RC CPU/GPU comparison",
    "",
    f"Suite status: {progress['status']}.",
    "",
    "The archived nominal CPU cases are reproduced on GPU: 70 states, E=[0,0,200] V/cm, "
    "B=[0,0,0.01] G, pure X J=2 F1=5/2 F=2 mF=0 initial state, constant optical and "
    "microwave Rabi rates each equal to Gamma, coherent EOM 1.6 MHz beta=3.8, "
    "1 MHz microwave polarization switching with 250 ns offset, absolute entry time 125 ns. "
    "These are historical numerical fixtures, not production experimental fields.",
    "",
    "GPU: fixed-step FP64 RK4 and staged sparse diagonal/real-imaginary implementation. "
    "CPU: archived native Rust D+S+U / adaptive Tsit5 full-final-density timings. "
    "No CPU ODEs were rerun. Each GPU batch contains independently integrated replicas of "
    "the corresponding single CPU benchmark case; no trajectory results are reused. "
    "GPU RHS checked against the original CPU basis at all four switching configurations, "
    "including arbitrary coherences. No density clipping or renormalization.",
    "",
    "## Measured timings and accuracy",
    "",
    "| Duration (us) | Batch | dt (ns) | GPU median (s) | Repeats | CPU single trajectory (s) | "
    "Serial CPU throughput ratio | Ratio assuming ideal 12-thread CPU scaling | "
    "GPU worst complex-entry error | CPU archived error | Both within 1%? |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for r in rows:
    error = max(r["accuracy_vs_archived_tsit5"]["max_relative_above_cutoff"],
                r["accuracy_vs_archived_dopri5"]["max_relative_above_cutoff"])
    cpu_error = accuracy[str(r["duration_us"])]["max_relative_above_cutoff"]
    lines.append(f"| {r['duration_us']} | {r['batch']} | {r['dt_ns']:.4g} | "
                 f"{r['median_s']:.4g} | {r['repeats']} | {r['cpu']['median_s']:.4g} | "
                 f"{r['speedup_vs_serial_cpu']:.2f}x | {r['speedup_vs_ideal_12thread_cpu']:.2f}x | "
                 f"{100*error:.4g}% | {100*cpu_error:.4g}% | "
                 f"{'Yes' if max(error,cpu_error)<=.01 else 'No'} |")
lines += [
    "",
    "Errors compare full complex entries, including phase and coherences, where the archived "
    "reference magnitude is at least 1e-6. GPU is checked against both archived strict "
    "Tsit5 and Dopri5 references; the larger error is tabulated. CPU error is recomputed "
    "from the archived accepted output against its strict Tsit5 reference. Below-cutoff "
    "absolute errors and trace, Hermiticity and eigenvalue diagnostics are in each JSON.",
    "",
    "## Interpretation limits",
    "",
    "The 12-thread column is an arithmetic comparison assuming ideal linear scaling; it "
    "is not a measured 12-thread CPU benchmark. The GPU batch times are measured. "
    "Historical CPU timings exclude model setup; GPU timing excludes compilation, capture "
    "and final host transfer, with full density retained on device. Current GPU runs can "
    "have production CPU activity in parallel; the archived CPU benchmark had low background "
    "activity. The rebuilt fixture and historical native binary/source provenance are "
    "recorded, and direct archived-density agreement is checked rather than assumed.",
    "",
    "The archived 10-us CPU timing used rtol=1e-7, atol=1e-9 and fails the subsequently "
    "adopted 1% complex-entry target. Its timing is included for provenance, not an "
    "accuracy-matched speedup claim. The 200-us CPU accepted timing used rtol=1e-9, "
    "atol=1e-11; 1-us CPU used rtol=1e-7, atol=1e-9. The long GPU cases are single timing "
    "observations, while the listed short cases have three repetitions.",
    "",
    "The coarse 0.977-ns GPU pilot is preserved as a failed-accuracy result. "
    "It must not be used as a qualified speedup. These fixtures have one nominal parameter "
    "point; they do not measure production full-grid throughput.",
    "",
    "## Sources",
    "",
]
for duration in sorted({r["duration_us"] for r in rows}):
    r = next(r for r in rows if r["duration_us"] == duration)
    path = Path(r["cpu"]["source"])
    lines.append(f"- {duration} us CPU: [{path.name}]({path.as_posix()}); "
                 "references and accepted output arrays alongside that archived suite.")
lines += ["", "GPU raw timings, density outputs and per-case diagnostics are alongside this report."]
if batch_rows:
    lines += ["", "## Exact archived parallel initial-state batches", "",
              "These compare the same initial-state list, nominal parameters, duration and "
              "full-density output against measured archived eight-worker CPU timings. "
              "CPU timing is a single observation; GPU timing is a median of three.", "",
              "| Initial states | GPU dt (ns) | GPU seconds | Archived 8-worker CPU seconds | "
              "Measured speed ratio | GPU entry error | CPU entry error | Both within 1%? |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(batch_rows, key=lambda r:(r['batch'],r['dt_ns'])):
        ge = r['accuracy_vs_archived_reference']['max_relative_above_cutoff']
        ce = r['cpu']['accuracy']['max_relative_above_cutoff']
        lines.append(f"| {r['initial_batch']} | {r['dt_ns']:.4g} | {r['median_s']:.4g} | "
                     f"{r['cpu']['measured_8thread_batch_s']:.4g} | "
                     f"{r['speedup_vs_measured_8thread_cpu']:.2f}x | {100*ge:.4g}% | "
                     f"{100*ce:.4g}% | {'Yes' if max(ge,ce)<=.01 else 'No'} |")
    lines += ["", "The saved batch reference was validated by the original CPU suite against "
              "Tsit5 and Dopri5; one reference array is archived per initial-state batch. "
              "Individual JSONs link the CPU timing source and list exact initial indices."]
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT / "REPORT.md")
