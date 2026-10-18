"""Summarize the remaining CUDA performance experiments."""
import json
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent/"cuda_remaining_results"
lines=["# Further CUDA experiments", "", "GPU: NVIDIA RTX 5070 Ti. Benchmark-only prototypes; production solver unchanged.", "",
       "Tests cover precomputed packed-density gather indices, a shared-column tiled direct RHS, "
       "FP32 product/RHS arithmetic with FP64 state and RK updates, trajectory-local persistent "
       "Tsit5, and fixed-shape buffer/graph reuse across changed detunings.", "",
       "The screen uses 512 trajectories and four randomized interleaved rounds; confirmation "
       "uses 1,024 and six rounds. RC spans 146-147 us near the field center with polarization "
       "switching, dt=4.4642857 ns. R(2) includes an EOM, spans 0.125-1.125 us, dt=2 ns. "
       "Archived distinct initial-state/detuning cases are repeated to fill batches. No CPU ODEs rerun.", "",
       "Evolution timings exclude setup/capture/precomputation and include final internal-density "
       "download. Every candidate runs through the public solver for archived-reference checks, "
       "and timed graph replays are checked again. State/stage storage remains FP64 even for "
       "FP32 arithmetic. The hybrid tiles one density column in shared memory, evaluates both "
       "sides of the commutator, applies decay and writes packed output directly.", "",
       "## Large-batch confirmation", "", "Positive percentages indicate reduced runtime vs the same RHS baseline. Parentheses "
       "give median seconds. Gather variants use their corresponding tiled/direct baseline; "
       "hybrid and FP32 variants use direct. Neither comparison changes dt or physics.", "",
       "| System | Method | Median seconds | Paired runtime reduction | Passed density checks |",
       "|---|---|---:|---:|---|"]
path=OUT/"confirm1024.json"
if path.exists():
    for case in json.loads(path.read_text())["cases"]:
        rows=case["variants"]
        for name,row in rows.items():
            times=np.asarray(row["timings"])
            base_name="base_tiled" if name.endswith("tiled") else "base_direct"
            base=np.asarray(rows[base_name]["timings"])
            reduction=float(np.median(100*(1-times/base)))
            lines.append(f"| {case['system']} | {name} | {np.median(times):.6f} | {reduction:+.2f}% | {row['passed']} |")
signpath=OUT/"gather_sign1024.json"
if signpath.exists():
    lines.extend(["", "### Precomputed indices plus conjugation signs", "",
                  "A separate four-round 1,024-trajectory check precomputes both packed indices "
                  "and conjugation signs. Comparisons are paired against the corresponding RHS.", "",
                  "| System | Method | Median seconds | Paired runtime reduction | Passed |",
                  "|---|---|---:|---:|---|"])
    for case in json.loads(signpath.read_text())["cases"]:
        for name in ["gather_sign_tiled","gather_sign_direct"]:
            row=case["variants"][name]
            times=np.asarray(row["timings"])
            base=np.asarray(case["variants"]["base_tiled" if name.endswith("tiled") else "base_direct"]["timings"])
            reduction=float(np.median(100*(1-times/base)))
            lines.append(f"| {case['system']} | {name} | {np.median(times):.6f} | {reduction:+.2f}% | {row['passed']} |")
lines.extend(["", "Full raw rounds and diagnostics: [screen.json](screen.json), [confirm1024.json](confirm1024.json), "
              "[gather_sign1024.json](gather_sign1024.json). "
              "The original 512-screen fp32_product_direct label used the same full-RHS lowering as "
              "fp32_rhs_direct; do not treat that duplicate as an independent product-only result. "
              "The product-only/direct lowering was separated before 1,024-trajectory confirmation. "
              "Generated CUDA sources are retained beside these JSON files.", "", "## Persistent integration", "",
              "One block owns one trajectory, keeping the base/current packed density and Hamiltonian "
              "in shared memory, using global FP64 RK stage arrays, block barriers between stages, "
              "FSAL across steps/chunks, and the original clipped time grid. Launches advance only "
              "2 or 16 steps, avoiding an unbounded long-lived kernel. Full density output only; "
              "photon integration was not added to this rejected prototype.", "",
              "| System | Batch | Product arithmetic | Baseline direct (s) | Best persistent (s) | Shared memory/block |",
              "|---|---:|---|---:|---:|---:|"])
for path in sorted(OUT.glob("persistent_*_b*.json")):
    for case in json.loads(path.read_text()):
        passing=[r for r in case["variants"].values() if r["passed"] and r["timings"]]
        best=min(float(np.median(r["timings"])) for r in passing)
        lines.append(f"| {case['system']} | {case['batch']} | {case.get('arithmetic','fp64')} | {np.median(case['baseline_times']):.6f} | {best:.6f} | {case['shared_bytes']} bytes |")
lines.extend(["", "Persistent comparisons use three warm repetitions per launch configuration. "
              "Kernel attributes and full complex errors are in the per-system JSON files. "
              "Large shared-memory allocations constrain occupancy; attributing the slowdown "
              "to a specific hardware bottleneck would require profiler counters.", "",
              "## Buffer/graph reuse", "",
              "Reuses buffers, captured graphs, static model and unchanged initial states for "
              "fixed-shape/support/varying-pattern/time-grid batches. Every call replaces scalar "
              "parameters, resets state/clock/error and regenerates common waveform tables. "
              "Detuning offsets +0.125, -0.25 and +0.5 MHz are compared with fresh public calls. "
              "Fresh-packed calls use prepacked initial states to separate packing from reuse. "
              "Three interleaved rounds per detuning setting; values agree to 1e-12 absolute "
              "and 1e-11 relative. Timings include parameter uploads and final public-packed "
              "density output. These are short-call wall times, not full-transit speedups.", "",
              "| System | Batch | Duration (us) | Fresh matrix input (s) | Fresh packed input (s) | Reused (s) |",
              "|---|---:|---:|---:|---:|---:|"])
for path in sorted(OUT.glob("session_reuse_b*.json")):
    for case in json.loads(path.read_text()):
        def median(key, rows=case["rows"]):
            values=[v for row in rows for v in row["timings"].get(key,[])]
            return f"{np.median(values):.6f}" if values else "not measured"
        duration=(case['span'][1]-case['span'][0])*1e6
        lines.append(f"| {case['system']} | {case['batch']} | {duration:g} | {median('fresh')} | {median('fresh_packed')} | {median('reused')} |")
longpath=OUT/"session_reuse_b1024_10us.json"
if longpath.exists():
    lines.extend(["", "### RC and R(2), 10 us reuse comparison", "",
                  "1,024 trajectories, identical prepacked inputs, current default tiled RHS, "
                  "fixed Tsit5. The original short-fixture start times and dt are retained, "
                  "extending the endpoint to 10 us after entry. Three changed-detuning batches "
                  "(+0.125/-0.25/+0.5 MHz offsets), two randomized interleaved rounds each: six "
                  "fresh/reused pairs per system. Full packed densities are compared at each "
                  "offset, atol=1e-12 and rtol=1e-11. This isolates setup reuse; it is not a "
                  "new timestep-accuracy or full-transit benchmark.", "",
                  "| System | Fresh packed median (s) | Reused median (s) | Paired median runtime reduction | Descriptive 95% interval | Max packed difference |",
                  "|---|---:|---:|---:|---:|---:|"])
    rng=np.random.default_rng(728)
    for case in json.loads(longpath.read_text()):
        fresh=np.array([v for row in case["rows"] for v in row["timings"]["fresh_packed"]])
        reused=np.array([v for row in case["rows"] for v in row["timings"]["reused"]])
        reductions=100*(1-reused/fresh)
        samples=rng.integers(0,len(fresh),size=(10000,len(fresh)))
        low,high=np.percentile(np.median(reductions[samples],axis=1),[2.5,97.5])
        error=max(row["max_packed_difference"] for row in case["rows"])
        lines.append(f"| {case['system']} | {np.median(fresh):.6f} | {np.median(reused):.6f} | {np.median(reductions):+.2f}% | [{low:+.2f}, {high:+.2f}]% | {error:.3g} |")
    lines.extend(["", "Positive percentages mean less runtime. Intervals resample the six paired "
                  "rounds and describe this small sample on this GPU. Raw data: "
                  "[session_reuse_b1024_10us.json](session_reuse_b1024_10us.json). Reproduce with "
                  "`benchmarks/check_cuda_session_reuse.py --batch 1024 --duration-us 10 "
                  "--repeats 2 --methods fresh_packed reused` in the benchmark GPU environment."])
lines.extend(["", "The RC/R(2) matrix-input timings above predate vectorized packing and "
              "CUDA grid prepacking in the library. That CPU overhead is now addressed by "
              "`layout.pack_batch` and by packing distinct CUDA-grid initial states before "
              "parameter expansion. Explicit packed snapshots remain reusable across calls."])
r0path=OUT/"r0_reuse.json"
if r0path.exists():
    lines.extend(["", "## R(0) reuse after the packing update", "",
                  "18-level R(0) F'=2 fixture, E=200 V/cm, B=0.01 G, optical Rabi rate Gamma. "
                  "Uniform J=0 population and a coherent J=0 initial state; three detunings "
                  "(-20/0/+20 MHz), repeated to fill 1,024 trajectories. EOM off (beta=0) or "
                  "beta=3.8 at 1.6 MHz. dt=1 ns; 1 us and 10 us intervals beginning at 125 ns. "
                  "Large rotational offsets of uncoupled compact spectators are excluded from "
                  "the initial step-size estimate. Six randomized interleaved rounds per case, "
                  "with changed detunings each round. This checks reuse equality, not independent "
                  "R(0) step convergence. All reused packed entries match fresh solves exactly.", "",
                  "| EOM beta | Duration (us) | Fresh packed (s) | Reused (s) | Per-round runtime reduction range |",
                  "|---:|---:|---:|---:|---:|"])
    for row in json.loads(r0path.read_text()):
        fresh=np.asarray(row["timings"]["fresh_packed"])
        reused=np.asarray(row["timings"]["reused"])
        reductions=100*(1-reused/fresh)
        lines.append(f"| {row['beta']:g} | {(row['span'][1]-row['span'][0])*1e6:g} | {np.median(fresh):.6f} | {np.median(reused):.6f} | {min(reductions):+.2f}% to {max(reductions):+.2f}% |")
    lines.extend(["", "Reuse saves roughly 6-7% in the short R(0) calls; longer calls are "
                  "effectively tied and include slower rounds. Reuse is not uniformly faster, "
                  "including within a single system. Raw samples: [r0_reuse.json](r0_reuse.json). "
                  "Reproduce with `benchmarks/check_r0_cuda_reuse.py` in the benchmark GPU environment."])
packingpath=OUT/"batch_packing_integration.json"
if packingpath.exists():
    lines.extend(["", "## Integrated packing", "",
                  "`PackedHermitianLayout.pack_batch` packs matrix batches with bounded NumPy "
                  "indexing. Batch solvers, dense preparation and matrix-snapshot conversion "
                  "use it automatically; CUDA grids validate/pack the distinct initial matrix "
                  "batch before replicating packed states over parameters. Repeated external "
                  "calls can reuse an explicitly packed owned snapshot. No identity-based "
                  "cache of mutable input arrays or graph-reuse API was introduced.", "",
                  "1,024 random complex Hermitian matrices at each representative model size; "
                  "one scalar reference timing, three vectorized timings, exact equality of "
                  "every packed coordinate. Timings below measure packing only, not ODE evolution.", "",
                  "| Size | States | Scalar pack (s) | Vectorized median (s) |",
                  "|---|---:|---:|---:|"])
    for row in json.loads(packingpath.read_text()):
        lines.append(f"| {row['model']} | {row['n']} | {row['scalar_seconds']:.6f} | {row['median_vectorized_seconds']:.6f} |")
    lines.extend(["", "Raw data: [batch_packing_integration.json](batch_packing_integration.json); "
                  "reproduce with `benchmarks/validate_batch_packing.py`. Tests verify scalar "
                  "equivalence, complex signs, strided/empty inputs, chunk boundaries and owned "
                  "snapshots; CUDA grid tests verify packing once before parameter expansion "
                  "and validation of matrix inputs before conversion."])
lines.extend(["", "## Full-transit mixed-arithmetic accuracy", "",
              "150 m/s; light-field exit and after 25 excited-state lifetimes with all optical/MW "
              "drives disabled. RC: nine distinct detuning/J-mixture cases. R(2): 60 distinct "
              "detuning/initial-state cases with an EOM. References are the archived finer-step "
              "full trajectories. This does not validate an entire apparatus grid.", "",
              "| System | Variant | Worst relative entry/coherence error | Trace error | Passed |",
              "|---|---|---:|---:|---|"])
path=OUT/"mixed_full.json"
if path.exists():
    for row in json.loads(path.read_text()):
        err=row["accuracy"]
        lines.append(f"| {row['system']} | {row['variant']} | {100*err['max_relative_above_cutoff']:.5g}% | {err['trace_error']:.3g} | {row['passed']} |")
lines.extend(["", "Acceptance: each complex entry with reference magnitude >=1e-6 must have "
              "<=1% relative error; smaller entries <=1e-6 absolute. Trace error <1e-6; "
              "minimum density eigenvalue >=-1e-7. Complete near-zero and positivity diagnostics "
              "are retained in [mixed_full.json](mixed_full.json). No tolerances or state-space "
              "pruning were changed to pass a candidate.", "", "## Findings", "",
              "None of these kernel prototypes improves evolution consistently over the current "
              "default on the tested RC/R(2) systems. Early 512-trajectory gather gains in RC did "
              "not survive 1,024-trajectory confirmation; adding precomputed conjugation signs "
              "also loses. The tiled-direct hybrid, including its gather combination, does not "
              "beat the best existing RHS consistently. Mixed sparse/RHS arithmetic costs roughly "
              "20% more runtime at large batch sizes despite passing the stated full-transit "
              "accuracy checks at 150 m/s. Persistent FP64 integration is about 1.6 times the "
              "baseline runtime at 1,024 trajectories; its FP32-product combination is about "
              "1.9-2.0 times baseline. Keep production evolution kernels unchanged.", "",
              "For repeated 1 us calls at 1,024 trajectories, graph/buffer reuse reduces wall time "
              "by approximately 2.9% (RC) and 2.3% (R2) compared with fresh calls using packed "
              "initial inputs. Most apparent setup saving against matrix inputs comes from "
              "avoiding repeated CPU packing: the already-supported packed input reduces these "
              "short fresh-call wall times by approximately 46% and 32%, respectively. That "
              "existing API choice should be used before adding a reusable-session API. These "
              "percentages do not predict full-transit speedups, where evolution dominates. "
              "The subsequent 10 us RC/R(2) comparison finds no reliable reuse gain either: "
              "both descriptive paired intervals include zero, and R(2) has a slower reused "
              "median. All endpoint packed densities match exactly. This supports leaving "
              "graph/buffer reuse out of the production API for these workloads. "
              "Any session implementation would need explicit invalidation for changed shapes, "
              "support/varying patterns, time grids and discontinuities; this prototype only "
              "reuses compatible detuning batches.", "", "## Reproduction", "",
              "Use `benchmarks/.venv-gpu-bench/Scripts/python.exe` for:", "",
              "- `benchmarks/check_cuda_remaining.py` (512 screen); add `--batch 1024 --rounds 6 "
              "--name confirm1024 --variants base_tiled base_direct gather_tiled gather_direct hybrid32 "
              "gather_hybrid32 fp32_product_direct` for confirmation.",
              "- `benchmarks/check_cuda_persistent.py` (use `--threads 256 --chunks 16 --batch 1024 "
              "--rounds 3` for large-batch confirmation; add `--arithmetic fp32_products` for the combination).",
              "- `benchmarks/check_cuda_session_reuse.py --batch 1024`.",
              "- `benchmarks/check_cuda_mixed_full.py`.",
              "- `benchmarks/report_cuda_remaining.py` regenerates this report.", ""])
(OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
