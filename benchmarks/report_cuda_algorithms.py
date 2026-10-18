"""Render the archived CUDA algorithm experiment results."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cuda_algorithm_results"


def median(values):
    return float(np.median(values))


def report():
    lines = [
        "# Full-density CUDA algorithm experiments",
        "",
        "RTX 5070 Ti; CuPy FP64; RC (70 levels) and R(2) with EOM (40 levels). "
        "Benchmark measurements followed by opt-in integration of exact reduction and periodic maps; "
        "production solver defaults unchanged.",
        "",
        "Acceptance checks complex density entries: <=1% relative error for reference "
        "magnitude >=1e-6, <=1e-6 absolute below that floor, trace error <=1e-6, "
        "and minimum eigenvalue >=-1e-7. No population-only accuracy shortcut.",
        "",
    ]
    for filename, title in [
        ("reduction.json", "Exact coordinate reduction: expanded sparse RHS"),
        ("structured_reduction.json", "Exact coordinate reduction: structured Hamiltonian RHS"),
    ]:
        path = OUT / filename
        if not path.exists():
            continue
        lines += [
            f"## {title}",
            "",
            "1024 trajectories, 10 us; three randomized interleaved rounds. "
            "Timing includes evolution and full-state download; compilation/capture is separate.",
            "",
            "| System | Method | Median seconds | Change vs current | Accuracy |",
            "|---|---|---:|---:|---|",
        ]
        reduction_rows = json.loads(path.read_text())
        for row in reduction_rows:
            base = median(row["timings"]["current_tiled"])
            for method, times in row["timings"].items():
                t = median(times)
                lines.append(
                    f"| {row['system']} | {method} | {t:.4f} | {(1 - t / base) * 100:+.1f}% | "
                    f"{'pass' if row['checks'][method]['passed'] else 'FAIL'} |"
                )
        for row in reduction_rows:
            if "retained" in row:
                lines += [
                    "",
                    f"{row['system']}: {row['retained']}/{row['coordinates']} real coordinates retained; "
                    f"baseline vs half-step reference passes: {row['baseline_half_step_check']['passed']}.",
                    "",
                ]
        lines += ["", f"Raw data: [{filename}]({filename}).", ""]
    path = OUT / "magnus.json"
    if path.exists():
        lines += [
            "## Matrix-free exponential actions / CF4 Magnus",
            "",
            "Two exponentials per fourth-order commutator-free Magnus step, using fixed-degree "
            "Taylor actions. No dense Liouvillian or eigendecomposition. Uses the existing "
            "tiled Hamiltonian/dissipator kernels, assembles H once per exponential, and clips "
            "every switch boundary. Both the Hamiltonian and the dissipator are scaled in each "
            "exponential. Fixed time steps remain synchronized across the batch.",
            "",
            "| System | Step ns | Degree | Median seconds | Speedup vs Tsit5 | Full density |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in json.loads(path.read_text()):
            base = median(row["baseline_seconds"])
            for candidate in row["rows"]:
                elapsed = median(candidate["timings"])
                lines.append(
                    f"| {row['system']} | {candidate['step_ns']:g} | {candidate['degree']} | "
                    f"{elapsed:.4f} | {base / elapsed:.2f}x | "
                    f"{'pass' if candidate['check']['passed'] else 'FAIL'} |"
                )
        lines += [
            "",
            "Accuracy is checked against half-step Tsit5 for every distinct physical initial-state/parameter point. "
            "These results apply to this CF4/Taylor implementation and tested settings; they do not exhaust "
            "Krylov, Leja, or other exponential-action implementations.",
            "",
            "Short screens also tested 4-64 ns steps and degrees 8-36. At 16 ns, increasing "
            "degree from 24 to 36 did not fix the error, indicating that time ordering, rather "
            "than the exponential polynomial, limits that setting. Raw screens: "
            "[initial](magnus_smoke.json), [higher degree](magnus_smoke_refined.json), "
            "[degree selection](magnus_degree_screen.json).",
            "",
            "Raw data: [magnus.json](magnus.json).",
            "",
        ]
    path = OUT / "r2_finalists.json"
    if path.exists():
        row = json.loads(path.read_text())
        lines += [
            "## R(2): comparison with CK54",
            "",
            "1024 trajectories, 10 us, three randomized interleaved rounds, full density. "
            "Tsit5 step 2 ns; CK54 step 1.95 ns; CF4 step 8 ns with degree-8 Taylor actions.",
            "",
            "| Method | Median seconds | Accuracy |",
            "|---|---:|---|",
        ]
        for method, times in row["timings"].items():
            lines.append(
                f"| {method} | {median(times):.4f} | {'pass' if row['checks'][method]['passed'] else 'FAIL'} |"
            )
        lines += ["", "Raw data: [r2_finalists.json](r2_finalists.json).", ""]
    path = OUT / "full_transit_finalists.json"
    if path.exists():
        lines += [
            "## Full-transit validation of the promising candidates",
            "",
            "Uses the existing strict full-transit reference artifacts; no CPU benchmarks rerun. "
            "RC tests structured reduction; R(2) tests CF4/Taylor8. Both check field-exit density "
            "and density after 25 excited-state lifetimes of dark decay, at 150 m/s. "
            "These are correctness checks on the distinct physical trajectories, not throughput timings.",
            "",
            "| System | Cases | Light time us | Worst relative at exit | Worst relative after decay | Pass |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in json.loads(path.read_text()):
            light, dark = row["light_check"], row["after_decay_check"]
            lines.append(
                f"| {row['system']} | {row['cases']} | {row['light_end_seconds'] * 1e6:.3f} | "
                f"{light['max_relative_above_1e6'] * 100:.4f}% | {dark['max_relative_above_1e6'] * 100:.4f}% | "
                f"{'pass' if light['passed'] and dark['passed'] else 'FAIL'} |"
            )
        lines += ["", "Raw data: [full_transit_finalists.json](full_transit_finalists.json).", ""]
    path = OUT / "periodic.json"
    if path.exists():
        lines += [
            "## Full modulation-cycle map on the GPU",
            "",
            "Fixed detuning 0 MHz. RC explicitly freezes the Gaussian microwave envelopes to unity; "
            "this is a constant-envelope comparison, not the Gaussian production recipe. "
            "R(2) retains its 1.6 MHz EOM. The period is 5 us for RC and 0.625 us for R(2). "
            "All reachable real density coordinates are propagated as a complete basis, so the map "
            "handles arbitrary initial densities in that invariant support. No truncated eigenmode/Krylov model. "
            "Direct references include an additional positive coherent initial-state probe.",
            "",
            "| System | Map build s | Map power s | Resident reuse s | Direct 1024 s | Direct distinct states s | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        periodic_rows = json.loads(path.read_text())
        for row in periodic_rows:
            lines.append(
                f"| {row['system']} | {row['construction_seconds']:.3f} | {row['power_seconds']:.3f} | "
                f"{median(row['reuse_timings']):.4f} | {median(row['direct_batch_timings']):.3f} | "
                f"{row['direct_unique_seconds']:.3f} ({row['physical_initial_states']} states) | "
                f"{'pass' if row['check']['passed'] else 'FAIL'} |"
            )
        for row in periodic_rows:
            lines += [
                "",
                f"{row['system']}: the compiled coefficient periodicity check has relative discrepancy "
                f"{row['coefficient_periodicity_error']:.3g}. The {row['long_cycles']}-cycle full-density "
                f"check ({row['period_seconds'] * row['long_cycles'] * 1e6:g} us) "
                f"{'passes' if row['long_check']['passed'] else 'FAILS'}; worst complex-entry relative "
                f"error {row['long_check']['max_relative_above_1e6'] * 100:.5f}%.",
                "",
            ]
        lines += [
            "",
            "Map build includes library setup/capture, period integration, download, and upload of all map columns. "
            "Reuse times include the full output download with initial states already resident. "
            "Map power is separate. Compare cold build+power+reuse against direct evolution when only one batch is needed. "
            "The 1024-state reuse benchmark repeats a small set of physical states: it measures map throughput, "
            "not 1024 distinct parameter keys. Detuning, power, phase, geometry, or envelope changes require new maps. "
            "For the RC production scan, only three J mixtures share each parameter key; the distinct-state "
            "direct cost is therefore the more relevant cold comparator.",
            "",
            "Raw data: [periodic.json](periodic.json).",
            "",
        ]
    lines += [
        "## Findings and scope",
        "",
        "The structured compact-density kernel improves RC by about 12% in these "
        "three interleaved rounds; R(2) is about 4% slower than its current tiled kernel. "
        "The reduction is exact for the union of the supplied initial supports and all "
        "Hamiltonian/dissipator basis terms. A new initial coherence outside that support "
        "requires recomputing the closure. The compact kernel does not expand the density "
        "back to the full layout at each RHS evaluation.",
        "",
        "For R(2), degree-8 CF4 at 8 ns uses about 19% less runtime than CK54 at 1.95 ns "
        "(about 1.24x speedup). Full-transit complex density error stays below 0.37%. "
        "For RC, the 8 ns CF4 setting is slower and fails the entry-wise relative criterion "
        "(about 1.89% worst error in the 10 us test); increasing degree does not fix it. "
        "These are model-specific results, not universal default settings.",
        "",
        "Complete periodic maps pass long-duration density checks and can be very fast "
        "for many initial densities sharing one strictly periodic generator. Cold construction "
        "is unfavorable for the small distinct-state batches at a parameter point: roughly "
        "39 s versus 2.53 s direct for RC, and 1.10 s versus 0.43 s for R(2), "
        "including one coherent validation state. Gaussian envelopes and actual multipass "
        "trajectories are outside this exact period-reuse experiment.",
        "",
        "At the benchmark stage there were no production library changes. Exact reduction and "
        "periodic maps have since been integrated as opt-in APIs (see "
        "[solver documentation](../../README_OBE_SOLVER.md)); CF4 remains a prototype. "
        "The reduction API supports existing photon/output modes; the periodic API supports "
        "full/population/selected densities without photon accumulation. Full density coordinates "
        "are retained or exactly reconstructed; no rate-equation or population-only model is used.",
        "",
        "Validation: four focused tests cover an independent complex commutator/dissipator, "
        "exact support closure, fourth-order convergence, and compact-kernel reconstruction "
        "with an initially coherent density. All four pass on the GPU environment.",
        "",
        "## Reproduction and prior work",
        "",
        "Use `benchmarks/.venv-gpu-bench/Scripts/python.exe`; scripts set the project CuPy cache directory. "
        "Run from the repository root:",
        "",
        "```powershell",
        "python benchmarks/check_cuda_algorithms.py reduction --batch 1024 --duration-us 10 --repeats 3",
        "python benchmarks/check_cuda_algorithms.py structured_reduction --batch 1024 --duration-us 10 --repeats 3",
        "python benchmarks/check_cuda_algorithms.py magnus --batch 1024 --duration-us 10 --repeats 3",
        "python benchmarks/check_cuda_algorithms.py periodic --batch 1024 --cycles 8 --repeats 3",
        "python benchmarks/check_cuda_algorithm_finalists.py",
        "python -m pytest benchmarks/test_cuda_algorithms.py -q -p no:cacheprovider",
        "python benchmarks/report_cuda_algorithms.py",
        "```",
        "",
        "Earlier CPU-only investigations exist: "
        "[density support reduction](../rust_specialized_rhs_results/density_reduction/REPORT.md) and "
        "[periodic map / Arnoldi](../rc_propagation_results/README.md). "
        "Those are separate from these CUDA experiments. The old diagonal integrating-factor and "
        "per-level rotating-frame experiments are also separate from the full-generator CF4 action here.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    report()
