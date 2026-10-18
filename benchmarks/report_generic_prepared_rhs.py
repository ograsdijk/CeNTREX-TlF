"""Write the permanent report once all prepared-data experiments have finished."""

import hashlib
import json

import bench_generic_prepared_rhs as bench


def main():
    output = bench.base.RESULTS
    reports = [
        json.loads((output / (name + "_generic_prepared.json")).read_text())
        for name in ["rc", "r2", "r0", "q1"]
    ]
    for report in reports:
        assert len(report["rhs"]) == 65
        assert all(
            len(r["samples_s"]) == 5 and r["accuracy"]["passed"] for r in report["trajectories"]
        )
    checks = json.loads((output / "generic_prepared_runtime_checks.json").read_text())
    assert checks["passed"]
    lines = [
        "# Generic prepared-data CPU RHS experiments",
        "",
        "**Follow-up:** [Matched integrated decay/pattern tests](INTEGRATED_DECAY_PATTERNS_PAIRED_REPORT.md)",
        "supersede the tentative interpretation of their separate RHS timings below.",
        "",
        "Completed 2026-10-16. One reusable Rust binary serves RC, R(2), R(0) and Q(1);",
        "no system-specific native arithmetic or recompilation between models is used.",
        "Production library code is unchanged. This tests the four remaining generic",
        "opportunities after the [generated-RHS experiments](SEPARATELY_GENERATED_RHS_REPORT.md).",
        "",
        "## Trajectory results",
        "",
        "Matched 10 us Tsit5 solves, five observations per candidate, randomized candidate",
        "order in each repeat after warmup. Medians in seconds. The best candidate is",
        "selected from the trajectory-tested finalists, not solely from RHS microtimings.",
        "",
        "| System | Library D+S+U | Adapter control | Best candidate | Reduction vs library | Candidate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for report in reports:
        trajectories = report["trajectories"]
        baseline = next(r for r in trajectories if r["mode"] == 0)["median_s"]
        control = next(r for r in trajectories if r["mode"] == 16)["median_s"]
        best = min(trajectories, key=lambda r: r["median_s"])
        lines.append(
            f"| {report['system']} | {baseline:.6f} | {control:.6f} | "
            f"{best['median_s']:.6f} | {100 * (1 - best['median_s'] / baseline):.2f}% | {best['method']} |"
        )
    lines += [
        "",
        "The adapter control implements the same assembly/decay/commutator arithmetic",
        "with a separate coefficient buffer and generic benchmark dispatch. Differences",
        "from the library baseline expose harness/layout overhead; do not attribute every",
        "control-relative improvement to a production speedup. Compare against the actual",
        "library column. Small differences among equivalent/no-op combinations are noise.",
        "",
        "Prepared Hamiltonian assembly alone reduces solve time by 7.5-9.2% across",
        "these four fixtures and is the clearest generic change to pursue. D+S+U SoA",
        "remains the winning commutator family on each. For RC, adding nonzero decay",
        "and active gate patterns raises the measured reduction to about 11%; the",
        "full four-change combination gives 10.6%. The other optical fixtures have",
        "no gate configurations to specialize and no repeated coefficient DAG nodes,",
        "so tiny differences between those enabled/disabled combinations are not evidence",
        "of useful CSE or switching gains. Enabling every change is not uniformly best.",
        "Some RC samples have large timing outliers; all raw observations are retained.",
        "A sensible integration order is prepared assembly first, then validate the",
        "additional RC combination in the library and a production-context fixture.",
        "",
        "## Individual changes and full combination with D+S+U",
        "",
        "| System | Variant | Solve seconds | Reduction vs library |",
        "|---|---|---:|---:|",
    ]
    for report in reports:
        rows = {r["mode"]: r for r in report["trajectories"]}
        baseline = rows[0]["median_s"]
        for mode in [16, 17, 18, 20, 24, 31]:
            row = rows[mode]
            label = "adapter control" if mode == 16 else bench.describe(mode)
            lines.append(
                f"| {report['system']} | {label} | {row['median_s']:.6f} | "
                f"{100 * (1 - row['median_s'] / baseline):.2f}% |"
            )
    lines += [
        "",
        "## Experimental matrix and implementation",
        "",
        "Each system has all 16 on/off combinations of these four changes crossed with",
        "four commutator variants: D+S+U with real/imaginary scratch (SoA), S+U SoA,",
        "D+S+U complex scratch, and S+U complex scratch. This gives 64 RHS candidates",
        "plus the unchanged library D+S+U reference. It is not a scan over every solver",
        "or all execution modes in the library.",
        "",
        "1. **prepared_H:** copy a prepared packed static triangle and apply cached",
        "   (upper offset, numeric multiplier) lists for each runtime coefficient.",
        "   Replaces repeated triangle clearing, dense-to-upper traversal and checked indices.",
        "2. **nonzero_decay:** precompute nonzero coherence damping and population transfer",
        "   operations; skip exactly zero damping. No extra physical cutoff is introduced.",
        "3. **shared_coefficient_DAG:** intern identical instruction subtrees across scalar",
        "   coefficient expressions and evaluate shared nodes once. The existing compound",
        "   parameter graph remains evaluated once per RHS; this is structural reuse, not",
        "   algebraic SymPy CSE or native expression compilation.",
        "4. **active_pattern:** prepare sparsity patterns for externally frozen binary gate",
        "   configurations; select at segment boundaries. Remove coefficients only when",
        "   zero propagation from a disabled gate proves them identically zero. Never",
        "   infer a permanent zero from an instantaneous EOM/drive value. RC prepares",
        "   16 combinations of its four PA/PY slots, of which four are physically used.",
        "",
        "The kernels use runtime plan dimensions, arrays and coefficients. The prototype",
        "requires decomposed scalar expressions and externally supplied binary gate states;",
        "its adapter recognizes the RC fixture's gate names. A library integration would",
        "need general gate registration/fallback, static-H caching and all supported",
        "expression types. Native compilation is shared, but preparation is per system.",
        "",
        "## Preparation and expression reuse",
        "",
        "| System | Prepared data creation (ms) | Original coefficient instructions | Shared DAG nodes |",
        "|---|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['system']} | {1000 * report['preparation_s']:.3f} | "
            f"{report['coefficient_instruction_count']} | {report['shared_DAG_nodes']} |"
        )
    lines += [
        "",
        f"One-time benchmark extension compilation: {reports[0]['build']['compile_s']:.3f} s",
        "with cached Cargo dependencies. Preparation above is one observation and includes",
        "all candidate data structures; it is not a carefully repeated preparation benchmark.",
        "The existing compound graph has already captured most coefficient reuse here.",
        "",
        "## Screening, accuracy and limits",
        "",
        "The RHS screen uses five samples of 1000 repetitions at four EOM times, with",
        "the initial gates fixed for timing. Correctness checks cover all four gate states.",
        "Trajectory finalists include the control, all four individual changes and the",
        "full combination on D+S+U, plus the three fastest RHS combinations in each",
        "commutator family. Trajectories cycle through all four RC states, including",
        "boundary selection overhead. Not every one of the 64 combinations gets a",
        "trajectory benchmark; the raw JSON records the exact shortlist.",
        "",
        "Same Ryzen 7 9800X3D, one CPU thread and one BLAS thread as the earlier tests.",
        "Same performance fixtures, E=[0,0,200] V/cm and B=[0,0,0.01] G; these are not",
        "the zero-electric-field production RC apparatus model. Optical EOM is 1.6 MHz",
        "with beta=3.8. Switching is frozen at exact 250 ns segment midpoints; EOM retains",
        "absolute time. Candidate atol/rtol 1e-11/1e-9; strict reference 1e-13/1e-11.",
        "",
        "All trajectory candidates pass the full complex-density entry criterion:",
        "1% relative error at reference magnitude >=1e-6, absolute error <=1e-6 below.",
        "RHS sum checks use random positive Hermitian densities, four EOM times and",
        "four RC configurations, normalized error <1e-10. Additional runtime-change",
        "checks cover all 64 candidates on all four systems at optical/microwave Rabi",
        "multipliers 0, 0.3 and 2, with detunings -3.7, 0 and +5.1 MHz. The same binary",
        "is reused. No long apparatus transit, parallel scan throughput or independent",
        "GPU comparison is claimed. Winners need production-context validation before promotion.",
        "",
        "## Reproduce and artifacts",
        "",
        "```",
        ".venv/Scripts/python.exe benchmarks/bench_generic_prepared_rhs.py",
        ".venv/Scripts/python.exe benchmarks/check_generic_prepared_rhs.py",
        ".venv/Scripts/python.exe benchmarks/report_generic_prepared_rhs.py",
        "```",
        "",
        "Use PYTHONIOENCODING=utf-8. Helpers: `generic_prepared_rhs.rs` and",
        "`generic_coefficient_dag.rs`. Raw samples: `*_generic_prepared.json`.",
        "Runtime checks: `generic_prepared_runtime_checks.json`. Hashes:",
        "`generic_prepared_source_manifest.json`. These sources and results are outside `.cache`.",
    ]
    (output / "GENERIC_PREPARED_RHS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    paths = [
        bench.base.HERE / name
        for name in [
            "bench_generic_prepared_rhs.py",
            "generic_prepared_rhs.rs",
            "generic_coefficient_dag.rs",
            "specialized_rhs_native.rs",
        ]
    ]
    paths += [
        bench.FOLDER / name for name in ["lib.rs", "rhs.rs", "eval.rs", bench.MODULE + ".pyd"]
    ]
    paths += [
        bench.base.REPO / "rust/src/lindblad" / name for name in ["rhs.rs", "plan.rs", "eval.rs"]
    ]
    (output / "generic_prepared_source_manifest.json").write_text(
        json.dumps(
            {
                str(path.relative_to(bench.base.REPO)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in paths
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bench.base.emit(
        "Generic prepared-data factorial benchmarks and runtime checks complete",
        report="GENERIC_PREPARED_RHS_REPORT.md",
    )


if __name__ == "__main__":
    main()
