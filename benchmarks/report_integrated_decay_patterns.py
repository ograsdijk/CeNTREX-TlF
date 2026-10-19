"""Summarize the paired integrated RC/optical decay and pattern experiment."""

import json

import check_rc_integrated_decay_patterns as bench


def estimate(item):
    lo, hi = item["paired_bootstrap_95_percent"]
    return f"{item['paired_median_reduction_percent']:+.2f}% [{lo:+.2f}, {hi:+.2f}]"


def main():
    path = bench.base.RESULTS / "integrated_decay_patterns_paired.json"
    report = json.loads(path.read_text())
    assert len(report["systems"]) == 4
    for row in report["systems"]:
        assert len(row["times_by_round_s"]) == bench.ROUNDS
        assert all(
            check["accuracy"]["passed"] and check["trace_error"] < 1e-9
            for checks in row["density_checks"]
            for check in checks
        )
        bench.statistics(row)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Paired decay and active-pattern tests on integrated D+S+U",
        "",
        "Completed 2026-10-17. This replaces the tentative interpretation of the earlier",
        "separate RHS timings. Prepared Hamiltonian assembly and the shared coefficient",
        "DAG are enabled in every run. No production code or defaults were changed.",
        "",
        "## Incremental solve-time changes",
        "",
        "Positive percentages mean less time; negative percentages mean more time.",
        "Values are median within-round reductions versus the current pipeline in the",
        "same benchmark build. Brackets show descriptive 95% paired bootstrap intervals.",
        "",
        "| System | Nonzero decay alone | Active patterns alone | Both | Both vs unpatched production control |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["systems"]:
        summary = row["summary"]
        lines.append(
            f"| {row['system']} | {estimate(summary[1])} | {estimate(summary[2])} | "
            f"{estimate(summary[3])} | {estimate(summary[3]['versus_unpatched_control'])} |"
        )
    lines += [
        "",
        "R(2), R(0) and Q(1) have no polarization switching here: their active pattern",
        "has the same sparsity as the union. Differences there measure incidental",
        "selection/layout overhead, not the removal of physical interactions.",
        "The unpatched-control comparison matters: a gain versus the hooked baseline",
        "is not automatically the same net gain over the installed algorithm.",
        "",
        "The paired RC results support small individual improvements: about 2.1% for",
        "decay and 0.6% for patterns, with 3.2% for both within the hooked pipeline.",
        "The net combined reduction versus the unmodified production control is 2.6%",
        "(descriptive interval 2.2-3.3%). The earlier statement that neither addition",
        "helps individually was too strong. A special combined interaction is not",
        "established by these samples. R(2) regresses; R(0)/Q(1) show no reliable",
        "incremental benefit. This supports an optional RC combination, not a new",
        "generic default. Long apparatus transits remain untested here.",
        "",
        "## RC interaction and marginal effects",
        "",
    ]
    rc = next(row for row in report["systems"] if row["system"] == "rc")
    for item in rc["marginal_effects"]:
        lines.append(f"- {item['effect']}: {estimate(item)}")
    interaction = rc["log_interaction"]
    lo, hi = interaction["bootstrap_95"]
    lines += [
        "",
        f"RC log interaction `log(T_both*T_current/(T_decay*T_patterns))`: "
        f"{interaction['median']:+.6f}, interval [{lo:+.6f}, {hi:+.6f}].",
        "Negative values favor both together beyond multiplicative independence; an",
        "interval spanning zero does not demonstrate such an interaction. Medians of",
        "paired ratios need not exactly equal ratios of separate medians.",
        "",
        "## Absolute solve-time medians and hook overhead",
        "",
        "| System | Current, hooked (s) | Decay (s) | Patterns (s) | Both (s) | Unpatched (s) | Hook overhead vs unpatched |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["systems"]:
        summary = row["summary"]
        values = " | ".join(f"{s['median_s']:.6f}" for s in summary)
        overhead = -summary[0]["versus_unpatched_control"]["paired_median_reduction_percent"]
        lines.append(f"| {row['system']} | {values} | {overhead:+.2f}% |")
    lines += [
        "",
        "## Method and scope",
        "",
        "The baseline and candidates call the current real production RHS, parameter",
        "evaluation, prepared assembly, shared coefficient DAG and Tsit5 source. A",
        "benchmark-only snapshot adds two hooks: select a prepared gate-conditioned",
        "sparse pattern, and replace the structured decay loop by nonzero damping/transfer",
        "lists. It does not use the earlier generic assembly adapter. All four candidates",
        "share one compiled binary and dispatch; a second binary contains the unmodified",
        "production RHS as a separate control. Sources and snapshot hashes are in the JSON.",
        "",
        "Each system has 12 paired rounds, each including all four configurations and",
        "the unpatched control. The four-configuration order uses randomized/relabelled",
        "four-round Williams blocks; the control is inserted at a cycling position.",
        "Preparation/build time is excluded. Warmup precedes sampling. One CPU thread",
        "and one BLAS thread are used; no unrelated benchmark runs concurrently.",
        "Intervals resample whole paired rounds (20,000 bootstrap draws, fixed seed),",
        "not individual timings. With only 12 rounds these are descriptive intervals;",
        "they do not cover compiler-layout changes, other workloads or systematic effects.",
        "",
        "All are the existing short performance fixtures: 10 us trajectories, E=[0,0,200]",
        "V/cm, B=[0,0,0.01] G, optical EOM at 1.6 MHz with beta=3.8. RC uses 1 MHz",
        "microwave polarization switching with quarter-period offset, all four states,",
        "exact 250 ns segment boundaries and continuously evaluated EOM phase. Pattern",
        "selection overhead at segment boundaries is included. This is not the zero-E",
        "apparatus model, a full transit or a parallel production scan.",
        "",
        "Candidate atol/rtol: 1e-11/1e-9. Strict unpatched reference: 1e-13/1e-11.",
        "Every trajectory checks complex density entries: 1% relative error at reference",
        "magnitude >=1e-6, absolute error <=1e-6 below, including coherences. Trace error",
        "is below 1e-9. RHS checks cover random positive Hermitian and initial densities,",
        "four EOM times/all RC gate states, Rabi multipliers 0, 0.3, 1 and 2 and changed",
        "detunings. Raw JSON retains errors, RHS/accepted/rejected counts, execution",
        "orders, every observation, and source hashes.",
        "",
        "Active patterns remove coefficients only when gate-zero propagation proves",
        "them identically zero throughout the segment; instantaneous EOM zeros do not",
        "qualify. Decay lists remove exactly zero operations, with no new physical cutoff.",
        "The `counts` array records union NNZ, damping entries, transfer entries, then",
        "NNZ for each binary gate configuration in native parameter-slot order.",
        "",
        "## Reproduce",
        "",
        "```",
        ".venv/Scripts/python.exe benchmarks/check_rc_integrated_decay_patterns.py",
        ".venv/Scripts/python.exe benchmarks/report_integrated_decay_patterns.py",
        "```",
        "",
        "Sources and results live outside `.cache`; only compiled extensions/snapshots",
        "live there. The native libraries are compiled once and reused across systems.",
    ]
    output = path.with_name("INTEGRATED_DECAY_PATTERNS_PAIRED_REPORT.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bench.base.emit(
        "Integrated decay/pattern paired comparison and density validation complete",
        report=output.name,
    )
    for row in report["systems"]:
        print(row["system"], [estimate(s) for s in row["summary"][1:4]])
    print("RC interaction:", interaction)


if __name__ == "__main__":
    main()
