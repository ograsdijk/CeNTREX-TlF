"""Plot accepted trajectory timings and accuracy for the RC kernel matrix.

Consumes saved JSON summaries only; it never runs simulations or RHS profiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "rc_propagation_results" / "rc_kernel_matrix_v1"
MODES = (
    "structured", "structured_upper", "expanded_sparse",
    "experimental_structured_sparse_dense",
    "experimental_structured_sparse_dense_diag",
    "experimental_structured_sparse_dense_soa",
    "experimental_structured_sparse_dense_single_unpack", "phase_gauge",
)
LABELS = {
    "structured": "structured",
    "structured_upper": "structured_upper",
    "expanded_sparse": "expanded_sparse",
    "experimental_structured_sparse_dense": "sparse-dense",
    "experimental_structured_sparse_dense_diag": "diag split",
    "experimental_structured_sparse_dense_soa": "real/imag scratch",
    "experimental_structured_sparse_dense_single_unpack": "single unpack",
    "phase_gauge": "phase gauge",
}
COLORS = dict(zip(MODES, plt.get_cmap("tab10").colors, strict=False))
DEFAULT_CASES = (
    "nominal", "detuning_-100MHz", "detuning_-20MHz", "detuning_-5MHz",
    "detuning_+5MHz", "detuning_+20MHz", "detuning_+100MHz",
    "optical_0.3Gamma", "optical_3Gamma", "microwave_0.3Gamma", "microwave_3Gamma",
)
CASE_MARKERS = dict(zip(DEFAULT_CASES, ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"), strict=True))


class GeneralScalarFormatter(ScalarFormatter):
    """ScalarFormatter that keeps power ticks concise (0.09, 1, 9)."""

    def _set_format(self) -> None:
        self.format = "%g"
        self._usetex = False


def _read(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def load_matrix(root: Path) -> tuple[dict[str, dict], dict[str, Any]]:
    manifest = _read(root / "run.json") or _read(root / "run.progress.json") or {}
    cases = {}
    for path in sorted(root.glob("*/case_summary.json")):
        row = _read(path)
        if row and row.get("case"):
            cases[str(row["case"])] = row
    for row in manifest.get("rows", []):
        if isinstance(row, dict) and row.get("case"):
            cases.setdefault(str(row["case"]), row)
    return cases, manifest


def _pass(row: dict) -> bool:
    return bool((row.get("accuracy_vs_reference") or {}).get("passes", False))


def _accepted(rows: list[dict], mode: str, phase: str, repeat: int | None = None) -> dict | None:
    candidates = [
        row for row in rows
        if row.get("mode") == mode
        and row.get("phase") in {phase, "accuracy_retry"}
        and (phase != "screening" or row.get("repeat", 1) == 1)
        and (repeat is None or row.get("repeat") == repeat)
        and _pass(row)
        and not row.get("superseded_by_tighter_retry", False)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (
        float(row.get("reltol", np.inf)), row.get("phase") != "accuracy_retry"
    ))


def _ratios(summary: dict, mode: str) -> tuple[list[float], list[bool]]:
    rows = summary.get("all_trial_rows") or summary.get("screening_rows") or []
    pairs = []
    base = _accepted(rows, "structured", "screening")
    candidate = _accepted(rows, mode, "screening")
    if base and candidate:
        pairs.append((candidate, base))
    for repeat in (2, 3):
        base = _accepted(rows, "structured", "paired_repeat", repeat)
        candidate = _accepted(rows, mode, "paired_repeat", repeat)
        if base and candidate:
            pairs.append((candidate, base))
    ratios = [float(a["wall_seconds"]) / float(b["wall_seconds"]) for a, b in pairs]
    mixed = [
        (a.get("reltol"), a.get("abstol")) != (b.get("reltol"), b.get("abstol"))
        for a, b in pairs
    ]
    return ratios, mixed


def _x_values(summary: dict) -> list[tuple[str, float]]:
    name = str(summary.get("case", ""))
    if name == "nominal":
        return [("detuning", 0.0), ("optical", 1.0), ("microwave", 1.0)]
    if name.startswith("detuning_") and name.endswith("MHz"):
        try:
            return [("detuning", float(name[9:-3]))]
        except ValueError:
            return []
    for group in ("optical", "microwave"):
        if name.startswith(group + "_") and name.endswith("Gamma"):
            try:
                ratio = float(name[len(group) + 1:-5])
            except ValueError:
                return []
            return [(group, ratio * ratio)]
    return []


def plot_runtime(cases: dict[str, dict], manifest: dict, output: Path) -> None:
    expected = manifest.get("case_names") or DEFAULT_CASES
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    groups = (
        ("detuning", axes[0], "Detuning (MHz)", [-100, -20, 0, 20, 100], False),
        ("optical", axes[1], "Optical power ratio Ω²/Γ²", [0.09, 1, 9], True),
        ("microwave", axes[2], "Microwave power ratio Ω²/Γ²", [0.09, 1, 9], True),
    )
    for group, ax, xlabel, ticks, logx in groups:
        group_low = np.inf
        group_high = 0.0
        ax.set_title({"detuning": "Detuning", "optical": "Optical power", "microwave": "Microwave power"}[group])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Full-trajectory wall ratio vs structured")
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(0.06, 14.0)
            ax.xaxis.set_major_locator(FixedLocator([0.09, 1.0, 9.0]))
            power_formatter = GeneralScalarFormatter()
            power_formatter.set_useOffset(False)
            power_formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(power_formatter)
        else:
            ax.set_xlim(-110.0, 110.0)
        ax.grid(True, which="both", alpha=0.22)
        for mode in MODES:
            points = []
            for case in expected:
                summary = cases.get(case)
                if not summary:
                    continue
                xv = [x for x in _x_values(summary) if x[0] == group]
                if not xv:
                    continue
                x = xv[0][1]
                ratios, mixed = _ratios(summary, mode)
                if ratios:
                    points.append((x, float(np.median(ratios)), min(ratios), max(ratios), len(ratios), any(mixed)))
            points.sort(key=lambda p: p[0])
            if not points:
                continue
            xs = np.array([p[0] for p in points])
            med = np.array([p[1] for p in points])
            low = np.array([p[2] for p in points])
            high = np.array([p[3] for p in points])
            color = COLORS[mode]
            ax.plot(xs, med, color=color, linewidth=1.0, alpha=0.8)
            for x, y, lo, hi, count, mismatch in points:
                group_low = min(group_low, lo)
                group_high = max(group_high, hi)
                marker = {1: "s", 2: "^", 3: "o"}.get(count, "D")
                ax.errorbar([x], [y], yerr=[[y-lo], [hi-y]], fmt=marker, color=color,
                            markersize=5, capsize=3, linewidth=1)
                if mismatch:
                    ax.annotate("†", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
        if not logx:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in ticks]))
        ax.yaxis.set_major_locator(FixedLocator([0.75, 1.0, 1.5, 2.0, 3.0, 4.0]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.75", "1", "1.5", "2", "3", "4"]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if group == "detuning":
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        if np.isfinite(group_low) and group_high > 0.0:
            ax.set_ylim(max(0.2, min(0.6, group_low * 0.85)), max(4.5, group_high * 1.15))

    handles = [Line2D([0], [0], color=COLORS[m], marker="o", label=LABELS[m]) for m in MODES]
    handles.extend([
        Line2D([0], [0], color="#444", marker="s", linestyle="none", label="1 timing"),
        Line2D([0], [0], color="#444", marker="^", linestyle="none", label="2 timings"),
        Line2D([0], [0], color="#444", marker="o", linestyle="none", label="3 timings; range"),
    ])
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.99))
    incomplete = [f"{name}({cases[name].get('status', 'unknown')})" for name in expected
                  if name in cases and cases[name].get("status") not in {"complete", "complete_with_failures"}]
    missing = [name for name in expected if name not in cases]
    aliases = {
        "structured": "str", "structured_upper": "up", "expanded_sparse": "exp",
        "experimental_structured_sparse_dense": "sd",
        "experimental_structured_sparse_dense_diag": "diag",
        "experimental_structured_sparse_dense_soa": "soa",
        "experimental_structured_sparse_dense_single_unpack": "copy",
        "phase_gauge": "gauge",
    }
    not_run, failed = [], []
    for name in expected:
        summary = cases.get(name)
        if summary is None:
            continue
        entries = summary.get("not_run_budget", [])
        if entries:
            compact = [f"{aliases.get(item.split(':')[0], item.split(':')[0])}/{item.split(':', 1)[1]}" for item in entries]
            not_run.append(f"{name}({','.join(compact)})")
        bad = []
        for row in summary.get("all_trial_rows", []):
            if not _pass(row) and not row.get("superseded_by_tighter_retry", False):
                mode = aliases.get(row.get("mode"), row.get("mode", "?"))
                phase = str(row.get("phase", "?"))
                tag = f"{mode}/{phase}"
                if tag not in bad:
                    bad.append(tag)
        if bad:
            failed.append(f"{name}({','.join(bad)})")
    complete_count = sum(
        bool(cases.get(name, {}).get("screening_complete"))
        and cases.get(name, {}).get("status") != "partial_budget"
        for name in expected
    )
    final = bool(manifest.get("complete")) and complete_count == len(expected)
    title_status = "FINAL" if final else f"PARTIAL {complete_count}/{len(expected)} cases"
    if missing or incomplete:
        status_line = "Case status: " + (
            (f"missing {','.join(missing)}; " if missing else "")
            + (f"incomplete {','.join(incomplete)}" if incomplete else "")
        )
    else:
        status_line = "Case status: all case summaries complete"
    status_lines = [
        status_line,
        "Not run: " + ("; ".join(not_run) if not_run else "none"),
        "Accuracy failed: " + ("; ".join(failed) if failed else "none (superseded retries omitted)"),
    ]
    for index, line in enumerate(status_lines):
        fig.text(0.012, 0.018 + index * 0.022, line, fontsize=5.5, va="bottom", wrap=True)
    fig.text(
        0.015, 0.09,
        "Accuracy-passing trajectory rows only; paired ratios share the structured run for that round. "
        "Bars show min–max, not confidence intervals; squares/triangles/circles indicate 1/2/3 timings. "
        "† marks candidate/base tolerance mismatch. Nominal Ω/Γ=1 appears at power ratio 1 in both power panels. "
        "RHS profiles are excluded.",
        fontsize=8,
    )
    fig.suptitle(f"RC kernel matrix ({title_status}): accepted full-trajectory runtime", y=1.035, fontsize=14)
    fig.subplots_adjust(top=0.82, bottom=0.24, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(cases: dict[str, dict], manifest: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7), constrained_layout=False)
    failures_by_case: dict[str, set[str]] = {}
    not_run_by_case: dict[str, set[str]] = {}
    missing = []
    labeled_modes: set[str] = set()
    for case, summary in cases.items():
        for row in summary.get("all_trial_rows", []):
            mode = row.get("mode")
            score = row.get("accuracy_vs_reference") or {}
            metric = (score.get("all_saved") or {}).get("density_max_abs")
            wall = row.get("wall_seconds")
            if mode not in MODES or metric is None or wall is None:
                continue
            if not np.isfinite(metric) or not np.isfinite(wall) or metric < 0 or wall <= 0:
                continue
            passed = _pass(row)
            label = LABELS[mode] if mode not in labeled_modes else None
            labeled_modes.add(mode)
            marker = CASE_MARKERS.get(case, "o")
            ax.scatter(wall, max(float(metric), 1e-18), color=COLORS[mode],
                       marker=marker, facecolors=COLORS[mode] if passed else "none",
                       edgecolors=COLORS[mode] if passed else "#b91c1c", s=40,
                       linewidths=1.1, alpha=0.82, label=label)
            if not passed:
                failures_by_case.setdefault(case, set()).add(LABELS[mode])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1e-7, color="#444", linestyle="--", linewidth=1, label="density target 1e-7")
    ax.set_xlabel("Full trajectory wall time (s; density plus photon solve)")
    ax.set_ylabel("Maximum saved-time density absolute error")
    expected = manifest.get("case_names") or DEFAULT_CASES
    complete_count = sum(
        bool(cases.get(name, {}).get("screening_complete"))
        and cases.get(name, {}).get("status") != "partial_budget"
        for name in expected
    )
    final = bool(manifest.get("complete")) and complete_count == len(expected)
    title_status = "FINAL" if final else f"PARTIAL {complete_count}/{len(expected)} cases"
    ax.set_title(f"Trajectory accuracy versus runtime ({title_status}); open red edges mark failed rows")
    ax.grid(True, which="both", alpha=0.22)
    mode_handles = [Line2D([0], [0], color=COLORS[m], marker="o", linestyle="none", label=LABELS[m]) for m in MODES]
    case_handles = [
        Line2D([0], [0], color="#555", marker=CASE_MARKERS[name], linestyle="none", label=name)
        for name in DEFAULT_CASES
    ]
    ax.legend(handles=mode_handles + case_handles, loc="best", fontsize=7, ncol=3)
    expected = manifest.get("case_names") or DEFAULT_CASES
    for name in expected:
        if name not in cases:
            missing.append(f"{name}: no summary")
        else:
            aliases = {
                "structured": "str", "structured_upper": "up", "expanded_sparse": "exp",
                "experimental_structured_sparse_dense": "sd",
                "experimental_structured_sparse_dense_diag": "diag",
                "experimental_structured_sparse_dense_soa": "soa",
                "experimental_structured_sparse_dense_single_unpack": "copy",
                "phase_gauge": "gauge",
            }
            for item in cases[name].get("not_run_budget", []):
                mode, _, stage = item.partition(":")
                not_run_by_case.setdefault(name, set()).add(f"{aliases.get(mode, mode)}/{stage}")
    failure_text = "; ".join(
        f"{case}[{','.join(sorted(modes))}]"
        for case, modes in sorted(failures_by_case.items())
    ) or "none"
    not_run_text = "; ".join(
        f"{case}[{','.join(sorted(items))}]"
        for case, items in sorted(not_run_by_case.items())
    ) or "none"
    status_lines = [
        "Accuracy failures (case[methods]): " + failure_text,
        "Not run (case[method/stage]): " + not_run_text,
        "Missing summaries: " + (",".join(missing) if missing else "none"),
    ]
    for index, line in enumerate(status_lines):
        fig.text(0.01, 0.045 + index * 0.025, line, fontsize=6.5, va="bottom", wrap=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    cases, manifest = load_matrix(args.root)
    stem = args.output_stem or args.root / "kernel_matrix"
    runtime = stem.with_name(stem.name + "_runtime.png")
    accuracy = stem.with_name(stem.name + "_accuracy.png")
    plot_runtime(cases, manifest, runtime)
    plot_accuracy(cases, manifest, accuracy)
    print(f"Loaded {len(cases)} case summaries from {args.root}")
    print(f"Wrote {runtime} and {accuracy}")


if __name__ == "__main__":
    main()
