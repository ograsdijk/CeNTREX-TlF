"""Plot error versus propagation time from an RC tolerance-sweep JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _positive(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return max(float(value), 1e-18)


def plot_sweep(json_path: Path, output_path: Path, repeats_path: Path | None = None) -> None:
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    if not artifact.get("complete") or not artifact.get("timings_accepted"):
        raise ValueError("refusing to plot an incomplete or timing-rejected sweep")
    if not artifact.get("reference_validated"):
        raise ValueError("the sweep has no validated reference")
    repeated_medians: dict[str, float] = {}
    if repeats_path is not None:
        repeats = json.loads(repeats_path.read_text(encoding="utf-8"))
        if not repeats.get("complete") or not repeats.get("timings_accepted"):
            raise ValueError("refusing to use incomplete or timing-rejected repeat data")
        repeated_medians = {
            row["method"]: float(row["median_seconds"])
            for row in repeats.get("rows", [])
            if row.get("median_seconds") is not None
        }

    colors = {
        ("dopri5", "scaled"): "#2563eb",
        ("dopri5", "fixed"): "#78a6e3",
        ("tsit5", "scaled"): "#d97706",
        ("tsit5", "fixed"): "#f1ad62",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    axes[0].set_title("Full density matrix")
    axes[0].set_ylabel("Maximum absolute element error")
    axes[1].set_title("Populations (all saved times)")
    axes[1].set_ylabel("Maximum error (absolute or relative)")
    axes[2].set_title("Photon yield")
    axes[2].set_ylabel("Absolute yield error")

    reference_photons = float(artifact["reference_photon_yield"])
    photon_limit = 1e-6 + 1e-5 * abs(reference_photons)
    all_rows: list[dict] = []
    for row in artifact.get("rows", []):
        runs = row.get("runs") or []
        run = runs[0] if runs else None
        if run is None:
            continue
        accuracy = run.get("accuracy") or {}
        all_saved = accuracy.get("all_saved") or {}
        endpoint = accuracy.get("endpoint") or {}
        wall = _positive(
            repeated_medians.get(row.get("method"), run.get("outer_propagation_wall_seconds"))
        )
        if wall is None:
            continue
        solver = row.get("solver", "")
        policy = "fixed" if "fixed_atol" in row.get("method", "") else "scaled"
        passed = bool(row.get("passes"))
        color = colors.get((solver, policy), "#555555")
        face = color if passed else "none"
        edge = color
        exponent = round(-np.log10(float(row["reltol"])))
        all_rows.append(row)

        points = (
            (axes[0], all_saved.get("density_max_abs"), "o"),
            (axes[0], endpoint.get("density_max_abs"), "s"),
            (axes[1], all_saved.get("population_max_abs"), "o"),
            (axes[1], all_saved.get("population_max_relative_above_1e-6"), "D"),
            (axes[2], all_saved.get("photon_absolute_error"), "o"),
        )
        for axis, raw_y, marker in points:
            y = _positive(raw_y)
            if y is None:
                continue
            axis.scatter(
                wall,
                y,
                marker=marker,
                s=48,
                facecolors=face,
                edgecolors=edge,
                linewidths=1.2,
                zorder=3,
            )
            if axis is axes[0] and marker == "o":
                axis.annotate(
                    f"{exponent}",
                    (wall, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=edge,
                )

    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Propagation wall time (s; full density + separate photon solve)")
        axis.grid(True, which="both", alpha=0.22)

    axes[0].axhline(1e-7, color="#555555", linestyle="--", linewidth=1)
    axes[0].text(0.02, 0.04, "density abs target 1e-7", transform=axes[0].transAxes, fontsize=8)
    axes[1].axhline(1e-7, color="#555555", linestyle="--", linewidth=1)
    axes[1].axhline(1e-3, color="#777777", linestyle=":", linewidth=1)
    axes[1].text(0.02, 0.04, "abs target 1e-7; rel target 1e-3", transform=axes[1].transAxes, fontsize=8)
    axes[2].axhline(photon_limit, color="#555555", linestyle="--", linewidth=1)
    axes[2].text(0.02, 0.04, f"photon target {photon_limit:.2g}", transform=axes[2].transAxes, fontsize=8)

    # Compact legends keep solver, tolerance-policy, and output-series semantics clear.
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[("dopri5", "scaled")], label="DOPRI5 scaled"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[("dopri5", "fixed")], label="DOPRI5 fixed atol"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[("tsit5", "scaled")], label="Tsit5 scaled"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[("tsit5", "fixed")], label="Tsit5 fixed atol"),
        Line2D([0], [0], marker="o", color="#444444", markerfacecolor="#444444", label="absolute error (all saved for rho/pop)"),
        Line2D([0], [0], marker="s", color="#444444", markerfacecolor="#444444", label="endpoint density"),
        Line2D([0], [0], marker="D", color="#444444", markerfacecolor="#444444", label="relative population error"),
        Line2D([0], [0], marker="o", color="#444444", markerfacecolor="none", label="fails any joint gate"),
        Line2D([0], [0], marker="o", color="#444444", markerfacecolor="#444444", label="passes all joint gates"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.suptitle(
        "Nominal 10 µs tolerance sweep - point labels are -log10(rtol); density panel distinguishes all saved times and endpoint",
        y=1.25,
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    repeat_note = f" and {repeats_path}" if repeats_path is not None else ""
    print(f"wrote {output_path} from {json_path}{repeat_note} ({len(all_rows)} finite timed rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=Path)
    args = parser.parse_args()
    plot_sweep(args.json, args.output, args.repeats)


if __name__ == "__main__":
    main()
