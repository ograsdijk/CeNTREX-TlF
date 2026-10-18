"""Plot the nominal RC tolerance sweep's full-density accuracy and cost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")


COLORS = {"dopri5": "#2563eb", "tsit5": "#d97706"}
MARKERS = {"scaled": "o", "fixed": "^"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(row: dict) -> dict:
    runs = row.get("runs") or []
    if not runs:
        raise ValueError(f"missing candidate run for {row.get('method')}")
    return runs[0]


def _policy(row: dict) -> str:
    return "fixed" if "fixed_atol" in row.get("method", "") else "scaled"


def _rho_error(row: dict) -> float:
    value = _run(row)["accuracy"]["all_saved"]["density_max_abs"]
    if not np.isfinite(value):
        raise ValueError(f"nonfinite full-density error for {row.get('method')}")
    return max(float(value), 1e-18)


def _repeat_info(repeats_path: Path, source_json: Path, source_npz: Path) -> dict[str, dict]:
    repeats = json.loads(repeats_path.read_text(encoding="utf-8"))
    if not repeats.get("complete") or not repeats.get("timings_accepted"):
        raise ValueError("repeat artifact is incomplete or its timing guard failed")
    if repeats.get("source_sweep_json_sha256") != _sha256(source_json):
        raise ValueError("repeat artifact does not match the tolerance-sweep JSON")
    if repeats.get("source_sweep_npz_sha256") != _sha256(source_npz):
        raise ValueError("repeat artifact does not match the tolerance-sweep density NPZ")

    repeated = {}
    for row in repeats.get("rows", []):
        walls = np.asarray(row.get("wall_seconds", []), dtype=float)
        if walls.size < 3 or not np.all(np.isfinite(walls)):
            raise ValueError(f"incomplete wall-time repeats for {row.get('method')}")
        if not row.get("all_accuracy_runs_pass"):
            raise ValueError(f"a repeated candidate did not pass all accuracy checks: {row.get('method')}")
        repeated[row["method"]] = {
            "median": float(np.median(walls)),
            "minimum": float(np.min(walls)),
            "maximum": float(np.max(walls)),
            "wall_seconds": walls,
        }
    return repeated


def plot_convergence(sweep_json: Path, repeats_json: Path, output_path: Path) -> None:
    artifact = json.loads(sweep_json.read_text(encoding="utf-8"))
    if not artifact.get("complete") or not artifact.get("timings_accepted"):
        raise ValueError("tolerance sweep is incomplete or its timing guard failed")
    if not artifact.get("reference_validated"):
        raise ValueError("tolerance sweep does not have a validated reference")
    sweep_npz = sweep_json.with_name(sweep_json.stem + "_densities.npz")
    repeated = _repeat_info(repeats_json, sweep_json, sweep_npz)

    # Confirm the repeated full trajectories reproduce the original density
    # errors exactly before combining their measured timing ranges with them.
    with np.load(sweep_npz) as original, np.load(
        repeats_json.with_name(repeats_json.stem + "_densities.npz")
    ) as repeat_arrays:
        reference = original["reference_density"]
        for method in repeated:
            for suffix in ("repeat_1", "repeat_2"):
                candidate = repeat_arrays[f"{method}_{suffix}"]
                if not np.array_equal(candidate, original[method]):
                    raise ValueError(f"repeated density differs from baseline: {method} {suffix}")
                error = float(np.max(np.abs(candidate - reference)))
                if error != _rho_error(next(row for row in artifact["rows"] if row["method"] == method)):
                    raise ValueError(f"repeated density error changed unexpectedly: {method} {suffix}")

    rows = artifact["rows"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    error_ax, timing_ax, pareto_ax = axes

    # Panel 1: error against requested relative tolerance. The lines connect
    # only the explicitly measured scaled-atol sweep points; fixed-atol
    # controls remain unconnected because their error-control policy differs.
    for solver in ("dopri5", "tsit5"):
        scaled = sorted(
            (row for row in rows if row.get("solver") == solver and _policy(row) == "scaled"),
            key=lambda row: float(row["reltol"]),
            reverse=True,
        )
        xs = [float(row["reltol"]) for row in scaled]
        ys = [_rho_error(row) for row in scaled]
        filled = [bool(row.get("passes")) for row in scaled]
        color = COLORS[solver]
        error_ax.plot(xs, ys, color=color, alpha=0.75, linewidth=1.3, zorder=1)
        for _row, x, y, pass_all in zip(scaled, xs, ys, filled, strict=True):
            error_ax.scatter(
                x,
                y,
                marker=MARKERS["scaled"],
                s=55,
                facecolors=color if pass_all else "none",
                edgecolors=color,
                linewidths=1.3,
                zorder=3,
            )

        controls = [row for row in rows if row.get("solver") == solver and _policy(row) == "fixed"]
        for row in controls:
            x, y = float(row["reltol"]), _rho_error(row)
            error_ax.scatter(
                x,
                y,
                marker=MARKERS["fixed"],
                s=70,
                facecolors=color if row.get("passes") else "none",
                edgecolors=color,
                linewidths=1.4,
                zorder=4,
            )

    error_ax.set_xscale("log")
    error_ax.set_yscale("log")
    error_ax.set_xlim(2e-2, 5e-10)  # loose tolerances on the left, tight on the right, with padding
    error_ax.axhline(1e-7, color="#444444", linestyle="--", linewidth=1.2)
    error_ax.text(0.03, 0.08, "full-density target 1e-7", transform=error_ax.transAxes, fontsize=8)
    checks = artifact["reference_check"]["checks"]
    refine_difference = checks["primary_to_refined"]["density_max_abs"]
    error_ax.text(
        0.03,
        0.025,
        f"tight-reference refinement difference: {refine_difference:.2e} (agreement scale, not an error bound)",
        transform=error_ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    error_ax.set_title("Full-density tolerance sweep")
    error_ax.set_xlabel("Requested rtol (loose  →  tight)")
    error_ax.set_ylabel("Max elementwise density error over saved times")

    # Panel 2: measured propagation wall versus rtol. Error bars show the
    # baseline plus two independent repeat times for the selected settings.
    for solver in ("dopri5", "tsit5"):
        scaled = sorted(
            (row for row in rows if row.get("solver") == solver and _policy(row) == "scaled"),
            key=lambda row: float(row["reltol"]),
            reverse=True,
        )
        xs, ys = [], []
        for row in scaled:
            method = row["method"]
            base_wall = float(_run(row)["outer_propagation_wall_seconds"])
            median = repeated.get(method, {}).get("median", base_wall)
            minimum = repeated.get(method, {}).get("minimum", base_wall)
            maximum = repeated.get(method, {}).get("maximum", base_wall)
            xs.append(float(row["reltol"]))
            ys.append(median)
            timing_ax.errorbar(
                float(row["reltol"]),
                median,
                yerr=[[median - minimum], [maximum - median]],
                marker=MARKERS["scaled"],
                color=COLORS[solver],
                markerfacecolor=COLORS[solver] if row.get("passes") else "none",
                markeredgecolor=COLORS[solver],
                capsize=2,
                linewidth=1,
                markersize=5,
                zorder=3,
            )
        timing_ax.plot(xs, ys, color=COLORS[solver], alpha=0.7, linewidth=1.1)
        controls = [row for row in rows if row.get("solver") == solver and _policy(row) == "fixed"]
        for row in controls:
            method = row["method"]
            base_wall = float(_run(row)["outer_propagation_wall_seconds"])
            median = repeated.get(method, {}).get("median", base_wall)
            minimum = repeated.get(method, {}).get("minimum", base_wall)
            maximum = repeated.get(method, {}).get("maximum", base_wall)
            timing_ax.errorbar(
                float(row["reltol"]),
                median,
                yerr=[[median - minimum], [maximum - median]],
                marker=MARKERS["fixed"],
                color=COLORS[solver],
                markerfacecolor=COLORS[solver] if row.get("passes") else "none",
                markeredgecolor=COLORS[solver],
                capsize=2,
                linewidth=1,
                markersize=6,
                zorder=4,
            )
    timing_ax.set_xscale("log")
    timing_ax.set_xlim(2e-2, 5e-10)
    timing_ax.set_ylim(bottom=0)
    timing_ax.set_title("Propagation wall time")
    timing_ax.set_xlabel("Requested rtol (loose  →  tight)")
    timing_ax.set_ylabel("Wall time (s; full density + separate photon solve)")
    timing_ax.text(
        0.03,
        0.94,
        "Triangle controls: atol=1e-10 at rtol=1e-7 and 1e-9",
        transform=timing_ax.transAxes,
        fontsize=8,
        va="top",
    )
    timing_ax.text(
        0.03,
        0.04,
        "Error bars: min-max of three trials where repeated; otherwise one measurement.",
        transform=timing_ax.transAxes,
        fontsize=7,
    )

    # Panel 3: full-density Pareto view, with linear time to make the measured
    # 3–12 s range readable and log error for the accepted/failing spread.
    for row in rows:
        method = row["method"]
        solver = row["solver"]
        policy = _policy(row)
        base_wall = float(_run(row)["outer_propagation_wall_seconds"])
        wall = repeated.get(method, {}).get("median", base_wall)
        error = _rho_error(row)
        color = COLORS[solver]
        pareto_ax.scatter(
            wall,
            error,
            marker=MARKERS[policy],
            s=62,
            facecolors=color if row.get("passes") else "none",
            edgecolors=color,
            linewidths=1.35,
            zorder=3,
        )
    pareto_ax.axhline(1e-7, color="#444444", linestyle="--", linewidth=1.2)
    pareto_ax.set_xlim(2.8, 12.0)
    pareto_ax.set_yscale("log")
    pareto_ax.set_title("Full-density error versus runtime")
    pareto_ax.set_xlabel("Propagation wall time (s; full density + separate photon solve)")
    pareto_ax.set_ylabel("Max elementwise density error over saved times")
    pareto_ax.annotate(
        "Tsit5 1e-7/1e-9\ndensity fails",
        xy=(4.6556554, 7.11904095e-7),
        xytext=(3.05, 1.8e-6),
        arrowprops={"arrowstyle": "->", "color": COLORS["tsit5"]},
        fontsize=8,
        color=COLORS["tsit5"],
    )
    pareto_ax.annotate(
        "Tsit5 1e-7/1e-10\nall joint gates pass",
        xy=(6.6120955, 5.37833569e-8),
        xytext=(8.25, 1.7e-8),
        arrowprops={"arrowstyle": "->", "color": COLORS["tsit5"]},
        fontsize=8,
        color=COLORS["tsit5"],
    )

    for axis in axes:
        axis.grid(True, which="both", alpha=0.22)

    legend = [
        Line2D([0], [0], color=COLORS["dopri5"], marker="o", label="DOPRI5"),
        Line2D([0], [0], color=COLORS["tsit5"], marker="o", label="Tsit5"),
        Line2D([0], [0], color="#444444", marker="o", label="scaled atol=rtol/100"),
        Line2D([0], [0], color="#444444", marker="^", label="fixed atol=1e-10 controls"),
        Line2D([0], [0], color="#444444", marker="o", markerfacecolor="none", label="fails at least one joint gate"),
        Line2D([0], [0], color="#444444", marker="o", markerfacecolor="#444444", label="passes all joint gates"),
    ]
    figure.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    figure.suptitle(
        "Nominal 10 µs full-density tolerance convergence and cost (paired rtol/atol sweep; not a timestep-order fit)",
        y=1.18,
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output_path}; verified repeat density arrays against the original sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_json", type=Path)
    parser.add_argument("repeats_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_convergence(args.sweep_json, args.repeats_json, args.output)


if __name__ == "__main__":
    main()
