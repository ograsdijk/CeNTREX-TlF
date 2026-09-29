"""Summarize raw shared-step benchmark CSVs and draw scaling figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "shared_step_batch_results"


def main() -> None:
    summary: dict = {}
    rhs_path = RESULTS / "rhs_microbench.csv"
    if rhs_path.exists():
        rhs = pd.read_csv(rhs_path)
        grouped = rhs.groupby(["system", "kind", "parameter_count", "initial_count"], as_index=False).agg(
            speedup=("speedup", "median"),
            independent_seconds=("independent_seconds", "median"),
            batched_seconds=("batched_seconds", "median"),
            max_abs_rhs_diff=("max_abs_rhs_diff", "max"),
        )
        grouped.to_csv(RESULTS / "rhs_summary.csv", index=False)
        summary["rhs"] = grouped.to_dict(orient="records")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for system in grouped.system.unique():
            selected = grouped[(grouped.system == system) & (grouped.kind == "rabi_narrow") & (grouped.initial_count == 1)]
            axes[0].plot(selected.parameter_count, selected.speedup, marker="o", label=system)
            selected = grouped[(grouped.system == system) & (grouped.kind == "rabi_narrow") & (grouped.parameter_count == 16)]
            axes[1].plot(selected.initial_count, selected.speedup, marker="o", label=system)
        for axis in axes:
            axis.axhline(1., color="black", linewidth=.8)
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
        axes[0].set(xlabel="parameter points", ylabel="independent / batched RHS time", xscale="log", title="Parameter scaling")
        axes[1].set(xlabel="initial projectors", ylabel="independent / batched RHS time", title="Population-basis scaling")
        fig.tight_layout()
        fig.savefig(RESULTS / "rhs_speedup.png", dpi=180)
        plt.close(fig)

    timing_path = RESULTS / "solver_timings.csv"
    if timing_path.exists():
        times = pd.read_csv(timing_path)
        grouped = times.groupby(["system", "kind", "parameter_count", "initial_count", "method"], as_index=False).agg(
            median_seconds=("seconds", "median"), min_seconds=("seconds", "min"),
            std_seconds=("seconds", "std"),
            median_steps=("accepted_steps", "median"),
            median_rejected=("rejected_steps", "median"),
            median_rhs_calls=("rhs_calls", "median"),
        )
        grouped["std_seconds"] = grouped.std_seconds.fillna(0.)
        pivot = grouped.pivot(index=["system", "kind", "parameter_count", "initial_count"],
                              columns="method", values="median_seconds").reset_index()
        if {"independent_serial", "independent_rayon", "shared_serial"}.issubset(pivot):
            pivot["shared_speedup_vs_serial"] = pivot.independent_serial / pivot.shared_serial
            pivot["shared_speedup_vs_rayon"] = pivot.independent_rayon / pivot.shared_serial
        if "independent_capped_serial" in pivot:
            pivot["shared_speedup_vs_capped_serial"] = pivot.independent_capped_serial / pivot.shared_serial
        if "independent_capped_parallel" in pivot:
            pivot["shared_speedup_vs_capped_parallel"] = pivot.independent_capped_parallel / pivot.shared_serial
        pivot.to_csv(RESULTS / "solver_summary.csv", index=False)
        summary["solver"] = pivot.replace({np.nan: None}).to_dict(orient="records")
        for system in pivot.system.unique():
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            subset = pivot[(pivot.system == system) & (pivot.initial_count == 1)]
            for kind in ("rabi_narrow", "rabi_broad", "detuning", "velocity_dynamic", "velocity_straggler_dynamic"):
                part = subset[subset.kind == kind]
                if part.empty:
                    continue
                serial = (part.shared_speedup_vs_capped_serial if kind.endswith("dynamic")
                          else part.shared_speedup_vs_serial)
                parallel = (part.shared_speedup_vs_capped_parallel if kind.endswith("dynamic")
                            else part.shared_speedup_vs_rayon)
                axes[0].plot(part.parameter_count, serial, marker="o", label=kind)
                axes[1].plot(part.parameter_count, parallel, marker="o", label=kind)
                axes[2].plot(part.parameter_count, part.parameter_count / part.shared_serial, marker="o", label=kind)
            for axis in axes:
                axis.grid(alpha=.3)
                axis.set_xscale("log")
                axis.set_xlabel("parameter points")
            for axis in axes[:2]:
                axis.axhline(1., color="black", linewidth=.8)
            axes[0].set(title="Shared vs serial", ylabel="speedup")
            axes[1].set(title="Shared vs parallel", ylabel="speedup")
            axes[2].set(title="Shared throughput", ylabel="trajectories/s")
            axes[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(RESULTS / f"solver_scaling_{system}.png", dpi=180)
            plt.close(fig)
        step_pivot = grouped.pivot(index=["system", "kind", "parameter_count", "initial_count"],
                                   columns="method", values="median_steps").reset_index()
        step_pivot.to_csv(RESULTS / "step_summary.csv", index=False)
        summary["steps"] = step_pivot.replace({np.nan: None}).to_dict(orient="records")
    for name in ("accuracy", "accuracy_full", "straggler_stats", "rhs_layout_microbench", "block_diag_microbench", "coefficient_profile"):
        path = RESULTS / f"{name}.csv"
        if path.exists():
            data = pd.read_csv(path)
            summary[name] = dict(rows=len(data))
            if "max_abs" in data:
                summary[name]["max_abs"] = float(data.max_abs.max())
            if "max_abs_rhs_diff" in data:
                summary[name]["max_abs_rhs_diff"] = float(data.max_abs_rhs_diff.max())
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
