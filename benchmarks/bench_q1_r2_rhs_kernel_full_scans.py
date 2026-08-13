from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = HERE / "q1_r2_rhs_kernel_static_dynamic_results"

MODES = {
    "baseline_packed": "experimental_expanded_sparse_baseline_packed",
    "current_split_inputs": "experimental_expanded_sparse_current_split_inputs",
    "static_dynamic": "expanded_sparse",
}


def load_notebook_namespace(path: Path, cell_indices: list[int], *, quiet: bool) -> dict[str, Any]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict[str, Any] = {}
    stream = io.StringIO()
    ctx = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()
    with ctx:
        for idx in cell_indices:
            exec("".join(nb["cells"][idx]["source"]), ns)
    return ns


def retained_system(ns: dict[str, Any], *, qn_compact: bool) -> Any:
    return ns["lindblad"].generate_OBE_system_transitions(
        [ns["transition"]],
        ns["transition_selectors"],
        qn_compact=qn_compact,
        E=ns["E_FIELD"],
        B=ns["B_FIELD"],
        retain_opposite_parity_levels=True,
        method="matrix",
    )


def prepare_q1(quiet: bool, *, qn_compact: bool) -> dict[str, Any]:
    ns = load_notebook_namespace(
        ROOT / "examples" / "lindblad" / "q1_opposite_parity_retention.ipynb",
        [1, 3, 5, 7, 9],
        quiet=quiet,
    )
    system = retained_system(ns, qn_compact=qn_compact)
    params, _ = ns["parameter_values"](system)
    prepared = ns["prepare_lindblad_problem"](
        system,
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    gamma = getattr(ns["hamiltonian"], "\u0393")  # noqa: B009
    weights = [(int(idx), float(gamma)) for idx in ns["index_sets"](system)["excited"]]
    return {
        "name": "q1_retained_compact" if qn_compact else "q1_retained",
        "lindblad": ns["lindblad"],
        "prepared": prepared,
        "rho0": ns["initial_density_matrix"](system),
        "t_end": ns["T_END"],
        "saveat": ns["SCAN_SAVEAT"],
        "detuning_mhz": ns["DETUNING_SCAN_MHZ"],
        "scan_symbol": vars(ns["transition_selectors"][0])["\u03b4"],
        "weights": weights,
        "n_levels": len(system.QN),
        "qn_compact": qn_compact,
    }


def prepare_r2(quiet: bool, *, qn_compact: bool) -> dict[str, Any]:
    ns = load_notebook_namespace(
        ROOT / "examples" / "lindblad" / "r2_opposite_parity_retention.ipynb",
        [1, 3, 5, 8, 10],
        quiet=quiet,
    )
    system = retained_system(ns, qn_compact=qn_compact)
    params, _ = ns["parameter_values"](system)
    prepared = ns["prepare_lindblad_problem"](
        system,
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    weights = [(int(idx), float(ns["GAMMA"])) for idx in ns["index_sets"](system)["excited"]]
    return {
        "name": "r2_retained_compact" if qn_compact else "r2_retained",
        "lindblad": ns["lindblad"],
        "prepared": prepared,
        "rho0": ns["initial_density_matrix"](system),
        "t_end": ns["T_END"],
        "saveat": ns["SCAN_SAVEAT"],
        "detuning_mhz": ns["DETUNING_SCAN_MHZ"],
        "scan_symbol": vars(ns["transition_selectors"][0])["\u03b4"],
        "weights": weights,
        "n_levels": len(system.QN),
        "qn_compact": qn_compact,
    }


def run_scan(model: dict[str, Any], label: str, *, pair: int, order_index: int, threads: int | None):
    mode = MODES[label]
    detuning_mhz = model["detuning_mhz"]
    start = time.perf_counter()
    result = model["lindblad"].grid_scan(
        model["prepared"],
        model["rho0"],
        (0.0, model["t_end"]),
        scan={model["scan_symbol"]: 2 * np.pi * 1e6 * detuning_mhz},
        solver="dopri5",
        execution_mode=mode,
        output="photon_integral",
        output_when="saveat",
        integral_weights=model["weights"],
        saveat=model["saveat"],
        dt=2e-9,
        reltol=1e-7,
        abstol=1e-9,
        parallel=True,
        threads=threads,
        collect_stats=True,
    )
    wall = time.perf_counter() - start
    photons = result.values.reshape(detuning_mhz.size, -1)[:, -1]
    stats = result.solver_stats
    peak_idx = int(np.argmax(photons))
    row = {
        "system": model["name"],
        "pair": pair,
        "order_index": order_index,
        "label": label,
        "mode": mode,
        "wall_seconds": wall,
        "stats_total_seconds": float(stats.get("total_seconds", wall)),
        "rhs_calls": int(stats.get("rhs_calls", 0)),
        "accepted_steps": int(stats.get("accepted_steps", 0)),
        "peak_detuning_MHz": float(detuning_mhz[peak_idx]),
        "peak_photons": float(photons[peak_idx]),
    }
    return row, photons


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(system: str, rows: list[dict[str, Any]], photon_refs: dict[str, np.ndarray]) -> dict[str, Any]:
    by_label = {label: [row for row in rows if row["label"] == label] for label in MODES}
    summary: dict[str, Any] = {"system": system}
    for label, group in by_label.items():
        walls = [float(row["wall_seconds"]) for row in group]
        totals = [float(row["stats_total_seconds"]) for row in group]
        summary[label] = {
            "n": len(group),
            "wall_mean_seconds": statistics.mean(walls) if walls else None,
            "wall_std_seconds": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "stats_total_mean_seconds": statistics.mean(totals) if totals else None,
            "stats_total_std_seconds": statistics.stdev(totals) if len(totals) > 1 else 0.0,
            "rhs_calls_unique": sorted({int(row["rhs_calls"]) for row in group}),
            "accepted_steps_unique": sorted({int(row["accepted_steps"]) for row in group}),
            "peak_detuning_unique": sorted({float(row["peak_detuning_MHz"]) for row in group}),
        }
    for comparison, numerator, denominator in (
        ("baseline_over_static_dynamic", "baseline_packed", "static_dynamic"),
        ("current_over_static_dynamic", "current_split_inputs", "static_dynamic"),
    ):
        left = by_label[numerator]
        right = by_label[denominator]
        count = min(len(left), len(right))
        speedups = [
            float(left[idx]["wall_seconds"]) / float(right[idx]["wall_seconds"])
            for idx in range(count)
        ]
        summary[f"paired_speedup_{comparison}"] = {
            "values": speedups,
            "mean": statistics.mean(speedups) if speedups else None,
            "std": statistics.stdev(speedups) if len(speedups) > 1 else 0.0,
        }
    if set(photon_refs) == set(MODES):
        diff = np.abs(photon_refs["baseline_packed"] - photon_refs["static_dynamic"])
        denom = max(float(np.max(np.abs(photon_refs["baseline_packed"]))), 1e-30)
        summary["validation"] = {
            "max_abs_photon_diff_first_comparison": float(np.max(diff)),
            "max_rel_photon_diff_first_comparison": float(np.max(diff) / denom),
        }
    return summary


def benchmark_model(model: dict[str, Any], *, repeats: int, threads: int | None) -> None:
    out_dir = RESULTS_DIR / model["name"]
    rows_path = out_dir / "full_scan_kernel_repeats.csv"
    summary_path = out_dir / "full_scan_kernel_repeats_summary.json"
    rows: list[dict[str, Any]] = []
    photon_refs: dict[str, np.ndarray] = {}
    labels = list(MODES)
    orders = [labels[offset:] + labels[:offset] for offset in range(len(labels))]
    print(
        f"{model['name']}: levels={model['n_levels']} "
        f"scan_points={model['detuning_mhz'].size} saveat={model['saveat'].size}",
        flush=True,
    )
    for pair in range(1, repeats + 1):
        order = orders[(pair - 1) % len(orders)]
        print(f"{model['name']} pair {pair}/{repeats}: {order}", flush=True)
        for order_index, label in enumerate(order, start=1):
            row, photons = run_scan(model, label, pair=pair, order_index=order_index, threads=threads)
            rows.append(row)
            photon_refs.setdefault(label, photons)
            write_rows(rows_path, rows)
            summary_path.write_text(
                json.dumps(summarize(model["name"], rows, photon_refs), indent=2),
                encoding="utf-8",
            )
            print({k: v for k, v in row.items() if k != "mode"}, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated full Q1/R2 retained-opposite-parity RHS kernel scan benchmarks."
    )
    parser.add_argument("--system", choices=["q1", "r2", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--qn-compact", action="store_true")
    parser.add_argument("--show-setup-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="Low overlap detected")
    quiet = not args.show_setup_output
    systems = []
    if args.system in {"q1", "both"}:
        systems.append(prepare_q1(quiet=quiet, qn_compact=args.qn_compact))
    if args.system in {"r2", "both"}:
        systems.append(prepare_r2(quiet=quiet, qn_compact=args.qn_compact))
    for model in systems:
        benchmark_model(model, repeats=args.repeats, threads=args.threads)


if __name__ == "__main__":
    main()
