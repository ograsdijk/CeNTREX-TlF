"""Stage-4 cross-system regime/reuse catalog and pilot-based budget estimator.

This is a plan-only module: it never builds fixtures or propagates trajectories.
Its output is a row catalog for a later, explicitly authorized stage-4 runner.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from centrex_tlf import hamiltonian

HERE = Path(__file__).resolve().parent
GAMMA = float(hamiltonian.Γ)
FIXTURES = ("r2_f4", "r3_f1_7o2_f4", "r3_f1_9o2_f4", "rc")
T_SPAN_S = {
    "r2_f4": (0.0, 1.0e-6),
    "r3_f1_7o2_f4": (0.0, 1.0e-6),
    "r3_f1_9o2_f4": (0.0, 1.0e-6),
    "rc": (125e-9, 1.125e-6),
}
KERNEL_PREFIX = "experimental_structured_sparse_dense"
KERNELS = (
    "structured", "expanded_sparse", "structured_upper", KERNEL_PREFIX,
    f"{KERNEL_PREFIX}_diag", f"{KERNEL_PREFIX}_soa", f"{KERNEL_PREFIX}_single_unpack",
    f"{KERNEL_PREFIX}_diag_soa", f"{KERNEL_PREFIX}_diag_single_unpack",
    f"{KERNEL_PREFIX}_soa_single_unpack", f"{KERNEL_PREFIX}_diag_soa_single_unpack",
)
PRODUCTS = (
    "full", "photon_integral", "populations", "selected_1", "selected_4",
    "group_sum", "full_and_integral",
)


@dataclass(frozen=True)
class Regime:
    fixture: str
    name: str
    axis: str
    detuning_mhz: float
    optical_rabi_over_gamma: float
    microwave_rabi_over_gamma: float = 1.0

    @property
    def detuning_rad_s(self) -> float:
        return 2.0 * np.pi * self.detuning_mhz * 1e6

    @property
    def optical_rabi_rad_s(self) -> float:
        return self.optical_rabi_over_gamma * GAMMA

    @property
    def microwave_rabi_rad_s(self) -> float:
        return self.microwave_rabi_over_gamma * GAMMA


def regimes_for_fixture(fixture: str) -> list[Regime]:
    if fixture not in FIXTURES:
        raise ValueError(f"unknown fixture {fixture!r}")
    rows: list[Regime] = []
    # One-axis optical detuning and optical power screens.
    for detuning in (-100.0, -20.0, 20.0, 100.0):
        rows.append(Regime(fixture, f"detuning_{detuning:+g}MHz", "detuning",
                           detuning, 1.0))
    for power in (0.3, 3.0):
        rows.append(Regime(fixture, f"optical_{power:g}Gamma", "optical_power",
                           0.0, power))
    if fixture == "rc":
        # Microwave-power axis is separate from optical power.
        for power in (0.3, 3.0):
            rows.append(Regime(fixture, f"microwave_{power:g}Gamma", "microwave_power",
                               0.0, 1.0, power))
        # Focused interaction corners: far detuning x optical extremes, and
        # optical x microwave extremes at resonance. No full Cartesian grid.
        for detuning in (-100.0, 100.0):
            for optical in (0.3, 3.0):
                rows.append(Regime(fixture, f"corner_det{detuning:+g}_opt{optical:g}Gamma",
                                   "detuning_x_optical", detuning, optical, 1.0))
        for optical in (0.3, 3.0):
            for microwave in (0.3, 3.0):
                rows.append(Regime(fixture, f"corner_opt{optical:g}_mw{microwave:g}Gamma",
                                   "optical_x_microwave", 0.0, optical, microwave))
    else:
        # Focused interaction corners for the optical systems.
        for detuning in (-100.0, 100.0):
            for power in (0.3, 3.0):
                rows.append(Regime(fixture, f"corner_det{detuning:+g}_opt{power:g}Gamma",
                                   "detuning_x_optical", detuning, power))
    return rows


def build_parameter_catalog() -> list[dict[str, Any]]:
    """11 kernels x 7 products at K=1 for each distinct nonnominal regime."""
    rows = []
    for fixture in FIXTURES:
        for regime in regimes_for_fixture(fixture):
            for kernel in KERNELS:
                for product in PRODUCTS:
                    overrides = (
                        {"Ωt0": regime.optical_rabi_rad_s,
                         "δ0": regime.detuning_rad_s,
                         "Ω1": regime.microwave_rabi_rad_s,
                         "Ω2": regime.microwave_rabi_rad_s}
                        if fixture == "rc" else
                        {"rabi": regime.optical_rabi_rad_s,
                         "detuning": regime.detuning_rad_s}
                    )
                    rows.append({
                        "stage": "parameter_regime", "fixture": fixture,
                        "regime": regime.name, "regime_axis": regime.axis,
                        "kernel": kernel, "product": product,
                        "initial_policy": "main_state", "initial_count": 1,
                        "detuning_mhz": regime.detuning_mhz,
                        "detuning_rad_s": regime.detuning_rad_s,
                        "optical_rabi_over_gamma": regime.optical_rabi_over_gamma,
                        "optical_rabi_rad_s": regime.optical_rabi_rad_s,
                        "microwave_rabi_over_gamma": regime.microwave_rabi_over_gamma,
                        "microwave_rabi_rad_s": regime.microwave_rabi_rad_s,
                        "parameter_overrides": overrides,
                        "t_span_s": list(T_SPAN_S[fixture]),
                        "solver": "dopri5", "rtol": 1e-7, "atol": 1e-9,
                        "dt_s": 2e-9,
                        "status": "planned",
                    })
    return rows


def build_reuse_catalog() -> list[dict[str, Any]]:
    """Nominal K>1 follow-up rows; actual counts come from resolved QN labels."""
    policies = {
        "r2_f4": (("selected_four", 4), ("all_resolved_X_J2", None)),
        "r3_f1_7o2_f4": (("selected_four", 4), ("all_resolved_X_J3", None)),
        "r3_f1_9o2_f4": (("selected_four", 4), ("all_resolved_X_J3", None)),
        "rc": (("all_resolved_X_J2", 20), ("all_model_states", 70)),
    }
    return [{
        "stage": "initial_batch_reuse", "fixture": fixture,
        "regime": "nominal", "kernel": kernel, "product": product,
        "initial_policy": policy, "initial_count": count,
        "initial_count_source": "actual QN resolution required at execution" if count is None else "declared model subset",
        "status": "planned_unestimated_until_batch_pilot",
    }
        for fixture, fixture_policies in policies.items()
        for policy, count in fixture_policies
        for kernel in KERNELS
        for product in PRODUCTS
    ]


def estimate_from_nominal_pilots(
    catalog: list[dict[str, Any]], pilot_root: Path,
) -> dict[str, Any]:
    """Estimate integration workload from completed nominal screen rows.

    The estimate is a pilot-proxy, not a guaranteed upper bound. It includes
    one fixture build and reference cost per regime when those are recorded in
    the nominal artifact. K>1 reuse rows remain unestimated until a real batch
    pilot exists; no linear-in-K extrapolation is presented as measured.
    """
    parameter_rows = [row for row in catalog if row["stage"] == "parameter_regime"]
    per_fixture = {}
    missing = []
    for fixture in FIXTURES:
        path = pilot_root / fixture / "screen.json"
        if not path.exists():
            missing.append(fixture)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        times = {(row.get("kernel"), row.get("product")): row.get("wall_seconds")
                 for row in payload.get("rows", []) if row.get("status") == "passed"
                 and isinstance(row.get("wall_seconds"), int | float)}
        fixture_regimes = {row["regime"] for row in parameter_rows if row["fixture"] == fixture}
        projected_s = 0.0
        timed_cells = 0
        missing_cells = 0
        for _regime in fixture_regimes:
            for kernel in KERNELS:
                for product in PRODUCTS:
                    value = times.get((kernel, product))
                    if value is None:
                        missing_cells += 1
                    else:
                        projected_s += float(value)
                        timed_cells += 1
        build_s = float(payload.get("fixture_build_seconds", 0.0))
        reference = payload.get("reference_checks", {}).get("reference_wall_seconds", {})
        reference_s = float(sum(reference.values())) if isinstance(reference, dict) else 0.0
        projected_s += len(fixture_regimes) * (build_s + reference_s)
        per_fixture[fixture] = {
            "regime_count": len(fixture_regimes),
            "planned_integration_cells": len(fixture_regimes) * len(KERNELS) * len(PRODUCTS),
            "pilot_timed_cells_used": timed_cells,
            "pilot_missing_cells": missing_cells,
            "pilot_proxy_wall_seconds_including_build_reference": projected_s,
            "pilot_kernel_product_wall_median_seconds": float(np.median(list(times.values()))) if times else None,
        }
    return {
        "estimate_kind": "nominal pilot proxy; not a bound and not parameter-adaptive",
        "missing_nominal_fixture_pilots": missing,
        "parameter_regime_count": len({(r["fixture"], r["regime"]) for r in parameter_rows}),
        "parameter_integration_cells": len(parameter_rows),
        "reuse_cells": sum(r["stage"] == "initial_batch_reuse" for r in catalog),
        "reuse_time_estimate": None,
        "per_fixture": per_fixture,
        "total_parameter_proxy_wall_seconds": (
            float(sum(row["pilot_proxy_wall_seconds_including_build_reference"]
                      for row in per_fixture.values())) if not missing else None
        ),
    }


def write_plan(path: Path, *, pilot_root: Path | None = None) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing stage-4 plan: {path}")
    catalog = build_parameter_catalog() + build_reuse_catalog()
    payload = {
        "schema_version": 1,
        "plan_only": True,
        "fixtures": list(FIXTURES),
        "kernels": list(KERNELS), "products": list(PRODUCTS),
        "parameter_regime_count": len({(r["fixture"], r["regime"])
                                       for r in catalog if r["stage"] == "parameter_regime"}),
        "parameter_screen_cell_count": sum(r["stage"] == "parameter_regime" for r in catalog),
        "reuse_screen_cell_count": sum(r["stage"] == "initial_batch_reuse" for r in catalog),
        "rows": catalog,
    }
    if pilot_root is not None:
        payload["pilot_budget"] = estimate_from_nominal_pilots(catalog, pilot_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path)
    args = parser.parse_args()
    print(write_plan(args.write_plan, pilot_root=args.pilot_root))


if __name__ == "__main__":
    main()
