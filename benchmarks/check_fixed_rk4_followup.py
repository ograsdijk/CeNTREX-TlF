"""Prepare fixed RK4 validation at intermediate full-density checkpoints.

Runs one continuous, 250 ns-segmented trajectory per solver and records density
at 2.5, 5, 7.5, and 10 us after t0=125 ns. Cases are nominal RC, RC at +0.5 MHz
optical detuning, and nominal R(2). Each OBE model is prepared once per process.
This driver is prepared for a later timing slot and has not been run.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import bench_rust_specialized_rhs as base
import numpy as np
from check_current_cpu_solvers import (
    ATOL,
    DURATION,
    METHODS,
    build_generic,
    density_accuracy,
)
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rust_specialized_rhs_results"
CHECKPOINTS = [2.5e-6, 5e-6, 7.5e-6, 10e-6]  # durations measured from t0=125 ns
FIXED_DTS = {"fixed_rk4_0.5ns": 0.5e-9, "fixed_rk4_0.4ns": 0.4e-9}
REPEATS = 12
REFERENCE_ATOL, REFERENCE_RTOL = 1e-13, 1e-11
BASELINE_ATOL, BASELINE_RTOL = ATOL, 1e-9

ORDERS = [
    ["tsit5", "fixed_rk4_0.5ns", "fixed_rk4_0.4ns"],
    ["tsit5", "fixed_rk4_0.4ns", "fixed_rk4_0.5ns"],
    ["fixed_rk4_0.5ns", "tsit5", "fixed_rk4_0.4ns"],
    ["fixed_rk4_0.5ns", "fixed_rk4_0.4ns", "tsit5"],
    ["fixed_rk4_0.4ns", "tsit5", "fixed_rk4_0.5ns"],
    ["fixed_rk4_0.4ns", "fixed_rk4_0.5ns", "tsit5"],
] * 2


def load_native():
    """Build an isolated extension before loading it (DLL replacement is locked on Windows)."""
    native, source_hash = build_generic(tag="fixed_rk4_followup")
    if not hasattr(native, "trajectory_checkpoints"):
        raise RuntimeError("Generic Rust extension lacks trajectory_checkpoints")
    return native, source_hash


def shifted_detuning_payload(payload):
    """Copy a payload with the optical delta0 slot shifted by +0.5 MHz."""
    shifted = copy.deepcopy(payload)
    graph = shifted["parameter_graph"]
    try:
        slot = graph["slot_names"].index("δ0")
    except ValueError as exc:
        raise RuntimeError(f"Optical detuning slot δ0 missing: {graph['slot_names']}") from exc
    value = graph["base_values"][slot]
    if not isinstance(value, dict) or "re" not in value:
        raise TypeError(f"Unexpected scalar payload for δ0: {value!r}")
    value["re"] = 2 * math.pi * 0.5e6
    value["im"] = 0.0
    return shifted


def checkpoint_errors(prepared, states, reference):
    return {
        f"{checkpoint:g}_s_after_t0": density_accuracy(prepared, states[index], reference[index])
        for index, checkpoint in enumerate(CHECKPOINTS)
    }


def update_summaries(row):
    runs = {candidate["method"]: candidate["runs"] for candidate in row["candidates"]}
    baseline = np.asarray([run["seconds"] for run in runs["tsit5"]])
    rng = np.random.default_rng(4721 + len(row["system"]) + len(row["parameter_point"]))
    indices = rng.integers(0, REPEATS, size=(20000, REPEATS))
    row["paired_summaries"] = {}
    for method in FIXED_DTS:
        candidate = np.asarray([run["seconds"] for run in runs[method]])
        ratios = candidate / baseline
        reduction_samples = 100 * (1 - np.median(ratios[indices], axis=1))
        row["paired_summaries"][method] = dict(
            fixed_over_tsit5_ratio_median=float(np.median(ratios)),
            fixed_runtime_reduction_vs_tsit5_percent=float(100 * (1 - np.median(ratios))),
            fixed_runtime_reduction_bootstrap_95_percent=[
                float(x) for x in np.quantile(reduction_samples, [0.025, 0.975])
            ],
        )


def save_progress(result, output, system, parameter_point, repeat, method):
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (RESULTS / "fixed_rk4_followup_progress.json").write_text(
        json.dumps(dict(system=system, parameter_point=parameter_point,
                        repeat=repeat, method=method), indent=2),
        encoding="utf-8",
    )


def run_scenario(native, system, prepared, rho0, payload, parameter_point, result, output):
    y0 = prepared.layout.pack(rho0)
    reference_run = native.trajectory_checkpoints(
        payload,
        y0,
        METHODS["tsit5"],
        DURATION,
        REFERENCE_ATOL,
        REFERENCE_RTOL,
        1e-10,
        CHECKPOINTS,
    )
    reference = np.asarray(reference_run[0]).reshape(len(CHECKPOINTS), -1)
    row = dict(
        system=system,
        parameter_point=parameter_point,
        checkpoint_durations_from_t0_s=CHECKPOINTS,
        absolute_checkpoint_times_s=[125e-9 + t for t in CHECKPOINTS],
        reference=dict(method="tsit5", atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL,
                       rhs_calls=reference_run[2], accepted_steps=reference_run[3],
                       rejected_steps=reference_run[4]),
        order_by_round=ORDERS,
        candidates=[dict(method=m, dt_s=(None if m == "tsit5" else FIXED_DTS[m]), runs=[])
                    for m in ["tsit5", *FIXED_DTS]],
    )
    result["scenarios"].append(row)
    save_progress(result, output, system, parameter_point, 0, "reference_complete")
    candidate_rows = {candidate["method"]: candidate for candidate in row["candidates"]}

    for repeat, order in enumerate(ORDERS, start=1):
        for method in order:
            if method == "tsit5":
                solver_id, dt = METHODS["tsit5"], 1e-10
            else:
                solver_id, dt = METHODS["fixed_rk4"], FIXED_DTS[method]
            run = native.trajectory_checkpoints(
                payload, y0, solver_id, DURATION, BASELINE_ATOL, BASELINE_RTOL, dt, CHECKPOINTS
            )
            states = np.asarray(run[0]).reshape(len(CHECKPOINTS), -1)
            accuracy = checkpoint_errors(prepared, states, reference)
            candidate_rows[method]["runs"].append(
                dict(seconds=run[1], rhs_calls=run[2], accepted_steps=run[3],
                     rejected_steps=run[4], accuracy_by_checkpoint=accuracy)
            )
            save_progress(result, output, system, parameter_point, repeat, method)

    for candidate in row["candidates"]:
        candidate["median_s"] = float(np.median([run["seconds"] for run in candidate["runs"]]))
        checks = [check for run in candidate["runs"]
                  for check in run["accuracy_by_checkpoint"].values()]
        finite = [check for check in checks if check["finite"]]
        candidate["all_accuracy_passed"] = all(check["passed"] for check in checks)
        candidate["finite"] = all(check["finite"] for check in checks)
        candidate["max_relative_error_fraction"] = max(
            (check["max_relative_above_1e6"] for check in finite), default=None
        )
        candidate["max_small_entry_absolute_error"] = max(
            (check["max_absolute_at_or_below_1e6"] for check in finite), default=None
        )
        candidate["max_trace_error_from_one"] = max(
            (check["trace_error_from_one"] for check in finite), default=None
        )
    update_summaries(row)
    save_progress(result, output, system, parameter_point, REPEATS, "scenario_complete")
    return row


def main():
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "fixed_rk4_followup.json"
    result = dict(
        description="One segmented full trajectory per solver, with save-at snapshots; each molecular model prepared once.",
        t0_s=125e-9,
        duration_s=DURATION,
        checkpoint_durations_from_t0_s=CHECKPOINTS,
        absolute_checkpoint_times_s=[125e-9 + t for t in CHECKPOINTS],
        fixed_dt_s=FIXED_DTS,
        repeats=REPEATS,
        adaptive_candidate_tolerances=dict(atol=BASELINE_ATOL, rtol=BASELINE_RTOL),
        reference=dict(solver="tsit5", atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL),
        accuracy_criterion="1% relative above density magnitude 1e-6; 1e-6 absolute below; trace error <=1e-9 at every checkpoint.",
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            centrex_tlf=importlib.metadata.version("centrex-tlf"),
            machine=platform.platform(),
        ),
        fixture_scope=dict(
            electric_field_V_cm=[0.0, 0.0, base.driver.E_FIELD_V_CM],
            magnetic_field_G=[0.0, 0.0, base.driver.B_FIELD_G],
            optical_transition="P(2), F1'=3/2, F'=1",
            microwave_transitions=["X J=1↔2", "X J=2↔3"],
            optical_and_microwave_rabi_over_gamma=1.0,
            optical_and_microwave_rabi_rad_s=base.driver.GAMMA,
            optical_power_mW=None,
            microwave_power_mW_each=None,
            optical_power_note="This numerical fixture sets Ω=Γ directly; no power-to-Rabi conversion or physical power scan is applied.",
            eom_beta=base.driver.BETA_EOM,
            eom_frequency_Hz=base.driver.F_EOM_HZ,
            microwave_switch_frequency_Hz=base.driver.F_SWITCH_HZ,
            microwave_switch_offset_s=base.driver.SWITCH_OFFSET_S,
            microwave_switch_phase_offset_rad=base.driver.SWITCH_PHASE_OFFSET_RAD,
            RC_initial_state="X J=2, F1=5/2, F=2, mF=0",
            R2_transition="R(2) F1=7/2, F=4",
            non_RC_initial_state="uniform diagonal over all modeled X-state levels",
            segment_duration_s=base.driver.SWITCH_OFFSET_S,
            duration_is_short_numerical_fixture=True,
        ),
        scenarios=[],
    )
    save_progress(result, output, "setup", "setup", 0, "native_build")
    native, source_hash = load_native()
    result["native_wrapper_source_sha256"] = source_hash
    save_progress(result, output, "setup", "setup", 0, "native_build_complete")
    with threadpool_limits(limits=1):
        # Both RC parameter points share one prepared model and initial density.
        prepared, rho0, _ = base.model("rc")
        payload = prepared.to_payload()
        run_scenario(native, "rc", prepared, rho0, payload, "nominal", result, output)
        run_scenario(native, "rc", prepared, rho0, shifted_detuning_payload(payload),
                     "delta0_plus_0.5_MHz", result, output)
        prepared, rho0, _ = base.model("r2")
        run_scenario(native, "r2", prepared, rho0, prepared.to_payload(), "nominal", result, output)


if __name__ == "__main__":
    main()
