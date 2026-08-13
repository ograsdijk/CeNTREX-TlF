"""Benchmark: does the per-level co-rotating frame prototype pay off?

Builds on `benchmarks/diagnose_step_size.py` (read that first, and
`benchmarks/step_size_diagnostics_results/step_size_diagnostics_report.md`
section 4(a)). That diagnostic found the r2-in-static-E-field system's dopri5
step size is *oscillation-limited* at dt ~= 4.52 ns by the driven B J=3
manifold's 73.6 MHz static diagonal spread -- accepted steps are flat across
four orders of magnitude in reltol.

`centrex_tlf.lindblad.generate_hamiltonian.apply_per_level_rotating_frame`
analytically removes the *numeric* static part of the Hamiltonian diagonal
via a per-level unitary `T = diag(exp(-i*E_i*t))`, at the cost of making the
off-diagonal coupling coefficients explicitly time-dependent (oscillating at
their own detunings, <= 50.4 MHz). This script measures whether that
trade-off is worth it:

1. Accepted/rejected steps, RHS calls, mean dt, and wall time -- original vs
   rotated frame -- at detunings {0, 25} MHz and reltol in {1e-5, 1e-7, 1e-9}.
   Also checks whether the rotated frame's step count now *scales* with
   reltol (accuracy-limited signature ~ reltol^(-1/5)) instead of being flat.
2. A photon-integral detuning scan (-5..30 MHz, 1 MHz steps) in both frames,
   checking the curves agree (frame-invariant observable) and comparing scan
   wall time.

Writes `benchmarks/step_size_diagnostics_results/per_level_frame_bench.csv`
and prints a summary used to fill in
`step_size_diagnostics_report.md` section 6.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np

from centrex_tlf import hamiltonian, states
from centrex_tlf.lindblad.generate_hamiltonian import apply_per_level_rotating_frame
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad

import diagnose_step_size as diag

RESULTS_DIR = Path(__file__).parent / "step_size_diagnostics_results"

GAMMA = getattr(hamiltonian, "Γ")

RELTOLS = [1e-5, 1e-7, 1e-9]
ABSTOL = 1e-9
DETUNINGS_MHZ = [0.0, 25.0]
WALL_REPEATS_RELTOL = 1e-7
WALL_REPEATS = 3

SCAN_DETUNINGS_MHZ = np.arange(-5.0, 30.0 + 0.5, 1.0)


def mhz(x: float) -> float:
    return float(x) / (2 * np.pi * 1e6)


def excited_indices(system) -> list[int]:
    return [
        idx
        for idx, state in enumerate(system.QN)
        if state.largest.electronic_state == states.ElectronicState.B
    ]


def run_step_table(system, ts, rotated, rabi_value: float, rows: list[dict]) -> None:
    print("\n== Step-count / wall-time table (original vs rotated) ==")
    for detuning_mhz in DETUNINGS_MHZ:
        detuning_rad = 2 * np.pi * detuning_mhz * 1e6
        params = diag.make_parameters(system, ts, rabi_value, detuning_rad)
        prepared = prepare_lindblad_problem(
            system, params, backend="rust", hamiltonian_representation="decomposed"
        )
        prepared_rot = prepare_lindblad_problem(
            rotated, params, backend="rust", hamiltonian_representation="decomposed"
        )
        rho0 = diag.build_rho0(system)

        for frame_name, prepared_problem in [("original", prepared), ("rotated", prepared_rot)]:
            for reltol in RELTOLS:
                n_repeats = WALL_REPEATS if reltol == WALL_REPEATS_RELTOL else 1
                wall_times = []
                stats = {}
                for _ in range(n_repeats):
                    start = time.perf_counter()
                    result = solve_lindblad(
                        prepared_problem,
                        rho0,
                        (0.0, diag.T_END),
                        solver="dopri5",
                        execution_mode="expanded_sparse",
                        output="populations",
                        output_when="final",
                        dense_output=False,
                        abstol=ABSTOL,
                        reltol=reltol,
                        dt=1e-10,
                        collect_stats=True,
                    )
                    wall_times.append(time.perf_counter() - start)
                    stats = result.solver_stats or {}
                wall_median = statistics.median(wall_times)
                accepted = int(stats.get("accepted_steps", 0))
                rejected = int(stats.get("rejected_steps", 0))
                rhs_calls = int(stats.get("rhs_calls", 0))
                mean_dt = diag.T_END / accepted if accepted else float("nan")
                print(
                    f"  {frame_name:8s} detuning={detuning_mhz:5.1f} MHz reltol={reltol:.0e}: "
                    f"accepted={accepted:7d} rejected={rejected:6d} rhs={rhs_calls:8d} "
                    f"mean_dt={mean_dt * 1e9:7.2f} ns wall_median={wall_median:6.3f} s "
                    f"(n={n_repeats})"
                )
                rows.append(
                    {
                        "frame": frame_name,
                        "detuning_MHz": detuning_mhz,
                        "reltol": reltol,
                        "accepted_steps": accepted,
                        "rejected_steps": rejected,
                        "rhs_calls": rhs_calls,
                        "mean_dt_ns": mean_dt * 1e9,
                        "wall_seconds_median": wall_median,
                        "wall_repeats": n_repeats,
                    }
                )


def run_photon_integral_scan(system, ts, rotated, rabi_value: float) -> dict:
    print("\n== Photon-integral detuning scan (-5..30 MHz, 1 MHz steps) ==")
    rho0 = diag.build_rho0(system)
    weights_orig = [(int(idx), float(GAMMA)) for idx in excited_indices(system)]
    weights_rot = [(int(idx), float(GAMMA)) for idx in excited_indices(rotated)]

    scan_results: dict[str, np.ndarray] = {}
    scan_wall: dict[str, float] = {}
    for frame_name, sys_obj, weights in [
        ("original", system, weights_orig),
        ("rotated", rotated, weights_rot),
    ]:
        photons = np.zeros(SCAN_DETUNINGS_MHZ.size)
        start = time.perf_counter()
        for i, detuning_mhz in enumerate(SCAN_DETUNINGS_MHZ):
            detuning_rad = 2 * np.pi * float(detuning_mhz) * 1e6
            params = diag.make_parameters(sys_obj, ts, rabi_value, detuning_rad)
            prepared = prepare_lindblad_problem(
                sys_obj, params, backend="rust", hamiltonian_representation="decomposed"
            )
            result = solve_lindblad(
                prepared,
                rho0,
                (0.0, diag.T_END),
                solver="dopri5",
                execution_mode="expanded_sparse",
                output="photon_integral",
                output_when="final",
                dense_output=False,
                integral_weights=weights,
                abstol=ABSTOL,
                reltol=1e-7,
                dt=1e-10,
            )
            photons[i] = float(np.asarray(result.values).reshape(-1)[0].real)
        elapsed = time.perf_counter() - start
        scan_results[frame_name] = photons
        scan_wall[frame_name] = elapsed
        print(f"  {frame_name:8s}: scan wall = {elapsed:6.2f} s")

    diff = scan_results["rotated"] - scan_results["original"]
    max_abs_diff = float(np.max(np.abs(diff)))
    denom = np.maximum(np.abs(scan_results["original"]), 1e-300)
    max_rel_diff = float(np.max(np.abs(diff) / denom))
    argmax_orig = int(np.argmax(scan_results["original"]))
    argmax_rot = int(np.argmax(scan_results["rotated"]))
    print(
        f"  max abs diff = {max_abs_diff:.3e}, max rel diff = {max_rel_diff:.3e}, "
        f"argmax detuning: original={SCAN_DETUNINGS_MHZ[argmax_orig]:.1f} MHz, "
        f"rotated={SCAN_DETUNINGS_MHZ[argmax_rot]:.1f} MHz"
    )
    return {
        "scan_wall_original_s": scan_wall["original"],
        "scan_wall_rotated_s": scan_wall["rotated"],
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "argmax_detuning_MHz_original": float(SCAN_DETUNINGS_MHZ[argmax_orig]),
        "argmax_detuning_MHz_rotated": float(SCAN_DETUNINGS_MHZ[argmax_rot]),
        "argmax_match": argmax_orig == argmax_rot,
    }


def main() -> None:
    print("Building r2-in-E-field system (notebook-identical)...")
    t0 = time.perf_counter()
    system, ts = diag.build_system()
    print(f"  built in {time.perf_counter() - t0:.2f} s, n_states={len(system.QN)}")

    rotated = apply_per_level_rotating_frame(system)
    print(
        f"  rotated frame built; H_symbolic free symbols include t: "
        f"{'t' in {str(s) for s in rotated.H_symbolic.free_symbols}}"
    )

    rabi_value = diag.power_to_rabi_rectangular_beam(
        diag.POWER_W, abs(system.couplings[0].main_coupling), diag.BEAM_WX, diag.BEAM_WY
    )
    print(f"  Rabi rate: {mhz(rabi_value):.4f} MHz (2pi), Gamma: {mhz(GAMMA):.3f} MHz")

    rows: list[dict] = []
    run_step_table(system, ts, rotated, rabi_value, rows)
    scan_summary = run_photon_integral_scan(system, ts, rotated, rabi_value)

    # net speedup at the notebook tolerance (reltol=1e-7)
    orig_wall = next(
        r["wall_seconds_median"]
        for r in rows
        if r["frame"] == "original" and r["reltol"] == 1e-7 and r["detuning_MHz"] == 0.0
    )
    rot_wall = next(
        r["wall_seconds_median"]
        for r in rows
        if r["frame"] == "rotated" and r["reltol"] == 1e-7 and r["detuning_MHz"] == 0.0
    )
    speedup = orig_wall / rot_wall if rot_wall else float("nan")
    print(f"\nNet speedup at reltol=1e-7, detuning=0 MHz: {speedup:.2f}x")

    orig_accepted = next(
        r["accepted_steps"]
        for r in rows
        if r["frame"] == "original" and r["reltol"] == 1e-7 and r["detuning_MHz"] == 0.0
    )
    rot_accepted = next(
        r["accepted_steps"]
        for r in rows
        if r["frame"] == "rotated" and r["reltol"] == 1e-7 and r["detuning_MHz"] == 0.0
    )
    step_reduction = orig_accepted / rot_accepted if rot_accepted else float("nan")
    print(f"Step-count reduction at reltol=1e-7, detuning=0 MHz: {step_reduction:.2f}x")

    # tolerance scaling: rotated accepted steps at 1e-5 vs 1e-9
    rot_1e5 = next(
        r["accepted_steps"]
        for r in rows
        if r["frame"] == "rotated" and r["reltol"] == 1e-5 and r["detuning_MHz"] == 0.0
    )
    rot_1e9 = next(
        r["accepted_steps"]
        for r in rows
        if r["frame"] == "rotated" and r["reltol"] == 1e-9 and r["detuning_MHz"] == 0.0
    )
    tol_scaling = rot_1e9 / rot_1e5 if rot_1e5 else float("nan")
    print(
        f"Rotated-frame accepted-step ratio reltol 1e-9 / 1e-5: {tol_scaling:.2f}x "
        f"(accuracy-limited RK5(4) predicts ~(1e-4)^(1/5) = {10 ** (4 / 5):.1f}x; "
        f"flat/oscillation-limited predicts ~1x)"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "per_level_frame_bench.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "detuning_MHz", "reltol", "accepted_steps", "rejected_steps",
                "rhs_calls", "mean_dt_ns", "wall_seconds_median", "wall_repeats",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {RESULTS_DIR / 'per_level_frame_bench.csv'}")
    print("\nScan equivalence summary:", scan_summary)


if __name__ == "__main__":
    main()
