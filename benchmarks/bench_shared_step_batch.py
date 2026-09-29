"""Reproducible benchmark of the isolated shared-step Rust experiment.

Run after `python -m maturin develop --release`. Raw results are appended so a
long realistic scan can be resumed. The production batch solver is the reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from centrex_tlf import transitions

from centrex_tlf.centrex_tlf_rust import (
    benchmark_shared_rhs_experiment_py, benchmark_shared_rhs_layout_py,
    solve_shared_step_experiment_py,
)
from centrex_tlf.lindblad.batch import _parameter_slot_indices, solve_lindblad_batch
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem

from bench_fixed_step_solvers import make_two_level_system
from bench_q1_r2_rhs_kernel_full_scans import prepare_q1, prepare_r2
from benchmark_obe import make_initial_state, make_parameters, setup_system

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "shared_step_batch_results"
MHZ = 2 * np.pi * 1e6


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prepare_model(name: str) -> dict:
    requested_name = name
    dynamic_velocity = name == "q1_velocity_envelope"
    if dynamic_velocity:
        name = "q1"
    compact = name.endswith("_compact")
    if dynamic_velocity:
        compact = True
    if name.endswith("_compact") or name.endswith("_full"):
        name = name.rsplit("_", 1)[0]
    if name == "two_level":
        system = make_two_level_system()
        prepared = prepare_lindblad_problem(
            system, {"Omega": 0.8, "delta": 0.0}, backend="rust",
            hamiltonian_representation="decomposed",
        )
        return dict(name=name, prepared=prepared, rho=np.array([1., 0., 0., 0.]),
                    basis=[0, 1], rabi_slot="Omega", detuning_slot="delta",
                    rabi=0.8, span=(0., 0.8), dt=1e-3, gamma=0.3,
                    weights=[(1, 0.3)], n=2, transition="synthetic",
                    E=None, B=None, polarization="synthetic", retained=False,
                    velocity=184., velocity_sigma=1.4, detuning_range=(-2., 2.))
    if name == "r0":
        system = setup_system()
        prepared = prepare_lindblad_problem(system, make_parameters(system), backend="rust",
                                             hamiltonian_representation="decomposed")
        n = len(system.QN)
        coupling = str(system.coupling_symbols[0])
        detuning = next((str(s) for s in system.H_symbolic.free_symbols
                         if s not in system.coupling_symbols and "delta" in str(s).lower()), None)
        if detuning is None:
            candidates = [str(s) for s in system.H_symbolic.free_symbols
                          if s not in system.coupling_symbols and "δ" in str(s)]
            detuning = candidates[0]
        driven = [i for i, state in enumerate(system.QN[:len(system.ground)])
                  if state.largest.J == transitions.R0_F1_3o2_F2.J_ground]
        driven.sort(key=lambda i: (system.QN[i].largest.F == 0, i))
        return dict(name=name, prepared=prepared, rho=prepared.layout.pack(make_initial_state(system)),
                    basis=driven, rabi_slot=coupling,
                    detuning_slot=detuning, rabi=2 * np.pi * 1.56e6,
                    span=(0., 10e-6), dt=1e-10, gamma=2 * np.pi * 1.56e6,
                    weights=[(i, 2 * np.pi * 1.56e6) for i in range(len(system.ground), n)],
                    n=n, transition="R0_F1_3o2_F2", E=[0., 0., 0.], B=[0., 0., 0.],
                    polarization="Z", retained=False, velocity=184., velocity_sigma=1.4,
                    detuning_range=(-30., 30.))
    from bench_q1_r2_rhs_kernel_full_scans import load_notebook_namespace, retained_system
    path = HERE.parent / "examples" / "lindblad" / f"{name}_opposite_parity_retention.ipynb"
    indices = [1, 3, 5, 7, 9] if name == "q1" else [1, 3, 5, 8, 10]
    ns = load_notebook_namespace(path, indices, quiet=True)
    system = retained_system(ns, qn_compact=compact)
    params, rabi = ns["parameter_values"](system)
    if dynamic_velocity:
        from centrex_tlf.lindblad.parameters import LindbladParameters, Time, gaussian, linear
        params = LindbladParameters()
        velocity_parameter = params.real("v", float(ns["VELOCITY"]))
        peak = params.real("omega_peak", float(rabi))
        position = linear(Time(), offset=-0.005, slope=velocity_parameter)
        envelope = gaussian(position, center=0.002, sigma=200e-6, amplitude=peak)
        polarization_symbols = {
            str(symbol)
            for group in system.polarization_symbols
            for symbol in (group if isinstance(group, (list, tuple)) else [group])
        }
        for symbol in system.H_symbolic.free_symbols:
            if symbol in system.coupling_symbols:
                params.bind(symbol, envelope, finalize=False)
            else:
                params.real(str(symbol), 1.0 if str(symbol) in polarization_symbols else 0.0)
        params._finalize()
    prepared = prepare_lindblad_problem(system, params, backend="rust",
                                         hamiltonian_representation="decomposed")
    rho = prepared.layout.pack(ns["initial_density_matrix"](system))
    initially = [i for i, state in enumerate(system.QN)
                 if state.largest.electronic_state == ns["states"].ElectronicState.X
                 and state.largest.J == ns["transition"].J_ground]
    first = int(np.argmax(rho[:len(system.QN)]))
    basis = [first] + [i for i in initially if i != first]
    detuning = str(vars(ns["transition_selectors"][0])["δ"])
    return dict(name=requested_name, prepared=prepared, rho=rho, basis=basis,
                rabi_slot="omega_peak" if dynamic_velocity else str(system.coupling_symbols[0]),
                detuning_slot=detuning,
                rabi=float(rabi), span=(0., 50e-6 if dynamic_velocity else float(ns["T_END"])),
                dt=1e-10 if dynamic_velocity else 2e-9,
                gamma=float(ns.get("GAMMA", getattr(ns["hamiltonian"], "Γ"))),
                weights=[(int(i), float(ns.get("GAMMA", getattr(ns["hamiltonian"], "Γ"))))
                         for i in ns["index_sets"](system)["excited"]],
                n=len(system.QN), transition=ns["transition"].name,
                E=ns["E_FIELD"].tolist(), B=ns["B_FIELD"].tolist(),
                polarization="X", retained=True, velocity=float(ns["VELOCITY"]),
                velocity_sigma=float(ns["TRANSVERSE_VELOCITY_SIGMA"]),
                forward_velocity_sigma=0.1 * float(ns["VELOCITY"]),
                field_envelope=("Gaussian along z=-5 mm+v*t; center=2 mm; sigma=0.2 mm"
                                if dynamic_velocity else "flat-top 2 cm interaction"),
                detuning_range=(float(ns["DETUNING_SCAN_MHZ"][0]),
                                 float(ns["DETUNING_SCAN_MHZ"][-1])))


def parameter_case(model: dict, kind: str, count: int) -> tuple[list[str], np.ndarray]:
    omega = model["rabi"]
    lo, hi = model["detuning_range"]
    if kind == "rabi_narrow":
        return [model["rabi_slot"]], np.linspace(0.8, 1.2, count)[:, None] * omega
    if kind == "rabi_broad":
        return [model["rabi_slot"]], np.linspace(0., 4., count)[:, None] * omega
    if kind == "rabi_straggler":
        v = np.ones(count) * omega
        v[-1] = 10 * omega
        return [model["rabi_slot"]], v[:, None]
    if kind in {"velocity_dynamic", "velocity_straggler_dynamic"}:
        v = np.linspace(.7, 1.3, count) * model["velocity"]
        if kind == "velocity_straggler_dynamic":
            v[:] = model["velocity"]
            v[-1] = 1.3 * model["velocity"]
        return ["v"], v[:, None]
    if kind == "rabi_velocity_dynamic":
        side = int(np.sqrt(count))
        if side * side != count:
            raise ValueError("combined count must be square")
        a, b = np.meshgrid(np.linspace(.5, 1.5, side) * omega,
                           np.linspace(.7, 1.3, side) * model["velocity"], indexing="ij")
        return ["omega_peak", "v"], np.stack((a.ravel(), b.ravel()), axis=1)
    if kind in {"detuning", "detuning_straggler", "velocity"}:
        if kind == "velocity":
            # Notebook physics: transverse velocity produces optical Doppler detuning.
            # Existing Q1/R2 examples use 1.1e15 Hz optical frequency and 1.4 m/s sigma.
            v = np.linspace(-3, 3, count) * model["velocity_sigma"]
            vals = 2 * np.pi * 1.1e15 * v / scipy.constants.c
        else:
            vals = MHZ * np.linspace(lo, hi, count)
            if kind == "detuning_straggler":
                vals[:] = 0.
                vals[-1] = MHZ * hi * 3
        return [model["detuning_slot"]], vals[:, None]
    if kind in {"rabi_detuning", "rabi_velocity"}:
        side = int(np.sqrt(count))
        if side * side != count:
            raise ValueError("combined count must be square")
        rabi = np.linspace(.5, 1.5, side) * omega
        if kind == "rabi_velocity":
            other = MHZ * (1.1e15 / scipy.constants.c / 1e6) * np.linspace(-3, 3, side) * model["velocity_sigma"]
        else:
            other = MHZ * np.linspace(lo, hi, side)
        a, b = np.meshgrid(rabi, other, indexing="ij")
        return [model["rabi_slot"], model["detuning_slot"]], np.stack((a.ravel(), b.ravel()), axis=1)
    raise ValueError(kind)


def run_case(model: dict, kind: str, count: int, initials: int, repeat: int,
             threads: int, output: str, saveat: np.ndarray | None) -> None:
    slots, values = parameter_case(model, kind, count)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    selected = model["basis"][:initials]
    if len(selected) != initials:
        return
    dim = model["n"] ** 2
    basis = np.zeros((initials, dim), dtype=np.float64)
    for column, state in enumerate(selected):
        basis[column, state] = 1.
    batch = np.ascontiguousarray(np.broadcast_to(basis, (count, initials, dim)))
    flat = batch.reshape(count * initials, dim)
    flat_params = np.ascontiguousarray(np.repeat(values, initials, axis=0))
    slot_indices = _parameter_slot_indices(model["prepared"], slots)
    output_shared = "weighted_integral" if output == "photon_integral" else output
    save = np.asarray([model["span"][1]]) if saveat is None else saveat

    def independent(parallel: bool, nthreads: int):
        return solve_lindblad_batch(
            model["prepared"], flat, model["span"], parameter_slots=slots,
            parameter_batch=flat_params, solver="dopri5", execution_mode="expanded_sparse",
            output=output, output_when="final" if saveat is None else "saveat",
            integral_weights=model["weights"] if output == "photon_integral" else None,
            saveat=saveat, dt=model["dt"], reltol=1e-7, abstol=1e-9,
            parallel=parallel, threads=nthreads, collect_stats=True,
        )

    def shared():
        maximum_step = 2e-7 if model["name"] == "q1_velocity_envelope" else None
        return solve_shared_step_experiment_py(
            model["prepared"].rust_plan, batch, slot_indices, values,
            *model["span"], 1e-9, 1e-7, model["dt"], save,
            output_shared, model["weights"] if output == "photon_integral" else None,
            100000, maximum_step,
        )

    def independent_capped(parallel: bool = False):
        series = []
        aggregate = {"accepted_steps": 0, "rejected_steps": 0, "rhs_calls": 0}
        def one(index: int):
            point, initial = divmod(index, initials)
            return solve_shared_step_experiment_py(
                model["prepared"].rust_plan,
                np.ascontiguousarray(batch[point:point + 1, initial:initial + 1]),
                slot_indices, np.ascontiguousarray(values[point:point + 1]),
                *model["span"], 1e-9, 1e-7, model["dt"], save,
                output_shared, model["weights"] if output == "photon_integral" else None,
                100000, 2e-7,
            )
        if parallel:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                responses = list(executor.map(one, range(count * initials)))
        else:
            responses = [one(index) for index in range(count * initials)]
        for _, values_one, width_one, stats_one in responses:
            series.append(np.asarray(values_one).reshape(len(save), width_one))
            for key in aggregate:
                aggregate[key] += stats_one[key]
        return SimpleNamespace(values=np.asarray(series).reshape(count * initials, len(save), -1),
                               solver_stats=aggregate)

    # Warm both paths once. Do not include preparation or JIT/build time.
    independent(False, 1)
    shared()
    results = {}
    operations = [("independent_serial", lambda: independent(False, 1)),
                  ("independent_rayon", lambda: independent(True, threads))]
    if model["name"] == "q1_velocity_envelope":
        operations.append(("independent_capped_serial", independent_capped))
        operations.append(("independent_capped_parallel", lambda: independent_capped(True)))
    operations.append(("shared_serial", shared))
    for label, operation in operations:
        start = time.perf_counter()
        result = operation()
        seconds = time.perf_counter() - start
        results[label] = result
        stats = dict(result[3]) if label == "shared_serial" else result.solver_stats
        write_row(RESULTS / "solver_timings.csv", dict(
            system=model["name"], kind=kind, parameter_count=count,
            initial_count=initials, trajectories=count * initials,
            repeat=repeat, method=label,
            threads=threads if label in {"independent_rayon", "independent_capped_parallel"} else 1,
            output=output, save_count=len(save), seconds=seconds,
            accepted_steps=stats["accepted_steps"], rejected_steps=stats["rejected_steps"],
            rhs_calls=stats["rhs_calls"], trajectories_per_second=count * initials / seconds,
        ))
    reference_label = ("independent_capped_serial" if model["name"] == "q1_velocity_envelope"
                       else "independent_serial")
    common = results[reference_label].values.reshape(count, initials, len(save), -1)
    times, flat_result, width, stats = results["shared_serial"]
    batched = np.asarray(flat_result).reshape(len(times), count, initials, width).transpose(1, 2, 0, 3)
    diff = np.abs(common - batched)
    write_row(RESULTS / "accuracy.csv", dict(
        system=model["name"], kind=kind, parameter_count=count, initial_count=initials,
        output=output, save_count=len(save), max_abs=float(diff.max()),
        max_rel=float(diff.max() / max(1., np.abs(common).max())),
    ))
    if reference_label != "independent_serial":
        unbounded = results["independent_serial"].values.reshape(count, initials, len(save), -1)
        write_row(RESULTS / "accuracy_unbounded_velocity.csv", dict(
            system=model["name"], kind=kind, parameter_count=count, initial_count=initials,
            max_abs_vs_capped=float(np.abs(unbounded - common).max()),
        ))
    counts = np.asarray(stats["controller_counts"])
    steps = np.asarray(stats["attempted_steps"])
    if "straggler" in kind:
        np.savez_compressed(
            RESULTS / f"steps_{model['name']}_{kind}_p{count}_i{initials}_r{repeat}.npz",
            attempted_dt=steps,
            controller_counts=counts,
            rejected_controller_counts=np.asarray(stats["rejected_controller_counts"]),
        )
    write_row(RESULTS / "straggler_stats.csv", dict(
        system=model["name"], kind=kind, parameter_count=count, initial_count=initials,
        accepted_steps=stats["accepted_steps"], rejected_steps=stats["rejected_steps"],
        controlling_trajectory=int(np.argmax(counts)), controlling_fraction=float(counts.max() / counts.sum()),
        distinct_controllers=int(np.count_nonzero(counts)), min_dt=float(steps.min()),
        median_dt=float(np.median(steps)), max_dt=float(steps.max()),
        controller_counts=json.dumps(counts.tolist()),
    ))
    print(model["name"], kind, count, initials, output,
          "max_error", float(diff.max()), flush=True)


def rhs_case(model: dict, kind: str, count: int, initials: int, repeat: int) -> None:
    slots, values = parameter_case(model, kind, count)
    selected = model["basis"][:initials]
    if len(selected) != initials:
        return
    batch = np.zeros((count, initials, model["n"] ** 2), dtype=np.float64)
    for initial, state in enumerate(selected):
        batch[:, initial, state] = 1.
    # Add a small Hermitian-like packed perturbation so off-diagonal paths run.
    batch[..., model["n"]:] = .001
    values = np.ascontiguousarray(values, dtype=np.complex128)
    slots = _parameter_slot_indices(model["prepared"], slots)
    repetitions = max(3, min(1000, 4000 // (count * initials)))
    independent, batched, error = benchmark_shared_rhs_experiment_py(
        model["prepared"].rust_plan, batch, slots, values,
        model["span"][0] + .25 * (model["span"][1] - model["span"][0]), repetitions,
    )
    write_row(RESULTS / "rhs_microbench.csv", dict(
        system=model["name"], kind=kind, parameter_count=count, initial_count=initials,
        trajectories=count * initials, repeat=repeat, evaluations=repetitions,
        independent_seconds=independent, batched_seconds=batched,
        speedup=independent / batched, max_abs_rhs_diff=error,
    ))
    print("RHS", model["name"], kind, count, initials,
          "speedup", round(independent / batched, 3), "error", error, flush=True)


def layout_case(model: dict, kind: str, count: int, initials: int, repeat: int) -> None:
    if model["n"] <= 40 or initials > len(model["basis"]):
        return
    slots, values = parameter_case(model, kind, count)
    batch = np.zeros((count, initials, model["n"] ** 2), dtype=np.float64)
    for initial, state in enumerate(model["basis"][:initials]):
        batch[:, initial, state] = 1.
    batch[..., model["n"]:] = .001
    values = np.ascontiguousarray(values, dtype=np.complex128)
    slots = _parameter_slot_indices(model["prepared"], slots)
    repetitions = max(3, min(1000, 4000 // (count * initials)))
    row, column, conversion, error = benchmark_shared_rhs_layout_py(
        model["prepared"].rust_plan, batch, slots, values,
        model["span"][0] + .25 * (model["span"][1] - model["span"][0]), repetitions,
    )
    write_row(RESULTS / "rhs_layout_microbench.csv", dict(
        system=model["name"], kind=kind, parameter_count=count, initial_count=initials,
        repeat=repeat, evaluations=repetitions, trajectory_major_seconds=row,
        state_major_seconds=column, transpose_once_seconds=conversion,
        state_major_speedup=row / column, max_abs_rhs_diff=error,
    ))
    print("LAYOUT", model["name"], kind, count, initials,
          "state-major speedup", round(row / column, 3), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["two_level", "r0", "q1_full", "q1_compact", "r2_full", "r2_compact", "q1_velocity_envelope"], required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--initials", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--kinds", nargs="+", default=["rabi_narrow", "rabi_broad", "detuning", "rabi_detuning"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", choices=["populations", "photon_integral", "full"], default="photon_integral")
    parser.add_argument("--save-points", type=int, default=0)
    parser.add_argument("--rhs-only", action="store_true")
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()
    model = prepare_model(args.system)
    RESULTS.mkdir(exist_ok=True)
    metadata = {k: v for k, v in model.items() if k not in {"prepared", "rho", "basis"}}
    metadata.update(packed_dim=model["n"] ** 2, population_basis_states=model["basis"],
                    reltol=1e-7, abstol=1e-9, solver="dopri5", output=args.output,
                    save_points=args.save_points,
                    maximum_step=2e-7 if args.system == "q1_velocity_envelope" else None)
    (RESULTS / f"system_{args.system}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    environment = dict(cpu=platform.processor(), logical_cores=os.cpu_count(), os=platform.platform(),
                       python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                       rust=subprocess.check_output(["rustc", "--version"], text=True).strip(),
                       build="maturin develop --release", threads=args.threads)
    (RESULTS / "benchmark_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    saveat = None if args.save_points == 0 else np.linspace(*model["span"], args.save_points)
    for kind in args.kinds:
        for count in args.counts:
            if kind.startswith("rabi_") and kind.endswith(("detuning", "velocity", "velocity_dynamic")) and int(np.sqrt(count)) ** 2 != count:
                continue
            for initials in args.initials:
                for repeat in range(args.repeats):
                    if args.layout_only:
                        layout_case(model, kind, count, initials, repeat)
                    elif args.rhs_only:
                        rhs_case(model, kind, count, initials, repeat)
                    else:
                        run_case(model, kind, count, initials, repeat, args.threads, args.output, saveat)


if __name__ == "__main__":
    main()
