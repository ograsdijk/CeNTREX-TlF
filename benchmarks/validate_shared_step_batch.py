"""Compare full packed states, coherences, saveat, and solver-native integrals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench_shared_step_batch import RESULTS, prepare_model, parameter_case, write_row
from centrex_tlf.centrex_tlf_rust import (
    solve_lindblad_batch_ode_py, solve_shared_step_experiment_py,
)
from centrex_tlf.lindblad.batch import _parameter_slot_indices


def compare(model_name: str, kind: str, parameter_count: int, initial_count: int) -> None:
    model = prepare_model(model_name)
    slots, values = parameter_case(model, kind, parameter_count)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    slot_indices = _parameter_slot_indices(model["prepared"], slots)
    dim = model["n"] ** 2
    basis = np.zeros((parameter_count, initial_count, dim), dtype=np.float64)
    for initial, state in enumerate(model["basis"][:initial_count]):
        basis[:, initial, state] = 1.
    initial_flat = basis.reshape(-1, dim)
    parameters_flat = np.ascontiguousarray(np.repeat(values, initial_count, axis=0))
    saveat = np.linspace(*model["span"], 7)
    for output, weights in (("full", None), ("weighted_integral", model["weights"])):
        maximum_step = 2e-7 if model_name == "q1_velocity_envelope" else None
        if maximum_step is None:
            reference_output = "photon_integral" if output == "weighted_integral" else output
            times_ref, flat_ref, width_ref, count_ref, _ = solve_lindblad_batch_ode_py(
                model["prepared"].rust_plan, initial_flat, *model["span"],
                1e-9, 1e-7, model["dt"], saveat, True, 100000,
                "expanded_sparse", "dopri5", reference_output, None, "saveat", weights,
                slot_indices, parameters_flat, False, 1, None, True, "solver",
            )
            reference = np.asarray(flat_ref).reshape(parameter_count, initial_count, count_ref, width_ref)
        else:
            # The uncapped independent controller can step completely over a narrow pulse.
            # Use independent singleton solves with the same resolved maximum step.
            rows = []
            for point in range(parameter_count):
                row = []
                for initial in range(initial_count):
                    times_ref, one, width_ref, _ = solve_shared_step_experiment_py(
                        model["prepared"].rust_plan,
                        np.ascontiguousarray(basis[point:point + 1, initial:initial + 1]),
                        slot_indices, np.ascontiguousarray(values[point:point + 1]),
                        *model["span"], 1e-9, 1e-7, model["dt"], saveat,
                        output, weights, 100000, maximum_step,
                    )
                    row.append(np.asarray(one).reshape(len(saveat), width_ref))
                rows.append(row)
            reference = np.asarray(rows)
        times, flat, width, stats = solve_shared_step_experiment_py(
            model["prepared"].rust_plan, basis, slot_indices, values,
            *model["span"], 1e-9, 1e-7, model["dt"], saveat, output, weights,
            100000, maximum_step,
        )
        assert width == width_ref
        assert np.allclose(times, times_ref)
        shared = np.asarray(flat).reshape(len(times), parameter_count, initial_count, width)
        shared = shared.transpose(1, 2, 0, 3)
        diff = np.abs(reference - shared)
        result = dict(system=model_name, kind=kind, parameter_count=parameter_count,
                      initial_count=initial_count, output=output, save_count=len(times),
                      max_abs=float(diff.max()),
                      max_rel=float(diff.max() / max(1., np.abs(reference).max())))
        write_row(RESULTS / "accuracy_full.csv", result)
        print(result, flush=True)
        if output == "full":
            for label, part in (("populations", slice(0, model["n"])),
                                ("coherences", slice(model["n"], None))):
                component = diff[..., part]
                component_ref = reference[..., part]
                detail = dict(result, output=label, max_abs=float(component.max()),
                              max_rel=float(component.max() / max(1., np.abs(component_ref).max())))
                write_row(RESULTS / "accuracy_full.csv", detail)
        if output == "full" and initial_count > 1:
            weights_pop = np.arange(1, initial_count + 1, dtype=float)
            weights_pop /= weights_pop.sum()
            reconstructed = np.einsum("i,pitd->ptd", weights_pop, shared)
            mixture = np.einsum("i,pid->pd", weights_pop, basis)
            mixture_params = values
            if maximum_step is None:
                _, mix_flat, _, mix_time_count, _ = solve_lindblad_batch_ode_py(
                    model["prepared"].rust_plan, np.ascontiguousarray(mixture), *model["span"],
                    1e-9, 1e-7, model["dt"], saveat, True, 100000,
                    "expanded_sparse", "dopri5", "full", None, "saveat", None,
                    slot_indices, mixture_params, False, 1, None, True, "solver",
                )
                direct = np.asarray(mix_flat).reshape(parameter_count, mix_time_count, dim)
            else:
                direct = []
                for point in range(parameter_count):
                    _, mix_flat, _, _ = solve_shared_step_experiment_py(
                        model["prepared"].rust_plan,
                        np.ascontiguousarray(mixture[point:point + 1, None, :]),
                        slot_indices, np.ascontiguousarray(values[point:point + 1]),
                        *model["span"], 1e-9, 1e-7, model["dt"], saveat,
                        "full", None, 100000, maximum_step,
                    )
                    direct.append(np.asarray(mix_flat).reshape(len(saveat), dim))
                direct = np.asarray(direct)
            mixture_diff = np.abs(direct - reconstructed)
            result = dict(system=model_name, kind=kind, parameter_count=parameter_count,
                          initial_count=initial_count, output="population_reconstruction",
                          save_count=len(times), max_abs=float(mixture_diff.max()),
                          max_rel=float(mixture_diff.max() / max(1., np.abs(direct).max())))
            write_row(RESULTS / "accuracy_full.csv", result)
            print(result, flush=True)


def compare_coherent(model_name: str) -> None:
    model = prepare_model(model_name)
    n = model["n"]
    rho = np.zeros((n, n), dtype=np.complex128)
    i, j = model["basis"][:2]
    rho[i, i] = rho[j, j] = .5
    rho[i, j] = .2 + .1j
    rho[j, i] = .2 - .1j
    rho2 = np.zeros_like(rho)
    rho2[i, i], rho2[j, j] = .7, .3
    rho2[i, j], rho2[j, i] = .1 - .05j, .1 + .05j
    explicit = np.stack((model["prepared"].layout.pack(rho),
                         model["prepared"].layout.pack(rho2)))
    batch = np.ascontiguousarray(np.broadcast_to(explicit, (4, 2, n * n)))
    slots, values = parameter_case(model, "rabi_broad", 4)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    indices = _parameter_slot_indices(model["prepared"], slots)
    saveat = np.linspace(*model["span"], 7)
    _, reference, width, sample_count, _ = solve_lindblad_batch_ode_py(
        model["prepared"].rust_plan, batch.reshape(8, -1), *model["span"],
        1e-9, 1e-7, model["dt"], saveat, True, 100000,
        "expanded_sparse", "dopri5", "full", None, "saveat", None,
        indices, np.ascontiguousarray(np.repeat(values, 2, axis=0)),
        False, 1, None, True, "solver",
    )
    _, shared, _, _ = solve_shared_step_experiment_py(
        model["prepared"].rust_plan, batch, indices, values,
        *model["span"], 1e-9, 1e-7, model["dt"], saveat, "full",
    )
    reference = np.asarray(reference).reshape(4, 2, sample_count, width)
    shared = np.asarray(shared).reshape(sample_count, 4, 2, width).transpose(1, 2, 0, 3)
    diff = np.abs(reference - shared)
    result = dict(system=model_name, kind="explicit_coherence", parameter_count=4,
                  initial_count=2, output="full", save_count=sample_count,
                  max_abs=float(diff.max()), max_rel=float(diff.max() / max(1., np.abs(reference).max())))
    write_row(RESULTS / "accuracy_full.csv", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--kind", default="rabi_narrow")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--initials", type=int, default=4)
    parser.add_argument("--coherent", action="store_true")
    args = parser.parse_args()
    if args.coherent:
        compare_coherent(args.system)
    else:
        compare(args.system, args.kind, args.count, args.initials)
