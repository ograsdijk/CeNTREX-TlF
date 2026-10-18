"""Validate the promoted opt-in APIs on the archived RC/R(2) fixtures."""

from __future__ import annotations

import argparse
import copy
import gc
import time

import check_cuda_algorithms as alg
import numpy as np

from centrex_tlf import lindblad


def reduction():
    p, packed, pars, names, span, dt = alg.fixture("RC", 1024, 1.0)
    options = {
        "full": lindblad.CudaSolverOptions(batch_size=1024),
        "reduced": lindblad.CudaSolverOptions(batch_size=1024, density_reduction=True),
    }

    def solve(option):
        return lindblad.solve_lindblad_batch(
            p,
            packed,
            span,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt,
            parameter_batch=pars,
            parameter_slots=names,
            output="full",
            collect_stats=True,
            cuda_options=option,
        )

    for option in options.values():
        solve(option)
    rng = np.random.default_rng(747)
    timings = {key: [] for key in options}
    answers = {}
    for _ in range(3):
        for key in rng.permutation(list(options)):
            tick = time.perf_counter()
            answers[key] = solve(options[key])
            timings[key].append(time.perf_counter() - tick)
    reduction = dict(
        timings=timings,
        check=alg.compare(p.layout, answers["reduced"].values, answers["full"].values),
        chunks=answers["reduced"].solver_stats["chunks"],
    )
    alg.save("production_reduction.json", reduction)
    print("Public RC reduction", reduction, flush=True)
    del answers
    gc.collect()
    alg.cp.get_default_memory_pool().free_all_blocks()


def periodic():
    rows = []
    for name in ("RC", "R2_EOM"):
        p, packed, pars, names, _, dt = alg.fixture(name, 3 if name == "RC" else 20, 1.0)
        if name == "RC":
            # The archived prepared snapshot omits its high-level Parameters.
            # Remove only its two known Gaussian-Rabi factors, in a copied IR.
            p = copy.copy(p)
            p.parameter_graph = copy.deepcopy(p.parameter_graph)
            p.rust_plan = None
            replaced = 0
            for compound in p.parameter_graph["compounds"]:
                instructions = compound["expression"]["instructions"]
                if any(x.get("function") == 15 for x in instructions):
                    assert len(instructions) == 6 and instructions[-2].get("function") == 15
                    assert instructions[0]["op"] == 2 and instructions[-1]["op"] == 7
                    compound["expression"]["instructions"] = instructions[:1]
                    replaced += 1
            assert replaced == 2
            period, length = 5e-6, 0.044
        else:
            period, length = 0.625e-6, 0.02
        values = dict(zip(names, pars[0], strict=True))
        values[names[0]] = 0.0
        start = 125e-9
        times = np.sort(start + length / np.array([150.0, 184.0, 250.0]))
        print(
            f"Public periodic map {name}; three forward velocities, nonzero entry phase", flush=True
        )
        tick = time.perf_counter()
        prop = lindblad.prepare_periodic_lindblad_propagator(
            p,
            period,
            packed,
            dt=dt,
            parameter_values=values,
            cuda_options=lindblad.CudaSolverOptions(batch_size=1024, density_reduction=True),
        )
        build = time.perf_counter() - tick
        got = prop.evaluate(packed, times, t0=start)
        reference = lindblad.solve_lindblad_batch(
            p,
            packed,
            (start, float(times[-1])),
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt,
            parameter_slots=list(values),
            parameter_batch=np.repeat(np.array(list(values.values()))[None], len(packed), axis=0),
            output="full",
            output_when="saveat",
            saveat=times,
            save_start=False,
            maxiters=1000000,
        )
        check = alg.compare(
            p.layout,
            got.reshape(-1, p.layout.packed_len),
            reference.values.reshape(-1, p.layout.packed_len),
        )
        row = dict(
            system=name,
            construction_seconds=build,
            initial_states=len(packed),
            velocities_m_s=[150.0, 184.0, 250.0],
            times=times.tolist(),
            entry_time=start,
            check=check,
            stats=prop.stats,
        )
        rows.append(row)
        alg.save("production_periodic.json", rows)
        print(row, flush=True)
        assert check["passed"]
        prop.clear_cuda_cache()
        del prop
        gc.collect()
        alg.cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--periodic-only", action="store_true")
    if not parser.parse_args().periodic_only:
        reduction()
    periodic()
