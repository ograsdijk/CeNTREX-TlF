"""Check localized Gaussian-velocity pulse convergence point by point."""
import numpy as np

from bench_shared_step_batch import RESULTS, prepare_model, parameter_case, write_row
from centrex_tlf.centrex_tlf_rust import solve_shared_step_experiment_py
from centrex_tlf.lindblad.batch import _parameter_slot_indices, solve_lindblad_batch


def main():
    model = prepare_model("q1_velocity_envelope")
    slots, values = parameter_case(model, "velocity_dynamic", 8)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    indices = _parameter_slot_indices(model["prepared"], slots)
    state = np.zeros((8, 1, model["n"] ** 2))
    state[:, 0, model["basis"][0]] = 1.
    production = solve_lindblad_batch(
        model["prepared"], state.reshape(8, -1), model["span"],
        parameter_slots=slots, parameter_batch=values,
        output="photon_integral", integral_weights=model["weights"],
        abstol=1e-9, reltol=1e-7, dt=model["dt"], parallel=False,
    ).values.reshape(8)
    for maximum_step in (None, 2e-7, 5e-8, 1e-8):
        _, flat, _, stats = solve_shared_step_experiment_py(
            model["prepared"].rust_plan, state, indices, values,
            *model["span"], 1e-9, 1e-7, model["dt"], None,
            "weighted_integral", model["weights"], 100000, maximum_step,
        )
        batched = np.asarray(flat).reshape(8)
        for point in range(8):
            single_state = np.ascontiguousarray(state[point:point + 1])
            single_values = np.ascontiguousarray(values[point:point + 1])
            _, one, _, one_stats = solve_shared_step_experiment_py(
                model["prepared"].rust_plan, single_state, indices, single_values,
                *model["span"], 1e-9, 1e-7, model["dt"], None,
                "weighted_integral", model["weights"], 100000, maximum_step,
            )
            row = dict(point=point, velocity=float(values[point, 0].real),
                       maximum_step=maximum_step, production=float(production[point]),
                       shared=float(batched[point]), independent_capped=float(one[0]),
                       shared_steps=stats["accepted_steps"], single_steps=one_stats["accepted_steps"])
            write_row(RESULTS / "velocity_diagnostic.csv", row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
