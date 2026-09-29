"""Measure existing expanded-sparse RHS phase costs without Python call overhead."""
from __future__ import annotations

from bench_shared_step_batch import RESULTS, prepare_model, write_row
import numpy as np
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py


for name in ("two_level", "r0", "q1_velocity_envelope", "r2_compact", "r2_full"):
    model = prepare_model(name)
    evaluator = create_lindblad_rhs_evaluator_py(model["prepared"].rust_plan, "expanded_sparse")
    state = np.zeros(model["n"] ** 2, dtype=np.float64)
    state[model["basis"][0]] = 1.0
    state[model["n"]:] = 0.001
    # Warm caches, then sample across the whole interval so dynamic expressions
    # see both field wings and the central pulse.
    evaluator.rhs_packed_py(state, model["span"][0])
    evaluator.enable_profile_py(True)
    evaluator.reset_profile_py()
    for t in np.linspace(*model["span"], 200):
        evaluator.rhs_packed_py(state, float(t))
    profile = dict(evaluator.profile_summary_py())
    write_row(RESULTS / "coefficient_profile.csv", dict(
        system=name, n_states=model["n"], packed_dim=model["n"] ** 2,
        calls=profile["calls"], total_seconds=profile["total_seconds"],
        parameter_eval_seconds=profile["parameter_eval_seconds"],
        hamiltonian_fill_seconds=profile["hamiltonian_fill_seconds"],
        commutator_seconds=profile["commutator_seconds"],
        parameter_fraction=profile["parameter_eval_seconds"] / profile["total_seconds"],
        coefficient_fraction=(profile["parameter_eval_seconds"] +
                              profile["hamiltonian_fill_seconds"]) / profile["total_seconds"],
    ))
    print(name, profile, flush=True)
