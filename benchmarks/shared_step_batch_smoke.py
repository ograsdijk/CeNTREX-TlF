"""Quick correctness smoke test for the isolated Rust experiment."""
import numpy as np

from bench_fixed_step_solvers import ground_state_density, make_two_level_system
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.batch import solve_lindblad_batch, _parameter_slot_indices
from centrex_tlf.centrex_tlf_rust import solve_shared_step_experiment_py


def main() -> None:
    system = make_two_level_system()
    prepared = prepare_lindblad_problem(
        system, {"Omega": 0.8, "delta": 0.0}, backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho = prepared.layout.pack(ground_state_density())
    batch = np.tile(rho, (3, 2, 1))
    batch[:, 1, :] = prepared.layout.pack(np.diag([0.0, 1.0]).astype(complex))
    values = np.array([[0.8], [1.0], [1.2]], dtype=np.complex128)
    slots = _parameter_slot_indices(prepared, ["Omega"])
    saveat = np.linspace(0.0, 0.8, 7)
    times, flat, width, stats = solve_shared_step_experiment_py(
        prepared.rust_plan, batch, slots, values, 0.0, 0.8,
        1e-10, 1e-8, 1e-3, saveat, "full",
    )
    shared = np.asarray(flat).reshape(len(times), 3, 2, width).transpose(1, 2, 0, 3)
    reference = solve_lindblad_batch(
        prepared, batch.reshape(6, -1), (0.0, 0.8),
        parameter_slots=["Omega"], parameter_batch=np.repeat(values, 2, axis=0),
        output="populations", output_when="saveat", saveat=saveat,
        abstol=1e-10, reltol=1e-8, dt=1e-3, parallel=False,
    )
    error = np.max(np.abs(shared[..., :2] - reference.values.reshape(3, 2, len(times), 2)))
    print("max population error", error)
    print("stats", dict(stats))
    assert error < 1e-7


if __name__ == "__main__":
    main()
