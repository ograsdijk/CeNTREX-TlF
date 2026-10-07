from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import sympy as smp

from centrex_tlf.lindblad import (
    grid_scan,
    initial_condition_scan,
    parameter_scan,
    prepare_dense_lindblad_propagator,
    prepare_lindblad_problem,
    solve_lindblad,
    solve_lindblad_batch,
)

pytest.importorskip("centrex_tlf.centrex_tlf_rust")


def model(n=2, parameters=None):
    omega, delta = smp.symbols("omega delta", real=True)
    H = smp.zeros(n)
    H[0, 1] = H[1, 0] = omega / 2
    H[1, 1] = -delta
    C = np.zeros((1 if n == 2 else 2, n, n), complex)
    C[0, 0, 1] = np.sqrt(0.3)
    if n > 2:
        C[1, 2, 1] = np.sqrt(0.2)
    p = dict(omega=1.2, delta=0.4) if parameters is None else parameters
    prepared = prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=C), p)
    return prepared, H, C


def complex_generator(H, C, **parameters):
    substitutions = {s: parameters[str(s)] for s in H.free_symbols}
    H = np.array(H.subs(substitutions), complex)
    identity = np.eye(len(H))
    L = -1j * (np.kron(identity, H) - np.kron(H.T, identity))
    for jump in C:
        K = jump.conj().T @ jump
        L += np.kron(jump.conj(), jump) - 0.5 * (np.kron(identity, K) + np.kron(K.T, identity))
    return L


def independent(H, C, rho, times, weights=None, omega=1.2, delta=0.4):
    L = complex_generator(H, C, omega=omega, delta=delta)
    state = rho.reshape(-1, order="F")
    if weights is None:
        return np.stack([(la.expm(L * t) @ state).reshape(rho.shape, order="F") for t in times])
    augmented = np.zeros((len(L) + 1, len(L) + 1), complex)
    augmented[:-1, :-1] = L
    for i, value in weights:
        augmented[-1, i * (len(rho) + 1)] += value
    return np.array([(la.expm(augmented * t) @ np.r_[state, 0])[-1].real for t in times])


def states(n=2):
    a = np.zeros((n, n), complex)
    a[0, 0] = 1
    b = np.zeros_like(a)
    b[:2, :2] = [[0.4, 0.1 + 0.15j], [0.1 - 0.15j, 0.6]]
    if n == 3:
        b *= 0.7
        b[2, 2] = 0.3
    return np.stack((a, b))


@pytest.mark.parametrize("n", [2, 3])
def test_density_and_sink_integrals_match_independent_exponential(n):
    prepared, H, C = model(n)
    initial = states(n)
    times = np.array([2.0, 2.03, 2.2, 2.7])
    factor = prepare_dense_lindblad_propagator(prepared, initial)
    density = factor.density_matrices(initial, times, t0=2.0)
    for i, rho in enumerate(initial):
        expected = independent(H, C, rho, times - 2.0)
        np.testing.assert_allclose(density[i], expected, atol=2e-13)
        assert np.min(la.eigvalsh(density[i])) > -1e-12
    np.testing.assert_allclose(np.trace(density, axis1=-2, axis2=-1), 1, atol=2e-13)
    for weights in [[(1, 0.5)], [(i, 1.0) for i in range(n)]]:
        photons = factor.evaluate(
            initial, times, t0=2.0, output="weighted_integral", integral_weights=weights
        )
        for i, rho in enumerate(initial):
            np.testing.assert_allclose(
                photons[i, :, 0], independent(H, C, rho, times - 2.0, weights), atol=2e-13
            )
    if n == 3:
        assert factor.stats["sink_count"] == 1
        with pytest.raises(ValueError, match="outside this reduced"):
            coherent_sink = initial.copy()
            coherent_sink[0, 0, 2] = coherent_sink[0, 2, 0] = 0.1
            factor.evaluate(coherent_sink, times)
        full = prepare_dense_lindblad_propagator(prepared, reduce=False)
        rho = np.ones((3, 3), complex) / 3
        np.testing.assert_allclose(
            full.density_matrices(rho[None], times - 2.0)[0],
            independent(H, C, rho, times - 2.0),
            atol=2e-13,
        )


def test_single_outputs_and_ode_agreement():
    prepared, _, _ = model()
    rho = states()[1]
    times = [0.0, 0.1, 0.7]
    dense = solve_lindblad(
        prepared, rho, (0.0, 0.7), solver="dense_eig", saveat=times, collect_stats=True
    )
    ode = solve_lindblad(
        prepared, rho, (0.0, 0.7), saveat=times, abstol=1e-12, reltol=1e-10, dt=1e-3
    )
    np.testing.assert_allclose(dense.density_matrices(), ode.density_matrices(), atol=2e-10)
    selected = solve_lindblad(
        prepared,
        rho,
        (0.0, 0.7),
        solver="dense_eig",
        saveat=times,
        output="selected",
        output_indices=[(0, 1), (1, 0), (0, 0)],
    )
    np.testing.assert_allclose(
        selected.values, dense.density_matrices()[:, [0, 1, 0], [1, 0, 0]], atol=1e-13
    )
    for when in ["saveat", "final"]:
        photons = solve_lindblad(
            prepared,
            rho,
            (0.0, 0.7),
            solver="dense_eig",
            saveat=times,
            output="photon_integral",
            integral_weights=[(1, 0.3)],
            output_when=when,
        )
        assert np.asarray(photons.values).shape == ((3,) if when == "saveat" else ())
    final = solve_lindblad(
        prepared, rho, (0.0, 0.7), solver="dense_eig", output="full", output_when="final"
    )
    np.testing.assert_allclose(final.packed_y[0], dense.packed_y[-1])
    assert dense.solver_stats["factorization_count"] == 1


def test_grouped_initial_states_parameter_snapshot_and_grid_order():
    prepared, H, C = model()
    initial = states()
    before = repr(prepared.parameter_graph)
    result = initial_condition_scan(
        prepared, initial, (0.0, 0.7), solver="dense_eig", collect_stats=True
    )
    assert result.solver_stats["factorization_count"] == 1
    assert result.values.shape == (2, 2)
    grid = grid_scan(
        prepared,
        None,
        (0.0, 0.7),
        rho0_batch=initial,
        scan={"omega": [1.2, 0.7], "delta": [0.4, -0.2]},
        solver="dense_eig",
        output="full",
        output_when="saveat",
        saveat=[0.0, 0.7],
        parallel=False,
        collect_stats=True,
    )
    assert grid.values.shape == (8, 2, 4)
    assert grid.solver_stats["factorization_count"] == 4
    assert grid.metadata["grid_shape"] == (2, 2)
    assert grid.metadata["initial_condition_count"] == 2
    assert repr(prepared.parameter_graph) == before
    for point, (omega, delta) in enumerate([(1.2, 0.4), (1.2, -0.2), (0.7, 0.4), (0.7, -0.2)]):
        for i, rho in enumerate(initial):
            expected = independent(H, C, rho, [0.7], omega=omega, delta=delta)[0]
            np.testing.assert_allclose(
                prepared.layout.unpack(grid.values[point * 2 + i, -1]), expected, atol=2e-13
            )
    scanned = parameter_scan(
        prepared,
        initial[0],
        (0.0, 0.7),
        parameter_slots=["omega"],
        parameter_batch=np.array([[1.2], [1.2], [0.7]]),
        solver="dense_eig",
        parallel=False,
        collect_stats=True,
    )
    assert scanned.solver_stats["factorization_count"] == 2
    np.testing.assert_allclose(scanned.values[0], scanned.values[1])


def test_parallel_process_scan_matches_serial():
    prepared, _, _ = model()
    options = dict(
        scan={"omega": [1.2, 0.7], "delta": [0.4, -0.2]},
        solver="dense_eig",
        output="photon_integral",
        integral_weights=[(1, 0.3)],
        output_when="saveat",
        saveat=[0.0, 0.7],
        collect_stats=True,
    )
    serial = grid_scan(prepared, states()[0], (0.0, 0.7), parallel=False, **options)
    parallel = grid_scan(
        prepared, states()[0], (0.0, 0.7), threads=2, profile_startup=True, **options
    )
    np.testing.assert_allclose(serial.values, parallel.values, atol=2e-13)
    assert parallel.solver_stats["workers"] == 2
    assert parallel.solver_stats["evaluator_count"] == 2
    assert parallel.solver_stats["worker_startup_seconds"] > 0
    assert parallel.solver_stats["steady_state_seconds"] > 0
    normal = grid_scan(prepared, states()[0], (0.0, 0.7), threads=2, **options)
    np.testing.assert_allclose(serial.values, normal.values, atol=2e-13)
    assert normal.solver_stats["worker_startup_seconds"] is None
    assert normal.solver_stats["steady_state_seconds"] is None
    assert not normal.solver_stats["startup_synchronized"]


def test_serial_scan_reuses_plan_and_evaluator(monkeypatch):
    import centrex_tlf.centrex_tlf_rust as rust

    prepared, _, _ = model()
    create = rust.create_lindblad_rhs_evaluator_py
    calls = []

    def counted(*args):
        calls.append(True)
        return create(*args)

    def no_rebuild(*args):
        raise AssertionError("must reuse the prepared Rust plan")

    monkeypatch.setattr(rust, "create_lindblad_rhs_evaluator_py", counted)
    monkeypatch.setattr(rust, "prepare_lindblad_problem_py", no_rebuild)
    result = grid_scan(
        prepared,
        states()[0],
        (0.0, 0.7),
        scan={"omega": [1.2, 0.7, 1.8], "delta": [0.4, -0.2]},
        solver="dense_eig",
        parallel=False,
        collect_stats=True,
    )
    assert len(calls) == 1
    assert result.solver_stats["factorization_count"] == 6
    assert result.solver_stats["evaluator_count"] == 1


def test_native_override_cache_invalidation_and_reset():
    import centrex_tlf.centrex_tlf_rust as rust

    prepared, _, _ = model(parameters=dict(drive=1.2, omega="drive", delta=0.4))
    evaluator = rust.create_lindblad_rhs_evaluator_py(prepared.rust_plan, "expanded_sparse")
    packed = prepared.layout.pack(states()[1])
    baseline_graph = repr(prepared.parameter_graph)
    baseline = evaluator.rhs_packed_py(packed, 0.0).copy()
    slot = prepared.parameter_graph["slot_names"].index("drive")
    for value in [0.7, 1.8, 0.0, 1.2]:
        evaluator.set_scalar_parameter_overrides_py([slot], np.array([value], complex))
        expected, _, _ = model(parameters=dict(drive=value, omega="drive", delta=0.4))
        reference = rust.create_lindblad_rhs_evaluator_py(expected.rust_plan, "expanded_sparse")
        # RHS calls warm the static coefficient cache before the next update.
        np.testing.assert_allclose(
            evaluator.rhs_packed_py(packed, 0.2), reference.rhs_packed_py(packed, 0.2), atol=1e-14
        )
        for candidate, oracle in zip(
            evaluator.jacobian_packed_sparse_py(0.0, 0.0, "analytic"),
            reference.jacobian_packed_sparse_py(0.0, 0.0, "analytic"),
            strict=True,
        ):
            np.testing.assert_array_equal(candidate, oracle)
    evaluator.set_scalar_parameter_overrides_py([], np.array([], complex))
    np.testing.assert_array_equal(evaluator.rhs_packed_py(packed, 0.0), baseline)
    for slots, values in [([slot], []), ([100], [1]), ([slot, slot], [1, 2]), ([slot], [np.nan])]:
        with pytest.raises(ValueError):
            evaluator.set_scalar_parameter_overrides_py(slots, np.array(values, complex))
        np.testing.assert_array_equal(evaluator.rhs_packed_py(packed, 0.0), baseline)
    assert repr(prepared.parameter_graph) == baseline_graph


@pytest.mark.parametrize(
    "parameters",
    [dict(omega="sin(t)", delta=0.4), dict(amplitude="sin(t)", omega="amplitude", delta=0.4)],
)
def test_rejects_direct_and_indirect_time_dependence(parameters):
    prepared, _, _ = model(parameters=parameters)
    with pytest.raises(ValueError, match="time-independent"):
        solve_lindblad(prepared, states()[0], (0.0, 0.7), solver="dense_eig")
    with pytest.raises(ValueError, match="time-independent"):
        grid_scan(prepared, states()[0], (0.0, 0.7), scan={"delta": [0.0]}, solver="dense_eig")


def test_explicit_time_in_hamiltonian_rejected():
    H = smp.Matrix([[0, smp.sin(smp.Symbol("t"))], [smp.sin(smp.Symbol("t")), 0]])
    prepared = prepare_lindblad_problem(
        SimpleNamespace(H_symbolic=H, C_array=np.zeros((0, 2, 2))), {}
    )
    with pytest.raises(ValueError, match="time-independent"):
        prepare_dense_lindblad_propagator(prepared)


def test_zero_generator_and_defective_generator():
    H = smp.zeros(3)
    zero = prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=np.zeros((0, 3, 3))), {})
    rho = np.eye(3) / 3
    factor = prepare_dense_lindblad_propagator(zero, rho[None])
    np.testing.assert_allclose(
        factor.density_matrices(rho[None], [0.0, 1.0])[0], np.stack([rho, rho])
    )
    integrals = factor.evaluate(
        rho[None], [0.0, 1.0], output="weighted_integral", integral_weights=[(0, 1)]
    )
    np.testing.assert_allclose(integrals[0, :, 0], [0, 1 / 3])
    C = np.zeros((2, 3, 3))
    C[0, 0, 1] = C[1, 1, 2] = 1
    defective = prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=C), {})
    rho = np.diag([0.0, 0.0, 1.0])[None]
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        prepare_dense_lindblad_propagator(defective, rho)


@pytest.mark.parametrize(
    "options",
    [
        dict(stop_event=object()),
        dict(integral_method="sampled"),
        dict(saveat=[-0.1], output_when="saveat"),
        dict(evolution_device="wrong"),
        dict(threads=0),
    ],
)
def test_invalid_options_fail_before_work(options):
    prepared, _, _ = model()
    with pytest.raises(ValueError):
        solve_lindblad_batch(prepared, states(), (0.0, 0.7), solver="dense_eig", **options)


def test_optional_gpu_and_missing_dependency(monkeypatch):
    import builtins

    from centrex_tlf.lindblad import dense

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="optional PyTorch"):
        dense._torch_device("cuda")


def test_cuda_projection_agrees_with_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    prepared, _, _ = model(3)
    options = dict(
        rho0_batch=states(3),
        scan={"omega": [1.2, 0.7], "delta": [0.4]},
        solver="dense_eig",
        output="photon_integral",
        integral_weights=[(1, 0.5), (2, 0.1)],
        output_when="saveat",
        saveat=[0.0, 0.03, 0.7],
        parallel=False,
    )
    cpu = grid_scan(prepared, None, (0.0, 0.7), **options)
    gpu = grid_scan(
        prepared, None, (0.0, 0.7), evolution_device="cuda", gpu_batch_size=2, **options
    )
    np.testing.assert_allclose(cpu.values, gpu.values, atol=2e-13)
