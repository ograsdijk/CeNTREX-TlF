from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.integrate as si
import scipy.linalg as la
import scipy.sparse as sp
import sympy as smp

from benchmarks import rc_propagation_methods as methods
from centrex_tlf import states
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout


def test_phase_frame_problem_derives_orientation_and_strips_phase(monkeypatch) -> None:
    qn = [
        SimpleNamespace(largest=SimpleNamespace(electronic_state=states.ElectronicState.X)),
        SimpleNamespace(largest=SimpleNamespace(electronic_state=states.ElectronicState.B)),
    ]
    omega_symbol = smp.Symbol("Ω0", real=True)
    ground = qn[0]
    excited = qn[1]
    couplings = [SimpleNamespace(ground_main=ground, excited_main=excited)]
    collapse = np.zeros((1, 2, 2), dtype=np.complex128)
    collapse[0, 0, 1] = 1.0
    system = SimpleNamespace(
        QN=qn,
        coupling_symbols=[omega_symbol],
        couplings=couplings,
        C_array=collapse,
        H_symbolic=smp.Matrix([[0, omega_symbol], [omega_symbol, 0]]),
        system=None,
        dissipator=None,
    )
    params = LindbladParameters(
        {"Ωt0": 2.0, "β0": 0.5, "ωphase0": 4.0},
        {"Ω0": "Ωt0*phase_modulation(t, β0, ωphase0)"},
    )
    prepared = SimpleNamespace(obe_system=system, parameters=params, backend="rust")
    captured: dict[str, object] = {}

    def fake_prepare(transformed, bound, **kwargs):
        captured["system"] = transformed
        captured["parameters"] = bound
        captured["kwargs"] = kwargs
        return "prepared"

    monkeypatch.setattr(methods, "prepare_lindblad_problem", fake_prepare)
    result, phase = methods.phase_frame_problem(prepared)

    assert result == "prepared"
    assert phase.diagonal_sign == -1  # ground row precedes excited column
    assert phase.excited_indices == (1,)
    frame_system = captured["system"]
    assert frame_system.H_symbolic[1, 1] == -2.0 * smp.cos(4.0 * smp.Symbol("t", real=True))
    assert "phase_modulation" not in str(captured["parameters"].compound_parameters["Ω0"])
    assert frame_system.C_array is system.C_array


def test_phase_frame_rejects_unmodulated_optical_coupling() -> None:
    qn = [
        SimpleNamespace(largest=SimpleNamespace(electronic_state=states.ElectronicState.X)),
        SimpleNamespace(largest=SimpleNamespace(electronic_state=states.ElectronicState.B)),
    ]
    omega, extra = smp.symbols("Ω0 Ω1", real=True)
    system = SimpleNamespace(
        QN=qn,
        coupling_symbols=[omega, extra],
        couplings=[SimpleNamespace(ground_main=qn[0], excited_main=qn[1])],
        C_array=np.zeros((0, 2, 2), dtype=complex),
        H_symbolic=smp.Matrix([[0, omega + extra], [omega + extra, 0]]),
    )
    params = LindbladParameters(
        {"Ωt0": 2.0, "β0": 0.5, "ωphase0": 4.0},
        {"Ω0": "Ωt0*phase_modulation(t, β0, ωphase0)"},
    )
    with pytest.raises(ValueError, match="every X-B optical coupling"):
        methods.infer_eom_phase(system, params)


def test_common_reachability_closes_union_across_switch_generators() -> None:
    layout = PackedHermitianLayout(3)
    dimension = layout.packed_len
    first = sp.csc_matrix(([1.0], ([1], [0])), shape=(dimension, dimension))
    second = sp.csc_matrix(([1.0], ([2], [1])), shape=(dimension, dimension))
    zero = sp.csc_matrix((dimension, dimension))
    phase = methods.EOMPhase(1.0, 2.0, (2,), ("Ω0",), -1)
    initial = np.zeros(dimension)
    initial[0] = 1.0

    first_only, _ = methods.common_reachable_coordinates([first], initial, layout, phase)
    second_only, _ = methods.common_reachable_coordinates([second], initial, layout, phase)
    common, _ = methods.common_reachable_coordinates(
        [first, second, zero, zero], initial, layout, phase
    )
    # The second switch state acts on coordinate 1, reached only in the first.
    assert 2 not in first_only
    assert 2 not in second_only
    assert {0, 1, 2}.issubset(set(common))


def test_common_reachability_pairs_phase_rotated_real_and_imaginary_parts() -> None:
    layout = PackedHermitianLayout(2)
    dimension = layout.packed_len
    zero = sp.csc_matrix((dimension, dimension))
    phase = methods.EOMPhase(0.7, 3.0, (1,), ("Ω0",), -1)
    initial = np.zeros(dimension)
    real = layout.upper_real_index(0, 1)
    imag = layout.upper_imag_index(0, 1)
    initial[real] = 1.0

    reachable, sinks = methods.common_reachable_coordinates(
        [zero, zero, zero, zero], initial, layout, phase
    )
    assert {real, imag}.issubset(set(reachable))
    assert real not in sinks
    assert imag not in sinks


def test_phase_frame_density_round_trip_and_populations() -> None:
    phase = methods.EOMPhase(1.2, 3.0, (1,), ("Ω0",), -1)
    rho = np.array([[0.4, 0.1 + 0.2j], [0.1 - 0.2j, 0.6]], dtype=complex)
    transformed = methods.phase_frame_density(rho, 0.37, phase)
    restored = methods.original_frame_density(transformed, 0.37, phase)
    np.testing.assert_allclose(restored, rho, atol=2e-15)
    np.testing.assert_allclose(np.diag(transformed), np.diag(rho), atol=2e-15)


def test_phase_frame_ode_matches_dissipative_eom_at_nonzero_entry() -> None:
    phase = methods.EOMPhase(0.7, 5.0, (1,), ("Ω0",), -1)
    h0 = np.array([[0.0, 0.6], [0.6, 0.2]], dtype=complex)
    rho0 = np.array([[0.65, 0.08 + 0.11j], [0.08 - 0.11j, 0.35]], dtype=complex)
    layout = PackedHermitianLayout(2)
    excited_projector = np.diag([0.0, 1.0])
    gamma = 0.37
    collapse = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    methods.validate_jump_frame(collapse[None, ...], phase)
    t0, t1 = 0.23, 1.19

    def integrate(initial: np.ndarray, rhs_hamiltonian: Callable[[float], np.ndarray]) -> np.ndarray:
        def rhs(time_s: float, packed: np.ndarray) -> np.ndarray:
            rho = layout.unpack(packed)
            h = rhs_hamiltonian(time_s)
            cdc = collapse.conj().T @ collapse
            dissipator = collapse @ rho @ collapse.conj().T - 0.5 * (
                cdc @ rho + rho @ cdc
            )
            return layout.pack(-1j * (h @ rho - rho @ h) + dissipator)

        solution = si.solve_ivp(
            rhs, (t0, t1), layout.pack(initial), method="DOP853", rtol=2e-12, atol=2e-14
        )
        return layout.unpack(solution.y[:, -1])

    def original_hamiltonian(time_s: float) -> np.ndarray:
        phi = phase.phi(time_s)
        h = h0.copy()
        h[0, 1] *= np.exp(1j * phi)
        h[1, 0] = np.conjugate(h[0, 1])
        return h

    def frame_hamiltonian(time_s: float) -> np.ndarray:
        h = h0.copy()
        shift = phase.diagonal_sign * phase.beta * phase.omega_rad_s * np.cos(
            phase.omega_rad_s * time_s
        )
        return h + shift * excited_projector

    rho_original_end = integrate(rho0, original_hamiltonian)
    rho_frame_end = integrate(methods.phase_frame_density(rho0, t0, phase), frame_hamiltonian)
    np.testing.assert_allclose(
        rho_frame_end,
        methods.phase_frame_density(rho_original_end, t1, phase),
        atol=2e-10,
        rtol=2e-10,
    )


class _StaticUnitaryFactor:
    def __init__(self, hamiltonian: np.ndarray):
        self.hamiltonian = hamiltonian

    def density_matrices(self, rho_batch: np.ndarray, times: list[float]) -> np.ndarray:
        duration = float(times[0])
        unitary = la.expm(-1j * self.hamiltonian * duration)
        return np.asarray([[unitary @ rho_batch[0] @ unitary.conj().T]])


def _phase_frame_reference(
    rho0: np.ndarray,
    t0: float,
    t1: float,
    h_static: np.ndarray,
    phase: methods.EOMPhase,
) -> np.ndarray:
    rho0_frame = methods.phase_frame_density(rho0, t0, phase)
    layout = PackedHermitianLayout(2)
    projector = np.diag([0.0, 1.0])

    def rhs(time_s: float, packed: np.ndarray) -> np.ndarray:
        rho = layout.unpack(packed)
        h = h_static + phase.diagonal_sign * phase.beta * phase.omega_rad_s * np.cos(
            phase.omega_rad_s * time_s
        ) * projector
        derivative = -1j * (h @ rho - rho @ h)
        return layout.pack(derivative)

    solution = si.solve_ivp(
        rhs,
        (t0, t1),
        layout.pack(rho0_frame),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    return methods.original_frame_density(layout.unpack(solution.y[:, -1]), t1, phase)


def test_strang_split_converges_when_phase_step_is_halved() -> None:
    h_static = np.array([[0.0, 0.8], [0.8, 0.3]], dtype=complex)
    phase = methods.EOMPhase(0.9, 8.0, (1,), ("Ω0",), -1)
    rho0 = np.array([[0.7, 0.1j], [-0.1j, 0.3]], dtype=complex)
    layout = PackedHermitianLayout(2)
    factors = {(a, b): _StaticUnitaryFactor(h_static) for a in (0, 1) for b in (0, 1)}
    split = methods.StaticSplit(layout, factors, phase, 0, {})
    exact = _phase_frame_reference(rho0, 0.17, 1.17, h_static, phase)

    coarse = split.propagate(
        rho0, (0.17, 1.17), switch_values=lambda _t: (0, 0),
        switch_boundaries=[0.17, 1.17], dt=0.02,
    )
    fine = split.propagate(
        rho0, (0.17, 1.17), switch_values=lambda _t: (0, 0),
        switch_boundaries=[0.17, 1.17], dt=0.01,
    )
    coarse_error = np.linalg.norm(coarse - exact)
    fine_error = np.linalg.norm(fine - exact)
    assert fine_error < coarse_error / 3.0


def test_sparse_expm_factor_reuses_generator_and_integrates_photons() -> None:
    layout = PackedHermitianLayout(2)
    gamma = 2.5
    generator = sp.csr_matrix(
        np.array(
            [
                [0.0, gamma, 0.0, 0.0],
                [0.0, -gamma, 0.0, 0.0],
                [0.0, 0.0, -gamma / 2.0, 0.0],
                [0.0, 0.0, 0.0, -gamma / 2.0],
            ]
        )
    )
    factor = methods.SparseExpmFactor(layout, generator, 0)
    rho0 = np.diag([0.25, 0.75]).astype(complex)
    duration = 0.6
    evolved = factor.density_matrices(rho0[None, ...], [duration])[0, 0]
    expected_excited = 0.75 * np.exp(-gamma * duration)
    assert evolved[1, 1].real == pytest.approx(expected_excited, rel=2e-13)
    assert evolved[0, 0].real == pytest.approx(1.0 - expected_excited, rel=2e-13)
    photons = factor.evaluate(
        rho0[None, ...], [duration], output="photon_integral", integral_weights=[(1, gamma)]
    )[0, 0, 0]
    assert photons == pytest.approx(0.75 - expected_excited, rel=2e-13)


def test_reduced_sparse_factor_reconstructs_sink_population_and_photons() -> None:
    layout = PackedHermitianLayout(3)
    gamma = 1.7
    generator = sp.csr_matrix(
        (
            [gamma, -gamma],
            ([0, 1], [1, 1]),
        ),
        shape=(layout.packed_len, layout.packed_len),
    )
    keep = np.array([0, 1, 2])
    full = methods.SparseExpmFactor(layout, generator, 0)
    reduced = methods.SparseExpmFactor(layout, generator, 0, keep)
    rho0 = np.diag([0.2, 0.6, 0.2]).astype(complex)
    duration = 0.4

    full_rho = full.density_matrices(rho0[None, ...], [duration])[0, 0]
    reduced_rho = reduced.density_matrices(rho0[None, ...], [duration])[0, 0]
    np.testing.assert_allclose(reduced_rho, full_rho, atol=2e-14)
    assert reduced_rho[2, 2] == pytest.approx(0.2)

    weights = [(1, gamma)]
    full_yield = full.evaluate(
        rho0[None, ...], [duration], output="photon_integral", integral_weights=weights
    )[0, 0, 0]
    reduced_yield = reduced.evaluate(
        rho0[None, ...], [duration], output="photon_integral", integral_weights=weights
    )[0, 0, 0]
    assert reduced_yield == pytest.approx(full_yield, abs=2e-14)
    assert reduced_yield == pytest.approx(0.6 * (1.0 - np.exp(-gamma * duration)))


def test_reduced_sparse_factor_rejects_excluded_input_support() -> None:
    layout = PackedHermitianLayout(2)
    generator = sp.csr_matrix((layout.packed_len, layout.packed_len))
    factor = methods.SparseExpmFactor(layout, generator, 0, np.array([0, 1]))
    rho0 = np.zeros((1, 2, 2), dtype=complex)
    rho0[0, 0, 1] = 0.1
    rho0[0, 1, 0] = 0.1
    with pytest.raises(ValueError, match="outside keep_coordinates"):
        factor.density_matrices(rho0, [0.5])


def test_periodic_map_gate_prevents_unaffordable_basis_actions() -> None:
    layout = PackedHermitianLayout(3)
    action_called = False

    def action(_rho: np.ndarray) -> np.ndarray:
        nonlocal action_called
        action_called = True
        raise AssertionError("gated action must not run")

    mapping, gate = methods.build_periodic_map(
        layout, action, 5e-6, max_bytes=10_000, max_basis_actions=4
    )
    assert mapping is None
    assert not gate.feasible
    assert not action_called


def test_periodic_map_reconstructs_full_density_action() -> None:
    layout = PackedHermitianLayout(2)
    unitary = la.expm(-1j * np.array([[0.0, 0.3], [0.3, 0.2]]) * 0.1)

    def action(rho: np.ndarray) -> np.ndarray:
        return unitary @ rho @ unitary.conj().T

    mapping, gate = methods.build_periodic_map(
        layout,
        action,
        0.1,
        max_bytes=100_000,
        max_basis_actions=4,
        entry_time_s=0.23,
    )
    assert gate.feasible and mapping is not None
    assert mapping.entry_time_s == pytest.approx(0.23)
    rho = np.array([[0.6, 0.15 + 0.1j], [0.15 - 0.1j, 0.4]])
    np.testing.assert_allclose(mapping.apply(rho), action(rho), atol=2e-14)
    np.testing.assert_allclose(mapping.repeat(rho, 2), action(action(rho)), atol=3e-14)


def test_periodic_map_repeats_batches_and_accumulates_linear_observable() -> None:
    layout = PackedHermitianLayout(2)

    def action(rho: np.ndarray) -> tuple[np.ndarray, float]:
        return rho.copy(), float(np.real(rho[1, 1]))

    mapping, gate = methods.build_periodic_map(
        layout, action, 1.0, max_bytes=100_000, max_basis_actions=4
    )
    assert gate.feasible and mapping is not None
    rho_a = np.diag([0.8, 0.2]).astype(complex)
    rho_b = np.diag([0.1, 0.9]).astype(complex)
    result, counts = mapping.repeat(np.stack([rho_a, rho_b]), 3, return_integral=True)
    np.testing.assert_allclose(result, np.stack([rho_a, rho_b]), atol=1e-14)
    np.testing.assert_allclose(counts, [0.6, 2.7], atol=1e-14)


def test_dense_split_memory_gate_runs_before_factorization(monkeypatch) -> None:
    class Prepared:
        def __init__(self, n: int):
            self.layout = PackedHermitianLayout(n)
            self.obe_system = SimpleNamespace(QN=list(range(n)), C_array=np.zeros((0, n, n)))

    plans = {(0, 0): Prepared(70), (0, 1): Prepared(70), (1, 0): Prepared(70), (1, 1): Prepared(70)}
    phase = methods.EOMPhase(1.0, 1.0, (1,), ("Ω0",), -1)
    called = False

    def factor(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("memory gate must run before factorization")

    monkeypatch.setattr(methods, "prepare_dense_lindblad_propagator", factor)
    with pytest.raises(MemoryError, match="exceeds max_total_bytes"):
        methods.prepare_static_split(plans, phase, max_total_bytes=100)
    assert not called
