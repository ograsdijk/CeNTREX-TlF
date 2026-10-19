from __future__ import annotations

import numpy as np
import pytest
import sympy as smp

from centrex_tlf import centrex_tlf_rust
from centrex_tlf.lindblad.adjoint import (
    population_observable,
    prepare_lindblad_adjoint_response,
    prepare_lindblad_adjoint_response_batch,
)
from centrex_tlf.lindblad.parameters import (
    LindbladParameters,
    Time,
    phase_modulation,
    square_wave,
)
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.utils_setup import OBESystem


def _system() -> OBESystem:
    omega, delta = smp.symbols("omega delta", real=True)
    hamiltonian = smp.Matrix([[0, omega / 2], [smp.conjugate(omega) / 2, -delta]])
    collapse = np.zeros((1, 2, 2), dtype=np.complex128)
    collapse[0, 0, 1] = np.sqrt(0.3)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    return OBESystem(
        ground=[], excited=[], QN=[], H_int=zeros, V_ref_int=zeros, couplings=[],
        H_symbolic=hamiltonian, C_array=collapse, system=None,
        coupling_symbols=[omega, delta], polarization_symbols=[],
    )


def _prepared(omega: float = 0.0, delta: float = 0.0):
    system = _system()
    params = LindbladParameters()
    omega0 = params.real("omega0", omega)
    delta0 = params.real("delta0", delta)
    params.bind(system.coupling_symbols[0], omega0, finalize=False)
    params.bind(system.coupling_symbols[1], delta0)
    return prepare_lindblad_problem(system, params, backend="rust")


def _coherent_rho() -> np.ndarray:
    rho = np.array([[0.65, 0.12 + 0.18j], [0.12 - 0.18j, 0.35]], dtype=np.complex128)
    assert np.linalg.eigvalsh(rho).min() > 0
    return rho


def _forward_final(prepared, rho: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    result = solve_lindblad(
        prepared, rho, span, solver="dopri5", execution_mode="structured",
        output="full", output_when="final", dense_output=False,
        dt=1e-3, abstol=1e-12, reltol=1e-10,
    )
    return result.density_matrices()[-1]


def test_terminal_adjoint_matches_forward_for_coherent_input() -> None:
    prepared = _prepared(omega=0.8, delta=0.23)
    terminal = np.array([[0.25, 0.18 + 0.11j], [0.18 - 0.11j, 0.75]], dtype=np.complex128)
    rho = _coherent_rho()
    response = prepare_lindblad_adjoint_response(
        prepared, (0.0, 0.8), terminal_observables={"terminal": terminal},
        solver="tsit5", dt=1e-3, abstol=1e-12, reltol=1e-10,
    )
    predicted = response.evaluate(rho, "terminal")
    actual = np.trace(terminal @ _forward_final(prepared, rho, (0.0, 0.8))).real
    assert predicted == pytest.approx(actual, rel=2e-8, abs=2e-10)
    assert response.responses["terminal"].shape == (2, 2)
    assert response.basis_responses("terminal").shape == (2,)


def test_photon_running_objective_matches_analytic_decay_and_forward() -> None:
    prepared = _prepared()
    rho = np.diag([0.0, 1.0]).astype(np.complex128)
    duration = 1.3
    response = prepare_lindblad_adjoint_response(
        prepared, (0.0, duration), integral_weights=[(1, 0.3)],
        abstol=1e-12, reltol=1e-10, dt=1e-3,
    )
    expected = 1.0 - np.exp(-0.3 * duration)
    assert response.evaluate(rho, "integral") == pytest.approx(expected, rel=2e-8)
    forward = solve_lindblad(
        prepared, rho, (0.0, duration), solver="dopri5", execution_mode="structured",
        output="photon_integral", output_when="final", integral_weights=[(1, 0.3)],
        dense_output=False, dt=1e-3, abstol=1e-12, reltol=1e-10,
    )
    assert float(np.asarray(forward.values).reshape(-1)[-1]) == pytest.approx(expected, rel=2e-8)


def test_switched_drive_uses_absolute_time_and_exact_split() -> None:
    system = _system()
    omega_symbol, delta_symbol = system.coupling_symbols
    params = LindbladParameters()
    omega0 = params.real("omega0", 0.9)
    frequency = params.real("switch_frequency", 2 * np.pi)
    phase = params.real("phase", 0.0)
    params.bind(omega_symbol, omega0 * square_wave(Time(), frequency, phase), finalize=False)
    params.bind(delta_symbol, 0.17)
    prepared = prepare_lindblad_problem(system, params, backend="rust")
    terminal = np.array([[0.7, 0.1 - 0.16j], [0.1 + 0.16j, 0.3]], dtype=np.complex128)
    rho = _coherent_rho()
    response = prepare_lindblad_adjoint_response(
        prepared, (0.0, 1.0), terminal_observables={"p": terminal},
        discontinuities=[0.5], dt=1e-3, abstol=1e-12, reltol=1e-10,
    )
    state = rho
    for span in ((0.0, 0.5), (0.5, 1.0)):
        state = _forward_final(prepared, state, span)
    assert response.evaluate(rho, "p") == pytest.approx(np.trace(terminal @ state).real, rel=3e-8, abs=3e-10)


def test_smooth_complex_drive_at_nonzero_absolute_times() -> None:
    system = _system()
    omega_symbol, delta_symbol = system.coupling_symbols
    params = LindbladParameters()
    omega0 = params.real("omega0", 0.75)
    beta = params.real("beta", 0.65)
    modulation_frequency = params.real("modulation_frequency", 1.4)
    params.bind(
        omega_symbol,
        omega0 * phase_modulation(Time(), beta, modulation_frequency),
        finalize=False,
    )
    params.bind(delta_symbol, -0.13)
    prepared = prepare_lindblad_problem(system, params, backend="rust")
    terminal = np.array([[0.3, 0.2 + 0.09j], [0.2 - 0.09j, 0.7]], dtype=np.complex128)
    rho = _coherent_rho()
    span = (-0.23, 0.61)
    response = prepare_lindblad_adjoint_response(
        prepared, span, terminal_observables={"smooth": terminal},
        dt=1e-3, abstol=1e-12, reltol=1e-10,
    )
    final = _forward_final(prepared, rho, span)
    assert response.evaluate(rho, "smooth") == pytest.approx(
        np.trace(terminal @ final).real, rel=3e-8, abs=3e-10
    )


def test_parameter_point_parallel_order_and_validation() -> None:
    prepared = _prepared(omega=0.4)
    terminal = np.diag([1.0, 0.0]).astype(np.complex128)
    kwargs = dict(t_span=(0.0, 0.15), terminal_observables={"g": terminal}, dt=1e-3)
    points = [{"omega0": 0.2}, {"omega0": 0.6}]
    serial = prepare_lindblad_adjoint_response_batch(
        prepared, parameter_values_batch=points, parallel=False, **kwargs
    )
    parallel = prepare_lindblad_adjoint_response_batch(
        prepared, parameter_values_batch=points, parallel=True, threads=2, **kwargs
    )
    assert len(serial) == len(parallel) == 2
    for left, right in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(left.responses["g"], right.responses["g"], rtol=1e-11, atol=1e-12)
    assert not np.allclose(serial[0].responses["g"], serial[1].responses["g"])
    rho = _coherent_rho()
    reference = _prepared(omega=0.2)
    forward = _forward_final(reference, rho, (0.0, 0.15))
    assert serial[0].evaluate(rho, "g") == pytest.approx(
        np.trace(terminal @ forward).real, rel=3e-8, abs=3e-10
    )
    with pytest.raises(ValueError, match="Hermitian"):
        prepare_lindblad_adjoint_response(prepared, (0, 0.1), terminal_observables={"bad": np.ones((2, 2)) * 1j})
    with pytest.raises(ValueError, match="discontinuities"):
        prepare_lindblad_adjoint_response(prepared, (0, 0.1), terminal_observables={"g": terminal}, discontinuities=[0.1])
    with pytest.raises(ValueError, match="indices"):
        population_observable(2, [0.5])


def test_general_collapse_operators_are_rejected_explicitly() -> None:
    prepared = _prepared()
    payload = prepared.to_payload()
    collapse = np.asarray(payload["dense_c_array"]).copy()
    collapse[0, 0, 0] = 0.2
    payload["dense_c_array"] = collapse
    prepared.rust_plan = centrex_tlf_rust.prepare_lindblad_problem_py(payload)
    with pytest.raises(ValueError, match="general collapse adjoints are unsupported"):
        prepare_lindblad_adjoint_response(
            prepared, (0.0, 0.1), integral_weights=[(1, 0.3)]
        )
