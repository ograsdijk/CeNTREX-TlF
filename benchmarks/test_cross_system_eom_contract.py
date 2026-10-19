"""Lightweight tests for the phase-modulated benchmark contract (no ODEs)."""
from __future__ import annotations

import numpy as np
import sympy as smp

from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py, fill_hamiltonian_py
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem

try:
    from cross_system_eom_fixtures import (
        ENTRY_PHASE_S, EOM_FIXTURES, SCREEN_DURATION_S, evaluate_phase_factor,
        phase_parameters,
    )
except ModuleNotFoundError:
    from benchmarks.cross_system_eom_fixtures import (
        ENTRY_PHASE_S, EOM_FIXTURES, SCREEN_DURATION_S, evaluate_phase_factor,
        phase_parameters,
    )

try:
    import run_rc_cross_system_screen_v3 as screen
except ModuleNotFoundError:
    from benchmarks import run_rc_cross_system_screen_v3 as screen


def test_phase_modulation_is_complex_unit_modulus_at_absolute_entry_time() -> None:
    beta = 3.8
    omega = 2 * np.pi * 1.6e6
    params, expression = phase_parameters(rabi_rad_s=1.0, beta=beta, omega_rad_s=omega)
    assert complex(params.base_parameters["β0"]).real == beta
    evaluated = complex(expression.evaluate(t=ENTRY_PHASE_S))
    expected = evaluate_phase_factor(ENTRY_PHASE_S, beta=beta, omega_rad_s=omega)
    assert abs(evaluated - expected) < 1e-14
    assert abs(evaluated.imag) > 1e-3
    assert abs(abs(evaluated) - 1.0) < 1e-14
    # A Hermitian optical Hamiltonian uses both Ω and its complex conjugate.
    assert abs(evaluated * evaluated.conjugate() - 1.0) < 1e-14


def _two_level_phase_hamiltonian(beta: float, t: float) -> np.ndarray:
    omega_symbol = smp.Symbol("Ω", complex=True)
    delta_symbol = smp.Symbol("δ", real=True)
    h_symbolic = smp.Matrix([[0, omega_symbol / 2],
                             [smp.conjugate(omega_symbol) / 2, -delta_symbol]])
    collapse = np.zeros((1, 2, 2), dtype=np.complex128)
    collapse[0, 0, 1] = np.sqrt(0.3)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    system = OBESystem(
        ground=[], excited=[], QN=[], H_int=zeros, V_ref_int=zeros, couplings=[],
        H_symbolic=h_symbolic, C_array=collapse, system=None,
        coupling_symbols=[omega_symbol, delta_symbol], polarization_symbols=[],
    )
    params, expression = phase_parameters(rabi_rad_s=1.0, beta=beta,
                                          omega_rad_s=2 * np.pi * 1.6e6)
    params.bind(omega_symbol, expression, finalize=False)
    params.real(str(delta_symbol), 0.0)
    params._finalize()
    prepared = prepare_lindblad_problem(system, params, backend="python")
    slots = evaluate_parameter_graph_py(prepared.parameter_graph, t)
    return fill_hamiltonian_py(prepared.hamiltonian_plan, slots, t)


def test_phase_plan_keeps_conjugate_hamiltonian_and_zero_beta_control() -> None:
    t0, t1 = ENTRY_PHASE_S, ENTRY_PHASE_S + 100e-9
    h0 = _two_level_phase_hamiltonian(3.8, t0)
    h1 = _two_level_phase_hamiltonian(3.8, t1)
    np.testing.assert_allclose(h0, h0.conj().T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(h1, h1.conj().T, rtol=0, atol=1e-14)
    assert abs(h1[0, 1] - h0[0, 1]) > 1e-3
    h_static = _two_level_phase_hamiltonian(0.0, t0)
    np.testing.assert_allclose(h_static[0, 1], 0.5 + 0j, rtol=0, atol=1e-14)


def test_eom_screen_catalog_is_231_cells_with_frozen_basis_fixtures() -> None:
    assert len(EOM_FIXTURES) == 3
    assert SCREEN_DURATION_S == 1e-6
    assert len(screen.KERNELS) == 11
    assert len(screen.PRODUCTS) == 7
    assert len(EOM_FIXTURES) * len(screen.KERNELS) * len(screen.PRODUCTS) == 231
