"""Independent complex-matrix oracle for the native adjoint response."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as smp
from scipy.linalg import expm

from centrex_tlf import lindblad
from centrex_tlf.lindblad.utils_setup import OBESystem


def test_complex_three_level_adjoint_against_matrix_exponential():
    h = np.array([[0.2, 0.3 + 0.2j, -0.1j],
                  [0.3 - 0.2j, -0.4, 0.17 + 0.08j],
                  [0.1j, 0.17 - 0.08j, 0.6]], dtype=complex)
    collapse = np.zeros((3, 3, 3), dtype=complex)
    collapse[0, 0, 1] = np.sqrt(0.7)
    collapse[1, 1, 2] = np.sqrt(0.4)
    collapse[2, 2, 2] = np.sqrt(0.2)  # Single-projector dephasing.
    zeros = np.zeros_like(h)
    system = OBESystem([], [], [], zeros, zeros, [], smp.Matrix(h), collapse,
                       None, [], [])
    prepared = lindblad.prepare_lindblad_problem(
        system, lindblad.LindbladParameters(), backend="rust"
    )
    generator = np.empty((9, 9), dtype=complex)
    for column in range(9):
        basis = np.zeros((3, 3), dtype=complex)
        basis.flat[column] = 1
        derivative = -1j * (h @ basis - basis @ h)
        for c in collapse:
            cc = c.conj().T @ c
            derivative += c @ basis @ c.conj().T - (cc @ basis + basis @ cc) / 2
        generator[:, column] = derivative.ravel()

    observable = np.array([[0.1, 0.2j, 0.12],
                           [-0.2j, 0.4, 0.06 - 0.09j],
                           [0.12, 0.06 + 0.09j, 0.8]])
    duration = 0.73
    expected = (expm(generator.conj().T * duration) @ observable.ravel()).reshape(3, 3)
    response = lindblad.prepare_lindblad_adjoint_response(
        prepared, (-0.2, duration - 0.2),
        terminal_observables={"coherent": observable},
        solver="tsit5", dt=1e-3, reltol=1e-11, abstol=1e-13,
    )
    np.testing.assert_allclose(response.responses["coherent"], expected,
                               rtol=1e-9, atol=2e-11)

    # A diagonal-only response or the wrong packed off-diagonal metric fails this.
    pure = np.array([1, 0.3j, -0.4 + 0.2j], dtype=complex)
    pure /= np.linalg.norm(pure)
    rho = np.outer(pure, pure.conj())
    final = (expm(generator * duration) @ rho.ravel()).reshape(3, 3)
    value = np.trace(observable @ final).real
    assert response.evaluate(rho, "coherent") == pytest.approx(value, abs=2e-11)
    values = response.evaluate(np.stack([rho, np.eye(3) / 3]), "coherent")
    assert values.shape == (2,)
    assert values[0] == pytest.approx(value, abs=2e-11)
