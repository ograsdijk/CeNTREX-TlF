"""Exact zero-field degeneracy must not mix the chosen mF sectors."""

import numpy as np
import pytest

from centrex_tlf import hamiltonian, states


def test_block_eigenvectors_preserve_degenerate_labels():
    matrix = np.array([[2.0, 0, 0.2, 0], [0, 2.0, 0, 0.2], [0.2, 0, 3.0, 0], [0, 0.2, 0, 3.0]])
    labels = np.array([-1, 1, -1, 1])
    result = hamiltonian.generate_diagonalized_hamiltonian(matrix, block_labels=labels)
    assert np.max(abs(result.V[labels[:, None] != labels[None, :]])) == 0
    np.testing.assert_allclose(np.sort(np.diag(result.H).real), np.linalg.eigvalsh(matrix))
    np.testing.assert_allclose(result.V.conj().T @ result.V, np.eye(4), atol=1e-14)
    with pytest.raises(ValueError, match="couples"):
        hamiltonian.generate_diagonalized_hamiltonian(matrix, block_labels=[0, 0, 1, 1])


@pytest.mark.parametrize("electronic", ["X", "B"])
def test_physical_zero_field_mf_blocks(electronic):
    if electronic == "X":
        qn = states.generate_coupled_states_X(states.QuantumSelector(J=[0, 1, 2]))
        result = hamiltonian.generate_reduced_X_hamiltonian(
            qn, E=np.zeros(3), B=np.zeros(3), Jmax=2
        )
    else:
        qn = states.generate_coupled_states_B(states.QuantumSelector(J=[1, 2], P=[-1, 1], Ω=1))
        result = hamiltonian.generate_reduced_B_hamiltonian(
            qn, E=np.zeros(3), B=np.zeros(3), Jmax=2
        )
    labels = np.array([s.mF for s in result.QN_construct])
    assert np.max(abs(result.V[labels[:, None] != labels[None, :]])) < 1e-12
    np.testing.assert_allclose(result.V.conj().T @ result.V, np.eye(len(labels)), atol=1e-12)
