import numpy as np
import pytest
import sympy as smp

from centrex_tlf.lindblad.generate_hamiltonian import generate_symbolic_hamiltonian
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.utils_compact import compact_symbolic_hamiltonian_indices


class TestGenerateSymbolicHamiltonianEdgeCases:
    def test_off_diagonal_H_int_raises(self):
        H_int = np.array([[1, 0.5], [0.5, 2]], dtype=complex)
        with pytest.raises(ValueError, match="off-diagonal"):
            generate_symbolic_hamiltonian(H_int, [], [], [])


class TestCompactSymbolicHamiltonianEdgeCases:
    def test_coupled_states_raise_value_error(self):
        hamiltonian = smp.ones(5)
        indices = np.array([2, 3])
        with pytest.raises(ValueError):
            compact_symbolic_hamiltonian_indices(hamiltonian, indices)


class TestSolveLindbladEdgeCases:
    def test_invalid_solver_raises(self):
        with pytest.raises(NotImplementedError, match="supported solvers"):
            solve_lindblad(
                None, np.eye(2, dtype=complex), (0.0, 1.0),
                solver="invalid_solver",
            )

    def test_invalid_execution_mode_raises(self):
        with pytest.raises(NotImplementedError, match="execution_mode"):
            solve_lindblad(
                None, np.eye(2, dtype=complex), (0.0, 1.0),
                execution_mode="invalid_mode",
            )

    def test_invalid_output_raises(self):
        with pytest.raises(NotImplementedError, match="output"):
            solve_lindblad(
                None, np.eye(2, dtype=complex), (0.0, 1.0),
                output="invalid_output",
            )

    def test_invalid_output_when_raises(self):
        with pytest.raises(NotImplementedError, match="output_when"):
            solve_lindblad(
                None, np.eye(2, dtype=complex), (0.0, 1.0),
                output_when="invalid",
            )
