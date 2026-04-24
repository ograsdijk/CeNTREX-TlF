import numpy as np
import pytest

from centrex_tlf import hamiltonian, states
from centrex_tlf.hamiltonian.utils import reduced_basis_hamiltonian


class TestGenerateHamiltonianEdgeCases:
    def test_coupled_hamiltonian_rejects_uncoupled_states(self):
        ubs = states.UncoupledBasisState(
            J=1, mJ=0, I1=0.5, m1=0.5, I2=0.5, m2=0.5,
            Omega=0, P=1, electronic_state=states.ElectronicState.X,
        )
        with pytest.raises(TypeError, match="CoupledBasisStates"):
            hamiltonian.generate_hamiltonian.generate_coupled_hamiltonian_B([ubs])

    def test_uncoupled_hamiltonian_rejects_coupled_states(self):
        cbs = states.CoupledBasisState(
            F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5,
            Omega=0, P=1, electronic_state=states.ElectronicState.X,
        )
        with pytest.raises(TypeError, match="UncoupledBasisStates"):
            hamiltonian.generate_hamiltonian.generate_uncoupled_hamiltonian_X([cbs])
