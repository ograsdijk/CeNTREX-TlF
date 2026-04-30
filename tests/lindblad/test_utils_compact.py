import pickle
from pathlib import Path

import numpy as np
import pytest
import sympy as smp

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions


def test_generate_qn_compact():
    trans = [
        transitions.OpticalTransition(transitions.OpticalTransitionType.R, 0, 3 / 2, 1)
    ]
    H_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(trans)
    qn_compact = lindblad.utils_compact.generate_qn_compact(trans, H_reduced)
    assert qn_compact == [
        states.QuantumSelector(J=2, electronic=states.ElectronicState.X)
    ]


def test_compact_symbolic_hamiltonian_indices():
    hamiltonian = smp.zeros(5)
    indices = np.array([2, 3])
    arr = lindblad.utils_compact.compact_symbolic_hamiltonian_indices(
        hamiltonian, indices
    )
    assert arr.shape == (4, 4)

    hamiltonian = smp.ones(5)
    indices = np.array([2, 3])
    with pytest.raises(ValueError):
        arr = lindblad.utils_compact.compact_symbolic_hamiltonian_indices(
            hamiltonian, indices
        )


def test_generate_obe_system_transitions_with_compact_selector_over_multiple_js():
    trans = transitions.R0_F1_1o2_F1
    transition_selectors = couplings.generate_transition_selectors(
        transitions=[trans],
        polarizations=[[couplings.polarization_Z]],
    )

    system = lindblad.generate_OBE_system_transitions(
        [trans],
        transition_selectors,
        qn_compact=states.QuantumSelector(
            J=[1, 2, 3], electronic=states.ElectronicState.X
        ),
        method="matrix",
    )

    assert system.QN_original is not None
    assert len(system.QN) < len(system.QN_original)
    assert any(state.largest.F1 is None for state in system.QN)
