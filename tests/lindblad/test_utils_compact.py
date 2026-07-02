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


def test_generate_obe_system_transitions_retains_opposite_parity_levels():
    trans = transitions.Q1_F1_1o2_F0
    transition_selectors = couplings.generate_transition_selectors(
        transitions=[trans],
        polarizations=[[couplings.polarization_X]],
    )

    default = lindblad.generate_OBE_system_transitions(
        [trans],
        transition_selectors,
        E=np.array([0.0, 0.0, 200.0]),
        method="matrix",
    )
    retained = lindblad.generate_OBE_system_transitions(
        [trans],
        transition_selectors,
        E=np.array([0.0, 0.0, 200.0]),
        retain_opposite_parity_levels=True,
        method="matrix",
    )
    retained_from_setup = lindblad.setup_OBE_system_transitions(
        [trans],
        transition_selectors,
        E=np.array([0.0, 0.0, 200.0]),
        retain_opposite_parity_levels=True,
        method="matrix",
    )

    default_parities = {
        state.largest.P
        for state in default.excited
        if state.largest.J == trans.J_excited
        and state.largest.F1 == trans.F1_excited
        and state.largest.F == trans.F_excited
    }
    retained_parities = {
        state.largest.P
        for state in retained.excited
        if state.largest.J == trans.J_excited
        and state.largest.F1 == trans.F1_excited
        and state.largest.F == trans.F_excited
    }
    initial_idx = min(
        [
            idx
            for idx, state in enumerate(retained.QN)
            if state.largest.electronic_state == states.ElectronicState.X
            and state.largest.J == trans.J_ground
            and state.largest.F == 1
            and state.largest.mF == 1
        ],
        key=lambda idx: np.real(retained.H_int[idx, idx]),
    )
    addressed_idx = next(
        idx
        for idx, state in enumerate(retained.QN)
        if state.largest.electronic_state == states.ElectronicState.B
        and state.largest.J == trans.J_excited
        and state.largest.F1 == trans.F1_excited
        and state.largest.F == trans.F_excited
        and state.largest.mF == 0
        and state.largest.P == trans.P_excited
    )

    assert default_parities == {trans.P_excited}
    assert retained_parities == {trans.P_excited, -trans.P_excited}
    assert len(retained.excited) == len(default.excited) + 1
    assert len(retained_from_setup.excited) == len(retained.excited)
    assert abs(retained.couplings[0].fields[0].field[initial_idx, addressed_idx]) > 0.1
