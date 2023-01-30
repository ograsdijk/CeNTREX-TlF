from pathlib import Path

import numpy as np

from centrex_tlf import couplings, states


def test_collapse_matrices():
    qn_select = states.QuantumSelector(J=1)
    ground_states = states.generate_coupled_states_X(qn_select)

    qn_select = states.QuantumSelector(J=1, F1=1 / 2, F=1, P=1, Ω=1)
    excited_states = states.generate_coupled_states_B(qn_select)

    QN = list(1 * np.append(ground_states, excited_states))
    ground_states = [1 * s for s in ground_states]
    excited_states = [1 * s for s in excited_states]

    C_array = couplings.collapse_matrices(
        QN, ground_states, excited_states, gamma=1.56e6
    )

    C_test = np.load(Path(__file__).parent / "collapse_matrices_test.npy")
    assert np.allclose(C_array, C_test)
