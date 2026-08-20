from pathlib import Path

import numpy as np
import pytest

from centrex_tlf import couplings, states


def _two_level_setup():
    qn_select = states.QuantumSelector(J=1)
    ground_states = states.generate_coupled_states_X(qn_select)

    qn_select = states.QuantumSelector(J=1, F1=1 / 2, F=1, P=1, Ω=1)
    excited_states = states.generate_coupled_states_B(qn_select)

    QN = list(1 * np.append(ground_states, excited_states))
    ground_states = [1 * s for s in ground_states]
    excited_states = [1 * s for s in excited_states]
    return QN, ground_states, excited_states


def test_collapse_matrices():
    QN, ground_states, excited_states = _two_level_setup()

    # Arbitrary decay rate used only to exercise the collapse-matrix algebra
    # against the pinned golden array below; not the physical TlF
    # BConstants.Γ molecular constant.
    decay_rate = 1.56e6
    C_array = couplings.collapse_matrices(QN, ground_states, excited_states, decay_rate=decay_rate)

    C_test = np.load(Path(__file__).parent / "collapse_matrices_test.npy")
    assert np.allclose(C_array, C_test)


def test_collapse_matrices_scale_as_sqrt_br_times_decay_rate():
    QN, ground_states, excited_states = _two_level_setup()

    decay_rate = 1.56e6
    C_array = couplings.collapse_matrices(QN, ground_states, excited_states, decay_rate=decay_rate)
    # Doubling the decay rate must scale every matrix element by sqrt(2).
    C_array_2x = couplings.collapse_matrices(
        QN, ground_states, excited_states, decay_rate=2 * decay_rate
    )
    np.testing.assert_allclose(C_array_2x, C_array * np.sqrt(2))


def test_collapse_matrices_deprecated_gamma_matches_decay_rate():
    QN, ground_states, excited_states = _two_level_setup()
    decay_rate = 1.56e6

    C_expected = couplings.collapse_matrices(
        QN, ground_states, excited_states, decay_rate=decay_rate
    )
    with pytest.deprecated_call():
        C_via_gamma = couplings.collapse_matrices(
            QN, ground_states, excited_states, gamma=decay_rate
        )
    np.testing.assert_allclose(C_via_gamma, C_expected)


def test_collapse_matrices_both_decay_rate_and_gamma_raises():
    QN, ground_states, excited_states = _two_level_setup()

    with pytest.raises(TypeError):
        couplings.collapse_matrices(
            QN, ground_states, excited_states, decay_rate=1.56e6, gamma=1.56e6
        )
