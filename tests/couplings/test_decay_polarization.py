import numpy as np
import pytest

from centrex_tlf import couplings, hamiltonian, states


def cartesian_ground_triplet():
    excited = 1 * next(
        iter(states.generate_coupled_states_B(states.QuantumSelector(J=1, F1=0.5, F=0, mF=0, P=1)))
    )
    ground = list(states.generate_coupled_states_X(states.QuantumSelector(J=1, F1=0.5, F=1)))
    plus = next(state for state in ground if state.mF == 1)
    minus = next(state for state in ground if state.mF == -1)
    zero = next(state for state in ground if state.mF == 0)
    ground_x = states.CoupledState([(1 / np.sqrt(2), plus), (-1 / np.sqrt(2), minus)])
    ground_y = states.CoupledState([(1 / np.sqrt(2), plus), (1 / np.sqrt(2), minus)])
    return [ground_x, ground_y, 1 * zero], excited


def rotate_z(state, angle):
    return states.CoupledState(
        [(amplitude * np.exp(-1j * basis.mF * angle), basis) for amplitude, basis in state]
    )


def test_cartesian_triplet_has_equal_decay_branches():
    ground, excited = cartesian_ground_triplet()
    np.testing.assert_allclose(
        couplings.calculate_br(excited, ground, tol=0), np.full(3, 1 / 3), atol=1e-14
    )


@pytest.mark.parametrize("angle", [np.pi / 6, np.pi / 2, np.pi])
def test_decay_branching_invariant_under_common_rotation(angle):
    ground, excited = cartesian_ground_triplet()
    expected = couplings.calculate_br(excited, ground, tol=0)
    actual = couplings.calculate_br(
        rotate_z(excited, angle), [rotate_z(state, angle) for state in ground], tol=0
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_collapse_rates_use_incoherent_polarization_sum():
    ground, excited = cartesian_ground_triplet()
    decay_rate = 2.4e6
    collapse = couplings.collapse_matrices(
        [*ground, excited], ground, [excited], decay_rate=decay_rate
    )
    rates = np.sum(abs(collapse[:, :3, 3]) ** 2, axis=0)
    np.testing.assert_allclose(rates, np.full(3, decay_rate / 3), atol=1e-8)
    np.testing.assert_allclose(rates.sum(), decay_rate, atol=1e-8)


def test_tilted_field_branching_matches_spherical_strength_sum():
    electric = np.array([0.0, 0.0, -170.0])
    magnetic = np.array([-0.01327, 0.19433, -0.48889])
    ground_labels = list(states.generate_coupled_states_X(states.QuantumSelector(J=2)))
    excited_labels = list(
        states.generate_coupled_states_B(states.QuantumSelector(J=3, F1=3.5, F=4, mF=0, P=1))
    )
    ground = hamiltonian.generate_reduced_X_hamiltonian(
        ground_labels, E=electric, B=magnetic
    ).QN_basis
    excited = hamiltonian.generate_reduced_B_hamiltonian(
        excited_labels, E=electric, B=magnetic
    ).QN_basis[0]
    spherical = np.array(
        [
            [-1 / np.sqrt(2), 1j / np.sqrt(2), 0],
            [0, 0, 1],
            [1 / np.sqrt(2), 1j / np.sqrt(2), 0],
        ],
        dtype=complex,
    )
    strengths = np.array(
        [
            sum(
                abs(
                    hamiltonian.generate_ED_ME_mixed_state(
                        state.remove_small_components(1e-3),
                        excited.remove_small_components(1e-3),
                        pol_vec=polarization,
                    )
                )
                ** 2
                for polarization in spherical
            )
            for state in ground
        ]
    )
    expected = strengths / strengths.sum()
    actual = couplings.calculate_br(excited, ground)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
    legacy_strengths = np.array(
        [
            abs(
                hamiltonian.generate_ED_ME_mixed_state(
                    state.remove_small_components(1e-3), excited.remove_small_components(1e-3)
                )
            )
            ** 2
            for state in ground
        ]
    )
    legacy = legacy_strengths / legacy_strengths.sum()
    assert np.max(abs(legacy - expected)) > 1e-4


def test_empty_ground_list_returns_empty_branching_array():
    _, excited = cartesian_ground_triplet()
    assert couplings.calculate_br(excited, []).shape == (0,)


def test_zero_total_dipole_strength_raises():
    _, excited = cartesian_ground_triplet()
    forbidden = 1 * next(
        iter(states.generate_coupled_states_X(states.QuantumSelector(J=0, F1=0.5, F=0, mF=0)))
    )
    with pytest.raises(ValueError, match="zero total dipole strength"):
        couplings.calculate_br(excited, [forbidden])
