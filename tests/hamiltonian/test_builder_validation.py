import warnings

import numpy as np
import pytest

from centrex_tlf import hamiltonian, states, transitions


@pytest.mark.parametrize("electronic", ["X", "B"])
@pytest.mark.parametrize("bounds", [(0, 1), (3, 4), (3, 1), (-1, 4)])
def test_reduced_builders_reject_bounds_excluding_requested_states(electronic, bounds):
    if electronic == "X":
        selected = states.generate_coupled_states_X(states.QuantumSelector(J=2, F1=2.5, F=3, mF=0))
        builder = hamiltonian.generate_reduced_X_hamiltonian
    else:
        selected = states.generate_coupled_states_B(
            states.QuantumSelector(J=2, F1=2.5, F=3, mF=0, P=1, Ω=1)
        )
        builder = hamiltonian.generate_reduced_B_hamiltonian
    with pytest.raises(ValueError, match="construction bounds|Construction bounds"):
        builder(selected, Jmin=bounds[0], Jmax=bounds[1])


def test_ground_j1_j2_mixing_requires_an_electric_field():
    basis = list(states.generate_uncoupled_states_ground([1, 2]))
    terms = hamiltonian.generate_uncoupled_hamiltonian_X(basis)
    function = hamiltonian.generate_uncoupled_hamiltonian_X_function(terms)
    lower = [index for index, state in enumerate(basis) if state.J == 1]
    upper = [index for index, state in enumerate(basis) if state.J == 2]
    block = np.ix_(lower, upper)
    zero = np.zeros(3)
    earth = np.array([-0.01327, 0.19433, -0.48889])

    assert not np.any(function(zero, zero)[block])
    assert not np.any(function(zero, earth)[block])
    assert np.any(function(np.array([0.0, 0.0, -170.0]), earth)[block])


@pytest.mark.parametrize("electric", [[170.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
def test_omega_matching_retains_distinct_parity_eigenstates(electric):
    ground = states.generate_coupled_states_ground([2])
    excited = states.generate_coupled_states_B(
        states.QuantumSelector(J=3, F1=3.5, F=4, P=[-1, 1], Ω=1)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reduced = hamiltonian.generate_total_reduced_hamiltonian(
            ground, excited, E=np.array(electric), B=np.array([-0.01327, 0.19433, -0.48889])
        )
    vectors = np.array(
        [
            state.transform_to_omega_basis().state_vector(reduced.B_hamiltonian.QN_construct)
            for state in reduced.B_states
        ]
    )
    assert len(reduced.B_states) == 18
    assert np.linalg.matrix_rank(vectors) == 18


def test_transition_discovery_uses_unique_excited_matching(monkeypatch):
    module = hamiltonian.reduced_hamiltonian
    original = module._match_excited_states
    matches = []

    def check_matches(approximate, dressed):
        matched = original(approximate, dressed)
        assert len({id(state) for state in matched}) == len(approximate)
        matches.append(len(matched))
        return matched

    monkeypatch.setattr(module, "_match_excited_states", check_matches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        module.generate_reduced_hamiltonian_transitions(
            [transitions.R2_F1_7o2_F4],
            E=np.array([170.0, 0.0, 0.0]),
            B=np.array([-0.01327, 0.19433, -0.48889]),
            retain_opposite_parity_levels=True,
        )
    assert matches == [18, 18]


def test_microwave_only_hamiltonian_has_empty_b_block():
    reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        [transitions.MicrowaveTransition(1, 2)]
    )
    assert reduced.B_states == []
    assert reduced.B_hamiltonian.H.shape == (0, 0)
    assert {state.J for state in reduced.X_states_basis} == {1, 2}
    assert reduced.H_int.shape == (32, 32)
