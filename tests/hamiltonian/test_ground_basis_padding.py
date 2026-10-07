import numpy as np
import pytest

from centrex_tlf import hamiltonian, states, transitions


@pytest.mark.parametrize("selected_js", [[0], [1], [2], [3], [2, 3]])
def test_default_ground_construction_pads_both_bounds(selected_js):
    selected = states.generate_coupled_states_ground(Js=selected_js)
    reduced = hamiltonian.generate_reduced_X_hamiltonian(selected)

    expected_js = set(range(max(0, min(selected_js) - 2), max(selected_js) + 3))
    assert {basis.J for basis in reduced.QN_construct} == expected_js
    assert len(reduced.QN_basis) == len(selected)


@pytest.mark.parametrize(
    "lower, upper, expected_js",
    [(1, None, {1, 2, 3, 4}), (None, 3, {0, 1, 2, 3}), (2, 2, {2})],
)
def test_explicit_ground_bounds_override_independently(lower, upper, expected_js):
    selected = states.generate_coupled_states_ground(Js=[2])
    reduced = hamiltonian.generate_reduced_X_hamiltonian(selected, Jmin=lower, Jmax=upper)

    assert {basis.J for basis in reduced.QN_construct} == expected_js
    assert len(reduced.QN_basis) == len(selected)


def test_padded_ground_spectrum_converges_at_experimental_fields():
    selected = states.generate_coupled_states_ground(Js=[2])
    fields = dict(E=np.array([0.0, 0.0, -170.0]), B=np.array([-0.01327, 0.19433, -0.48889]))
    default = hamiltonian.generate_reduced_X_hamiltonian(selected, **fields)
    reference = hamiltonian.generate_reduced_X_hamiltonian(selected, Jmin=0, Jmax=6, **fields)

    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(default.H)),
        np.sort(np.linalg.eigvalsh(reference.H)),
        rtol=0,
        atol=2 * np.pi,
    )
    for default_state, reference_state in zip(default.QN_basis, reference.QN_basis, strict=True):
        default_vector = default_state.state_vector(reference.QN_construct)
        reference_vector = reference_state.state_vector(reference.QN_construct)
        fidelity = abs(np.vdot(default_vector, reference_vector)) ** 2 / (
            np.vdot(default_vector, default_vector).real
            * np.vdot(reference_vector, reference_vector).real
        )
        assert fidelity > 1 - 1e-6


def test_relative_optical_line_positions_converge_with_ground_padding():
    ground = list(states.generate_coupled_states_ground(Js=[2]))
    excited = list(
        states.generate_coupled_states_B(states.QuantumSelector(J=3, F1=3.5, F=4, P=[-1, 1], Ω=1))
    )
    fields = dict(E=np.array([0.0, 0.0, -170.0]), B=np.array([-0.01327, 0.19433, -0.48889]))
    default = hamiltonian.generate_total_reduced_hamiltonian(ground, excited, **fields)
    reference = hamiltonian.generate_total_reduced_hamiltonian(
        ground, excited, Jmin_X=0, Jmax_X=6, **fields
    )

    def relative_line_positions(reduced):
        energies = np.diag(reduced.H_int).real
        ground_offsets = energies[: len(ground)] - energies[0]
        excited_offsets = energies[len(ground) :] - energies[len(ground)]
        return excited_offsets[:, None] - ground_offsets[None, :]

    assert default.QN_basis == reference.QN_basis
    assert {basis.J for basis in default.X_hamiltonian.QN_construct} == set(range(5))
    assert {basis.J for basis in reference.X_hamiltonian.QN_construct} == set(range(7))
    np.testing.assert_allclose(
        relative_line_positions(default),
        relative_line_positions(reference),
        rtol=0,
        atol=2 * np.pi,
    )


def test_transition_builder_uses_padded_ground_construction():
    reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        [transitions.R2_F1_7o2_F4],
        E=np.array([0.0, 0.0, -170.0]),
        B=np.array([-0.01327, 0.19433, -0.48889]),
        retain_opposite_parity_levels=True,
    )
    retained_js = {int(basis.J) for basis in reduced.X_states_basis}
    expected_js = set(range(max(0, min(retained_js) - 2), max(retained_js) + 3))

    assert retained_js == {2, 3, 4, 5}
    assert {basis.J for basis in reduced.X_hamiltonian.QN_construct} == expected_js
    assert len(reduced.X_states) == 128
