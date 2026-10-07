import warnings

import numpy as np
import pytest

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions


@pytest.fixture
def custom_construction():
    uncoupled = list(states.generate_uncoupled_states_ground(range(5)))
    coupled = list(states.generate_coupled_states_ground(range(5)))
    transform = hamiltonian.generate_transform_matrix(uncoupled, coupled)
    terms = hamiltonian.generate_uncoupled_hamiltonian_X(uncoupled)
    function = hamiltonian.generate_uncoupled_hamiltonian_X_function(terms)
    bounds = dict(Jmin_X=0, Jmax_X=4, Jmin_B=1, Jmax_B=3)
    return transform, function, bounds


@pytest.mark.parametrize("use_omega_basis", [True, False])
def test_custom_b_hamiltonian_controls_discovery_and_final_states(use_omega_basis):
    selector = states.QuantumSelector(
        J=[1, 2, 3], P=None if use_omega_basis else [-1, 1], Ω=[-1, 1] if use_omega_basis else 1
    )
    basis = list(states.generate_coupled_states_B(selector))
    function = hamiltonian.generate_coupled_hamiltonian_B_function(
        hamiltonian.generate_coupled_hamiltonian_B(basis)
    )
    calls = []

    def custom_b(electric, magnetic):
        calls.append((electric.copy(), magnetic.copy()))
        return function(np.array([0.0, 0.0, 200.0]), magnetic)

    arguments = dict(
        E=np.zeros(3),
        B=np.array([0.0, 0.0, 0.001]),
        Jmin_X=0,
        Jmax_X=4,
        Jmin_B=1,
        Jmax_B=3,
        use_omega_basis=use_omega_basis,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        baseline = hamiltonian.generate_reduced_hamiltonian_transitions(
            [transitions.R0_F1_1o2_F1], **arguments
        )
        custom = hamiltonian.generate_reduced_hamiltonian_transitions(
            [transitions.R0_F1_1o2_F1], H_func_B=custom_b, **arguments
        )
    assert len(calls) == 2
    assert all(np.array_equal(electric, arguments["E"]) for electric, _ in calls)
    assert {state.J for state in baseline.X_states_basis} == {0, 2}
    assert any(state.J % 2 for state in custom.X_states_basis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reference = hamiltonian.generate_total_reduced_hamiltonian(
            custom.X_states_basis, custom.B_states_basis, H_func_B=custom_b, **arguments
        )
    np.testing.assert_allclose(custom.H_int, reference.H_int, rtol=0, atol=0.01)


@pytest.mark.parametrize(
    "builder", [lindblad.generate_OBE_system_transitions, lindblad.setup_OBE_system_transitions]
)
def test_obe_transition_builders_apply_custom_functions_and_transform(custom_construction, builder):
    transform, ground_function, bounds = custom_construction
    transition = transitions.R0_F1_1o2_F1
    selectors = couplings.generate_transition_selectors([transition], [[couplings.polarization_X]])
    excited_basis = list(
        states.generate_coupled_states_B(states.QuantumSelector(J=[1, 2, 3], P=None, Ω=[-1, 1]))
    )
    excited_function = hamiltonian.generate_coupled_hamiltonian_B_function(
        hamiltonian.generate_coupled_hamiltonian_B(excited_basis)
    )
    calls = {"X": 0, "B": 0}

    def custom_x(electric, magnetic):
        calls["X"] += 1
        return ground_function(electric, magnetic) + 1e6 * np.eye(transform.shape[0])

    def custom_b(electric, magnetic):
        calls["B"] += 1
        return excited_function(electric, magnetic)

    baseline = builder([transition], selectors, B=np.array([0.0, 0.0, 0.001]), **bounds)
    custom = builder(
        [transition],
        selectors,
        B=np.array([0.0, 0.0, 0.001]),
        transform=transform,
        H_func_X=custom_x,
        H_func_B=custom_b,
        **bounds,
    )
    assert calls == {"X": 1, "B": 2}
    diagonal_change = np.diag(custom.H_int - baseline.H_int).real
    np.testing.assert_allclose(diagonal_change[: len(custom.ground)], 1e6, rtol=0, atol=0.01)
    np.testing.assert_allclose(diagonal_change[len(custom.ground) :], 0, rtol=0, atol=0.01)


def test_transition_hamiltonian_uses_supplied_transform(custom_construction, monkeypatch):
    transform, ground_function, bounds = custom_construction

    def forbidden_transform(*arguments):
        raise AssertionError("Provided transform must not be regenerated")

    monkeypatch.setattr(
        hamiltonian.reduced_hamiltonian, "generate_transform_matrix", forbidden_transform
    )
    reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        [transitions.R0_F1_1o2_F1], transform=transform, H_func_X=ground_function, **bounds
    )
    assert reduced.X_hamiltonian.transform is transform


@pytest.mark.parametrize("electronic", ["X", "B"])
def test_custom_hamiltonian_shape_is_validated(custom_construction, electronic):
    _, _, bounds = custom_construction

    def wrong_shape(electric, magnetic):
        return np.zeros((1, 1))

    arguments = {f"H_func_{electronic}": wrong_shape}
    with pytest.raises(ValueError, match=f"H_func_{electronic} returned shape"):
        hamiltonian.generate_reduced_hamiltonian_transitions(
            [transitions.R0_F1_1o2_F1], **arguments, **bounds
        )


@pytest.mark.parametrize("invalid", ["rectangular", "nonunitary"])
def test_custom_transform_is_validated(custom_construction, invalid):
    transform, _, bounds = custom_construction
    supplied = transform[:, :-1] if invalid == "rectangular" else np.zeros_like(transform)
    with pytest.raises(ValueError, match="shape of transform|unitary"):
        hamiltonian.generate_reduced_hamiltonian_transitions(
            [transitions.R0_F1_1o2_F1], transform=supplied, **bounds
        )


@pytest.mark.parametrize("electronic", ["X", "B"])
@pytest.mark.parametrize(
    "invalid",
    ["nan", "inf", "negative_inf", "imaginary_inf", "imaginary_diagonal", "upper", "lower"],
)
def test_invalid_custom_matrices_are_rejected_before_diagonalization(
    electronic, invalid, monkeypatch
):
    if electronic == "X":
        selected = states.generate_coupled_states_X(states.QuantumSelector(J=2, F1=2.5, F=3, mF=0))
        dimension = len(states.generate_uncoupled_states_ground(range(5)))
        builder = hamiltonian.generate_reduced_X_hamiltonian
        bounds = dict(Jmin=0, Jmax=4)
    else:
        selected = states.generate_coupled_states_B(
            states.QuantumSelector(J=1, F1=0.5, F=1, mF=0, P=-1, Ω=1)
        )
        dimension = len(
            states.generate_coupled_states_B(states.QuantumSelector(J=[1, 2, 3], P=[-1, 1], Ω=1))
        )
        builder = hamiltonian.generate_reduced_B_hamiltonian
        bounds = dict(Jmin=1, Jmax=3)
    matrix = 1e12 * np.eye(dimension, dtype=complex)
    if invalid == "imaginary_diagonal":
        matrix[0, 0] += 1e6j
    elif invalid == "upper":
        matrix[0, 1] = 1.0
    elif invalid == "lower":
        matrix[1, 0] = 1.0
    else:
        matrix[0, 0] = {
            "nan": np.nan,
            "inf": np.inf,
            "negative_inf": -np.inf,
            "imaginary_inf": complex(0, np.inf),
        }[invalid]

    def callback(electric, magnetic):
        return matrix

    def forbidden_diagonalization(*arguments, **keywords):
        raise AssertionError("Invalid custom matrices must fail before diagonalization")

    monkeypatch.setattr(
        hamiltonian.reduced_hamiltonian,
        "generate_diagonalized_hamiltonian",
        forbidden_diagonalization,
    )
    message = "Hermitian" if invalid in {"imaginary_diagonal", "upper", "lower"} else "finite"
    with pytest.raises(ValueError, match=f"H_func_{electronic}.*{message}"):
        builder(selected, H_func=callback, **bounds)


def test_hamiltonian_validation_accepts_complex_hermitian_matrices():
    matrix = np.array([[1.0, 2.0 + 3.0j], [2.0 - 3.0j, 4.0]])
    hamiltonian.reduced_hamiltonian._validate_hamiltonian_matrix(matrix, 2, "custom")


def test_hermiticity_validation_tolerance_tracks_roundoff_scale():
    matrix = np.eye(2, dtype=complex)
    matrix[0, 1] = 1e-4
    with pytest.raises(ValueError, match="Hermitian"):
        hamiltonian.reduced_hamiltonian._validate_hamiltonian_matrix(matrix, 2, "custom")
    matrix += 1e12 * np.eye(2)
    hamiltonian.reduced_hamiltonian._validate_hamiltonian_matrix(matrix, 2, "custom")
