import pickle
import warnings
from pathlib import Path

import numpy as np

from centrex_tlf import hamiltonian, states
from centrex_tlf.hamiltonian.reduced_hamiltonian import (
    ReducedHamiltonian,
    ReducedHamiltonianTotal,
)


def test_generate_reduced_X_hamiltonian():
    X_states_approx = states.generate_coupled_states_ground(Js=[0, 1, 2, 3])

    H_reduced = hamiltonian.reduced_hamiltonian.generate_reduced_X_hamiltonian(
        X_states_approx=X_states_approx,
        Jmin=0,
        Jmax=3,
    )

    with open(Path(__file__).parent / "X_reduced.pkl", "rb") as f:
        H_reduced_test: hamiltonian.reduced_hamiltonian.ReducedHamiltonian = (
            pickle.load(f)
        )

    assert np.allclose(H_reduced.H, H_reduced_test.H, rtol=1e-15, atol=1e-3)
    # assert np.allclose(H_reduced.V, H_reduced_test.V)
    assert len(H_reduced.QN_basis) == len(H_reduced_test.QN_basis)
    assert len(H_reduced.QN_construct) == len(H_reduced_test.QN_construct)
    assert H_reduced.QN_construct == H_reduced_test.QN_construct

    state_vectors = np.array(
        [s.state_vector(H_reduced.QN_basis) for s in H_reduced_test.QN_basis]
    )

    assert np.allclose(np.trace(state_vectors), len(state_vectors), rtol=1e-8)


def test_generate_reduced_B_hamiltonian_omega():
    # omega basis
    B_states_approx = states.generate_coupled_states_excited(
        Js=[1, 2, 3], Ps=None, Omegas=[-1, 1]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H_reduced_omega = (
            hamiltonian.reduced_hamiltonian.generate_reduced_B_hamiltonian(
                B_states_approx=B_states_approx, Jmin=1, Jmax=5, rtol=1e-5
            )
        )

    with open(Path(__file__).parent / "B_reduced_omega.pkl", "rb") as f:
        H_reduced_test_omega: hamiltonian.reduced_hamiltonian.ReducedHamiltonian = (
            pickle.load(f)
        )

    assert np.allclose(H_reduced_omega.H, H_reduced_test_omega.H, rtol=1e-15, atol=1e-3)
    # assert np.allclose(H_reduced_omega.V, H_reduced_test_omega.V)
    assert len(H_reduced_omega.QN_basis) == len(H_reduced_test_omega.QN_basis)
    assert len(H_reduced_omega.QN_construct) == len(H_reduced_test_omega.QN_construct)
    assert H_reduced_omega.QN_construct == H_reduced_test_omega.QN_construct

    state_vectors = np.array(
        [
            s.state_vector(H_reduced_omega.QN_basis)
            for s in H_reduced_test_omega.QN_basis
        ]
    )

    assert np.allclose(np.trace(state_vectors), len(state_vectors), rtol=1e-6)


def test_generate_reduced_B_hamiltonian_parity():
    # parity basis
    B_states_approx = states.generate_coupled_states_excited(
        Js=[1, 2, 3], Ps=[-1, 1], Omegas=1
    )

    H_reduced_parity = hamiltonian.reduced_hamiltonian.generate_reduced_B_hamiltonian(
        B_states_approx=B_states_approx, Jmin=1, Jmax=5, rtol=1e-5
    )

    with open(Path(__file__).parent / "B_reduced_parity.pkl", "rb") as f:
        H_reduced_parity_test: ReducedHamiltonian = pickle.load(f)

    assert np.allclose(
        H_reduced_parity.H, H_reduced_parity_test.H, rtol=1e-15, atol=1e-3
    )
    # assert np.allclose(H_reduced_parity.V, H_reduced_parity_test.V)
    assert len(H_reduced_parity.QN_basis) == len(H_reduced_parity_test.QN_basis)
    assert len(H_reduced_parity.QN_construct) == len(H_reduced_parity_test.QN_construct)
    assert H_reduced_parity.QN_construct == H_reduced_parity_test.QN_construct

    state_vectors = np.array(
        [
            s.state_vector(H_reduced_parity.QN_basis)
            for s in H_reduced_parity_test.QN_basis
        ]
    )

    assert np.allclose(np.trace(state_vectors), len(state_vectors), rtol=1e-6)


def test_generate_total_reduced_hamiltonian():
    B_states_approx = states.generate_coupled_states_excited(
        Js=[1, 2, 3], Ps=[-1, 1], Omegas=1
    )
    X_states_approx = states.generate_coupled_states_ground(Js=[0, 1, 2, 3])

    H_reduced_total = (
        hamiltonian.reduced_hamiltonian.generate_total_reduced_hamiltonian(
            X_states_approx=X_states_approx,
            B_states_approx=B_states_approx,
            rtol=None,
            stol=1e-3,
            Jmin_X=0,
            Jmax_X=3,
            Jmin_B=1,
            Jmax_B=5,
        )
    )

    with open(Path(__file__).parent / "H_reduced_total.pkl", "rb") as f:
        H_reduced_total_test: ReducedHamiltonianTotal = pickle.load(f)

    assert np.allclose(
        H_reduced_total.H_int, H_reduced_total_test.H_int, rtol=1e-15, atol=1e-3
    )
    # assert np.allclose(H_reduced_total.V_ref_int, H_reduced_total_test.V_ref_int)
    assert len(H_reduced_total.QN) == len(H_reduced_total_test.QN)
    assert len(H_reduced_total.QN_basis) == len(H_reduced_total_test.QN_basis)
    assert H_reduced_total.QN_basis == H_reduced_total_test.QN_basis

    state_vectors = np.array(
        [s.state_vector(H_reduced_total.QN) for s in H_reduced_total_test.QN]
    )

    assert np.allclose(np.trace(state_vectors), len(state_vectors), rtol=1e-6)
