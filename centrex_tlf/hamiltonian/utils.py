from typing import List, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt

from centrex_tlf.states import (
    CoupledBasisState,
    CoupledState,
    UncoupledBasisState,
    UncoupledState,
)

__all__ = ["reorder_evecs", "matrix_to_states", "reduced_basis_hamiltonian"]


def reorder_evecs(
    V_in: npt.NDArray[np.complex128],
    E_in: npt.NDArray[np.complex128],
    V_ref: npt.NDArray[np.complex128],
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Reshuffle eigenvectors and eigenergies based on a reference

    Args:
        V_in (np.ndarray): eigenvector matrix to be reorganized
        E_in (np.ndarray): energy vector to be reorganized
        V_ref (np.ndarray): reference eigenvector matrix

    Returns:
        (np.ndarray, np.ndarray): energy vector, eigenvector matrix
    """
    # take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T), V_ref))

    # find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors, axis=1))
    # store energy and state
    E_out = E_in[index]
    V_out = V_in[:, index]

    return E_out, V_out


@overload
def matrix_to_states(
    V: npt.NDArray[np.complex128], QN: Sequence[CoupledBasisState]
) -> List[CoupledState]: ...


@overload
def matrix_to_states(
    V: npt.NDArray[np.complex128], QN: Sequence[UncoupledBasisState]
) -> List[UncoupledState]: ...


def matrix_to_states(V, QN):
    """Turn a matrix of eigenvectors into a list of state objects
    QN is in the basis the diagonal Hamiltonian H was formed from corresponding to the
    eigenvectors V.

    Args:
        V (npt.NDArray[np.complex128]): array with columns corresponding to eigenvectors
        QN (Sequence[BasisState]): list of State objects
        E (List, optional): list of energies corresponding to the states.
                            Defaults to None.

    Returns:
        List[State]: list of eigenstates expressed as State objects
    """
    # find dimensions of matrix
    matrix_dimensions = V.shape

    # initialize a list for storing eigenstates
    eigenstates = []

    for i in range(0, matrix_dimensions[1]):
        # find state vector
        state_vector = V[:, i]

        # ensure that largest component has positive sign
        index = np.argmax(np.abs(state_vector))
        state_vector = state_vector * np.sign(state_vector[index])

        data = []

        # get data in correct format for initializing state object
        for j, amp in enumerate(state_vector):
            data.append((amp, QN[j]))

        # store the state in the list
        if isinstance(QN[0], CoupledBasisState):
            state = CoupledState(data)
        elif isinstance(QN[1], UncoupledBasisState):
            state = UncoupledState(data)
        else:
            raise ValueError(
                f"QN should be list of CoupledBasisState or UncoupledBasisState, not {type(QN[0])}"
            )
        eigenstates.append(state)

    # return the list of states
    return eigenstates


def reduced_basis_hamiltonian(
    basis_original: Sequence[CoupledState],
    H_original: npt.NDArray[np.complex128],
    basis_reduced: Sequence[CoupledState],
) -> npt.NDArray[np.complex128]:
    """Generate Hamiltonian for a sub-basis of the original basis

    Args:
        basis_original (Sequence[State],): sequence of states of original basis
        H_original (npt.NDArray[np.complex128]): original Hamiltonian
        basis_reduced (Sequence[State]): sequence of states of sub-basis

    Returns:
        npt.NDArray[np.complex128]: Hamiltonian in sub-basis
    """

    # Determine the indices of each of the reduced basis states
    index_red = np.zeros(len(basis_reduced), dtype=int)
    for i, state_red in enumerate(basis_reduced):
        index_red[i] = basis_original.index(state_red)

    # Initialize matrix for Hamiltonian in reduced basis
    H_red = np.zeros((len(basis_reduced), len(basis_reduced)), dtype=complex)

    # Loop over reduced basis states and pick out the correct matrix elements
    # for the Hamiltonian in the reduced basis
    for i, state_i in enumerate(basis_reduced):
        for j, state_j in enumerate(basis_reduced):
            H_red[i, j] = H_original[index_red[i], index_red[j]]

    return H_red
