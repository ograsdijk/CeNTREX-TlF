from typing import List, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt

from centrex_tlf.states import (
    CoupledBasisState,
    CoupledState,
    UncoupledBasisState,
    UncoupledState,
)
from centrex_tlf.states.utils import reorder_evecs

__all__ = ["reorder_evecs", "matrix_to_states", "reduced_basis_hamiltonian"]


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
    if len(QN) == 0:
        raise ValueError("QN must contain at least one basis state")

    # find dimensions of matrix
    matrix_dimensions = V.shape
    basis_type = type(QN[0])

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
        elif isinstance(QN[0], UncoupledBasisState):
            state = UncoupledState(data)
        else:
            raise ValueError(
                f"QN should be list of CoupledBasisState or UncoupledBasisState, not {basis_type}"
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
