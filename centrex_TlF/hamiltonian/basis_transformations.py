from typing import Any, Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf.states import BasisState

__all__ = ["generate_transform_matrix"]


def generate_transform_matrix(
    basis1: Union[Sequence[BasisState], npt.NDArray[Any]],
    basis2: Union[Sequence[BasisState], npt.NDArray[Any]],
) -> npt.NDArray[np.complex128]:
    """
    Function that generates a transform matrix that takes Hamiltonian expressed
    in basis1 to basis2: H_2 = S.conj().T @ H_1 @ S

    inputs:
    basis1 = list of basis states that defines basis1
    basis2 = list of basis states that defines basis2
    progress = boolean to display tqdm progress bar

    returns:
    S = transformation matrix that takes Hamiltonian (or any operator) from
    basis1 to basis2
    """

    # Check that the two bases have the same dimension
    assert len(basis1) == len(basis2), "Bases don't have the same dimension"

    # Initialize S
    S = np.zeros((len(basis1), len(basis1)), dtype=complex)

    # Loop over the bases and calculate inner products
    for i, state1 in enumerate(basis1):
        for j, state2 in enumerate(basis2):
            S[i, j] = state1 @ state2

    # Return the transform matrix
    return S
