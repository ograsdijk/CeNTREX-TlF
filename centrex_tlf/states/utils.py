from functools import lru_cache
from typing import List, Sequence, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from sympy.physics.quantum.cg import CG

__all__ = ["CGc", "parity_X", "reorder_evecs"]


@lru_cache(maxsize=int(1e6))
def CGc(j1: float, m1: float, j2: float, m2: float, j3: float, m3: float) -> complex:
    """Calculate Clebsch-Gordan coefficient.

    Computes ⟨j1 m1 j2 m2 | j3 m3⟩ using sympy's quantum CG coefficient.
    Results are cached for performance.

    Args:
        j1: First angular momentum quantum number
        m1: First magnetic quantum number
        j2: Second angular momentum quantum number
        m2: Second magnetic quantum number
        j3: Total angular momentum quantum number
        m3: Total magnetic quantum number

    Returns:
        Complex Clebsch-Gordan coefficient
    """
    return complex(CG(j1, m1, j2, m2, j3, m3).doit())


def parity_X(J: int) -> int:
    """Calculate parity of X electronic state for given J.

    The parity of the ground X¹Σ⁺ state is (-1)^J.

    Args:
        J: Rotational quantum number

    Returns:
        Parity: +1 or -1
    """
    return (-1) ** J


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


DType = TypeVar("DType")


def get_unique_list(states: Sequence[DType]) -> List[DType]:
    """get a list/array of unique entries in the list/array

    Args:
        states (Union[list, np.ndarray]): list/array

    Returns:
        Union[list, np.ndarray]: list/array with unique entries
    """
    states_unique = []
    for state in states:
        if state not in states_unique:
            states_unique.append(state)

    if isinstance(states, np.ndarray):
        return np.asarray(states_unique)
    else:
        return states_unique
