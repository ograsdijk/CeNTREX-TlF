from functools import lru_cache
from typing import List, Sequence, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
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

    Column k of the returned eigenvector matrix is the input eigenvector matched to
    column k of ``V_ref``; if ``V_ref`` has fewer columns than ``V_in``, the unmatched
    eigenvectors follow in their original order. The matching is the assignment that maximizes the total
    overlap ``Σ_k |⟨V_in[:, i_k] | V_ref[:, k]⟩|`` (Hungarian algorithm), which
    guarantees a one-to-one mapping. A greedy per-eigenvector argmax does not: when
    two input eigenvectors have their largest overlap with the same reference column
    - which happens once states mix strongly, e.g. in X at electric fields above
    roughly 20 kV/cm - it silently produces an arbitrary ordering instead.

    Args:
        V_in (np.ndarray): eigenvector matrix to be reorganized
        E_in (np.ndarray): energy vector to be reorganized
        V_ref (np.ndarray): reference eigenvector matrix

    Returns:
        (np.ndarray, np.ndarray): energy vector, eigenvector matrix
    """
    # take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T), V_ref))

    # optimal one-to-one matching of input eigenvectors to reference eigenvectors
    row_ind, col_ind = linear_sum_assignment(-overlap_vectors)

    # index[k] is the input eigenvector assigned to reference eigenvector k
    index = row_ind[np.argsort(col_ind)]

    # with fewer reference vectors than eigenvectors only min(N, M) get assigned; keep
    # the rest, in their original order, so the output always holds every eigenvector
    if index.size < V_in.shape[1]:
        remaining = np.setdiff1d(np.arange(V_in.shape[1]), index)
        index = np.concatenate([index, remaining])

    # store energy and state
    E_out = E_in[index]
    V_out = V_in[:, index]

    return E_out, V_out


DType = TypeVar("DType")


def get_unique_list(states: Sequence[DType]) -> List[DType]:
    """Get a list/array of unique entries in the list/array.

    Args:
        states: list or array of items supporting hash and equality

    Returns:
        list or array with unique entries, preserving first-occurrence order
    """
    seen: dict = {}
    states_unique = []
    for state in states:
        h = hash(state)
        bucket = seen.get(h)
        if bucket is None:
            seen[h] = [state]
            states_unique.append(state)
        else:
            is_dup = False
            for existing in bucket:
                if existing == state:
                    is_dup = True
                    break
            if not is_dup:
                bucket.append(state)
                states_unique.append(state)

    if isinstance(states, np.ndarray):
        return np.asarray(states_unique)
    else:
        return states_unique
