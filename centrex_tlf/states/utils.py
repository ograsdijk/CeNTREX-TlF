from functools import lru_cache
from typing import List, Sequence, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from sympy.physics.quantum.cg import CG

__all__ = ["CGc", "parity_X", "reorder_evecs", "eigenstate_quantum_numbers"]


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


def _coupled_transform(QN: Sequence) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Return ``U[c, u] = <coupled c|uncoupled u>``, and F and F1 per coupled index."""
    decompositions = [q.transform_to_coupled() for q in QN]
    keys: dict = {}
    for decomposition in decompositions:
        for _, coupled in decomposition.data:
            key = (coupled.electronic_state, coupled.J, coupled.F1, coupled.F, coupled.mF)
            keys.setdefault(key, len(keys))

    U = np.zeros((len(keys), len(QN)), dtype=complex)
    for u, decomposition in enumerate(decompositions):
        for amplitude, coupled in decomposition.data:
            key = (coupled.electronic_state, coupled.J, coupled.F1, coupled.F, coupled.mF)
            U[keys[key], u] += amplitude

    ordered = sorted(keys, key=keys.get)
    F_of_coupled = np.array([key[3] for key in ordered], dtype=float)
    F1_of_coupled = np.array([key[2] for key in ordered], dtype=float)
    return U, F_of_coupled, F1_of_coupled


def _spread(weights: npt.NDArray, values: npt.NDArray) -> npt.NDArray:
    mean = weights.T @ values
    return np.sqrt(np.maximum(weights.T @ values**2 - mean**2, 0.0))


def eigenstate_quantum_numbers(V: npt.NDArray, QN: Sequence) -> dict:
    """(J, F1, F, mF) expectation values and spreads for each column of ``V``.

    Identifies eigenstates by what they are rather than by where they sit in an
    array. That matters wherever an index has been carried across a parameter
    sweep or a trajectory: `reorder_evecs` labels adiabatically, and at a level
    crossing the population follows its diabatic branch while the label follows
    the other, silently. Selecting on quantum numbers is immune to this.

    Every returned array has one entry per eigenvector, and each quantum number
    is accompanied by a ``*_spread``. A spread near zero means the quantum number
    is good for that state and the label is meaningful; a large spread means the
    state is a mixture and it is not. Check the spread rather than assuming: in a
    strong Stark field F is badly mixed and only becomes good near zero field,
    while mF = mJ + m1 + m2 is exact in the uncoupled basis throughout.

    Note that ``(J, F, mF)`` is not a complete label -- J=1 has two F=1 levels,
    one from F1=1/2 and one from F1=3/2 -- so F1 is returned as well.

    Args:
        V: eigenvector matrix, columns are eigenvectors, in the ``QN`` basis
        QN: the basis states ``V`` is expressed in

    Returns:
        dict of arrays: ``J``, ``F1``, ``F``, ``mF`` and a ``*_spread`` for each
    """
    V = np.asarray(V, dtype=complex)
    if V.ndim == 1:
        V = V[:, np.newaxis]

    weights = np.abs(V) ** 2
    weights = weights / weights.sum(axis=0, keepdims=True)

    mF_uncoupled = np.array([q.mJ + q.m1 + q.m2 for q in QN], dtype=float)
    J_uncoupled = np.array([q.J for q in QN], dtype=float)

    U, F_of_coupled, F1_of_coupled = _coupled_transform(QN)
    coupled_weights = np.abs(U @ V) ** 2
    coupled_weights = coupled_weights / coupled_weights.sum(axis=0, keepdims=True)
    mean_FF1 = coupled_weights.T @ (F_of_coupled * (F_of_coupled + 1))

    return {
        "J": weights.T @ J_uncoupled,
        "J_spread": _spread(weights, J_uncoupled),
        "F1": coupled_weights.T @ F1_of_coupled,
        "F1_spread": _spread(coupled_weights, F1_of_coupled),
        "F": (-1 + np.sqrt(1 + 4 * mean_FF1)) / 2,
        "F_spread": _spread(coupled_weights, F_of_coupled),
        "mF": weights.T @ mF_uncoupled,
        "mF_spread": _spread(weights, mF_uncoupled),
    }
