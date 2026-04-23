from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from centrex_tlf import states


def _as_field_vector(
    value: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    if np.isscalar(value):
        return np.array([0.0, 0.0, float(value)], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Field vector must have shape (3,), not {arr.shape}")
    return arr


def _parameter_at_time(
    value: float | complex | Callable[[float], float | complex],
    t: float,
) -> float | complex:
    if callable(value):
        return value(float(t))
    return value


def _state_vector(
    state: states.CoupledState,
    basis: Sequence[states.CoupledBasisState],
) -> np.ndarray:
    return np.asarray((1.0 * state).state_vector(basis), dtype=np.complex128)


def _basis_transform(
    reduced_states: Sequence[states.CoupledState],
    construct_basis: Sequence[states.CoupledBasisState],
) -> np.ndarray:
    return np.column_stack([_state_vector(state, construct_basis) for state in reduced_states])


def _reduce_square_matrix(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.ix_(indices, indices)], dtype=np.complex128)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def _safe_denominator(value: float, floor: float) -> float:
    if abs(value) >= floor:
        return value
    if value == 0.0:
        return floor
    return float(np.sign(value)) * floor


def _state_label(
    state: states.CoupledState | states.CoupledBasisState,
) -> str:
    return str(state.largest if hasattr(state, "largest") else state)


def _dominant_state_index(
    target_state: states.CoupledState,
    basis_states: Sequence[states.CoupledState],
) -> int:
    vec = np.asarray((1.0 * target_state).state_vector(basis_states), dtype=np.complex128)
    idx = int(np.argmax(np.abs(vec) ** 2))
    overlap = float(np.abs(vec[idx]) ** 2)
    if overlap < 0.5:
        raise ValueError(
            "Could not identify a unique dominant state in the fixed basis; "
            f"largest overlap was only {overlap:.3f}."
        )
    return idx


def _selector_indices(
    qn: Sequence[states.CoupledState],
    selector: states.QuantumSelector,
) -> np.ndarray:
    indices = selector.get_indices(qn)
    return np.asarray(indices, dtype=np.int64)


def _exact_full_eigensubspace(
    hamiltonian_full: np.ndarray,
    p_indices: np.ndarray,
    n_keep: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(hamiltonian_full))
    p_weights = np.sum(np.abs(eigenvectors[p_indices, :]) ** 2, axis=0)
    selected = np.argsort(p_weights)[-n_keep:]
    selected = selected[np.argsort(np.real(eigenvalues[selected]))]
    return (
        np.asarray(eigenvalues[selected], dtype=np.float64),
        np.asarray(eigenvectors[:, selected], dtype=np.complex128),
        np.asarray(p_weights[selected], dtype=np.float64),
    )


def _polar_unitary(matrix: np.ndarray) -> np.ndarray:
    u_mat, _, vh_mat = np.linalg.svd(np.asarray(matrix, dtype=np.complex128), full_matrices=False)
    return np.asarray(u_mat @ vh_mat, dtype=np.complex128)


def _state_key(state: states.CoupledState) -> str:
    if isinstance(state, states.CoupledBasisState):
        return repr(1.0 * state)
    return repr(state)


def _psd_project(matrix: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    matrix = _symmetrize(np.asarray(matrix, dtype=np.complex128))
    evals, evecs = np.linalg.eigh(matrix)
    evals = np.where(np.real(evals) > tol, np.real(evals), 0.0)
    if np.count_nonzero(evals) == 0:
        return np.zeros_like(matrix, dtype=np.complex128)
    return _symmetrize((evecs * evals) @ evecs.conj().T)
