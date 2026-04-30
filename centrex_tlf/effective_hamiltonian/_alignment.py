from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, logm
from scipy.optimize import linear_sum_assignment

from centrex_tlf.effective_hamiltonian._utility import _polar_unitary, _symmetrize


def _union_block_alignment(
    reference_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    ground_indices: np.ndarray,
    sink_indices: np.ndarray,
    excited_indices: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    reference_basis = np.asarray(reference_basis, dtype=np.complex128)
    target_basis = np.asarray(target_basis, dtype=np.complex128)
    rotation = np.eye(reference_basis.shape[1], dtype=np.complex128)
    reference_active = np.linalg.norm(reference_basis, axis=0) > tol
    target_active = np.linalg.norm(target_basis, axis=0) > tol
    for sector_indices in (ground_indices, sink_indices, excited_indices):
        common = np.array(
            [int(index) for index in sector_indices.tolist() if reference_active[int(index)] and target_active[int(index)]],
            dtype=np.int64,
        )
        if common.size == 0:
            continue
        overlap = reference_basis[:, common].conj().T @ target_basis[:, common]
        sector_rotation = _polar_unitary(overlap)
        rotation[np.ix_(common, common)] = sector_rotation
    aligned_basis = np.asarray(target_basis @ rotation.conj().T, dtype=np.complex128)
    return aligned_basis, rotation


def _tracking_rotation(
    reference_basis: np.ndarray,
    target_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_basis = np.asarray(reference_basis, dtype=np.complex128)
    target_basis = np.asarray(target_basis, dtype=np.complex128)
    n_states = int(reference_basis.shape[1])
    if target_basis.shape[1] != n_states:
        raise ValueError(
            f"Tracking rotation requires matching dimensions, got {n_states} and {target_basis.shape[1]}."
        )
    overlap = reference_basis.conj().T @ target_basis
    row_ind, col_ind = linear_sum_assignment(-np.abs(overlap))
    if not np.array_equal(row_ind, np.arange(n_states, dtype=np.int64)):
        raise ValueError("Unexpected nontrivial row permutation from linear_sum_assignment.")
    permutation = np.zeros((n_states, n_states), dtype=np.complex128)
    for new_col, old_col in enumerate(col_ind.tolist()):
        permutation[int(old_col), int(new_col)] = 1.0
    reordered = np.asarray(target_basis @ permutation, dtype=np.complex128)
    reordered_overlap = reference_basis.conj().T @ reordered
    phases = np.ones(n_states, dtype=np.complex128)
    for idx in range(n_states):
        value = complex(reordered_overlap[idx, idx])
        if abs(value) > 0.0:
            phases[idx] = np.exp(-1j * np.angle(value))
    phase_matrix = np.diag(phases.astype(np.complex128))
    rotation = np.asarray(permutation @ phase_matrix, dtype=np.complex128)
    tracked_basis = np.asarray(target_basis @ rotation, dtype=np.complex128)
    return tracked_basis, rotation


def _coherent_gauge_connection(
    basis_minus: np.ndarray,
    basis_plus: np.ndarray,
    delta_field: float,
) -> np.ndarray:
    if delta_field == 0.0:
        raise ValueError("delta_field must be nonzero for gauge-connection estimation.")
    overlap = np.asarray(basis_minus.conj().T @ basis_plus, dtype=np.complex128)
    unitary_overlap = _polar_unitary(overlap)
    generator = np.asarray(logm(unitary_overlap), dtype=np.complex128) / float(delta_field)
    connection = 1j * generator
    return _symmetrize(np.asarray(connection, dtype=np.complex128))


def _sector_block_alignment(
    reference_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    ground_slice: slice = slice(0, 4),
    excited_slice: slice = slice(4, 7),
) -> tuple[np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    for sector in (ground_slice, excited_slice):
        overlap = reference_basis[:, sector].conj().T @ target_basis[:, sector]
        blocks.append(_polar_unitary(overlap))
    physical_rotation = block_diag(blocks[0], blocks[1]).astype(np.complex128)
    full_rotation = block_diag(blocks[0], np.eye(3, dtype=np.complex128), blocks[1]).astype(np.complex128)
    return physical_rotation, full_rotation
