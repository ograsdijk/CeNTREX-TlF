from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from centrex_tlf.effective_hamiltonian._utility import _psd_project, _symmetrize


def _jump_rate_operator(
    c_array: np.ndarray,
    *,
    loss_operator: np.ndarray | None = None,
) -> np.ndarray:
    if c_array.size == 0:
        if loss_operator is None:
            return np.zeros((0, 0), dtype=np.complex128)
        return np.asarray(loss_operator, dtype=np.complex128)
    jump_rate = np.sum(np.array([c.conj().T @ c for c in c_array], dtype=np.complex128), axis=0)
    if loss_operator is not None and loss_operator.size:
        jump_rate = jump_rate + loss_operator
    return jump_rate


def _excited_to_sector_rates(
    c_array: np.ndarray,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
) -> np.ndarray:
    rates = np.zeros(len(source_indices), dtype=np.float64)
    if c_array.size == 0 or len(destination_indices) == 0:
        return rates
    for c_op in c_array:
        block = np.asarray(c_op[np.ix_(destination_indices, source_indices)], dtype=np.complex128)
        rates += np.sum(np.abs(block) ** 2, axis=0)
    return rates


def _sector_decay_kernel(
    c_array: np.ndarray,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
) -> np.ndarray:
    n_source = int(source_indices.size)
    kernel = np.zeros((n_source, n_source), dtype=np.complex128)
    if c_array.size == 0 or destination_indices.size == 0:
        return kernel
    for c_op in c_array:
        block = np.asarray(c_op[np.ix_(destination_indices, source_indices)], dtype=np.complex128)
        kernel += block.conj().T @ block
    return _symmetrize(kernel)


def _single_target_decay_kernels(
    c_array: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(
        _sector_decay_kernel(
            np.asarray(c_array, dtype=np.complex128),
            np.asarray(source_indices, dtype=np.int64),
            np.array([int(target_index)], dtype=np.int64),
        )
        for target_index in np.asarray(target_indices, dtype=np.int64).tolist()
    )


def _full_recycling_decay_kernel(
    c_array: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    n_composite = int(target_indices.size * source_indices.size)
    kernel = np.zeros((n_composite, n_composite), dtype=np.complex128)
    if c_array.size == 0 or source_indices.size == 0 or target_indices.size == 0:
        return kernel
    for c_op in np.asarray(c_array, dtype=np.complex128):
        block = np.asarray(c_op[np.ix_(target_indices, source_indices)], dtype=np.complex128)
        vector = block.reshape(-1)
        kernel += np.outer(vector, vector.conj())
    return _symmetrize(kernel)


def _full_recycling_decay_kernel_from_superoperator(
    dissipator_superop: np.ndarray,
    jump_rate_operator: np.ndarray,
) -> np.ndarray:
    jump_rate_operator = np.asarray(jump_rate_operator, dtype=np.complex128)
    n_states = int(jump_rate_operator.shape[0])
    identity = np.eye(n_states, dtype=np.complex128)
    recycling_superop = np.asarray(dissipator_superop, dtype=np.complex128)
    recycling_superop = recycling_superop + 0.5 * np.kron(identity, jump_rate_operator.T)
    recycling_superop = recycling_superop + 0.5 * np.kron(jump_rate_operator, identity)
    recycling_tensor = recycling_superop.reshape((n_states, n_states, n_states, n_states))
    kernel_tensor = np.transpose(recycling_tensor, (0, 2, 1, 3))
    return _symmetrize(kernel_tensor.reshape((n_states * n_states, n_states * n_states)))


def _c_array_from_full_recycling_decay_kernel(
    *,
    target_indices: np.ndarray,
    source_indices: np.ndarray,
    kernel: np.ndarray,
    total_dimension: int,
    tol: float = 1e-12,
) -> np.ndarray:
    target_indices = np.asarray(target_indices, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    kernel_psd = _psd_project(kernel, tol=tol)
    evals, evecs = np.linalg.eigh(kernel_psd)
    jumps: list[np.ndarray] = []
    target_count = int(target_indices.size)
    source_count = int(source_indices.size)
    for eval_value, evec in zip(np.real(evals).tolist(), evecs.T):
        if eval_value <= tol:
            continue
        c_op = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
        block = math.sqrt(float(eval_value)) * np.asarray(evec, dtype=np.complex128).reshape(
            (target_count, source_count)
        )
        c_op[np.ix_(target_indices, source_indices)] = block
        jumps.append(c_op)
    if not jumps:
        return np.zeros((0, total_dimension, total_dimension), dtype=np.complex128)
    return np.asarray(jumps, dtype=np.complex128)


def _c_array_from_target_decay_kernels(
    *,
    target_indices: np.ndarray,
    excited_indices: np.ndarray,
    kernels: Sequence[np.ndarray],
    total_dimension: int,
    tol: float = 1e-12,
) -> np.ndarray:
    target_indices = np.asarray(target_indices, dtype=np.int64)
    excited_indices = np.asarray(excited_indices, dtype=np.int64)
    jumps: list[np.ndarray] = []
    for target_index, kernel in zip(target_indices.tolist(), kernels):
        kernel_psd = _psd_project(kernel, tol=tol)
        evals, evecs = np.linalg.eigh(kernel_psd)
        for eval_value, evec in zip(np.real(evals).tolist(), evecs.T):
            if eval_value <= tol:
                continue
            c_op = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
            c_op[int(target_index), excited_indices] = math.sqrt(float(eval_value)) * np.asarray(
                evec, dtype=np.complex128
            ).conj()
            jumps.append(c_op)
    if not jumps:
        return np.zeros((0, total_dimension, total_dimension), dtype=np.complex128)
    return np.asarray(jumps, dtype=np.complex128)


def _sink_jump_operators(
    rate_matrix: np.ndarray,
    *,
    sink_index: int,
    active_positions: np.ndarray,
    total_dimension: int,
    tol: float = 1e-12,
) -> list[np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(_symmetrize(rate_matrix))
    jump_ops: list[np.ndarray] = []
    for eigval, eigvec in zip(eigvals, eigvecs.T):
        if eigval <= tol:
            continue
        jump = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
        jump[sink_index, active_positions] = np.sqrt(eigval) * np.conj(eigvec)
        jump_ops.append(jump)
    return jump_ops
