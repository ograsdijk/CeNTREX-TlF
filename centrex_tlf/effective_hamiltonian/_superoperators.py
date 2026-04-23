from __future__ import annotations

import numpy as np


def _lindblad_dissipator(
    c_array: np.ndarray,
    rho: np.ndarray,
    *,
    loss_operator: np.ndarray | None = None,
) -> np.ndarray:
    dissipator = np.zeros_like(rho, dtype=np.complex128)
    for c_op in c_array:
        c_dag = c_op.conj().T
        c_dag_c = c_dag @ c_op
        dissipator += c_op @ rho @ c_dag
        dissipator -= 0.5 * (c_dag_c @ rho + rho @ c_dag_c)
    if loss_operator is not None and loss_operator.size:
        dissipator -= 0.5 * (loss_operator @ rho + rho @ loss_operator)
    return dissipator


def _dissipator_superoperator(
    c_array: np.ndarray,
    *,
    loss_operator: np.ndarray | None = None,
) -> np.ndarray:
    if c_array.ndim != 3:
        raise ValueError("c_array must have shape (n_ops, n_states, n_states)")
    n_states = c_array.shape[1]
    identity = np.eye(n_states, dtype=np.complex128)
    superoperator = np.zeros((n_states * n_states, n_states * n_states), dtype=np.complex128)
    for c_op in c_array:
        c_dag_c = c_op.conj().T @ c_op
        superoperator += np.kron(c_op, c_op.conj())
        superoperator -= 0.5 * np.kron(identity, c_dag_c.T)
        superoperator -= 0.5 * np.kron(c_dag_c, identity)
    if loss_operator is not None and loss_operator.size:
        superoperator -= 0.5 * np.kron(identity, loss_operator.T)
        superoperator -= 0.5 * np.kron(loss_operator, identity)
    return superoperator


def _hamiltonian_superoperator(hamiltonian_total: np.ndarray) -> np.ndarray:
    n_states = hamiltonian_total.shape[0]
    identity = np.eye(n_states, dtype=np.complex128)
    return -1j * (
        np.kron(hamiltonian_total, identity)
        - np.kron(identity, hamiltonian_total.T)
    )


def _unitary_superoperator(unitary: np.ndarray) -> np.ndarray:
    unitary = np.asarray(unitary, dtype=np.complex128)
    return np.asarray(np.kron(unitary, unitary.conj()), dtype=np.complex128)


def _transform_superoperator(superoperator: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    transform = _unitary_superoperator(unitary)
    return np.asarray(transform @ np.asarray(superoperator, dtype=np.complex128) @ transform.conj().T)


def _embed_superoperator_to_union_layout(
    superoperator: np.ndarray,
    local_to_union: dict[int, int],
    *,
    n_local: int,
    n_union: int,
) -> np.ndarray:
    embedding = np.zeros((n_union * n_union, n_local * n_local), dtype=np.complex128)
    for local_i, union_i in local_to_union.items():
        for local_j, union_j in local_to_union.items():
            embedding[union_i * n_union + union_j, local_i * n_local + local_j] = 1.0
    return np.asarray(
        embedding @ np.asarray(superoperator, dtype=np.complex128) @ embedding.conj().T,
        dtype=np.complex128,
    )
