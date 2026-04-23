from __future__ import annotations

from itertools import permutations
from dataclasses import dataclass, replace
import math
import re
from typing import Callable, Sequence

import numpy as np
import sympy as smp
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, eigvals, logm, solve_sylvester
from scipy.optimize import linear_sum_assignment

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions


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


def _finite_difference_weights(
    offsets: np.ndarray,
    derivative_order: int,
) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64)
    n_points = offsets.size
    powers = np.arange(n_points, dtype=np.int64)
    vandermonde = np.vstack([offsets**power for power in powers])
    rhs = np.zeros(n_points, dtype=np.float64)
    rhs[derivative_order] = float(math.factorial(derivative_order))
    return np.linalg.solve(vandermonde, rhs)


def _polynomial_coefficients_from_samples(
    sample_values: Sequence[np.ndarray | float | complex],
    offsets: np.ndarray,
    order: int,
) -> list[np.ndarray]:
    coeffs: list[np.ndarray] = []
    for derivative_order in range(order + 1):
        weights = _finite_difference_weights(offsets, derivative_order) / float(
            math.factorial(derivative_order)
        )
        coeff = sum(
            complex(weight) * np.asarray(value, dtype=np.complex128)
            for weight, value in zip(weights, sample_values)
        )
        coeffs.append(np.asarray(coeff, dtype=np.complex128))
    return coeffs


def _evaluate_polynomial_coefficients(
    coefficients: Sequence[np.ndarray | float | complex],
    delta: float,
) -> np.ndarray:
    result = np.asarray(coefficients[0], dtype=np.complex128).copy()
    power = 1.0
    for coefficient in coefficients[1:]:
        power *= float(delta)
        result = result + power * np.asarray(coefficient, dtype=np.complex128)
    return np.asarray(result, dtype=np.complex128)


def _polar_unitary(matrix: np.ndarray) -> np.ndarray:
    u_mat, _, vh_mat = np.linalg.svd(np.asarray(matrix, dtype=np.complex128), full_matrices=False)
    return np.asarray(u_mat @ vh_mat, dtype=np.complex128)


def _unitary_superoperator(unitary: np.ndarray) -> np.ndarray:
    unitary = np.asarray(unitary, dtype=np.complex128)
    return np.asarray(np.kron(unitary, unitary.conj()), dtype=np.complex128)


def _transform_superoperator(superoperator: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    transform = _unitary_superoperator(unitary)
    return np.asarray(transform @ np.asarray(superoperator, dtype=np.complex128) @ transform.conj().T)


def _transform_operator_bundle(
    bundle: OperatorBundle,
    unitary: np.ndarray,
    *,
    decay_kernel_ground: np.ndarray | None = None,
    decay_kernels_sinks: tuple[np.ndarray, ...] | None = None,
) -> OperatorBundle:
    unitary = np.asarray(unitary, dtype=np.complex128)
    h_internal = _symmetrize(unitary @ np.asarray(bundle.h_internal, dtype=np.complex128) @ unitary.conj().T)
    h_opt = _symmetrize(unitary @ np.asarray(bundle.h_opt, dtype=np.complex128) @ unitary.conj().T)
    h_det = _symmetrize(unitary @ np.asarray(bundle.h_det, dtype=np.complex128) @ unitary.conj().T)
    h_full_internal = _symmetrize(
        unitary @ np.asarray(bundle.h_full_internal, dtype=np.complex128) @ unitary.conj().T
    )
    h_lab_internal = _symmetrize(
        unitary @ np.asarray(bundle.h_lab_internal, dtype=np.complex128) @ unitary.conj().T
    )
    c_array = np.asarray(
        [
            unitary @ np.asarray(c_op, dtype=np.complex128) @ unitary.conj().T
            for c_op in np.asarray(bundle.c_array, dtype=np.complex128)
        ],
        dtype=np.complex128,
    )
    dissipator_superop = _transform_superoperator(bundle.dissipator_superoperator(), unitary)
    jump_rate_operator = _symmetrize(
        unitary @ np.asarray(bundle.jump_rate_operator(), dtype=np.complex128) @ unitary.conj().T
    )

    if decay_kernel_ground is None:
        decay_kernel_ground = bundle.decay_kernel_ground
    if decay_kernels_sinks is None:
        decay_kernels_sinks = bundle.decay_kernels_sinks

    excited_to_ground_rates_hz = None
    if decay_kernel_ground is not None:
        decay_kernel_ground = _symmetrize(np.asarray(decay_kernel_ground, dtype=np.complex128))
        excited_to_ground_rates_hz = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)

    excited_to_sink_rates_hz = None
    if decay_kernels_sinks is not None:
        decay_kernels_sinks = tuple(_symmetrize(np.asarray(kernel, dtype=np.complex128)) for kernel in decay_kernels_sinks)
        excited_to_sink_rates_hz = np.sum(
            np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
            axis=0,
        ) / (2.0 * np.pi)

    return OperatorBundle(
        electric_field=np.asarray(bundle.electric_field, dtype=np.float64),
        magnetic_field=np.asarray(bundle.magnetic_field, dtype=np.float64),
        omega_reference=float(bundle.omega_reference),
        h_internal=h_internal,
        h_opt=h_opt,
        h_det=h_det,
        c_array=c_array,
        excited_indices=np.asarray(bundle.excited_indices, dtype=np.int64),
        loss_operator=np.zeros_like(h_internal, dtype=np.complex128),
        h_full_internal=h_full_internal,
        h_lab_internal=h_lab_internal,
        dissipator_superop=dissipator_superop,
        perturbative_ratio_max=bundle.perturbative_ratio_max,
        hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
        sylvester_residual_norm=bundle.sylvester_residual_norm,
        spectral_separation_min=bundle.spectral_separation_min,
        generator=bundle.generator,
        p_indices=bundle.p_indices,
        q_indices=bundle.q_indices,
        excited_to_ground_rates_hz=excited_to_ground_rates_hz,
        excited_to_sink_rates_hz=excited_to_sink_rates_hz,
        decay_kernel_ground=decay_kernel_ground,
        decay_kernels_sinks=decay_kernels_sinks,
        jump_rate_operator_override=jump_rate_operator,
    )


def _compact_effective_order_indices(
    system: lindblad.utils_setup.OBESystem,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    ground = _selector_indices(
        system.QN,
        states.QuantumSelector(J=0, electronic=states.ElectronicState.X),
    )
    excited = _selector_indices(
        system.QN,
        states.QuantumSelector(electronic=states.ElectronicState.B),
    )
    sink_groups = tuple(
        _selector_indices(
            system.QN,
            states.QuantumSelector(J=J, electronic=states.ElectronicState.X),
        )
        for J in (1, 2, 3)
    )
    return ground, sink_groups, excited


def _state_key(state: states.CoupledState) -> str:
    if isinstance(state, states.CoupledBasisState):
        return repr(1.0 * state)
    return repr(state)


def _compact_patch_role_indices(
    system: lindblad.utils_setup.OBESystem,
    bundle: OperatorBundle,
    *,
    optical_tol: float = 1e-10,
    ground_selector: states.QuantumSelector | None = None,
    parent_basis: Sequence[states.CoupledState] | None = None,
    reference_ground_basis: np.ndarray | None = None,
    n_ground_keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    excited = _selector_indices(
        system.QN,
        states.QuantumSelector(electronic=states.ElectronicState.B),
    )
    lower = np.setdiff1d(np.arange(bundle.h_internal.shape[0], dtype=np.int64), excited, assume_unique=True)
    if lower.size == 0:
        return lower, np.array([], dtype=np.int64), excited
    if ground_selector is not None:
        ground = _selector_indices(system.QN, ground_selector).astype(np.int64)
        sinks = np.setdiff1d(lower, ground, assume_unique=True).astype(np.int64)
        return ground, sinks, excited.astype(np.int64)
    if parent_basis is not None and reference_ground_basis is not None and n_ground_keep is not None:
        candidate_basis = _basis_transform(
            [system.QN[int(index)] for index in lower.tolist()],
            parent_basis,
        )
        reference_ground_basis = np.asarray(reference_ground_basis, dtype=np.complex128)
        projector = reference_ground_basis @ reference_ground_basis.conj().T
        weights = np.real(
            np.einsum(
                "ij,ji->i",
                np.asarray(candidate_basis, dtype=np.complex128).conj().T,
                projector @ np.asarray(candidate_basis, dtype=np.complex128),
            )
        )
        ordering = np.argsort(-weights, kind="stable")
        selected = np.sort(lower[ordering[: int(n_ground_keep)]].astype(np.int64))
        sinks = np.setdiff1d(lower, selected, assume_unique=True).astype(np.int64)
        return selected, sinks, excited.astype(np.int64)
    if excited.size == 0:
        return lower, np.array([], dtype=np.int64), excited
    strengths = np.sum(
        np.abs(np.asarray(bundle.h_opt[np.ix_(lower, excited)], dtype=np.complex128)) ** 2,
        axis=1,
    )
    scale = max(float(np.max(strengths)), 1.0)
    ground = lower[strengths > optical_tol * scale].astype(np.int64)
    sinks = np.setdiff1d(lower, ground, assume_unique=True).astype(np.int64)
    return ground, sinks, excited.astype(np.int64)


def _build_union_compact_state_order(
    patch_systems: Sequence[lindblad.utils_setup.OBESystem],
    patch_bundles: Sequence[OperatorBundle],
    parent_basis: Sequence[states.CoupledState],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    parent_order = {_state_key(state): idx for idx, state in enumerate(parent_basis)}
    ground_keys: dict[str, None] = {}
    sink_keys: dict[str, None] = {}
    excited_keys: dict[str, None] = {}
    for system, bundle in zip(patch_systems, patch_bundles):
        ground, sinks, excited = _compact_patch_role_indices(system, bundle)
        for index in ground.tolist():
            key = _state_key(system.QN[int(index)])
            ground_keys[key] = None
            sink_keys.pop(key, None)
        for index in sinks.tolist():
            key = _state_key(system.QN[int(index)])
            if key not in ground_keys:
                sink_keys[key] = None
        for index in excited.tolist():
            excited_keys[_state_key(system.QN[int(index)])] = None

    def key_sorter(key: str) -> tuple[int, str]:
        return (int(parent_order.get(key, 10**9)), key)

    ordered_ground = sorted(ground_keys.keys(), key=key_sorter)
    ordered_sinks = sorted(sink_keys.keys(), key=key_sorter)
    ordered_excited = sorted(excited_keys.keys(), key=key_sorter)
    union_keys = ordered_ground + ordered_sinks + ordered_excited
    ground_indices = np.arange(0, len(ordered_ground), dtype=np.int64)
    sink_indices = np.arange(len(ordered_ground), len(ordered_ground) + len(ordered_sinks), dtype=np.int64)
    excited_indices = np.arange(
        len(ordered_ground) + len(ordered_sinks),
        len(union_keys),
        dtype=np.int64,
    )
    return union_keys, ground_indices, sink_indices, excited_indices


def _embed_compact_bundle_to_union_order(
    system: lindblad.utils_setup.OBESystem,
    bundle: OperatorBundle,
    parent_basis: Sequence[states.CoupledState],
    union_keys: Sequence[str],
    union_excited_indices: np.ndarray,
) -> tuple[np.ndarray, OperatorBundle]:
    union_lookup = {key: idx for idx, key in enumerate(union_keys)}
    mapping: dict[int, int] = {}
    for old_idx, state in enumerate(system.QN):
        key = _state_key(state)
        if key not in union_lookup:
            raise ValueError(f"Compact state {key} is missing from the union basis.")
        mapping[int(old_idx)] = int(union_lookup[key])

    n_new = len(union_keys)
    basis_vectors_old = _basis_transform(list(system.QN), parent_basis)
    basis_vectors = np.zeros((basis_vectors_old.shape[0], n_new), dtype=np.complex128)
    for old_idx, new_idx in mapping.items():
        basis_vectors[:, new_idx] = basis_vectors_old[:, old_idx]

    def embed_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.complex128)
        embedded = np.zeros((n_new, n_new), dtype=np.complex128)
        for old_i, new_i in mapping.items():
            for old_j, new_j in mapping.items():
                embedded[new_i, new_j] = matrix[old_i, old_j]
        return embedded

    c_array_old = np.asarray(bundle.c_array, dtype=np.complex128)
    c_array_new = np.zeros((c_array_old.shape[0], n_new, n_new), dtype=np.complex128)
    for op_index, c_op in enumerate(c_array_old):
        for old_i, new_i in mapping.items():
            for old_j, new_j in mapping.items():
                c_array_new[op_index, new_i, new_j] = c_op[old_i, old_j]

    _, _, excited_old = _compact_patch_role_indices(system, bundle)
    union_excited_lookup = {int(index): idx for idx, index in enumerate(union_excited_indices.tolist())}

    def embed_excited_kernel(kernel: np.ndarray | None) -> np.ndarray | None:
        if kernel is None:
            return None
        kernel = np.asarray(kernel, dtype=np.complex128)
        embedded = np.zeros((len(union_excited_indices), len(union_excited_indices)), dtype=np.complex128)
        for old_i_local, old_i in enumerate(excited_old.tolist()):
            new_i_local = union_excited_lookup[mapping[int(old_i)]]
            for old_j_local, old_j in enumerate(excited_old.tolist()):
                new_j_local = union_excited_lookup[mapping[int(old_j)]]
                embedded[new_i_local, new_j_local] = kernel[old_i_local, old_j_local]
        return _symmetrize(embedded)

    decay_kernel_ground = embed_excited_kernel(bundle.decay_kernel_ground)
    decay_kernels_sinks = None
    if bundle.decay_kernels_sinks is not None:
        decay_kernels_sinks = tuple(
            embed_excited_kernel(kernel) for kernel in bundle.decay_kernels_sinks
        )

    excited_to_ground_rates_hz = None
    if decay_kernel_ground is not None:
        excited_to_ground_rates_hz = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)

    excited_to_sink_rates_hz = None
    if decay_kernels_sinks is not None:
        excited_to_sink_rates_hz = np.sum(
            np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
            axis=0,
        ) / (2.0 * np.pi)

    embedded_h_internal = embed_matrix(bundle.h_internal)
    bundle_embedded = OperatorBundle(
        electric_field=np.asarray(bundle.electric_field, dtype=np.float64),
        magnetic_field=np.asarray(bundle.magnetic_field, dtype=np.float64),
        omega_reference=float(bundle.omega_reference),
        h_internal=_symmetrize(embedded_h_internal),
        h_opt=_symmetrize(embed_matrix(bundle.h_opt)),
        h_det=_symmetrize(embed_matrix(bundle.h_det)),
        c_array=c_array_new,
        excited_indices=np.asarray(union_excited_indices, dtype=np.int64),
        loss_operator=np.zeros((n_new, n_new), dtype=np.complex128),
        h_full_internal=_symmetrize(embed_matrix(bundle.h_full_internal)),
        h_lab_internal=_symmetrize(embed_matrix(bundle.h_lab_internal)),
        hermiticity_error=float(np.max(np.abs(embedded_h_internal - embedded_h_internal.conj().T))),
        excited_to_ground_rates_hz=excited_to_ground_rates_hz,
        excited_to_sink_rates_hz=excited_to_sink_rates_hz,
        decay_kernel_ground=decay_kernel_ground,
        decay_kernels_sinks=decay_kernels_sinks,
    )
    return basis_vectors, bundle_embedded


def _optically_bright_ground_index(
    bundle: OperatorBundle,
    excited_indices: np.ndarray,
) -> int:
    ground_indices = np.setdiff1d(
        np.arange(bundle.h_internal.shape[0], dtype=np.int64),
        np.asarray(excited_indices, dtype=np.int64),
        assume_unique=True,
    )
    strengths = np.sum(
        np.abs(np.asarray(bundle.h_opt[np.ix_(ground_indices, excited_indices)], dtype=np.complex128)) ** 2,
        axis=1,
    )
    return int(ground_indices[int(np.argmax(strengths))])


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


def _make_bundle_sinks_dissipative_only(
    bundle: OperatorBundle,
    sink_indices: np.ndarray,
) -> OperatorBundle:
    sink_indices = np.asarray(sink_indices, dtype=np.int64)
    if sink_indices.size == 0:
        return bundle

    def zero_sink_blocks(matrix: np.ndarray) -> np.ndarray:
        reduced = np.asarray(matrix, dtype=np.complex128).copy()
        reduced[sink_indices, :] = 0.0
        reduced[:, sink_indices] = 0.0
        return _symmetrize(reduced)

    return OperatorBundle(
        electric_field=np.asarray(bundle.electric_field, dtype=np.float64),
        magnetic_field=np.asarray(bundle.magnetic_field, dtype=np.float64),
        omega_reference=float(bundle.omega_reference),
        h_internal=zero_sink_blocks(bundle.h_internal),
        h_opt=zero_sink_blocks(bundle.h_opt),
        h_det=zero_sink_blocks(bundle.h_det),
        c_array=np.asarray(bundle.c_array, dtype=np.complex128),
        excited_indices=np.asarray(bundle.excited_indices, dtype=np.int64),
        loss_operator=np.asarray(bundle.loss_operator, dtype=np.complex128),
        h_full_internal=zero_sink_blocks(bundle.h_full_internal),
        h_lab_internal=zero_sink_blocks(bundle.h_lab_internal),
        dissipator_superop=np.asarray(bundle.dissipator_superop, dtype=np.complex128)
        if bundle.dissipator_superop is not None
        else None,
        perturbative_ratio_max=bundle.perturbative_ratio_max,
        hermiticity_error=float(
            np.max(
                np.abs(
                    zero_sink_blocks(bundle.h_internal)
                    - zero_sink_blocks(bundle.h_internal).conj().T
                )
            )
        ),
        sylvester_residual_norm=bundle.sylvester_residual_norm,
        spectral_separation_min=bundle.spectral_separation_min,
        generator=bundle.generator,
        p_indices=bundle.p_indices,
        q_indices=bundle.q_indices,
        excited_to_ground_rates_hz=np.asarray(bundle.excited_to_ground_rates_hz, dtype=np.float64)
        if bundle.excited_to_ground_rates_hz is not None
        else None,
        excited_to_sink_rates_hz=np.asarray(bundle.excited_to_sink_rates_hz, dtype=np.float64)
        if bundle.excited_to_sink_rates_hz is not None
        else None,
        decay_kernel_ground=np.asarray(bundle.decay_kernel_ground, dtype=np.complex128)
        if bundle.decay_kernel_ground is not None
        else None,
        decay_kernels_sinks=tuple(np.asarray(kernel, dtype=np.complex128) for kernel in bundle.decay_kernels_sinks)
        if bundle.decay_kernels_sinks is not None
        else None,
        jump_rate_operator_override=np.asarray(bundle.jump_rate_operator_override, dtype=np.complex128)
        if bundle.jump_rate_operator_override is not None
        else None,
    )


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


def _build_compact_union_layout(
    patch_systems: Sequence[lindblad.utils_setup.OBESystem],
    patch_bundles: Sequence[OperatorBundle],
    parent_basis: Sequence[states.CoupledState],
    *,
    master_index: int,
    ground_selector: states.QuantumSelector,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    parent_order = {_state_key(state): idx for idx, state in enumerate(parent_basis)}
    master_system = patch_systems[master_index]
    master_bundle = patch_bundles[master_index]
    master_ground, _, master_excited = _compact_patch_role_indices(
        master_system,
        master_bundle,
        ground_selector=ground_selector,
    )
    n_ground = int(master_ground.size)
    n_excited = int(master_excited.size)

    sink_keys: dict[str, None] = {}
    for system, bundle in zip(patch_systems, patch_bundles):
        ground, sinks, excited = _compact_patch_role_indices(
            system,
            bundle,
            ground_selector=ground_selector,
        )
        if ground.size != n_ground:
            raise ValueError(
                "Compact atlas patches must have a consistent number of optically active lower states; "
                f"got {ground.size} versus master {n_ground}."
            )
        if excited.size != n_excited:
            raise ValueError(
                "Compact atlas patches must have a consistent number of excited states; "
                f"got {excited.size} versus master {n_excited}."
            )
        for index in sinks.tolist():
            sink_keys[_state_key(system.QN[int(index)])] = None

    def key_sorter(key: str) -> tuple[int, str]:
        return (int(parent_order.get(key, 10**9)), key)

    ordered_sink_keys = tuple(sorted(sink_keys.keys(), key=key_sorter))
    ground_indices = np.arange(0, n_ground, dtype=np.int64)
    sink_indices = np.arange(n_ground, n_ground + len(ordered_sink_keys), dtype=np.int64)
    excited_indices = np.arange(
        n_ground + len(ordered_sink_keys),
        n_ground + len(ordered_sink_keys) + n_excited,
        dtype=np.int64,
    )
    reference_ground_basis = _basis_transform(
        [master_system.QN[int(index)] for index in master_ground.tolist()],
        parent_basis,
    )
    reference_excited_basis = _basis_transform(
        [master_system.QN[int(index)] for index in master_excited.tolist()],
        parent_basis,
    )
    return (
        ground_indices,
        sink_indices,
        excited_indices,
        ordered_sink_keys,
        np.asarray(reference_ground_basis, dtype=np.complex128),
        np.asarray(reference_excited_basis, dtype=np.complex128),
    )


def _embed_aligned_compact_patch_to_union_layout(
    system: lindblad.utils_setup.OBESystem,
    bundle: OperatorBundle,
    parent_basis: Sequence[states.CoupledState],
    *,
    ground_selector: states.QuantumSelector,
    sink_union_keys: Sequence[str],
    ground_indices_union: np.ndarray,
    excited_indices_union: np.ndarray,
    reference_ground_basis: np.ndarray,
    reference_excited_basis: np.ndarray,
    use_identity_rotation: bool = False,
) -> tuple[np.ndarray, OperatorBundle]:
    ground, sinks, excited = _compact_patch_role_indices(
        system,
        bundle,
        ground_selector=ground_selector,
    )
    n_ground = int(ground_indices_union.size)
    n_excited = int(excited_indices_union.size)
    if ground.size != n_ground:
        raise ValueError(f"Patch has {ground.size} ground states, expected {n_ground}.")
    if excited.size != n_excited:
        raise ValueError(f"Patch has {excited.size} excited states, expected {n_excited}.")

    permutation_order = np.concatenate([ground, sinks, excited]).astype(np.int64)
    permutation = np.eye(bundle.h_internal.shape[0], dtype=np.complex128)[permutation_order, :]

    sink_basis = _basis_transform(
        [system.QN[int(index)] for index in sinks.tolist()],
        parent_basis,
    ) if sinks.size else np.zeros((len(parent_basis), 0), dtype=np.complex128)
    ground_basis = _basis_transform(
        [system.QN[int(index)] for index in ground.tolist()],
        parent_basis,
    )
    excited_basis = _basis_transform(
        [system.QN[int(index)] for index in excited.tolist()],
        parent_basis,
    )
    if use_identity_rotation:
        ground_rotation = np.eye(n_ground, dtype=np.complex128)
        excited_rotation = np.eye(n_excited, dtype=np.complex128)
    else:
        ground_rotation = _polar_unitary(
            reference_ground_basis.conj().T @ np.asarray(ground_basis, dtype=np.complex128)
        )
        excited_rotation = _polar_unitary(
            reference_excited_basis.conj().T @ np.asarray(excited_basis, dtype=np.complex128)
        )
    local_rotation = block_diag(
        ground_rotation,
        np.eye(sinks.size, dtype=np.complex128),
        excited_rotation,
    ).astype(np.complex128)
    total_unitary = np.asarray(local_rotation @ permutation, dtype=np.complex128)
    transformed_bundle = _transform_operator_bundle(
        bundle,
        total_unitary,
        decay_kernel_ground=_symmetrize(
            excited_rotation
            @ np.asarray(bundle.decay_kernel_ground, dtype=np.complex128)
            @ excited_rotation.conj().T
        )
        if bundle.decay_kernel_ground is not None
        else None,
        decay_kernels_sinks=tuple(
            _symmetrize(
                excited_rotation @ np.asarray(kernel, dtype=np.complex128) @ excited_rotation.conj().T
            )
            for kernel in (bundle.decay_kernels_sinks or ())
        )
        if bundle.decay_kernels_sinks is not None
        else None,
    )

    n_union = int(ground_indices_union.size + len(sink_union_keys) + excited_indices_union.size)
    sink_lookup = {key: idx for idx, key in enumerate(sink_union_keys)}
    local_to_union: dict[int, int] = {}
    for local_idx in range(n_ground):
        local_to_union[local_idx] = int(ground_indices_union[local_idx])
    for sink_offset, old_idx in enumerate(sinks.tolist()):
        sink_key = _state_key(system.QN[int(old_idx)])
        if sink_key not in sink_lookup:
            raise ValueError(f"Sink state {sink_key} is missing from the union sink layout.")
        local_to_union[n_ground + sink_offset] = int(ground_indices_union.size + sink_lookup[sink_key])
    for excited_offset in range(n_excited):
        local_to_union[n_ground + sinks.size + excited_offset] = int(excited_indices_union[excited_offset])

    aligned_ground_basis = np.asarray(ground_basis, dtype=np.complex128) @ ground_rotation.conj().T
    aligned_sink_basis = np.asarray(sink_basis, dtype=np.complex128)
    aligned_excited_basis = np.asarray(excited_basis, dtype=np.complex128) @ excited_rotation.conj().T
    basis_vectors = np.zeros((len(parent_basis), n_union), dtype=np.complex128)
    for local_idx in range(n_ground):
        basis_vectors[:, local_to_union[local_idx]] = aligned_ground_basis[:, local_idx]
    for sink_offset in range(sinks.size):
        basis_vectors[:, local_to_union[n_ground + sink_offset]] = aligned_sink_basis[:, sink_offset]
    for excited_offset in range(n_excited):
        basis_vectors[:, local_to_union[n_ground + sinks.size + excited_offset]] = aligned_excited_basis[:, excited_offset]

    def embed_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.complex128)
        embedded = np.zeros((n_union, n_union), dtype=np.complex128)
        for local_i, union_i in local_to_union.items():
            for local_j, union_j in local_to_union.items():
                embedded[union_i, union_j] = matrix[local_i, local_j]
        return embedded

    c_array_old = np.asarray(transformed_bundle.c_array, dtype=np.complex128)
    c_array_new = np.zeros((c_array_old.shape[0], n_union, n_union), dtype=np.complex128)
    for op_index, c_op in enumerate(c_array_old):
        for local_i, union_i in local_to_union.items():
            for local_j, union_j in local_to_union.items():
                c_array_new[op_index, union_i, union_j] = c_op[local_i, local_j]

    embedded_h_internal = embed_matrix(transformed_bundle.h_internal)
    embedded_dissipator = _embed_superoperator_to_union_layout(
        transformed_bundle.dissipator_superoperator(),
        local_to_union,
        n_local=transformed_bundle.h_internal.shape[0],
        n_union=n_union,
    )
    embedded_jump_rate = _symmetrize(embed_matrix(transformed_bundle.jump_rate_operator()))
    bundle_embedded = OperatorBundle(
        electric_field=np.asarray(bundle.electric_field, dtype=np.float64),
        magnetic_field=np.asarray(bundle.magnetic_field, dtype=np.float64),
        omega_reference=float(transformed_bundle.omega_reference),
        h_internal=_symmetrize(embedded_h_internal),
        h_opt=_symmetrize(embed_matrix(transformed_bundle.h_opt)),
        h_det=_symmetrize(embed_matrix(transformed_bundle.h_det)),
        c_array=c_array_new,
        excited_indices=np.asarray(excited_indices_union, dtype=np.int64),
        loss_operator=np.zeros((n_union, n_union), dtype=np.complex128),
        h_full_internal=_symmetrize(embed_matrix(transformed_bundle.h_full_internal)),
        h_lab_internal=_symmetrize(embed_matrix(transformed_bundle.h_lab_internal)),
        dissipator_superop=embedded_dissipator,
        hermiticity_error=float(np.max(np.abs(embedded_h_internal - embedded_h_internal.conj().T))),
        excited_to_ground_rates_hz=transformed_bundle.excited_to_ground_rates_hz,
        excited_to_sink_rates_hz=transformed_bundle.excited_to_sink_rates_hz,
        decay_kernel_ground=transformed_bundle.decay_kernel_ground,
        decay_kernels_sinks=transformed_bundle.decay_kernels_sinks,
        jump_rate_operator_override=embedded_jump_rate,
    )
    return basis_vectors, bundle_embedded


def _embed_tracked_compact_patch_to_union_layout(
    system: lindblad.utils_setup.OBESystem,
    bundle: OperatorBundle,
    parent_basis: Sequence[states.CoupledState],
    *,
    ground_selector: states.QuantumSelector,
    sink_union_keys: Sequence[str],
    ground_indices_union: np.ndarray,
    excited_indices_union: np.ndarray,
    reference_ground_basis: np.ndarray | None = None,
    reference_excited_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, OperatorBundle]:
    ground, sinks, excited = _compact_patch_role_indices(
        system,
        bundle,
        ground_selector=ground_selector,
    )
    n_ground = int(ground_indices_union.size)
    n_excited = int(excited_indices_union.size)
    if ground.size != n_ground:
        raise ValueError(f"Patch has {ground.size} ground states, expected {n_ground}.")
    if excited.size != n_excited:
        raise ValueError(f"Patch has {excited.size} excited states, expected {n_excited}.")

    permutation_order = np.concatenate([ground, sinks, excited]).astype(np.int64)
    permutation = np.eye(bundle.h_internal.shape[0], dtype=np.complex128)[permutation_order, :]

    sink_basis = (
        _basis_transform([system.QN[int(index)] for index in sinks.tolist()], parent_basis)
        if sinks.size
        else np.zeros((len(parent_basis), 0), dtype=np.complex128)
    )
    ground_basis = _basis_transform(
        [system.QN[int(index)] for index in ground.tolist()],
        parent_basis,
    )
    excited_basis = _basis_transform(
        [system.QN[int(index)] for index in excited.tolist()],
        parent_basis,
    )
    if reference_ground_basis is None:
        tracked_ground_basis = np.asarray(ground_basis, dtype=np.complex128)
        ground_rotation = np.eye(n_ground, dtype=np.complex128)
    else:
        tracked_ground_basis, ground_rotation = _tracking_rotation(
            np.asarray(reference_ground_basis, dtype=np.complex128),
            np.asarray(ground_basis, dtype=np.complex128),
        )
    if reference_excited_basis is None:
        tracked_excited_basis = np.asarray(excited_basis, dtype=np.complex128)
        excited_rotation = np.eye(n_excited, dtype=np.complex128)
    else:
        tracked_excited_basis, excited_rotation = _tracking_rotation(
            np.asarray(reference_excited_basis, dtype=np.complex128),
            np.asarray(excited_basis, dtype=np.complex128),
        )
    local_rotation = block_diag(
        ground_rotation,
        np.eye(sinks.size, dtype=np.complex128),
        excited_rotation,
    ).astype(np.complex128)
    total_unitary = np.asarray(local_rotation.conj().T @ permutation, dtype=np.complex128)
    transformed_bundle = _transform_operator_bundle(
        bundle,
        total_unitary,
        decay_kernel_ground=_symmetrize(
            excited_rotation.conj().T
            @ np.asarray(bundle.decay_kernel_ground, dtype=np.complex128)
            @ excited_rotation
        )
        if bundle.decay_kernel_ground is not None
        else None,
        decay_kernels_sinks=tuple(
            _symmetrize(
                excited_rotation.conj().T @ np.asarray(kernel, dtype=np.complex128) @ excited_rotation
            )
            for kernel in (bundle.decay_kernels_sinks or ())
        )
        if bundle.decay_kernels_sinks is not None
        else None,
    )

    n_union = int(ground_indices_union.size + len(sink_union_keys) + excited_indices_union.size)
    sink_lookup = {key: idx for idx, key in enumerate(sink_union_keys)}
    local_to_union: dict[int, int] = {}
    for local_idx in range(n_ground):
        local_to_union[local_idx] = int(ground_indices_union[local_idx])
    for sink_offset, old_idx in enumerate(sinks.tolist()):
        sink_key = _state_key(system.QN[int(old_idx)])
        if sink_key not in sink_lookup:
            raise ValueError(f"Sink state {sink_key} is missing from the union sink layout.")
        local_to_union[n_ground + sink_offset] = int(ground_indices_union.size + sink_lookup[sink_key])
    for excited_offset in range(n_excited):
        local_to_union[n_ground + sinks.size + excited_offset] = int(excited_indices_union[excited_offset])

    coherent_basis = np.concatenate([tracked_ground_basis, tracked_excited_basis], axis=1)
    basis_vectors = np.zeros((len(parent_basis), n_union), dtype=np.complex128)
    for local_idx in range(n_ground):
        basis_vectors[:, local_to_union[local_idx]] = tracked_ground_basis[:, local_idx]
    for sink_offset in range(sinks.size):
        basis_vectors[:, local_to_union[n_ground + sink_offset]] = sink_basis[:, sink_offset]
    for excited_offset in range(n_excited):
        basis_vectors[:, local_to_union[n_ground + sinks.size + excited_offset]] = tracked_excited_basis[:, excited_offset]

    def embed_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.complex128)
        embedded = np.zeros((n_union, n_union), dtype=np.complex128)
        for local_i, union_i in local_to_union.items():
            for local_j, union_j in local_to_union.items():
                embedded[union_i, union_j] = matrix[local_i, local_j]
        return embedded

    c_array_old = np.asarray(transformed_bundle.c_array, dtype=np.complex128)
    c_array_new = np.zeros((c_array_old.shape[0], n_union, n_union), dtype=np.complex128)
    for op_index, c_op in enumerate(c_array_old):
        for local_i, union_i in local_to_union.items():
            for local_j, union_j in local_to_union.items():
                c_array_new[op_index, union_i, union_j] = c_op[local_i, local_j]

    embedded_h_internal = embed_matrix(transformed_bundle.h_internal)
    embedded_dissipator = _embed_superoperator_to_union_layout(
        transformed_bundle.dissipator_superoperator(),
        local_to_union,
        n_local=transformed_bundle.h_internal.shape[0],
        n_union=n_union,
    )
    embedded_jump_rate = _symmetrize(embed_matrix(transformed_bundle.jump_rate_operator()))
    bundle_embedded = OperatorBundle(
        electric_field=np.asarray(bundle.electric_field, dtype=np.float64),
        magnetic_field=np.asarray(bundle.magnetic_field, dtype=np.float64),
        omega_reference=float(transformed_bundle.omega_reference),
        h_internal=_symmetrize(embedded_h_internal),
        h_opt=_symmetrize(embed_matrix(transformed_bundle.h_opt)),
        h_det=_symmetrize(embed_matrix(transformed_bundle.h_det)),
        c_array=c_array_new,
        excited_indices=np.asarray(excited_indices_union, dtype=np.int64),
        loss_operator=np.zeros((n_union, n_union), dtype=np.complex128),
        h_full_internal=_symmetrize(embed_matrix(transformed_bundle.h_full_internal)),
        h_lab_internal=_symmetrize(embed_matrix(transformed_bundle.h_lab_internal)),
        dissipator_superop=embedded_dissipator,
        hermiticity_error=float(np.max(np.abs(embedded_h_internal - embedded_h_internal.conj().T))),
        excited_to_ground_rates_hz=transformed_bundle.excited_to_ground_rates_hz,
        excited_to_sink_rates_hz=transformed_bundle.excited_to_sink_rates_hz,
        decay_kernel_ground=transformed_bundle.decay_kernel_ground,
        decay_kernels_sinks=transformed_bundle.decay_kernels_sinks,
        jump_rate_operator_override=embedded_jump_rate,
    )
    return basis_vectors, coherent_basis, bundle_embedded


def _embed_compact_bundle_to_effective_order(
    system: lindblad.utils_setup.OBESystem,
    bundle: OperatorBundle,
    parent_basis: Sequence[states.CoupledState],
) -> tuple[np.ndarray, OperatorBundle]:
    ground, sink_groups, excited = _compact_effective_order_indices(system)
    n_old = bundle.h_internal.shape[0]
    n_new = 10
    mapping: dict[int, int] = {}
    for new_idx, old_idx in enumerate(ground.tolist()):
        mapping[int(old_idx)] = int(new_idx)
    for sink_offset, sink_indices in enumerate(sink_groups):
        if sink_indices.size == 0:
            continue
        if sink_indices.size != 1:
            raise ValueError(
                "Compact sink sectors are expected to be single-state sectors in the exact compact model."
            )
        mapping[int(sink_indices[0])] = 4 + sink_offset
    for excited_offset, old_idx in enumerate(excited.tolist()):
        mapping[int(old_idx)] = 7 + excited_offset

    physical_order = np.concatenate([ground, excited]).astype(np.int64)
    basis_vectors = _basis_transform([system.QN[idx] for idx in physical_order.tolist()], parent_basis)

    def embed_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.complex128)
        embedded = np.zeros((n_new, n_new), dtype=np.complex128)
        for old_i, new_i in mapping.items():
            for old_j, new_j in mapping.items():
                embedded[new_i, new_j] = matrix[old_i, old_j]
        return embedded

    c_array_old = np.asarray(bundle.c_array, dtype=np.complex128)
    c_array_new = np.zeros((c_array_old.shape[0], n_new, n_new), dtype=np.complex128)
    for op_index, c_op in enumerate(c_array_old):
        for old_i, new_i in mapping.items():
            for old_j, new_j in mapping.items():
                c_array_new[op_index, new_i, new_j] = c_op[old_i, old_j]

    bundle_embedded = OperatorBundle(
        electric_field=np.asarray(bundle.electric_field, dtype=np.float64),
        magnetic_field=np.asarray(bundle.magnetic_field, dtype=np.float64),
        omega_reference=float(bundle.omega_reference),
        h_internal=_symmetrize(embed_matrix(bundle.h_internal)),
        h_opt=_symmetrize(embed_matrix(bundle.h_opt)),
        h_det=_symmetrize(embed_matrix(bundle.h_det)),
        c_array=c_array_new,
        excited_indices=np.arange(7, 10, dtype=np.int64),
        loss_operator=np.zeros((n_new, n_new), dtype=np.complex128),
        h_full_internal=_symmetrize(embed_matrix(bundle.h_full_internal)),
        h_lab_internal=_symmetrize(embed_matrix(bundle.h_lab_internal)),
        hermiticity_error=float(np.max(np.abs(embed_matrix(bundle.h_internal) - embed_matrix(bundle.h_internal).conj().T))),
        excited_to_ground_rates_hz=np.asarray(bundle.excited_to_ground_rates_hz, dtype=np.float64)
        if bundle.excited_to_ground_rates_hz is not None
        else None,
        excited_to_sink_rates_hz=np.asarray(bundle.excited_to_sink_rates_hz, dtype=np.float64)
        if bundle.excited_to_sink_rates_hz is not None
        else None,
        decay_kernel_ground=np.asarray(bundle.decay_kernel_ground, dtype=np.complex128)
        if bundle.decay_kernel_ground is not None
        else None,
        decay_kernels_sinks=tuple(np.asarray(kernel, dtype=np.complex128) for kernel in bundle.decay_kernels_sinks)
        if bundle.decay_kernels_sinks is not None
        else None,
    )
    return basis_vectors, bundle_embedded


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


def _smallest_left_singular_vector(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_vecs, singular_values, _ = np.linalg.svd(
        np.asarray(matrix, dtype=np.complex128),
        full_matrices=True,
    )
    return singular_values, np.asarray(left_vecs[:, -1], dtype=np.complex128)


def _singular_value_dark_dimension(
    singular_values: np.ndarray,
    n_ground: int,
    *,
    tol: float = 1e-8,
) -> int:
    if singular_values.size == 0:
        return n_ground
    scale = max(float(np.max(np.abs(singular_values))), 1.0)
    rank = int(np.count_nonzero(np.abs(singular_values) > tol * scale))
    return max(0, n_ground - rank)


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


def _psd_project(matrix: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    matrix = _symmetrize(np.asarray(matrix, dtype=np.complex128))
    evals, evecs = np.linalg.eigh(matrix)
    evals = np.where(np.real(evals) > tol, np.real(evals), 0.0)
    if np.count_nonzero(evals) == 0:
        return np.zeros_like(matrix, dtype=np.complex128)
    return _symmetrize((evecs * evals) @ evecs.conj().T)


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


def _density_from_state_vector(
    state_vector: np.ndarray,
    positions: np.ndarray,
    total_dimension: int,
) -> np.ndarray:
    density = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
    density[np.ix_(positions, positions)] = np.outer(state_vector, state_vector.conj())
    return density


def _matrix_touching_positions_norm(
    matrix: np.ndarray,
    positions: np.ndarray,
) -> float:
    if positions.size == 0:
        return 0.0
    mask = np.zeros(matrix.shape, dtype=bool)
    mask[positions, :] = True
    mask[:, positions] = True
    return float(np.linalg.norm(matrix[mask]))


def _dark_bright_basis(
    dark_vector: np.ndarray,
) -> np.ndarray:
    dark = np.asarray(dark_vector, dtype=np.complex128)
    dark /= np.linalg.norm(dark)
    basis = [dark]
    identity = np.eye(dark.size, dtype=np.complex128)
    for candidate in identity.T:
        vector = candidate.astype(np.complex128, copy=True)
        for existing in basis:
            vector -= existing * np.vdot(existing, vector)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            continue
        basis.append(vector / norm)
        if len(basis) == dark.size:
            break
    return np.column_stack(basis)


def _dark_to_bright_mixing_scale(
    h_ground: np.ndarray,
    dark_vector: np.ndarray,
) -> float:
    basis = _dark_bright_basis(dark_vector)
    rotated = basis.conj().T @ np.asarray(h_ground, dtype=np.complex128) @ basis
    return float(np.linalg.norm(rotated[1:, 0]))


def _truncated_state_transform(
    generator: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    if order < 0:
        raise ValueError(f"State-transform order must be non-negative, not {order}.")
    generator = np.asarray(generator, dtype=np.complex128)
    n_total = generator.shape[0]
    identity = np.eye(n_total, dtype=np.complex128)
    transform = identity.copy()
    term = identity.copy()
    for k in range(1, order + 1):
        term = (term @ generator) / float(k)
        transform += term
    return np.asarray(transform, dtype=np.complex128)


def _dressed_state_bases(
    generator: np.ndarray,
    n_p: int,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = _truncated_state_transform(generator, order=order)
    return (
        np.asarray(transform[:, :n_p], dtype=np.complex128),
        np.asarray(transform[:, n_p:], dtype=np.complex128),
    )


def _mixed_order_kept_vectors(
    generator: np.ndarray,
    n_p: int,
    excited_columns: np.ndarray,
    *,
    base_order: int = 2,
    excited_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_vectors_base, q_vectors = _dressed_state_bases(generator, n_p, order=base_order)
    p_vectors_mixed = np.array(p_vectors_base, copy=True)
    if excited_columns.size and excited_order != base_order:
        p_vectors_excited, _ = _dressed_state_bases(generator, n_p, order=excited_order)
        p_vectors_mixed[:, excited_columns] = p_vectors_excited[:, excited_columns]
    return p_vectors_base, q_vectors, p_vectors_mixed


def _matrix_elements_dressed_states(
    operator: np.ndarray,
    left_vectors: np.ndarray,
    right_vectors: np.ndarray | None = None,
) -> np.ndarray:
    right = left_vectors if right_vectors is None else right_vectors
    return np.asarray(
        left_vectors.conj().T @ np.asarray(operator, dtype=np.complex128) @ right,
        dtype=np.complex128,
    )


def _partitioned_vectors_to_full_basis(
    vectors_partitioned: np.ndarray,
    p_indices: np.ndarray,
    q_indices: np.ndarray,
) -> np.ndarray:
    vectors_partitioned = np.asarray(vectors_partitioned, dtype=np.complex128)
    n_total = int(p_indices.size + q_indices.size)
    n_vectors = int(vectors_partitioned.shape[1])
    vectors_full = np.zeros((n_total, n_vectors), dtype=np.complex128)
    vectors_full[np.ix_(p_indices, np.arange(n_vectors, dtype=np.int64))] = vectors_partitioned[
        : p_indices.size, :
    ]
    vectors_full[np.ix_(q_indices, np.arange(n_vectors, dtype=np.int64))] = vectors_partitioned[
        p_indices.size :, :
    ]
    return vectors_full


def _match_columns_by_overlap(
    reference_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlaps = np.abs(
        np.asarray(reference_vectors, dtype=np.complex128).conj().T
        @ np.asarray(candidate_vectors, dtype=np.complex128)
    )
    n = overlaps.shape[0]
    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    for perm in permutations(range(n)):
        score = float(sum(overlaps[i, perm[i]] for i in range(n)))
        if score > best_score:
            best_score = score
            best_perm = perm
    if best_perm is None:
        raise ValueError("Could not match state columns by overlap.")
    return overlaps, np.asarray(best_perm, dtype=np.int64)


def _top_component_summary(
    vector: np.ndarray,
    labels: Sequence[str],
    *,
    top_k: int = 8,
) -> list[tuple[str, float]]:
    weights = np.abs(np.asarray(vector, dtype=np.complex128)) ** 2
    order = np.argsort(weights)[::-1][:top_k]
    return [(labels[int(idx)], float(weights[int(idx)])) for idx in order if weights[int(idx)] > 0.0]


def _embed_block(
    matrix: np.ndarray,
    total_dimension: int,
    row_offset: int,
    col_offset: int,
) -> np.ndarray:
    embedded = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
    n_rows, n_cols = matrix.shape
    embedded[row_offset : row_offset + n_rows, col_offset : col_offset + n_cols] = matrix
    return embedded


def _embed_subspace_matrix(
    matrix: np.ndarray,
    positions: np.ndarray,
    total_dimension: int,
) -> np.ndarray:
    embedded = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
    embedded[np.ix_(positions, positions)] = matrix
    return embedded


@dataclass(frozen=True)
class MatrixBlocks:
    pp: np.ndarray
    pq: np.ndarray
    qp: np.ndarray
    qq: np.ndarray


def _zero_blocks_like(blocks: MatrixBlocks) -> MatrixBlocks:
    return MatrixBlocks(
        pp=np.zeros_like(blocks.pp),
        pq=np.zeros_like(blocks.pq),
        qp=np.zeros_like(blocks.qp),
        qq=np.zeros_like(blocks.qq),
    )


def _partition_matrix(
    matrix: np.ndarray,
    p_indices: np.ndarray,
    q_indices: np.ndarray,
) -> MatrixBlocks:
    return MatrixBlocks(
        pp=np.asarray(matrix[np.ix_(p_indices, p_indices)], dtype=np.complex128),
        pq=np.asarray(matrix[np.ix_(p_indices, q_indices)], dtype=np.complex128),
        qp=np.asarray(matrix[np.ix_(q_indices, p_indices)], dtype=np.complex128),
        qq=np.asarray(matrix[np.ix_(q_indices, q_indices)], dtype=np.complex128),
    )


def _assemble_full_from_blocks(blocks: MatrixBlocks) -> np.ndarray:
    return np.block([[blocks.pp, blocks.pq], [blocks.qp, blocks.qq]]).astype(np.complex128)


def _block_add(
    left: MatrixBlocks,
    right: MatrixBlocks,
    *,
    scale_right: complex = 1.0,
) -> MatrixBlocks:
    return MatrixBlocks(
        pp=left.pp + scale_right * right.pp,
        pq=left.pq + scale_right * right.pq,
        qp=left.qp + scale_right * right.qp,
        qq=left.qq + scale_right * right.qq,
    )


def _block_matmul(left: MatrixBlocks, right: MatrixBlocks) -> MatrixBlocks:
    return MatrixBlocks(
        pp=left.pp @ right.pp + left.pq @ right.qp,
        pq=left.pp @ right.pq + left.pq @ right.qq,
        qp=left.qp @ right.pp + left.qq @ right.qp,
        qq=left.qp @ right.pq + left.qq @ right.qq,
    )


def _block_commutator(left: MatrixBlocks, right: MatrixBlocks) -> MatrixBlocks:
    product_lr = _block_matmul(left, right)
    product_rl = _block_matmul(right, left)
    return MatrixBlocks(
        pp=product_lr.pp - product_rl.pp,
        pq=product_lr.pq - product_rl.pq,
        qp=product_lr.qp - product_rl.qp,
        qq=product_lr.qq - product_rl.qq,
    )


def _combine_field_blocks(
    reference_blocks: MatrixBlocks,
    hsx_blocks: MatrixBlocks,
    hsy_blocks: MatrixBlocks,
    hsz_blocks: MatrixBlocks,
    electric_field: np.ndarray,
    *,
    hzx_blocks: MatrixBlocks | None = None,
    hzy_blocks: MatrixBlocks | None = None,
    hzz_blocks: MatrixBlocks | None = None,
    magnetic_delta: np.ndarray | None = None,
) -> MatrixBlocks:
    magnetic = np.zeros(3, dtype=np.float64) if magnetic_delta is None else magnetic_delta
    zero = MatrixBlocks(
        pp=np.zeros_like(reference_blocks.pp),
        pq=np.zeros_like(reference_blocks.pq),
        qp=np.zeros_like(reference_blocks.qp),
        qq=np.zeros_like(reference_blocks.qq),
    )
    hzx = zero if hzx_blocks is None else hzx_blocks
    hzy = zero if hzy_blocks is None else hzy_blocks
    hzz = zero if hzz_blocks is None else hzz_blocks
    return MatrixBlocks(
        pp=(
            reference_blocks.pp
            + electric_field[0] * hsx_blocks.pp
            + electric_field[1] * hsy_blocks.pp
            + electric_field[2] * hsz_blocks.pp
            + magnetic[0] * hzx.pp
            + magnetic[1] * hzy.pp
            + magnetic[2] * hzz.pp
        ),
        pq=(
            reference_blocks.pq
            + electric_field[0] * hsx_blocks.pq
            + electric_field[1] * hsy_blocks.pq
            + electric_field[2] * hsz_blocks.pq
            + magnetic[0] * hzx.pq
            + magnetic[1] * hzy.pq
            + magnetic[2] * hzz.pq
        ),
        qp=(
            reference_blocks.qp
            + electric_field[0] * hsx_blocks.qp
            + electric_field[1] * hsy_blocks.qp
            + electric_field[2] * hsz_blocks.qp
            + magnetic[0] * hzx.qp
            + magnetic[1] * hzy.qp
            + magnetic[2] * hzz.qp
        ),
        qq=(
            reference_blocks.qq
            + electric_field[0] * hsx_blocks.qq
            + electric_field[1] * hsy_blocks.qq
            + electric_field[2] * hsz_blocks.qq
            + magnetic[0] * hzx.qq
            + magnetic[1] * hzy.qq
            + magnetic[2] * hzz.qq
        ),
    )


def _hamiltonian_superoperator(hamiltonian_total: np.ndarray) -> np.ndarray:
    n_states = hamiltonian_total.shape[0]
    identity = np.eye(n_states, dtype=np.complex128)
    return -1j * (
        np.kron(hamiltonian_total, identity)
        - np.kron(identity, hamiltonian_total.T)
    )


def _operator_from_transition(
    qn_full: Sequence[states.CoupledState],
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    transition_selector: couplings.TransitionSelector,
) -> tuple[np.ndarray, complex, int, int]:
    if transition_selector.ground_main is None or transition_selector.excited_main is None:
        raise ValueError("Transition selector must contain ground_main and excited_main states")

    ground_main_index = _dominant_state_index(transition_selector.ground_main, qn_full)
    excited_main_index = _dominant_state_index(transition_selector.excited_main, qn_full)
    qn_coupling = [
        (1.0 * state if isinstance(state, states.CoupledBasisState) else state)
        for state in qn_full
    ]
    ground_coupling = [
        (1.0 * state if isinstance(state, states.CoupledBasisState) else state)
        for state in ground_states
    ]
    excited_coupling = [
        (1.0 * state if isinstance(state, states.CoupledBasisState) else state)
        for state in excited_states
    ]
    ground_main_state = qn_coupling[ground_main_index]
    excited_main_state = qn_coupling[excited_main_index]
    main_coupling = hamiltonian.generate_ED_ME_mixed_state(
        excited_main_state,
        ground_main_state,
        pol_vec=transition_selector.polarizations[0],
        normalize_pol=True,
    )
    if main_coupling == 0:
        raise ValueError("Main optical matrix element is zero in the fixed basis")

    optical_matrix = couplings.generate_coupling_matrix(
        qn_coupling,
        ground_coupling,
        excited_coupling,
        pol_vec=np.asarray(transition_selector.polarizations[0], dtype=np.complex128),
        reduced=False,
        normalize_pol=True,
    )
    return (
        np.asarray(optical_matrix / main_coupling, dtype=np.complex128),
        complex(main_coupling),
        ground_main_index,
        excited_main_index,
    )


def _compact_transition_frequency(
    system: lindblad.utils_setup.OBESystem,
    *,
    transition: transitions.OpticalTransition,
    optical_polarization: couplings.Polarization,
) -> float:
    transition_selector = couplings.generate_transition_selectors(
        transitions=[transition],
        polarizations=[[optical_polarization]],
    )[0]
    qn_full = list(system.QN_original)
    ground_states = [
        state for state in qn_full if state.largest.electronic_state == states.ElectronicState.X
    ]
    excited_states = [
        state for state in qn_full if state.largest.electronic_state == states.ElectronicState.B
    ]
    _, _, ground_main_index, excited_main_index = _operator_from_transition(
        qn_full,
        ground_states,
        excited_states,
        transition_selector,
    )
    diag = np.real(np.diag(np.asarray(system.H_int, dtype=np.complex128)))
    return float(diag[excited_main_index] - diag[ground_main_index])


def _shift_bundle_to_common_frequency_frame(
    bundle: OperatorBundle,
    *,
    delta_omega: float,
    common_omega_reference: float,
) -> OperatorBundle:
    delta_omega = float(delta_omega)
    if abs(delta_omega) <= 1e-18:
        return replace(bundle, omega_reference=float(common_omega_reference))
    correction = float(delta_omega) * np.asarray(bundle.h_det, dtype=np.complex128)
    return replace(
        bundle,
        omega_reference=float(common_omega_reference),
        h_internal=_symmetrize(np.asarray(bundle.h_internal, dtype=np.complex128) + correction),
        h_full_internal=_symmetrize(np.asarray(bundle.h_full_internal, dtype=np.complex128) + correction),
        h_lab_internal=_symmetrize(np.asarray(bundle.h_lab_internal, dtype=np.complex128) + correction),
    )


def _build_generator_s(
    hamiltonian_full: np.ndarray,
    p_indices: np.ndarray,
    q_indices: np.ndarray,
    denominator_floor: float,
) -> tuple[np.ndarray, float]:
    n_total = hamiltonian_full.shape[0]
    generator = np.zeros((n_total, n_total), dtype=np.complex128)
    diag = np.real(np.diag(hamiltonian_full))
    max_ratio = 0.0
    for i in p_indices:
        for a in q_indices:
            coupling = hamiltonian_full[i, a]
            if coupling == 0:
                continue
            denom_raw = float(diag[i] - diag[a])
            denom = _safe_denominator(denom_raw, denominator_floor)
            ratio = abs(coupling) / abs(denom)
            max_ratio = max(max_ratio, float(ratio))
            generator[i, a] = coupling / denom
            generator[a, i] = -np.conj(generator[i, a])
    return generator, max_ratio


def _build_generator_s_blocks(
    h_pp: np.ndarray,
    v_pq: np.ndarray,
    h_qq: np.ndarray,
    denominator_floor: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    e_p = np.real(np.diag(h_pp))
    e_q = np.real(np.diag(h_qq))
    denominators = e_p[:, None] - e_q[None, :]
    safe = np.where(
        np.abs(denominators) >= denominator_floor,
        denominators,
        np.where(denominators == 0.0, denominator_floor, np.sign(denominators) * denominator_floor),
    ).astype(np.float64)
    x = solve_sylvester(h_pp, -h_qq, v_pq)
    s_pq = np.asarray(x, dtype=np.complex128)
    s_qp = -s_pq.conj().T
    max_ratio = 0.0 if v_pq.size == 0 else float(np.max(np.abs(v_pq) / np.abs(safe)))
    residual = h_pp @ s_pq - s_pq @ h_qq - v_pq
    sylvester_residual_norm = float(np.linalg.norm(residual))
    eig_pp = eigvals(h_pp)
    eig_qq = eigvals(h_qq)
    spectral_separation = float(
        np.min(np.abs(eig_pp[:, None] - eig_qq[None, :]))
    ) if eig_pp.size and eig_qq.size else float("inf")
    return s_pq, s_qp, max_ratio, sylvester_residual_norm, spectral_separation


def _second_order_hamiltonian_correction(
    hamiltonian_full: np.ndarray,
    p_indices: np.ndarray,
    q_indices: np.ndarray,
    denominator_floor: float,
) -> np.ndarray:
    diag = np.real(np.diag(hamiltonian_full))
    n_keep = p_indices.size
    correction = np.zeros((n_keep, n_keep), dtype=np.complex128)
    for i_local, i in enumerate(p_indices):
        for j_local, j in enumerate(p_indices):
            value = 0.0 + 0.0j
            for a in q_indices:
                v_ia = hamiltonian_full[i, a]
                v_aj = hamiltonian_full[a, j]
                if v_ia == 0 or v_aj == 0:
                    continue
                denom_i = _safe_denominator(float(diag[i] - diag[a]), denominator_floor)
                denom_j = _safe_denominator(float(diag[j] - diag[a]), denominator_floor)
                value += 0.5 * v_ia * v_aj * ((1.0 / denom_i) + (1.0 / denom_j))
            correction[i_local, j_local] = value
    return _symmetrize(correction)


def _second_order_hamiltonian_correction_blocks(
    s_pq: np.ndarray,
    s_qp: np.ndarray,
    v_pq: np.ndarray,
    v_qp: np.ndarray,
) -> np.ndarray:
    return _symmetrize(0.5 * (s_pq @ v_qp - v_pq @ s_qp))


def _dress_operator_first_order_full(operator: np.ndarray, generator: np.ndarray) -> np.ndarray:
    return np.asarray(operator + _commutator(operator, generator), dtype=np.complex128)


def _dress_operator_first_order_pp(
    operator_blocks: MatrixBlocks,
    s_pq: np.ndarray,
    s_qp: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        operator_blocks.pp + operator_blocks.pq @ s_qp - s_pq @ operator_blocks.qp,
        dtype=np.complex128,
    )


def _dress_operator_first_order_qp(
    operator_blocks: MatrixBlocks,
    s_qp: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        operator_blocks.qp + operator_blocks.qq @ s_qp - s_qp @ operator_blocks.pp,
        dtype=np.complex128,
    )


def _generator_blocks(
    s_pq: np.ndarray,
    s_qp: np.ndarray,
) -> MatrixBlocks:
    return MatrixBlocks(
        pp=np.zeros((s_pq.shape[0], s_pq.shape[0]), dtype=np.complex128),
        pq=np.asarray(s_pq, dtype=np.complex128),
        qp=np.asarray(s_qp, dtype=np.complex128),
        qq=np.zeros((s_qp.shape[0], s_qp.shape[0]), dtype=np.complex128),
    )


def _dress_operator_second_order_blocks(
    operator_blocks: MatrixBlocks,
    s_pq: np.ndarray,
    s_qp: np.ndarray,
) -> MatrixBlocks:
    s_blocks = _generator_blocks(s_pq, s_qp)
    comm_1 = _block_commutator(operator_blocks, s_blocks)
    comm_2 = _block_commutator(comm_1, s_blocks)
    return _block_add(_block_add(operator_blocks, comm_1), comm_2, scale_right=0.5)




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


@dataclass
class OperatorBundle:
    electric_field: np.ndarray
    magnetic_field: np.ndarray
    omega_reference: float
    h_internal: np.ndarray
    h_opt: np.ndarray
    h_det: np.ndarray
    c_array: np.ndarray
    excited_indices: np.ndarray
    loss_operator: np.ndarray
    h_full_internal: np.ndarray
    h_lab_internal: np.ndarray
    dissipator_superop: np.ndarray | None = None
    perturbative_ratio_max: float | None = None
    hermiticity_error: float | None = None
    sylvester_residual_norm: float | None = None
    spectral_separation_min: float | None = None
    generator: np.ndarray | None = None
    p_indices: np.ndarray | None = None
    q_indices: np.ndarray | None = None
    excited_to_ground_rates_hz: np.ndarray | None = None
    excited_to_sink_rates_hz: np.ndarray | None = None
    decay_kernel_ground: np.ndarray | None = None
    decay_kernels_sinks: tuple[np.ndarray, ...] | None = None
    jump_rate_operator_override: np.ndarray | None = None

    def total_hamiltonian(
        self,
        *,
        rabi_rate: float | complex = 0.0,
        detuning: float = 0.0,
    ) -> np.ndarray:
        return (
            np.asarray(self.h_internal, dtype=np.complex128)
            + 0.5 * complex(rabi_rate) * np.asarray(self.h_opt, dtype=np.complex128)
            + float(detuning) * np.asarray(self.h_det, dtype=np.complex128)
        )

    def dissipator(self, rho: np.ndarray) -> np.ndarray:
        if self.dissipator_superop is not None:
            return np.asarray(self.dissipator_superop @ rho.reshape(-1), dtype=np.complex128).reshape(
                rho.shape
            )
        return _lindblad_dissipator(self.c_array, rho, loss_operator=self.loss_operator)

    def dissipator_superoperator(self) -> np.ndarray:
        if self.dissipator_superop is not None:
            return np.asarray(self.dissipator_superop, dtype=np.complex128)
        return _dissipator_superoperator(self.c_array, loss_operator=self.loss_operator)

    def jump_rate_operator(self) -> np.ndarray:
        if self.jump_rate_operator_override is not None:
            return np.asarray(self.jump_rate_operator_override, dtype=np.complex128)
        return _jump_rate_operator(self.c_array, loss_operator=self.loss_operator)

    def liouvillian_superoperator(
        self,
        *,
        rabi_rate: float | complex = 0.0,
        detuning: float = 0.0,
    ) -> np.ndarray:
        h_total = self.total_hamiltonian(rabi_rate=rabi_rate, detuning=detuning)
        return _hamiltonian_superoperator(h_total) + self.dissipator_superoperator()


@dataclass
class PreparedEffectiveHamiltonianModel:
    transition: transitions.OpticalTransition
    optical_polarization: couplings.Polarization
    transition_selector: couplings.TransitionSelector
    space_generation_electric_field: np.ndarray
    space_generation_magnetic_field: np.ndarray
    reference_electric_field: np.ndarray
    reference_magnetic_field: np.ndarray
    qn_full: list[states.CoupledState]
    x_states: list[states.CoupledState]
    b_states: list[states.CoupledState]
    h_reference_internal: np.ndarray
    h_sx: np.ndarray
    h_sy: np.ndarray
    h_sz: np.ndarray
    h_zx: np.ndarray
    h_zy: np.ndarray
    h_zz: np.ndarray
    h0_blocks: MatrixBlocks
    hsx_blocks: MatrixBlocks
    hsy_blocks: MatrixBlocks
    hsz_blocks: MatrixBlocks
    hzx_blocks: MatrixBlocks
    hzy_blocks: MatrixBlocks
    hzz_blocks: MatrixBlocks
    h_opt_full: np.ndarray
    h_det_full: np.ndarray
    c_array_full: np.ndarray
    h_opt_blocks: MatrixBlocks
    h_det_blocks: MatrixBlocks
    jump_blocks: tuple[MatrixBlocks, ...]
    p_indices: np.ndarray
    q_indices: np.ndarray
    p_ground_indices: np.ndarray
    p_excited_indices: np.ndarray
    full_excited_indices: np.ndarray
    p_excited_indices_local: np.ndarray
    sink_indices_full: tuple[np.ndarray, ...]
    sink_indices_q: tuple[np.ndarray, ...]
    sink_labels: tuple[str, ...]
    ground_main_index_full: int
    excited_main_index_full: int
    ground_main_index_p: int
    excited_main_index_p: int
    main_coupling: complex
    gamma: float
    denominator_floor: float
    kept_state_dressing_order: int
    excited_state_dressing_order: int
    compact_reference_bundle: OperatorBundle | None = None
    perturbative_reference_bundle: OperatorBundle | None = None

    @property
    def p_states(self) -> list[states.CoupledState | states.CoupledBasisState]:
        return [self.qn_full[idx] for idx in self.p_indices]

    @property
    def n_active_states(self) -> int:
        return int(self.p_indices.size)

    @property
    def n_sink_states(self) -> int:
        return len(self.sink_indices_full)

    @property
    def n_effective_states(self) -> int:
        return self.n_active_states + self.n_sink_states

    @property
    def sink_positions(self) -> np.ndarray:
        n_ground = int(self.p_ground_indices.size)
        return np.arange(n_ground, n_ground + self.n_sink_states, dtype=np.int64)

    @property
    def active_positions(self) -> np.ndarray:
        n_ground = int(self.p_ground_indices.size)
        return np.concatenate(
            [
                np.arange(n_ground, dtype=np.int64),
                np.arange(n_ground + self.n_sink_states, self.n_effective_states, dtype=np.int64),
            ]
        )

    @property
    def excited_positions(self) -> np.ndarray:
        n_ground = int(self.p_ground_indices.size)
        return np.arange(
            n_ground + self.n_sink_states,
            self.n_effective_states,
            dtype=np.int64,
        )

    def assemble_internal_blocks(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> MatrixBlocks:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        delta_b = magnetic - self.reference_magnetic_field
        return _combine_field_blocks(
            self.h0_blocks,
            self.hsx_blocks,
            self.hsy_blocks,
            self.hsz_blocks,
            electric,
            hzx_blocks=self.hzx_blocks,
            hzy_blocks=self.hzy_blocks,
            hzz_blocks=self.hzz_blocks,
            magnetic_delta=delta_b,
        )

    def assemble_lab_internal_hamiltonian(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        return _symmetrize(
            _assemble_full_from_blocks(
                self.assemble_internal_blocks(electric_field, magnetic_field)
            )
        )

    def reference_transition_frequency(self, h_lab_internal: np.ndarray) -> float:
        diag = np.real(np.diag(h_lab_internal))
        return float(diag[self.excited_main_index_full] - diag[self.ground_main_index_full])

    def exact_bundle(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> OperatorBundle:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        h_lab = self.assemble_lab_internal_hamiltonian(electric, magnetic)
        omega_reference = self.reference_transition_frequency(h_lab)
        h_internal = _symmetrize(h_lab - omega_reference * self.h_det_full)
        return OperatorBundle(
            electric_field=electric,
            magnetic_field=magnetic,
            omega_reference=omega_reference,
            h_internal=h_internal,
            h_opt=np.asarray(self.h_opt_full, dtype=np.complex128),
            h_det=np.asarray(self.h_det_full, dtype=np.complex128),
            c_array=np.asarray(self.c_array_full, dtype=np.complex128),
            excited_indices=np.asarray(self.full_excited_indices, dtype=np.int64),
            loss_operator=np.zeros_like(self.h_det_full, dtype=np.complex128),
            h_full_internal=h_internal,
            h_lab_internal=h_lab,
            hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
        )

    def _effective_bundle_perturbative(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> OperatorBundle:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        h_blocks_lab = self.assemble_internal_blocks(electric, magnetic)
        h_lab_full = _symmetrize(_assemble_full_from_blocks(h_blocks_lab))

        omega_reference = float(
            np.real(np.diag(h_blocks_lab.pp))[self.excited_main_index_p]
            - np.real(np.diag(h_blocks_lab.pp))[self.ground_main_index_p]
        )

        s_pq, s_qp, max_ratio, sylvester_residual_norm, spectral_separation = _build_generator_s_blocks(
            h_blocks_lab.pp,
            h_blocks_lab.pq,
            h_blocks_lab.qq,
            self.denominator_floor,
        )
        generator = np.block(
            [
                [np.zeros_like(h_blocks_lab.pp), s_pq],
                [s_qp, np.zeros_like(h_blocks_lab.qq)],
            ]
        ).astype(np.complex128)
        n_active = h_blocks_lab.pp.shape[0]
        delta_h = _second_order_hamiltonian_correction_blocks(
            s_pq,
            s_qp,
            h_blocks_lab.pq,
            h_blocks_lab.qp,
        )
        h_eff_lab_active = _symmetrize(h_blocks_lab.pp + delta_h)

        p_vectors_base, q_vectors, p_vectors_mixed = _mixed_order_kept_vectors(
            generator,
            n_active,
            np.asarray(self.p_excited_indices_local, dtype=np.int64),
            base_order=self.kept_state_dressing_order,
            excited_order=self.excited_state_dressing_order,
        )
        h_opt_partitioned = _assemble_full_from_blocks(self.h_opt_blocks)
        h_det_partitioned = _assemble_full_from_blocks(self.h_det_blocks)
        h_opt_active = _symmetrize(
            _matrix_elements_dressed_states(h_opt_partitioned, p_vectors_mixed)
        )
        h_det_active = _symmetrize(
            _matrix_elements_dressed_states(h_det_partitioned, p_vectors_base)
        )
        h_internal_active = _symmetrize(h_eff_lab_active - omega_reference * h_det_active)

        n_total = self.n_effective_states
        h_internal_eff = np.zeros((n_total, n_total), dtype=np.complex128)
        h_internal_eff[np.ix_(self.active_positions, self.active_positions)] = h_internal_active

        full_diag_lab = np.real(np.diag(h_blocks_lab.qq))
        for sink_offset, sink_indices_q in enumerate(self.sink_indices_q):
            sink_index = int(self.sink_positions[sink_offset])
            h_internal_eff[sink_index, sink_index] = float(np.mean(full_diag_lab[sink_indices_q]))

        n_ground = int(self.p_ground_indices.size)
        n_excited = int(self.p_excited_indices_local.size)
        ground_rows = np.arange(n_ground, dtype=np.int64)
        excited_rows = np.asarray(self.p_excited_indices_local, dtype=np.int64)
        source_columns = np.asarray(self.excited_positions, dtype=np.int64)
        dissipator_eff = np.zeros((n_total * n_total, n_total * n_total), dtype=np.complex128)
        loss_operator = np.zeros((n_total, n_total), dtype=np.complex128)
        decay_kernel_ground = np.zeros((n_excited, n_excited), dtype=np.complex128)
        decay_kernels_sinks = [
            np.zeros((n_excited, n_excited), dtype=np.complex128)
            for _ in self.sink_indices_q
        ]
        ground_vectors = p_vectors_base[:, ground_rows]
        excited_vectors = p_vectors_mixed[:, excited_rows]
        sink_vectors = tuple(q_vectors[:, sink_indices_q] for sink_indices_q in self.sink_indices_q)

        def vector_index(row: int, col: int) -> int:
            return row * n_total + col

        def source_basis(a: int, b: int) -> np.ndarray:
            rho_e = np.zeros((n_excited, n_excited), dtype=np.complex128)
            rho_e[a, b] = 1.0
            return rho_e

        for jump_blocks in self.jump_blocks:
            jump_partitioned = _assemble_full_from_blocks(jump_blocks)
            c_ge = _matrix_elements_dressed_states(
                jump_partitioned,
                ground_vectors,
                excited_vectors,
            )
            if np.any(np.abs(c_ge) > 0):
                decay_kernel_ground += c_ge.conj().T @ c_ge
            c_ee = _matrix_elements_dressed_states(
                jump_partitioned,
                p_vectors_mixed,
                excited_vectors,
            )
            if np.any(np.abs(c_ee) > 0):
                loss_operator[np.ix_(source_columns, source_columns)] += c_ee.conj().T @ c_ee

            sink_channels: list[np.ndarray] = []
            for sink_offset, sink_vectors_group in enumerate(sink_vectors):
                c_sink = _matrix_elements_dressed_states(
                    jump_partitioned,
                    sink_vectors_group,
                    excited_vectors,
                )
                sink_channels.append(c_sink)
                if np.any(np.abs(c_sink) > 0):
                    decay_kernels_sinks[sink_offset] += c_sink.conj().T @ c_sink
                    loss_operator[np.ix_(source_columns, source_columns)] += c_sink.conj().T @ c_sink

            for a in range(n_excited):
                for b in range(n_excited):
                    rho_e = source_basis(a, b)
                    source_idx = vector_index(int(source_columns[a]), int(source_columns[b]))

                    if np.any(np.abs(c_ge) > 0):
                        rho_g = c_ge @ rho_e @ c_ge.conj().T
                        for i in range(n_ground):
                            for j in range(n_ground):
                                dissipator_eff[
                                    vector_index(int(ground_rows[i]), int(ground_rows[j])),
                                    source_idx,
                                ] += rho_g[i, j]

                    for sink_offset, c_sink in enumerate(sink_channels):
                        if np.any(np.abs(c_sink) > 0):
                            sink_pop = np.trace(c_sink @ rho_e @ c_sink.conj().T)
                            sink_pos = int(self.sink_positions[sink_offset])
                            dissipator_eff[vector_index(sink_pos, sink_pos), source_idx] += sink_pop

        dissipator_eff += _dissipator_superoperator(
            np.zeros((0, n_total, n_total), dtype=np.complex128),
            loss_operator=loss_operator,
        )
        decay_kernel_ground = _symmetrize(decay_kernel_ground)
        decay_kernels_sinks = tuple(_symmetrize(kernel) for kernel in decay_kernels_sinks)
        excited_to_ground_rates = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)
        excited_to_sink_rates = (
            np.sum(
                np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
                axis=0,
            )
            if decay_kernels_sinks
            else np.zeros(n_excited, dtype=np.float64)
        )
        c_array_eff = np.zeros((0, n_total, n_total), dtype=np.complex128)
        h_opt_eff = _embed_subspace_matrix(h_opt_active, self.active_positions, n_total)
        h_det_eff = _embed_subspace_matrix(h_det_active, self.active_positions, n_total)

        return OperatorBundle(
            electric_field=electric,
            magnetic_field=magnetic,
            omega_reference=omega_reference,
            h_internal=_symmetrize(h_internal_eff),
            h_opt=h_opt_eff,
            h_det=h_det_eff,
            c_array=c_array_eff,
            excited_indices=np.asarray(self.excited_positions, dtype=np.int64),
            loss_operator=loss_operator,
            h_full_internal=h_lab_full - omega_reference * self.h_det_full,
            h_lab_internal=h_lab_full,
            dissipator_superop=dissipator_eff,
            perturbative_ratio_max=max_ratio,
            hermiticity_error=float(np.max(np.abs(h_internal_eff - h_internal_eff.conj().T))),
            sylvester_residual_norm=sylvester_residual_norm,
            spectral_separation_min=spectral_separation,
            generator=generator,
            p_indices=np.asarray(self.p_indices, dtype=np.int64),
            q_indices=np.asarray(self.q_indices, dtype=np.int64),
            excited_to_ground_rates_hz=excited_to_ground_rates,
            excited_to_sink_rates_hz=excited_to_sink_rates,
            decay_kernel_ground=decay_kernel_ground,
            decay_kernels_sinks=decay_kernels_sinks,
        )

    def effective_bundle(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> OperatorBundle:
        perturbative = self._effective_bundle_perturbative(electric_field, magnetic_field)
        compact_reference = self.compact_reference_bundle
        perturbative_reference = self.perturbative_reference_bundle
        if (
            compact_reference is None
            or perturbative_reference is None
            or compact_reference.h_internal.shape != perturbative.h_internal.shape
        ):
            return perturbative

        compact_dissipator = compact_reference.dissipator_superoperator()
        perturbative_dissipator = perturbative.dissipator_superoperator()
        perturbative_reference_dissipator = perturbative_reference.dissipator_superoperator()
        compact_jump_rate = compact_reference.jump_rate_operator()
        perturbative_jump_rate = perturbative.jump_rate_operator()
        perturbative_reference_jump_rate = perturbative_reference.jump_rate_operator()

        h_internal = _symmetrize(
            compact_reference.h_internal
            + (perturbative.h_internal - perturbative_reference.h_internal)
        )
        h_opt = _symmetrize(
            compact_reference.h_opt
            + (perturbative.h_opt - perturbative_reference.h_opt)
        )
        h_det = _symmetrize(
            compact_reference.h_det
            + (perturbative.h_det - perturbative_reference.h_det)
        )
        dissipator_superop = np.asarray(
            compact_dissipator + (perturbative_dissipator - perturbative_reference_dissipator),
            dtype=np.complex128,
        )
        jump_rate_operator = _symmetrize(
            compact_jump_rate + (perturbative_jump_rate - perturbative_reference_jump_rate)
        )

        decay_kernel_ground = None
        if (
            compact_reference.decay_kernel_ground is not None
            and perturbative.decay_kernel_ground is not None
            and perturbative_reference.decay_kernel_ground is not None
        ):
            decay_kernel_ground = _symmetrize(
                compact_reference.decay_kernel_ground
                + (
                    perturbative.decay_kernel_ground
                    - perturbative_reference.decay_kernel_ground
                )
            )

        decay_kernels_sinks = None
        if (
            compact_reference.decay_kernels_sinks is not None
            and perturbative.decay_kernels_sinks is not None
            and perturbative_reference.decay_kernels_sinks is not None
        ):
            decay_kernels_sinks = tuple(
                _symmetrize(
                    compact_kernel + (kernel - kernel_ref)
                )
                for compact_kernel, kernel, kernel_ref in zip(
                    compact_reference.decay_kernels_sinks,
                    perturbative.decay_kernels_sinks,
                    perturbative_reference.decay_kernels_sinks,
                )
            )

        excited_to_ground_rates = None
        if decay_kernel_ground is not None:
            excited_to_ground_rates = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)

        excited_to_sink_rates = None
        if decay_kernels_sinks is not None:
            excited_to_sink_rates = np.sum(
                np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
                axis=0,
            )
            excited_to_sink_rates = excited_to_sink_rates / (2.0 * np.pi)

        return OperatorBundle(
            electric_field=np.asarray(perturbative.electric_field, dtype=np.float64),
            magnetic_field=np.asarray(perturbative.magnetic_field, dtype=np.float64),
            omega_reference=float(compact_reference.omega_reference)
            + float(perturbative.omega_reference - perturbative_reference.omega_reference),
            h_internal=h_internal,
            h_opt=h_opt,
            h_det=h_det,
            c_array=np.zeros_like(compact_reference.c_array),
            excited_indices=np.asarray(compact_reference.excited_indices, dtype=np.int64),
            loss_operator=np.zeros_like(h_internal, dtype=np.complex128),
            h_full_internal=np.asarray(perturbative.h_full_internal, dtype=np.complex128),
            h_lab_internal=np.asarray(perturbative.h_lab_internal, dtype=np.complex128),
            dissipator_superop=dissipator_superop,
            perturbative_ratio_max=perturbative.perturbative_ratio_max,
            hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
            sylvester_residual_norm=perturbative.sylvester_residual_norm,
            spectral_separation_min=perturbative.spectral_separation_min,
            generator=perturbative.generator,
            p_indices=np.asarray(perturbative.p_indices, dtype=np.int64)
            if perturbative.p_indices is not None
            else None,
            q_indices=np.asarray(perturbative.q_indices, dtype=np.int64)
            if perturbative.q_indices is not None
            else None,
            excited_to_ground_rates_hz=excited_to_ground_rates,
            excited_to_sink_rates_hz=excited_to_sink_rates,
            decay_kernel_ground=decay_kernel_ground,
            decay_kernels_sinks=decay_kernels_sinks,
            jump_rate_operator_override=jump_rate_operator,
        )


@dataclass
class PreparedLocalOperatorExpansionModel:
    base_model: PreparedEffectiveHamiltonianModel
    expansion_axis: int
    expansion_step: float
    expansion_order: int
    reference_bundle: OperatorBundle
    h_internal_coefficients: tuple[np.ndarray, ...]
    h_opt_coefficients: tuple[np.ndarray, ...]
    h_det_coefficients: tuple[np.ndarray, ...]
    dissipator_coefficients: tuple[np.ndarray, ...]
    jump_rate_coefficients: tuple[np.ndarray, ...]
    h_full_internal_coefficients: tuple[np.ndarray, ...]
    h_lab_internal_coefficients: tuple[np.ndarray, ...]
    omega_reference_coefficients: tuple[np.ndarray, ...]
    decay_kernel_ground_coefficients: tuple[np.ndarray, ...] | None = None
    decay_kernels_sinks_coefficients: tuple[tuple[np.ndarray, ...], ...] | None = None

    def __getattr__(self, name: str):
        return getattr(self.base_model, name)

    def effective_bundle(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> OperatorBundle:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.base_model.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        for axis in range(3):
            if axis == self.expansion_axis:
                continue
            if abs(float(electric[axis] - self.base_model.reference_electric_field[axis])) > 1e-12:
                raise ValueError(
                    "PreparedLocalOperatorExpansionModel currently supports variation only "
                    f"along electric-field axis {self.expansion_axis}."
                )
        if np.max(np.abs(magnetic - self.base_model.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedLocalOperatorExpansionModel currently supports only the reference "
                "magnetic field."
            )

        delta = float(electric[self.expansion_axis] - self.base_model.reference_electric_field[self.expansion_axis])
        h_internal = _symmetrize(_evaluate_polynomial_coefficients(self.h_internal_coefficients, delta))
        h_opt = _symmetrize(_evaluate_polynomial_coefficients(self.h_opt_coefficients, delta))
        h_det = _symmetrize(_evaluate_polynomial_coefficients(self.h_det_coefficients, delta))
        dissipator_superop = np.asarray(
            _evaluate_polynomial_coefficients(self.dissipator_coefficients, delta),
            dtype=np.complex128,
        )
        jump_rate_operator = _symmetrize(
            _evaluate_polynomial_coefficients(self.jump_rate_coefficients, delta)
        )
        h_full_internal = np.asarray(
            _evaluate_polynomial_coefficients(self.h_full_internal_coefficients, delta),
            dtype=np.complex128,
        )
        h_lab_internal = np.asarray(
            _evaluate_polynomial_coefficients(self.h_lab_internal_coefficients, delta),
            dtype=np.complex128,
        )
        omega_reference = float(
            np.real(_evaluate_polynomial_coefficients(self.omega_reference_coefficients, delta))
        )

        decay_kernel_ground = None
        excited_to_ground_rates_hz = None
        if self.decay_kernel_ground_coefficients is not None:
            decay_kernel_ground = _symmetrize(
                _evaluate_polynomial_coefficients(self.decay_kernel_ground_coefficients, delta)
            )
            excited_to_ground_rates_hz = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)

        decay_kernels_sinks = None
        excited_to_sink_rates_hz = None
        if self.decay_kernels_sinks_coefficients is not None:
            decay_kernels_sinks = tuple(
                _symmetrize(_evaluate_polynomial_coefficients(coefficients, delta))
                for coefficients in self.decay_kernels_sinks_coefficients
            )
            excited_to_sink_rates_hz = np.sum(
                np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
                axis=0,
            ) / (2.0 * np.pi)

        return OperatorBundle(
            electric_field=electric,
            magnetic_field=magnetic,
            omega_reference=omega_reference,
            h_internal=h_internal,
            h_opt=h_opt,
            h_det=h_det,
            c_array=np.zeros_like(self.reference_bundle.c_array),
            excited_indices=np.asarray(self.reference_bundle.excited_indices, dtype=np.int64),
            loss_operator=np.zeros_like(h_internal, dtype=np.complex128),
            h_full_internal=h_full_internal,
            h_lab_internal=h_lab_internal,
            dissipator_superop=dissipator_superop,
            hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
            excited_to_ground_rates_hz=excited_to_ground_rates_hz,
            excited_to_sink_rates_hz=excited_to_sink_rates_hz,
            decay_kernel_ground=decay_kernel_ground,
            decay_kernels_sinks=decay_kernels_sinks,
            jump_rate_operator_override=jump_rate_operator,
        )


@dataclass
class InterpolatedEffectivePatch:
    electric_field: np.ndarray
    aligned_basis_vectors: np.ndarray
    bundle: OperatorBundle


@dataclass
class LindbladSafeCompactInterpolatedPatch:
    electric_field: np.ndarray
    aligned_basis_vectors: np.ndarray
    bundle: OperatorBundle
    target_decay_kernels: tuple[np.ndarray, ...]
    full_recycling_decay_kernel: np.ndarray


@dataclass
class PopulationSinkEffectivePatch:
    electric_field: np.ndarray
    coherent_bundle: OperatorBundle
    sink_feed_matrix: np.ndarray


@dataclass
class InstantaneousInterpolatedEffectivePatch:
    electric_field: np.ndarray
    full_basis_vectors: np.ndarray
    coherent_basis_vectors: np.ndarray
    gauge_connection: np.ndarray
    bundle: OperatorBundle


@dataclass
class PreparedInterpolatedEffectiveHamiltonianModel:
    transition: transitions.OpticalTransition
    optical_polarization: couplings.Polarization
    reference_magnetic_field: np.ndarray
    parent_basis_qn: list[states.CoupledState]
    union_state_keys: tuple[str, ...]
    field_points: np.ndarray
    master_field: float
    ground_indices: np.ndarray
    sink_indices: np.ndarray
    excited_indices: np.ndarray
    ground_main_index: int
    patches: tuple[InterpolatedEffectivePatch, ...]
    keep_diagnostics: bool = True

    @property
    def n_effective_states(self) -> int:
        return int(self.patches[0].bundle.h_internal.shape[0])

    @property
    def ground_main_index_p(self) -> int:
        return int(self.ground_main_index)

    def _interpolation_indices(self, field_z: float) -> tuple[int, int, float]:
        fields = np.asarray(self.field_points, dtype=np.float64)
        if field_z <= fields[0]:
            return 0, 0, 0.0
        if field_z >= fields[-1]:
            last = int(fields.size - 1)
            return last, last, 0.0
        upper = int(np.searchsorted(fields, field_z, side="right"))
        lower = upper - 1
        x0 = float(fields[lower])
        x1 = float(fields[upper])
        weight = (float(field_z) - x0) / (x1 - x0)
        return lower, upper, float(weight)

    def _interpolate_matrix(self, values: Sequence[np.ndarray], field_z: float) -> np.ndarray:
        lower, upper, weight = self._interpolation_indices(field_z)
        if lower == upper:
            return np.asarray(values[lower], dtype=np.complex128)
        return np.asarray(
            (1.0 - weight) * np.asarray(values[lower], dtype=np.complex128)
            + weight * np.asarray(values[upper], dtype=np.complex128),
            dtype=np.complex128,
        )

    def effective_bundle(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> OperatorBundle:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        if np.max(np.abs(magnetic - self.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedInterpolatedEffectiveHamiltonianModel currently supports only the "
                "reference magnetic field."
            )
        if np.max(np.abs(electric[:2])) > 1e-12:
            raise ValueError(
                "PreparedInterpolatedEffectiveHamiltonianModel currently supports interpolation "
                "only along Ez."
            )

        field_z = float(electric[2])
        h_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_internal for patch in self.patches], field_z)
        )
        h_opt = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_opt for patch in self.patches], field_z)
        )
        h_det = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_det for patch in self.patches], field_z)
        )
        dissipator_superop = self._interpolate_matrix(
            [patch.bundle.dissipator_superoperator() for patch in self.patches],
            field_z,
        )
        jump_rate_operator = _symmetrize(
            self._interpolate_matrix([patch.bundle.jump_rate_operator() for patch in self.patches], field_z)
        )
        h_full_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_full_internal for patch in self.patches], field_z)
        )
        h_lab_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_lab_internal for patch in self.patches], field_z)
        )
        omega_reference = float(
            np.real(
                self._interpolate_matrix(
                    [np.array(patch.bundle.omega_reference, dtype=np.complex128) for patch in self.patches],
                    field_z,
                )
            )
        )

        decay_kernel_ground = None
        excited_to_ground_rates_hz = None
        if self.keep_diagnostics and all(patch.bundle.decay_kernel_ground is not None for patch in self.patches):
            decay_kernel_ground = _symmetrize(
                self._interpolate_matrix(
                    [np.asarray(patch.bundle.decay_kernel_ground, dtype=np.complex128) for patch in self.patches],
                    field_z,
                )
            )
            excited_to_ground_rates_hz = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)

        decay_kernels_sinks = None
        excited_to_sink_rates_hz = None
        if self.keep_diagnostics and all(patch.bundle.decay_kernels_sinks is not None for patch in self.patches):
            n_sinks = len(self.patches[0].bundle.decay_kernels_sinks or ())
            decay_kernels_sinks = tuple(
                _symmetrize(
                    self._interpolate_matrix(
                        [
                            np.asarray(patch.bundle.decay_kernels_sinks[sink_index], dtype=np.complex128)  # type: ignore[index]
                            for patch in self.patches
                        ],
                        field_z,
                    )
                )
                for sink_index in range(n_sinks)
            )
            excited_to_sink_rates_hz = np.sum(
                np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
                axis=0,
            ) / (2.0 * np.pi)

        return OperatorBundle(
            electric_field=electric,
            magnetic_field=magnetic,
            omega_reference=omega_reference,
            h_internal=h_internal,
            h_opt=h_opt,
            h_det=h_det,
            c_array=np.zeros_like(self.patches[0].bundle.c_array),
            excited_indices=np.asarray(self.excited_indices, dtype=np.int64),
            loss_operator=np.zeros_like(h_internal, dtype=np.complex128),
            h_full_internal=h_full_internal,
            h_lab_internal=h_lab_internal,
            dissipator_superop=dissipator_superop,
            hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
            excited_to_ground_rates_hz=excited_to_ground_rates_hz,
            excited_to_sink_rates_hz=excited_to_sink_rates_hz,
            decay_kernel_ground=decay_kernel_ground,
            decay_kernels_sinks=decay_kernels_sinks,
            jump_rate_operator_override=jump_rate_operator,
        )


@dataclass
class PreparedLindbladSafeCompactInterpolatedHamiltonianModel:
    transition: transitions.OpticalTransition
    optical_polarization: couplings.Polarization
    reference_magnetic_field: np.ndarray
    parent_basis_qn: list[states.CoupledState]
    union_state_keys: tuple[str, ...]
    field_points: np.ndarray
    master_field: float
    ground_indices: np.ndarray
    sink_indices: np.ndarray
    excited_indices: np.ndarray
    target_indices: np.ndarray
    ground_main_index: int
    common_omega_reference: float
    patch_transition_frequencies: np.ndarray
    patches: tuple[LindbladSafeCompactInterpolatedPatch, ...]
    keep_diagnostics: bool = True

    @property
    def n_effective_states(self) -> int:
        return int(self.patches[0].bundle.h_internal.shape[0])

    @property
    def ground_main_index_p(self) -> int:
        return int(self.ground_main_index)

    def _interpolation_indices(self, field_z: float) -> tuple[int, int, float]:
        fields = np.asarray(self.field_points, dtype=np.float64)
        if field_z <= fields[0]:
            return 0, 0, 0.0
        if field_z >= fields[-1]:
            last = int(fields.size - 1)
            return last, last, 0.0
        upper = int(np.searchsorted(fields, field_z, side="right"))
        lower = upper - 1
        x0 = float(fields[lower])
        x1 = float(fields[upper])
        weight = (float(field_z) - x0) / (x1 - x0)
        return lower, upper, float(weight)

    def _interpolate_matrix(self, values: Sequence[np.ndarray], field_z: float) -> np.ndarray:
        lower, upper, weight = self._interpolation_indices(field_z)
        if lower == upper:
            return np.asarray(values[lower], dtype=np.complex128)
        return np.asarray(
            (1.0 - weight) * np.asarray(values[lower], dtype=np.complex128)
            + weight * np.asarray(values[upper], dtype=np.complex128),
            dtype=np.complex128,
        )

    def _interpolate_decay_kernels(self, field_z: float) -> tuple[np.ndarray, ...]:
        kernels: list[np.ndarray] = []
        for kernel_index in range(int(self.target_indices.size)):
            interpolated = self._interpolate_matrix(
                [np.asarray(patch.target_decay_kernels[kernel_index], dtype=np.complex128) for patch in self.patches],
                field_z,
            )
            kernels.append(_psd_project(interpolated))
        return tuple(kernels)

    def _interpolate_full_recycling_decay_kernel(self, field_z: float) -> np.ndarray:
        return _psd_project(
            self._interpolate_matrix(
                [
                    np.asarray(patch.full_recycling_decay_kernel, dtype=np.complex128)
                    for patch in self.patches
                ],
                field_z,
            )
        )

    def effective_bundle(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> OperatorBundle:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        if np.max(np.abs(magnetic - self.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedLindbladSafeCompactInterpolatedHamiltonianModel currently supports only the "
                "reference magnetic field."
            )
        if np.max(np.abs(electric[:2])) > 1e-12:
            raise ValueError(
                "PreparedLindbladSafeCompactInterpolatedHamiltonianModel currently supports interpolation "
                "only along Ez."
            )

        field_z = float(electric[2])
        n_total = int(self.n_effective_states)
        lower_indices = np.concatenate([self.ground_indices, self.sink_indices]).astype(np.int64)
        n_lower = int(lower_indices.size)

        lower_block = _symmetrize(
            self._interpolate_matrix(
                [
                    np.asarray(patch.bundle.h_internal[np.ix_(lower_indices, lower_indices)], dtype=np.complex128)
                    for patch in self.patches
                ],
                field_z,
            )
        )
        excited_block = _symmetrize(
            self._interpolate_matrix(
                [
                    np.asarray(
                        patch.bundle.h_internal[np.ix_(self.excited_indices, self.excited_indices)],
                        dtype=np.complex128,
                    )
                    for patch in self.patches
                ],
                field_z,
            )
        )
        h_internal = np.zeros((n_total, n_total), dtype=np.complex128)
        h_internal[np.ix_(lower_indices, lower_indices)] = lower_block
        h_internal[np.ix_(self.excited_indices, self.excited_indices)] = excited_block
        h_internal = _symmetrize(h_internal)

        ge_block = self._interpolate_matrix(
            [
                np.asarray(patch.bundle.h_opt[np.ix_(self.ground_indices, self.excited_indices)], dtype=np.complex128)
                for patch in self.patches
            ],
            field_z,
        )
        h_opt = np.zeros((n_total, n_total), dtype=np.complex128)
        h_opt[np.ix_(self.ground_indices, self.excited_indices)] = ge_block
        h_opt[np.ix_(self.excited_indices, self.ground_indices)] = ge_block.conj().T
        h_opt = _symmetrize(h_opt)

        h_det = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_det for patch in self.patches], field_z)
        )

        h_full_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_full_internal for patch in self.patches], field_z)
        )
        h_lab_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_lab_internal for patch in self.patches], field_z)
        )
        omega_reference = float(
            np.real(
                self._interpolate_matrix(
                    [np.array(patch.bundle.omega_reference, dtype=np.complex128) for patch in self.patches],
                    field_z,
                )
            )
        )

        target_decay_kernels = self._interpolate_decay_kernels(field_z)
        # Default Lindblad-safe dissipator path: interpolate the PSD kernel of the
        # full collapse-operator matrix entries, then factor it back into collapse
        # operators. The older per-target excited kernels are retained only for
        # rate diagnostics because they do not encode all lower-manifold coherences.
        full_recycling_decay_kernel = self._interpolate_full_recycling_decay_kernel(field_z)
        full_indices = np.arange(n_total, dtype=np.int64)
        c_array = _c_array_from_full_recycling_decay_kernel(
            target_indices=full_indices,
            source_indices=full_indices,
            kernel=full_recycling_decay_kernel,
            total_dimension=n_total,
        )

        decay_kernel_ground = None
        excited_to_ground_rates_hz = None
        if self.ground_indices.size:
            ground_kernels = target_decay_kernels[: int(self.ground_indices.size)]
            decay_kernel_ground = _symmetrize(
                np.sum(np.array(ground_kernels, dtype=np.complex128), axis=0)
            )
            excited_to_ground_rates_hz = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)

        decay_kernels_sinks = None
        excited_to_sink_rates_hz = None
        if self.sink_indices.size:
            sink_kernels = target_decay_kernels[int(self.ground_indices.size) :]
            decay_kernels_sinks = tuple(_symmetrize(kernel) for kernel in sink_kernels)
            excited_to_sink_rates_hz = np.sum(
                np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
                axis=0,
            ) / (2.0 * np.pi)

        return OperatorBundle(
            electric_field=electric,
            magnetic_field=magnetic,
            omega_reference=omega_reference,
            h_internal=h_internal,
            h_opt=h_opt,
            h_det=h_det,
            c_array=c_array,
            excited_indices=np.asarray(self.excited_indices, dtype=np.int64),
            loss_operator=np.zeros((n_total, n_total), dtype=np.complex128),
            h_full_internal=h_full_internal,
            h_lab_internal=h_lab_internal,
            hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
            excited_to_ground_rates_hz=excited_to_ground_rates_hz,
            excited_to_sink_rates_hz=excited_to_sink_rates_hz,
            decay_kernel_ground=decay_kernel_ground,
            decay_kernels_sinks=decay_kernels_sinks,
        )


@dataclass
class PreparedPopulationSinkEffectiveHamiltonianModel:
    transition: transitions.OpticalTransition
    optical_polarization: couplings.Polarization
    reference_magnetic_field: np.ndarray
    field_points: np.ndarray
    master_field: float
    ground_indices: np.ndarray
    excited_indices: np.ndarray
    ground_main_index: int
    sink_labels: tuple[str, ...]
    patches: tuple[PopulationSinkEffectivePatch, ...]

    @property
    def n_coherent_states(self) -> int:
        return int(self.patches[0].coherent_bundle.h_internal.shape[0])

    @property
    def n_sink_populations(self) -> int:
        return len(self.sink_labels)

    @property
    def n_effective_states(self) -> int:
        return self.n_coherent_states

    @property
    def coherent_ground_indices(self) -> np.ndarray:
        return np.arange(self.ground_indices.size, dtype=np.int64)

    @property
    def coherent_excited_indices(self) -> np.ndarray:
        start = int(self.ground_indices.size)
        stop = start + int(self.excited_indices.size)
        return np.arange(start, stop, dtype=np.int64)

    @property
    def ground_main_index_p(self) -> int:
        return int(self.ground_main_index)

    def _interpolation_indices(self, field_z: float) -> tuple[int, int, float]:
        fields = np.asarray(self.field_points, dtype=np.float64)
        if field_z <= fields[0]:
            return 0, 0, 0.0
        if field_z >= fields[-1]:
            last = int(fields.size - 1)
            return last, last, 0.0
        upper = int(np.searchsorted(fields, field_z, side="right"))
        lower = upper - 1
        x0 = float(fields[lower])
        x1 = float(fields[upper])
        weight = (float(field_z) - x0) / (x1 - x0)
        return lower, upper, float(weight)

    def _interpolate_matrix(self, values: Sequence[np.ndarray], field_z: float) -> np.ndarray:
        lower, upper, weight = self._interpolation_indices(field_z)
        if lower == upper:
            return np.asarray(values[lower], dtype=np.complex128)
        return np.asarray(
            (1.0 - weight) * np.asarray(values[lower], dtype=np.complex128)
            + weight * np.asarray(values[upper], dtype=np.complex128),
            dtype=np.complex128,
        )

    def coherent_bundle(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> OperatorBundle:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        if np.max(np.abs(magnetic - self.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedPopulationSinkEffectiveHamiltonianModel currently supports only the "
                "reference magnetic field."
            )
        if np.max(np.abs(electric[:2])) > 1e-12:
            raise ValueError(
                "PreparedPopulationSinkEffectiveHamiltonianModel currently supports interpolation "
                "only along Ez."
            )

        field_z = float(electric[2])
        h_internal = _symmetrize(
            self._interpolate_matrix([patch.coherent_bundle.h_internal for patch in self.patches], field_z)
        )
        h_opt = _symmetrize(
            self._interpolate_matrix([patch.coherent_bundle.h_opt for patch in self.patches], field_z)
        )
        h_det = _symmetrize(
            self._interpolate_matrix([patch.coherent_bundle.h_det for patch in self.patches], field_z)
        )
        dissipator_superop = self._interpolate_matrix(
            [patch.coherent_bundle.dissipator_superoperator() for patch in self.patches],
            field_z,
        )
        jump_rate_operator = _symmetrize(
            self._interpolate_matrix(
                [patch.coherent_bundle.jump_rate_operator() for patch in self.patches],
                field_z,
            )
        )
        loss_operator = _symmetrize(
            self._interpolate_matrix([patch.coherent_bundle.loss_operator for patch in self.patches], field_z)
        )
        h_full_internal = _symmetrize(
            self._interpolate_matrix([patch.coherent_bundle.h_full_internal for patch in self.patches], field_z)
        )
        h_lab_internal = _symmetrize(
            self._interpolate_matrix([patch.coherent_bundle.h_lab_internal for patch in self.patches], field_z)
        )

        return OperatorBundle(
            electric_field=electric,
            magnetic_field=magnetic,
            omega_reference=0.0,
            h_internal=h_internal,
            h_opt=h_opt,
            h_det=h_det,
            c_array=np.zeros_like(self.patches[0].coherent_bundle.c_array),
            excited_indices=np.asarray(self.coherent_excited_indices, dtype=np.int64),
            loss_operator=loss_operator,
            h_full_internal=h_full_internal,
            h_lab_internal=h_lab_internal,
            dissipator_superop=dissipator_superop,
            hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
            excited_to_ground_rates_hz=None,
            excited_to_sink_rates_hz=None,
            decay_kernel_ground=None,
            decay_kernels_sinks=None,
            jump_rate_operator_override=jump_rate_operator,
        )

    def sink_feed_matrix(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        electric = _as_field_vector(electric_field)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        if np.max(np.abs(magnetic - self.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedPopulationSinkEffectiveHamiltonianModel currently supports only the "
                "reference magnetic field."
            )
        if np.max(np.abs(electric[:2])) > 1e-12:
            raise ValueError(
                "PreparedPopulationSinkEffectiveHamiltonianModel currently supports interpolation "
                "only along Ez."
            )
        field_z = float(electric[2])
        return np.asarray(
            self._interpolate_matrix([patch.sink_feed_matrix for patch in self.patches], field_z),
            dtype=np.complex128,
        )


@dataclass
class PreparedInstantaneousInterpolatedEffectiveHamiltonianModel:
    transition: transitions.OpticalTransition
    optical_polarization: couplings.Polarization
    reference_magnetic_field: np.ndarray
    parent_basis_qn: list[states.CoupledState]
    union_state_keys: tuple[str, ...]
    field_points: np.ndarray
    master_field: float
    ground_indices: np.ndarray
    sink_indices: np.ndarray
    excited_indices: np.ndarray
    coherent_indices: np.ndarray
    ground_main_index: int
    patches: tuple[InstantaneousInterpolatedEffectivePatch, ...]
    keep_diagnostics: bool = True

    @property
    def n_effective_states(self) -> int:
        return int(self.patches[0].bundle.h_internal.shape[0])

    @property
    def ground_main_index_p(self) -> int:
        return int(self.ground_main_index)

    def _interpolation_indices(self, field_z: float) -> tuple[int, int, float]:
        fields = np.asarray(self.field_points, dtype=np.float64)
        if field_z <= fields[0]:
            return 0, 0, 0.0
        if field_z >= fields[-1]:
            last = int(fields.size - 1)
            return last, last, 0.0
        upper = int(np.searchsorted(fields, field_z, side="right"))
        lower = upper - 1
        x0 = float(fields[lower])
        x1 = float(fields[upper])
        weight = (float(field_z) - x0) / (x1 - x0)
        return lower, upper, float(weight)

    def _interpolate_matrix(self, values: Sequence[np.ndarray], field_z: float) -> np.ndarray:
        lower, upper, weight = self._interpolation_indices(field_z)
        if lower == upper:
            return np.asarray(values[lower], dtype=np.complex128)
        return np.asarray(
            (1.0 - weight) * np.asarray(values[lower], dtype=np.complex128)
            + weight * np.asarray(values[upper], dtype=np.complex128),
            dtype=np.complex128,
        )

    def effective_bundle(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
        magnetic_field: float | Sequence[float] | np.ndarray | None = None,
        electric_field_derivative: float | Sequence[float] | np.ndarray = 0.0,
    ) -> OperatorBundle:
        electric = _as_field_vector(electric_field)
        electric_dot = _as_field_vector(electric_field_derivative)
        magnetic = (
            np.asarray(self.reference_magnetic_field, dtype=np.float64)
            if magnetic_field is None
            else _as_field_vector(magnetic_field)
        )
        if np.max(np.abs(magnetic - self.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedInstantaneousInterpolatedEffectiveHamiltonianModel currently supports only the "
                "reference magnetic field."
            )
        if np.max(np.abs(electric[:2])) > 1e-12 or np.max(np.abs(electric_dot[:2])) > 1e-12:
            raise ValueError(
                "PreparedInstantaneousInterpolatedEffectiveHamiltonianModel currently supports interpolation "
                "only along Ez."
            )

        field_z = float(electric[2])
        gauge_connection = _symmetrize(
            self._interpolate_matrix([patch.gauge_connection for patch in self.patches], field_z)
        )
        h_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_internal for patch in self.patches], field_z)
            - float(electric_dot[2]) * gauge_connection
        )
        h_opt = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_opt for patch in self.patches], field_z)
        )
        h_det = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_det for patch in self.patches], field_z)
        )
        dissipator_superop = self._interpolate_matrix(
            [patch.bundle.dissipator_superoperator() for patch in self.patches],
            field_z,
        )
        jump_rate_operator = _symmetrize(
            self._interpolate_matrix([patch.bundle.jump_rate_operator() for patch in self.patches], field_z)
        )
        h_full_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_full_internal for patch in self.patches], field_z)
        )
        h_lab_internal = _symmetrize(
            self._interpolate_matrix([patch.bundle.h_lab_internal for patch in self.patches], field_z)
        )
        omega_reference = float(
            np.real(
                self._interpolate_matrix(
                    [np.array(patch.bundle.omega_reference, dtype=np.complex128) for patch in self.patches],
                    field_z,
                )
            )
        )

        decay_kernel_ground = None
        excited_to_ground_rates_hz = None
        if self.keep_diagnostics and all(patch.bundle.decay_kernel_ground is not None for patch in self.patches):
            decay_kernel_ground = _symmetrize(
                self._interpolate_matrix(
                    [np.asarray(patch.bundle.decay_kernel_ground, dtype=np.complex128) for patch in self.patches],
                    field_z,
                )
            )
            excited_to_ground_rates_hz = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)

        decay_kernels_sinks = None
        excited_to_sink_rates_hz = None
        if self.keep_diagnostics and all(patch.bundle.decay_kernels_sinks is not None for patch in self.patches):
            n_sinks = len(self.patches[0].bundle.decay_kernels_sinks or ())
            decay_kernels_sinks = tuple(
                _symmetrize(
                    self._interpolate_matrix(
                        [
                            np.asarray(patch.bundle.decay_kernels_sinks[sink_idx], dtype=np.complex128)
                            for patch in self.patches
                        ],
                        field_z,
                    )
                )
                for sink_idx in range(n_sinks)
            )
            excited_to_sink_rates_hz = np.sum(
                np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
                axis=0,
            ) / (2.0 * np.pi)

        return OperatorBundle(
            electric_field=electric,
            magnetic_field=magnetic,
            omega_reference=omega_reference,
            h_internal=h_internal,
            h_opt=h_opt,
            h_det=h_det,
            c_array=np.zeros_like(self.patches[0].bundle.c_array),
            excited_indices=np.asarray(self.excited_indices, dtype=np.int64),
            loss_operator=np.zeros_like(h_internal, dtype=np.complex128),
            h_full_internal=h_full_internal,
            h_lab_internal=h_lab_internal,
            dissipator_superop=dissipator_superop,
            hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
            excited_to_ground_rates_hz=excited_to_ground_rates_hz,
            excited_to_sink_rates_hz=excited_to_sink_rates_hz,
            decay_kernel_ground=decay_kernel_ground,
            decay_kernels_sinks=decay_kernels_sinks,
            jump_rate_operator_override=jump_rate_operator,
        )


@dataclass
class PatchwiseInstantaneousSolveResult:
    t: np.ndarray
    y: np.ndarray
    patch_indices: np.ndarray
    switch_times: np.ndarray
    success: bool = True
    message: str = ""


@dataclass
class PreparedPatchwiseInstantaneousEffectiveHamiltonianModel:
    base_model: PreparedInstantaneousInterpolatedEffectiveHamiltonianModel
    interval_boundaries: np.ndarray
    forward_transforms: tuple[np.ndarray, ...]

    def __getattr__(self, name: str):
        return getattr(self.base_model, name)

    def patch_index_for_field(
        self,
        electric_field: float | Sequence[float] | np.ndarray,
    ) -> int:
        electric = _as_field_vector(electric_field)
        if np.max(np.abs(electric[:2])) > 1e-12:
            raise ValueError(
                "PreparedPatchwiseInstantaneousEffectiveHamiltonianModel currently supports only Ez."
            )
        boundaries = np.asarray(self.interval_boundaries, dtype=np.float64)
        if boundaries.size == 0:
            return 0
        return int(np.searchsorted(boundaries, float(electric[2]), side="right"))

    def patch_bundle(self, patch_index: int) -> OperatorBundle:
        return self.base_model.patches[int(patch_index)].bundle

    def transform_between_patches(
        self,
        from_index: int,
        to_index: int,
    ) -> np.ndarray:
        from_index = int(from_index)
        to_index = int(to_index)
        n = int(self.n_effective_states)
        if from_index == to_index:
            return np.eye(n, dtype=np.complex128)
        transform = np.eye(n, dtype=np.complex128)
        if to_index > from_index:
            for idx in range(from_index, to_index):
                transform = np.asarray(self.forward_transforms[idx] @ transform, dtype=np.complex128)
        else:
            for idx in range(from_index - 1, to_index - 1, -1):
                transform = np.asarray(self.forward_transforms[idx].conj().T @ transform, dtype=np.complex128)
        return np.asarray(transform, dtype=np.complex128)


def prepare_effective_hamiltonian_model(
    *,
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    space_generation_electric_field: Sequence[float] | np.ndarray = (0.0, 0.0, 100.0),
    space_generation_magnetic_field: Sequence[float] | np.ndarray | None = None,
    reference_electric_field: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    reference_magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    include_opposite_parity_manifold_in_p: bool = False,
    kept_state_dressing_order: int = 3,
    excited_state_dressing_order: int = 4,
    gamma: float = hamiltonian.Γ,
    denominator_floor: float = 2.0 * np.pi * 1e3,
) -> PreparedEffectiveHamiltonianModel:
    space_generation_electric = _as_field_vector(space_generation_electric_field)
    reference_electric = _as_field_vector(reference_electric_field)
    reference_magnetic = _as_field_vector(reference_magnetic_field)
    space_generation_magnetic = (
        np.asarray(reference_magnetic, dtype=np.float64)
        if space_generation_magnetic_field is None
        else _as_field_vector(space_generation_magnetic_field)
    )

    transition_selector = couplings.generate_transition_selectors(
        transitions=[transition],
        polarizations=[[optical_polarization]],
    )[0]

    auto_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        [transition],
        E=space_generation_electric,
        B=space_generation_magnetic,
        use_omega_basis=False,
    )
    b_parent_states = list(auto_reduced.B_hamiltonian.QN_construct)
    b_parent_j_values = np.unique([int(state.J) for state in b_parent_states]).astype(int)
    reference_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        auto_reduced.X_states_basis,
        b_parent_states,
        E=reference_electric,
        B=reference_magnetic,
        Jmin_B=int(b_parent_j_values.min()),
        Jmax_B=int(b_parent_j_values.max()),
        use_omega_basis=False,
    )

    qn_full = list(reference_reduced.QN)
    x_states = list(reference_reduced.X_states)
    b_states = list(reference_reduced.B_states)

    x_transform_ref = _basis_transform(
        reference_reduced.X_states,
        reference_reduced.X_hamiltonian.QN_construct,
    )
    x_construct_transform = reference_reduced.X_hamiltonian.transform
    if x_construct_transform is None:
        raise ValueError("X-state construction transform is missing")
    x_terms = reference_reduced.X_hamiltonian.hamiltonian
    if x_terms is None:
        raise ValueError("X-state Hamiltonian terms are missing")

    def reduce_x_term(term_name: str) -> np.ndarray:
        term_uncoupled = np.asarray(getattr(x_terms, term_name), dtype=np.complex128)
        term_coupled = x_construct_transform.conj().T @ term_uncoupled @ x_construct_transform
        return _symmetrize(x_transform_ref.conj().T @ term_coupled @ x_transform_ref)

    b_transform_ref = _basis_transform(
        reference_reduced.B_states,
        reference_reduced.B_hamiltonian.QN_construct,
    )
    b_terms = reference_reduced.B_hamiltonian.hamiltonian
    if b_terms is None:
        raise ValueError("B-state Hamiltonian terms are missing")

    def reduce_b_term(term_name: str) -> np.ndarray:
        term = np.asarray(getattr(b_terms, term_name), dtype=np.complex128)
        return _symmetrize(b_transform_ref.conj().T @ term @ b_transform_ref)

    x_h_sx = reduce_x_term("HSx")
    x_h_sy = reduce_x_term("HSy")
    x_h_sz = reduce_x_term("HSz")
    x_h_zx = reduce_x_term("HZx")
    x_h_zy = reduce_x_term("HZy")
    x_h_zz = reduce_x_term("HZz")

    b_h_sx = reduce_b_term("HSx")
    b_h_sy = reduce_b_term("HSy")
    b_h_sz = reduce_b_term("HSz")
    b_h_zx = reduce_b_term("HZx")
    b_h_zy = reduce_b_term("HZy")
    b_h_zz = reduce_b_term("HZz")

    h_sx = block_diag(x_h_sx, b_h_sx).astype(np.complex128)
    h_sy = block_diag(x_h_sy, b_h_sy).astype(np.complex128)
    h_sz = block_diag(x_h_sz, b_h_sz).astype(np.complex128)
    h_zx = block_diag(x_h_zx, b_h_zx).astype(np.complex128)
    h_zy = block_diag(x_h_zy, b_h_zy).astype(np.complex128)
    h_zz = block_diag(x_h_zz, b_h_zz).astype(np.complex128)
    h_reference_internal = np.asarray(reference_reduced.H_int, dtype=np.complex128)

    h_opt_full, main_coupling, ground_main_index_full, excited_main_index_full = _operator_from_transition(
        qn_full,
        x_states,
        b_states,
        transition_selector,
    )

    h_det_full = np.zeros_like(h_reference_internal, dtype=np.complex128)
    full_excited_indices = np.arange(len(x_states), len(qn_full), dtype=np.int64)
    h_det_full[np.ix_(full_excited_indices, full_excited_indices)] = np.eye(
        len(full_excited_indices),
        dtype=np.complex128,
    )
    qn_full_coupling = [
        (1.0 * state if isinstance(state, states.CoupledBasisState) else state)
        for state in qn_full
    ]
    x_states_coupling = [
        (1.0 * state if isinstance(state, states.CoupledBasisState) else state)
        for state in x_states
    ]
    b_states_coupling = [
        (1.0 * state if isinstance(state, states.CoupledBasisState) else state)
        for state in b_states
    ]
    c_array_full = couplings.collapse_matrices(
        qn_full_coupling,
        x_states_coupling,
        b_states_coupling,
        gamma=gamma,
    ).astype(np.complex128)

    p_ground_indices = _selector_indices(
        qn_full,
        states.QuantumSelector(J=0, electronic=states.ElectronicState.X),
    )
    p_excited_indices_b = states.find_exact_states_indices(
        [1 * state for state in auto_reduced.B_states_basis],
        reference_reduced.B_hamiltonian.QN_construct,
        V=reference_reduced.B_hamiltonian.V,
    ).astype(np.int64)
    p_excited_indices = (len(x_states) + p_excited_indices_b).astype(np.int64)
    if include_opposite_parity_manifold_in_p:
        opposite_parity_indices = _selector_indices(
            qn_full,
            states.QuantumSelector(
                J=1,
                F1=1 / 2,
                F=1,
                P=+1,
                electronic=states.ElectronicState.B,
            ),
        )
        p_excited_indices = np.union1d(
            p_excited_indices,
            opposite_parity_indices.astype(np.int64),
        ).astype(np.int64)
    sink_indices_full = tuple(
        _selector_indices(
            qn_full,
            states.QuantumSelector(J=J, electronic=states.ElectronicState.X),
        )
        for J in (1, 2, 3)
    )
    sink_labels = tuple(f"X, J={J} sink" for J in (1, 2, 3))
    p_indices = np.sort(np.concatenate([p_ground_indices, p_excited_indices])).astype(np.int64)
    q_indices = np.setdiff1d(np.arange(len(qn_full), dtype=np.int64), p_indices, assume_unique=True)

    covered_indices = np.sort(np.concatenate([p_indices, q_indices])).astype(np.int64)
    if not np.array_equal(covered_indices, np.arange(len(qn_full), dtype=np.int64)):
        raise ValueError(
            "The chosen active subspace plus eliminated subspace do not cover the full "
            "parent Hilbert space."
        )
    for sink_indices in sink_indices_full:
        if not np.all(np.isin(sink_indices, q_indices)):
            raise ValueError("Compact sink manifolds must be contained in the eliminated Q subspace.")

    if ground_main_index_full not in p_indices or excited_main_index_full not in p_indices:
        raise ValueError("The automatically chosen main optical states must lie in the kept subspace P")

    ground_main_index_p = int(np.flatnonzero(p_indices == ground_main_index_full)[0])
    excited_main_index_p = int(np.flatnonzero(p_indices == excited_main_index_full)[0])
    p_excited_indices_local = np.flatnonzero(np.isin(p_indices, full_excited_indices)).astype(
        np.int64
    )
    q_lookup = {int(index): idx for idx, index in enumerate(q_indices.tolist())}
    sink_indices_q = tuple(
        np.array([q_lookup[int(index)] for index in sink_indices], dtype=np.int64)
        for sink_indices in sink_indices_full
    )

    h0_blocks = _partition_matrix(h_reference_internal, p_indices, q_indices)
    hsx_blocks = _partition_matrix(h_sx, p_indices, q_indices)
    hsy_blocks = _partition_matrix(h_sy, p_indices, q_indices)
    hsz_blocks = _partition_matrix(h_sz, p_indices, q_indices)
    hzx_blocks = _partition_matrix(h_zx, p_indices, q_indices)
    hzy_blocks = _partition_matrix(h_zy, p_indices, q_indices)
    hzz_blocks = _partition_matrix(h_zz, p_indices, q_indices)
    h_opt_blocks = _partition_matrix(h_opt_full, p_indices, q_indices)
    h_det_blocks = _partition_matrix(h_det_full, p_indices, q_indices)
    jump_blocks = tuple(_partition_matrix(c_op, p_indices, q_indices) for c_op in c_array_full)

    model = PreparedEffectiveHamiltonianModel(
        transition=transition,
        optical_polarization=optical_polarization,
        transition_selector=transition_selector,
        space_generation_electric_field=space_generation_electric,
        space_generation_magnetic_field=space_generation_magnetic,
        reference_electric_field=reference_electric,
        reference_magnetic_field=reference_magnetic,
        qn_full=qn_full,
        x_states=x_states,
        b_states=b_states,
        h_reference_internal=h_reference_internal,
        h_sx=h_sx,
        h_sy=h_sy,
        h_sz=h_sz,
        h_zx=h_zx,
        h_zy=h_zy,
        h_zz=h_zz,
        h0_blocks=h0_blocks,
        hsx_blocks=hsx_blocks,
        hsy_blocks=hsy_blocks,
        hsz_blocks=hsz_blocks,
        hzx_blocks=hzx_blocks,
        hzy_blocks=hzy_blocks,
        hzz_blocks=hzz_blocks,
        h_opt_full=h_opt_full,
        h_det_full=h_det_full,
        c_array_full=c_array_full,
        h_opt_blocks=h_opt_blocks,
        h_det_blocks=h_det_blocks,
        jump_blocks=jump_blocks,
        p_indices=p_indices,
        q_indices=q_indices,
        p_ground_indices=p_ground_indices,
        p_excited_indices=p_excited_indices,
        full_excited_indices=full_excited_indices,
        p_excited_indices_local=p_excited_indices_local,
        sink_indices_full=sink_indices_full,
        sink_indices_q=sink_indices_q,
        sink_labels=sink_labels,
        ground_main_index_full=ground_main_index_full,
        excited_main_index_full=excited_main_index_full,
        ground_main_index_p=ground_main_index_p,
        excited_main_index_p=excited_main_index_p,
        main_coupling=main_coupling,
        gamma=gamma,
        denominator_floor=float(denominator_floor),
        kept_state_dressing_order=int(kept_state_dressing_order),
        excited_state_dressing_order=int(excited_state_dressing_order),
    )
    if (
        int(model.n_effective_states) == 10
        and int(model.p_ground_indices.size) == 4
        and int(model.p_excited_indices_local.size) == 3
    ):
        _, compact_reference_bundle = build_compact_reference_decomposed_bundle(
            transition=transition,
            optical_polarization=optical_polarization,
            electric_field=reference_electric,
            magnetic_field=reference_magnetic,
        )
        model.compact_reference_bundle = compact_reference_bundle
        model.perturbative_reference_bundle = model._effective_bundle_perturbative(
            reference_electric,
            reference_magnetic,
        )
    return model


def prepare_local_operator_expansion_model(
    *,
    reference_electric_field: float | Sequence[float] | np.ndarray,
    expansion_axis: int = 2,
    expansion_order: int = 2,
    expansion_step: float = 10.0,
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    space_generation_electric_field: Sequence[float] | np.ndarray | None = None,
    reference_magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    gamma: float = hamiltonian.Γ,
    denominator_floor: float = 2.0 * np.pi * 1e3,
    kept_state_dressing_order: int = 3,
    excited_state_dressing_order: int = 4,
) -> PreparedLocalOperatorExpansionModel:
    if expansion_axis not in (0, 1, 2):
        raise ValueError(f"expansion_axis must be 0, 1, or 2, not {expansion_axis}")
    if expansion_order < 0:
        raise ValueError(f"expansion_order must be non-negative, not {expansion_order}")
    if expansion_step <= 0.0:
        raise ValueError(f"expansion_step must be positive, not {expansion_step}")

    reference_electric = _as_field_vector(reference_electric_field)
    space_generation_electric = (
        np.asarray(reference_electric, dtype=np.float64)
        if space_generation_electric_field is None
        else _as_field_vector(space_generation_electric_field)
    )
    base_model = prepare_effective_hamiltonian_model(
        transition=transition,
        optical_polarization=optical_polarization,
        space_generation_electric_field=space_generation_electric,
        reference_electric_field=reference_electric,
        reference_magnetic_field=reference_magnetic_field,
        gamma=gamma,
        denominator_floor=denominator_floor,
        kept_state_dressing_order=kept_state_dressing_order,
        excited_state_dressing_order=excited_state_dressing_order,
    )

    stencil_indices = np.arange(-expansion_order, expansion_order + 1, dtype=np.float64)
    offsets = stencil_indices * float(expansion_step)
    sampled_fields: list[np.ndarray] = []
    sampled_bundles: list[OperatorBundle] = []
    for offset in offsets:
        electric = np.asarray(reference_electric, dtype=np.float64).copy()
        electric[expansion_axis] += float(offset)
        sampled_fields.append(electric)
        sampled_bundles.append(base_model.effective_bundle(electric))

    h_internal_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [bundle.h_internal for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )
    h_opt_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [bundle.h_opt for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )
    h_det_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [bundle.h_det for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )
    dissipator_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [bundle.dissipator_superoperator() for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )
    jump_rate_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [bundle.jump_rate_operator() for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )
    h_full_internal_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [bundle.h_full_internal for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )
    h_lab_internal_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [bundle.h_lab_internal for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )
    omega_reference_coefficients = tuple(
        _polynomial_coefficients_from_samples(
            [np.array(bundle.omega_reference, dtype=np.complex128) for bundle in sampled_bundles],
            offsets,
            expansion_order,
        )
    )

    decay_kernel_ground_coefficients = None
    if all(bundle.decay_kernel_ground is not None for bundle in sampled_bundles):
        decay_kernel_ground_coefficients = tuple(
            _polynomial_coefficients_from_samples(
                [bundle.decay_kernel_ground for bundle in sampled_bundles],
                offsets,
                expansion_order,
            )
        )

    decay_kernels_sinks_coefficients = None
    if all(bundle.decay_kernels_sinks is not None for bundle in sampled_bundles):
        n_sinks = len(sampled_bundles[0].decay_kernels_sinks or ())
        decay_kernels_sinks_coefficients = tuple(
            tuple(
                _polynomial_coefficients_from_samples(
                    [bundle.decay_kernels_sinks[sink_index] for bundle in sampled_bundles],  # type: ignore[index]
                    offsets,
                    expansion_order,
                )
            )
            for sink_index in range(n_sinks)
        )

    return PreparedLocalOperatorExpansionModel(
        base_model=base_model,
        expansion_axis=int(expansion_axis),
        expansion_step=float(expansion_step),
        expansion_order=int(expansion_order),
        reference_bundle=sampled_bundles[int(expansion_order)],
        h_internal_coefficients=h_internal_coefficients,
        h_opt_coefficients=h_opt_coefficients,
        h_det_coefficients=h_det_coefficients,
        dissipator_coefficients=dissipator_coefficients,
        jump_rate_coefficients=jump_rate_coefficients,
        h_full_internal_coefficients=h_full_internal_coefficients,
        h_lab_internal_coefficients=h_lab_internal_coefficients,
        omega_reference_coefficients=omega_reference_coefficients,
        decay_kernel_ground_coefficients=decay_kernel_ground_coefficients,
        decay_kernels_sinks_coefficients=decay_kernels_sinks_coefficients,
    )


def prepare_interpolated_effective_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    interpolation_kind: str = "linear",
    keep_diagnostics: bool = True,
) -> PreparedInterpolatedEffectiveHamiltonianModel:
    if interpolation_kind != "linear":
        raise ValueError(
            f"Only linear interpolation is currently supported, not {interpolation_kind!r}."
        )
    if len(field_points) < 2:
        raise ValueError("At least two field_points are required to build an interpolated model.")

    parsed_fields = np.array([_as_field_vector(field)[2] for field in field_points], dtype=np.float64)
    if np.max(np.abs(np.array([_as_field_vector(field)[:2] for field in field_points], dtype=np.float64))) > 1e-12:
        raise ValueError("prepare_interpolated_effective_model currently supports only Ez field points.")
    unique_sorted = np.array(sorted(set(float(value) for value in parsed_fields.tolist())), dtype=np.float64)
    if unique_sorted.size < 2:
        raise ValueError("field_points must contain at least two distinct Ez values.")

    master_value = (
        float(unique_sorted[len(unique_sorted) // 2])
        if master_field is None
        else float(_as_field_vector(master_field)[2])
    )
    if master_value not in unique_sorted:
        raise ValueError("master_field must match one of the supplied field_points.")

    magnetic = _as_field_vector(magnetic_field)
    master_electric = np.array([0.0, 0.0, master_value], dtype=np.float64)
    auto_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        [transition],
        E=master_electric,
        B=magnetic,
        use_omega_basis=False,
    )
    b_parent_states = list(auto_reduced.B_hamiltonian.QN_construct)
    b_parent_j_values = np.unique([int(state.J) for state in b_parent_states]).astype(int)
    parent_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        auto_reduced.X_states_basis,
        b_parent_states,
        E=master_electric,
        B=magnetic,
        Jmin_B=int(b_parent_j_values.min()),
        Jmax_B=int(b_parent_j_values.max()),
        use_omega_basis=False,
    )
    parent_basis_qn = list(parent_reduced.QN)
    ground_selector = states.QuantumSelector(
        J=int(transition.J_ground),
        electronic=transition.electronic_ground,
    )

    patch_systems: list[lindblad.utils_setup.OBESystem] = []
    patch_bundles: list[OperatorBundle] = []
    for field_z in unique_sorted.tolist():
        electric = np.array([0.0, 0.0, float(field_z)], dtype=np.float64)
        system, bundle = build_compact_reference_decomposed_bundle(
            transition=transition,
            optical_polarization=optical_polarization,
            electric_field=electric,
            magnetic_field=magnetic,
        )
        patch_systems.append(system)
        patch_bundles.append(bundle)

    master_index = int(np.where(unique_sorted == master_value)[0][0])
    ground_indices, sink_indices, excited_indices, sink_union_keys, reference_ground_basis, reference_excited_basis = _build_compact_union_layout(
        patch_systems,
        patch_bundles,
        parent_basis_qn,
        master_index=master_index,
        ground_selector=ground_selector,
    )

    raw_bases: list[np.ndarray] = []
    raw_bundles: list[OperatorBundle] = []
    for system, bundle in zip(patch_systems, patch_bundles):
        basis_vectors, reordered_bundle = _embed_aligned_compact_patch_to_union_layout(
            system,
            bundle,
            parent_basis_qn,
            ground_selector=ground_selector,
            sink_union_keys=sink_union_keys,
            ground_indices_union=ground_indices,
            excited_indices_union=excited_indices,
            reference_ground_basis=reference_ground_basis,
            reference_excited_basis=reference_excited_basis,
            use_identity_rotation=(system is patch_systems[master_index] and bundle is patch_bundles[master_index]),
        )
        raw_bases.append(np.asarray(basis_vectors, dtype=np.complex128))
        raw_bundles.append(_make_bundle_sinks_dissipative_only(reordered_bundle, sink_indices))

    aligned_bases: list[np.ndarray | None] = [None] * len(unique_sorted)
    aligned_bundles: list[OperatorBundle | None] = [None] * len(unique_sorted)
    aligned_bases[master_index] = raw_bases[master_index]
    aligned_bundles[master_index] = raw_bundles[master_index]

    for index in range(master_index + 1, len(unique_sorted)):
        previous_basis = np.asarray(aligned_bases[index - 1], dtype=np.complex128)
        current_basis = raw_bases[index]
        aligned_basis, rotation = _union_block_alignment(
            previous_basis,
            current_basis,
            ground_indices=ground_indices,
            sink_indices=sink_indices,
            excited_indices=excited_indices,
        )
        aligned_bases[index] = aligned_basis
        excited_rotation = rotation[np.ix_(excited_indices, excited_indices)]
        aligned_bundles[index] = _make_bundle_sinks_dissipative_only(
            _transform_operator_bundle(
                raw_bundles[index],
                rotation,
                decay_kernel_ground=_symmetrize(
                    excited_rotation
                    @ np.asarray(raw_bundles[index].decay_kernel_ground, dtype=np.complex128)
                    @ excited_rotation.conj().T
                )
                if raw_bundles[index].decay_kernel_ground is not None
                else None,
                decay_kernels_sinks=tuple(
                    _symmetrize(
                        excited_rotation
                        @ np.asarray(kernel, dtype=np.complex128)
                        @ excited_rotation.conj().T
                    )
                    for kernel in (raw_bundles[index].decay_kernels_sinks or ())
                )
                if raw_bundles[index].decay_kernels_sinks is not None
                else None,
            ),
            sink_indices,
        )

    for index in range(master_index - 1, -1, -1):
        previous_basis = np.asarray(aligned_bases[index + 1], dtype=np.complex128)
        current_basis = raw_bases[index]
        aligned_basis, rotation = _union_block_alignment(
            previous_basis,
            current_basis,
            ground_indices=ground_indices,
            sink_indices=sink_indices,
            excited_indices=excited_indices,
        )
        aligned_bases[index] = aligned_basis
        excited_rotation = rotation[np.ix_(excited_indices, excited_indices)]
        aligned_bundles[index] = _make_bundle_sinks_dissipative_only(
            _transform_operator_bundle(
                raw_bundles[index],
                rotation,
                decay_kernel_ground=_symmetrize(
                    excited_rotation
                    @ np.asarray(raw_bundles[index].decay_kernel_ground, dtype=np.complex128)
                    @ excited_rotation.conj().T
                )
                if raw_bundles[index].decay_kernel_ground is not None
                else None,
                decay_kernels_sinks=tuple(
                    _symmetrize(
                        excited_rotation
                        @ np.asarray(kernel, dtype=np.complex128)
                        @ excited_rotation.conj().T
                    )
                    for kernel in (raw_bundles[index].decay_kernels_sinks or ())
                )
                if raw_bundles[index].decay_kernels_sinks is not None
                else None,
            ),
            sink_indices,
        )

    patches = tuple(
        InterpolatedEffectivePatch(
            electric_field=np.array([0.0, 0.0, float(field_z)], dtype=np.float64),
            aligned_basis_vectors=np.asarray(aligned_basis, dtype=np.complex128),
            bundle=aligned_bundle,
        )
        for field_z, aligned_basis, aligned_bundle in zip(
            unique_sorted.tolist(),
            aligned_bases,
            aligned_bundles,
        )
    )
    return PreparedInterpolatedEffectiveHamiltonianModel(
        transition=transition,
        optical_polarization=optical_polarization,
        reference_magnetic_field=magnetic,
        parent_basis_qn=parent_basis_qn,
        union_state_keys=tuple(sink_union_keys),
        field_points=unique_sorted,
        master_field=float(master_value),
        ground_indices=np.asarray(ground_indices, dtype=np.int64),
        sink_indices=np.asarray(sink_indices, dtype=np.int64),
        excited_indices=np.asarray(excited_indices, dtype=np.int64),
        ground_main_index=int(_optically_bright_ground_index(raw_bundles[master_index], excited_indices)),
        patches=patches,
        keep_diagnostics=bool(keep_diagnostics),
    )


def prepare_lindblad_safe_compact_interpolated_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    interpolation_kind: str = "linear",
    keep_diagnostics: bool = True,
) -> PreparedLindbladSafeCompactInterpolatedHamiltonianModel:
    base_model = prepare_interpolated_effective_model(
        field_points=field_points,
        transition=transition,
        optical_polarization=optical_polarization,
        magnetic_field=magnetic_field,
        master_field=master_field,
        interpolation_kind=interpolation_kind,
        keep_diagnostics=keep_diagnostics,
    )
    patch_transition_frequencies: list[float] = []
    for field_z in np.asarray(base_model.field_points, dtype=np.float64).tolist():
        electric = np.array([0.0, 0.0, float(field_z)], dtype=np.float64)
        system, _ = build_compact_reference_decomposed_bundle(
            transition=transition,
            optical_polarization=optical_polarization,
            electric_field=electric,
            magnetic_field=magnetic_field,
        )
        patch_transition_frequencies.append(
            _compact_transition_frequency(
                system,
                transition=transition,
                optical_polarization=optical_polarization,
            )
        )
    patch_transition_frequencies_arr = np.asarray(patch_transition_frequencies, dtype=np.float64)
    master_index = int(np.argmin(np.abs(np.asarray(base_model.field_points, dtype=np.float64) - float(base_model.master_field))))
    common_omega_reference = float(patch_transition_frequencies_arr[master_index])
    target_indices = np.concatenate([base_model.ground_indices, base_model.sink_indices]).astype(np.int64)
    lindblad_safe_patches: list[LindbladSafeCompactInterpolatedPatch] = []
    for index, patch in enumerate(base_model.patches):
        shifted_bundle = _shift_bundle_to_common_frequency_frame(
            patch.bundle,
            delta_omega=float(patch_transition_frequencies_arr[index] - common_omega_reference),
            common_omega_reference=common_omega_reference,
        )
        lindblad_safe_patches.append(
            LindbladSafeCompactInterpolatedPatch(
                electric_field=np.asarray(patch.electric_field, dtype=np.float64),
                aligned_basis_vectors=np.asarray(patch.aligned_basis_vectors, dtype=np.complex128),
                bundle=shifted_bundle,
                target_decay_kernels=_single_target_decay_kernels(
                    np.asarray(patch.bundle.c_array, dtype=np.complex128),
                    np.asarray(base_model.excited_indices, dtype=np.int64),
                    target_indices,
                ),
                full_recycling_decay_kernel=_full_recycling_decay_kernel(
                    np.asarray(patch.bundle.c_array, dtype=np.complex128),
                    np.arange(int(patch.bundle.h_internal.shape[0]), dtype=np.int64),
                    np.arange(int(patch.bundle.h_internal.shape[0]), dtype=np.int64),
                ),
            )
        )
    patches = tuple(lindblad_safe_patches)
    return PreparedLindbladSafeCompactInterpolatedHamiltonianModel(
        transition=base_model.transition,
        optical_polarization=base_model.optical_polarization,
        reference_magnetic_field=np.asarray(base_model.reference_magnetic_field, dtype=np.float64),
        parent_basis_qn=list(base_model.parent_basis_qn),
        union_state_keys=tuple(base_model.union_state_keys),
        field_points=np.asarray(base_model.field_points, dtype=np.float64),
        master_field=float(base_model.master_field),
        ground_indices=np.asarray(base_model.ground_indices, dtype=np.int64),
        sink_indices=np.asarray(base_model.sink_indices, dtype=np.int64),
        excited_indices=np.asarray(base_model.excited_indices, dtype=np.int64),
        target_indices=target_indices,
        ground_main_index=int(base_model.ground_main_index),
        common_omega_reference=float(common_omega_reference),
        patch_transition_frequencies=np.asarray(patch_transition_frequencies_arr, dtype=np.float64),
        patches=patches,
        keep_diagnostics=bool(keep_diagnostics),
    )


def _population_sink_label_from_union_key(key: str, fallback_index: int) -> str:
    match = re.search(r"J\s*=\s*([0-9]+(?:\.[0-9]+)?)", key)
    if match is None:
        return f"sink_{fallback_index}"
    j_value = match.group(1)
    if j_value.endswith(".0"):
        j_value = j_value[:-2]
    return f"X, J={j_value} sink"


def _population_sink_bundle_from_patch(
    bundle: OperatorBundle,
    *,
    ground_indices: np.ndarray,
    sink_indices: np.ndarray,
    excited_indices: np.ndarray,
) -> tuple[OperatorBundle, np.ndarray]:
    coherent_indices = np.concatenate(
        [
            np.asarray(ground_indices, dtype=np.int64),
            np.asarray(excited_indices, dtype=np.int64),
        ]
    )
    coherent_dim = int(coherent_indices.size)
    local_excited_indices = np.arange(
        int(ground_indices.size),
        coherent_dim,
        dtype=np.int64,
    )
    coherent_h_internal = _symmetrize(
        np.asarray(bundle.h_internal[np.ix_(coherent_indices, coherent_indices)], dtype=np.complex128)
    )
    coherent_h_opt = _symmetrize(
        np.asarray(bundle.h_opt[np.ix_(coherent_indices, coherent_indices)], dtype=np.complex128)
    )
    coherent_h_det = _symmetrize(
        np.asarray(bundle.h_det[np.ix_(coherent_indices, coherent_indices)], dtype=np.complex128)
    )
    coherent_h_full_internal = _symmetrize(
        np.asarray(bundle.h_full_internal[np.ix_(coherent_indices, coherent_indices)], dtype=np.complex128)
    )
    coherent_h_lab_internal = _symmetrize(
        np.asarray(bundle.h_lab_internal[np.ix_(coherent_indices, coherent_indices)], dtype=np.complex128)
    )
    full_dim = int(bundle.h_internal.shape[0])
    full_liouvillian = np.asarray(bundle.dissipator_superoperator(), dtype=np.complex128)
    jump_rate_full = np.asarray(bundle.jump_rate_operator(), dtype=np.complex128)

    embed = np.zeros((full_dim * full_dim, coherent_dim * coherent_dim), dtype=np.complex128)
    extract = np.zeros((coherent_dim * coherent_dim, full_dim * full_dim), dtype=np.complex128)
    full_lookup = {int(full_index): int(local_index) for local_index, full_index in enumerate(coherent_indices.tolist())}
    for full_i, local_i in full_lookup.items():
        for full_j, local_j in full_lookup.items():
            embed[full_i * full_dim + full_j, local_i * coherent_dim + local_j] = 1.0
            extract[local_i * coherent_dim + local_j, full_i * full_dim + full_j] = 1.0

    dissipator_superop = np.asarray(extract @ full_liouvillian @ embed, dtype=np.complex128)
    sink_feed_matrix = np.zeros((len(sink_indices), coherent_dim * coherent_dim), dtype=np.complex128)
    for sink_local, sink_index in enumerate(np.asarray(sink_indices, dtype=np.int64).tolist()):
        sink_feed_matrix[sink_local, :] = (
            np.eye(full_dim * full_dim, dtype=np.complex128)[int(sink_index) * full_dim + int(sink_index), :]
            @ full_liouvillian
            @ embed
        )

    jump_rate_operator = _symmetrize(
        np.asarray(jump_rate_full[np.ix_(coherent_indices, coherent_indices)], dtype=np.complex128)
    )

    coherent_bundle = OperatorBundle(
        electric_field=np.asarray(bundle.electric_field, dtype=np.float64),
        magnetic_field=np.asarray(bundle.magnetic_field, dtype=np.float64),
        omega_reference=float(bundle.omega_reference),
        h_internal=coherent_h_internal,
        h_opt=coherent_h_opt,
        h_det=coherent_h_det,
        c_array=np.zeros((0, coherent_dim, coherent_dim), dtype=np.complex128),
        excited_indices=np.asarray(local_excited_indices, dtype=np.int64),
        loss_operator=np.zeros((coherent_dim, coherent_dim), dtype=np.complex128),
        h_full_internal=coherent_h_full_internal,
        h_lab_internal=coherent_h_lab_internal,
        dissipator_superop=dissipator_superop,
        hermiticity_error=float(np.max(np.abs(coherent_h_internal - coherent_h_internal.conj().T))),
        excited_to_ground_rates_hz=None,
        excited_to_sink_rates_hz=None,
        decay_kernel_ground=None,
        decay_kernels_sinks=None,
        jump_rate_operator_override=jump_rate_operator,
    )
    return coherent_bundle, np.asarray(sink_feed_matrix, dtype=np.complex128)


def prepare_population_sink_effective_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    interpolation_kind: str = "linear",
) -> PreparedPopulationSinkEffectiveHamiltonianModel:
    if interpolation_kind != "linear":
        raise ValueError(
            f"Only linear interpolation is currently supported, not {interpolation_kind!r}."
        )
    if len(field_points) < 2:
        raise ValueError("At least two field_points are required to build a population-sink model.")

    parsed_fields = np.array([_as_field_vector(field)[2] for field in field_points], dtype=np.float64)
    if np.max(np.abs(np.array([_as_field_vector(field)[:2] for field in field_points], dtype=np.float64))) > 1e-12:
        raise ValueError("prepare_population_sink_effective_model currently supports only Ez field points.")
    unique_sorted = np.array(sorted(set(float(value) for value in parsed_fields.tolist())), dtype=np.float64)
    if unique_sorted.size < 2:
        raise ValueError("field_points must contain at least two distinct Ez values.")

    master_value = (
        float(unique_sorted[len(unique_sorted) // 2])
        if master_field is None
        else float(_as_field_vector(master_field)[2])
    )
    if master_value not in unique_sorted:
        raise ValueError("master_field must match one of the supplied field_points.")

    magnetic = _as_field_vector(magnetic_field)
    master_electric = np.array([0.0, 0.0, master_value], dtype=np.float64)
    auto_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        [transition],
        E=master_electric,
        B=magnetic,
        use_omega_basis=False,
    )
    b_parent_states = list(auto_reduced.B_hamiltonian.QN_construct)
    b_parent_j_values = np.unique([int(state.J) for state in b_parent_states]).astype(int)
    parent_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        auto_reduced.X_states_basis,
        b_parent_states,
        E=master_electric,
        B=magnetic,
        Jmin_B=int(b_parent_j_values.min()),
        Jmax_B=int(b_parent_j_values.max()),
        use_omega_basis=False,
    )
    parent_basis_qn = list(parent_reduced.QN)
    ground_selector = states.QuantumSelector(
        J=int(transition.J_ground),
        electronic=transition.electronic_ground,
    )

    patch_systems: list[lindblad.utils_setup.OBESystem] = []
    patch_bundles: list[OperatorBundle] = []
    for field_z in unique_sorted.tolist():
        electric = np.array([0.0, 0.0, float(field_z)], dtype=np.float64)
        system, bundle = build_compact_reference_decomposed_bundle(
            transition=transition,
            optical_polarization=optical_polarization,
            electric_field=electric,
            magnetic_field=magnetic,
        )
        patch_systems.append(system)
        patch_bundles.append(bundle)

    master_index = int(np.where(unique_sorted == master_value)[0][0])
    (
        ground_indices,
        sink_indices,
        excited_indices,
        sink_union_keys,
        reference_ground_basis,
        reference_excited_basis,
    ) = _build_compact_union_layout(
        patch_systems,
        patch_bundles,
        parent_basis_qn,
        master_index=master_index,
        ground_selector=ground_selector,
    )

    raw_bases: list[np.ndarray] = []
    raw_bundles: list[OperatorBundle] = []
    for system, bundle in zip(patch_systems, patch_bundles):
        basis_vectors, reordered_bundle = _embed_aligned_compact_patch_to_union_layout(
            system,
            bundle,
            parent_basis_qn,
            ground_selector=ground_selector,
            sink_union_keys=sink_union_keys,
            ground_indices_union=ground_indices,
            excited_indices_union=excited_indices,
            reference_ground_basis=reference_ground_basis,
            reference_excited_basis=reference_excited_basis,
            use_identity_rotation=(system is patch_systems[master_index] and bundle is patch_bundles[master_index]),
        )
        raw_bases.append(np.asarray(basis_vectors, dtype=np.complex128))
        raw_bundles.append(reordered_bundle)

    aligned_bases: list[np.ndarray | None] = [None] * len(unique_sorted)
    aligned_bundles: list[OperatorBundle | None] = [None] * len(unique_sorted)
    aligned_bases[master_index] = raw_bases[master_index]
    aligned_bundles[master_index] = raw_bundles[master_index]

    for index in range(master_index + 1, len(unique_sorted)):
        previous_basis = np.asarray(aligned_bases[index - 1], dtype=np.complex128)
        current_basis = raw_bases[index]
        aligned_basis, rotation = _union_block_alignment(
            previous_basis,
            current_basis,
            ground_indices=ground_indices,
            sink_indices=sink_indices,
            excited_indices=excited_indices,
        )
        aligned_bases[index] = aligned_basis
        excited_rotation = rotation[np.ix_(excited_indices, excited_indices)]
        aligned_bundles[index] = _transform_operator_bundle(
            raw_bundles[index],
            rotation,
            decay_kernel_ground=_symmetrize(
                excited_rotation
                @ np.asarray(raw_bundles[index].decay_kernel_ground, dtype=np.complex128)
                @ excited_rotation.conj().T
            )
            if raw_bundles[index].decay_kernel_ground is not None
            else None,
            decay_kernels_sinks=tuple(
                _symmetrize(
                    excited_rotation @ np.asarray(kernel, dtype=np.complex128) @ excited_rotation.conj().T
                )
                for kernel in (raw_bundles[index].decay_kernels_sinks or ())
            )
            if raw_bundles[index].decay_kernels_sinks is not None
            else None,
        )

    for index in range(master_index - 1, -1, -1):
        previous_basis = np.asarray(aligned_bases[index + 1], dtype=np.complex128)
        current_basis = raw_bases[index]
        aligned_basis, rotation = _union_block_alignment(
            previous_basis,
            current_basis,
            ground_indices=ground_indices,
            sink_indices=sink_indices,
            excited_indices=excited_indices,
        )
        aligned_bases[index] = aligned_basis
        excited_rotation = rotation[np.ix_(excited_indices, excited_indices)]
        aligned_bundles[index] = _transform_operator_bundle(
            raw_bundles[index],
            rotation,
            decay_kernel_ground=_symmetrize(
                excited_rotation
                @ np.asarray(raw_bundles[index].decay_kernel_ground, dtype=np.complex128)
                @ excited_rotation.conj().T
            )
            if raw_bundles[index].decay_kernel_ground is not None
            else None,
            decay_kernels_sinks=tuple(
                _symmetrize(
                    excited_rotation @ np.asarray(kernel, dtype=np.complex128) @ excited_rotation.conj().T
                )
                for kernel in (raw_bundles[index].decay_kernels_sinks or ())
            )
            if raw_bundles[index].decay_kernels_sinks is not None
            else None,
        )

    coherent_patches: list[PopulationSinkEffectivePatch] = []
    for field_z, aligned_bundle in zip(unique_sorted.tolist(), aligned_bundles):
        coherent_bundle, sink_feed_matrix = _population_sink_bundle_from_patch(
            aligned_bundle,
            ground_indices=ground_indices,
            sink_indices=sink_indices,
            excited_indices=excited_indices,
        )
        coherent_patches.append(
            PopulationSinkEffectivePatch(
                electric_field=np.array([0.0, 0.0, float(field_z)], dtype=np.float64),
                coherent_bundle=coherent_bundle,
                sink_feed_matrix=sink_feed_matrix,
            )
        )

    sink_labels = tuple(
        _population_sink_label_from_union_key(key, index)
        for index, key in enumerate(sink_union_keys)
    )

    return PreparedPopulationSinkEffectiveHamiltonianModel(
        transition=transition,
        optical_polarization=optical_polarization,
        reference_magnetic_field=magnetic,
        field_points=unique_sorted,
        master_field=float(master_value),
        ground_indices=np.asarray(ground_indices, dtype=np.int64),
        excited_indices=np.asarray(excited_indices, dtype=np.int64),
        ground_main_index=int(_optically_bright_ground_index(raw_bundles[master_index], excited_indices)),
        sink_labels=sink_labels,
        patches=tuple(coherent_patches),
    )


def prepare_instantaneous_interpolated_effective_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    keep_diagnostics: bool = True,
) -> PreparedInstantaneousInterpolatedEffectiveHamiltonianModel:
    if len(field_points) < 2:
        raise ValueError(
            "At least two field_points are required to build an instantaneous interpolated model."
        )

    parsed_fields = np.array([_as_field_vector(field)[2] for field in field_points], dtype=np.float64)
    if np.max(np.abs(np.array([_as_field_vector(field)[:2] for field in field_points], dtype=np.float64))) > 1e-12:
        raise ValueError(
            "prepare_instantaneous_interpolated_effective_model currently supports only Ez field points."
        )
    unique_sorted = np.array(sorted(set(float(value) for value in parsed_fields.tolist())), dtype=np.float64)
    if unique_sorted.size < 2:
        raise ValueError("field_points must contain at least two distinct Ez values.")

    master_value = (
        float(unique_sorted[len(unique_sorted) // 2])
        if master_field is None
        else float(_as_field_vector(master_field)[2])
    )
    if master_value not in unique_sorted:
        raise ValueError("master_field must match one of the supplied field_points.")

    magnetic = _as_field_vector(magnetic_field)
    master_electric = np.array([0.0, 0.0, master_value], dtype=np.float64)
    auto_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        [transition],
        E=master_electric,
        B=magnetic,
        use_omega_basis=False,
    )
    b_parent_states = list(auto_reduced.B_hamiltonian.QN_construct)
    b_parent_j_values = np.unique([int(state.J) for state in b_parent_states]).astype(int)
    parent_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        auto_reduced.X_states_basis,
        b_parent_states,
        E=master_electric,
        B=magnetic,
        Jmin_B=int(b_parent_j_values.min()),
        Jmax_B=int(b_parent_j_values.max()),
        use_omega_basis=False,
    )
    parent_basis_qn = list(parent_reduced.QN)
    ground_selector = states.QuantumSelector(
        J=int(transition.J_ground),
        electronic=transition.electronic_ground,
    )

    patch_systems: list[lindblad.utils_setup.OBESystem] = []
    patch_bundles: list[OperatorBundle] = []
    for field_z in unique_sorted.tolist():
        electric = np.array([0.0, 0.0, float(field_z)], dtype=np.float64)
        system, bundle = build_compact_reference_decomposed_bundle(
            transition=transition,
            optical_polarization=optical_polarization,
            electric_field=electric,
            magnetic_field=magnetic,
        )
        patch_systems.append(system)
        patch_bundles.append(bundle)

    master_index = int(np.where(unique_sorted == master_value)[0][0])
    ground_indices, sink_indices, excited_indices, sink_union_keys, _, _ = _build_compact_union_layout(
        patch_systems,
        patch_bundles,
        parent_basis_qn,
        master_index=master_index,
        ground_selector=ground_selector,
    )
    coherent_indices = np.concatenate([ground_indices, excited_indices]).astype(np.int64)

    instantaneous_patches: list[InstantaneousInterpolatedEffectivePatch | None] = [None] * len(unique_sorted)
    tracked_ground_basis_prev: np.ndarray | None = None
    tracked_excited_basis_prev: np.ndarray | None = None

    # Build master patch first, then continue outward with overlap/phase tracking.
    build_order = [master_index] + list(range(master_index + 1, len(unique_sorted))) + list(
        range(master_index - 1, -1, -1)
    )
    references: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in build_order:
        if index == master_index:
            ref_ground = None
            ref_excited = None
        elif index > master_index:
            ref_ground, ref_excited = references[index - 1]
        else:
            ref_ground, ref_excited = references[index + 1]

        basis_vectors, coherent_basis, bundle_embedded = _embed_tracked_compact_patch_to_union_layout(
            patch_systems[index],
            patch_bundles[index],
            parent_basis_qn,
            ground_selector=ground_selector,
            sink_union_keys=sink_union_keys,
            ground_indices_union=ground_indices,
            excited_indices_union=excited_indices,
            reference_ground_basis=ref_ground,
            reference_excited_basis=ref_excited,
        )
        bundle_embedded = _make_bundle_sinks_dissipative_only(bundle_embedded, sink_indices)
        instantaneous_patches[index] = InstantaneousInterpolatedEffectivePatch(
            electric_field=np.array([0.0, 0.0, float(unique_sorted[index])], dtype=np.float64),
            full_basis_vectors=np.asarray(basis_vectors, dtype=np.complex128),
            coherent_basis_vectors=np.asarray(coherent_basis, dtype=np.complex128),
            gauge_connection=np.zeros((ground_indices.size + sink_indices.size + excited_indices.size,) * 2, dtype=np.complex128),
            bundle=bundle_embedded,
        )
        references[index] = (
            np.asarray(coherent_basis[:, : ground_indices.size], dtype=np.complex128),
            np.asarray(coherent_basis[:, ground_indices.size :], dtype=np.complex128),
        )

    built_patches = [patch for patch in instantaneous_patches if patch is not None]
    if len(built_patches) != len(unique_sorted):
        raise RuntimeError("Failed to build all instantaneous patches.")

    gauge_connections: list[np.ndarray] = []
    n_union = int(ground_indices.size + sink_indices.size + excited_indices.size)
    for index in range(len(unique_sorted)):
        if index == 0:
            basis_minus = instantaneous_patches[index].coherent_basis_vectors
            basis_plus = instantaneous_patches[index + 1].coherent_basis_vectors
            delta_field = float(unique_sorted[index + 1] - unique_sorted[index])
        elif index == len(unique_sorted) - 1:
            basis_minus = instantaneous_patches[index - 1].coherent_basis_vectors
            basis_plus = instantaneous_patches[index].coherent_basis_vectors
            delta_field = float(unique_sorted[index] - unique_sorted[index - 1])
        else:
            basis_minus = instantaneous_patches[index - 1].coherent_basis_vectors
            basis_plus = instantaneous_patches[index + 1].coherent_basis_vectors
            delta_field = float(unique_sorted[index + 1] - unique_sorted[index - 1])
        connection_coherent = _coherent_gauge_connection(
            np.asarray(basis_minus, dtype=np.complex128),
            np.asarray(basis_plus, dtype=np.complex128),
            delta_field,
        )
        connection = np.zeros((n_union, n_union), dtype=np.complex128)
        connection[np.ix_(coherent_indices, coherent_indices)] = connection_coherent
        gauge_connections.append(_symmetrize(connection))

    patches = tuple(
        InstantaneousInterpolatedEffectivePatch(
            electric_field=np.asarray(patch.electric_field, dtype=np.float64),
            full_basis_vectors=np.asarray(patch.full_basis_vectors, dtype=np.complex128),
            coherent_basis_vectors=np.asarray(patch.coherent_basis_vectors, dtype=np.complex128),
            gauge_connection=np.asarray(gauge_connection, dtype=np.complex128),
            bundle=patch.bundle,
        )
        for patch, gauge_connection in zip(instantaneous_patches, gauge_connections)
    )

    return PreparedInstantaneousInterpolatedEffectiveHamiltonianModel(
        transition=transition,
        optical_polarization=optical_polarization,
        reference_magnetic_field=magnetic,
        parent_basis_qn=parent_basis_qn,
        union_state_keys=tuple(sink_union_keys),
        field_points=unique_sorted,
        master_field=float(master_value),
        ground_indices=np.asarray(ground_indices, dtype=np.int64),
        sink_indices=np.asarray(sink_indices, dtype=np.int64),
        excited_indices=np.asarray(excited_indices, dtype=np.int64),
        coherent_indices=np.asarray(coherent_indices, dtype=np.int64),
        ground_main_index=int(_optically_bright_ground_index(patches[master_index].bundle, excited_indices)),
        patches=patches,
        keep_diagnostics=bool(keep_diagnostics),
    )


def prepare_patchwise_instantaneous_effective_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    keep_diagnostics: bool = True,
) -> PreparedPatchwiseInstantaneousEffectiveHamiltonianModel:
    base_model = prepare_instantaneous_interpolated_effective_model(
        field_points=field_points,
        transition=transition,
        optical_polarization=optical_polarization,
        magnetic_field=magnetic_field,
        master_field=master_field,
        keep_diagnostics=keep_diagnostics,
    )
    fields = np.asarray(base_model.field_points, dtype=np.float64)
    boundaries = 0.5 * (fields[:-1] + fields[1:])
    coherent_indices = np.asarray(base_model.coherent_indices, dtype=np.int64)
    sink_indices = np.asarray(base_model.sink_indices, dtype=np.int64)
    transforms: list[np.ndarray] = []
    n_union = int(base_model.n_effective_states)
    for idx in range(len(base_model.patches) - 1):
        basis_old = np.asarray(base_model.patches[idx].coherent_basis_vectors, dtype=np.complex128)
        basis_new = np.asarray(base_model.patches[idx + 1].coherent_basis_vectors, dtype=np.complex128)
        coherent_overlap = np.asarray(basis_new.conj().T @ basis_old, dtype=np.complex128)
        coherent_transform = _polar_unitary(coherent_overlap)
        transform = np.eye(n_union, dtype=np.complex128)
        transform[np.ix_(coherent_indices, coherent_indices)] = coherent_transform
        transform[np.ix_(sink_indices, sink_indices)] = np.eye(len(sink_indices), dtype=np.complex128)
        transforms.append(np.asarray(transform, dtype=np.complex128))
    return PreparedPatchwiseInstantaneousEffectiveHamiltonianModel(
        base_model=base_model,
        interval_boundaries=np.asarray(boundaries, dtype=np.float64),
        forward_transforms=tuple(transforms),
    )


def default_full_density_matrix(
    model: PreparedEffectiveHamiltonianModel,
) -> np.ndarray:
    density = np.zeros((len(model.qn_full), len(model.qn_full)), dtype=np.complex128)
    density[model.ground_main_index_full, model.ground_main_index_full] = 1.0
    return density


def default_effective_density_matrix(
    model: PreparedEffectiveHamiltonianModel,
) -> np.ndarray:
    density = np.zeros((model.n_effective_states, model.n_effective_states), dtype=np.complex128)
    density[model.ground_main_index_p, model.ground_main_index_p] = 1.0
    return density


def project_full_density_to_p(
    model: PreparedEffectiveHamiltonianModel,
    rho_full: np.ndarray,
) -> np.ndarray:
    return rho_full[np.ix_(model.p_indices, model.p_indices)]


def project_full_density_to_effective(
    model: PreparedEffectiveHamiltonianModel,
    rho_full: np.ndarray,
) -> np.ndarray:
    rho_effective = np.zeros(
        (model.n_effective_states, model.n_effective_states),
        dtype=np.complex128,
    )
    n_ground = int(model.p_ground_indices.size)
    active_positions = np.concatenate(
        [
            np.arange(n_ground, dtype=np.int64),
            np.arange(n_ground + model.n_sink_states, model.n_effective_states, dtype=np.int64),
        ]
    )
    rho_effective[np.ix_(active_positions, active_positions)] = rho_full[
        np.ix_(model.p_indices, model.p_indices)
    ]
    for sink_offset, sink_indices in enumerate(model.sink_indices_full):
        sink_index = n_ground + sink_offset
        rho_effective[sink_index, sink_index] = np.trace(
            rho_full[np.ix_(sink_indices, sink_indices)]
        )
    return rho_effective


def solve_density_matrix_model(
    bundle_evaluator: Callable[[float], OperatorBundle],
    *,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    rabi_rate: float | complex | Callable[[float], float | complex],
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    rho0 = np.asarray(rho0, dtype=np.complex128)
    n_states = rho0.shape[0]
    if rho0.shape != (n_states, n_states):
        raise ValueError("rho0 must be a square density matrix")

    def rhs(t: float, rho_flat: np.ndarray) -> np.ndarray:
        bundle = bundle_evaluator(float(t))
        rho = rho_flat.reshape((n_states, n_states))
        h_total = bundle.total_hamiltonian(
            rabi_rate=_parameter_at_time(rabi_rate, t),
            detuning=float(_parameter_at_time(detuning, t)),
        )
        drho = -1j * (h_total @ rho - rho @ h_total) + bundle.dissipator(rho)
        return drho.reshape(-1)

    return solve_ivp(
        rhs,
        t_span=t_span,
        y0=rho0.reshape(-1),
        t_eval=t_eval,
        method=method,
    )


def solve_static_density_matrix_bundle(
    bundle: OperatorBundle,
    *,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    rabi_rate: float | complex = 0.0,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    rho0 = np.asarray(rho0, dtype=np.complex128)
    liouvillian = bundle.liouvillian_superoperator(
        rabi_rate=rabi_rate,
        detuning=detuning,
    )

    def rhs(_: float, rho_flat: np.ndarray) -> np.ndarray:
        return liouvillian @ rho_flat

    return solve_ivp(
        rhs,
        t_span=t_span,
        y0=rho0.reshape(-1),
        t_eval=t_eval,
        method=method,
    )


def solve_exact_full_model(
    model: PreparedEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray] | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex | Callable[[float], float | complex] = 2.0 * np.pi * 1e6,
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_full_density_matrix(model)

    def bundle_evaluator(t: float) -> OperatorBundle:
        e_val = _parameter_at_time(electric_field, t)
        if magnetic_field is None:
            b_val: float | Sequence[float] | np.ndarray = model.reference_magnetic_field
        else:
            b_val = _parameter_at_time(magnetic_field, t)
        return model.exact_bundle(e_val, b_val)

    return solve_density_matrix_model(
        bundle_evaluator,
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def solve_static_exact_full_model(
    model: PreparedEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_full_density_matrix(model)
    return solve_static_density_matrix_bundle(
        model.exact_bundle(electric_field, magnetic_field),
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def solve_effective_model(
    model: PreparedEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray] | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex | Callable[[float], float | complex] = 2.0 * np.pi * 1e6,
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)

    def bundle_evaluator(t: float) -> OperatorBundle:
        e_val = _parameter_at_time(electric_field, t)
        if magnetic_field is None:
            b_val: float | Sequence[float] | np.ndarray = model.reference_magnetic_field
        else:
            b_val = _parameter_at_time(magnetic_field, t)
        return model.effective_bundle(e_val, b_val)

    return solve_density_matrix_model(
        bundle_evaluator,
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def solve_static_effective_model(
    model: PreparedEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)
    return solve_static_density_matrix_bundle(
        model.effective_bundle(electric_field, magnetic_field),
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def solve_lindblad_safe_compact_interpolated_model(
    model: PreparedLindbladSafeCompactInterpolatedHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray] | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex | Callable[[float], float | complex] = 2.0 * np.pi * 1e6,
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)
    rho0 = np.asarray(rho0, dtype=np.complex128)
    n_states = rho0.shape[0]
    if rho0.shape != (n_states, n_states):
        raise ValueError("rho0 must be a square density matrix")

    fields = np.asarray(model.field_points, dtype=np.float64)
    endpoint_bundles = tuple(
        model.effective_bundle((0.0, 0.0, float(field_z)), model.reference_magnetic_field)
        for field_z in fields.tolist()
    )
    h_internal_superops = tuple(
        _hamiltonian_superoperator(np.asarray(bundle.h_internal, dtype=np.complex128))
        for bundle in endpoint_bundles
    )
    h_opt_superops = tuple(
        _hamiltonian_superoperator(np.asarray(bundle.h_opt, dtype=np.complex128))
        for bundle in endpoint_bundles
    )
    h_det_superops = tuple(
        _hamiltonian_superoperator(np.asarray(bundle.h_det, dtype=np.complex128))
        for bundle in endpoint_bundles
    )
    dissipator_superops = tuple(
        np.asarray(bundle.dissipator_superoperator(), dtype=np.complex128)
        for bundle in endpoint_bundles
    )

    def _interval_differences(
        values: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        if len(values) <= 1:
            return tuple()
        return tuple(
            np.asarray(values[idx + 1] - values[idx], dtype=np.complex128)
            for idx in range(len(values) - 1)
        )

    dh_internal_superops = _interval_differences(h_internal_superops)
    dh_opt_superops = _interval_differences(h_opt_superops)
    dh_det_superops = _interval_differences(h_det_superops)
    ddissipator_superops = _interval_differences(dissipator_superops)

    def rhs(t: float, rho_flat: np.ndarray) -> np.ndarray:
        electric = _as_field_vector(_parameter_at_time(electric_field, float(t)))
        if np.max(np.abs(electric[:2])) > 1e-12:
            raise ValueError(
                "PreparedLindbladSafeCompactInterpolatedHamiltonianModel currently supports "
                "interpolation only along Ez."
            )
        if magnetic_field is None:
            magnetic = np.asarray(model.reference_magnetic_field, dtype=np.float64)
        else:
            magnetic = _as_field_vector(_parameter_at_time(magnetic_field, float(t)))
        if np.max(np.abs(magnetic - model.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedLindbladSafeCompactInterpolatedHamiltonianModel currently supports only "
                "the reference magnetic field."
            )

        lower, upper, weight = model._interpolation_indices(float(electric[2]))
        if lower == upper:
            h_int_super = h_internal_superops[lower]
            h_opt_super = h_opt_superops[lower]
            h_det_super = h_det_superops[lower]
            dissipator_super = dissipator_superops[lower]
        else:
            h_int_super = h_internal_superops[lower] + weight * dh_internal_superops[lower]
            h_opt_super = h_opt_superops[lower] + weight * dh_opt_superops[lower]
            h_det_super = h_det_superops[lower] + weight * dh_det_superops[lower]
            dissipator_super = dissipator_superops[lower] + weight * ddissipator_superops[lower]

        liouvillian = (
            h_int_super
            + 0.5 * complex(_parameter_at_time(rabi_rate, float(t))) * h_opt_super
            + float(_parameter_at_time(detuning, float(t))) * h_det_super
            + dissipator_super
        )
        return np.asarray(liouvillian @ rho_flat, dtype=np.complex128)

    return solve_ivp(
        rhs,
        t_span=t_span,
        y0=rho0.reshape(-1),
        t_eval=t_eval,
        method=method,
    )


def solve_static_lindblad_safe_compact_interpolated_model(
    model: PreparedLindbladSafeCompactInterpolatedHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)
    return solve_static_density_matrix_bundle(
        model.effective_bundle(electric_field, magnetic_field),
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def solve_instantaneous_interpolated_model(
    model: PreparedInstantaneousInterpolatedEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    electric_field_derivative: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray] | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex | Callable[[float], float | complex] = 2.0 * np.pi * 1e6,
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)

    def bundle_evaluator(t: float) -> OperatorBundle:
        e_val = _parameter_at_time(electric_field, t)
        e_dot_val = _parameter_at_time(electric_field_derivative, t)
        if magnetic_field is None:
            b_val: float | Sequence[float] | np.ndarray = model.reference_magnetic_field
        else:
            b_val = _parameter_at_time(magnetic_field, t)
        return model.effective_bundle(e_val, b_val, electric_field_derivative=e_dot_val)

    return solve_density_matrix_model(
        bundle_evaluator,
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def solve_static_instantaneous_interpolated_model(
    model: PreparedInstantaneousInterpolatedEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)
    return solve_static_density_matrix_bundle(
        model.effective_bundle(electric_field, magnetic_field, electric_field_derivative=0.0),
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def default_population_sink_state(
    model: PreparedPopulationSinkEffectiveHamiltonianModel,
) -> np.ndarray:
    rho = np.zeros((model.n_coherent_states, model.n_coherent_states), dtype=np.complex128)
    rho[model.ground_main_index_p, model.ground_main_index_p] = 1.0
    state = np.zeros(model.n_coherent_states * model.n_coherent_states + model.n_sink_populations, dtype=np.complex128)
    state[: model.n_coherent_states * model.n_coherent_states] = rho.reshape(-1)
    return state


def population_sink_density_trajectory(
    model: PreparedPopulationSinkEffectiveHamiltonianModel,
    solution: solve_ivp,
) -> np.ndarray:
    n_coherent = int(model.n_coherent_states)
    rho_flat = np.asarray(solution.y[: n_coherent * n_coherent, :], dtype=np.complex128).T
    return rho_flat.reshape((-1, n_coherent, n_coherent))


def population_sink_populations_trajectory(
    model: PreparedPopulationSinkEffectiveHamiltonianModel,
    solution: solve_ivp,
) -> np.ndarray:
    n_coherent = int(model.n_coherent_states)
    return np.asarray(solution.y[n_coherent * n_coherent :, :], dtype=np.complex128).T


def population_sink_scattering_signal(
    model: PreparedPopulationSinkEffectiveHamiltonianModel,
    solution: solve_ivp,
    *,
    electric_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray] | None = None,
) -> np.ndarray:
    rho_t = population_sink_density_trajectory(model, solution)
    signals = np.zeros(rho_t.shape[0], dtype=np.float64)
    for index, t_value in enumerate(np.asarray(solution.t, dtype=np.float64)):
        e_val = _parameter_at_time(electric_field, float(t_value))
        if magnetic_field is None:
            b_val: float | Sequence[float] | np.ndarray = model.reference_magnetic_field
        else:
            b_val = _parameter_at_time(magnetic_field, float(t_value))
        bundle = model.coherent_bundle(e_val, b_val)
        signals[index] = float(np.real(np.trace(rho_t[index] @ bundle.jump_rate_operator())))
    return signals


def solve_population_sink_model(
    model: PreparedPopulationSinkEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray] | None = None,
    y0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex | Callable[[float], float | complex] = 2.0 * np.pi * 1e6,
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if y0 is None:
        y0 = default_population_sink_state(model)
    y0 = np.asarray(y0, dtype=np.complex128)
    n_coherent = int(model.n_coherent_states)
    n_sink = int(model.n_sink_populations)
    expected = n_coherent * n_coherent + n_sink
    if y0.shape != (expected,):
        raise ValueError(f"y0 must have shape ({expected},), not {y0.shape}.")

    coherent_excited_indices = np.asarray(model.coherent_excited_indices, dtype=np.int64)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        e_val = _parameter_at_time(electric_field, t)
        if magnetic_field is None:
            b_val: float | Sequence[float] | np.ndarray = model.reference_magnetic_field
        else:
            b_val = _parameter_at_time(magnetic_field, t)
        bundle = model.coherent_bundle(e_val, b_val)
        rho = np.asarray(y[: n_coherent * n_coherent], dtype=np.complex128).reshape((n_coherent, n_coherent))
        h_total = bundle.total_hamiltonian(
            rabi_rate=_parameter_at_time(rabi_rate, t),
            detuning=float(_parameter_at_time(detuning, t)),
        )
        drho = -1j * (h_total @ rho - rho @ h_total) + bundle.dissipator(rho)
        sink_feed = model.sink_feed_matrix(e_val, b_val)
        sink_rates = np.real(np.asarray(sink_feed @ rho.reshape(-1), dtype=np.complex128))
        if sink_rates.size != n_sink:
            raise ValueError(
                f"Interpolated sink-rate count {sink_rates.size} does not match model sink count {n_sink}."
            )
        return np.concatenate([drho.reshape(-1), sink_rates.astype(np.complex128)])

    return solve_ivp(
        rhs,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
    )


def solve_static_population_sink_model(
    model: PreparedPopulationSinkEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    y0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    return solve_population_sink_model(
        model,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        y0=y0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def _bisect_root(
    func: Callable[[float], float],
    left: float,
    right: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> float:
    f_left = float(func(left))
    f_right = float(func(right))
    if abs(f_left) <= tol:
        return float(left)
    if abs(f_right) <= tol:
        return float(right)
    if f_left * f_right > 0.0:
        raise ValueError("Bisection interval does not bracket a root.")
    a = float(left)
    b = float(right)
    fa = f_left
    fb = f_right
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = float(func(mid))
        if abs(fm) <= tol or abs(b - a) <= tol:
            return float(mid)
        if fa * fm <= 0.0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return float(0.5 * (a + b))


def _find_patch_switch_times(
    electric_field: Callable[[float], float | Sequence[float] | np.ndarray],
    t_span: tuple[float, float],
    interval_boundaries: np.ndarray,
    *,
    n_probe: int = 4000,
) -> np.ndarray:
    boundaries = np.asarray(interval_boundaries, dtype=np.float64)
    if boundaries.size == 0:
        return np.array([], dtype=np.float64)
    t0, t1 = map(float, t_span)
    probe_times = np.linspace(t0, t1, int(max(8, n_probe)))
    e_probe = np.array([_as_field_vector(electric_field(t))[2] for t in probe_times], dtype=np.float64)
    crossings: list[float] = []
    for boundary in boundaries.tolist():
        values = e_probe - float(boundary)
        for i in range(len(probe_times) - 1):
            a_t = float(probe_times[i])
            b_t = float(probe_times[i + 1])
            a_v = float(values[i])
            b_v = float(values[i + 1])
            if abs(a_v) <= 1e-12:
                crossings.append(a_t)
                continue
            if a_v * b_v < 0.0:
                root = _bisect_root(
                    lambda t, boundary=boundary: float(_as_field_vector(electric_field(t))[2] - boundary),
                    a_t,
                    b_t,
                )
                crossings.append(root)
    if not crossings:
        return np.array([], dtype=np.float64)
    crossings = sorted(float(value) for value in crossings if t0 < value < t1)
    unique: list[float] = []
    for value in crossings:
        if not unique or abs(value - unique[-1]) > 1e-9:
            unique.append(value)
    return np.asarray(unique, dtype=np.float64)


def solve_patchwise_instantaneous_model(
    model: PreparedPatchwiseInstantaneousEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray | Callable[[float], float | Sequence[float] | np.ndarray] | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex | Callable[[float], float | complex] = 2.0 * np.pi * 1e6,
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    switch_probe_points: int = 4000,
) -> PatchwiseInstantaneousSolveResult:
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)
    rho_current = np.asarray(rho0, dtype=np.complex128)

    if callable(electric_field):
        electric_field_fn = electric_field
    else:
        constant_field = _as_field_vector(electric_field)
        electric_field_fn = lambda _t, value=constant_field: value

    if magnetic_field is not None:
        if callable(magnetic_field):
            magnetic_probe = _as_field_vector(magnetic_field(float(t_span[0])))
        else:
            magnetic_probe = _as_field_vector(magnetic_field)
        if np.max(np.abs(magnetic_probe - model.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedPatchwiseInstantaneousEffectiveHamiltonianModel currently supports only the "
                "reference magnetic field."
            )

    t0, t1 = map(float, t_span)
    switch_times = _find_patch_switch_times(
        electric_field_fn,
        (t0, t1),
        model.interval_boundaries,
        n_probe=switch_probe_points,
    )
    segment_edges = np.concatenate([[t0], switch_times, [t1]]).astype(np.float64)

    t_segments: list[np.ndarray] = []
    y_segments: list[np.ndarray] = []
    patch_trace: list[int] = []
    current_patch_index: int | None = None
    for start, stop in zip(segment_edges[:-1], segment_edges[1:]):
        midpoint = 0.5 * (float(start) + float(stop))
        patch_index = int(model.patch_index_for_field(electric_field_fn(midpoint)))
        patch_bundle = model.patch_bundle(patch_index)
        if current_patch_index is not None and patch_index != current_patch_index:
            transform = model.transform_between_patches(current_patch_index, patch_index)
            rho_current = np.asarray(transform @ rho_current @ transform.conj().T, dtype=np.complex128)
        current_patch_index = patch_index

        segment_eval = None
        if t_eval is not None:
            t_eval_array = np.asarray(t_eval, dtype=np.float64)
            mask = (t_eval_array >= float(start) - 1e-15) & (t_eval_array <= float(stop) + 1e-15)
            segment_eval = t_eval_array[mask]
            if segment_eval.size == 0:
                segment_eval = np.array([float(stop)], dtype=np.float64)
        solution = solve_density_matrix_model(
            lambda _t, bundle=patch_bundle: bundle,
            rho0=rho_current,
            t_span=(float(start), float(stop)),
            rabi_rate=rabi_rate,
            detuning=detuning,
            t_eval=segment_eval,
            method=method,
        )
        rho_current = solution.y[:, -1].reshape((model.n_effective_states, model.n_effective_states))
        if t_segments:
            duplicate_start = (
                solution.t.size > 0
                and t_segments[-1].size > 0
                and abs(float(solution.t[0]) - float(t_segments[-1][-1])) <= 1e-12
            )
            start_idx = 1 if duplicate_start else 0
            t_segments.append(solution.t[start_idx:])
            y_segments.append(solution.y[:, start_idx:])
            patch_trace.extend([patch_index] * max(solution.t.size - start_idx, 0))
        else:
            t_segments.append(solution.t)
            y_segments.append(solution.y)
            patch_trace.extend([patch_index] * solution.t.size)

    t_out = np.concatenate(t_segments) if t_segments else np.array([t0], dtype=np.float64)
    y_out = np.concatenate(y_segments, axis=1) if y_segments else rho_current.reshape(-1, 1)
    if patch_trace:
        patch_indices = np.asarray(patch_trace[: t_out.size], dtype=np.int64)
    else:
        patch_indices = np.array([], dtype=np.int64)
    return PatchwiseInstantaneousSolveResult(
        t=t_out,
        y=y_out,
        patch_indices=patch_indices,
        switch_times=np.asarray(switch_times, dtype=np.float64),
        success=True,
        message="",
    )


def solve_static_patchwise_instantaneous_model(
    model: PreparedPatchwiseInstantaneousEffectiveHamiltonianModel,
    *,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> solve_ivp:
    if magnetic_field is not None:
        magnetic = _as_field_vector(magnetic_field)
        if np.max(np.abs(magnetic - model.reference_magnetic_field)) > 1e-12:
            raise ValueError(
                "PreparedPatchwiseInstantaneousEffectiveHamiltonianModel currently supports only the "
                "reference magnetic field."
            )
    if rho0 is None:
        rho0 = default_effective_density_matrix(model)
    patch_index = model.patch_index_for_field(electric_field)
    return solve_static_density_matrix_bundle(
        model.patch_bundle(patch_index),
        rho0=rho0,
        t_span=t_span,
        rabi_rate=rabi_rate,
        detuning=detuning,
        t_eval=t_eval,
        method=method,
    )


def solution_to_density_matrices(solution: solve_ivp, n_states: int) -> np.ndarray:
    return solution.y.T.reshape((-1, n_states, n_states))


def scattering_signal(
    rho_t: np.ndarray,
    jump_rate_operator: np.ndarray,
) -> np.ndarray:
    return np.real(np.einsum("tij,ji->t", rho_t, jump_rate_operator))


def integrated_scattering_probability(
    t_eval: np.ndarray,
    rho_t: np.ndarray,
    jump_rate_operator: np.ndarray,
) -> float:
    return float(np.trapezoid(scattering_signal(rho_t, jump_rate_operator), x=t_eval))


def compact_reference_diagnostics(
    model: PreparedEffectiveHamiltonianModel,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    *,
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    polarization_scale: float = 1.0,
) -> dict:
    magnetic = (
        np.asarray(model.reference_magnetic_field, dtype=np.float64)
        if magnetic_field is None
        else _as_field_vector(magnetic_field)
    )
    system, compact = build_compact_reference_bundle(
        transition=model.transition,
        optical_polarization=model.optical_polarization,
        electric_field=electric_field,
        magnetic_field=magnetic,
        rabi_rate=rabi_rate,
        detuning=detuning,
        polarization_scale=polarization_scale,
    )
    effective = model.effective_bundle(electric_field, magnetic)

    effective_ground = np.arange(model.p_ground_indices.size, dtype=np.int64)
    effective_excited = np.asarray(model.excited_positions, dtype=np.int64)
    effective_sinks = np.asarray(model.sink_positions, dtype=np.int64)

    compact_ground = _selector_indices(
        system.QN,
        states.QuantumSelector(J=0, electronic=states.ElectronicState.X),
    )
    compact_excited = _selector_indices(
        system.QN,
        states.QuantumSelector(electronic=states.ElectronicState.B),
    )
    compact_sink_groups = tuple(
        _selector_indices(
            system.QN,
            states.QuantumSelector(J=J, electronic=states.ElectronicState.X),
        )
        for J in (1, 2, 3)
    )
    compact_sinks = np.concatenate(compact_sink_groups).astype(np.int64)

    effective_h_ge = effective.h_opt[np.ix_(effective_ground, effective_excited)]
    compact_h_ge = compact.h_internal[np.ix_(compact_ground, compact_excited)]
    effective_singular_values, effective_dark_ground = _smallest_left_singular_vector(
        effective_h_ge
    )
    compact_singular_values, compact_dark_ground = _smallest_left_singular_vector(
        compact_h_ge
    )
    dark_overlap = abs(np.vdot(compact_dark_ground, effective_dark_ground))
    effective_dark_dimension = _singular_value_dark_dimension(
        effective_singular_values,
        int(effective_ground.size),
    )
    compact_dark_dimension = _singular_value_dark_dimension(
        compact_singular_values,
        int(compact_ground.size),
    )

    effective_kernel_ground = (
        np.asarray(effective.decay_kernel_ground, dtype=np.complex128)
        if effective.decay_kernel_ground is not None
        else _sector_decay_kernel(effective.c_array, effective_excited, effective_ground)
    )
    effective_kernel_sinks = (
        tuple(np.asarray(kernel, dtype=np.complex128) for kernel in effective.decay_kernels_sinks)
        if effective.decay_kernels_sinks is not None
        else tuple(
            _sector_decay_kernel(
                effective.c_array,
                effective_excited,
                np.array([int(pos)], dtype=np.int64),
            )
            for pos in effective_sinks
        )
    )
    compact_kernel_ground = _sector_decay_kernel(
        compact.c_array,
        compact_excited,
        compact_ground,
    )
    compact_kernel_sinks = tuple(
        _sector_decay_kernel(compact.c_array, compact_excited, sink_group)
        for sink_group in compact_sink_groups
    )

    if (
        effective.excited_to_ground_rates_hz is not None
        and effective.excited_to_sink_rates_hz is not None
    ):
        effective_branch_ground = np.asarray(
            effective.excited_to_ground_rates_hz,
            dtype=np.float64,
        )
        effective_branch_sinks = np.asarray(
            effective.excited_to_sink_rates_hz,
            dtype=np.float64,
        )
    else:
        effective_branch_ground = _excited_to_sector_rates(
            effective.c_array, effective_excited, effective_ground
        ) / (2.0 * np.pi)
        effective_branch_sinks = _excited_to_sector_rates(
            effective.c_array, effective_excited, effective_sinks
        ) / (2.0 * np.pi)
    compact_branch_ground = _excited_to_sector_rates(
        compact.c_array, compact_excited, compact_ground
    ) / (2.0 * np.pi)
    compact_branch_sinks = _excited_to_sector_rates(
        compact.c_array, compact_excited, compact_sinks
    ) / (2.0 * np.pi)

    rho_dark_effective = _density_from_state_vector(
        effective_dark_ground,
        effective_ground,
        effective.h_internal.shape[0],
    )
    rho_dark_compact = _density_from_state_vector(
        compact_dark_ground,
        compact_ground,
        compact.h_internal.shape[0],
    )
    effective_h_action = -1j * (
        effective.total_hamiltonian(rabi_rate=rabi_rate, detuning=detuning) @ rho_dark_effective
        - rho_dark_effective @ effective.total_hamiltonian(rabi_rate=rabi_rate, detuning=detuning)
    )
    effective_d_action = effective.dissipator(rho_dark_effective)
    compact_h_action = -1j * (
        compact.total_hamiltonian(rabi_rate=rabi_rate, detuning=detuning) @ rho_dark_compact
        - rho_dark_compact @ compact.total_hamiltonian(rabi_rate=rabi_rate, detuning=detuning)
    )
    compact_d_action = compact.dissipator(rho_dark_compact)

    effective_ground_block = effective.h_internal[np.ix_(effective_ground, effective_ground)]
    compact_ground_block = compact.h_internal[np.ix_(compact_ground, compact_ground)]

    return {
        "effective_optical_singular_values": effective_singular_values,
        "compact_optical_singular_values": compact_singular_values,
        "effective_dark_dimension": effective_dark_dimension,
        "compact_dark_dimension": compact_dark_dimension,
        "effective_dark_ground_abs": np.abs(effective_dark_ground),
        "compact_dark_ground_abs": np.abs(compact_dark_ground),
        "dark_ground_overlap_abs": float(dark_overlap),
        "effective_excited_to_ground_Hz": effective_branch_ground,
        "effective_excited_to_sinks_Hz": effective_branch_sinks,
        "compact_excited_to_ground_Hz": compact_branch_ground,
        "compact_excited_to_sinks_Hz": compact_branch_sinks,
        "effective_decay_kernel_ground_Hz": effective_kernel_ground / (2.0 * np.pi),
        "compact_decay_kernel_ground_Hz": compact_kernel_ground / (2.0 * np.pi),
        "effective_decay_kernel_sinks_Hz": tuple(
            kernel / (2.0 * np.pi) for kernel in effective_kernel_sinks
        ),
        "compact_decay_kernel_sinks_Hz": tuple(
            kernel / (2.0 * np.pi) for kernel in compact_kernel_sinks
        ),
        "effective_excited_to_sink_sectors_Hz": tuple(
            np.real(np.diag(kernel)) / (2.0 * np.pi) for kernel in effective_kernel_sinks
        ),
        "compact_excited_to_sink_sectors_Hz": tuple(
            np.real(np.diag(kernel)) / (2.0 * np.pi) for kernel in compact_kernel_sinks
        ),
        "effective_dark_hamiltonian_action_norm": float(np.linalg.norm(effective_h_action)),
        "effective_dark_dissipative_action_norm": float(np.linalg.norm(effective_d_action)),
        "effective_dark_total_action_norm": float(
            np.linalg.norm(effective_h_action + effective_d_action)
        ),
        "compact_dark_hamiltonian_action_norm": float(np.linalg.norm(compact_h_action)),
        "compact_dark_dissipative_action_norm": float(np.linalg.norm(compact_d_action)),
        "compact_dark_total_action_norm": float(
            np.linalg.norm(compact_h_action + compact_d_action)
        ),
        "effective_dark_hamiltonian_excited_support_norm": _matrix_touching_positions_norm(
            effective_h_action,
            effective_excited,
        ),
        "effective_dark_dissipative_excited_support_norm": _matrix_touching_positions_norm(
            effective_d_action,
            effective_excited,
        ),
        "compact_dark_hamiltonian_excited_support_norm": _matrix_touching_positions_norm(
            compact_h_action,
            compact_excited,
        ),
        "compact_dark_dissipative_excited_support_norm": _matrix_touching_positions_norm(
            compact_d_action,
            compact_excited,
        ),
        "effective_dark_to_bright_mixing_Hz": _dark_to_bright_mixing_scale(
            effective_ground_block,
            effective_dark_ground,
        )
        / (2.0 * np.pi),
        "compact_dark_to_bright_mixing_Hz": _dark_to_bright_mixing_scale(
            compact_ground_block,
            compact_dark_ground,
        )
        / (2.0 * np.pi),
    }


def summarize_static_comparison(
    model: PreparedEffectiveHamiltonianModel,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
) -> dict:
    exact = model.exact_bundle(electric_field, magnetic_field)
    effective = model.effective_bundle(electric_field, magnetic_field)
    compact_diag = compact_reference_diagnostics(
        model,
        electric_field,
        magnetic_field,
    )

    exact_eigs, _, p_weights = _exact_full_eigensubspace(
        exact.h_internal,
        model.p_indices,
        model.p_indices.size,
    )
    n_ground = int(model.p_ground_indices.size)
    active_positions = np.concatenate(
        [
            np.arange(n_ground, dtype=np.int64),
            np.arange(n_ground + model.n_sink_states, model.n_effective_states, dtype=np.int64),
        ]
    )
    eff_eigs = np.linalg.eigvalsh(
        effective.h_internal[np.ix_(active_positions, active_positions)]
    )
    excited_main_index_effective = (
        n_ground + model.n_sink_states + (model.excited_main_index_p - n_ground)
    )

    return {
        "n_full_states": len(model.qn_full),
        "n_active_states": int(model.n_active_states),
        "n_effective_states": int(model.n_effective_states),
        "p_state_labels": [_state_label(state) for state in model.p_states],
        "sink_labels": list(model.sink_labels),
        "full_reference_frequency_MHz": exact.omega_reference / (2.0 * np.pi * 1e6),
        "perturbative_ratio_max": effective.perturbative_ratio_max,
        "effective_hermiticity_error": effective.hermiticity_error,
        "sylvester_residual_norm": effective.sylvester_residual_norm,
        "spectral_separation_min_MHz": (
            None
            if effective.spectral_separation_min is None
            else effective.spectral_separation_min / (2.0 * np.pi * 1e6)
        ),
        "exact_p_like_eigenvalues_MHz": np.real(exact_eigs) / (2.0 * np.pi * 1e6),
        "exact_p_like_weights": p_weights,
        "effective_eigenvalues_MHz": np.real(eff_eigs) / (2.0 * np.pi * 1e6),
        "main_optical_element_exact": complex(model.h_opt_full[model.ground_main_index_full, model.excited_main_index_full]),
        "main_optical_element_effective": complex(
            effective.h_opt[model.ground_main_index_p, excited_main_index_effective]
        ),
        "effective_loss_rates_Hz": np.real(np.diag(effective.loss_operator)) / (2.0 * np.pi),
        "effective_optical_singular_values": compact_diag["effective_optical_singular_values"],
        "compact_optical_singular_values": compact_diag["compact_optical_singular_values"],
        "effective_dark_dimension": compact_diag["effective_dark_dimension"],
        "compact_dark_dimension": compact_diag["compact_dark_dimension"],
        "effective_dark_ground_abs": compact_diag["effective_dark_ground_abs"],
        "compact_dark_ground_abs": compact_diag["compact_dark_ground_abs"],
        "dark_ground_overlap_abs": compact_diag["dark_ground_overlap_abs"],
        "effective_excited_to_ground_Hz": compact_diag["effective_excited_to_ground_Hz"],
        "effective_excited_to_sinks_Hz": compact_diag["effective_excited_to_sinks_Hz"],
        "compact_excited_to_ground_Hz": compact_diag["compact_excited_to_ground_Hz"],
        "compact_excited_to_sinks_Hz": compact_diag["compact_excited_to_sinks_Hz"],
        "effective_decay_kernel_ground_Hz": compact_diag["effective_decay_kernel_ground_Hz"],
        "compact_decay_kernel_ground_Hz": compact_diag["compact_decay_kernel_ground_Hz"],
        "effective_decay_kernel_sinks_Hz": compact_diag["effective_decay_kernel_sinks_Hz"],
        "compact_decay_kernel_sinks_Hz": compact_diag["compact_decay_kernel_sinks_Hz"],
        "effective_excited_to_sink_sectors_Hz": compact_diag["effective_excited_to_sink_sectors_Hz"],
        "compact_excited_to_sink_sectors_Hz": compact_diag["compact_excited_to_sink_sectors_Hz"],
        "effective_dark_hamiltonian_action_norm": compact_diag["effective_dark_hamiltonian_action_norm"],
        "effective_dark_dissipative_action_norm": compact_diag["effective_dark_dissipative_action_norm"],
        "effective_dark_total_action_norm": compact_diag["effective_dark_total_action_norm"],
        "compact_dark_hamiltonian_action_norm": compact_diag["compact_dark_hamiltonian_action_norm"],
        "compact_dark_dissipative_action_norm": compact_diag["compact_dark_dissipative_action_norm"],
        "compact_dark_total_action_norm": compact_diag["compact_dark_total_action_norm"],
        "effective_dark_hamiltonian_excited_support_norm": compact_diag[
            "effective_dark_hamiltonian_excited_support_norm"
        ],
        "effective_dark_dissipative_excited_support_norm": compact_diag[
            "effective_dark_dissipative_excited_support_norm"
        ],
        "compact_dark_hamiltonian_excited_support_norm": compact_diag[
            "compact_dark_hamiltonian_excited_support_norm"
        ],
        "compact_dark_dissipative_excited_support_norm": compact_diag[
            "compact_dark_dissipative_excited_support_norm"
        ],
        "effective_dark_to_bright_mixing_Hz": compact_diag["effective_dark_to_bright_mixing_Hz"],
        "compact_dark_to_bright_mixing_Hz": compact_diag["compact_dark_to_bright_mixing_Hz"],
    }


def _effective_sector_populations(rho_effective: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(np.real(np.trace(rho_effective[:4, :4]))),
            float(np.real(np.trace(rho_effective[4:7, 4:7]))),
            float(np.real(np.trace(rho_effective[7:10, 7:10]))),
        ],
        dtype=np.float64,
    )


def _compact_sector_populations(rho_compact: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(np.real(np.trace(rho_compact[:4, :4]))),
            float(np.real(np.trace(rho_compact[4:7, 4:7]))),
            float(np.real(np.trace(rho_compact[7:10, 7:10]))),
        ],
        dtype=np.float64,
    )


def scan_local_reference_window(
    *,
    reference_electric_field: float | Sequence[float] | np.ndarray,
    scan_fields: Sequence[float | Sequence[float] | np.ndarray],
    magnetic_field: float | Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    kept_state_dressing_order: int = 3,
    excited_state_dressing_order: int = 4,
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_span: tuple[float, float] = (0.0, 20e-6),
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> dict:
    reference_electric = _as_field_vector(reference_electric_field)
    magnetic = _as_field_vector(magnetic_field)
    if t_eval is None:
        t_eval = np.array([t_span[1]], dtype=np.float64)
    else:
        t_eval = np.asarray(t_eval, dtype=np.float64)

    model = prepare_effective_hamiltonian_model(
        transition=transition,
        optical_polarization=optical_polarization,
        reference_electric_field=reference_electric,
        space_generation_electric_field=reference_electric,
        reference_magnetic_field=magnetic,
        space_generation_magnetic_field=magnetic,
        kept_state_dressing_order=kept_state_dressing_order,
        excited_state_dressing_order=excited_state_dressing_order,
    )

    results: list[dict] = []
    for field in scan_fields:
        electric = _as_field_vector(field)
        summary = summarize_static_comparison(
            model,
            electric_field=electric,
            magnetic_field=magnetic,
        )
        side = side_excited_subspace_diagnostics(
            model,
            electric_field=electric,
            magnetic_field=magnetic,
            rabi_rate=rabi_rate,
            detuning=detuning,
        )
        effective_ground_kernel = np.asarray(
            side["effective_side_decay_kernel_ground_eigvals_Hz"],
            dtype=np.float64,
        )
        compact_ground_kernel = np.asarray(
            side["compact_side_decay_kernel_ground_eigvals_Hz"],
            dtype=np.float64,
        )
        effective_sink_kernel = np.sum(
            np.array(
                [
                    np.asarray(values, dtype=np.float64)
                    for values in side["effective_side_decay_kernel_sink_eigvals_Hz"]
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        compact_sink_kernel = np.sum(
            np.array(
                [
                    np.asarray(values, dtype=np.float64)
                    for values in side["compact_side_decay_kernel_sink_eigvals_Hz"]
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        rel_ground_error = float(
            np.max(
                np.abs(effective_ground_kernel - compact_ground_kernel)
                / np.maximum(np.abs(compact_ground_kernel), 1.0)
            )
        )
        rel_sink_error = float(
            np.max(
                np.abs(effective_sink_kernel - compact_sink_kernel)
                / np.maximum(np.abs(compact_sink_kernel), 1.0)
            )
        )

        solution_effective = solve_static_effective_model(
            model,
            electric_field=electric,
            magnetic_field=magnetic,
            t_span=t_span,
            rabi_rate=rabi_rate,
            detuning=detuning,
            t_eval=t_eval,
            method=method,
        )
        rho_effective = solution_to_density_matrices(
            solution_effective,
            model.n_effective_states,
        )[-1]
        _, compact_bundle, solution_compact = solve_static_compact_reference_model(
            transition=transition,
            optical_polarization=optical_polarization,
            electric_field=electric,
            magnetic_field=magnetic,
            rabi_rate=rabi_rate,
            detuning=detuning,
            t_span=t_span,
            t_eval=t_eval,
            method=method,
        )
        rho_compact = solution_to_density_matrices(
            solution_compact,
            compact_bundle.h_internal.shape[0],
        )[-1]
        effective_sector = _effective_sector_populations(rho_effective)
        compact_sector = _compact_sector_populations(rho_compact)
        max_sector_abs_diff = float(np.max(np.abs(effective_sector - compact_sector)))

        results.append(
            {
                "electric_field": electric,
                "field_z_Vcm": float(electric[2]),
                "perturbative_ratio_max": summary["perturbative_ratio_max"],
                "dark_ground_overlap_abs": summary["dark_ground_overlap_abs"],
                "side_subspace_singular_values": np.asarray(
                    side["side_subspace_singular_values"],
                    dtype=np.float64,
                ),
                "side_projector_error_norm": float(side["side_projector_error_norm"]),
                "side_rel_ground_kernel_error": rel_ground_error,
                "side_rel_sink_kernel_error": rel_sink_error,
                "effective_sector_populations": effective_sector,
                "compact_sector_populations": compact_sector,
                "max_sector_abs_diff": max_sector_abs_diff,
            }
        )

    return {
        "reference_electric_field": reference_electric,
        "magnetic_field": magnetic,
        "t_span": t_span,
        "t_eval": t_eval,
        "results": results,
    }


def scan_local_operator_expansion_window(
    *,
    reference_electric_field: float | Sequence[float] | np.ndarray,
    scan_fields: Sequence[float | Sequence[float] | np.ndarray],
    expansion_axis: int = 2,
    expansion_order: int = 2,
    expansion_step: float = 10.0,
    magnetic_field: float | Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    kept_state_dressing_order: int = 3,
    excited_state_dressing_order: int = 4,
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_span: tuple[float, float] = (0.0, 20e-6),
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> dict:
    reference_electric = _as_field_vector(reference_electric_field)
    magnetic = _as_field_vector(magnetic_field)
    if t_eval is None:
        t_eval = np.array([t_span[1]], dtype=np.float64)
    else:
        t_eval = np.asarray(t_eval, dtype=np.float64)

    model = prepare_local_operator_expansion_model(
        reference_electric_field=reference_electric,
        expansion_axis=expansion_axis,
        expansion_order=expansion_order,
        expansion_step=expansion_step,
        kept_state_dressing_order=kept_state_dressing_order,
        excited_state_dressing_order=excited_state_dressing_order,
        transition=transition,
        optical_polarization=optical_polarization,
        reference_magnetic_field=magnetic,
    )

    results: list[dict] = []
    for field in scan_fields:
        electric = _as_field_vector(field)
        summary = summarize_static_comparison(
            model,  # type: ignore[arg-type]
            electric_field=electric,
            magnetic_field=magnetic,
        )
        side = side_excited_subspace_diagnostics(
            model,  # type: ignore[arg-type]
            electric_field=electric,
            magnetic_field=magnetic,
            rabi_rate=rabi_rate,
            detuning=detuning,
        )
        effective_ground_kernel = np.asarray(
            side["effective_side_decay_kernel_ground_eigvals_Hz"],
            dtype=np.float64,
        )
        compact_ground_kernel = np.asarray(
            side["compact_side_decay_kernel_ground_eigvals_Hz"],
            dtype=np.float64,
        )
        effective_sink_kernel = np.sum(
            np.array(
                [
                    np.asarray(values, dtype=np.float64)
                    for values in side["effective_side_decay_kernel_sink_eigvals_Hz"]
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        compact_sink_kernel = np.sum(
            np.array(
                [
                    np.asarray(values, dtype=np.float64)
                    for values in side["compact_side_decay_kernel_sink_eigvals_Hz"]
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        rel_ground_error = float(
            np.max(
                np.abs(effective_ground_kernel - compact_ground_kernel)
                / np.maximum(np.abs(compact_ground_kernel), 1.0)
            )
        )
        rel_sink_error = float(
            np.max(
                np.abs(effective_sink_kernel - compact_sink_kernel)
                / np.maximum(np.abs(compact_sink_kernel), 1.0)
            )
        )

        solution_effective = solve_static_effective_model(
            model,  # type: ignore[arg-type]
            electric_field=electric,
            magnetic_field=magnetic,
            t_span=t_span,
            rabi_rate=rabi_rate,
            detuning=detuning,
            t_eval=t_eval,
            method=method,
        )
        rho_effective = solution_to_density_matrices(
            solution_effective,
            model.n_effective_states,
        )[-1]
        _, compact_bundle, solution_compact = solve_static_compact_reference_model(
            transition=transition,
            optical_polarization=optical_polarization,
            electric_field=electric,
            magnetic_field=magnetic,
            rabi_rate=rabi_rate,
            detuning=detuning,
            t_span=t_span,
            t_eval=t_eval,
            method=method,
        )
        rho_compact = solution_to_density_matrices(
            solution_compact,
            compact_bundle.h_internal.shape[0],
        )[-1]
        effective_sector = _effective_sector_populations(rho_effective)
        compact_sector = _compact_sector_populations(rho_compact)
        max_sector_abs_diff = float(np.max(np.abs(effective_sector - compact_sector)))

        results.append(
            {
                "electric_field": electric,
                "field_z_Vcm": float(electric[2]),
                "perturbative_ratio_max": summary["perturbative_ratio_max"],
                "dark_ground_overlap_abs": summary["dark_ground_overlap_abs"],
                "side_subspace_singular_values": np.asarray(
                    side["side_subspace_singular_values"],
                    dtype=np.float64,
                ),
                "side_projector_error_norm": float(side["side_projector_error_norm"]),
                "side_rel_ground_kernel_error": rel_ground_error,
                "side_rel_sink_kernel_error": rel_sink_error,
                "effective_sector_populations": effective_sector,
                "compact_sector_populations": compact_sector,
                "max_sector_abs_diff": max_sector_abs_diff,
            }
        )

    return {
        "reference_electric_field": reference_electric,
        "magnetic_field": magnetic,
        "expansion_axis": int(expansion_axis),
        "expansion_order": int(expansion_order),
        "expansion_step": float(expansion_step),
        "t_span": t_span,
        "t_eval": t_eval,
        "results": results,
    }


def _aligned_exact_compact_bundle_for_field(
    model: PreparedInterpolatedEffectiveHamiltonianModel | PreparedLindbladSafeCompactInterpolatedHamiltonianModel,
    electric_field: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, OperatorBundle]:
    electric = _as_field_vector(electric_field)
    system, bundle = build_compact_reference_decomposed_bundle(
        transition=model.transition,
        optical_polarization=model.optical_polarization,
        electric_field=electric,
        magnetic_field=model.reference_magnetic_field,
    )
    patch_fields = np.asarray(model.field_points, dtype=np.float64)
    nearest_index = int(np.argmin(np.abs(patch_fields - float(electric[2]))))
    reference_basis = np.asarray(model.patches[nearest_index].aligned_basis_vectors, dtype=np.complex128)
    reference_ground_basis = reference_basis[:, model.ground_indices]
    reference_excited_basis = reference_basis[:, model.excited_indices]
    ground_selector = states.QuantumSelector(
        J=int(model.transition.J_ground),
        electronic=model.transition.electronic_ground,
    )
    raw_basis, reordered_bundle = _embed_aligned_compact_patch_to_union_layout(
        system,
        bundle,
        model.parent_basis_qn,
        ground_selector=ground_selector,
        sink_union_keys=model.union_state_keys,
        ground_indices_union=model.ground_indices,
        excited_indices_union=model.excited_indices,
        reference_ground_basis=reference_ground_basis,
        reference_excited_basis=reference_excited_basis,
    )
    aligned_basis, rotation = _union_block_alignment(
        reference_basis,
        raw_basis,
        ground_indices=model.ground_indices,
        sink_indices=model.sink_indices,
        excited_indices=model.excited_indices,
    )
    excited_rotation = rotation[np.ix_(model.excited_indices, model.excited_indices)]
    aligned_bundle = _transform_operator_bundle(
        reordered_bundle,
        rotation,
        decay_kernel_ground=_symmetrize(
            excited_rotation
            @ np.asarray(reordered_bundle.decay_kernel_ground, dtype=np.complex128)
            @ excited_rotation.conj().T
        )
        if reordered_bundle.decay_kernel_ground is not None
        else None,
        decay_kernels_sinks=tuple(
            _symmetrize(
                excited_rotation @ np.asarray(kernel, dtype=np.complex128) @ excited_rotation.conj().T
            )
            for kernel in (reordered_bundle.decay_kernels_sinks or ())
        )
        if reordered_bundle.decay_kernels_sinks is not None
        else None,
    )
    if hasattr(model, "common_omega_reference"):
        patch_omega = _compact_transition_frequency(
            system,
            transition=model.transition,
            optical_polarization=model.optical_polarization,
        )
        aligned_bundle = _shift_bundle_to_common_frequency_frame(
            aligned_bundle,
            delta_omega=float(patch_omega - float(getattr(model, "common_omega_reference"))),
            common_omega_reference=float(getattr(model, "common_omega_reference")),
        )
    return aligned_basis, aligned_bundle


def _tracked_exact_compact_bundle_for_field(
    model: PreparedInstantaneousInterpolatedEffectiveHamiltonianModel,
    electric_field: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, OperatorBundle]:
    electric = _as_field_vector(electric_field)
    system, bundle = build_compact_reference_decomposed_bundle(
        transition=model.transition,
        optical_polarization=model.optical_polarization,
        electric_field=electric,
        magnetic_field=model.reference_magnetic_field,
    )
    patch_fields = np.asarray(model.field_points, dtype=np.float64)
    nearest_index = int(np.argmin(np.abs(patch_fields - float(electric[2]))))
    reference_coherent_basis = np.asarray(
        model.patches[nearest_index].coherent_basis_vectors,
        dtype=np.complex128,
    )
    n_ground = int(model.ground_indices.size)
    reference_ground_basis = reference_coherent_basis[:, :n_ground]
    reference_excited_basis = reference_coherent_basis[:, n_ground:]
    ground_selector = states.QuantumSelector(
        J=int(model.transition.J_ground),
        electronic=model.transition.electronic_ground,
    )
    raw_basis, coherent_basis, tracked_bundle = _embed_tracked_compact_patch_to_union_layout(
        system,
        bundle,
        model.parent_basis_qn,
        ground_selector=ground_selector,
        sink_union_keys=model.union_state_keys,
        ground_indices_union=model.ground_indices,
        excited_indices_union=model.excited_indices,
        reference_ground_basis=reference_ground_basis,
        reference_excited_basis=reference_excited_basis,
    )
    return raw_basis, coherent_basis, _make_bundle_sinks_dissipative_only(tracked_bundle, model.sink_indices)


def scan_interpolated_effective_model_window(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    scan_fields: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    magnetic_field: float | Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 20e-6),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    keep_diagnostics: bool = True,
) -> dict:
    magnetic = _as_field_vector(magnetic_field)
    if t_eval is None:
        t_eval = np.array([t_span[1]], dtype=np.float64)
    else:
        t_eval = np.asarray(t_eval, dtype=np.float64)

    model = prepare_interpolated_effective_model(
        field_points=field_points,
        transition=transition,
        optical_polarization=optical_polarization,
        magnetic_field=magnetic,
        master_field=master_field,
        keep_diagnostics=keep_diagnostics,
    )

    results: list[dict] = []
    for field in scan_fields:
        electric = _as_field_vector(field)
        bundle_interpolated = model.effective_bundle(electric, magnetic)
        _, bundle_exact = _aligned_exact_compact_bundle_for_field(model, electric)

        h_internal_rel_error = float(
            np.linalg.norm(bundle_interpolated.h_internal - bundle_exact.h_internal)
            / max(np.linalg.norm(bundle_exact.h_internal), 1.0)
        )
        h_opt_rel_error = float(
            np.linalg.norm(bundle_interpolated.h_opt - bundle_exact.h_opt)
            / max(np.linalg.norm(bundle_exact.h_opt), 1.0)
        )
        dissipator_rel_error = float(
            np.linalg.norm(
                bundle_interpolated.dissipator_superoperator() - bundle_exact.dissipator_superoperator()
            )
            / max(np.linalg.norm(bundle_exact.dissipator_superoperator()), 1.0)
        )
        ground_kernel_rel_error = None
        sink_kernel_rel_error = None
        if (
            bundle_interpolated.decay_kernel_ground is not None
            and bundle_exact.decay_kernel_ground is not None
            and bundle_interpolated.decay_kernels_sinks is not None
            and bundle_exact.decay_kernels_sinks is not None
        ):
            ground_kernel_rel_error = float(
                np.linalg.norm(bundle_interpolated.decay_kernel_ground - bundle_exact.decay_kernel_ground)
                / max(np.linalg.norm(bundle_exact.decay_kernel_ground), 1.0)
            )
            sink_interp = np.sum(
                np.array(bundle_interpolated.decay_kernels_sinks, dtype=np.complex128),
                axis=0,
            )
            sink_exact = np.sum(
                np.array(bundle_exact.decay_kernels_sinks, dtype=np.complex128),
                axis=0,
            )
            sink_kernel_rel_error = float(
                np.linalg.norm(sink_interp - sink_exact) / max(np.linalg.norm(sink_exact), 1.0)
            )

        solution_interpolated = solve_static_effective_model(
            model,  # type: ignore[arg-type]
            electric_field=electric,
            magnetic_field=magnetic,
            t_span=t_span,
            rabi_rate=rabi_rate,
            detuning=detuning,
            t_eval=t_eval,
            method=method,
        )
        rho_interpolated = solution_to_density_matrices(
            solution_interpolated,
            model.n_effective_states,
        )[-1]
        solution_compact = solve_static_density_matrix_bundle(
            bundle_exact,
            rho0=default_effective_density_matrix(model),  # type: ignore[arg-type]
            t_span=t_span,
            rabi_rate=rabi_rate,
            detuning=detuning,
            t_eval=t_eval,
            method=method,
        )
        rho_compact = solution_to_density_matrices(solution_compact, model.n_effective_states)[-1]
        interpolated_sector = _effective_sector_populations(rho_interpolated)
        compact_sector = _effective_sector_populations(rho_compact)
        max_sector_abs_diff = float(np.max(np.abs(interpolated_sector - compact_sector)))

        results.append(
            {
                "electric_field": electric,
                "field_z_Vcm": float(electric[2]),
                "h_internal_rel_error": h_internal_rel_error,
                "h_opt_rel_error": h_opt_rel_error,
                "dissipator_rel_error": dissipator_rel_error,
                "ground_kernel_rel_error": ground_kernel_rel_error,
                "sink_kernel_rel_error": sink_kernel_rel_error,
                "interpolated_sector_populations": interpolated_sector,
                "compact_sector_populations": compact_sector,
                "max_sector_abs_diff": max_sector_abs_diff,
            }
        )

    return {
        "field_points": np.asarray(model.field_points, dtype=np.float64),
        "master_field": float(model.master_field),
        "magnetic_field": magnetic,
        "t_span": t_span,
        "t_eval": t_eval,
        "results": results,
    }


def excited_state_composition_diagnostics(
    model: PreparedEffectiveHamiltonianModel,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    *,
    top_k: int = 8,
) -> dict:
    if int(model.p_excited_indices_local.size) != 3:
        raise ValueError(
            "excited_state_composition_diagnostics currently expects a 3-state kept excited manifold."
        )
    electric = _as_field_vector(electric_field)
    magnetic = (
        np.asarray(model.reference_magnetic_field, dtype=np.float64)
        if magnetic_field is None
        else _as_field_vector(magnetic_field)
    )

    h_blocks_lab = model.assemble_internal_blocks(electric, magnetic)
    s_pq, s_qp, max_ratio, sylvester_residual_norm, spectral_separation = _build_generator_s_blocks(
        h_blocks_lab.pp,
        h_blocks_lab.pq,
        h_blocks_lab.qq,
        model.denominator_floor,
    )
    generator = np.block(
        [
            [np.zeros_like(h_blocks_lab.pp), s_pq],
            [s_qp, np.zeros_like(h_blocks_lab.qq)],
        ]
    ).astype(np.complex128)
    p_vectors_base, _, p_vectors_mixed = _mixed_order_kept_vectors(
        generator,
        model.n_active_states,
        np.asarray(model.p_excited_indices_local, dtype=np.int64),
        base_order=model.kept_state_dressing_order,
        excited_order=model.excited_state_dressing_order,
    )
    p_vectors_full = _partitioned_vectors_to_full_basis(
        p_vectors_mixed,
        model.p_indices,
        model.q_indices,
    )
    h_det_partitioned = _assemble_full_from_blocks(model.h_det_blocks)
    h_det_active = _symmetrize(
        _matrix_elements_dressed_states(h_det_partitioned, p_vectors_base)
    )
    omega_reference = float(
        np.real(np.diag(h_blocks_lab.pp))[model.excited_main_index_p]
        - np.real(np.diag(h_blocks_lab.pp))[model.ground_main_index_p]
    )
    delta_h = _second_order_hamiltonian_correction_blocks(
        s_pq,
        s_qp,
        h_blocks_lab.pq,
        h_blocks_lab.qp,
    )
    h_eff_lab_active = _symmetrize(h_blocks_lab.pp + delta_h)
    h_internal_active = _symmetrize(h_eff_lab_active - omega_reference * h_det_active)
    excited_rows = np.asarray(model.p_excited_indices_local, dtype=np.int64)
    effective_excited_block = np.asarray(
        h_internal_active[np.ix_(excited_rows, excited_rows)],
        dtype=np.complex128,
    )
    effective_excited_eigvals, effective_excited_eigvecs_local = np.linalg.eigh(
        effective_excited_block
    )
    effective_excited_vectors = np.asarray(
        p_vectors_full[:, excited_rows] @ effective_excited_eigvecs_local,
        dtype=np.complex128,
    )

    exact = model.exact_bundle(electric, magnetic)
    exact_eigvals, exact_eigvecs = np.linalg.eigh(_symmetrize(exact.h_lab_internal))
    exact_eigvecs = np.asarray(exact_eigvecs, dtype=np.complex128)
    exact_excited_weights = np.sum(
        np.abs(exact_eigvecs[model.p_excited_indices, :]) ** 2,
        axis=0,
    )
    n_excited = int(model.p_excited_indices_local.size)
    selected = np.argsort(exact_excited_weights)[-n_excited:]
    selected = selected[np.argsort(np.real(exact_eigvals[selected]))]
    exact_excited_vectors = np.asarray(exact_eigvecs[:, selected], dtype=np.complex128)
    exact_selected_eigvals = np.asarray(exact_eigvals[selected], dtype=np.float64)
    exact_selected_weights = np.asarray(exact_excited_weights[selected], dtype=np.float64)

    overlap_matrix_abs, matched_perm = _match_columns_by_overlap(
        effective_excited_vectors,
        exact_excited_vectors,
    )
    exact_excited_vectors = exact_excited_vectors[:, matched_perm]
    exact_selected_eigvals = exact_selected_eigvals[matched_perm]
    exact_selected_weights = exact_selected_weights[matched_perm]

    for idx in range(n_excited):
        phase = np.vdot(exact_excited_vectors[:, idx], effective_excited_vectors[:, idx])
        if abs(phase) > 0:
            exact_excited_vectors[:, idx] *= np.exp(-1j * np.angle(phase))

    x_indices = np.arange(len(model.x_states), dtype=np.int64)
    kept_excited_indices = np.asarray(model.p_excited_indices, dtype=np.int64)
    all_b_indices = np.arange(len(model.x_states), len(model.qn_full), dtype=np.int64)
    omitted_b_indices = np.setdiff1d(all_b_indices, kept_excited_indices, assume_unique=True)
    same_manifold_indices = _selector_indices(
        model.qn_full,
        states.QuantumSelector(
            J=1,
            F1=1 / 2,
            F=1,
            electronic=states.ElectronicState.B,
        ),
    )
    omitted_same_manifold_indices = np.setdiff1d(
        same_manifold_indices,
        kept_excited_indices,
        assume_unique=False,
    )
    other_b_indices = np.setdiff1d(
        omitted_b_indices,
        omitted_same_manifold_indices,
        assume_unique=False,
    )
    labels = [_state_label(state) for state in model.qn_full]

    def sector_weights(vector: np.ndarray) -> dict[str, float]:
        weights = np.abs(np.asarray(vector, dtype=np.complex128)) ** 2
        return {
            "x_total": float(np.sum(weights[x_indices])),
            "kept_excited": float(np.sum(weights[kept_excited_indices])),
            "omitted_same_manifold": float(np.sum(weights[omitted_same_manifold_indices])),
            "omitted_other_b": float(np.sum(weights[other_b_indices])),
        }

    state_summaries: list[dict] = []
    matched_overlaps = np.abs(
        np.sum(effective_excited_vectors.conj() * exact_excited_vectors, axis=0)
    )
    for idx in range(n_excited):
        state_summaries.append(
            {
                "state_index": idx,
                "matched_overlap_abs": float(matched_overlaps[idx]),
                "effective_eigenvalue_MHz": float(
                    np.real(effective_excited_eigvals[idx]) / (2.0 * np.pi * 1e6)
                ),
                "exact_selected_eigenvalue_MHz": float(
                    np.real(exact_selected_eigvals[idx]) / (2.0 * np.pi * 1e6)
                ),
                "exact_selected_kept_excited_weight": float(exact_selected_weights[idx]),
                "effective_sector_weights": sector_weights(effective_excited_vectors[:, idx]),
                "exact_sector_weights": sector_weights(exact_excited_vectors[:, idx]),
                "effective_top_components": _top_component_summary(
                    effective_excited_vectors[:, idx],
                    labels,
                    top_k=top_k,
                ),
                "exact_top_components": _top_component_summary(
                    exact_excited_vectors[:, idx],
                    labels,
                    top_k=top_k,
                ),
            }
        )

    return {
        "perturbative_ratio_max": max_ratio,
        "sylvester_residual_norm": sylvester_residual_norm,
        "spectral_separation_min_MHz": spectral_separation / (2.0 * np.pi * 1e6),
        "overlap_matrix_abs": overlap_matrix_abs,
        "matched_exact_indices": selected[matched_perm],
        "matched_exact_eigenvalues_MHz": np.real(exact_selected_eigvals) / (2.0 * np.pi * 1e6),
        "state_summaries": state_summaries,
    }


def side_excited_subspace_diagnostics(
    model: PreparedEffectiveHamiltonianModel,
    electric_field: float | Sequence[float] | np.ndarray,
    magnetic_field: float | Sequence[float] | np.ndarray | None = None,
    *,
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    polarization_scale: float = 1.0,
) -> dict:
    if int(model.p_excited_indices_local.size) != 3:
        raise ValueError(
            "side_excited_subspace_diagnostics currently expects a 3-state kept excited manifold."
        )
    electric = _as_field_vector(electric_field)
    magnetic = (
        np.asarray(model.reference_magnetic_field, dtype=np.float64)
        if magnetic_field is None
        else _as_field_vector(magnetic_field)
    )

    h_blocks_lab = model.assemble_internal_blocks(electric, magnetic)
    s_pq, s_qp, _, sylvester_residual_norm, spectral_separation = _build_generator_s_blocks(
        h_blocks_lab.pp,
        h_blocks_lab.pq,
        h_blocks_lab.qq,
        model.denominator_floor,
    )
    generator = np.block(
        [
            [np.zeros_like(h_blocks_lab.pp), s_pq],
            [s_qp, np.zeros_like(h_blocks_lab.qq)],
        ]
    ).astype(np.complex128)
    p_vectors_base, _, p_vectors_mixed = _mixed_order_kept_vectors(
        generator,
        model.n_active_states,
        np.asarray(model.p_excited_indices_local, dtype=np.int64),
        base_order=model.kept_state_dressing_order,
        excited_order=model.excited_state_dressing_order,
    )
    p_vectors_full = _partitioned_vectors_to_full_basis(
        p_vectors_mixed,
        model.p_indices,
        model.q_indices,
    )
    h_det_partitioned = _assemble_full_from_blocks(model.h_det_blocks)
    h_det_active = _symmetrize(
        _matrix_elements_dressed_states(h_det_partitioned, p_vectors_base)
    )
    omega_reference = float(
        np.real(np.diag(h_blocks_lab.pp))[model.excited_main_index_p]
        - np.real(np.diag(h_blocks_lab.pp))[model.ground_main_index_p]
    )
    delta_h = _second_order_hamiltonian_correction_blocks(
        s_pq,
        s_qp,
        h_blocks_lab.pq,
        h_blocks_lab.qp,
    )
    h_eff_lab_active = _symmetrize(h_blocks_lab.pp + delta_h)
    h_internal_active = _symmetrize(h_eff_lab_active - omega_reference * h_det_active)
    excited_rows = np.asarray(model.p_excited_indices_local, dtype=np.int64)
    effective_excited_block = np.asarray(
        h_internal_active[np.ix_(excited_rows, excited_rows)],
        dtype=np.complex128,
    )
    effective_excited_eigvals, effective_excited_eigvecs_local = np.linalg.eigh(
        effective_excited_block
    )
    effective_excited_vectors = np.asarray(
        p_vectors_full[:, excited_rows] @ effective_excited_eigvecs_local,
        dtype=np.complex128,
    )

    exact = model.exact_bundle(electric, magnetic)
    exact_eigvals, exact_eigvecs = np.linalg.eigh(_symmetrize(exact.h_lab_internal))
    exact_eigvecs = np.asarray(exact_eigvecs, dtype=np.complex128)
    exact_excited_weights = np.sum(
        np.abs(exact_eigvecs[model.p_excited_indices, :]) ** 2,
        axis=0,
    )
    n_excited = int(model.p_excited_indices_local.size)
    selected = np.argsort(exact_excited_weights)[-n_excited:]
    selected = selected[np.argsort(np.real(exact_eigvals[selected]))]
    exact_excited_vectors = np.asarray(exact_eigvecs[:, selected], dtype=np.complex128)
    exact_selected_eigvals = np.asarray(exact_eigvals[selected], dtype=np.float64)

    overlap_matrix_abs, matched_perm = _match_columns_by_overlap(
        effective_excited_vectors,
        exact_excited_vectors,
    )
    exact_excited_vectors = exact_excited_vectors[:, matched_perm]
    exact_selected_eigvals = exact_selected_eigvals[matched_perm]
    matched_overlaps = np.abs(
        np.sum(effective_excited_vectors.conj() * exact_excited_vectors, axis=0)
    )

    middle_index = int(np.argmax(matched_overlaps))
    side_indices = np.array([idx for idx in range(n_excited) if idx != middle_index], dtype=np.int64)
    effective_side_vectors = np.asarray(effective_excited_vectors[:, side_indices], dtype=np.complex128)
    exact_side_vectors = np.asarray(exact_excited_vectors[:, side_indices], dtype=np.complex128)
    side_overlap_singular_values = np.linalg.svd(
        effective_side_vectors.conj().T @ exact_side_vectors,
        compute_uv=False,
    )
    effective_side_projector = effective_side_vectors @ effective_side_vectors.conj().T
    exact_side_projector = exact_side_vectors @ exact_side_vectors.conj().T
    side_projector_error_norm = float(
        np.linalg.norm(effective_side_projector - exact_side_projector)
    )

    compact_diag = compact_reference_diagnostics(
        model,
        electric,
        magnetic,
        rabi_rate=rabi_rate,
        detuning=detuning,
        polarization_scale=polarization_scale,
    )
    effective_branch_ground = np.asarray(compact_diag["effective_excited_to_ground_Hz"], dtype=np.float64)
    compact_branch_ground = np.asarray(compact_diag["compact_excited_to_ground_Hz"], dtype=np.float64)
    effective_middle_kernel_index = int(np.argmax(effective_branch_ground))
    compact_middle_kernel_index = int(np.argmax(compact_branch_ground))
    effective_side_kernel_indices = np.array(
        [idx for idx in range(n_excited) if idx != effective_middle_kernel_index],
        dtype=np.int64,
    )
    compact_side_kernel_indices = np.array(
        [idx for idx in range(n_excited) if idx != compact_middle_kernel_index],
        dtype=np.int64,
    )

    effective_kernel_ground = np.asarray(compact_diag["effective_decay_kernel_ground_Hz"], dtype=np.complex128)
    compact_kernel_ground = np.asarray(compact_diag["compact_decay_kernel_ground_Hz"], dtype=np.complex128)
    effective_kernel_sinks = tuple(
        np.asarray(kernel, dtype=np.complex128)
        for kernel in compact_diag["effective_decay_kernel_sinks_Hz"]
    )
    compact_kernel_sinks = tuple(
        np.asarray(kernel, dtype=np.complex128)
        for kernel in compact_diag["compact_decay_kernel_sinks_Hz"]
    )

    def project_kernel(kernel: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return np.asarray(kernel[np.ix_(indices, indices)], dtype=np.complex128)

    effective_side_kernel_ground = project_kernel(
        effective_kernel_ground,
        effective_side_kernel_indices,
    )
    compact_side_kernel_ground = project_kernel(
        compact_kernel_ground,
        compact_side_kernel_indices,
    )
    effective_side_kernel_sinks = tuple(
        project_kernel(kernel, effective_side_kernel_indices) for kernel in effective_kernel_sinks
    )
    compact_side_kernel_sinks = tuple(
        project_kernel(kernel, compact_side_kernel_indices) for kernel in compact_kernel_sinks
    )

    return {
        "sylvester_residual_norm": sylvester_residual_norm,
        "spectral_separation_min_MHz": spectral_separation / (2.0 * np.pi * 1e6),
        "effective_excited_eigenvalues_MHz": np.real(effective_excited_eigvals) / (2.0 * np.pi * 1e6),
        "exact_selected_eigenvalues_MHz": np.real(exact_selected_eigvals) / (2.0 * np.pi * 1e6),
        "overlap_matrix_abs": overlap_matrix_abs,
        "matched_overlaps_abs": matched_overlaps,
        "middle_index": middle_index,
        "side_indices": side_indices,
        "side_subspace_singular_values": side_overlap_singular_values,
        "side_projector_error_norm": side_projector_error_norm,
        "effective_side_kernel_indices": effective_side_kernel_indices,
        "compact_side_kernel_indices": compact_side_kernel_indices,
        "effective_side_decay_kernel_ground_Hz": effective_side_kernel_ground,
        "compact_side_decay_kernel_ground_Hz": compact_side_kernel_ground,
        "effective_side_decay_kernel_ground_eigvals_Hz": np.linalg.eigvalsh(
            _symmetrize(effective_side_kernel_ground)
        ),
        "compact_side_decay_kernel_ground_eigvals_Hz": np.linalg.eigvalsh(
            _symmetrize(compact_side_kernel_ground)
        ),
        "effective_side_decay_kernel_sinks_Hz": effective_side_kernel_sinks,
        "compact_side_decay_kernel_sinks_Hz": compact_side_kernel_sinks,
        "effective_side_decay_kernel_sink_eigvals_Hz": tuple(
            np.linalg.eigvalsh(_symmetrize(kernel)) for kernel in effective_side_kernel_sinks
        ),
        "compact_side_decay_kernel_sink_eigvals_Hz": tuple(
            np.linalg.eigvalsh(_symmetrize(kernel)) for kernel in compact_side_kernel_sinks
        ),
    }


def build_compact_reference_bundle(
    *,
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    electric_field: Sequence[float] | np.ndarray = (0.0, 0.0, 100.0),
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    polarization_scale: float = 1.0,
) -> tuple[lindblad.utils_setup.OBESystem, OperatorBundle]:
    electric = _as_field_vector(electric_field)
    magnetic = _as_field_vector(magnetic_field)
    transition_selectors = couplings.generate_transition_selectors(
        transitions=[transition],
        polarizations=[[optical_polarization]],
    )
    system = lindblad.generate_OBE_system_transitions(
        [transition],
        transition_selectors,
        E=electric,
        B=magnetic,
        qn_compact=True,
    )

    symbols = list(system.H_symbolic.free_symbols)
    ham_lambdify = smp.lambdify(symbols, system.H_symbolic, modules="numpy", cse=True)
    values: dict[str, float | complex] = {}
    for symbol in symbols:
        name = str(symbol)
        if name.startswith("Ω"):
            values[name] = complex(rabi_rate)
        elif name.startswith("δ"):
            values[name] = float(detuning)
        elif name.startswith("PZ"):
            values[name] = float(polarization_scale)
        elif name.startswith("PX") or name.startswith("PY"):
            values[name] = 0.0
        else:
            raise ValueError(f"Unsupported compact-reference symbol {name}")

    ham = np.asarray(ham_lambdify(**values), dtype=np.complex128)
    excited_indices = _selector_indices(
        system.QN,
        states.QuantumSelector(electronic=states.ElectronicState.B),
    )
    bundle = OperatorBundle(
        electric_field=electric,
        magnetic_field=magnetic,
        omega_reference=0.0,
        h_internal=ham,
        h_opt=np.zeros_like(ham, dtype=np.complex128),
        h_det=np.zeros_like(ham, dtype=np.complex128),
        c_array=np.asarray(system.C_array, dtype=np.complex128),
        excited_indices=excited_indices,
        loss_operator=np.zeros_like(ham, dtype=np.complex128),
        h_full_internal=ham,
        h_lab_internal=ham,
        hermiticity_error=float(np.max(np.abs(ham - ham.conj().T))),
    )
    return system, bundle


def build_compact_reference_decomposed_bundle(
    *,
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    electric_field: Sequence[float] | np.ndarray = (0.0, 0.0, 100.0),
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    polarization_scale: float = 1.0,
) -> tuple[lindblad.utils_setup.OBESystem, OperatorBundle]:
    electric = _as_field_vector(electric_field)
    magnetic = _as_field_vector(magnetic_field)
    transition_selectors = couplings.generate_transition_selectors(
        transitions=[transition],
        polarizations=[[optical_polarization]],
    )
    system = lindblad.generate_OBE_system_transitions(
        [transition],
        transition_selectors,
        E=electric,
        B=magnetic,
        qn_compact=True,
    )

    symbols = list(system.H_symbolic.free_symbols)
    ham_lambdify = smp.lambdify(symbols, system.H_symbolic, modules="numpy", cse=True)

    def evaluate_compact_hamiltonian(
        *,
        rabi_value: complex = 0.0,
        detuning_value: float = 0.0,
    ) -> np.ndarray:
        values: dict[str, float | complex] = {}
        for symbol in symbols:
            name = str(symbol)
            if name.startswith("Ω"):
                values[name] = complex(rabi_value)
            elif name.startswith("δ"):
                values[name] = float(detuning_value)
            elif name.startswith("PZ"):
                values[name] = float(polarization_scale)
            elif name.startswith("PX") or name.startswith("PY"):
                values[name] = 0.0
            else:
                raise ValueError(f"Unsupported compact-reference symbol {name}")
        return np.asarray(ham_lambdify(**values), dtype=np.complex128)

    h_internal = _symmetrize(evaluate_compact_hamiltonian(rabi_value=0.0, detuning_value=0.0))
    h_with_rabi = _symmetrize(evaluate_compact_hamiltonian(rabi_value=1.0, detuning_value=0.0))
    h_with_detuning = _symmetrize(
        evaluate_compact_hamiltonian(rabi_value=0.0, detuning_value=1.0)
    )
    h_opt = _symmetrize(2.0 * (h_with_rabi - h_internal))
    h_det = _symmetrize(h_with_detuning - h_internal)
    c_array = np.asarray(system.C_array, dtype=np.complex128)
    excited_indices = _selector_indices(
        system.QN,
        states.QuantumSelector(electronic=states.ElectronicState.B),
    )
    ground_indices = _selector_indices(
        system.QN,
        states.QuantumSelector(J=0, electronic=states.ElectronicState.X),
    )
    sink_groups = tuple(
        _selector_indices(
            system.QN,
            states.QuantumSelector(J=J, electronic=states.ElectronicState.X),
        )
        for J in (1, 2, 3)
    )
    decay_kernel_ground = _sector_decay_kernel(c_array, excited_indices, ground_indices)
    decay_kernels_sinks = tuple(
        _sector_decay_kernel(c_array, excited_indices, sink_group)
        for sink_group in sink_groups
    )
    excited_to_ground_rates = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)
    excited_to_sink_rates = (
        np.sum(
            np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
            axis=0,
        )
        / (2.0 * np.pi)
    )

    bundle = OperatorBundle(
        electric_field=electric,
        magnetic_field=magnetic,
        omega_reference=0.0,
        h_internal=h_internal,
        h_opt=h_opt,
        h_det=h_det,
        c_array=c_array,
        excited_indices=excited_indices,
        loss_operator=np.zeros_like(h_internal, dtype=np.complex128),
        h_full_internal=h_internal,
        h_lab_internal=h_internal,
        hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
        excited_to_ground_rates_hz=excited_to_ground_rates,
        excited_to_sink_rates_hz=excited_to_sink_rates,
        decay_kernel_ground=decay_kernel_ground,
        decay_kernels_sinks=decay_kernels_sinks,
    )
    return system, bundle


def solve_static_compact_reference_model(
    *,
    transition: transitions.OpticalTransition = transitions.R0_F1_1o2_F1,
    optical_polarization: couplings.Polarization = couplings.polarization_Z,
    electric_field: Sequence[float] | np.ndarray = (0.0, 0.0, 100.0),
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    rabi_rate: float | complex = 2.0 * np.pi * 1e6,
    detuning: float = 0.0,
    polarization_scale: float = 1.0,
    rho0: np.ndarray | None = None,
    t_span: tuple[float, float] = (0.0, 50e-6),
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
) -> tuple[lindblad.utils_setup.OBESystem, OperatorBundle, solve_ivp]:
    system, bundle = build_compact_reference_bundle(
        transition=transition,
        optical_polarization=optical_polarization,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        rabi_rate=rabi_rate,
        detuning=detuning,
        polarization_scale=polarization_scale,
    )
    if rho0 is None:
        rho0 = np.zeros((bundle.h_internal.shape[0], bundle.h_internal.shape[0]), dtype=np.complex128)
        rho0[0, 0] = 1.0
    solution = solve_static_density_matrix_bundle(
        bundle,
        rho0=rho0,
        t_span=t_span,
        t_eval=t_eval,
        method=method,
    )
    return system, bundle, solution
