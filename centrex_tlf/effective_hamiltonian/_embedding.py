from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import block_diag

from centrex_tlf import lindblad, states
from centrex_tlf.effective_hamiltonian._utility import (
    _basis_transform,
    _polar_unitary,
    _selector_indices,
    _state_key,
    _symmetrize,
)
from centrex_tlf.effective_hamiltonian._superoperators import _embed_superoperator_to_union_layout
from centrex_tlf.effective_hamiltonian.operator_bundle import (
    OperatorBundle,
    _transform_operator_bundle,
)
from centrex_tlf.effective_hamiltonian._alignment import _tracking_rotation


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
