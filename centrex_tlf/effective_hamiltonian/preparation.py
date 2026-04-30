from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.effective_hamiltonian._utility import (
    _as_field_vector,
    _symmetrize,
)
from centrex_tlf.effective_hamiltonian._decay import (
    _full_recycling_decay_kernel,
    _single_target_decay_kernels,
)
from centrex_tlf.effective_hamiltonian._alignment import (
    _coherent_gauge_connection,
    _union_block_alignment,
)
from centrex_tlf.effective_hamiltonian._embedding import (
    _build_compact_union_layout,
    _compact_patch_role_indices,
    _embed_aligned_compact_patch_to_union_layout,
    _embed_tracked_compact_patch_to_union_layout,
    _make_bundle_sinks_dissipative_only,
    _optically_bright_ground_index,
)
from centrex_tlf.effective_hamiltonian.operator_bundle import (
    OperatorBundle,
    _compact_transition_frequency,
    _shift_bundle_to_common_frequency_frame,
    _transform_operator_bundle,
)
from centrex_tlf.effective_hamiltonian.compact_reference import (
    build_compact_reference_decomposed_bundle,
)
from centrex_tlf.effective_hamiltonian.models import (
    InterpolatedEffectivePatch,
    InstantaneousInterpolatedEffectivePatch,
    LindbladSafeCompactInterpolatedPatch,
    PreparedInterpolatedEffectiveHamiltonianModel,
    PreparedLindbladSafeCompactInterpolatedHamiltonianModel,
    PreparedInstantaneousInterpolatedEffectiveHamiltonianModel,
)


def _relative_adjacent_variation(values: Sequence[np.ndarray]) -> list[float]:
    variations: list[float] = []
    for left, right in zip(values[:-1], values[1:]):
        left_arr = np.asarray(left, dtype=np.complex128)
        right_arr = np.asarray(right, dtype=np.complex128)
        scale = max(float(np.linalg.norm(left_arr)), float(np.linalg.norm(right_arr)), 1e-300)
        variations.append(float(np.linalg.norm(right_arr - left_arr) / scale))
    return variations


def _operator_grid_variation_diagnostics(
    field_points: Sequence[float],
    patches: Sequence[
        InterpolatedEffectivePatch
        | LindbladSafeCompactInterpolatedPatch
        | InstantaneousInterpolatedEffectivePatch
    ],
) -> dict[str, object]:
    operator_values = {
        "h_internal": [patch.bundle.h_internal for patch in patches],
        "h_opt": [patch.bundle.h_opt for patch in patches],
        "h_det": [patch.bundle.h_det for patch in patches],
        "dissipator_superop": [patch.bundle.dissipator_superoperator() for patch in patches],
        "jump_rate_operator": [patch.bundle.jump_rate_operator() for patch in patches],
    }
    intervals = [
        (float(left), float(right))
        for left, right in zip(field_points[:-1], field_points[1:])
    ]
    by_operator: dict[str, dict[str, object]] = {}
    max_variation = 0.0
    max_operator = ""
    max_interval: tuple[float, float] | None = None

    for name, values in operator_values.items():
        variations = _relative_adjacent_variation(values)
        operator_max = max(variations, default=0.0)
        operator_interval = (
            intervals[int(np.argmax(variations))]
            if variations
            else None
        )
        by_operator[name] = {
            "adjacent_relative_variation": variations,
            "max_relative_variation": operator_max,
            "max_interval": operator_interval,
        }
        if operator_max > max_variation:
            max_variation = operator_max
            max_operator = name
            max_interval = operator_interval

    return {
        "metric": "adjacent_relative_frobenius_norm",
        "by_operator": by_operator,
        "max_relative_variation": max_variation,
        "max_operator": max_operator,
        "max_interval": max_interval,
    }


def _warn_large_operator_grid_variation(
    diagnostics: dict[str, object],
    threshold: float | None,
) -> None:
    if threshold is None:
        return
    max_variation = float(diagnostics.get("max_relative_variation", 0.0))
    if max_variation <= float(threshold):
        return
    warnings.warn(
        "Effective Hamiltonian operator grid has large adjacent variation: "
        f"{diagnostics.get('max_operator')} changes by {max_variation:.3g} "
        f"across field interval {diagnostics.get('max_interval')}. "
        "Consider adding intermediate field points before relying on interpolation.",
        RuntimeWarning,
        stacklevel=3,
    )


def prepare_interpolated_effective_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition,
    optical_polarization: couplings.Polarization,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    interpolation_kind: str = "linear",
    keep_diagnostics: bool = True,
    grid_variation_warning_threshold: float | None = 1.0,
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
    grid_variation_diagnostics = _operator_grid_variation_diagnostics(
        unique_sorted.tolist(),
        patches,
    )
    _warn_large_operator_grid_variation(
        grid_variation_diagnostics,
        grid_variation_warning_threshold,
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
        grid_variation_diagnostics=grid_variation_diagnostics,
    )


def prepare_lindblad_safe_compact_interpolated_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition,
    optical_polarization: couplings.Polarization,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    interpolation_kind: str = "linear",
    keep_diagnostics: bool = True,
    grid_variation_warning_threshold: float | None = 1.0,
) -> PreparedLindbladSafeCompactInterpolatedHamiltonianModel:
    base_model = prepare_interpolated_effective_model(
        field_points=field_points,
        transition=transition,
        optical_polarization=optical_polarization,
        magnetic_field=magnetic_field,
        master_field=master_field,
        interpolation_kind=interpolation_kind,
        keep_diagnostics=keep_diagnostics,
        grid_variation_warning_threshold=None,
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
    grid_variation_diagnostics = _operator_grid_variation_diagnostics(
        np.asarray(base_model.field_points, dtype=np.float64).tolist(),
        patches,
    )
    _warn_large_operator_grid_variation(
        grid_variation_diagnostics,
        grid_variation_warning_threshold,
    )
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
        grid_variation_diagnostics=grid_variation_diagnostics,
    )


def prepare_instantaneous_interpolated_effective_model(
    *,
    field_points: Sequence[float | Sequence[float] | np.ndarray],
    transition: transitions.OpticalTransition,
    optical_polarization: couplings.Polarization,
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    master_field: float | Sequence[float] | np.ndarray | None = None,
    keep_diagnostics: bool = True,
    grid_variation_warning_threshold: float | None = 1.0,
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
    grid_variation_diagnostics = _operator_grid_variation_diagnostics(
        unique_sorted.tolist(),
        patches,
    )
    _warn_large_operator_grid_variation(
        grid_variation_diagnostics,
        grid_variation_warning_threshold,
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
        grid_variation_diagnostics=grid_variation_diagnostics,
    )
