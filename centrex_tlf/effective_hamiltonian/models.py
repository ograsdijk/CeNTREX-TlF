from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from centrex_tlf import couplings, states, transitions
from centrex_tlf.effective_hamiltonian._utility import (
    _as_field_vector,
    _psd_project,
    _symmetrize,
)
from centrex_tlf.effective_hamiltonian._decay import (
    _c_array_from_full_recycling_decay_kernel,
)
from centrex_tlf.effective_hamiltonian.operator_bundle import OperatorBundle


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
    grid_variation_diagnostics: dict[str, object] | None = None

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
                            np.asarray(patch.bundle.decay_kernels_sinks[sink_index], dtype=np.complex128)
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
    grid_variation_diagnostics: dict[str, object] | None = None

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
    grid_variation_diagnostics: dict[str, object] | None = None

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
