from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.effective_hamiltonian._decay import _jump_rate_operator
from centrex_tlf.effective_hamiltonian._superoperators import (
    _dissipator_superoperator,
    _hamiltonian_superoperator,
    _lindblad_dissipator,
    _transform_superoperator,
    _unitary_superoperator,
)
from centrex_tlf.effective_hamiltonian._utility import (
    _dominant_state_index,
    _symmetrize,
)


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
    jump_rate_op = _symmetrize(
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
        jump_rate_operator_override=jump_rate_op,
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
