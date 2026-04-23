from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from centrex_tlf.effective_hamiltonian._utility import _as_field_vector, _parameter_at_time
from centrex_tlf.effective_hamiltonian._superoperators import _hamiltonian_superoperator
from centrex_tlf.effective_hamiltonian.operator_bundle import OperatorBundle
from centrex_tlf.effective_hamiltonian.models import (
    PreparedLindbladSafeCompactInterpolatedHamiltonianModel,
    PreparedInstantaneousInterpolatedEffectiveHamiltonianModel,
)
from centrex_tlf.effective_hamiltonian.initial_state import default_effective_density_matrix


def solve_density_matrix_model(
    bundle_evaluator: Callable[[float], OperatorBundle],
    *,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    rabi_rate: float | complex | Callable[[float], float | complex],
    detuning: float | Callable[[float], float] = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
):
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
):
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
):
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
):
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
):
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
):
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
