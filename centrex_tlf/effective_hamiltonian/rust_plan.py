from __future__ import annotations

from typing import Sequence

import numpy as np

from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.ir import lower_parameter_graph
from centrex_tlf.effective_hamiltonian._superoperators import (
    _dissipator_superoperator,
    _hamiltonian_superoperator,
)
from centrex_tlf.effective_hamiltonian.models import (
    PreparedLindbladSafeCompactInterpolatedHamiltonianModel,
)


def _complex_superop_to_split_real(L: np.ndarray) -> np.ndarray:
    n2 = L.shape[0]
    A = np.real(L)
    B = np.imag(L)
    L_real = np.zeros((2 * n2, 2 * n2), dtype=np.float64)
    L_real[:n2, :n2] = A
    L_real[:n2, n2:] = -B
    L_real[n2:, :n2] = B
    L_real[n2:, n2:] = A
    return L_real


def _complex_rho_to_split_real(rho: np.ndarray) -> np.ndarray:
    flat = rho.reshape(-1)
    return np.concatenate([np.real(flat), np.imag(flat)]).astype(np.float64)


def _split_real_to_complex_rho(y: np.ndarray, n_states: int) -> np.ndarray:
    n2 = n_states * n_states
    re = y[:n2]
    im = y[n2:]
    return (re + 1j * im).reshape(n_states, n_states)


def prepare_effective_lindblad_rust_plan(
    model: PreparedLindbladSafeCompactInterpolatedHamiltonianModel,
    parameters: LindbladParameters,
):
    from centrex_tlf.centrex_tlf_rust import prepare_effective_lindblad_plan_py

    n_states = model.n_effective_states
    n2 = n_states * n_states
    real_dim = 2 * n2
    n_grid = len(model.field_points)

    l_combined_list = []
    l_opt_list = []
    l_det_list = []

    excited_indices = np.asarray(model.excited_indices, dtype=np.int64)
    rwa_shift = model.common_omega_reference

    for field_z in model.field_points.tolist():
        bundle = model.effective_bundle(float(field_z))
        h_internal_rwa = np.array(bundle.h_internal, dtype=np.complex128).copy()
        for idx in excited_indices:
            h_internal_rwa[idx, idx] += rwa_shift
        L_int = _hamiltonian_superoperator(h_internal_rwa)
        L_opt = _hamiltonian_superoperator(bundle.h_opt)
        L_det = _hamiltonian_superoperator(bundle.h_det)
        L_diss = bundle.dissipator_superoperator()
        L_combined = L_int + L_diss

        l_combined_list.append(_complex_superop_to_split_real(L_combined).ravel())
        l_opt_list.append(_complex_superop_to_split_real(L_opt).ravel())
        l_det_list.append(_complex_superop_to_split_real(L_det).ravel())

    l_combined = np.concatenate(l_combined_list).astype(np.float64)
    l_opt = np.concatenate(l_opt_list).astype(np.float64)
    l_det = np.concatenate(l_det_list).astype(np.float64)

    mat_size = real_dim * real_dim
    dl_combined_parts = []
    dl_opt_parts = []
    dl_det_parts = []
    for i in range(n_grid - 1):
        base_lo = i * mat_size
        base_hi = (i + 1) * mat_size
        dl_combined_parts.append(l_combined[base_hi:base_hi + mat_size] - l_combined[base_lo:base_lo + mat_size])
        dl_opt_parts.append(l_opt[base_hi:base_hi + mat_size] - l_opt[base_lo:base_lo + mat_size])
        dl_det_parts.append(l_det[base_hi:base_hi + mat_size] - l_det[base_lo:base_lo + mat_size])

    dl_combined = np.concatenate(dl_combined_parts).astype(np.float64) if dl_combined_parts else np.array([], dtype=np.float64)
    dl_opt = np.concatenate(dl_opt_parts).astype(np.float64) if dl_opt_parts else np.array([], dtype=np.float64)
    dl_det = np.concatenate(dl_det_parts).astype(np.float64) if dl_det_parts else np.array([], dtype=np.float64)

    param_graph_payload = lower_parameter_graph(parameters)

    field_sym = None
    rabi_sym = None
    det_sym = None
    slot_names = param_graph_payload["slot_names"]
    for idx, name in enumerate(slot_names):
        if name == "Ez" or name == "field_coordinate":
            field_sym = idx
        if name.startswith("\u03a9") and ("0" in name):
            if rabi_sym is None:
                rabi_sym = idx
        if name.startswith("\u03b4") and ("0" in name):
            if det_sym is None:
                det_sym = idx

    if field_sym is None:
        for idx, name in enumerate(slot_names):
            if "field" in name.lower() or "ez" in name.lower():
                field_sym = idx
                break
    if rabi_sym is None:
        for idx, name in enumerate(slot_names):
            if "\u03a9" in name or "omega" in name.lower() or "rabi" in name.lower():
                rabi_sym = idx
                break
    if det_sym is None:
        for idx, name in enumerate(slot_names):
            if "\u03b4" in name or "delta" in name.lower() or "detuning" in name.lower():
                det_sym = idx
                break

    if field_sym is None:
        raise ValueError("Could not identify field_coordinate slot in parameter graph")
    if rabi_sym is None:
        raise ValueError("Could not identify rabi_rate slot in parameter graph")
    if det_sym is None:
        raise ValueError("Could not identify detuning slot in parameter graph")

    is_time_dependent = True

    plan_dict = {
        "n_states": n_states,
        "real_dim": real_dim,
        "n_grid": n_grid,
        "field_grid": np.asarray(model.field_points, dtype=np.float64),
        "l_combined": l_combined,
        "l_opt": l_opt,
        "l_det": l_det,
        "dl_combined": dl_combined,
        "dl_opt": dl_opt,
        "dl_det": dl_det,
        "excited_indices": np.asarray(model.excited_indices, dtype=np.int64),
        "ground_indices": np.asarray(model.ground_indices, dtype=np.int64),
        "sink_indices": np.asarray(model.sink_indices, dtype=np.int64),
        "field_coordinate_slot": field_sym,
        "rabi_rate_slot": rabi_sym,
        "detuning_slot": det_sym,
        "is_time_dependent": is_time_dependent,
        "parameter_graph": param_graph_payload,
    }

    return prepare_effective_lindblad_plan_py(plan_dict)


def solve_effective_lindblad(
    plan,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    *,
    saveat: np.ndarray | None = None,
    reltol: float = 1e-7,
    abstol: float = 1e-9,
    dt: float = 1e-10,
    save_start: bool = True,
    maxiters: int = 100_000,
):
    from centrex_tlf.centrex_tlf_rust import solve_effective_lindblad_py

    n_states = rho0.shape[0]
    y0 = _complex_rho_to_split_real(rho0)

    times, states = solve_effective_lindblad_py(
        plan,
        y0,
        t_span[0],
        t_span[1],
        float(abstol),
        float(reltol),
        float(dt),
        None if saveat is None else np.asarray(saveat, dtype=np.float64),
        bool(save_start),
        int(maxiters),
    )

    times = np.asarray(times, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    n_points = states.shape[0]
    rho_t = np.array([
        _split_real_to_complex_rho(states[i], n_states) for i in range(n_points)
    ])
    return EffectiveLindbladResult(t=times, rho=rho_t, n_states=n_states)


class EffectiveLindbladResult:
    def __init__(self, t: np.ndarray, rho: np.ndarray, n_states: int):
        self.t = t
        self.rho = rho
        self.n_states = n_states

    def populations(self) -> np.ndarray:
        return np.real(np.diagonal(self.rho, axis1=1, axis2=2))

    def density_matrices(self) -> np.ndarray:
        return self.rho
