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
    operator_interpolation: str = "linear",
):
    from centrex_tlf.centrex_tlf_rust import prepare_effective_lindblad_plan_py

    n_states = model.n_effective_states
    n2 = n_states * n_states
    real_dim = 2 * n2
    n_grid = len(model.field_points)

    l_combined_dense = []
    l_opt_dense = []
    l_det_dense = []

    for field_z in model.field_points.tolist():
        bundle = model.effective_bundle(float(field_z))
        L_int = _hamiltonian_superoperator(bundle.h_internal)
        L_opt = _hamiltonian_superoperator(bundle.h_opt)
        L_det = _hamiltonian_superoperator(bundle.h_det)
        L_diss = bundle.dissipator_superoperator()
        L_combined = L_int + L_diss

        l_combined_dense.append(_complex_superop_to_split_real(L_combined))
        l_opt_dense.append(_complex_superop_to_split_real(L_opt))
        l_det_dense.append(_complex_superop_to_split_real(L_det))

    def _build_sparse_operator(matrices, name, interp=operator_interpolation):
        tol = 1e-15
        union_mask = np.zeros((real_dim, real_dim), dtype=bool)
        for m in matrices:
            union_mask |= np.abs(m) > tol
        row_ptrs = [0]
        col_indices = []
        for i in range(real_dim):
            cols = np.where(union_mask[i])[0]
            col_indices.extend(cols.tolist())
            row_ptrs.append(len(col_indices))
        nnz = len(col_indices)
        density = nnz / (real_dim * real_dim) * 100
        values_per_grid = []
        for m in matrices:
            vals = []
            for i in range(real_dim):
                for j_pos in range(row_ptrs[i], row_ptrs[i + 1]):
                    j = col_indices[j_pos]
                    vals.append(m[i, j])
            values_per_grid.append(np.array(vals, dtype=np.float64))
        grid = np.asarray(model.field_points, dtype=np.float64)
        n_g = len(grid)
        all_coeffs = []
        for entry_idx in range(nnz):
            entry_values = np.array([values_per_grid[g][entry_idx] for g in range(n_g)], dtype=np.float64)
            h = np.diff(grid)
            delta = np.diff(entry_values) / h
            if interp == "pchip":
                d = np.zeros(n_g, dtype=np.float64)
                d[0] = delta[0] if n_g > 1 else 0.0
                d[-1] = delta[-1] if n_g > 1 else 0.0
                for k in range(1, n_g - 1):
                    if delta[k - 1] * delta[k] <= 0.0:
                        d[k] = 0.0
                    else:
                        w1 = 2.0 * h[k] + h[k - 1]
                        w2 = h[k] + 2.0 * h[k - 1]
                        d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
                for k in range(n_g - 1):
                    c0 = entry_values[k]
                    c1 = d[k]
                    c2 = (3.0 * delta[k] - 2.0 * d[k] - d[k + 1]) / h[k]
                    c3 = (d[k] + d[k + 1] - 2.0 * delta[k]) / (h[k] * h[k])
                    all_coeffs.extend([c0, c1, c2, c3])
            else:
                for k in range(n_g - 1):
                    c0 = entry_values[k]
                    c1 = delta[k]
                    all_coeffs.extend([c0, c1, 0.0, 0.0])
        pchip_coeffs_flat = np.zeros((n_g - 1) * nnz * 4, dtype=np.float64)
        for interval in range(n_g - 1):
            for entry_idx in range(nnz):
                src_offset = entry_idx * (n_g - 1) * 4 + interval * 4
                dst_offset = interval * nnz * 4 + entry_idx * 4
                pchip_coeffs_flat[dst_offset:dst_offset + 4] = all_coeffs[src_offset:src_offset + 4]
        return {
            "row_ptrs": np.array(row_ptrs, dtype=np.int64),
            "col_indices": np.array(col_indices, dtype=np.int64),
            "pchip_coeffs": pchip_coeffs_flat,
            "grid": grid,
            "nnz": nnz,
            "density": density,
            "name": name,
        }

    sparse_combined = _build_sparse_operator(l_combined_dense, "L_combined")
    sparse_opt = _build_sparse_operator(l_opt_dense, "L_opt")
    sparse_det = _build_sparse_operator(l_det_dense, "L_det")

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
        "sparse_combined": sparse_combined,
        "sparse_opt": sparse_opt,
        "sparse_det": sparse_det,
        "excited_indices": np.asarray(model.excited_indices, dtype=np.int64),
        "ground_indices": np.asarray(model.ground_indices, dtype=np.int64),
        "sink_indices": np.asarray(model.sink_indices, dtype=np.int64),
        "field_coordinate_slot": field_sym,
        "rabi_rate_slot": rabi_sym,
        "detuning_slot": det_sym,
        "is_time_dependent": is_time_dependent,
        "operator_interpolation": operator_interpolation,
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
    solver: str = "dopri5",
    output: str = "full",
    output_indices: list[tuple[int, int]] | None = None,
    output_when: str = "saveat",
    integral_weights: list[tuple[int, float]] | None = None,
):
    from centrex_tlf.centrex_tlf_rust import solve_effective_lindblad_py

    n_states = rho0.shape[0]
    y0 = _complex_rho_to_split_real(rho0)

    effective_saveat = saveat
    effective_save_start = save_start
    if output_when == "final":
        effective_saveat = np.array([t_span[1]], dtype=np.float64)
        effective_save_start = False

    times, values, width = solve_effective_lindblad_py(
        plan,
        y0,
        t_span[0],
        t_span[1],
        float(abstol),
        float(reltol),
        float(dt),
        None if effective_saveat is None else np.asarray(effective_saveat, dtype=np.float64),
        bool(effective_save_start),
        int(maxiters),
        solver,
        output,
        output_indices,
        integral_weights,
    )

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values)

    if output in ("weighted_integral", "photon_integral", "excited_population"):
        return EffectiveLindbladResult(t=times, rho=values, n_states=n_states)

    if output == "full":
        n_points = len(times)
        states = np.asarray(values, dtype=np.float64)
        rho_t = np.array([
            _split_real_to_complex_rho(states[i * plan.real_dim:(i + 1) * plan.real_dim], n_states)
            for i in range(n_points)
        ])
        return EffectiveLindbladResult(t=times, rho=rho_t, n_states=n_states)

    values_arr = np.asarray(values, dtype=np.float64 if output == "populations" else np.complex128)
    if times.size > 0 and width > 0:
        values_arr = values_arr.reshape((times.size, int(width)))
    return EffectiveLindbladResult(t=times, rho=values_arr, n_states=n_states)


class EffectiveLindbladResult:
    def __init__(self, t: np.ndarray, rho: np.ndarray, n_states: int):
        self.t = t
        self.rho = rho
        self.n_states = n_states

    def populations(self) -> np.ndarray:
        return np.real(np.diagonal(self.rho, axis1=1, axis2=2))

    def density_matrices(self) -> np.ndarray:
        return self.rho


from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from centrex_tlf.lindblad.parameters import Parameter

ParameterSlot = str | Parameter


@dataclass
class EffectiveLindbladBatchResult:
    t: np.ndarray
    values: np.ndarray
    output: str
    trajectory_count: int
    parameter_slots: list[str] | None = None
    parameter_values: np.ndarray | None = None
    solver_stats: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parameter_slot_name(slot: ParameterSlot) -> str:
    if isinstance(slot, Parameter):
        return slot.name
    return str(slot)


def _parameter_slot_indices(
    plan,
    parameter_slots: Sequence[ParameterSlot] | None,
) -> list[int]:
    if parameter_slots is None:
        return []
    slot_names = plan.slot_names
    n_base = plan.n_grid  # base params are first n_base slots (approximation)
    indices = []
    for slot in parameter_slots:
        name = _parameter_slot_name(slot)
        try:
            index = slot_names.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown parameter slot {name!r}") from exc
        indices.append(index)
    return indices


def _normalize_saveat_effective(
    saveat: None | float | Sequence[float] | np.ndarray,
    t_span: tuple[float, float],
    save_start: bool,
) -> np.ndarray | None:
    if saveat is None:
        return None
    if isinstance(saveat, (float, int, np.floating, np.integer)):
        step = float(saveat)
        if step <= 0:
            raise ValueError("saveat step must be positive")
        values = np.arange(t_span[0], t_span[1] + 0.5 * step, step, dtype=np.float64)
    else:
        values = np.asarray(saveat, dtype=np.float64)
    if not save_start and values.size > 0 and np.isclose(values[0], t_span[0]):
        values = values[1:]
    return values


def _call_batch_rust(
    plan,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    slot_indices: list[int],
    parameter_batch: np.ndarray | None,
    trajectory_count: int,
    solver: str,
    saveat: np.ndarray | None,
    save_start: bool,
    abstol: float,
    reltol: float,
    dt: float,
    maxiters: int,
    output: str,
    output_when: str,
    parallel: bool,
    threads: int | None,
) -> tuple[np.ndarray, np.ndarray, int, int, dict]:
    from centrex_tlf.centrex_tlf_rust import solve_effective_lindblad_batch_py

    effective_saveat = saveat
    effective_save_start = save_start
    if output_when == "final":
        effective_saveat = np.array([t_span[1]], dtype=np.float64)
        effective_save_start = False

    times, flat_values, width, time_count, solver_stats = solve_effective_lindblad_batch_py(
        plan,
        y0,
        t_span[0],
        t_span[1],
        float(abstol),
        float(reltol),
        float(dt),
        None if effective_saveat is None else np.asarray(effective_saveat, dtype=np.float64),
        bool(effective_save_start),
        int(maxiters),
        solver,
        output,
        slot_indices if slot_indices else None,
        parameter_batch,
        trajectory_count,
        bool(parallel),
        threads,
    )

    times_array = np.asarray(times, dtype=np.float64)
    values_array = np.asarray(flat_values, dtype=np.float64)
    if output_when == "final":
        values_array = values_array.reshape((trajectory_count, int(width)))
    else:
        values_array = values_array.reshape((trajectory_count, int(time_count), int(width)))

    return times_array, values_array, width, time_count, dict(solver_stats)


def parameter_scan(
    plan,
    rho0: np.ndarray,
    t_span: Sequence[float],
    *,
    parameter_slots: Sequence[ParameterSlot],
    parameter_batch: np.ndarray,
    solver: str = "dopri5",
    abstol: float = 1e-8,
    reltol: float = 1e-6,
    dt: float = 1e-10,
    saveat: None | float | Sequence[float] | np.ndarray = None,
    save_start: bool = True,
    maxiters: int = 100_000,
    output: str = "populations",
    output_when: str = "final",
    parallel: bool = True,
    threads: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EffectiveLindbladBatchResult:
    if output not in {"populations", "full"}:
        raise NotImplementedError("effective Hamiltonian batch supports output='populations' or 'full'")
    if output_when not in {"final", "saveat"}:
        raise ValueError("output_when must be 'final' or 'saveat'")
    if output_when == "saveat" and saveat is None:
        raise ValueError("output_when='saveat' requires explicit saveat values")
    if threads is not None and threads <= 0:
        raise ValueError("threads must be positive when provided")

    t_span_tuple = (float(t_span[0]), float(t_span[1]))
    y0 = _complex_rho_to_split_real(rho0)
    values = np.asarray(parameter_batch, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("parameter_batch must be 2D (n_trajectories, n_parameters)")
    trajectory_count = values.shape[0]
    slot_indices = _parameter_slot_indices(plan, parameter_slots)
    parameter_slot_names = [_parameter_slot_name(s) for s in parameter_slots]
    saveat_values = _normalize_saveat_effective(saveat, t_span_tuple, save_start)

    times, vals, width, time_count, stats = _call_batch_rust(
        plan, y0, t_span_tuple,
        slot_indices=slot_indices,
        parameter_batch=np.ascontiguousarray(values),
        trajectory_count=trajectory_count,
        solver=solver, saveat=saveat_values, save_start=save_start,
        abstol=abstol, reltol=reltol, dt=dt, maxiters=maxiters,
        output=output, output_when=output_when,
        parallel=parallel, threads=threads,
    )

    return EffectiveLindbladBatchResult(
        t=times,
        values=vals,
        output=output,
        trajectory_count=trajectory_count,
        parameter_slots=parameter_slot_names,
        parameter_values=values,
        solver_stats=stats,
        metadata={} if metadata is None else dict(metadata),
    )


def grid_scan(
    plan,
    rho0: np.ndarray,
    t_span: Sequence[float],
    *,
    scan: Mapping[ParameterSlot, Sequence[float] | np.ndarray],
    solver: str = "dopri5",
    abstol: float = 1e-8,
    reltol: float = 1e-6,
    dt: float = 1e-10,
    saveat: None | float | Sequence[float] | np.ndarray = None,
    save_start: bool = True,
    maxiters: int = 100_000,
    output: str = "populations",
    output_when: str = "final",
    parallel: bool = True,
    threads: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EffectiveLindbladBatchResult:
    if not scan:
        raise ValueError("scan must contain at least one parameter")
    if output not in {"populations", "full"}:
        raise NotImplementedError("effective Hamiltonian batch supports output='populations' or 'full'")
    if output_when not in {"final", "saveat"}:
        raise ValueError("output_when must be 'final' or 'saveat'")
    if output_when == "saveat" and saveat is None:
        raise ValueError("output_when='saveat' requires explicit saveat values")

    t_span_tuple = (float(t_span[0]), float(t_span[1]))
    y0 = _complex_rho_to_split_real(rho0)
    parameter_slots = list(scan.keys())
    axes = [np.asarray(values, dtype=np.float64).reshape(-1) for values in scan.values()]
    if any(axis.size == 0 for axis in axes):
        raise ValueError("scan axes must be non-empty")

    grid = np.meshgrid(*axes, indexing="ij")
    flat_grid = np.column_stack([g.ravel() for g in grid])
    trajectory_count = flat_grid.shape[0]
    grid_shape = tuple(axis.size for axis in axes)

    slot_indices = _parameter_slot_indices(plan, parameter_slots)
    parameter_slot_names = [_parameter_slot_name(s) for s in parameter_slots]
    saveat_values = _normalize_saveat_effective(saveat, t_span_tuple, save_start)

    times, vals, width, time_count, stats = _call_batch_rust(
        plan, y0, t_span_tuple,
        slot_indices=slot_indices,
        parameter_batch=np.ascontiguousarray(flat_grid),
        trajectory_count=trajectory_count,
        solver=solver, saveat=saveat_values, save_start=save_start,
        abstol=abstol, reltol=reltol, dt=dt, maxiters=maxiters,
        output=output, output_when=output_when,
        parallel=parallel, threads=threads,
    )

    result = EffectiveLindbladBatchResult(
        t=times,
        values=vals,
        output=output,
        trajectory_count=trajectory_count,
        parameter_slots=parameter_slot_names,
        parameter_values=flat_grid,
        solver_stats=stats,
        metadata={} if metadata is None else dict(metadata),
    )
    result.metadata.update({
        "scan_kind": "grid",
        "grid_shape": grid_shape,
        "grid_axes": {name: axis for name, axis in zip(parameter_slot_names, axes)},
    })
    return result
