from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import sympy as smp

from .parameters import Parameter
from .plan_static import PreparedLindbladProblem

__all__ = [
    "LindbladBatchResult",
    "grid_scan",
    "initial_condition_scan",
    "parameter_scan",
    "solve_lindblad_batch",
]


ParameterSlot = str | smp.Symbol | Parameter


@dataclass
class LindbladBatchResult:
    t: npt.NDArray[np.float64]
    values: npt.NDArray[Any]
    output: str
    output_indices: list[tuple[int, int]] | None
    trajectory_count: int
    parameter_slots: list[str] | None = None
    parameter_values: npt.NDArray[np.complex128] | None = None
    solver_stats: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_t_span(t_span: Sequence[float]) -> tuple[float, float]:
    if len(t_span) != 2:
        raise ValueError("t_span must contain exactly two values")
    return (float(t_span[0]), float(t_span[1]))


def _normalize_saveat(
    saveat: None | float | Sequence[float] | npt.NDArray[np.floating],
    t_span: tuple[float, float],
    save_start: bool,
) -> None | np.ndarray:
    if saveat is None:
        return None
    if isinstance(saveat, float | int | np.floating | np.integer):
        step = float(saveat)
        if step <= 0:
            raise ValueError("saveat step must be positive")
        values = np.arange(t_span[0], t_span[1] + 0.5 * step, step, dtype=np.float64)
    else:
        values = np.asarray(saveat, dtype=np.float64)
    if not save_start and values.size > 0 and np.isclose(values[0], t_span[0]):
        values = values[1:]
    return values


def _pack_rho0_batch(prepared: PreparedLindbladProblem, rho0_batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(rho0_batch)
    if batch.ndim == 2:
        packed = np.asarray(batch, dtype=np.float64)
        if packed.shape[1] != prepared.layout.packed_len:
            raise ValueError(
                "2D rho0_batch must have shape (n_trajectories, packed_len); "
                f"expected packed_len={prepared.layout.packed_len}, got {packed.shape[1]}"
            )
        return np.ascontiguousarray(packed)
    if batch.ndim != 3:
        raise ValueError(
            "rho0_batch must have shape (n_trajectories, packed_len) or "
            "(n_trajectories, n_states, n_states)"
        )
    if batch.shape[1:] != (prepared.layout.n, prepared.layout.n):
        raise ValueError(
            "matrix rho0_batch must have shape "
            f"(n_trajectories, {prepared.layout.n}, {prepared.layout.n})"
        )
    packed = np.empty((batch.shape[0], prepared.layout.packed_len), dtype=np.float64)
    for idx, rho in enumerate(np.asarray(batch, dtype=np.complex128)):
        packed[idx] = prepared.layout.pack(rho)
    return np.ascontiguousarray(packed)


def _parameter_slot_indices(
    prepared: PreparedLindbladProblem,
    parameter_slots: Sequence[ParameterSlot] | None,
) -> list[int]:
    if parameter_slots is None:
        return []
    slot_names = list(prepared.parameter_graph["slot_names"])
    base_count = len(prepared.parameter_graph["base_values"])
    indices = []
    for slot in parameter_slots:
        name = _parameter_slot_name(slot)
        try:
            index = slot_names.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown parameter slot {name!r}") from exc
        if index >= base_count:
            raise ValueError(
                f"parameter slot {name!r} is a compound parameter; only base parameters can be scanned"
            )
        indices.append(index)
    return indices


def _parameter_slot_name(slot: ParameterSlot) -> str:
    if isinstance(slot, Parameter):
        return slot.name
    return str(slot)


def _parameter_slot_names(parameter_slots: Sequence[ParameterSlot] | None) -> list[str] | None:
    if parameter_slots is None:
        return None
    return [_parameter_slot_name(slot) for slot in parameter_slots]


def _solver_stats_dict(
    solver_stats: Any,
    *,
    solver: str,
    saved_points: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    stats = dict(solver_stats)
    stats["solver"] = solver
    stats.setdefault("function_evaluations", stats.get("rhs_calls", 0))
    stats.setdefault("saved_points", saved_points)
    stats.setdefault("rhs_seconds", elapsed_seconds)
    stats.setdefault("total_seconds", elapsed_seconds)
    return stats


def solve_lindblad_batch(
    prepared: PreparedLindbladProblem,
    rho0_batch: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
    t_span: Sequence[float],
    *,
    parameter_batch: npt.NDArray[np.complex128] | None = None,
    parameter_slots: Sequence[ParameterSlot] | None = None,
    solver: str = "dopri5",
    execution_mode: str = "expanded_sparse",
    abstol: float = 1e-7,
    reltol: float = 1e-4,
    dt: float = 1e-8,
    saveat: None | float | Sequence[float] | npt.NDArray[np.floating] = None,
    save_start: bool = True,
    maxiters: int = 100_000,
    collect_stats: bool = False,
    output: str = "populations",
    output_indices: Sequence[tuple[int, int]] | None = None,
    output_when: str = "final",
    integral_weights: Sequence[tuple[int, float]] | None = None,
    dense_output: bool = True,
    parallel: bool = True,
    threads: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LindbladBatchResult:
    if prepared.rust_plan is None:
        raise RuntimeError("solve_lindblad_batch requires a Rust prepared plan")
    if solver not in {"dopri5", "tsit5"}:
        raise NotImplementedError("batch solving currently supports 'dopri5' and 'tsit5'")
    integral_outputs = {"weighted_integral", "photon_integral", "excited_population"}
    if output not in {"populations", "selected", *integral_outputs}:
        raise NotImplementedError(
            "batch solving currently supports output='populations', 'selected', "
            "'weighted_integral', 'photon_integral', or 'excited_population'"
        )
    if output == "selected" and output_indices is None:
        raise ValueError("output='selected' requires output_indices")
    if output != "selected" and output_indices is not None:
        raise ValueError("output_indices is only valid with output='selected'")
    if output in integral_outputs and integral_weights is None:
        raise ValueError(f"output={output!r} requires integral_weights")
    if output not in integral_outputs and integral_weights is not None:
        raise ValueError("integral_weights are only valid with integral output modes")
    if output_when not in {"final", "saveat"}:
        raise ValueError("output_when must be 'final' or 'saveat'")
    if output_when == "saveat" and saveat is None:
        raise ValueError("batch output_when='saveat' requires explicit saveat values")
    if output in integral_outputs and saveat is None:
        raise ValueError(f"output={output!r} requires explicit saveat values")
    if threads is not None and threads <= 0:
        raise ValueError("threads must be positive when provided")

    t_span_tuple = _normalize_t_span(t_span)
    packed_batch = _pack_rho0_batch(prepared, np.asarray(rho0_batch))
    trajectory_count = packed_batch.shape[0]
    if trajectory_count == 0:
        raise ValueError("rho0_batch must contain at least one trajectory")

    slot_indices = _parameter_slot_indices(prepared, parameter_slots)
    parameter_values = None
    if parameter_batch is not None:
        if parameter_slots is None:
            raise ValueError("parameter_slots are required when parameter_batch is provided")
        parameter_values = np.ascontiguousarray(parameter_batch, dtype=np.complex128)
        if parameter_values.shape != (trajectory_count, len(slot_indices)):
            raise ValueError(
                "parameter_batch must have shape "
                f"({trajectory_count}, {len(slot_indices)}), got {parameter_values.shape}"
            )
    elif parameter_slots is not None:
        raise ValueError("parameter_slots were provided without parameter_batch")

    saveat_values = _normalize_saveat(saveat, t_span_tuple, save_start)
    if not dense_output and output_when == "saveat" and saveat_values is not None:
        raise ValueError("dense_output=False is only supported with output_when='final'")

    from ..centrex_tlf_rust import solve_lindblad_batch_ode_py

    start = time.perf_counter()
    times, flat_values, width, time_count, solver_stats = solve_lindblad_batch_ode_py(
        prepared.rust_plan,
        packed_batch,
        t_span_tuple[0],
        t_span_tuple[1],
        float(abstol),
        float(reltol),
        float(dt),
        None if saveat_values is None else np.asarray(saveat_values, dtype=np.float64),
        bool(save_start),
        int(maxiters),
        execution_mode,
        solver,
        output,
        None if output_indices is None else list(output_indices),
        output_when,
        None if integral_weights is None else list(integral_weights),
        slot_indices or None,
        parameter_values,
        bool(parallel),
        threads,
    )
    elapsed = time.perf_counter() - start

    dtype = np.float64 if output in {"populations", *integral_outputs} else np.complex128
    values = np.asarray(flat_values, dtype=dtype)
    if output_when == "final":
        values = values.reshape((trajectory_count, int(width)))
    else:
        values = values.reshape((trajectory_count, int(time_count), int(width)))

    return LindbladBatchResult(
        t=np.asarray(times, dtype=np.float64),
        values=values,
        output=output,
        output_indices=None if output_indices is None else list(output_indices),
        trajectory_count=trajectory_count,
        parameter_slots=_parameter_slot_names(parameter_slots),
        parameter_values=parameter_values,
        solver_stats=(
            _solver_stats_dict(
                solver_stats,
                solver=solver,
                saved_points=int(time_count),
                elapsed_seconds=elapsed,
            )
            if collect_stats
            else None
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def initial_condition_scan(
    prepared: PreparedLindbladProblem,
    rho0_batch: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
    t_span: Sequence[float],
    **kwargs: Any,
) -> LindbladBatchResult:
    return solve_lindblad_batch(prepared, rho0_batch, t_span, **kwargs)


def parameter_scan(
    prepared: PreparedLindbladProblem,
    rho0: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
    t_span: Sequence[float],
    *,
    parameter_slots: Sequence[ParameterSlot],
    parameter_batch: npt.NDArray[np.complex128],
    **kwargs: Any,
) -> LindbladBatchResult:
    values = np.asarray(parameter_batch, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("parameter_batch must be 2D")
    rho = np.asarray(rho0)
    if rho.ndim == 1:
        packed = np.asarray(rho, dtype=np.float64)
    elif rho.ndim == 2:
        packed = prepared.layout.pack(np.asarray(rho, dtype=np.complex128))
    else:
        raise ValueError("rho0 must be a packed vector or density matrix")
    rho0_batch = np.repeat(packed.reshape(1, -1), values.shape[0], axis=0)
    return solve_lindblad_batch(
        prepared,
        rho0_batch,
        t_span,
        parameter_slots=parameter_slots,
        parameter_batch=values,
        **kwargs,
    )


def grid_scan(
    prepared: PreparedLindbladProblem,
    rho0: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
    t_span: Sequence[float],
    *,
    scan: Mapping[ParameterSlot, Sequence[complex] | npt.NDArray[np.complexfloating]],
    **kwargs: Any,
) -> LindbladBatchResult:
    if not scan:
        raise ValueError("scan must contain at least one parameter")
    parameter_slots = list(scan)
    parameter_slot_names = _parameter_slot_names(parameter_slots)
    axes = [np.asarray(values, dtype=np.complex128).reshape(-1) for values in scan.values()]
    if any(axis.size == 0 for axis in axes):
        raise ValueError("scan axes must be non-empty")
    slot_indices = _parameter_slot_indices(prepared, parameter_slots)
    rho = np.asarray(rho0)
    if rho.ndim == 1:
        packed = np.asarray(rho, dtype=np.float64)
        if packed.size != prepared.layout.packed_len:
            raise ValueError(
                f"packed rho0 must have length {prepared.layout.packed_len}, got {packed.size}"
            )
    elif rho.ndim == 2:
        packed = prepared.layout.pack(np.asarray(rho, dtype=np.complex128))
    else:
        raise ValueError("rho0 must be a packed vector or density matrix")

    t_span_tuple = _normalize_t_span(t_span)
    save_start = bool(kwargs.pop("save_start", True))
    saveat_values = _normalize_saveat(kwargs.pop("saveat", None), t_span_tuple, save_start)
    solver = kwargs.pop("solver", "dopri5")
    execution_mode = kwargs.pop("execution_mode", "expanded_sparse")
    abstol = float(kwargs.pop("abstol", 1e-7))
    reltol = float(kwargs.pop("reltol", 1e-4))
    dt = float(kwargs.pop("dt", 1e-8))
    maxiters = int(kwargs.pop("maxiters", 100_000))
    collect_stats = bool(kwargs.pop("collect_stats", False))
    output = kwargs.pop("output", "populations")
    output_indices = kwargs.pop("output_indices", None)
    output_when = kwargs.pop("output_when", "final")
    integral_weights = kwargs.pop("integral_weights", None)
    dense_output = bool(kwargs.pop("dense_output", True))
    parallel = bool(kwargs.pop("parallel", True))
    threads = kwargs.pop("threads", None)
    metadata = kwargs.pop("metadata", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected grid_scan keyword argument(s): {unknown}")
    if prepared.rust_plan is None:
        raise RuntimeError("grid_scan requires a Rust prepared plan")
    if solver not in {"dopri5", "tsit5"}:
        raise NotImplementedError("grid_scan currently supports 'dopri5' and 'tsit5'")
    integral_outputs = {"weighted_integral", "photon_integral", "excited_population"}
    if output not in {"populations", "selected", *integral_outputs}:
        raise NotImplementedError(
            "grid_scan currently supports output='populations', 'selected', "
            "'weighted_integral', 'photon_integral', or 'excited_population'"
        )
    if output == "selected" and output_indices is None:
        raise ValueError("output='selected' requires output_indices")
    if output != "selected" and output_indices is not None:
        raise ValueError("output_indices is only valid with output='selected'")
    if output in integral_outputs and integral_weights is None:
        raise ValueError(f"output={output!r} requires integral_weights")
    if output not in integral_outputs and integral_weights is not None:
        raise ValueError("integral_weights are only valid with integral output modes")
    if output_when == "saveat" and saveat_values is None:
        raise ValueError("grid_scan output_when='saveat' requires explicit saveat values")
    if output in integral_outputs and saveat_values is None:
        raise ValueError(f"output={output!r} requires explicit saveat values")
    if not dense_output and output_when == "saveat" and saveat_values is not None:
        raise ValueError("dense_output=False is only supported with output_when='final'")
    if threads is not None and threads <= 0:
        raise ValueError("threads must be positive when provided")

    axis_lengths = [int(axis.size) for axis in axes]
    trajectory_count = int(np.prod(axis_lengths, dtype=np.int64))
    flat_axes = np.ascontiguousarray(np.concatenate(axes), dtype=np.complex128)

    from ..centrex_tlf_rust import solve_lindblad_grid_ode_py

    start = time.perf_counter()
    times, flat_values, width, time_count, solver_stats = (
        solve_lindblad_grid_ode_py(
            prepared.rust_plan,
            np.ascontiguousarray(packed, dtype=np.float64),
            t_span_tuple[0],
            t_span_tuple[1],
            abstol,
            reltol,
            dt,
            slot_indices,
            flat_axes,
            axis_lengths,
            None if saveat_values is None else np.asarray(saveat_values, dtype=np.float64),
            save_start,
            maxiters,
            execution_mode,
            solver,
            output,
            None if output_indices is None else list(output_indices),
            output_when,
            None if integral_weights is None else list(integral_weights),
            parallel,
            threads,
        )
    )
    elapsed = time.perf_counter() - start
    dtype = np.float64 if output in {"populations", *integral_outputs} else np.complex128
    values = np.asarray(flat_values, dtype=dtype)
    if output_when == "final":
        values = values.reshape((trajectory_count, int(width)))
    else:
        values = values.reshape((trajectory_count, int(time_count), int(width)))

    result = LindbladBatchResult(
        t=np.asarray(times, dtype=np.float64),
        values=values,
        output=output,
        output_indices=None if output_indices is None else list(output_indices),
        trajectory_count=trajectory_count,
        parameter_slots=parameter_slot_names,
        parameter_values=None,
        solver_stats=(
            _solver_stats_dict(
                solver_stats,
                solver=solver,
                saved_points=int(time_count),
                elapsed_seconds=elapsed,
            )
            if collect_stats
            else None
        ),
        metadata={} if metadata is None else dict(metadata),
    )
    result.metadata.update(
        {
            "scan_kind": "grid",
            "grid_shape": tuple(int(axis.size) for axis in axes),
            "grid_axes": {
                name: axis for name, axis in zip(parameter_slot_names or [], axes, strict=True)
            },
            "compact_grid": True,
        }
    )
    return result
