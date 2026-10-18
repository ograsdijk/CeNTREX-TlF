"""Opt-in native adjoint responses for structured Lindblad plans.

An adjoint response evaluates linear terminal and running observables for many
initial density matrices after one reverse-time solve. This does not compute or
return forward density trajectories.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .batch import _parameter_slot_name
from .parameters import Parameter
from .plan_static import PreparedLindbladProblem

__all__ = [
    "AdjointResponse",
    "population_observable",
    "prepare_lindblad_adjoint_response",
    "prepare_lindblad_adjoint_response_batch",
]


def population_observable(
    n_states: int, indices: Sequence[int], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Build a diagonal terminal observable for a selected population/group."""
    if not isinstance(n_states, int) or n_states < 1:
        raise ValueError("n_states must be a positive integer")
    raw_idx = np.asarray(indices)
    if raw_idx.ndim != 1 or raw_idx.size == 0 or not np.issubdtype(raw_idx.dtype, np.integer):
        raise ValueError("indices must be a nonempty 1D sequence of integers")
    idx = raw_idx.astype(np.int64, copy=False)
    if np.any(idx < 0) or np.any(idx >= n_states):
        raise ValueError("indices must be a nonempty 1D list of valid state indices")
    values = np.ones(idx.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if values.shape != idx.shape or not np.isfinite(values).all():
        raise ValueError("weights must be finite and have one value per index")
    out = np.zeros((n_states, n_states), dtype=np.complex128)
    np.add.at(out, (idx, idx), values)
    return out


@dataclass(frozen=True)
class AdjointResponse:
    """Hermitian adjoint matrices keyed by objective name."""

    responses: Mapping[str, npt.NDArray[np.complex128]]
    solver_stats: Mapping[str, Mapping[str, Any]]
    t_span: tuple[float, float]

    @property
    def n_states(self) -> int:
        return next(iter(self.responses.values())).shape[0]

    def basis_responses(self, objective: str) -> npt.NDArray[np.float64]:
        """Return one response for each initially occupied basis state."""
        try:
            return np.asarray(self.responses[objective].diagonal().real).copy()
        except KeyError as exc:
            raise KeyError(f"unknown objective {objective!r}") from exc

    def evaluate(self, rho0: Any, objective: str | None = None) -> Any:
        """Evaluate the response for a coherent density matrix or a matrix batch."""
        rho = np.asarray(rho0, dtype=np.complex128)
        n = self.n_states
        if rho.shape not in {(n, n)} and not (rho.ndim == 3 and rho.shape[1:] == (n, n)):
            raise ValueError(f"rho0 must have shape ({n}, {n}) or (batch, {n}, {n})")
        if not np.isfinite(rho).all() or not np.allclose(rho, rho.swapaxes(-1, -2).conj(), rtol=1e-11, atol=1e-12):
            raise ValueError("rho0 must be finite and Hermitian")
        selected = self.responses if objective is None else {objective: self.responses[objective]}
        result: dict[str, Any] = {}
        for name, matrix in selected.items():
            values = np.einsum("ij,...ji->...", matrix, rho)
            if np.max(np.abs(np.asarray(values).imag), initial=0.0) > 1e-9 * max(1.0, float(np.max(np.abs(values.real), initial=0.0))):
                raise ArithmeticError("Hermitian adjoint evaluation produced a non-real response")
            real = np.asarray(values.real)
            result[name] = float(real) if real.ndim == 0 else real
        return result if objective is None else result[objective]


def _parameter_overrides(
    prepared: PreparedLindbladProblem, values: Mapping[Any, complex] | None
) -> tuple[list[int], list[complex]]:
    if values is None:
        return [], []
    names = list(prepared.parameter_graph["slot_names"])
    base_count = len(prepared.parameter_graph["base_values"])
    slots: list[int] = []
    result: list[complex] = []
    seen: set[str] = set()
    for key, value in values.items():
        name = _parameter_slot_name(key) if isinstance(key, Parameter) else str(key)
        if name in seen:
            raise ValueError(f"duplicate parameter override {name!r}")
        seen.add(name)
        try:
            idx = names.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown parameter slot {name!r}") from exc
        if idx >= base_count:
            raise ValueError(f"parameter slot {name!r} is compound and cannot be overridden")
        scalar = complex(value)
        if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
            raise ValueError(f"parameter override {name!r} must be finite")
        slots.append(idx)
        result.append(scalar)
    return slots, result


def _terminal_matrix(value: Any, n: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (n, n):
        raise ValueError(f"terminal observable {name!r} must be an ({n}, {n}) matrix")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.conj().T, rtol=1e-11, atol=1e-12):
        raise ValueError(f"terminal observable {name!r} must be finite and Hermitian")
    return np.ascontiguousarray(0.5 * (matrix + matrix.conj().T))


def prepare_lindblad_adjoint_response(
    prepared: PreparedLindbladProblem,
    t_span: Sequence[float],
    *,
    terminal_observables: Mapping[str, Any] | None = None,
    integral_weights: Sequence[tuple[int, float]] | None = None,
    integral_name: str = "integral",
    discontinuities: Sequence[float] | None = None,
    solver: str = "dopri5",
    abstol: float = 1e-10,
    reltol: float = 1e-8,
    dt: float = 1e-9,
    maxiters: int = 100000,
    parameter_values: Mapping[Any, complex] | None = None,
    parallel: bool = False,
    threads: int | None = None,
) -> AdjointResponse:
    """Prepare named terminal and running-observable responses.

    ``integral_weights`` defines a separate running objective ``integral_name``
    equal to the forward integral of the weighted populations. It is not added
    implicitly to any terminal objective. ``terminal_observables`` values are
    Hermitian matrices; use :func:`population_observable` for population groups.
    ``discontinuities`` lists exact absolute-time edges at which integration is
    split, e.g. for switched drives.
    """
    if prepared.backend != "rust" or prepared.rust_plan is None:
        raise ValueError("adjoint responses require a Rust-prepared Lindblad problem")
    if len(t_span) != 2:
        raise ValueError("t_span must contain two values")
    t0, t1 = map(float, t_span)
    if not np.isfinite([t0, t1]).all() or t1 <= t0:
        raise ValueError("t_span must be finite with t1 > t0")
    edges = [] if discontinuities is None else list(map(float, discontinuities))
    if any(not np.isfinite(x) or x <= t0 or x >= t1 for x in edges):
        raise ValueError("discontinuities must lie strictly inside t_span")
    if len(set(edges)) != len(edges):
        raise ValueError("discontinuities must be unique")
    n = prepared.layout.n
    objectives: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, value in (terminal_observables or {}).items():
        key = str(name)
        if not key or key in objectives:
            raise ValueError("objective names must be nonempty and unique")
        objectives[key] = (_terminal_matrix(value, n, key), np.zeros(n, dtype=np.float64))
    if integral_weights is not None:
        if not integral_name or integral_name in objectives:
            raise ValueError("integral_name must be nonempty and distinct from terminal objective names")
        weights = np.zeros(n, dtype=np.float64)
        for idx, weight in integral_weights:
            if not isinstance(idx, int | np.integer) or idx < 0 or idx >= n:
                raise ValueError(f"integral weight index {idx!r} is outside [0, {n})")
            if not np.isfinite(weight):
                raise ValueError("integral weights must be finite")
            weights[int(idx)] += float(weight)
        objectives[integral_name] = (np.zeros((n, n), dtype=np.complex128), weights)
    if not objectives:
        raise ValueError("provide terminal_observables and/or integral_weights")
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ValueError("threads must be a positive integer")
    slots, values = _parameter_overrides(prepared, parameter_values)
    from ..centrex_tlf_rust import solve_lindblad_adjoint_py

    def run(item: tuple[str, tuple[np.ndarray, np.ndarray]]) -> tuple[str, np.ndarray, dict[str, Any]]:
        name, (terminal, weights) = item
        response, stats = solve_lindblad_adjoint_py(
            prepared.rust_plan,
            terminal,
            weights.tolist(),
            t0,
            t1,
            edges,
            abstol,
            reltol,
            dt,
            maxiters,
            solver,
            slots,
            values,
        )
        return name, np.asarray(response, dtype=np.complex128), dict(stats)

    items = list(objectives.items())
    if parallel and len(items) > 1:
        with ThreadPoolExecutor(max_workers=min(len(items), threads or 4)) as pool:
            outputs = list(pool.map(run, items))
    else:
        outputs = [run(item) for item in items]
    return AdjointResponse(
        responses={name: matrix for name, matrix, _ in outputs},
        solver_stats={name: stats for name, _, stats in outputs},
        t_span=(t0, t1),
    )


def prepare_lindblad_adjoint_response_batch(
    prepared: PreparedLindbladProblem,
    t_span: Sequence[float],
    parameter_values_batch: Sequence[Mapping[Any, complex]],
    *,
    terminal_observables: Mapping[str, Any] | None = None,
    integral_weights: Sequence[tuple[int, float]] | None = None,
    integral_name: str = "integral",
    discontinuities: Sequence[float] | None = None,
    solver: str = "dopri5",
    abstol: float = 1e-10,
    reltol: float = 1e-8,
    dt: float = 1e-9,
    maxiters: int = 100000,
    parallel: bool = True,
    threads: int | None = None,
) -> list[AdjointResponse]:
    """Solve responses at multiple independent parameter points.

    Parallelism is across parameter points, capped at four workers by default
    (or the explicit ``threads`` value). Each native solve releases the GIL;
    objective solves remain serial within each point to avoid nested pools.
    Results preserve the input parameter-point order.
    """
    points = list(parameter_values_batch)
    if not points:
        raise ValueError("parameter_values_batch must be nonempty")
    if any(not isinstance(point, Mapping) for point in points):
        raise TypeError("each parameter point must be a mapping")
    kwargs = dict(
        terminal_observables=terminal_observables,
        integral_weights=integral_weights,
        integral_name=integral_name,
        discontinuities=discontinuities,
        solver=solver,
        abstol=abstol,
        reltol=reltol,
        dt=dt,
        maxiters=maxiters,
        parallel=False,
    )
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ValueError("threads must be a positive integer")
    run = lambda point: prepare_lindblad_adjoint_response(  # noqa: E731
        prepared, t_span, parameter_values=point, **kwargs
    )
    if parallel and len(points) > 1:
        with ThreadPoolExecutor(max_workers=min(len(points), threads or 4)) as pool:
            return list(pool.map(run, points))
    return [run(point) for point in points]
