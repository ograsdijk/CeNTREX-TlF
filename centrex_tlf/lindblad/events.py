from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as smp

from .ir import _compile_sympy_expr
from .parameters import Parameter, RuntimeExpression
from .plan_static import PreparedLindbladProblem

__all__ = [
    "PopulationEvent",
    "population",
]


@dataclass(frozen=True)
class PopulationEvent:
    indices: tuple[int, ...]
    threshold: float = 0.0
    name: str = "population"


def population(
    index_or_indices: int | Sequence[int],
    *,
    threshold: float = 0.0,
    name: str | None = None,
) -> PopulationEvent:
    if isinstance(index_or_indices, int | np.integer):
        indices = (int(index_or_indices),)
    else:
        indices = tuple(int(index) for index in index_or_indices)
    if not indices:
        raise ValueError("population event requires at least one index")
    event_name = name if name is not None else "population"
    return PopulationEvent(indices=indices, threshold=float(threshold), name=event_name)


def _runtime_expression_from_event(stop_event: Any) -> RuntimeExpression | None:
    if isinstance(stop_event, RuntimeExpression):
        return stop_event
    if isinstance(stop_event, Parameter):
        return stop_event.as_expression()
    if isinstance(stop_event, smp.Expr):
        return RuntimeExpression(stop_event, {})
    return None


def normalize_stop_event(
    stop_event: Any | None,
    prepared: PreparedLindbladProblem,
) -> dict[str, Any] | None:
    if stop_event is None:
        return None
    n_states = int(prepared.layout.n)
    if isinstance(stop_event, PopulationEvent):
        indices = tuple(int(index) for index in stop_event.indices)
        for index in indices:
            if index < 0 or index >= n_states:
                raise ValueError(f"population event index {index} out of bounds for {n_states} states")
        return {
            "kind": "population",
            "indices": list(indices),
            "threshold": float(stop_event.threshold),
            "name": stop_event.name,
        }
    runtime = _runtime_expression_from_event(stop_event)
    if runtime is None:
        raise TypeError(
            "stop_event must be a RuntimeExpression, sympy expression, Parameter, "
            "PopulationEvent, or None"
        )
    slot_names = list(prepared.parameter_graph["slot_names"])
    slot_index_by_name = {name: idx for idx, name in enumerate(slot_names)}
    missing = sorted(
        str(symbol)
        for symbol in runtime.expr.free_symbols
        if str(symbol) != "t" and str(symbol) not in slot_index_by_name
    )
    if missing:
        raise ValueError(f"stop_event references unknown parameter(s): {', '.join(missing)}")
    compiled = _compile_sympy_expr(runtime.expr, slot_index_by_name)
    if compiled.output_is_tuple:
        raise ValueError("stop_event RuntimeExpression must evaluate to a scalar")
    return {
        "kind": "runtime_expression",
        "expression": compiled.to_payload(),
        "runtime": runtime,
        "name": repr(runtime),
    }


def scipy_event_function(
    event_spec: dict[str, Any] | None,
    *,
    packed_layout: Any,
    matrix_state: bool = False,
) -> Any | None:
    if event_spec is None:
        return None
    if event_spec["kind"] == "population":
        indices = tuple(int(index) for index in event_spec["indices"])
        threshold = float(event_spec["threshold"])

        def event(_t: float, y: np.ndarray) -> float:
            if matrix_state:
                n = int(packed_layout.n)
                matrix = np.asarray(y, dtype=np.complex128).reshape((n, n))
                return float(np.real(np.diag(matrix)[list(indices)]).sum() - threshold)
            return float(np.sum(np.asarray(y, dtype=np.float64)[list(indices)]) - threshold)

    else:
        runtime = event_spec["runtime"]
        func = runtime.compile_callable("t")

        def event(t: float, _y: np.ndarray) -> float:
            value = func(float(t))
            if np.iscomplexobj(value) and abs(complex(value).imag) > 1e-12:
                raise ValueError("stop_event RuntimeExpression evaluated to a complex value")
            return float(np.real(value))

    event.terminal = True
    event.direction = 0
    return event
