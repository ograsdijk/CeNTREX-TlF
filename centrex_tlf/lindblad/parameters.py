from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
import numbers
from typing import Any, Mapping

import numpy as np
import sympy as smp
from centrex_tlf import hamiltonian
from sympy.parsing import sympy_parser

from .helper_functions import HELPER_FUNCTIONS

__all__ = [
    "Parameter",
    "RuntimeExpression",
    "Time",
    "LindbladParameters",
    "adapt_lindblad_parameters",
    "generate_lindblad_parameters",
    "gaussian",
    "linear",
    "sine",
    "square_wave_profile",
    "tabulated",
    "pchip_tabulated",
]


RuntimeScalar = int | float | complex | np.number


def _normalize_symbol_name(value: str) -> str:
    return value.replace("\u03d5", "\u03c6")


def _is_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_base_value(value: Any) -> complex | tuple[complex, ...]:
    if isinstance(value, bool):
        return complex(float(value), 0.0)
    if isinstance(value, np.generic):
        return _coerce_base_value(value.item())
    if isinstance(value, numbers.Number):
        return complex(value)
    if _is_sequence(value):
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            return tuple(_coerce_base_value(v.item()) for v in flat)
        return tuple(_coerce_base_value(v) for v in value)
    raise TypeError(f"unsupported base parameter type {type(value).__name__}")


def _base_values_equal(
    left: complex | tuple[complex, ...],
    right: complex | tuple[complex, ...],
) -> bool:
    if isinstance(left, tuple) or isinstance(right, tuple):
        return isinstance(left, tuple) and isinstance(right, tuple) and left == right
    return complex(left) == complex(right)


def _parse_expr(expr: str) -> smp.Expr:
    local_dict = {name: smp.Function(name) for name in HELPER_FUNCTIONS}
    return sympy_parser.parse_expr(_normalize_symbol_name(expr), local_dict=local_dict)


def _symbol_from_name(name: str, kind: str) -> smp.Symbol:
    if kind == "real":
        return smp.Symbol(name, real=True)
    if kind == "complex":
        return smp.Symbol(name, complex=True)
    raise ValueError("kind must be 'real' or 'complex'")


def _infer_kind(default: Any) -> str:
    coerced = _coerce_base_value(default)
    if isinstance(coerced, tuple):
        return "complex" if any(abs(value.imag) > 0.0 for value in coerced) else "real"
    return "complex" if abs(coerced.imag) > 0.0 else "real"


def _merge_parameters(
    left: Mapping[str, Parameter],
    right: Mapping[str, Parameter],
) -> dict[str, Parameter]:
    merged = dict(left)
    for name, parameter in right.items():
        existing = merged.get(name)
        if existing is not None and existing is not parameter:
            raise ValueError(f"duplicate Parameter name {name!r}")
        merged[name] = parameter
    return merged


@dataclass(frozen=True, eq=False)
class Parameter:
    """Registered runtime parameter that can be scanned or used in expressions."""

    name: str
    default: Any = 0.0
    kind: str = "real"
    symbol: smp.Symbol = field(init=False)

    def __post_init__(self) -> None:
        normalized = _normalize_symbol_name(self.name)
        if self.kind not in {"real", "complex"}:
            raise ValueError("kind must be 'real' or 'complex'")
        object.__setattr__(self, "name", normalized)
        object.__setattr__(self, "symbol", _symbol_from_name(normalized, self.kind))

    def as_expression(self) -> RuntimeExpression:
        return RuntimeExpression(self.symbol, {self.name: self})

    def __hash__(self) -> int:
        return id(self)

    def __neg__(self) -> RuntimeExpression:
        return -self.as_expression()

    def __add__(self, other: ExpressionLike) -> RuntimeExpression:
        return self.as_expression() + other

    def __radd__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) + self.as_expression()

    def __sub__(self, other: ExpressionLike) -> RuntimeExpression:
        return self.as_expression() - other

    def __rsub__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) - self.as_expression()

    def __mul__(self, other: ExpressionLike) -> RuntimeExpression:
        return self.as_expression() * other

    def __rmul__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) * self.as_expression()

    def __truediv__(self, other: ExpressionLike) -> RuntimeExpression:
        return self.as_expression() / other

    def __rtruediv__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) / self.as_expression()

    def __pow__(self, other: ExpressionLike) -> RuntimeExpression:
        return self.as_expression() ** other

    def __rpow__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) ** self.as_expression()


@dataclass(frozen=True)
class RuntimeExpression:
    """SymPy expression plus the registered parameters it depends on."""

    expr: smp.Expr
    parameters: Mapping[str, Parameter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expr", smp.sympify(self.expr))
        object.__setattr__(self, "parameters", dict(self.parameters))

    def _binary(self, other: ExpressionLike, op: Any) -> RuntimeExpression:
        rhs = _to_runtime_expression(other)
        return RuntimeExpression(
            op(self.expr, rhs.expr),
            _merge_parameters(self.parameters, rhs.parameters),
        )

    def __neg__(self) -> RuntimeExpression:
        return RuntimeExpression(-self.expr, self.parameters)

    def __add__(self, other: ExpressionLike) -> RuntimeExpression:
        return self._binary(other, lambda left, right: left + right)

    def __radd__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) + self

    def __sub__(self, other: ExpressionLike) -> RuntimeExpression:
        return self._binary(other, lambda left, right: left - right)

    def __rsub__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) - self

    def __mul__(self, other: ExpressionLike) -> RuntimeExpression:
        return self._binary(other, lambda left, right: left * right)

    def __rmul__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) * self

    def __truediv__(self, other: ExpressionLike) -> RuntimeExpression:
        return self._binary(other, lambda left, right: left / right)

    def __rtruediv__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) / self

    def __pow__(self, other: ExpressionLike) -> RuntimeExpression:
        return self._binary(other, lambda left, right: left**right)

    def __rpow__(self, other: ExpressionLike) -> RuntimeExpression:
        return _to_runtime_expression(other) ** self


ExpressionLike = RuntimeExpression | Parameter | smp.Expr | RuntimeScalar


def _to_runtime_expression(value: ExpressionLike) -> RuntimeExpression:
    if isinstance(value, RuntimeExpression):
        return value
    if isinstance(value, Parameter):
        return value.as_expression()
    if isinstance(value, smp.Expr):
        return RuntimeExpression(value, {})
    if isinstance(value, np.generic):
        return RuntimeExpression(smp.sympify(value.item()), {})
    if isinstance(value, numbers.Number):
        return RuntimeExpression(smp.sympify(value), {})
    raise TypeError(f"unsupported runtime expression type {type(value).__name__}")


def _call_function(name: str, *args: ExpressionLike) -> RuntimeExpression:
    coerced = [_to_runtime_expression(arg) for arg in args]
    parameters: dict[str, Parameter] = {}
    for arg in coerced:
        parameters = _merge_parameters(parameters, arg.parameters)
    return RuntimeExpression(smp.Function(name)(*(arg.expr for arg in coerced)), parameters)


def Time() -> RuntimeExpression:
    return RuntimeExpression(smp.Symbol("t", real=True), {})


def linear(
    x: ExpressionLike,
    offset: ExpressionLike = 0.0,
    slope: ExpressionLike = 1.0,
) -> RuntimeExpression:
    return _to_runtime_expression(offset) + (
        _to_runtime_expression(slope) * _to_runtime_expression(x)
    )


def sine(
    x: ExpressionLike,
    offset: ExpressionLike = 0.0,
    amplitude: ExpressionLike = 1.0,
    angular_frequency: ExpressionLike = 1.0,
    phase: ExpressionLike = 0.0,
) -> RuntimeExpression:
    x_expr = _to_runtime_expression(x)
    omega = _to_runtime_expression(angular_frequency)
    phase_expr = _to_runtime_expression(phase)
    arg = omega * x_expr + phase_expr
    return _to_runtime_expression(offset) + _to_runtime_expression(amplitude) * RuntimeExpression(
        smp.sin(arg.expr),
        arg.parameters,
    )


def gaussian(
    x: ExpressionLike,
    center: ExpressionLike = 0.0,
    sigma: ExpressionLike = 1.0,
    amplitude: ExpressionLike = 1.0,
    baseline: ExpressionLike = 0.0,
) -> RuntimeExpression:
    dx = _to_runtime_expression(x) - center
    width = _to_runtime_expression(sigma)
    exponent = -(dx**2) / (2.0 * width**2)
    return _to_runtime_expression(baseline) + _to_runtime_expression(amplitude) * RuntimeExpression(
        smp.exp(exponent.expr),
        exponent.parameters,
    )


def square_wave_profile(
    x: ExpressionLike,
    low: ExpressionLike = 0.0,
    high: ExpressionLike = 1.0,
    period: ExpressionLike = 1.0,
    phase: ExpressionLike = 0.0,
    duty: ExpressionLike = 0.5,
) -> RuntimeExpression:
    on = _call_function(
        "variable_on_off_duty",
        x,
        duty,
        1.0 / _to_runtime_expression(period),
        phase,
    )
    return _to_runtime_expression(low) + (_to_runtime_expression(high) - low) * on


def tabulated(x: ExpressionLike, grid: Parameter, values: Parameter) -> RuntimeExpression:
    if not isinstance(grid, Parameter) or not isinstance(values, Parameter):
        raise TypeError("tabulated requires grid and values to be registered tuple Parameters")
    return _call_function("linear_interp", x, grid, values)


def pchip_tabulated(x: ExpressionLike, grid: Parameter, values: Parameter) -> RuntimeExpression:
    if not isinstance(grid, Parameter) or not isinstance(values, Parameter):
        raise TypeError("pchip_tabulated requires grid and values to be registered tuple Parameters")
    return _call_function("pchip_interp", x, grid, values)


class LindbladParameters:
    """Container for OBE runtime parameters and Hamiltonian symbol bindings."""

    def __init__(
        self,
        base_parameters: Mapping[Any, Any] | None = None,
        compound_parameters: Mapping[Any, Any] | None = None,
    ) -> None:
        self.base_parameters: OrderedDict[str, complex | tuple[complex, ...]] = OrderedDict()
        self.compound_parameters: OrderedDict[str, str] = OrderedDict()
        self._compound_expressions: OrderedDict[str, smp.Expr] = OrderedDict()
        self._parameters_by_name: dict[str, Parameter] = {}

        if base_parameters is not None:
            if compound_parameters is None and self._looks_like_binding_mapping(base_parameters):
                for symbol, expression in base_parameters.items():
                    self.bind(symbol, expression, finalize=False)
            else:
                for name, value in base_parameters.items():
                    self._set_base(str(name), value)
        if compound_parameters is not None:
            for name, expression in compound_parameters.items():
                self._set_compound(str(name), expression)
        self._finalize()

    @staticmethod
    def _looks_like_binding_mapping(mapping: Mapping[Any, Any]) -> bool:
        return any(
            isinstance(key, smp.Symbol)
            or isinstance(value, (RuntimeExpression, Parameter, smp.Expr))
            for key, value in mapping.items()
        )

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> LindbladParameters:
        if any(
            isinstance(value, (RuntimeExpression, Parameter, smp.Expr))
            for value in kwargs.values()
        ):
            return cls(kwargs)
        base = OrderedDict()
        compound = OrderedDict()
        for key, value in kwargs.items():
            if isinstance(value, str):
                compound[key] = value
            else:
                base[key] = value
        return cls(base, compound)

    @classmethod
    def from_legacy(cls, legacy: Any) -> LindbladParameters:
        if not hasattr(legacy, "_parameters") or not hasattr(legacy, "_compound_vars"):
            raise TypeError("legacy object does not look like odeParameters")
        base = OrderedDict(
            (name, getattr(legacy, name)) for name in list(getattr(legacy, "_parameters"))
        )
        compound = OrderedDict(
            (name, getattr(legacy, name))
            for name in list(getattr(legacy, "_compound_vars"))
        )
        return cls(base, compound)

    def parameter(self, name: str, default: Any = 0.0, kind: str | None = None) -> Parameter:
        parameter = Parameter(name, default, _infer_kind(default) if kind is None else kind)
        self._register_parameter(parameter)
        return parameter

    def real(self, name: str, default: Any = 0.0) -> Parameter:
        return self.parameter(name, default, kind="real")

    def complex(self, name: str, default: Any = 0.0) -> Parameter:
        return self.parameter(name, default, kind="complex")

    def time(self) -> RuntimeExpression:
        return Time()

    def bind(
        self,
        symbol: Any,
        expression: ExpressionLike,
        *,
        overwrite: bool = True,
        finalize: bool = True,
    ) -> LindbladParameters:
        name = _normalize_symbol_name(str(symbol))
        runtime = _to_runtime_expression(expression)
        for parameter in runtime.parameters.values():
            self._register_parameter(parameter)
        if name in self.compound_parameters and not overwrite:
            raise ValueError(f"symbol {name!r} is already bound")
        expr_name = str(runtime.expr)
        if expr_name == name and name in self.base_parameters:
            self.compound_parameters.pop(name, None)
            self._compound_expressions.pop(name, None)
        else:
            self._set_compound(name, runtime.expr)
        if finalize:
            self._finalize()
        return self

    def drive(
        self,
        selector: Any,
        *,
        rabi: ExpressionLike,
        detuning: ExpressionLike,
        overwrite: bool = True,
    ) -> LindbladParameters:
        self.bind(getattr(selector, "Ω"), rabi, overwrite=overwrite, finalize=False)
        self.bind(getattr(selector, "δ"), detuning, overwrite=overwrite, finalize=False)
        self._finalize()
        return self

    @property
    def all_parameter_names(self) -> list[str]:
        return list(self.base_parameters) + list(self.compound_parameters)

    @property
    def slot_names(self) -> list[str]:
        return self.all_parameter_names

    @property
    def slot_index_by_name(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.slot_names)}

    @property
    def parsed_compound_parameters(self) -> OrderedDict[str, smp.Expr]:
        return OrderedDict(self._compound_expressions)

    def validate_symbols(self, symbols: Sequence[str]) -> None:
        allowed = set(self.all_parameter_names)
        missing = sorted(
            {
                _normalize_symbol_name(str(name))
                for name in symbols
                if _normalize_symbol_name(str(name)) != "t"
                and _normalize_symbol_name(str(name)) not in allowed
            }
        )
        if missing:
            raise AssertionError(f"Symbol(s) not defined: {', '.join(missing)}")

    def check_hamiltonian_symbols(self, hamiltonian: smp.Matrix) -> None:
        self.validate_symbols([str(symbol) for symbol in hamiltonian.free_symbols])

    def _set_base(self, name: str, value: Any) -> None:
        normalized = _normalize_symbol_name(name)
        self.base_parameters[normalized] = _coerce_base_value(value)

    def _set_compound(self, name: str, expression: str | smp.Expr | RuntimeExpression) -> None:
        normalized = _normalize_symbol_name(name)
        if isinstance(expression, RuntimeExpression):
            expr = expression.expr
        elif isinstance(expression, smp.Expr):
            expr = expression
        elif isinstance(expression, str):
            expr = _parse_expr(expression)
        else:
            raise TypeError(f"unsupported compound expression type {type(expression).__name__}")
        self.compound_parameters[normalized] = _normalize_symbol_name(str(expr))
        self._compound_expressions[normalized] = expr

    def _register_parameter(self, parameter: Parameter) -> None:
        existing = self._parameters_by_name.get(parameter.name)
        if existing is not None and existing is not parameter:
            raise ValueError(f"duplicate Parameter name {parameter.name!r}")
        self._parameters_by_name[parameter.name] = parameter
        value = _coerce_base_value(parameter.default)
        existing_value = self.base_parameters.get(parameter.name)
        if existing_value is not None and not _base_values_equal(existing_value, value):
            raise ValueError(f"base parameter {parameter.name!r} already has a different default")
        self.base_parameters[parameter.name] = value

    def _defined_names(self) -> set[str]:
        return set(self.all_parameter_names) | {"t"}

    def _numeric_names(self) -> set[str]:
        return set(self.base_parameters) | {"t"}

    def _check_symbols_defined(self) -> None:
        defined = self._defined_names()
        missing: list[str] = []
        for expr in self._compound_expressions.values():
            for symbol in expr.free_symbols:
                name = _normalize_symbol_name(str(symbol))
                if name not in defined:
                    missing.append(name)
        if missing:
            unique = ", ".join(sorted(set(missing)))
            raise AssertionError(f"Symbol(s) not defined: {unique}")

    def _order_compound_parameters(self) -> None:
        numeric = self._numeric_names()
        unordered = list(self.compound_parameters)
        ordered: list[str] = []
        while unordered:
            progressed = False
            for name in unordered:
                symbols = {
                    _normalize_symbol_name(str(symbol))
                    for symbol in self._compound_expressions[name].free_symbols
                }
                if all(symbol in numeric or symbol in ordered for symbol in symbols):
                    ordered.append(name)
                    progressed = True
            if not progressed:
                raise ValueError("could not resolve compound parameter dependency order")
            unordered = [name for name in unordered if name not in ordered]
        self.compound_parameters = OrderedDict(
            (name, self.compound_parameters[name]) for name in ordered
        )
        self._compound_expressions = OrderedDict(
            (name, self._compound_expressions[name]) for name in ordered
        )

    def _finalize(self) -> None:
        self._check_symbols_defined()
        self._order_compound_parameters()


def adapt_lindblad_parameters(parameters: Any) -> LindbladParameters:
    if isinstance(parameters, LindbladParameters):
        return parameters
    if isinstance(parameters, Mapping):
        if any(
            not isinstance(key, str) or isinstance(value, (RuntimeExpression, Parameter, smp.Expr))
            for key, value in parameters.items()
        ):
            return LindbladParameters(parameters)
        return LindbladParameters.from_kwargs(**dict(parameters))
    if hasattr(parameters, "_parameters") and hasattr(parameters, "_compound_vars"):
        return LindbladParameters.from_legacy(parameters)
    raise TypeError(
        "parameters must be a LindbladParameters instance, a mapping, "
        "or an odeParameters-like object"
    )


def generate_lindblad_parameters(
    transition_selectors: Sequence[Any],
    **kwargs: Any,
) -> LindbladParameters:
    parameters: list[tuple[str, Any]] = []
    for idx, selector in enumerate(transition_selectors):
        if getattr(selector, "phase_modulation", False):
            parameters.extend(
                [
                    (str(selector.Ω), f"Ωt{idx}*phase_modulation(t, β{idx}, ωphase{idx})"),
                    (f"Ωt{idx}", hamiltonian.Γ),
                    (f"β{idx}", 3.8),
                    (f"ωphase{idx}", hamiltonian.Γ),
                ]
            )
        else:
            parameters.append((str(selector.Ω), hamiltonian.Γ))
        parameters.append((str(selector.δ), 0.0))
        symbols = list(getattr(selector, "polarization_symbols", []))
        if len(symbols) == 1:
            parameters.append((str(symbols[0]), 1.0))
        elif len(symbols) == 2:
            parameters.extend(
                [
                    (f"ω{idx}", hamiltonian.Γ),
                    (f"φ{idx}", 0.0),
                    (str(symbols[0]), f"square_wave(t, ω{idx}, φ{idx})"),
                    (str(symbols[1]), f"1 - {symbols[0]}"),
                ]
            )
        elif len(symbols) > 2:
            raise ValueError("polarization switching with more than two symbols is not supported")
    parameter_dict = OrderedDict(parameters)
    for key, value in kwargs.items():
        parameter_dict[key] = value
    return adapt_lindblad_parameters(parameter_dict)
