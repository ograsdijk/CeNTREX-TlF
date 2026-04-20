from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import numbers
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import sympy as smp
from centrex_tlf import hamiltonian
from sympy.parsing import sympy_parser

from .helper_functions import HELPER_FUNCTIONS

__all__ = [
    "LindbladParameters",
    "adapt_lindblad_parameters",
    "generate_lindblad_parameters",
]


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


def _parse_expr(expr: str) -> smp.Expr:
    local_dict = {name: smp.Function(name) for name in HELPER_FUNCTIONS}
    return sympy_parser.parse_expr(expr, local_dict=local_dict)


@dataclass
class LindbladParameters:
    base_parameters: OrderedDict[str, complex | tuple[complex, ...]]
    compound_parameters: OrderedDict[str, str]

    def __post_init__(self) -> None:
        self.base_parameters = OrderedDict(
            (_normalize_symbol_name(key), _coerce_base_value(value))
            for key, value in self.base_parameters.items()
        )
        self.compound_parameters = OrderedDict(
            (_normalize_symbol_name(key), value.replace("\u03d5", "\u03c6"))
            for key, value in self.compound_parameters.items()
        )
        self._check_symbols_defined()
        self._order_compound_parameters()

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> LindbladParameters:
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
        return OrderedDict(
            (name, _parse_expr(expr)) for name, expr in self.compound_parameters.items()
        )

    def validate_symbols(self, symbols: Sequence[str]) -> None:
        allowed = set(self.all_parameter_names)
        missing = sorted({name for name in symbols if name != "t" and name not in allowed})
        if missing:
            raise AssertionError(f"Symbol(s) not defined: {', '.join(missing)}")

    def check_hamiltonian_symbols(self, hamiltonian: smp.Matrix) -> None:
        self.validate_symbols([str(symbol) for symbol in hamiltonian.free_symbols])

    def _defined_symbols(self) -> set[smp.Symbol]:
        names = self.all_parameter_names + ["t"]
        return {smp.Symbol(name) for name in names}

    def _numeric_symbols(self) -> set[smp.Symbol]:
        names = list(self.base_parameters) + ["t"]
        return {smp.Symbol(name) for name in names}

    def _check_symbols_defined(self) -> None:
        defined = self._defined_symbols()
        expressions = [_parse_expr(expr) for expr in self.compound_parameters.values()]
        missing: list[str] = []
        for expr in expressions:
            for symbol in expr.free_symbols:
                if symbol not in defined:
                    missing.append(str(symbol))
        if missing:
            unique = ", ".join(sorted(set(missing)))
            raise AssertionError(f"Symbol(s) not defined: {unique}")

    def _order_compound_parameters(self) -> None:
        numeric = self._numeric_symbols()
        unordered = list(self.compound_parameters)
        ordered: list[str] = []
        while unordered:
            progressed = False
            for name in unordered:
                symbols = _parse_expr(self.compound_parameters[name]).free_symbols
                if all(symbol in numeric or str(symbol) in ordered for symbol in symbols):
                    ordered.append(name)
                    progressed = True
            if not progressed:
                raise ValueError("could not resolve compound parameter dependency order")
            unordered = [name for name in unordered if name not in ordered]
        self.compound_parameters = OrderedDict(
            (name, self.compound_parameters[name]) for name in ordered
        )


def adapt_lindblad_parameters(parameters: Any) -> LindbladParameters:
    if isinstance(parameters, LindbladParameters):
        return parameters
    if isinstance(parameters, Mapping):
        return LindbladParameters.from_kwargs(**dict(parameters))
    if hasattr(parameters, "_parameters") and hasattr(parameters, "_compound_vars"):
        return LindbladParameters.from_legacy(parameters)
    raise TypeError(
        "parameters must be a LindbladParameters instance, a mapping, or an odeParameters-like object"
    )


def generate_lindblad_parameters(transition_selectors: Sequence[Any], **kwargs: Any) -> LindbladParameters:
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
                    (f"P{idx}", f"sin(ω{idx}*t+φ{idx})"),
                    (str(symbols[0]), f"P{idx} > 0"),
                    (str(symbols[1]), f"P{idx} <= 0"),
                ]
            )
        elif len(symbols) > 2:
            raise ValueError("polarization switching with more than two symbols is not supported")
    parameter_dict = OrderedDict(parameters)
    for key, value in kwargs.items():
        parameter_dict[key] = value
    return adapt_lindblad_parameters(parameter_dict)
