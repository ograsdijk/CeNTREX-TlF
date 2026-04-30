from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
import cmath
from typing import Any

import numpy as np
import numpy.typing as npt
import sympy as smp
from sympy.core.relational import Relational

from .helper_functions import HELPER_FUNCTION_IDS, HELPER_FUNCTION_NAMES, HELPER_FUNCTIONS

__all__ = [
    "BuiltinFunctionId",
    "CompiledExpression",
    "InstructionOp",
    "evaluate_parameter_graph_py",
    "fill_hamiltonian_py",
    "lower_expanded_sparse_rhs",
    "lower_hamiltonian_upper_triangle",
    "lower_parameter_graph",
]


class InstructionOp(IntEnum):
    CONST = 1
    SLOT = 2
    TEMP = 3
    TIME = 4
    ADD = 5
    SUB = 6
    MUL = 7
    DIV = 8
    POW = 9
    NEG = 10
    CONJ = 11
    BUILTIN_FUNC = 12
    HELPER_FUNC = 13
    GT = 14
    GE = 15
    LT = 16
    LE = 17
    EQ = 18
    NE = 19


class BuiltinFunctionId(IntEnum):
    SIN = 1
    COS = 2
    TAN = 3
    EXP = 4
    ABS = 5
    REAL = 6
    IMAG = 7


BUILTIN_FUNCTION_IDS: dict[str, BuiltinFunctionId] = {
    "sin": BuiltinFunctionId.SIN,
    "cos": BuiltinFunctionId.COS,
    "tan": BuiltinFunctionId.TAN,
    "exp": BuiltinFunctionId.EXP,
    "Abs": BuiltinFunctionId.ABS,
    "re": BuiltinFunctionId.REAL,
    "im": BuiltinFunctionId.IMAG,
}


ScalarValue = complex
RuntimeValue = ScalarValue | tuple[ScalarValue, ...]


def encode_runtime_value(value: RuntimeValue) -> dict[str, Any]:
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [encode_runtime_value(item) for item in value]}
    return {"kind": "scalar", "re": float(value.real), "im": float(value.imag)}


def decode_runtime_value(payload: dict[str, Any]) -> RuntimeValue:
    kind = payload["kind"]
    if kind == "scalar":
        return complex(payload["re"], payload["im"])
    if kind == "tuple":
        items = [decode_runtime_value(item) for item in payload["items"]]
        if any(isinstance(item, tuple) for item in items):
            raise TypeError("nested tuple runtime values are not supported")
        return tuple(item for item in items if not isinstance(item, tuple))
    raise ValueError(f"unknown value kind {kind}")


def _is_scalar(value: RuntimeValue) -> bool:
    return not isinstance(value, tuple)


def _as_scalar(value: RuntimeValue) -> ScalarValue:
    if isinstance(value, tuple):
        raise TypeError("sequence value used where scalar was required")
    return value


def _as_real(value: RuntimeValue) -> float:
    scalar = _as_scalar(value)
    if abs(scalar.imag) > 1e-12:
        raise TypeError("complex value used where real scalar was required")
    return float(scalar.real)


def _binary_elementwise(
    left: RuntimeValue,
    right: RuntimeValue,
    op: Any,
) -> RuntimeValue:
    if _is_scalar(left) and _is_scalar(right):
        return op(_as_scalar(left), _as_scalar(right))
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("tuple lengths do not match for elementwise operation")
        return tuple(op(l, r) for l, r in zip(left, right))
    if isinstance(left, tuple):
        scalar = _as_scalar(right)
        return tuple(op(item, scalar) for item in left)
    scalar = _as_scalar(left)
    if not isinstance(right, tuple):
        raise TypeError(f"expected right operand to be a tuple, got {type(right)}")
    return tuple(op(scalar, item) for item in right)


def _apply_builtin(function_id: int, args: list[RuntimeValue]) -> RuntimeValue:
    builtin = BuiltinFunctionId(function_id)
    value = _as_scalar(args[0])
    if builtin == BuiltinFunctionId.SIN:
        return cmath.sin(value)
    if builtin == BuiltinFunctionId.COS:
        return cmath.cos(value)
    if builtin == BuiltinFunctionId.TAN:
        return cmath.tan(value)
    if builtin == BuiltinFunctionId.EXP:
        return cmath.exp(value)
    if builtin == BuiltinFunctionId.ABS:
        return complex(abs(value), 0.0)
    if builtin == BuiltinFunctionId.REAL:
        return complex(value.real, 0.0)
    if builtin == BuiltinFunctionId.IMAG:
        return complex(value.imag, 0.0)
    raise ValueError(f"unsupported builtin function id {function_id}")


def _apply_helper(function_id: int, args: list[RuntimeValue]) -> RuntimeValue:
    name = HELPER_FUNCTION_NAMES[function_id]
    helper = HELPER_FUNCTIONS[name]
    coerced: list[Any] = []
    for arg in args:
        if isinstance(arg, tuple):
            coerced.append(tuple(_as_real(item) for item in arg))
        else:
            scalar = arg
            coerced.append(scalar if abs(scalar.imag) > 1e-12 else float(scalar.real))
    result = helper(*coerced)
    if isinstance(result, tuple):
        return tuple(complex(item) for item in result)
    return complex(result)


def _compare(left: RuntimeValue, right: RuntimeValue, op: InstructionOp) -> RuntimeValue:
    lhs = _as_real(left)
    rhs = _as_real(right)
    if op == InstructionOp.GT:
        return complex(1.0 if lhs > rhs else 0.0, 0.0)
    if op == InstructionOp.GE:
        return complex(1.0 if lhs >= rhs else 0.0, 0.0)
    if op == InstructionOp.LT:
        return complex(1.0 if lhs < rhs else 0.0, 0.0)
    if op == InstructionOp.LE:
        return complex(1.0 if lhs <= rhs else 0.0, 0.0)
    if op == InstructionOp.EQ:
        return complex(1.0 if lhs == rhs else 0.0, 0.0)
    if op == InstructionOp.NE:
        return complex(1.0 if lhs != rhs else 0.0, 0.0)
    raise ValueError(f"unsupported comparison op {op}")


@dataclass(frozen=True)
class CompiledExpression:
    instructions: list[dict[str, Any]]
    scalar_only: bool = False
    output_is_tuple: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "instructions": self.instructions,
            "scalar_only": self.scalar_only,
            "output_is_tuple": self.output_is_tuple,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CompiledExpression:
        return cls(
            instructions=list(payload["instructions"]),
            scalar_only=bool(payload.get("scalar_only", False)),
            output_is_tuple=bool(payload.get("output_is_tuple", False)),
        )


def _constant_instruction(value: complex) -> dict[str, Any]:
    return {"op": int(InstructionOp.CONST), "re": float(value.real), "im": float(value.imag)}


def _compile_sympy_expr(
    expr: smp.Expr,
    slot_index_by_name: dict[str, int],
    temp_index_by_name: dict[str, int] | None = None,
    tuple_value_names: set[str] | None = None,
    tuple_temp_names: set[str] | None = None,
) -> CompiledExpression:
    if temp_index_by_name is None:
        temp_index_by_name = {}
    if tuple_value_names is None:
        tuple_value_names = set()
    if tuple_temp_names is None:
        tuple_temp_names = set()
    instructions: list[dict[str, Any]] = []

    def emit(node: smp.Basic) -> tuple[bool, bool]:
        if node == smp.I:
            instructions.append(_constant_instruction(1j))
            return True, False
        if isinstance(node, smp.Symbol):
            name = str(node)
            if name == "t":
                instructions.append({"op": int(InstructionOp.TIME)})
                return True, False
            elif name in temp_index_by_name:
                instructions.append({"op": int(InstructionOp.TEMP), "index": int(temp_index_by_name[name])})
                is_tuple = name in tuple_temp_names
                return not is_tuple, is_tuple
            elif name in slot_index_by_name:
                instructions.append({"op": int(InstructionOp.SLOT), "index": int(slot_index_by_name[name])})
                is_tuple = name in tuple_value_names
                return not is_tuple, is_tuple
            else:
                raise KeyError(f"symbol {name} not found in slot or temp namespace")
        if isinstance(node, smp.Number) or node.is_number:
            instructions.append(_constant_instruction(complex(node.evalf())))
            return True, False
        if isinstance(node, Relational):
            lhs_scalar, lhs_tuple = emit(node.lhs)
            rhs_scalar, rhs_tuple = emit(node.rhs)
            op_map = {
                "StrictGreaterThan": InstructionOp.GT,
                "GreaterThan": InstructionOp.GE,
                "StrictLessThan": InstructionOp.LT,
                "LessThan": InstructionOp.LE,
                "Equality": InstructionOp.EQ,
                "Unequality": InstructionOp.NE,
            }
            try:
                op = op_map[type(node).__name__]
            except KeyError as exc:
                raise NotImplementedError(f"unsupported relational {type(node).__name__}") from exc
            instructions.append({"op": int(op)})
            return lhs_scalar and rhs_scalar and not (lhs_tuple or rhs_tuple), False
        if isinstance(node, smp.Add):
            args = list(node.args)
            scalar_only, tuple_output = emit(args[0])
            for arg in args[1:]:
                arg_scalar, arg_tuple = emit(arg)
                instructions.append({"op": int(InstructionOp.ADD)})
                scalar_only = scalar_only and arg_scalar
                tuple_output = tuple_output or arg_tuple
            return scalar_only and not tuple_output, tuple_output
        if isinstance(node, smp.Mul):
            args = list(node.args)
            scalar_only, tuple_output = emit(args[0])
            for arg in args[1:]:
                arg_scalar, arg_tuple = emit(arg)
                instructions.append({"op": int(InstructionOp.MUL)})
                scalar_only = scalar_only and arg_scalar
                tuple_output = tuple_output or arg_tuple
            return scalar_only and not tuple_output, tuple_output
        if isinstance(node, smp.Pow):
            lhs_scalar, lhs_tuple = emit(node.args[0])
            rhs_scalar, rhs_tuple = emit(node.args[1])
            instructions.append({"op": int(InstructionOp.POW)})
            tuple_output = lhs_tuple or rhs_tuple
            return lhs_scalar and rhs_scalar and not tuple_output, tuple_output
        if node.func == smp.conjugate:
            child_scalar, child_tuple = emit(node.args[0])
            instructions.append({"op": int(InstructionOp.CONJ)})
            return child_scalar and not child_tuple, child_tuple
        if node.is_Function:
            func_name = getattr(node.func, "__name__", str(node.func))
            scalar_only = True
            saw_tuple_arg = False
            for arg in node.args:
                arg_scalar, arg_tuple = emit(arg)
                scalar_only = scalar_only and arg_scalar
                saw_tuple_arg = saw_tuple_arg or arg_tuple
            if func_name in BUILTIN_FUNCTION_IDS:
                instructions.append(
                    {
                        "op": int(InstructionOp.BUILTIN_FUNC),
                        "function": int(BUILTIN_FUNCTION_IDS[func_name]),
                        "argc": len(node.args),
                    }
                )
                return scalar_only and not saw_tuple_arg, False
            if func_name in HELPER_FUNCTION_IDS:
                instructions.append(
                    {
                        "op": int(InstructionOp.HELPER_FUNC),
                        "function": int(HELPER_FUNCTION_IDS[func_name]),
                        "argc": len(node.args),
                    }
                )
                return scalar_only and not saw_tuple_arg, False
            raise NotImplementedError(f"unsupported function {func_name}")
        raise NotImplementedError(f"unsupported sympy node {type(node).__name__}")

    scalar_only, output_is_tuple = emit(expr)
    return CompiledExpression(
        instructions=instructions,
        scalar_only=scalar_only,
        output_is_tuple=output_is_tuple,
    )


def lower_parameter_graph(parameters: Any) -> dict[str, Any]:
    slot_index_by_name = parameters.slot_index_by_name
    base_values = [encode_runtime_value(value) for value in parameters.base_parameters.values()]
    tuple_value_names = {
        name for name, value in parameters.base_parameters.items() if isinstance(value, tuple)
    }
    compounds = []
    for name, expr in parameters.parsed_compound_parameters.items():
        compiled = _compile_sympy_expr(
            expr,
            slot_index_by_name,
            tuple_value_names=tuple_value_names,
        )
        compounds.append(
            {
                "slot": int(slot_index_by_name[name]),
                "expression": compiled.to_payload(),
            }
        )
        if compiled.output_is_tuple:
            tuple_value_names.add(str(name))
    pchip_tables = _extract_pchip_tables(parameters)

    return {
        "slot_names": parameters.slot_names,
        "n_base": len(base_values),
        "base_values": base_values,
        "compounds": compounds,
        "pchip_tables": pchip_tables,
    }


def _extract_pchip_tables(parameters: Any) -> list[dict[str, Any]]:
    import numpy as np

    tables: list[dict[str, Any]] = []
    base_items = list(parameters.base_parameters.items())
    seen_pairs: set[tuple[int, int]] = set()
    for ci, (cname, cval) in enumerate(base_items):
        if not isinstance(cval, tuple) or len(cval) < 2:
            continue
        try:
            grid = [float(np.real(v)) for v in cval]
        except (TypeError, ValueError):
            continue
        if not all(grid[i] < grid[i + 1] for i in range(len(grid) - 1)):
            continue
        for vi, (vname, vval) in enumerate(base_items):
            if vi == ci or not isinstance(vval, tuple):
                continue
            if len(vval) != len(cval):
                continue
            pair = (ci, vi)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            try:
                values = [float(np.real(v)) for v in vval]
            except (TypeError, ValueError):
                continue
            import numpy as _np
            tables.append({
                "grid": _np.array(grid, dtype=_np.float64),
                "values": _np.array(values, dtype=_np.float64),
            })
    return tables


def lower_hamiltonian_upper_triangle(
    hamiltonian: smp.Matrix,
    slot_index_by_name: dict[str, int],
    tuple_value_names: set[str] | None = None,
    representation: str = "auto",
) -> dict[str, Any]:
    if representation not in {"auto", "entrywise", "decomposed"}:
        raise ValueError(
            "representation must be one of 'auto', 'entrywise', or 'decomposed'"
        )
    if tuple_value_names is None:
        tuple_value_names = set()

    entrywise_plan = _lower_hamiltonian_upper_triangle_entrywise(
        hamiltonian,
        slot_index_by_name,
        tuple_value_names=tuple_value_names,
    )
    decomposed_plan = _lower_hamiltonian_upper_triangle_decomposed(
        hamiltonian,
        slot_index_by_name,
        tuple_value_names=tuple_value_names,
    )

    if representation == "entrywise":
        return entrywise_plan
    if representation == "decomposed":
        return decomposed_plan

    entrywise_cost = len(entrywise_plan["temps"]) + len(entrywise_plan["entries"])
    diagnostics = decomposed_plan.get("diagnostics", {})
    decomposed_cost = len(decomposed_plan["coefficients"]) + 0.15 * diagnostics.get(
        "basis_term_count", 0
    )
    has_static = bool(np.any(np.abs(decomposed_plan["static_matrix"]) > 0))
    if has_static:
        decomposed_cost += 1
    if decomposed_cost < entrywise_cost:
        return decomposed_plan
    return entrywise_plan


def lower_expanded_sparse_rhs(
    hamiltonian_plan: dict[str, Any],
    source_decay_rates: npt.ArrayLike | np.ndarray,
    incoming_transfers_by_target: list[list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Lower the upper-triangle Lindblad RHS into a compiler-free sparse program."""
    if hamiltonian_plan.get("kind", "entrywise") != "decomposed":
        return None

    n = int(hamiltonian_plan["n"])
    upper_offsets: list[int] = []
    offset = 0
    for row in range(n):
        upper_offsets.append(offset)
        offset += n - row
    upper_len = offset

    def upper_index(i: int, j: int) -> int:
        if i > j:
            raise ValueError(f"expected upper-triangle index with i <= j, got {(i, j)}")
        return upper_offsets[i] + (j - i)

    def rho_ref(i: int, j: int) -> tuple[int, bool]:
        if i <= j:
            return upper_index(i, j), False
        return upper_index(j, i), True

    # Each full H entry is represented as (other_index, coefficient_index,
    # coefficient_conjugated, numeric_factor).
    h_rows: list[list[tuple[int, int, bool, complex]]] = [[] for _ in range(n)]
    h_cols: list[list[tuple[int, int, bool, complex]]] = [[] for _ in range(n)]

    def add_full_h_entry(
        row: int,
        col: int,
        coefficient_index: int,
        coefficient_conj: bool,
        factor: complex,
    ) -> None:
        if factor == 0:
            return
        h_rows[row].append((col, coefficient_index, coefficient_conj, factor))
        h_cols[col].append((row, coefficient_index, coefficient_conj, factor))

    def add_upper_h_entry(row: int, col: int, coefficient_index: int, factor: complex) -> None:
        add_full_h_entry(row, col, coefficient_index, False, factor)
        if row != col:
            add_full_h_entry(
                col,
                row,
                coefficient_index,
                coefficient_index >= 0,
                np.conjugate(factor),
            )

    static_matrix = np.asarray(hamiltonian_plan["static_matrix"], dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            value = complex(static_matrix[i, j])
            if value != 0:
                add_upper_h_entry(i, j, -1, value)

    for coefficient_index, coefficient in enumerate(hamiltonian_plan["coefficients"]):
        for term in coefficient["basis_terms"]:
            value = complex(float(term["re"]), float(term["im"]))
            if value != 0:
                add_upper_h_entry(
                    int(term["i"]),
                    int(term["j"]),
                    coefficient_index,
                    value,
                )

    source_decay = np.asarray(source_decay_rates, dtype=np.float64)
    terms_by_output: list[dict[tuple[int, int, bool, bool], complex]] = [
        defaultdict(complex) for _ in range(upper_len)
    ]

    def add_rhs_term(
        output_i: int,
        output_j: int,
        input_i: int,
        input_j: int,
        coefficient_index: int,
        coefficient_conj: bool,
        factor: complex,
    ) -> None:
        if factor == 0:
            return
        output = upper_index(output_i, output_j)
        input_index, input_conj = rho_ref(input_i, input_j)
        key = (input_index, coefficient_index, coefficient_conj, input_conj)
        terms_by_output[output][key] += factor

    for i in range(n):
        for j in range(i, n):
            for k, coefficient_index, coefficient_conj, h_factor in h_rows[i]:
                add_rhs_term(
                    i,
                    j,
                    k,
                    j,
                    coefficient_index,
                    coefficient_conj,
                    -1j * h_factor,
                )
            for k, coefficient_index, coefficient_conj, h_factor in h_cols[j]:
                add_rhs_term(
                    i,
                    j,
                    i,
                    k,
                    coefficient_index,
                    coefficient_conj,
                    1j * h_factor,
                )

            if i == j:
                if source_decay[i] != 0.0:
                    add_rhs_term(i, j, i, i, -1, False, -float(source_decay[i]))
                for transfer in incoming_transfers_by_target[i]:
                    rate = float(transfer["rate"])
                    if rate != 0.0:
                        add_rhs_term(i, j, int(transfer["source"]), int(transfer["source"]), -1, False, rate)
            else:
                rate = 0.5 * float(source_decay[i] + source_decay[j])
                if rate != 0.0:
                    add_rhs_term(i, j, i, j, -1, False, -rate)

    output_ptrs: list[int] = [0]
    input_indices: list[int] = []
    coeff_indices: list[int] = []
    coeff_conj: list[bool] = []
    input_conj: list[bool] = []
    factors_re: list[float] = []
    factors_im: list[float] = []
    for output_terms in terms_by_output:
        for key, factor in sorted(output_terms.items()):
            if abs(factor) <= 1e-15:
                continue
            input_index, coefficient_index, coefficient_is_conj, input_is_conj = key
            input_indices.append(input_index)
            coeff_indices.append(coefficient_index)
            coeff_conj.append(coefficient_is_conj)
            input_conj.append(input_is_conj)
            factors_re.append(float(factor.real))
            factors_im.append(float(factor.imag))
        output_ptrs.append(len(input_indices))

    hamiltonian_term_count = sum(len(row) for row in h_rows)
    return {
        "kind": "expanded_sparse",
        "n": n,
        "output_ptrs": output_ptrs,
        "input_indices": input_indices,
        "coeff_indices": coeff_indices,
        "coeff_conj": coeff_conj,
        "input_conj": input_conj,
        "factors_re": factors_re,
        "factors_im": factors_im,
        "diagnostics": {
            "upper_len": upper_len,
            "term_count": len(input_indices),
            "hamiltonian_full_term_count": hamiltonian_term_count,
            "coefficient_count": len(hamiltonian_plan["coefficients"]),
        },
    }


def _lower_hamiltonian_upper_triangle_entrywise(
    hamiltonian: smp.Matrix,
    slot_index_by_name: dict[str, int],
    tuple_value_names: set[str] | None = None,
) -> dict[str, Any]:
    if tuple_value_names is None:
        tuple_value_names = set()
    n = int(hamiltonian.rows)
    upper_triangle_exprs: list[smp.Expr] = []
    indices: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i, n):
            upper_triangle_exprs.append(smp.simplify(hamiltonian[i, j]))
            indices.append((i, j))

    temp_index_by_name: dict[str, int] = {}
    tuple_temp_names: set[str] = set()
    temp_payloads: list[dict[str, Any]] = []
    replacements, reduced_exprs = smp.cse(
        upper_triangle_exprs,
        symbols=smp.numbered_symbols("_h"),
    )
    for temp_index, (symbol, expr) in enumerate(replacements):
        compiled = _compile_sympy_expr(
            expr,
            slot_index_by_name,
            temp_index_by_name=temp_index_by_name,
            tuple_value_names=tuple_value_names,
            tuple_temp_names=tuple_temp_names,
        )
        temp_name = str(symbol)
        temp_index_by_name[temp_name] = temp_index
        if compiled.output_is_tuple:
            tuple_temp_names.add(temp_name)
        temp_payloads.append(compiled.to_payload())

    entry_payloads = []
    for (i, j), expr in zip(indices, reduced_exprs):
        compiled = _compile_sympy_expr(
            expr,
            slot_index_by_name,
            temp_index_by_name=temp_index_by_name,
            tuple_value_names=tuple_value_names,
            tuple_temp_names=tuple_temp_names,
        )
        entry_payloads.append(
            {
                "i": i,
                "j": j,
                "expression": compiled.to_payload(),
            }
        )

    return {
        "kind": "entrywise",
        "n": n,
        "temps": temp_payloads,
        "entries": entry_payloads,
    }


def _lower_hamiltonian_upper_triangle_decomposed(
    hamiltonian: smp.Matrix,
    slot_index_by_name: dict[str, int],
    tuple_value_names: set[str] | None = None,
) -> dict[str, Any]:
    if tuple_value_names is None:
        tuple_value_names = set()

    def split_numeric_factor(term: smp.Expr) -> tuple[complex, smp.Expr]:
        if term == 0:
            return 0.0 + 0.0j, smp.Integer(1)
        if term.is_number:
            return complex(term.evalf()), smp.Integer(1)
        numeric = smp.Integer(1)
        symbolic_factors: list[smp.Expr] = []
        for factor in smp.Mul.make_args(term):
            if factor.is_number:
                numeric *= factor
            else:
                symbolic_factors.append(factor)
        symbolic = smp.Mul(*symbolic_factors) if symbolic_factors else smp.Integer(1)
        symbolic = smp.simplify(smp.factor_terms(symbolic))
        return complex(numeric.evalf()), symbolic

    def group_basis_rows(basis_upper: np.ndarray) -> list[dict[str, Any]]:
        row_segments: list[dict[str, Any]] = []
        for row in range(n):
            active_cols = [col for col in range(row, n) if basis_upper[row, col] != 0]
            if not active_cols:
                continue
            start = active_cols[0]
            prev = start
            values: list[complex] = [complex(basis_upper[row, start])]
            for col in active_cols[1:]:
                if col == prev + 1:
                    values.append(complex(basis_upper[row, col]))
                else:
                    row_segments.append(
                        {
                            "row": row,
                            "start_col": start,
                            "values_re": [float(value.real) for value in values],
                            "values_im": [float(value.imag) for value in values],
                        }
                    )
                    start = col
                    values = [complex(basis_upper[row, col])]
                prev = col
            row_segments.append(
                {
                    "row": row,
                    "start_col": start,
                    "values_re": [float(value.real) for value in values],
                    "values_im": [float(value.imag) for value in values],
                }
            )
        return row_segments

    n = int(hamiltonian.rows)
    static_matrix = np.zeros((n, n), dtype=np.complex128)
    coefficient_terms: dict[str, tuple[smp.Expr, np.ndarray]] = {}
    for i in range(n):
        for j in range(i, n):
            expr = smp.expand(smp.factor_terms(smp.simplify(hamiltonian[i, j])))
            for term in smp.Add.make_args(expr):
                numeric, symbolic = split_numeric_factor(term)
                if abs(numeric) == 0.0:
                    continue
                if symbolic == 1:
                    static_matrix[i, j] += numeric
                    if i != j:
                        static_matrix[j, i] += np.conjugate(numeric)
                    continue
                key = smp.srepr(symbolic)
                if key not in coefficient_terms:
                    coefficient_terms[key] = (
                        symbolic,
                        np.zeros((n, n), dtype=np.complex128),
                    )
                coefficient_terms[key][1][i, j] += numeric

    coefficient_payload = []
    coefficient_basis_matrices: list[np.ndarray] = []
    basis_term_count = 0
    row_segment_count = 0
    for symbolic, basis_upper in coefficient_terms.values():
        coefficient_basis_matrices.append(basis_upper.copy())
        compiled = _compile_sympy_expr(
            symbolic,
            slot_index_by_name,
            tuple_value_names=tuple_value_names,
        )
        basis_terms = []
        for i in range(n):
            for j in range(i, n):
                value = complex(basis_upper[i, j])
                if value == 0:
                    continue
                basis_terms.append(
                    {
                        "i": i,
                        "j": j,
                        "re": float(value.real),
                        "im": float(value.imag),
                    }
                )
        basis_rows = group_basis_rows(basis_upper)
        basis_term_count += len(basis_terms)
        row_segment_count += len(basis_rows)
        coefficient_payload.append(
            {
                "expression": compiled.to_payload(),
                "basis_terms": basis_terms,
                "basis_row_segments": basis_rows,
            }
        )

    row_plans: list[dict[str, Any]] = []
    row_plan_segment_count = 0
    for row in range(n):
        active_cols = [
            col
            for col in range(row, n)
            if any(matrix[row, col] != 0 for matrix in coefficient_basis_matrices)
        ]
        if not active_cols:
            continue
        segments: list[dict[str, Any]] = []
        start = active_cols[0]
        prev = start
        current_cols = [start]
        for col in active_cols[1:]:
            if col == prev + 1:
                current_cols.append(col)
            else:
                coeff_indices: list[int] = []
                values_re: list[list[float]] = []
                values_im: list[list[float]] = []
                for coeff_index, matrix in enumerate(coefficient_basis_matrices):
                    values = [complex(matrix[row, value_col]) for value_col in current_cols]
                    if not any(value != 0 for value in values):
                        continue
                    coeff_indices.append(coeff_index)
                    values_re.append([float(value.real) for value in values])
                    values_im.append([float(value.imag) for value in values])
                segments.append(
                    {
                        "start_col": start,
                        "coeff_indices": coeff_indices,
                        "values_re": values_re,
                        "values_im": values_im,
                    }
                )
                start = col
                current_cols = [col]
            prev = col
        coeff_indices = []
        values_re = []
        values_im = []
        for coeff_index, matrix in enumerate(coefficient_basis_matrices):
            values = [complex(matrix[row, value_col]) for value_col in current_cols]
            if not any(value != 0 for value in values):
                continue
            coeff_indices.append(coeff_index)
            values_re.append([float(value.real) for value in values])
            values_im.append([float(value.imag) for value in values])
        segments.append(
            {
                "start_col": start,
                "coeff_indices": coeff_indices,
                "values_re": values_re,
                "values_im": values_im,
            }
        )
        row_plan_segment_count += len(segments)
        row_plans.append({"row": row, "segments": segments})

    diagnostics = {
        "representation": "decomposed",
        "entrywise_upper_count": n * (n + 1) // 2,
        "coefficient_count": len(coefficient_payload),
        "basis_term_count": basis_term_count,
        "basis_row_segment_count": row_segment_count,
        "row_plan_count": len(row_plans),
        "row_plan_segment_count": row_plan_segment_count,
        "static_nonzero_count": int(np.count_nonzero(static_matrix)),
    }
    if diagnostics["entrywise_upper_count"]:
        diagnostics["compression_ratio"] = (
            basis_term_count / diagnostics["entrywise_upper_count"]
        )
    else:
        diagnostics["compression_ratio"] = 0.0

    return {
        "kind": "decomposed",
        "n": n,
        "static_matrix": static_matrix,
        "coefficients": coefficient_payload,
        "row_plans": row_plans,
        "dense_fill_mode": "direct",
        "diagnostics": diagnostics,
    }


def _evaluate_expression(
    expression: CompiledExpression,
    slots: list[RuntimeValue],
    t: float,
    temps: list[RuntimeValue] | None = None,
) -> RuntimeValue:
    if temps is None:
        temps = []
    stack: list[RuntimeValue] = []
    for instr in expression.instructions:
        op = InstructionOp(instr["op"])
        if op == InstructionOp.CONST:
            stack.append(complex(instr["re"], instr["im"]))
        elif op == InstructionOp.SLOT:
            stack.append(slots[instr["index"]])
        elif op == InstructionOp.TEMP:
            stack.append(temps[instr["index"]])
        elif op == InstructionOp.TIME:
            stack.append(complex(t, 0.0))
        elif op == InstructionOp.ADD:
            right, left = stack.pop(), stack.pop()
            stack.append(_binary_elementwise(left, right, lambda a, b: a + b))
        elif op == InstructionOp.SUB:
            right, left = stack.pop(), stack.pop()
            stack.append(_binary_elementwise(left, right, lambda a, b: a - b))
        elif op == InstructionOp.MUL:
            right, left = stack.pop(), stack.pop()
            stack.append(_binary_elementwise(left, right, lambda a, b: a * b))
        elif op == InstructionOp.DIV:
            right, left = stack.pop(), stack.pop()
            stack.append(_binary_elementwise(left, right, lambda a, b: a / b))
        elif op == InstructionOp.POW:
            right, left = stack.pop(), stack.pop()
            stack.append(_binary_elementwise(left, right, lambda a, b: a**b))
        elif op == InstructionOp.NEG:
            value = stack.pop()
            if isinstance(value, tuple):
                stack.append(tuple(-item for item in value))
            else:
                stack.append(-value)
        elif op == InstructionOp.CONJ:
            value = stack.pop()
            if isinstance(value, tuple):
                stack.append(tuple(np.conjugate(item) for item in value))
            else:
                stack.append(np.conjugate(value))
        elif op == InstructionOp.BUILTIN_FUNC:
            argc = instr["argc"]
            args = stack[-argc:]
            del stack[-argc:]
            stack.append(_apply_builtin(instr["function"], args))
        elif op == InstructionOp.HELPER_FUNC:
            argc = instr["argc"]
            args = stack[-argc:]
            del stack[-argc:]
            stack.append(_apply_helper(instr["function"], args))
        elif op in {
            InstructionOp.GT,
            InstructionOp.GE,
            InstructionOp.LT,
            InstructionOp.LE,
            InstructionOp.EQ,
            InstructionOp.NE,
        }:
            right, left = stack.pop(), stack.pop()
            stack.append(_compare(left, right, op))
        else:
            raise ValueError(f"unsupported instruction op {op}")
    if len(stack) != 1:
        raise ValueError("expression evaluation did not end with a single stack value")
    return stack[0]


def evaluate_parameter_graph_py(parameter_graph: dict[str, Any], t: float) -> list[RuntimeValue]:
    slots = [decode_runtime_value(payload) for payload in parameter_graph["base_values"]]
    n_slots = len(parameter_graph["slot_names"])
    slots.extend([complex(0.0, 0.0)] * (n_slots - len(slots)))
    for compound in parameter_graph["compounds"]:
        expression = CompiledExpression.from_payload(compound["expression"])
        slots[compound["slot"]] = _evaluate_expression(expression, slots, t)
    return slots


def fill_hamiltonian_py(
    hamiltonian_plan: dict[str, Any],
    parameter_values: list[RuntimeValue],
    t: float,
) -> np.ndarray:
    kind = hamiltonian_plan.get("kind", "entrywise")
    n = int(hamiltonian_plan["n"])
    if kind == "decomposed":
        matrix = np.array(hamiltonian_plan["static_matrix"], dtype=np.complex128, copy=True)
        for coefficient in hamiltonian_plan["coefficients"]:
            expression = CompiledExpression.from_payload(coefficient["expression"])
            value = _as_scalar(_evaluate_expression(expression, parameter_values, t))
            if "basis_row_segments" in coefficient:
                for segment in coefficient["basis_row_segments"]:
                    row = int(segment["row"])
                    start_col = int(segment["start_col"])
                    values = [
                        complex(re, im)
                        for re, im in zip(segment["values_re"], segment["values_im"])
                    ]
                    for offset, entry in enumerate(values):
                        col = start_col + offset
                        contrib = value * entry
                        matrix[row, col] += contrib
                        if row != col:
                            matrix[col, row] += np.conjugate(contrib)
            elif "basis_terms" in coefficient:
                for term in coefficient["basis_terms"]:
                    i = int(term["i"])
                    j = int(term["j"])
                    entry = complex(term["re"], term["im"])
                    contrib = value * entry
                    matrix[i, j] += contrib
                    if i != j:
                        matrix[j, i] += np.conjugate(contrib)
            else:
                basis_upper = np.asarray(coefficient["basis_upper"], dtype=np.complex128)
                for i in range(n):
                    for j in range(i, n):
                        entry = basis_upper[i, j]
                        if entry == 0:
                            continue
                        contrib = value * entry
                        matrix[i, j] += contrib
                        if i != j:
                            matrix[j, i] += np.conjugate(contrib)
        return matrix

    matrix = np.zeros((n, n), dtype=np.complex128)
    temps: list[RuntimeValue] = []
    for temp_payload in hamiltonian_plan["temps"]:
        expression = CompiledExpression.from_payload(temp_payload)
        temps.append(_evaluate_expression(expression, parameter_values, t, temps))
    for entry in hamiltonian_plan["entries"]:
        expression = CompiledExpression.from_payload(entry["expression"])
        value = _as_scalar(_evaluate_expression(expression, parameter_values, t, temps))
        i = int(entry["i"])
        j = int(entry["j"])
        matrix[i, j] = value
        if i != j:
            matrix[j, i] = np.conjugate(value)
    return matrix
