from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import cmath
from typing import Any

import numpy as np
import sympy as smp
from sympy.core.relational import Relational

from .helper_functions import HELPER_FUNCTION_IDS, HELPER_FUNCTION_NAMES, HELPER_FUNCTIONS

__all__ = [
    "BuiltinFunctionId",
    "CompiledExpression",
    "InstructionOp",
    "evaluate_parameter_graph_py",
    "fill_hamiltonian_py",
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
    assert isinstance(right, tuple)
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
    return {
        "slot_names": parameters.slot_names,
        "n_base": len(base_values),
        "base_values": base_values,
        "compounds": compounds,
    }


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
