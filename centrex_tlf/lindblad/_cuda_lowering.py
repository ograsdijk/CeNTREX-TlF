"""Lower the existing scalar parameter IR to CUDA, without model-specific slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as smp

from .ir import BuiltinFunctionId as Builtin
from .ir import InstructionOp as Op

TIME = smp.Symbol("cuda_time", real=True)


@dataclass(eq=False)
class Node:
    op: str
    args: tuple
    symbolic: Any
    dependencies: frozenset[int]


def constant(value: complex) -> Node:
    value = complex(value)
    return Node(
        "const", (value,), smp.Float(value.real) + smp.I * smp.Float(value.imag), frozenset()
    )


class ScalarCompiler:
    """A small typed DAG; tuple values stay compile-time tuple structures."""

    def __init__(self, graph: dict, expressions: list[dict]):
        self.graph = graph
        self.interpolators: dict[tuple, str] = {}
        self.helpers: list[str] = []
        self.switches: list[tuple[Any, Any, Any]] = []
        self.slots: list[Any] = [None] * len(graph["slot_names"])
        for index, payload in enumerate(graph["base_values"]):
            if payload["kind"] == "tuple":
                self.slots[index] = tuple(
                    constant(complex(v["re"], v["im"])) for v in payload["items"]
                )
            else:
                self.slots[index] = Node(
                    "param", (index,), smp.Symbol(f"p{index}", real=True), frozenset([index])
                )
        for compound in graph["compounds"]:
            self.slots[compound["slot"]] = self.expression(compound["expression"])
        self.coefficients = [constant(1)] + [self.expression(expr) for expr in expressions]
        if any(isinstance(c, tuple) for c in self.coefficients):
            raise ValueError("CUDA Hamiltonian coefficients must be scalar")

    def make(self, op: str, args: tuple, symbolic: Any) -> Node:
        dependencies = frozenset().union(*(a.dependencies for a in args if isinstance(a, Node)))
        return Node(op, args, symbolic, dependencies)

    def binary(self, op: Op, a: Any, b: Any) -> Any:
        if isinstance(a, tuple) or isinstance(b, tuple):
            if isinstance(a, tuple) and isinstance(b, tuple) and len(a) != len(b):
                raise ValueError("CUDA tuple operand lengths differ")
            count = len(a) if isinstance(a, tuple) else len(b)
            return tuple(
                self.binary(
                    op, a[k] if isinstance(a, tuple) else a, b[k] if isinstance(b, tuple) else b
                )
                for k in range(count)
            )
        operations = {
            Op.ADD: lambda x, y: x + y,
            Op.SUB: lambda x, y: x - y,
            Op.MUL: lambda x, y: x * y,
            Op.DIV: lambda x, y: x / y,
            Op.POW: lambda x, y: x**y,
            Op.GT: smp.Gt,
            Op.GE: smp.Ge,
            Op.LT: smp.Lt,
            Op.LE: smp.Le,
            Op.EQ: smp.Eq,
            Op.NE: smp.Ne,
        }
        return self.make(op.name.lower(), (a, b), operations[op](a.symbolic, b.symbolic))

    def expression(self, payload: dict) -> Any:
        stack = []
        for inst in payload["instructions"]:
            op = Op(inst["op"])
            if op == Op.CONST:
                stack.append(constant(complex(inst["re"], inst["im"])))
            elif op == Op.SLOT:
                value = self.slots[inst["index"]]
                if value is None:
                    raise ValueError("CUDA parameter graph is not topologically ordered")
                stack.append(value)
            elif op == Op.TIME:
                stack.append(Node("time", (), TIME, frozenset()))
            elif op in {Op.NEG, Op.CONJ}:
                value = stack.pop()

                def one(x, op=op):
                    symbolic = -x.symbolic if op == Op.NEG else smp.conjugate(x.symbolic)
                    return self.make(op.name.lower(), (x,), symbolic)

                stack.append(
                    tuple(one(x) for x in value) if isinstance(value, tuple) else one(value)
                )
            elif op in {Op.BUILTIN_FUNC, Op.HELPER_FUNC}:
                count = inst.get("argc", 1)
                args = tuple(stack[-count:])
                del stack[-count:]
                fid = inst["function"]
                if op == Op.BUILTIN_FUNC:
                    funcs = {
                        1: smp.sin,
                        2: smp.cos,
                        3: smp.tan,
                        4: smp.exp,
                        5: smp.Abs,
                        6: smp.re,
                        7: smp.im,
                    }
                    node = self.make("builtin", (fid, *args), funcs[fid](args[0].symbolic))
                else:
                    if fid not in range(1, 17):
                        raise ValueError(f"unsupported CUDA helper function {fid}")
                    symargs = [
                        smp.Tuple(*(v.symbolic for v in a)) if isinstance(a, tuple) else a.symbolic
                        for a in args
                    ]
                    node = self.make(
                        "helper", (fid, *args), smp.Function(f"helper_{fid}")(*symargs)
                    )
                    # Include dependencies inside tuple arguments as well.
                    node.dependencies = frozenset().union(
                        *(
                            v.dependencies
                            for a in args
                            for v in (a if isinstance(a, tuple) else (a,))
                        )
                    )
                    x = [a.symbolic if isinstance(a, Node) else None for a in args]
                    if fid in {4, 6}:
                        self.switches.append(
                            (x[0] * x[1] + x[2], smp.pi if fid == 4 else 2 * smp.pi, 0)
                        )
                    elif fid == 7:
                        phase = x[0] / (x[1] + x[2]) + x[3] / (2 * smp.pi)
                        self.switches.extend([(phase, 1, 0), (phase, 1, x[1] / (x[1] + x[2]))])
                    elif fid == 12:
                        phase = x[0] * x[2] + x[3] / (2 * smp.pi)
                        self.switches.extend([(phase, 1, 0), (phase, 1, x[1])])
                    elif fid == 13:
                        self.switches.append(((x[0] - x[1]) / x[2], 1, 0))
                stack.append(node)
            elif op == Op.TEMP:
                raise ValueError(
                    "CUDA decomposed coefficients cannot reference entrywise temporaries"
                )
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(self.binary(op, a, b))
        if len(stack) != 1:
            raise ValueError("invalid CUDA scalar expression stack")
        return stack[0]

    def factor_multipass(self) -> None:
        """Expose exact transverse factors for unrotated, common-height passes."""
        memo = {}

        def rewrite(node):
            if isinstance(node, tuple):
                return tuple(rewrite(n) for n in node)
            if node in memo:
                return memo[node]
            args = tuple(rewrite(a) if isinstance(a, Node | tuple) else a for a in node.args)
            result = Node(node.op, args, node.symbolic, node.dependencies)
            if node.op == "helper" and args[0] in {8, 10}:
                fid, *a = args
                centers = a[4]
                # Equality of the expressions, not equality at a sampled point.
                if (
                    isinstance(centers, tuple)
                    and centers
                    and all(y.symbolic == centers[0].symbolic for y in centers)
                    and not any(x.symbolic.has(TIME) for x in (a[1], centers[0], a[6]))
                    and all(x.symbolic.is_real is True for x in (a[1], centers[0], a[6]))
                ):
                    y, sy = a[1], a[6]
                    z = self.binary(Op.DIV, self.binary(Op.SUB, y, centers[0]), sy)
                    exponent = self.binary(
                        Op.MUL, constant(-0.5 if fid == 8 else -0.25), self.binary(Op.MUL, z, z)
                    )
                    attenuation = self.make(
                        "builtin", (int(Builtin.EXP), exponent), smp.exp(exponent.symbolic)
                    )
                    a[1] = centers[0]
                    on_axis = self.make(
                        "helper",
                        (fid, *a),
                        smp.Function(f"helper_{fid}")(
                            *[
                                smp.Tuple(*(x.symbolic for x in v))
                                if isinstance(v, tuple)
                                else v.symbolic
                                for v in a
                            ]
                        ),
                    )
                    on_axis.dependencies = frozenset().union(
                        *(x.dependencies for v in a for x in (v if isinstance(v, tuple) else (v,)))
                    )
                    result = self.binary(Op.MUL, on_axis, attenuation)
            memo[node] = result
            return result

        self.coefficients = [rewrite(n) for n in self.coefficients]

    def function(
        self,
        name: str,
        indices: list[int],
        *,
        roots=None,
        substitutions=None,
        extra_arguments="",
        root_times=None,
    ) -> str:
        lines, memo = [], {}

        def render(node: Any, time_name: str = "stage_t") -> Any:
            if isinstance(node, tuple):
                return tuple(render(n, time_name) for n in node)
            if substitutions and (node, time_name) in substitutions:
                return substitutions[node, time_name]
            key = (node, time_name)
            if key in memo:
                return memo[key]
            op, args = node.op, node.args
            if op == "const":
                value = args[0]
                return f"C({value.real:.17g},{value.imag:.17g})"
            if op == "param":
                return f"params[{args[0]}*B+b]"
            if op == "time":
                return f"C({time_name},0.)"
            # Discontinuous helpers use the same interior value at every stage
            # of a clipped step, including both boundary derivatives.
            tn = "switch_t" if op == "helper" and args[0] in {4, 6, 7, 12, 13} else time_name
            if op in {"helper", "builtin"}:
                values = [render(a, tn) for a in args[1:]]
                if op == "builtin":
                    fid = Builtin(args[0])
                    x = values[0]
                    functions = {
                        Builtin.SIN: "sin",
                        Builtin.COS: "cos",
                        Builtin.TAN: "tan",
                        Builtin.EXP: "exp",
                    }
                    code = (
                        f"{functions[fid]}({x})"
                        if fid in functions
                        else f"C(thrust::abs({x}),0.)"
                        if fid == Builtin.ABS
                        else f"C({x}.real(),0.)"
                        if fid == Builtin.REAL
                        else f"C({x}.imag(),0.)"
                    )
                else:
                    code = self.helper(args[0], values, node)
            else:
                values = [render(a, time_name) for a in args]
                if op in {"neg", "conj"}:
                    code = f"-{values[0]}" if op == "neg" else f"conj({values[0]})"
                elif op == "pow":
                    code = f"thrust::pow({values[0]},{values[1]})"
                elif op in {"gt", "ge", "lt", "le", "eq", "ne"}:
                    cmp = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}[op]
                    code = (
                        f"C(double(real_arg({values[0]},error){cmp}real_arg({values[1]},error)),0.)"
                    )
                else:
                    code = f"({values[0]}{ {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[op] }{values[1]})"
            var = f"v{len(memo)}"
            memo[key] = var
            lines.append(f"C {var}={code};")
            return var

        for k in indices:
            result = render(
                (self.coefficients if roots is None else roots)[k],
                "stage_t" if root_times is None else root_times[k],
            )
            lines.append(f"out[{k}*S+(S==1?0:b)]={result};")
        return (
            f"__device__ void {name}(const C* params,int B,int b,double stage_t,double switch_t,C* out,int S,int* error{extra_arguments}){{\n"
            + "\n".join(lines)
            + "\n}\n"
        )

    def helper(self, fid: int, values: list, node: Node) -> str:
        a = [
            tuple(f"real_arg({x},error)" for x in v)
            if isinstance(v, tuple)
            else f"real_arg({v},error)"
            for v in values
        ]
        if fid == 1:
            return f"C(g2({','.join(a)}),0.)"
        if fid == 2:
            return f"C(g2rot({','.join(a)}),0.)"
        if fid == 3:
            return f"phase_mod({','.join(a)})"
        if fid == 4:
            return f"C(sin({a[1]}*{a[0]}+{a[2]})>=0.?1.:0.,0.)"
        if fid == 5:
            return f"pol_mod({','.join(a)})"
        if fid == 6:
            return f"C(frac(({a[1]}*{a[0]}+{a[2]})/(2.*PI)),0.)"
        if fid == 7:
            return f"C(frac({a[0]}/({a[1]}+{a[2]})+{a[3]}/(2.*PI))<{a[1]}/({a[1]}+{a[2]})?1.:0.,0.)"
        if fid == 12:
            return f"C(gate_duty({','.join(a)}),0.)"
        if fid == 13:
            return f"C(frac(floor(({a[0]}-{a[1]})/{a[2]})/2.)==0.?1.:-1.,0.)"
        if fid == 15:
            return f"C(g1({','.join(a)}),0.)"
        if fid in {14, 16}:
            try:
                grid = tuple(complex(x.symbolic) for x in node.args[2])
                vals = tuple(complex(x.symbolic) for x in node.args[3])
                if any(x.imag != 0 for x in grid + vals):
                    raise ValueError("complex interpolation data")
                grid, vals = tuple(x.real for x in grid), tuple(x.real for x in vals)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "CUDA interpolation requires constant real grid/value tuples"
                ) from exc
            if (
                not grid
                or len(grid) != len(vals)
                or not np.isfinite(grid + vals).all()
                or np.any(np.diff(grid) <= 0)
            ):
                raise ValueError(
                    "CUDA interpolation requires finite, equally sized tuples and an increasing grid"
                )
            key = (fid, grid, vals)
            if key not in self.interpolators:
                name = f"interpolate_{len(self.interpolators)}"
                self.interpolators[key] = name
                body = f"if(x<={grid[0]:.17g})return {vals[0]:.17g};if(x>={grid[-1]:.17g})return {vals[-1]:.17g};"
                if fid == 16 and len(grid) > 1:
                    from scipy.interpolate import PchipInterpolator

                    coeff = PchipInterpolator(grid, vals).c
                for i in range(len(grid) - 1):
                    if fid == 14:
                        expression = f"({vals[i]:.17g})+(x-({grid[i]:.17g}))*({(vals[i + 1] - vals[i]) / (grid[i + 1] - grid[i]):.17g})"
                    else:
                        c = coeff[:, i]
                        expression = (
                            f"((({c[0]:.17g})*d+({c[1]:.17g}))*d+({c[2]:.17g}))*d+({c[3]:.17g})"
                        )
                    body += f"if(x<={grid[i + 1]:.17g}){{double d=x-({grid[i]:.17g});return {expression};}}"
                self.helpers.append(
                    f"__device__ double {name}(double x){{{body}return {vals[-1]:.17g};}}"
                )
            return f"C({self.interpolators[key]}({a[0]}),0.)"
        from centrex_tlf.constants import ED_XtB

        def rabi(intensity, coupling, dipole):
            return f"sqrt(({intensity})*2./(299792458.*8.8541878128e-12))*({coupling})*({dipole})/1.0545718176461565e-34"

        if fid == 9:
            return f"C({rabi(a[0], a[1], a[2] if len(a) > 2 else repr(float(ED_XtB)))},0.)"
        if fid == 11:
            intensity = f"g2({','.join(a[:7])})"
            return f"C({rabi(intensity, a[7], a[8] if len(a) > 8 else repr(float(ED_XtB)))},0.)"
        if fid in {8, 10}:
            if (
                not all(isinstance(a[k], tuple) for k in [2, 3, 4])
                or len({len(a[k]) for k in [2, 3, 4]}) != 1
            ):
                raise ValueError(
                    "CUDA multipass amplitudes and locations must be equally sized tuples"
                )
            intensity = (
                "("
                + "+".join(
                    f"g2({a[0]},{a[1]},{amp},{x},{y},{a[5]},{a[6]})"
                    for amp, x, y in zip(a[2], a[3], a[4], strict=False)
                )
                + ")"
                if a[2]
                else "0."
            )
            return f"C({intensity if fid == 8 else rabi(intensity, a[7], a[8] if len(a) > 8 else repr(float(ED_XtB)))},0.)"
        raise ValueError(f"unsupported CUDA helper {fid}")

    def discontinuities(
        self, parameters: np.ndarray, span: tuple[float, float], explicit: Any
    ) -> np.ndarray:
        supplied = np.asarray([] if explicit is None else explicit, dtype=float)
        if supplied.ndim != 1 or not np.isfinite(supplied).all():
            raise ValueError("CUDA discontinuities must be a finite one-dimensional sequence")
        edges = list(supplied[(supplied > span[0]) & (supplied < span[1])])
        symbols = [smp.Symbol(f"p{k}", real=True) for k in range(len(self.graph["base_values"]))]
        for phase, period, offset in self.switches:
            phase = smp.expand(phase)
            slope = smp.diff(phase, TIME)
            intercept = phase.subs(TIME, 0)
            if TIME in slope.free_symbols or TIME in smp.sympify(offset).free_symbols:
                if explicit is None:
                    raise ValueError("nonlinear CUDA switching requires explicit discontinuities")
                continue
            functions = [
                smp.lambdify(symbols, x, "numpy") for x in [slope, intercept, period, offset]
            ]
            columns = [parameters[:, k] for k in range(parameters.shape[1])]
            vals = [
                np.broadcast_to(np.asarray(f(*columns), complex), (len(parameters),))
                for f in functions
            ]
            if any(
                not np.isfinite(v).all() or np.max(abs(v.imag), initial=0) > 1e-12 for v in vals
            ):
                raise ValueError("CUDA switch frequencies and phases must be finite and real")
            for a, c, p, o in np.unique(np.stack(vals, axis=1).real, axis=0):
                if p <= 0:
                    raise ValueError("CUDA switching period must be positive")
                if a == 0:
                    continue
                lo, hi = sorted(((a * span[0] + c - o) / p, (a * span[1] + c - o) / p))
                if hi - lo > 10_000_000:
                    raise ValueError("too many CUDA switching boundaries")
                k = np.arange(np.ceil(lo), np.floor(hi) + 1)
                t = (k * p + o - c) / a
                edges.extend(t[(t > span[0]) & (t < span[1])])
        return np.unique(edges)
