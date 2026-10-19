"""Conservative structural periodicity checks on the CUDA scalar graph."""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as smp

from ._cuda_lowering import TIME, Node, ScalarCompiler


def assert_periodic(
    compiler: ScalarCompiler, parameters: np.ndarray, period: float, active: np.ndarray
) -> None:
    """Prove supported periodic compositions; reject unproven time dependence.

    Sampling a Gaussian at a few points cannot establish periodicity. Instead
    check constant expressions, periodic compositions, and commensurate affine
    phases of the supported trigonometric and switching helpers.
    """
    substitutions = {smp.Symbol(f"p{k}", real=True): complex(v) for k, v in enumerate(parameters)}
    memo: dict[Node, bool] = {}

    def expr(node: Node) -> Any:
        return smp.simplify(node.symbolic.subs(substitutions))

    def winding(phase: Any, cycle: float) -> bool:
        phase = smp.expand(phase)
        slope = smp.diff(phase, TIME)
        if TIME in slope.free_symbols:
            return False
        try:
            q = complex(slope * period / cycle)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(q) and abs(q.imag) < 1e-12 and abs(q.real - round(q.real)) <= 1e-10)

    def check(node: Any) -> bool:
        if isinstance(node, tuple):
            return all(check(x) for x in node)
        if not isinstance(node, Node):
            return True
        if node in memo:
            return memo[node]
        if TIME not in expr(node).free_symbols:
            memo[node] = True
            return True
        op, args = node.op, node.args
        result = False
        if op in {"helper", "builtin"}:
            fid = int(args[0])
            values = args[1:]
            result = all(check(x) for x in values)
            if not result and op == "builtin" and fid in {1, 2, 3, 4}:
                phase = expr(values[0])
                result = winding(
                    phase / (smp.I if fid == 4 else 1), np.pi if fid == 3 else 2 * np.pi
                )
            if not result and op == "helper" and fid in {3, 4, 5, 6, 7, 12, 13}:
                a = [expr(x) for x in values]
                if all(TIME not in x.free_symbols for x in a[1:]):
                    if fid in {3, 5}:
                        result = a[1] == 0 or winding(a[0] * a[2], 2 * np.pi)
                    elif fid in {4, 6}:
                        result = winding(a[0] * a[1] + a[2], 2 * np.pi)
                    elif fid == 7:
                        result = winding(a[0] / (a[1] + a[2]) + a[3] / (2 * smp.pi), 1.0)
                    elif fid == 12:
                        result = winding(a[0] * a[2] + a[3] / (2 * smp.pi), 1.0)
                    else:
                        result = winding((a[0] - a[1]) / a[2], 2.0)
        elif op not in {"time", "param", "const"}:
            result = all(check(x) for x in args)
        memo[node] = bool(result)
        return bool(result)

    for k, node in enumerate(compiler.coefficients):
        if active[k] and not check(node):
            raise ValueError(
                "periodic map cannot establish periodicity of Hamiltonian coefficient "
                f"{k}; use constant envelopes and commensurate drives, or an ODE solve"
            )
