from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

import sympy as smp
from sympy.matrices.dense import MutableDenseMatrix


@dataclass
class DarkBrightTransformation:
    n_levels: int
    a_symbols: list
    Δ_symbols: list
    T: MutableDenseMatrix


def generate_transform_matrix_symbolic(
    n_levels: int = 4, ind_T_zero: Sequence[int] = [0, 2]
) -> DarkBrightTransformation:
    """Generate symbolic transformation to dark/bright state basis.
    
    Constructs a symbolic unitary transformation matrix T that transforms a system
    of n_levels states (n_levels-1 ground states coupled to 1 excited state) into
    a dark/bright basis. In this basis:
    - 1 bright state couples to the excited state
    - n_levels-2 dark states have no direct coupling to the excited state
    
    The transformation diagonalizes the coupling structure to identify dark states
    that are decoupled from the optical field.
    
    Args:
        n_levels (int): Total number of quantum levels (ground + excited). 
            Defaults to 4.
        ind_T_zero (Sequence[int]): Indices [i, j] to set T[i,j]=0, making state i 
            the "darkest" with no direct coupling. Defaults to [0, 2].
    
    Returns:
        DarkBrightTransformation: Dataclass containing:
            - n_levels: Number of levels
            - a_symbols: Symbolic coupling amplitudes [a₀, a₁, ...]
            - Δ_symbols: Symbolic detunings [Δ₀, Δ₁, ...]
            - T: Symbolic unitary transformation matrix
    
    Note:
        The transformation satisfies T†T = I (unitarity) and creates a basis where
        only one state (the bright state) couples to the excited level.
    """
    δ = smp.symbols("δ", real=True)
    Ω = smp.Symbol("Ω", real=True)

    # T = smp.MatrixSymbol('T',,4)
    T = smp.zeros(n_levels, n_levels)
    for idx, idy in product(*[range(n_levels - 1)] * 2):
        T[idx, idy] = smp.Symbol(f"T{idx}{idy}", real=True)
    T[n_levels - 1, n_levels - 1] = 1

    a = smp.symbols([f"a_{i}" for i in range(n_levels - 1)], real=True)
    Δ = smp.symbols([f"Δ_{i}" for i in range(n_levels - 1)], real=True)

    H = smp.zeros(n_levels, n_levels)
    for i in range(n_levels - 1):
        H[i, i] = -δ + Δ[i]

    for i in range(n_levels - 1):
        H[i, n_levels - 1] = 1 / 2 * a[i] * Ω
        H[n_levels - 1, i] = smp.conjugate(H[i, n_levels - 1])

    transformed = smp.simplify(T.T @ H @ T)

    # ensure one bright state and n_levels-2 dark states
    eqns = [smp.Eq(transformed[i, n_levels - 1], 0) for i in range(1, n_levels - 1)]

    sol = smp.solve(eqns)

    T = T.subs(sol[0])

    # ensure the transformation matrix is unitary
    eqns = []

    # set a coupling to zero to the first state, such that this state is the
    # 'darkest' state with no direct coupling to the bright state
    if (n_levels > 3) & (ind_T_zero is not None):
        eqns.append(smp.Eq(T[ind_T_zero[0], ind_T_zero[1]], 0))

    for i in range(n_levels - 1):
        eqns.append(smp.Eq(T[:, i].dot(T[:, i]), 1))
    for i, j in combinations(range(n_levels - 1), 2):
        eqns.append(smp.Eq(T[:, i].dot(T[:, j]), 0))

    solve_symbols = [sym for sym in T.free_symbols if sym not in [Ω, δ, *a, *Δ]]
    # for some reason this order of symbols solves faster
    solve_symbols = [
        T[i, j]
        for i, j in product(*[range(n_levels - 1)] * 2)
        if T[i, j] in solve_symbols
    ]

    sol = smp.solve(eqns, *solve_symbols)
    T = T.subs([(s, v) for s, v in zip(solve_symbols, sol[0])])

    return DarkBrightTransformation(n_levels, a, Δ, T)
    return DarkBrightTransformation(n_levels, a, Δ, T)
