import numpy as np
import sympy as smp

__all__: list[str] = []


def has_off_diagonal_elements(arr: np.ndarray, tol: float = 0.0) -> bool:
    """
    Check if a square NumPy array has any nonzero off-diagonal elements.

    Parameters:
        arr (np.ndarray): A 2D square NumPy array.
        tol (float): Optional tolerance. Any absolute value larger than this is considered nonzero.

    Returns:
        bool: True if any off-diagonal element has absolute value > tol, False otherwise.
    """
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Array must be square")

    off_diag = arr.copy()
    np.fill_diagonal(off_diag, 0)

    return bool(np.any(np.abs(off_diag) > tol))


def strip_float_ones(expr: smp.Expr) -> smp.Expr:
    """
    Remove multiplicative Float(±1.0) factors in products, turning
    1.0*x -> x and -1.0*x -> -x, without touching other floats.
    """
    expr = smp.sympify(expr)

    def rec(e: smp.Expr) -> smp.Expr:
        if e.is_Atom:
            return e

        if isinstance(e, smp.Mul):
            coeff, rest = e.as_coeff_Mul(rational=False)

            # Recurse into the non-numeric part
            rest2 = rec(rest)

            # coeff might be Integer(-1) * Float(1.0), etc.
            coeff2 = smp.sympify(coeff)

            # If coeff is exactly ±1.0 as a Float, drop it
            if isinstance(coeff2, smp.Float):
                if coeff2 == smp.Float(1.0):
                    return rest2
                if coeff2 == smp.Float(-1.0):
                    return -rest2

            # If coeff is something like -1.0 (Float) *something already handled above,
            # otherwise keep it
            return coeff2 * rest2

        return e.func(*[rec(a) for a in e.args])

    return rec(expr)
