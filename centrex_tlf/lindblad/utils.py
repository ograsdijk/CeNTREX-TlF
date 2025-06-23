import numpy as np
import sympy as smp

__all__ = ["generate_density_matrix_symbolic"]


def recursive_subscript(i: int) -> str:
    # chr(0x2080+i) is unicode for
    # subscript num(i), resulting in x₀₀ for example
    if i < 10:
        return chr(0x2080 + i)
    else:
        return recursive_subscript(i // 10) + chr(0x2080 + i % 10)


def generate_density_matrix_symbolic(
    levels: int,
) -> smp.matrices.dense.MutableDenseMatrix:
    """
    Generate a symbolic density matrix for a given number of levels.
    This function creates a symbolic representation of a density matrix
    with dimensions `levels x levels`. Each element of the matrix is
    represented as a symbolic variable using SymPy. The diagonal elements
    are assigned unique symbols, while the off-diagonal elements are
    symmetrically assigned corresponding symbols.
    Args:
        levels (int): The number of levels in the system, which determines
                      the dimensions of the density matrix.
    Returns:
        smp.matrices.dense.MutableDenseMatrix: A symbolic density matrix
        with dimensions `levels x levels`, where each element is a SymPy
        symbolic variable.
    Notes:
        - The Unicode character `\u03c1` (ρ) is used as a prefix for the
          symbolic variable names.
        - The `recursive_subscript` function is used to format the indices
          of the symbolic variables.
    """
    ρ = smp.zeros(levels, levels)
    levels = levels
    for i in range(levels):
        for j in range(i, levels):
            # \u03C1 is unicode for ρ,
            if i == j:
                ρ[i, j] = smp.Symbol(
                    "\u03c1{0},{1}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
            else:
                ρ[i, j] = smp.Symbol(
                    "\u03c1{0},{1}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
                ρ[j, i] = smp.Symbol(
                    "\u03c1{1},{0}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
    return ρ


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
