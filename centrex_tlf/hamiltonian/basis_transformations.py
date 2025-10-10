from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf.states import BasisState

__all__ = ["generate_transform_matrix"]


def generate_transform_matrix(
    basis1: Union[Sequence[BasisState], npt.NDArray],
    basis2: Union[Sequence[BasisState], npt.NDArray],
) -> npt.NDArray[np.complex128]:
    """Generate transformation matrix between two quantum state bases.
    
    Computes the transformation matrix S that converts operators from basis1 to basis2:
        H₂ = S† @ H₁ @ S
    
    where S[i,j] = ⟨basis1[i]|basis2[j]⟩ is the overlap matrix between the two bases.
    
    Args:
        basis1 (Sequence[BasisState] | npt.NDArray): Sequence or array of basis states 
            defining the first basis
        basis2 (Sequence[BasisState] | npt.NDArray): Sequence or array of basis states 
            defining the second basis
    
    Returns:
        npt.NDArray[np.complex128]: Transformation matrix S of shape (n, n) that 
            converts operators from basis1 to basis2
    
    Raises:
        ValueError: If the two bases have different dimensions or if either basis is empty
    
    Example:
        >>> # Transform uncoupled to coupled basis
        >>> S = generate_transform_matrix(uncoupled_basis, coupled_basis)
        >>> H_coupled = S.conj().T @ H_uncoupled @ S
    
    Note:
        For large bases (n > 100), consider using numba JIT compilation or 
        vectorized operations for better performance.
    """
    # Input validation
    n1 = len(basis1)
    n2 = len(basis2)
    
    if n1 == 0 or n2 == 0:
        raise ValueError("Both bases must be non-empty")
    
    if n1 != n2:
        raise ValueError(
            f"Bases must have the same dimension: basis1 has {n1} states, "
            f"basis2 has {n2} states"
        )
    
    # Initialize transformation matrix
    n = n1
    S = np.zeros((n, n), dtype=np.complex128)

    # Calculate inner products: S[i,j] = ⟨basis1[i]|basis2[j]⟩
    for i, state1 in enumerate(basis1):
        for j, state2 in enumerate(basis2):
            S[i, j] = state1 @ state2

    return S
