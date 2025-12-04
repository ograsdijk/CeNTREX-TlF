from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf.states import (
    BasisState,
    CoupledBasisState,
    UncoupledBasisState,
    UncoupledState,
)

try:
    from ..centrex_tlf_rust import (
        generate_transform_matrix_py as _generate_transform_matrix_rust,
    )

    HAS_RUST = True
except ImportError:
    _generate_transform_matrix_rust = None  # type: ignore[assignment]
    HAS_RUST = False


__all__ = ["generate_transform_matrix"]


def _generate_transform_matrix_original(
    basis1: Union[Sequence[BasisState], npt.NDArray],
    basis2: Union[Sequence[BasisState], npt.NDArray],
) -> npt.NDArray[np.complex128]:
    """Naive Python implementation of the transformation matrix generation.

    This implementation iterates over all pairs of states and computes the inner product.
    It is used as a fallback when the optimized Python implementation cannot be used
    (e.g. when bases contain repeated states or mixed types that are not handled).

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


def _generate_transform_matrix_python(
    basis1: Union[Sequence[BasisState], npt.NDArray],
    basis2: Union[Sequence[BasisState], npt.NDArray],
) -> npt.NDArray[np.complex128]:
    """Optimized Python implementation with caching and simple fast paths.

    This function is used when the Rust extension is not available.
    It includes optimizations for:
    1. Identity transform (same basis object)
    2. Permutation transform (same basis states, different order)
    3. Caching of state transformations for mixed coupled/uncoupled bases

    Uses only the public BasisState / State API so changes to internals
    of the state classes should not break behavior.

    Args:
        basis1 (Sequence[BasisState] | npt.NDArray): Sequence or array of basis states
            defining the first basis
        basis2 (Sequence[BasisState] | npt.NDArray): Sequence or array of basis states
            defining the second basis

    Returns:
        npt.NDArray[np.complex128]: Transformation matrix S
    """
    b1 = np.asarray(basis1, dtype=object)
    b2 = np.asarray(basis2, dtype=object)

    n1 = b1.size
    n2 = b2.size

    if n1 == 0 or n2 == 0:
        raise ValueError("Both bases must be non-empty")

    if n1 != n2:
        raise ValueError(
            f"Bases must have the same dimension: basis1 has {n1} states, "
            f"basis2 has {n2} states"
        )

    n = n1

    # Trivial fast path: exactly the same array object
    if b1 is b2:
        return np.eye(n, dtype=np.complex128)

    # Detect homogeneous basis types
    all1_unc = all(isinstance(s, UncoupledBasisState) for s in b1)
    all2_unc = all(isinstance(s, UncoupledBasisState) for s in b2)
    all1_cpl = all(isinstance(s, CoupledBasisState) for s in b1)
    all2_cpl = all(isinstance(s, CoupledBasisState) for s in b2)

    # Same basis type on both sides: try permutation fast path
    if (all1_unc and all2_unc) or (all1_cpl and all2_cpl):
        index2: dict[BasisState, int] = {}
        for j, s in enumerate(b2):
            index2[s] = j

        # Require that each state appears exactly once in each basis
        # so the overlap is just a permutation matrix
        perm_ok = len(index2) == n and all(s in index2 for s in b1)
        if perm_ok:
            S = np.zeros((n, n), dtype=np.complex128)
            for i, s in enumerate(b1):
                j = index2[s]
                S[i, j] = 1.0
            return S

        # Same type but not a simple permutation (e.g. repeated states)
        # fall back to original behavior
        return _generate_transform_matrix_original(b1, b2)

    # Mixed coupled/uncoupled cases: cache transform_to_uncoupled and single-state wrappers
    S = np.empty((n, n), dtype=np.complex128)

    # Cache for coupled -> uncoupled expansions
    unc_cache: dict[CoupledBasisState, UncoupledState] = {}
    single_unc_cache: dict[UncoupledBasisState, UncoupledState] = {}

    def as_single_unc_state(u: UncoupledBasisState) -> UncoupledState:
        """Return |u⟩ as an UncoupledState, cached."""
        st = single_unc_cache.get(u)
        if st is None:
            st = UncoupledState([(1.0, u)])
            single_unc_cache[u] = st
        return st

    def coupled_to_unc_state(c: CoupledBasisState) -> UncoupledState:
        """Return uncoupled expansion of a coupled basis state, cached."""
        st = unc_cache.get(c)
        if st is None:
            st = c.transform_to_uncoupled()
            unc_cache[c] = st
        return st

    if all1_cpl and all2_unc:
        # S[i, j] = basis1[i] @ basis2[j] for coupled_i, uncoupled_j
        # Original semantics:
        #   CoupledBasisState.__matmul__(UncoupledBasisState) ->
        #       UncoupledState([(1, other_unc)]) @ self.transform_to_uncoupled()
        for i in range(n):
            c_i = b1[i]
            unc_i = coupled_to_unc_state(c_i)  # expanded |coupled_i> in uncoupled basis
            row = S[i]
            for j in range(n):
                u_j = b2[j]
                row[j] = as_single_unc_state(u_j) @ unc_i
        return S

    if all1_unc and all2_cpl:
        # S[i, j] = basis1[i] @ basis2[j] for uncoupled_i, coupled_j
        # Original semantics:
        #   UncoupledBasisState.__matmul__(CoupledBasisState) ->
        #       UncoupledState([(1, self)]) @ other.transform_to_uncoupled()
        for i in range(n):
            u_i = b1[i]
            unc_i = as_single_unc_state(u_i)
            row = S[i]
            for j in range(n):
                c_j = b2[j]
                row[j] = unc_i @ coupled_to_unc_state(c_j)
        return S

    return _generate_transform_matrix_original(b1, b2)


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
        This function uses a Rust implementation if available for better performance.
    """
    if HAS_RUST and _generate_transform_matrix_rust is not None:
        return _generate_transform_matrix_rust(basis1, basis2)
    else:
        return _generate_transform_matrix_python(basis1, basis2)
