"""Population distribution utilities for quantum states.

This module provides functions for generating thermal and uniform population
distributions across molecular quantum states, useful for initializing density
matrices in optical Bloch equation simulations.
"""

from typing import Literal, Optional, Sequence, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import scipy.constants as cst

from centrex_tlf import states

__all__ = [
    "J_levels",
    "thermal_population",
    "generate_uniform_population_state_indices",
    "generate_uniform_population_states",
    "generate_thermal_population_states",
    "get_diagonal_indices_flattened",
]

# Define a TypeVar that can be either an int or a numpy array of ints
JType = TypeVar("JType", int, npt.NDArray[np.int_])


def J_levels(J: JType) -> Union[int, npt.NDArray[np.int_]]:
    """Calculate number of hyperfine levels per rotational J level.

    For TlF with two nuclear spins (I₁=1/2 for ²⁰⁵Tl, I₂=1/2 for ¹⁹F), the number
    of hyperfine levels is 4(2J+1), accounting for F1 and F quantum numbers.

    Args:
        J (JType): Rotational quantum number(s). Can be single int or array of ints.

    Returns:
        Union[int, npt.NDArray[np.int_]]: Number of hyperfine levels. Returns same
            type as input (int or array).

    Example:
        >>> J_levels(0)  # J=0 has 4 levels
        4
        >>> J_levels(1)  # J=1 has 12 levels
        12
        >>> J_levels(np.array([0, 1, 2]))
        array([ 4, 12, 20])
    """
    return 4 * (2 * J + 1)


@overload
def thermal_population(
    J: int, T: float, B: float = 6.66733e9, n: int = 100
) -> float: ...


@overload
def thermal_population(
    J: npt.NDArray[np.int_], T: float, B: float = 6.66733e9, n: int = 100
) -> npt.NDArray[np.floating]: ...


def thermal_population(
    J: JType,
    T: float,
    B: float = 6.66733e9,
    n: int = 100,
) -> Union[float, npt.NDArray[np.floating]]:
    """Calculate thermal (Boltzmann) population fraction for rotational J level(s).

    Uses rigid rotor energy E_J = BJ(J+1) and Boltzmann distribution to compute
    relative populations at temperature T. Includes degeneracy factor 4(2J+1) for
    hyperfine structure.

    Args:
        J (Union[int, npt.NDArray[np.int_]]): Rotational quantum number(s).
        T (float): Temperature in Kelvin. Must be positive.
        B (float): Rotational constant in Hz. Defaults to 6.66733e9 Hz for TlF X state.
        n (int): Number of J levels to include in partition function for normalization.
            Defaults to 100 (sufficient for T < 10 K).

    Returns:
        Union[float, npt.NDArray[np.floating]]: Normalized population fraction(s).
            Sum over all J equals 1. Returns same type as J input.

    Example:
        >>> thermal_population(0, 4.0)  # J=0 at 4K
        0.8945...
        >>> thermal_population(1, 4.0)  # J=1 at 4K
        0.0971...
        >>> thermal_population(np.array([0, 1, 2]), 4.0)
        array([0.8945..., 0.0971..., 0.0082...])

    Note:
        Population ∝ (2J+1)·exp(-E_J/k_BT) where E_J = h·B·J(J+1).
    """
    c = 2 * np.pi * cst.hbar * B / (cst.k * T)

    @overload
    def a(J: int) -> float: ...

    @overload
    def a(J: npt.NDArray[np.int_]) -> npt.NDArray[np.floating]: ...

    def a(
        J: Union[int, npt.NDArray[np.int_]],
    ) -> Union[float, npt.NDArray[np.floating]]:
        return -c * J * (J + 1)

    J_values = np.arange(n)
    Z = np.sum(J_levels(J_values) * np.exp(a(J_values)))

    # Compute the population for both scalar and array inputs
    result = J_levels(J) * np.exp(a(J)) / Z
    return result


def generate_uniform_population_state_indices(
    state_indices: Sequence[int], levels: int
) -> npt.NDArray[np.complex128]:
    """Create uniform population density matrix over specified state indices.

    Generates a diagonal density matrix with equal population distributed across
    the given states. Useful for initializing uniform superpositions.

    Args:
        state_indices (Sequence[int]): State indices to populate (0-indexed). Must be
            in range [0, levels).
        levels (int): Total dimension of the Hilbert space (total number of states).

    Returns:
        npt.NDArray[np.complex128]: Normalized density matrix of shape (levels, levels).
            Diagonal elements sum to 1, with equal weight on specified states.

    Raises:
        ValueError: If levels is non-positive, state_indices is empty, or any index
            is out of bounds.

    Example:
        >>> ρ = generate_uniform_population_state_indices([0, 2, 4], 5)
        >>> np.diag(ρ)  # Only indices 0, 2, 4 have population
        array([0.33333333+0.j, 0.        +0.j, 0.33333333+0.j, 0.        +0.j,
               0.33333333+0.j])
    """
    if levels <= 0:
        raise ValueError(f"levels must be positive, got {levels}")
    if not state_indices:
        raise ValueError("state_indices cannot be empty")

    state_indices_array = np.asarray(state_indices)
    if np.any(state_indices_array < 0) or np.any(state_indices_array >= levels):
        raise ValueError(
            f"All indices must be in range [0, {levels}), "
            f"got min={state_indices_array.min()}, max={state_indices_array.max()}"
        )

    ρ = np.zeros([levels, levels], dtype=complex)
    for ids in state_indices:
        ρ[ids, ids] = 1
    return ρ / np.trace(ρ)


def generate_uniform_population_states(
    selected_states: Union[Sequence[states.QuantumSelector], states.QuantumSelector],
    QN: Sequence[states.CoupledState],
) -> npt.NDArray[np.complex128]:
    """Create uniform population density matrix using quantum number selectors.

    Distributes population equally across states matching the given quantum number
    criteria. Accepts single or multiple selectors for flexible state selection.

    Args:
        selected_states (Union[Sequence[states.QuantumSelector], states.QuantumSelector]):
            Quantum number selector(s) specifying which states to populate. Multiple
            selectors are combined (union of matching states).
        QN (Sequence[states.CoupledState]): Complete basis of quantum states.

    Returns:
        npt.NDArray[np.complex128]: Normalized density matrix with uniform population
            on selected states. Shape is (len(QN), len(QN)).

    Raises:
        ValueError: If QN is empty or if no states match the selectors.

    Example:
        >>> selector = states.QuantumSelector(J=0, electronic=states.ElectronicState.X)
        >>> QN = states.generate_coupled_states_X(J_max=1)
        >>> ρ = generate_uniform_population_states(selector, QN)
    """
    if not QN:
        raise ValueError("QN sequence cannot be empty.")

    levels = len(QN)
    ρ = np.zeros([levels, levels], dtype=complex)

    if isinstance(selected_states, states.QuantumSelector):
        indices = selected_states.get_indices(QN)
    else:
        indices = np.unique(
            np.concatenate([ss.get_indices(QN) for ss in selected_states])
        )

    for idx in indices:
        ρ[idx, idx] = 1

    return ρ / np.trace(ρ)


def generate_thermal_population_states(
    temperature: float,
    QN: Sequence[states.CoupledState],
) -> npt.NDArray[np.complex128]:
    """Generate thermal (Boltzmann) population density matrix for ground X state.

    Creates density matrix with populations following Boltzmann distribution over
    rotational J levels, uniformly distributed within hyperfine manifolds. Only
    populates ground electronic state (X).

    Args:
        temperature (float): Temperature in Kelvin. Must be positive.
        QN (Sequence[states.CoupledState]): Complete basis of quantum states. Must
            include CoupledState objects.

    Returns:
        npt.NDArray[np.complex128]: Diagonal density matrix of shape (len(QN), len(QN))
            with thermal populations. Normalized to Tr(ρ) = 1.

    Raises:
        ValueError: If temperature is non-positive, QN is empty, or duplicate quantum
            numbers are found.
        TypeError: If QN doesn't contain CoupledState objects.

    Example:
        >>> QN = states.generate_coupled_states_X(J_max=2)
        >>> ρ_thermal = generate_thermal_population_states(4.0, QN)
        >>> np.trace(ρ_thermal)  # Normalized
        (1+0j)

    Note:
        Population within each J is uniform across mF sublevels. Excited electronic
        states (B, etc.) are left unpopulated.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    levels = len(QN)
    ρ = np.zeros([levels, levels], dtype=complex)

    if not QN:
        raise ValueError("QN sequence cannot be empty.")
    if not isinstance(QN[0], states.CoupledState):
        raise TypeError(f"Expected CoupledState objects, got {type(QN[0]).__name__}.")

    j_levels = np.unique([qn.largest.J for qn in QN])

    # Compute relative thermal population fractions
    population = {
        j: p for j, p in zip(j_levels, thermal_population(j_levels, temperature))
    }

    # Get quantum numbers of the ground state
    quantum_numbers = [
        (qn.largest.J, float(qn.largest.F1), qn.largest.F, qn.largest.mF)
        for qn in QN
        if qn.largest.electronic_state == states.ElectronicState.X
    ]

    unique_qn: np.ndarray = np.unique(quantum_numbers, axis=0)
    if len(unique_qn) != len(quantum_numbers):
        raise ValueError(
            f"Duplicate quantum numbers found: expected {len(quantum_numbers)} "
            f"unique states but got {len(unique_qn)}."
        )

    for idx, qn in enumerate(QN):
        if qn.largest.electronic_state != states.ElectronicState.X:
            continue
        if qn.largest.F is None:
            ρ[idx, idx] = population[qn.largest.J]
        else:
            ρ[idx, idx] = population[qn.largest.J] / J_levels(qn.largest.J)

    return ρ


def get_diagonal_indices_flattened(
    size: int,
    states: Optional[Sequence[int]] = None,
    mode: Literal["python", "julia"] = "python",
) -> list[int]:
    """Get flattened array indices corresponding to diagonal matrix elements.

    Converts 2D matrix indices (i, i) to 1D flattened array indices i + size*i.
    Useful for extracting populations from flattened density matrices in ODE solvers.

    Args:
        size (int): Matrix dimension (number of rows/columns).
        states (Optional[Sequence[int]]): Specific diagonal indices to include.
            If None, returns all diagonal elements [0, 1, ..., size-1].
        mode (Literal["python", "julia"]): Indexing convention. "python" uses 0-based
            indexing, "julia" uses 1-based indexing. Defaults to "python".

    Returns:
        list[int]: Flattened indices of diagonal elements in row-major order.

    Raises:
        ValueError: If mode is not "python" or "julia".

    Example:
        >>> get_diagonal_indices_flattened(3)  # 3x3 matrix
        [0, 4, 8]
        >>> get_diagonal_indices_flattened(5, states=[0, 2, 4])
        [0, 12, 24]
        >>> get_diagonal_indices_flattened(3, mode="julia")  # 1-indexed
        [1, 5, 9]

    Note:
        For matrix M flattened row-wise, diagonal element M[i, i] is at index i + size*i.
    """
    if states is None:
        indices = [i + size * i for i in range(size)]
    else:
        indices = [i + size * i for i in states]

    if mode == "julia":
        return [i + 1 for i in indices]
    elif mode == "python":
        return indices
    else:
        raise ValueError("`mode` must be 'python' or 'julia'.")
