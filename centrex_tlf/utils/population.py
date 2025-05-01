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
    "get_diagonal_indices_flattened",
]

# Define a TypeVar that can be either an int or a numpy array of ints
JType = TypeVar("JType", int, npt.NDArray[np.int_])


def J_levels(J: JType) -> Union[int, npt.NDArray[np.int_]]:
    """
    Number of hyperfine levels per J rotational level.

    Args:
        J (JType): Rotational level(s), either an int or a numpy array of ints.

    Returns:
        Union[int, npt.NDArray[np.int_]]: Number of hyperfine levels for the given J level(s).
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
    """
    Thermal population of a given J sublevel.

    Args:
        J (Union[int, npt.NDArray[np.int_]]): Rotational level(s).
        T (float): Temperature [Kelvin].
        B (float, optional): Rotational constant. Defaults to 6.66733e9.
        n (int, optional): Number of rotational levels to normalize with. Defaults to 100.

    Returns:
        Union[float, npt.NDArray[np.floating]]: Relative population(s) in the rotational level(s).
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
    """
    spread population uniformly over the given state indices

    Args:
        state_indices (Sequence[int]): indices to put population into
        levels (int): total states involved

    Returns:
        npt.NDArray[np.complex128]: density matrix
    """
    ρ = np.zeros([levels, levels], dtype=complex)
    for ids in state_indices:
        ρ[ids, ids] = 1
    return ρ / np.trace(ρ)


def generate_uniform_population_states(
    selected_states: Union[Sequence[states.QuantumSelector], states.QuantumSelector],
    QN: Sequence[states.CoupledState],
) -> npt.NDArray[np.complex128]:
    """
    Spread population uniformly over the given states.

    Args:
        selected_states (Union[Sequence[states.QuantumSelector], states.QuantumSelector]):
            State selector(s).
        QN (Sequence[states.CoupledState]): All states involved.

    Returns:
        npt.NDArray[np.complex128]: Density matrix.
    """
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
    """
    Generate a thermal population density matrix.

    Args:
        temperature (float): Temperature [Kelvin].
        QN (Sequence[states.CoupledState]): All states involved.

    Returns:
        npt.NDArray[np.complex128]: Density matrix.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    levels = len(QN)
    ρ = np.zeros([levels, levels], dtype=complex)

    assert isinstance(QN[0], states.CoupledState), "No State objects supplied."

    j_levels = np.unique([qn.largest.J for qn in QN])

    # Compute relative thermal population fractions
    population = {
        j: p for j, p in zip(j_levels, thermal_population(j_levels, temperature))
    }

    # Get quantum numbers of the ground state
    quantum_numbers = [
        (qn.largest.J, qn.largest.F1, qn.largest.F, qn.largest.mF)
        for qn in QN
        if qn.largest.electronic_state == states.ElectronicState.X
    ]

    assert len(np.unique(quantum_numbers, axis=0)) == len(quantum_numbers), (
        "Duplicate quantum numbers."
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
    """
    Get the flattened diagonal indices of a matrix.

    Args:
        size (int): Size of the matrix.
        states (Optional[Sequence[int]], optional): Specific states to include. Defaults to None.
        mode (Literal["python", "julia"], optional): Indexing mode. Defaults to "python".

    Returns:
        list[int]: Flattened diagonal indices.
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
