import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy import constants as cst

from .find_states import QuantumSelector
from .states import CoupledState
from .utils_compact import compact_QN_coupled_indices

__all__ = [
    "thermal_population",
    "J_levels",
    "generate_thermal_population_states",
    "generate_population_states",
]


def thermal_population(J: int, T: float, B: float = 6.66733e9, n: int = 100) -> float:
    """Calculate the thermal population of a given J sublevel.

    Uses Boltzmann distribution to calculate the relative population of a rotational
    sublevel at a given temperature.

    Args:
        J (int): Rotational level
        T (float): Temperature in Kelvin
        B (float, optional): Rotational constant in Hz. Defaults to 6.66733e9 (TlF ground state)
        n (int, optional): Number of rotational levels to include in partition function normalization.
                          Defaults to 100.

    Returns:
        float: Relative population in the rotational sublevel (normalized by partition function)

    Raises:
        ValueError: If T is non-positive
    """
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got {T} K")

    c = 2 * np.pi * cst.hbar * B / (cst.k * T)

    def energy_factor(J_level: int) -> float:
        """Calculate the Boltzmann factor for a given J level."""
        return -c * J_level * (J_level + 1)

    # Partition function
    Z = np.sum([J_levels(i) * np.exp(energy_factor(i)) for i in range(n)])
    return J_levels(J) * np.exp(energy_factor(J)) / Z


def J_levels(J: int) -> int:
    """Calculate the number of hyperfine sublevels per J rotational level.

    For TlF, each J level has 4*(2J+1) hyperfine sublevels due to the two nuclear
    spins (I1=1/2 for Tl, I2=1/2 for F).

    Args:
        J (int): Rotational quantum number

    Returns:
        int: Number of hyperfine sublevels for the given J
    """
    return 4 * (2 * J + 1)


def generate_thermal_population_states(
    states_to_fill: Union[Sequence[QuantumSelector], QuantumSelector],
    states: Sequence[CoupledState],
    T: float,
    qn_compact: Optional[Union[Sequence[QuantumSelector], QuantumSelector]] = None,
) -> npt.NDArray[np.complex128]:
    """Generate a thermal distribution over specified states.

    Creates a density matrix with thermal (Boltzmann) population distribution over
    the states specified by states_to_fill. Population is distributed uniformly within
    each J manifold according to the thermal population of that J level.

    Args:
        states_to_fill (Union[Sequence[QuantumSelector], QuantumSelector]): QuantumSelector
            or sequence of QuantumSelectors specifying which states to populate thermally
        states (Sequence[CoupledState]): All states used in the simulation
        T (float): Temperature in Kelvin
        qn_compact (Optional[Union[Sequence[QuantumSelector], QuantumSelector]], optional):
            QuantumSelector(s) specifying states to compact in the OBE system. Population
            in compacted states is summed together. Defaults to None.

    Returns:
        npt.NDArray[np.complex128]: Density matrix with trace normalized to 1

    Raises:
        ValueError: If states_to_fill doesn't have J levels defined
        TypeError: If states_to_fill is not a QuantumSelector or sequence of them
    """
    # branch for single QuantumSelector use
    if isinstance(states_to_fill, QuantumSelector):
        # get all involved Js
        Js = states_to_fill.J
        if Js is None:
            raise ValueError(
                "states_to_fill needs rotational levels (J) defined to generate "
                "thermal population density"
            )
        # check if J was a list
        _Js = (
            np.array([Js])
            if not isinstance(Js, (np.ndarray, list, tuple, Sequence))
            else np.array(Js)
        )
        # get indices of states to fill
        indices_to_fill = states_to_fill.get_indices(states)
    # branch for multiple QuantumSelectors use
    elif isinstance(states_to_fill, (list, np.ndarray, tuple)):
        if not isinstance(states_to_fill[0], QuantumSelector):
            raise TypeError(
                f"Need to supply a sequence of QuantumSelectors, not {type(states_to_fill[0])}"
            )
        # get all involved Js
        _Js = np.array([], dtype=np.int_)
        for stf in states_to_fill:
            J = stf.J
            # check if J was a list
            if J is not None:
                _J = (
                    [J] if not isinstance(J, (np.ndarray, list, tuple, Sequence)) else J
                )
                _Js = np.append(_Js, _J)
        if len(_Js) == 0:
            raise ValueError(
                "states_to_fill requires QuantumSelectors with rotational levels (J) defined"
            )
        # get indices of states to fill
        indices_to_fill = np.array([], dtype=np.int_)
        for stf in states_to_fill:
            indices_to_fill = np.append(
                indices_to_fill, stf.get_indices(states)
            ).astype(int)
    else:
        raise TypeError(
            "states_to_fill required to be a QuantumsSelector or a Sequence of "
            f"QuantumSelectors, not {type(states_to_fill)}"
        )

    # remove duplicates from Js and indices_to_fill
    _Js = np.unique(_Js)
    indices_to_fill = np.unique(indices_to_fill)

    # thermal population per hyperfine level for each involved J
    thermal_populations = dict(
        [(Ji, thermal_population(Ji, T) / J_levels(Ji)) for Ji in _Js]
    )
    # generate an empty density matrix
    density = np.zeros(len(states), dtype=complex)
    # fill the density matrix
    for idd in indices_to_fill:
        state = states[idd].largest
        thermal_pop = thermal_populations[state.J]
        density[idd] = thermal_pop

    if qn_compact is not None:
        states_compact = copy.copy(states)
        if isinstance(qn_compact, (list, tuple, np.ndarray, Sequence)):
            for qnc in qn_compact:
                indices_compact = qnc.get_indices(states_compact)
                pop_compact = density[indices_compact].sum()
                density = np.array(
                    [di for idd, di in enumerate(density) if idd not in indices_compact]
                )
                density[indices_compact[0]] = pop_compact
                states_compact = compact_QN_coupled_indices(
                    states_compact, indices_compact
                )
        elif isinstance(qn_compact, QuantumSelector):
            indices_compact = qn_compact.get_indices(states_compact)
            pop_compact = density[indices_compact].sum()
            density = np.array(
                [di for idd, di in enumerate(density) if idd not in indices_compact]
            )
            density[indices_compact[0]] = pop_compact
            states_compact = compact_QN_coupled_indices(states_compact, indices_compact)
        else:
            raise TypeError(
                f"qn_compact must be a QuantumSelector or a sequence of QuantumSelectors, "
                f"not {type(qn_compact)}"
            )

    density = np.eye(len(density), dtype=np.complex128) * density

    # normalize the trace to 1 and return the density matrix
    return density / np.trace(density)


def generate_population_states(
    states: Union[List[int], npt.NDArray[np.int_]], levels: int
) -> npt.NDArray[np.complex128]:
    """Generate a uniform population distribution over specified states.

    Creates a density matrix with equal population in each of the specified states,
    normalized so the trace equals 1.

    Args:
        states (Union[List[int], npt.NDArray[np.int_]]): Indices of states to populate
        levels (int): Total number of levels in the system

    Returns:
        npt.NDArray[np.complex128]: Density matrix with uniform population in specified
            states, trace normalized to 1

    Raises:
        ValueError: If any state index is out of bounds
    """
    states_array = np.asarray(states)
    if np.any(states_array < 0) or np.any(states_array >= levels):
        raise ValueError(
            f"State indices must be in range [0, {levels}), got indices: {states_array}"
        )

    density = np.zeros([levels, levels], dtype=np.complex128)
    for state in states_array:
        density[state, state] = 1
    return density / np.trace(density)
