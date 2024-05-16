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
    """calculate the thermal population of a given J sublevel

    Args:
        J (int): rotational level
        T (float): temperature [Kelvin]
        B (float, optional): rotational constant. Defaults to 6.66733e9.
        n (int, optional): number of rotational levels to normalize with.
                            Defaults to 100.

    Returns:
        float: relative population in a rotational sublevel
    """
    c = 2 * np.pi * cst.hbar * B / (cst.k * T)

    def a(J):
        return -c * J * (J + 1)

    Z = np.sum([J_levels(i) * np.exp(a(i)) for i in range(n)])
    return J_levels(J) * np.exp(a(J)) / Z


def J_levels(J: int) -> int:
    """calculate the number of hyperfine sublevels per J rotational level

    Args:
        J (int): rotational level

    Returns:
        int: number of levels
    """
    return 4 * (2 * J + 1)


def generate_thermal_population_states(
    states_to_fill: Union[Sequence[QuantumSelector], QuantumSelector],
    states: Sequence[CoupledState],
    T: float,
    qn_compact: Optional[Union[Sequence[QuantumSelector], QuantumSelector]] = None,
) -> npt.NDArray[np.complex_]:
    """Generate a thermal distrubtion over the states specified in
    states_to_fill, a QuantumSelector or list of Quantumselectors

    Args:
        states_to_fill (QuantumSelector): Quantumselector specifying states to
        fill
        states (list, np.ndarray): all states used in simulation
        T (float): temperature in Kelvin
        qn_compact (Optional[Union[Sequence[QuantumSelector], QuantumSelector]]):
            defaults to None. Quantum selector to specify which states are compacted
            in the OBE system.

    Returns:
        np.ndarray: density matrix with trace normalized to 1
    """
    # branch for single QuantumSelector use
    if isinstance(states_to_fill, QuantumSelector):
        # get all involved Js
        Js = states_to_fill.J
        assert Js is not None, (
            "states_to_fill needs rotational levels defined to generate the thermal "
            "population density"
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
        assert isinstance(states_to_fill[0], QuantumSelector), (
            "need to supply a sequence of QuantumSelectors, not "
            f"{type(states_to_fill[0])}"
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
        assert (
            len(_Js) != 0
        ), "requires states_to_fill with QuantumSelectors with rotation levels defined"
        # get indices of states to fill
        indices_to_fill = np.array([], dtype=np.int_)
        for stf in states_to_fill:
            indices_to_fill = np.append(
                indices_to_fill, stf.get_indices(states)
            ).astype(int)

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
            raise AssertionError(
                "qn_compact required to be a QuantumsSelector or a Sequence of "
                f"QuantumSelectors, not {type(qn_compact)}"
            )

    density = np.eye(len(density), dtype=np.complex_) * density

    # normalize the trace to 1 and return the density matrix
    return density / np.trace(density)


def generate_population_states(
    states: Union[List[int], npt.NDArray[np.int_]], levels: int
):
    """generate a uniform population distribution with population in the
    specified states

    Args:
        states (list, np.ndarray): indices to put population into
        levels (int): total number of levels

    Returns:
        np.ndarray: density matrix
    """
    density = np.zeros([levels, levels], dtype=complex)
    for state in states:
        density[state, state] = 1
    return density / np.trace(density)
