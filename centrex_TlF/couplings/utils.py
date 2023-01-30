from typing import List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf import states
from centrex_tlf.states.states import CoupledBasisState

__all__: List[str] = []


def check_transition_coupled_allowed(
    ground: states.CoupledBasisState, excited: states.CoupledBasisState
):
    """
    Check whether the transition between the two states is allowed based on the quantum
    numbers

    Args:
        ground (states.CoupledBasisState): ground CoupledBasisState
        excited (states.CoupledBasisState): excited CoupledBasisState
    """
    assert ground.P is not None, "parity is required to be set for ground"
    assert excited.P is not None, "parity is required to be set for excited"

    ΔF = int(excited.F - ground.F)
    ΔP = int(excited.P - ground.P)

    flag_ΔP = abs(ΔP) != 2
    flag_ΔF = abs(ΔF) > 1

    return not (flag_ΔP | flag_ΔF)


def check_transition_coupled_allowed_polarization(
    ground_state: states.CoupledBasisState,
    excited_state: states.CoupledBasisState,
    ΔmF_allowed: int,
    return_err: float = True,
    ΔmF_absolute=False,
) -> Union[bool, Tuple[bool, str]]:
    """Check whether the transition is allowed based on the quantum numbers

    Args:
        ground_state (CoupledBasisState): ground CoupledBasisState
        state2 (CoupledBasisState): excited CoupledBasisState
        ΔmF_allowed (int): allowed ΔmF for the transition
        return_err (boolean): boolean flag for returning the error message

    Returns:
        tuple: (allowed boolean, error message)
    """
    assert ground_state.P is not None, "parity is required to be set for ground_state"
    assert excited_state.P is not None, "parity is required to be set for excited_state"

    ΔF = int(excited_state.F - ground_state.F)
    ΔmF = int(excited_state.mF - ground_state.mF)
    if ΔmF_absolute:
        ΔmF = np.abs(ΔmF)
        ΔmF_allowed = np.abs(ΔmF_allowed)
    ΔP = int(excited_state.P - ground_state.P)

    flag_ΔP = abs(ΔP) != 2
    flag_ΔF = abs(ΔF) > 1
    flag_ΔmF = ΔmF != ΔmF_allowed
    flag_ΔFΔmF = (not flag_ΔmF) & ((ΔF == 0) & (ΔmF == 0) & (ground_state.mF == 0))

    errors = ""
    if flag_ΔP:
        errors += "parity invalid"
    if flag_ΔF:
        if len(errors) != 0:
            errors += ", "
        errors += f"ΔF invalid -> ΔF = {ΔF}"
    if flag_ΔmF:
        if len(errors) != 0:
            errors += ", "
        errors += f"ΔmF invalid -> ΔmF = {ΔmF}"

    if flag_ΔFΔmF:
        if len(errors) != 0:
            errors += ", "
        errors += "ΔF = 0 & ΔmF = 0 invalid"

    if len(errors) != 0:
        errors = f"transition not allowed; {errors}"

    if return_err:
        return not (flag_ΔP | flag_ΔF | flag_ΔmF | flag_ΔFΔmF), errors
    else:
        return not (flag_ΔP | flag_ΔF | flag_ΔmF | flag_ΔFΔmF)


def assert_transition_coupled_allowed(
    ground_state: states.CoupledBasisState,
    excited_state: states.CoupledBasisState,
    ΔmF_allowed: int,
) -> bool:
    """Check whether the transition is allowed based on the quantum numbers.
    Raises an AssertionError if the transition is not allowed.

    Args:
        ground_state (CoupledBasisState): ground CoupledBasisState
        excited_state (CoupledBasisState): excited CoupledBasisState

    Returns:
        tuple: allowed boolean
    """
    ret = check_transition_coupled_allowed_polarization(
        ground_state, excited_state, ΔmF_allowed, return_err=True
    )
    if isinstance(ret, tuple):
        allowed, errors = ret
        assert allowed, errors
        return allowed
    else:
        return ret


def select_main_states(
    ground_states: Sequence[states.State],
    excited_states: Sequence[states.State],
    polarization: npt.NDArray[np.complex_],
) -> Tuple[states.State, states.State]:
    """Select main states for calculating the transition strength to normalize
    the Rabi rate with

    Args:
        ground_states (Sequence[states.State]): Sequence of ground states for the
                                                transition
        excited_states (Sequence[states.State]): Sequence of excited states for the
                                                transition
        polarization (npt.NDArray[np.float_]): polarization vector
    """
    ΔmF = 0 if polarization[2] != 0 else 1

    allowed_transitions = []
    indices_gnd_mF0 = []
    for ide, exc in enumerate(excited_states):
        exc_basisstate: CoupledBasisState = exc.largest  # type: ignore
        for idg, gnd in enumerate(ground_states):
            gnd_basisstate: CoupledBasisState = gnd.largest  # type: ignore
            if check_transition_coupled_allowed_polarization(
                gnd_basisstate, exc_basisstate, ΔmF, return_err=False
            ):
                allowed_transitions.append((idg, ide, exc_basisstate.mF))
                if gnd_basisstate.mF == 0:
                    indices_gnd_mF0.append((idg, ide, gnd_basisstate.mF))

    assert (
        len(allowed_transitions) > 0
    ), "none of the supplied ground and excited states have allowed transitions"

    if len(indices_gnd_mF0) > 0:
        excited_state = excited_states[indices_gnd_mF0[-1][1]]
        ground_state = ground_states[indices_gnd_mF0[-1][0]]
    else:
        idt = len(allowed_transitions) // 2
        excited_state = excited_states[allowed_transitions[idt][1]]
        ground_state = ground_states[allowed_transitions[idt][0]]

    return ground_state, excited_state
