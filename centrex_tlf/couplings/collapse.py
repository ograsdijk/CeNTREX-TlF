import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf import states

from .branching import calculate_br
from .utils_compact import compact_C_array_indices

__all__ = ["collapse_matrices"]


def collapse_matrices(
    QN: Sequence[states.CoupledState],
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    gamma: float = 1,
    tol: float = 1e-4,
    qn_compact: Optional[
        Union[states.QuantumSelector, Sequence[states.QuantumSelector]]
    ] = None,
) -> npt.NDArray[np.float_]:
    """
    Function that generates the collapse matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states
    excited_states = list of excited states that are coupled to the ground states
    gamma = decay rate of excited states
    tol = couplings smaller than tol/sqrt(gamma) are set to zero to speed up computation
    qn_compact = list of QuantumSelectors or lists of QuantumSelectors with each
                QuantumSelector containing the quantum numbers to compact into a
                single state. Defaults to None.

    outputs:
    C_list = array of collapse matrices
    """
    # Initialize list of collapse matrices
    C_list: List[npt.NDArray[np.float_]] = []

    # Start looping over ground and excited states
    for excited_state in excited_states:
        j = QN.index(excited_state)
        BRs = calculate_br(excited_state, ground_states)
        if np.sum(BRs) > 1:
            print(f"Warning: Branching ratio sum > 1, difference = {np.sum(BRs)-1:.2e}")
        for ground_state, BR in zip(ground_states, BRs):
            i = QN.index(ground_state)

            if np.sqrt(BR) > tol:
                # Initialize the coupling matrix
                H = np.zeros((len(QN), len(QN)), dtype=np.float64)
                H[i, j] = np.sqrt(BR * gamma)

                C_list.append(H)

    C_array = np.array(C_list)

    if qn_compact:
        if isinstance(qn_compact, states.QuantumSelector):
            qn_compact = [qn_compact]
        QN_compact = copy.deepcopy(QN)
        for qnc in qn_compact:
            indices_compact = states.get_indices_quantumnumbers(qnc, QN_compact)
            QN_compact = states.compact_QN_coupled_indices(QN_compact, indices_compact)
            C_array = compact_C_array_indices(C_array, gamma, indices_compact)
    return C_array
