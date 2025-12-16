import copy
import warnings
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
) -> npt.NDArray[np.floating]:
    """Generate collapse (jump) matrices for spontaneous emission from excited states.

    Creates Lindblad collapse operators C that describe spontaneous decay from excited
    states to ground states via electric dipole transitions. Each operator has the form:
        C[i,j] = √(BR * γ)
    where BR is the branching ratio and γ is the decay rate. Couplings smaller than
    tol are set to zero for computational efficiency.

    Args:
        QN (Sequence[CoupledState]): Complete basis of states for the calculation
        ground_states (Sequence[CoupledState]): Ground states coupled to excited states
        excited_states (Sequence[CoupledState]): Excited states that can decay
        gamma (float): Decay rate of excited states in rad/s. Defaults to 1.
        tol (float): Threshold for keeping couplings. Couplings with √(BR) < tol are
            set to zero. Defaults to 1e-4.
        qn_compact (QuantumSelector | Sequence[QuantumSelector] | None): Quantum number
            selectors for compacting multiple states into single manifolds. Defaults to
            None.

    Returns:
        npt.NDArray[np.floating]: Array of collapse matrices, shape (n_ops, n_states,
            n_states), where n_ops is the number of non-zero transitions

    Warns:
        UserWarning: If branching ratios sum to more than 1 (numerical error)

    Example:
        >>> # Create collapse operators for X→B transitions
        >>> C_array = collapse_matrices(QN, ground_states, excited_states, gamma=2π*36e6)
    """
    # Initialize list of collapse matrices
    C_list: List[npt.NDArray[np.floating]] = []

    # Start looping over ground and excited states
    for excited_state in excited_states:
        j = QN.index(excited_state)
        BRs = calculate_br(excited_state, ground_states)
        if not np.allclose(np.sum(BRs), 1.0):
            warnings.warn(
                f"Branching ratio sum > 1, difference = {np.sum(BRs) - 1:.2e}"
            )
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
