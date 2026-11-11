import copy
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf import states

from .branching import _branching_ratios_reduced, _reduce_for_branching, calculate_br
from .utils_compact import (
    CollapseOperators,
    compact_C_array_indices,
    compact_collapse_operators,
)

__all__ = ["collapse_matrices"]


def collapse_matrices(
    QN: Sequence[states.CoupledState],
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    decay_rate: Optional[float] = None,
    tol: float = 1e-4,
    qn_compact: Optional[Union[states.QuantumSelector, Sequence[states.QuantumSelector]]] = None,
    *,
    gamma: Optional[float] = None,
) -> npt.NDArray[np.floating]:
    """Generate collapse (jump) matrices for spontaneous emission from excited states.

    Creates Lindblad collapse operators C that describe spontaneous decay from excited
    states to ground states via electric dipole transitions. Each operator has the form:
        C[i,j] = √(BR * Γ)
    where BR is the branching ratio and Γ is the excited-state population decay rate.
    Couplings smaller than tol are set to zero for computational efficiency.

    Args:
        QN (Sequence[CoupledState]): Complete basis of states for the calculation
        ground_states (Sequence[CoupledState]): Ground states coupled to excited states
        excited_states (Sequence[CoupledState]): Excited states that can decay
        decay_rate (float | None): Excited-state population decay rate Γ = 1/τ, in
            s^-1. This is NOT the ordinary-frequency linewidth γ = Γ/(2π) in Hz.
            Defaults to 1.
        tol (float): Threshold for keeping couplings. Couplings with √(BR) < tol are
            set to zero. Defaults to 1e-4.
        qn_compact (QuantumSelector | Sequence[QuantumSelector] | None): Quantum number
            selectors for compacting multiple states into single manifolds. Defaults to
            None.
        gamma (float | None): Deprecated alias for `decay_rate`. Do not use both
            `decay_rate` and `gamma` in the same call.

    Returns:
        npt.NDArray[np.floating]: Array of collapse matrices, shape (n_ops, n_states,
            n_states), where n_ops is the number of non-zero transitions

    Warns:
        UserWarning: If branching ratios sum to more than 1 (numerical error)
        DeprecationWarning: If the deprecated `gamma` keyword is used

    Raises:
        TypeError: If both `decay_rate` and `gamma` are supplied

    Example:
        >>> # Create collapse operators for X→B transitions, with the B-state
        >>> # population decay rate Γ = 1/τ (s^-1), not the linewidth γ = Γ/(2π).
        >>> C_array = collapse_matrices(QN, ground_states, excited_states,
        ...                             decay_rate=hamiltonian.Γ)
    """
    if gamma is not None:
        if decay_rate is not None:
            raise TypeError(
                "collapse_matrices() got both 'decay_rate' and the deprecated "
                "'gamma' keyword; pass only 'decay_rate'."
            )
        warnings.warn(
            "The 'gamma' keyword of collapse_matrices() has been renamed to "
            "'decay_rate': the value is the population decay rate Γ = 1/τ "
            "(s^-1), not the ordinary-frequency linewidth γ = Γ/(2π) (Hz). "
            "Pass decay_rate=... instead; 'gamma' will be removed in a "
            "future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        decay_rate = gamma
    elif decay_rate is None:
        decay_rate = 1

    # One collapse operator per decay path, each with a single nonzero element:
    # kept as (row, column, value) until the dense array is built at the end.
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    qn_index = {id(state): idx for idx, state in enumerate(QN)}
    for state in ground_states:
        if id(state) not in qn_index:
            qn_index[id(state)] = QN.index(state)
    for state in excited_states:
        if id(state) not in qn_index:
            qn_index[id(state)] = QN.index(state)

    # the ground-state reduction is the same for every excited state
    ground_states_reduced = [_reduce_for_branching(state, 1e-3) for state in ground_states]

    for excited_state in excited_states:
        j = qn_index[id(excited_state)]
        if len(ground_states):
            BRs = _branching_ratios_reduced(
                _reduce_for_branching(excited_state, 1e-3), ground_states_reduced
            )
        else:
            BRs = calculate_br(excited_state, ground_states)
        if not np.allclose(np.sum(BRs), 1.0):
            warnings.warn(f"Branching ratio sum > 1, difference = {np.sum(BRs) - 1:.2e}")
        for ground_state, BR in zip(ground_states, BRs):
            i = qn_index[id(ground_state)]

            if np.sqrt(BR) > tol:
                rows.append(i)
                cols.append(j)
                values.append(np.sqrt(BR * decay_rate))

    operators = CollapseOperators(
        np.asarray(rows, dtype=np.intp),
        np.asarray(cols, dtype=np.intp),
        np.asarray(values, dtype=np.float64),
        len(QN),
    )

    if not qn_compact:
        return operators.to_dense()

    if isinstance(qn_compact, states.QuantumSelector):
        qn_compact = [qn_compact]
    QN_compact = copy.deepcopy(QN)
    C_array: Optional[npt.NDArray[np.floating]] = None
    for qnc in qn_compact:
        indices_compact = states.get_indices_quantumnumbers(qnc, QN_compact)
        QN_compact = states.compact_QN_coupled_indices(QN_compact, indices_compact)
        if C_array is None:
            compacted = compact_collapse_operators(operators, decay_rate, indices_compact)
            if compacted is not None:
                operators = compacted
                continue
            # not expressible as single-element operators: finish on dense arrays
            C_array = operators.to_dense()
        C_array = compact_C_array_indices(C_array, decay_rate, indices_compact)
    return operators.to_dense() if C_array is None else C_array
