from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
import sympy as smp

from centrex_tlf import hamiltonian, states, transitions

__all__ = ["compact_symbolic_hamiltonian_indices", "generate_qn_compact"]


def compact_symbolic_hamiltonian_indices(
    hamiltonian: smp.matrices.dense.MutableDenseMatrix,
    indices_compact: npt.NDArray[np.int_],
) -> smp.matrices.dense.MutableDenseMatrix:
    """compact a sympy hamiltonian by combining all indices in indices_compact
    into a single state

    Args:
        hamiltonian (sympy matrix): hamiltonian
        indices_compact (list, array): indices to compact

    Returns:
        sympy matrix: compacted hamiltonian
    """
    arr = hamiltonian.copy()
    diagonal = arr.diagonal()
    diagonal = [diagonal[idd] for idd in indices_compact]
    # free_symbols = np.unique([val.free_symbols for val in diagonal])
    check_free_symbols = np.sum([len(val.free_symbols) for val in diagonal])
    assert (
        check_free_symbols == 0
    ), "diagonal elements for states to compact have symbols, cannot compact"

    # delete the rows and columns to compact, except a single one that's needed
    # to put the decays into
    deleted = 0
    for idx in indices_compact[1:]:
        row = arr[idx - deleted, :]
        col = arr[:, idx - deleted]
        # check if couplings are present, raise AssertionError if true
        assert (
            np.sum(row) - row[idx - deleted]
        ) == 0, "couplings exist for states to compact, cannot compact"
        assert (
            np.sum(col) - row[idx - deleted]
        ) == 0, "couplings exist for states to compact, cannot compact"
        arr.row_del(idx - deleted)
        arr.col_del(idx - deleted)
        deleted += 1

    # setting the diagonal element to be the mean off the entire state
    # pretty much irrelevant since the states only have decays and should be
    # far enough away from the others
    arr[idx - deleted, idx - deleted] = np.mean(diagonal)
    return arr


def generate_qn_compact(
    transitions: Sequence[
        Union[transitions.OpticalTransition, transitions.MicrowaveTransition]
    ],
    H_reduced: hamiltonian.reduced_hamiltonian.ReducedHamiltonianTotal,
):
    J_transitions_ground = []
    for transition in transitions:
        J_transitions_ground.append(transition.J_ground)
    J_compact = [
        Ji
        for Ji in np.unique([s.J for s in H_reduced.X_states_basis])
        if Ji not in J_transitions_ground
    ]
    qn_compact = [
        states.QuantumSelector(J=Ji, electronic=states.ElectronicState.X)
        for Ji in J_compact
    ]

    return qn_compact
