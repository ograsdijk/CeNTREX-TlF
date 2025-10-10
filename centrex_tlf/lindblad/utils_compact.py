"""Utilities for compacting symbolic Hamiltonians by combining degenerate states.

This module provides functions to reduce the dimensionality of symbolic Hamiltonians
by identifying and combining quantum states that are not coupled to the system of
interest. This compacting reduces computational cost while maintaining physical accuracy.
"""

from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
import sympy as smp

from centrex_tlf import hamiltonian, states, transitions

__all__ = ["compact_symbolic_hamiltonian_indices", "generate_qn_compact"]


def compact_symbolic_hamiltonian_indices(
    hamiltonian_matrix: smp.Matrix,
    indices_compact: npt.NDArray[np.int_],
) -> smp.Matrix:
    """Compact a symbolic Hamiltonian by combining multiple indices into a single state.

    This function reduces the dimensionality of a Hamiltonian matrix by merging
    multiple quantum states (specified by indices) into a single representative state.
    The states to be compacted must satisfy two conditions:
    1. Their diagonal elements must be numeric (no symbolic variables)
    2. They must not have couplings to other states (off-diagonal elements must be zero)

    The compacting process:
    1. Validates that diagonal elements of states to compact are numeric
    2. Validates that these states have no couplings (off-diagonal elements are zero)
    3. Deletes all rows/columns except the first one (the representative state)
    4. Sets the representative state's diagonal element to the mean energy of all
       compacted states

    Args:
        hamiltonian_matrix: The symbolic Hamiltonian matrix to compact. Must be a
            square sympy Matrix with both symbolic and numeric elements.
        indices_compact: Array of matrix indices to compact into a single state.
            Must contain at least one index. The first index becomes the representative
            state that remains in the compacted Hamiltonian.

    Returns:
        A new compacted Hamiltonian matrix with reduced dimensionality. The returned
        matrix has `len(indices_compact) - 1` fewer rows/columns than the input.

    Raises:
        AssertionError: If diagonal elements of states to compact contain symbolic
            variables (have free symbols), indicating these states cannot be safely
            compacted.
        AssertionError: If any of the states to compact have non-zero off-diagonal
            elements (couplings), indicating they are coupled to the system and
            cannot be removed.

    Example:
        >>> import sympy as smp
        >>> import numpy as np
        >>> # Create a 5x5 Hamiltonian with states 3 and 4 being uncoupled
        >>> H = smp.Matrix([[1, 0.1, 0, 0, 0],
        ...                 [0.1, 2, 0, 0, 0],
        ...                 [0, 0, 100, 0, 0],
        ...                 [0, 0, 0, 200, 0],
        ...                 [0, 0, 0, 0, 300]])
        >>> indices = np.array([2, 3, 4])  # Compact states 2, 3, 4
        >>> H_compact = compact_symbolic_hamiltonian_indices(H, indices)
        >>> H_compact.shape
        (3, 3)
        >>> H_compact[2, 2]  # Mean of 100, 200, 300
        200.0

    Note:
        This function is typically used to remove high-J rotational states that are
        far detuned from the transitions of interest and have no population transfer.
        The mean energy is used for the representative state, though this value is
        largely irrelevant since these states only contribute decay terms.
    """
    # Create a working copy to avoid modifying the original
    hamiltonian_compact = hamiltonian_matrix.copy()

    # Extract diagonal elements for the states to compact
    diagonal_full = hamiltonian_compact.diagonal()
    diagonal_to_compact = [diagonal_full[idx] for idx in indices_compact]

    # Verify diagonal elements are numeric (no symbolic variables)
    # This ensures the states are sufficiently far detuned and can be compacted
    num_free_symbols = sum(len(val.free_symbols) for val in diagonal_to_compact)
    assert num_free_symbols == 0, (
        "Diagonal elements for states to compact contain symbolic variables. "
        "These states may be coupled to the system and cannot be compacted safely."
    )

    # Delete rows and columns for all states except the first (representative) state
    # We keep the first index as the representative that will hold the mean energy
    num_deleted = 0
    representative_idx = indices_compact[0]

    for idx in indices_compact[1:]:
        # Adjust index for previously deleted rows/columns
        current_idx = idx - num_deleted

        # Get row and column for this state
        row = hamiltonian_compact[current_idx, :]
        col = hamiltonian_compact[:, current_idx]

        # Verify no off-diagonal couplings exist
        # Sum of row/column minus the diagonal element should be zero
        # Access diagonal element directly from the matrix
        diagonal_element = hamiltonian_compact[current_idx, current_idx]
        row_coupling = np.sum(row) - diagonal_element  # type: ignore[arg-type]
        col_coupling = np.sum(col) - diagonal_element  # type: ignore[arg-type]

        assert row_coupling == 0, (
            f"Row couplings exist for state at index {idx}. "
            f"Sum of off-diagonal elements: {row_coupling}. Cannot compact."
        )
        assert col_coupling == 0, (
            f"Column couplings exist for state at index {idx}. "
            f"Sum of off-diagonal elements: {col_coupling}. Cannot compact."
        )

        # Delete this row and column
        hamiltonian_compact.row_del(current_idx)
        hamiltonian_compact.col_del(current_idx)
        num_deleted += 1

    # Set the representative state's diagonal element to the mean energy
    # Note: representative_idx doesn't change since we only delete indices after it
    mean_energy = float(np.mean(diagonal_to_compact))
    hamiltonian_compact[representative_idx, representative_idx] = mean_energy

    return hamiltonian_compact


def generate_qn_compact(
    transition_list: Sequence[
        Union[transitions.OpticalTransition, transitions.MicrowaveTransition]
    ],
    H_reduced: hamiltonian.reduced_hamiltonian.ReducedHamiltonianTotal,
) -> list[states.QuantumSelector]:
    """Generate quantum selectors for ground states that can be compacted.

    This function identifies ground electronic state (X) rotational levels that are
    not involved in any of the specified transitions. These states can be safely
    compacted (merged) in the Hamiltonian since they don't participate in the
    dynamics of interest.

    The function works by:
    1. Extracting all J quantum numbers involved in transitions (ground states)
    2. Finding all J values present in the X-state basis
    3. Identifying J values not involved in transitions (candidates for compacting)
    4. Creating QuantumSelector objects for these uncoupled states

    Args:
        transition_list: Sequence of optical or microwave transitions defining the
            system's couplings. Each transition specifies ground and excited states
            involved in the dynamics.
        H_reduced: The reduced Hamiltonian containing the basis states for the
            ground electronic state (X). Used to identify all available rotational
            levels in the system.

    Returns:
        A list of QuantumSelector objects identifying ground (X) state rotational
        levels (specified by J quantum number) that are not involved in any
        transitions and can therefore be compacted.

    Example:
        >>> import centrex_tlf as tlf
        >>> # Define a J=1 transition
        >>> transitions = [
        ...     tlf.couplings.TransitionSelector(
        ...         ground=tlf.states.generate_coupled_states_X(
        ...             tlf.states.QuantumSelector(J=1)
        ...         ),
        ...         excited=tlf.states.generate_coupled_states_B(
        ...             tlf.states.QuantumSelector(J=1)
        ...         ),
        ...     )
        ... ]
        >>> # Create Hamiltonian with J=1 and J=3 ground states
        >>> H_reduced = tlf.hamiltonian.generate_total_reduced_hamiltonian(
        ...     X_states_approx=[
        ...         *tlf.states.generate_coupled_states_X(tlf.states.QuantumSelector(J=1)),
        ...         *tlf.states.generate_coupled_states_X(tlf.states.QuantumSelector(J=3)),
        ...     ],
        ...     B_states_approx=tlf.states.generate_coupled_states_B(
        ...         tlf.states.QuantumSelector(J=1)
        ...     ),
        ... )
        >>> # Get selectors for states to compact (J=3 is not in transitions)
        >>> qn_compact = generate_qn_compact(transitions, H_reduced)
        >>> qn_compact
        [QuantumSelector(J=3, electronic=ElectronicState.X)]

    Note:
        This function only identifies ground (X) state levels for compacting.
        Excited states (B) are typically always involved in transitions and are
        not considered for compacting. The returned selectors can be passed to
        `compact_symbolic_hamiltonian_indices` to actually perform the compacting.
    """
    # Extract J quantum numbers from all ground states in transitions
    J_in_transitions = [transition.J_ground for transition in transition_list]

    # Find all unique J values in the X-state basis
    J_values_in_basis = np.unique([state.J for state in H_reduced.X_states_basis])

    # Identify J values not involved in any transitions (candidates for compacting)
    J_to_compact = [J for J in J_values_in_basis if J not in J_in_transitions]

    # Create QuantumSelector objects for ground states to compact
    qn_compact = [
        states.QuantumSelector(J=J, electronic=states.ElectronicState.X)
        for J in J_to_compact
    ]

    return qn_compact
