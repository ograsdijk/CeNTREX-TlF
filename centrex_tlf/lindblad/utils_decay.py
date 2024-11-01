import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import sympy as smp

from centrex_tlf import states

__all__ = ["DecayChannel"]


@dataclass
class DecayChannel:
    ground: states.CoupledState
    excited: states.QuantumSelector
    branching: float
    description: str = ""


def add_levels_symbolic_hamiltonian(
    hamiltonian: smp.matrices.dense.MutableDenseMatrix,
    decay_channels: Sequence[DecayChannel],
    QN: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
) -> Tuple[List[int], smp.matrices.dense.MutableDenseMatrix]:
    arr = hamiltonian.copy()
    indices = get_insert_level_indices(decay_channels, QN, excited_states)
    for idx in indices:
        arr = add_level_symbolic_hamiltonian(arr, idx)
    return indices, arr


def get_insert_level_indices(
    decay_channels: Sequence[DecayChannel],
    QN: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
):
    indices = [i + len(QN) - len(excited_states) for i in range(len(decay_channels))]
    return indices


def add_level_symbolic_hamiltonian(
    hamiltonian: smp.matrices.dense.MutableDenseMatrix, idx: int
) -> smp.matrices.dense.MutableDenseMatrix:
    arr = hamiltonian.copy()
    arr = arr.row_insert(idx, smp.zeros(1, arr.shape[1]))
    arr = arr.col_insert(idx, smp.zeros(arr.shape[0], 1))
    return arr


def add_states_QN(
    decay_channels: Sequence[DecayChannel],
    QN: List[states.CoupledState],
    indices: List[int],
) -> List[states.CoupledState]:
    states = copy.copy(QN)
    for idx, decay_channel in zip(indices, decay_channels):
        states.insert(idx, decay_channel.ground)
    return states


def add_levels_C_array(
    C_array: npt.NDArray[np.floating], indices: List[int]
) -> npt.NDArray[np.floating]:
    """
    Adding levels to the C arrays. Used when adding new decay channels.

    Args:
        C_array (npt.NDArray[np.floating]): original C arrays
        indices (List[int]): indices where to add levels

    Returns:
        npt.NDArray[np.floating]: modified C arrays
    """
    arr = C_array.copy()
    # inserting rows and columns of zeros to account for the new decay levels
    for idx in indices:
        arr = np.insert(arr, idx, np.zeros(arr.shape[2]), 1)
        arr = np.insert(arr, idx, np.zeros(arr.shape[1]), 2)
    return arr


def add_decays_C_arrays(
    decay_channels: Sequence[DecayChannel],
    indices: List[int],
    QN: Sequence[states.CoupledState],
    C_array: npt.NDArray[np.floating],
    Γ: float,
) -> npt.NDArray[np.floating]:
    """
    Add decays to the C arrays. Note that QN has to be the original QN before adding
    extra states!

    Args:
        decay_channels (Sequence[DecayChannel]): Additional decay channels
        indices (List[int]): indices where to add the decay channels
        QN (Sequence[states.CoupledState]): original CoupledStates
        C_array (npt.NDArray[np.floating]): original C arrays

    Returns:
        npt.NDArray[np.floating]: modified C arrays
    """
    # converting the C arrays to branching ratio arrays and adding the new
    # levels
    BR = add_levels_C_array(C_array, indices)
    BR = BR**2 / Γ

    # getting the excited state indices, account for the fact that the C_arrays have
    # been extended already by adding len(decay_channels) to the indices
    indices_excited = [
        decay_channel.excited.get_indices(QN) + len(decay_channels)
        for decay_channel in decay_channels
    ]
    # getting the total added branching ratios for each excited state
    BR_added: Dict[int, float] = {}
    for ides, decay_channel in zip(indices_excited, decay_channels):
        for ide in ides:
            if BR_added.get(ide) is None:
                BR_added[ide] = decay_channel.branching
            else:
                BR_added[ide] += decay_channel.branching
    # renormalizing the old branching ratios to ensure the sum is 1 when adding
    # the new branching ratios
    for ide, BR_add in BR_added.items():
        BR[:, :, ide] *= 1 - BR_add

    # adding the new branching ratios
    for idg, ides, decay_channel in zip(indices, indices_excited, decay_channels):
        for ide in ides:
            BR_new = np.zeros([1, *BR[0, :, :].shape], dtype=complex)
            BR_new[:, idg, ide] = decay_channel.branching
            BR = np.append(BR, BR_new, axis=0)
    # converting the branching ratios to C arrays
    return np.sqrt(BR * Γ)
