"""Utilities for compacting quantum state manifolds in coupling calculations.

This module provides functions for reducing the dimensionality of coupling matrices
and branching ratio arrays by combining degenerate or nearly-degenerate states into
single manifolds.
"""
import copy
from typing import List, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf import states

from .coupling_matrix import CouplingField, CouplingFields

__all__: List[str] = [
    "delete_row_column_indices",
    "insert_row_column_indices",
    "BR_to_C_array",
    "C_array_to_BR",
    "compact_BR_array_indices",
    "compact_C_array_indices",
    "compact_coupling_field",
    "compact_coupling_field_indices",
]


DType = TypeVar("DType", bound=np.generic)


def delete_row_column_indices(
    arr: npt.NDArray[DType], indices_compact: Union[Sequence[int], npt.NDArray[np.int_]]
) -> npt.NDArray[DType]:
    """Delete specified rows and columns from a 2D array, excluding the first index.
    
    Removes rows and columns at indices_compact[1:] from the array, preserving
    indices_compact[0]. Used when compacting states where the first index becomes
    the representative.
    
    Args:
        arr (npt.NDArray[DType]): 2D array to modify
        indices_compact (Sequence[int] | npt.NDArray[np.int_]): Indices to remove
            (all except first)
    
    Returns:
        npt.NDArray[DType]: Array with specified rows and columns removed
    """
    return np.delete(
        np.delete(arr, indices_compact[1:], axis=0), indices_compact[1:], axis=1
    )


def insert_row_column_indices(
    arr: npt.NDArray[DType], indices_insert: Union[Sequence[int], npt.NDArray[np.int_]]
) -> npt.NDArray[DType]:
    """Insert zero-filled rows and columns at specified indices.
    
    Expands a 2D array by inserting zero rows and columns at the specified indices.
    Inverse operation of delete_row_column_indices.
    
    Args:
        arr (npt.NDArray[DType]): 2D array to expand
        indices_insert (Sequence[int] | npt.NDArray[np.int_]): Indices where to insert
            zeros
    
    Returns:
        npt.NDArray[DType]: Expanded array with zeros at specified locations
    """
    arr = arr.copy()
    insert = np.zeros(arr.shape[1], dtype=arr.dtype)
    for idi in indices_insert:
        arr = np.insert(arr, idi, insert, axis=0)
    insert = np.zeros(arr.shape[0], dtype=arr.dtype)
    for idi in indices_insert:
        arr = np.insert(arr, idi, insert, axis=1)
    return arr


def BR_to_C_array(BR: npt.NDArray[np.floating], Gamma: float) -> npt.NDArray[np.floating]:
    """Convert branching ratios to collapse operator matrix elements.
    
    Computes C = √(BR * Γ) for Lindblad collapse operators.
    
    Args:
        BR (npt.NDArray[np.floating]): Branching ratio array
        Gamma (float): Decay rate in rad/s
    
    Returns:
        npt.NDArray[np.floating]: Collapse operator array
    """
    return np.sqrt(BR * Gamma)


def C_array_to_BR(
    C_array: npt.NDArray[np.floating], Gamma: float
) -> npt.NDArray[np.floating]:
    """Convert collapse operator matrix elements to branching ratios.
    
    Computes BR = C² / Γ, inverse of BR_to_C_array.
    
    Args:
        C_array (npt.NDArray[np.floating]): Collapse operator array
        Gamma (float): Decay rate in rad/s
    
    Returns:
        npt.NDArray[np.floating]: Branching ratio array
    """
    return C_array**2 / Gamma


def compact_BR_array_indices(
    BR_array: npt.NDArray[np.floating], indices_compact: npt.NDArray[np.int_]
) -> npt.NDArray[np.floating]:
    new_shape = np.asarray(BR_array[0].shape) - len(indices_compact) + 1
    BR_array_new = []
    for BR in BR_array:
        id1, id2 = np.nonzero(BR)
        id1 = id1[0]
        id2 = id2[0]
        if (id1 not in indices_compact) & (id2 not in indices_compact):
            BR_array_new.append(delete_row_column_indices(BR, indices_compact))

    BR_sum = np.zeros(new_shape, "complex")
    BR_sum[indices_compact[0], :] = np.sum(BR_array, axis=0)[indices_compact].sum(
        axis=0
    )[
        -new_shape[0] :  # noqa: 203
    ]
    for id1, id2 in zip(*np.nonzero(BR_sum)):
        BR_new = np.zeros(new_shape, "complex")
        BR_new[id1, id2] = BR_sum[id1, id2]
        BR_array_new.append(BR_new)
    return np.array(BR_array_new)


def compact_C_array_indices(
    C_array: npt.NDArray[np.floating],
    Gamma: float,
    indices_compact: npt.NDArray[np.int_],
) -> npt.NDArray[np.floating]:
    BR = C_array_to_BR(C_array, Gamma)
    BR_compact = compact_BR_array_indices(BR, indices_compact)
    C_array_compact = BR_to_C_array(BR_compact, Gamma)
    return C_array_compact


def compact_coupling_field(
    fields: CouplingFields,
    QN: Sequence[states.CoupledState],
    qn_compact: Union[Sequence[states.QuantumSelector], states.QuantumSelector],
) -> CouplingFields:
    if isinstance(qn_compact, states.QuantumSelector):
        qn_compact = [qn_compact]

    QN_compact = copy.deepcopy(QN)
    for qnc in qn_compact:
        indices_compact = states.get_indices_quantumnumbers(qnc, QN_compact)
        QN_compact = states.compact_QN_coupled_indices(QN_compact, indices_compact)
        fields = compact_coupling_field_indices(fields, indices_compact)
    return fields


def compact_coupling_field_indices(
    fields: CouplingFields,
    indices_compact: Union[Sequence[int], npt.NDArray[np.int_]],
) -> CouplingFields:
    field = []
    for cf in fields.fields:
        f = copy.deepcopy(cf.field)
        f = delete_row_column_indices(f, indices_compact)
        field.append(CouplingField(polarization=cf.polarization, field=f))

    return CouplingFields(
        ground_main=fields.ground_main,
        excited_main=fields.excited_main,
        main_coupling=fields.main_coupling,
        ground_states=fields.ground_states,
        excited_states=fields.excited_states,
        fields=field,
    )


def insert_levels_coupling_field(
    fields: CouplingFields,
    indices_insert: Sequence[int],
) -> CouplingFields:
    field = []
    for cf in fields.fields:
        f = copy.deepcopy(cf.field)
        f = insert_row_column_indices(f, indices_insert)
        field.append(CouplingField(polarization=cf.polarization, field=f))
    return CouplingFields(
        ground_main=fields.ground_main,
        excited_main=fields.excited_main,
        main_coupling=fields.main_coupling,
        ground_states=fields.ground_states,
        excited_states=fields.excited_states,
        fields=field,
    )
