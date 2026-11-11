"""Utilities for compacting quantum state manifolds in coupling calculations.

This module provides functions for reducing the dimensionality of coupling matrices
and branching ratio arrays by combining degenerate or nearly-degenerate states into
single manifolds.
"""
import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, TypeVar, Union

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
    "CollapseOperators",
    "compact_collapse_operators",
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
        Gamma (float): Population decay rate in s^-1
    
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
        Gamma (float): Population decay rate in s^-1
    
    Returns:
        npt.NDArray[np.floating]: Branching ratio array
    """
    return C_array**2 / Gamma


def compact_BR_array_indices(
    BR_array: npt.NDArray[np.floating], indices_compact: npt.NDArray[np.int_]
) -> npt.NDArray[np.floating]:
    new_shape = np.asarray(BR_array.shape[1:]) - len(indices_compact) + 1
    if BR_array.shape[0] == 0:
        return np.zeros((0, *new_shape), dtype=BR_array.dtype)
    # Row and column of the first nonzero element of every matrix, in one pass
    # instead of one np.nonzero call per matrix.
    nonzero = BR_array.reshape(BR_array.shape[0], -1) != 0
    if not nonzero.any(axis=1).all():
        raise IndexError("branching ratio matrix without a nonzero element")
    id1, id2 = np.divmod(nonzero.argmax(axis=1), BR_array.shape[2])
    keep = ~np.isin(id1, indices_compact) & ~np.isin(id2, indices_compact)
    BR_array_new = list(
        np.delete(
            np.delete(BR_array[keep], indices_compact[1:], axis=1),
            indices_compact[1:],
            axis=2,
        )
    )

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


@dataclass
class CollapseOperators:
    """Collapse operators that each have a single nonzero element.

    Operator ``k`` is the ``n x n`` matrix that is zero except for ``values[k]``
    at ``(rows[k], cols[k])``. This is what `collapse_matrices` produces, and
    carrying the three arrays instead of ``(n_operators, n, n)`` dense matrices
    keeps compaction proportional to the number of operators.
    """

    rows: npt.NDArray[np.intp]
    cols: npt.NDArray[np.intp]
    values: npt.NDArray[Any]
    n: int

    def to_dense(self) -> npt.NDArray[Any]:
        dense = np.zeros((len(self.values), self.n, self.n), dtype=self.values.dtype)
        dense[np.arange(len(self.values)), self.rows, self.cols] = self.values
        return dense


def compact_collapse_operators(
    operators: CollapseOperators,
    Gamma: float,
    indices_compact: npt.NDArray[np.int_],
) -> Optional[CollapseOperators]:
    """`compact_C_array_indices` for single-element collapse operators.

    Performs the same arithmetic, in the same order and with the same dtypes, as
    the dense routine applied to ``operators.to_dense()``, so the result is
    identical to it. Returns None for inputs the dense routine treats specially
    (no operators, zero elements, two operators on the same element, an empty
    result); the caller then uses the dense routine.
    """
    rows, cols, n = operators.rows, operators.cols, operators.n
    indices_compact = np.asarray(indices_compact)
    if (
        len(rows) == 0
        or len(indices_compact) == 0
        or not np.all(operators.values != 0)
        or len(np.unique(rows * n + cols)) != len(rows)
    ):
        return None
    removed = np.sort(indices_compact[1:])
    n_new = n - len(removed)
    representative = int(indices_compact[0])
    if not 0 <= representative < n_new:
        return None

    BR = C_array_to_BR(operators.values, Gamma)

    # operators that do not touch the compacted states: renumber past the
    # removed rows and columns
    keep = ~np.isin(rows, indices_compact) & ~np.isin(cols, indices_compact)
    kept_rows = rows[keep] - np.searchsorted(removed, rows[keep])
    kept_cols = cols[keep] - np.searchsorted(removed, cols[keep])

    # decays into the compacted states: per column, the sum over the compacted
    # rows in the order they are listed
    column_sum = np.zeros(n, dtype=BR.dtype)
    for row in indices_compact:
        in_row = rows == row
        column_sum[cols[in_row]] += BR[in_row]
    # the dense routine keeps the last n_new columns of this sum
    merged = column_sum[n - n_new :].astype(complex)
    merged_cols = np.flatnonzero(merged)

    if len(merged_cols):
        BR_new = np.concatenate([BR[keep].astype(complex), merged[merged_cols]])
    else:
        BR_new = BR[keep]
    if len(BR_new) == 0:
        return None

    return CollapseOperators(
        np.concatenate([kept_rows, np.full(len(merged_cols), representative, dtype=np.intp)]),
        np.concatenate([kept_cols, merged_cols]),
        BR_to_C_array(BR_new, Gamma),
        n_new,
    )


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
