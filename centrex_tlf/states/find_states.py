import warnings
from dataclasses import dataclass
from itertools import product
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from .states import (
    BasisState,
    CoupledBasisState,
    CoupledState,
    ElectronicState,
    UncoupledBasisState,
    UncoupledState,
)
from .utils import get_unique_list, reorder_evecs

__all__ = [
    "QuantumSelector",
    "find_state_idx_from_state",
    "find_exact_states_indices",
    "find_exact_states",
    "find_closest_vector_idx",
    "check_approx_state_exact_state",
    "get_indices_quantumnumbers_base",
    "get_indices_quantumnumbers",
    "get_unique_basisstates_from_basisstates",
    "get_unique_basisstates_from_states",
]


@dataclass
class QuantumSelector:
    """Class for setting quantum numbers for selecting a subset of states
    from a larger set of states

    Args:
        J: Rotational quantum number (int or sequence of ints)
        F1: Intermediate hyperfine quantum number (float or sequence of floats)
        F: Total hyperfine quantum number (int or sequence of ints)
        mF: Magnetic quantum number (int/float or sequence)
        electronic: Electronic state (ElectronicState or sequence)
        P: Parity quantum number (int, callable, or sequence). Can be None, 1, -1,
           or a callable that takes a state and returns bool
        Ω: Omega quantum number (currently not supported in get_indices)

    Note:
        Any quantum number set to None will match all states (no filtering on that number).
        Quantum numbers can be single values or sequences (list, tuple, array) to match
        multiple possible values.
    """

    J: Optional[Union[Sequence[int], npt.NDArray[np.int_], int]] = None
    F1: Optional[Union[Sequence[float], npt.NDArray[np.floating], float]] = None
    F: Optional[Union[Sequence[int], npt.NDArray[np.int_], int]] = None
    mF: Optional[Union[Sequence[int], npt.NDArray[np.int_], float]] = None
    electronic: Optional[
        Union[Sequence[ElectronicState], npt.NDArray[Any], ElectronicState]
    ] = None
    P: Optional[Union[Callable, Sequence[int], npt.NDArray[np.int_], int]] = None
    Ω: Optional[Union[Sequence[int], npt.NDArray[np.int_], int]] = None

    def get_indices(
        self,
        QN: Union[
            Sequence[CoupledState], Sequence[CoupledBasisState], npt.NDArray[Any]
        ],
        mode: str = "python",
    ) -> npt.NDArray[np.int_]:
        return get_indices_quantumnumbers_base(self, QN, mode)


def find_state_idx_from_state(
    H: npt.NDArray[np.complex128],
    reference_state: CoupledState,
    QN: Sequence[CoupledState],
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
) -> int:
    """Determine the index of the state vector most closely corresponding to an
    input state.

    Args:
        H: Hamiltonian matrix to diagonalize
        reference_state: State to find closest eigenstate match for
        QN: List of state objects defining the basis for H
        V_ref: Optional reference eigenvectors for consistent ordering

    Returns:
        Index of the eigenstate with highest overlap with reference_state

    Note:
        This function uses a greedy approach (argmax). For finding multiple
        states simultaneously, consider using find_exact_states_indices which
        uses optimal assignment to prevent duplicates.
    """
    # Determine state vector of reference state
    reference_state_vec = reference_state.state_vector(QN)

    # Find eigenvectors of the given Hamiltonian
    E, V = np.linalg.eigh(H)

    if V_ref is not None:
        E, V = reorder_evecs(V, E, V_ref)

    # Calculate overlap probabilities: |<ref|exact>|²
    # Fix: Use np.abs before squaring to get proper probabilities
    overlaps = np.dot(np.conj(reference_state_vec), V)
    probabilities = np.abs(overlaps) ** 2

    idx = int(np.argmax(probabilities))

    return idx


def find_closest_vector_idx(
    state_vec: npt.NDArray[np.complex128], vector_array: npt.NDArray[np.complex128]
) -> int:
    """Function that finds the index of the vector in vector_array that most closely
    matches state_vec. vector_array is array where each column is a vector, typically
    corresponding to an eigenstate of some Hamiltonian.

    inputs:
    state_vec = Numpy array, 1D
    vector_array = Numpy array, 2D

    returns:
    idx = index that corresponds to closest matching vector
    """

    overlaps = np.abs(state_vec.conj().T @ vector_array)
    idx = int(np.argmax(overlaps))

    return idx


def check_approx_state_exact_state(approx: CoupledState, exact: CoupledState) -> None:
    """Check if the exact found states match the approximate states.

    The exact states are found from the eigenvectors of the hamiltonian and are
    often a superposition of various states. The approximate states are used in
    initial setup of the hamiltonian. This function validates that the largest
    component of the exact state matches the approximate state's quantum numbers.

    Args:
        approx: Approximate state to validate against
        exact: Exact eigenstate found from diagonalization

    Raises:
        TypeError: If the largest components are not both CoupledBasisState
        ValueError: If quantum numbers don't match
        NotImplementedError: If state types are not supported
    """
    approx_largest = approx.find_largest_component()
    exact_largest = exact.find_largest_component()

    # Check if both are the same type using isinstance
    if not isinstance(approx_largest, type(exact_largest)):
        raise TypeError(
            f"can't compare approx ({type(approx_largest).__name__}) and exact "
            f"({type(exact_largest).__name__}), not equal types"
        )

    if isinstance(approx_largest, CoupledBasisState) and isinstance(
        exact_largest, CoupledBasisState
    ):
        # Check electronic state
        if approx_largest.electronic_state != exact_largest.electronic_state:
            raise ValueError(
                f"mismatch in electronic state: {approx_largest.electronic_state} != "
                f"{exact_largest.electronic_state}"
            )

        # Check J
        if approx_largest.J != exact_largest.J:
            raise ValueError(f"mismatch in J: {approx_largest.J} != {exact_largest.J}")

        # Check F
        if approx_largest.F != exact_largest.F:
            raise ValueError(f"mismatch in F: {approx_largest.F} != {exact_largest.F}")

        # Check F1
        if approx_largest.F1 != exact_largest.F1:
            raise ValueError(
                f"mismatch in F1: {approx_largest.F1} != {exact_largest.F1}"
            )

        # Check mF
        if approx_largest.mF != exact_largest.mF:
            raise ValueError(
                f"mismatch in mF: {approx_largest.mF} != {exact_largest.mF}"
            )
    else:
        raise NotImplementedError(
            f"check_approx_state_exact_state not implemented for state type "
            f"{type(approx_largest).__name__}"
        )


@overload
def find_exact_states_indices(
    states_approx: CoupledState,
    QN_construct: CoupledBasisState,
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
    overlap_threshold: float = 0.5,
    use_optimal_assignment: bool = True,
) -> npt.NDArray[np.int_]: ...


@overload
def find_exact_states_indices(
    states_approx: UncoupledState,
    QN_construct: UncoupledBasisState,
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
    overlap_threshold: float = 0.5,
    use_optimal_assignment: bool = True,
) -> npt.NDArray[np.int_]: ...


def find_exact_states_indices(
    states_approx,
    QN_construct,
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
    overlap_threshold: float = 0.5,
    use_optimal_assignment: bool = True,
) -> npt.NDArray[np.int_]:
    """
    Find the indices for the closest approximate eigenstates corresponding to
    states_approx for a Hamiltonian constructed from the quantum states QN_construct.

    Uses the Hungarian algorithm (linear_sum_assignment) to find the optimal one-to-one
    mapping between approximate states and eigenstates that maximizes total overlap.

    Args:
        states_approx (Sequence[State]): approximate states to find the indices for
        QN_construct (Sequence[State]): basis states from which H was constructed
        H (Optional[npt.NDArray[np.complex128]], optional): Hamiltonian matrix.
            Must be provided if V is None. Defaults to None.
        V (Optional[npt.NDArray[np.complex128]], optional): Eigenvectors of H in the
            construction basis. If None, computed from H. Defaults to None.
        V_ref (Optional[npt.NDArray[np.complex128]], optional): Reference eigenvectors
            for consistent ordering across calculations. Defaults to None.
        overlap_threshold (float, optional): Minimum overlap to warn about poor state
            matching. Defaults to 0.5.
        use_optimal_assignment (bool, optional): If True, use linear_sum_assignment
            for optimal one-to-one mapping. If False, use greedy argmax approach
            (kept for backward compatibility). Defaults to True.

    Returns:
        npt.NDArray[np.int_]: Array of indices mapping each state in states_approx
            to its closest match in the eigenvectors.

    Raises:
        ValueError: If neither H nor V is provided, or if optimal assignment fails.
        UserWarning: If any overlap is below overlap_threshold.
    """
    if V is None and H is None:
        raise ValueError("Must provide either H (Hamiltonian) or V (eigenvectors)")

    # Generate state vectors for states_approx in the construction basis
    # Note: These are NOT eigenstates, just representations in the construction basis
    state_vecs = np.array([s.state_vector(QN_construct) for s in states_approx])

    # Get or compute eigenvectors
    if V is None:
        assert H is not None  # Already checked above, but helps type checker
        _V = np.linalg.eigh(H)[1]
    else:
        _V = V.copy()  # Avoid modifying input

    # Reorder eigenvectors if reference provided
    if V_ref is not None:
        _, _V = reorder_evecs(
            _V, np.ones(len(QN_construct), dtype=np.complex128), V_ref
        )

    # Calculate overlap probabilities: |<approx|exact>|²
    # Fix: Use np.abs before squaring to get proper probabilities
    overlaps = np.abs(np.dot(np.conj(state_vecs), _V)) ** 2

    n_approx = len(states_approx)
    n_eigen = _V.shape[1]

    if use_optimal_assignment:
        # Use Hungarian algorithm for optimal one-to-one assignment
        # Convert overlap (similarity) to cost (dissimilarity)
        cost_matrix = 1.0 - overlaps

        if n_approx <= n_eigen:
            # More eigenstates than approximate states - optimal case
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            indices = col_ind
        else:
            # More approximate states than eigenstates - problematic
            # Pad with high cost dummy eigenstates to enable assignment
            padded_cost = np.ones((n_approx, n_approx))
            padded_cost[:, :n_eigen] = cost_matrix
            row_ind, col_ind = linear_sum_assignment(padded_cost)

            # Check if any approximate state was assigned to dummy eigenstate
            if np.any(col_ind >= n_eigen):
                unassigned = np.where(col_ind >= n_eigen)[0]
                raise ValueError(
                    f"Cannot uniquely assign all approximate states to eigenstates. "
                    f"Number of approximate states ({n_approx}) exceeds number of "
                    f"eigenstates ({n_eigen}). Unassigned approximate states: {unassigned.tolist()}"
                )
            indices = col_ind

        max_overlaps = overlaps[row_ind, indices]
    else:
        # Original greedy approach (kept for backward compatibility)
        indices = np.argmax(overlaps, axis=1)
        max_overlaps = np.max(overlaps, axis=1)

        # Check for duplicate assignments
        unique_indices, counts = np.unique(indices, return_counts=True)
        if len(unique_indices) != len(indices):
            duplicate_idx = unique_indices[counts > 1]
            conflicting_states = [
                i for i, idx in enumerate(indices) if idx in duplicate_idx
            ]
            raise ValueError(
                f"Multiple approximate states map to the same eigenstate. "
                f"Conflicting approximate state indices: {conflicting_states}, "
                f"mapping to eigenstate indices: {indices[conflicting_states].tolist()}. "
                f"Consider using use_optimal_assignment=True."
            )

    # Warn about poor overlaps
    poor_overlaps = max_overlaps < overlap_threshold
    if np.any(poor_overlaps):
        poor_indices = np.where(poor_overlaps)[0]
        warnings.warn(
            f"Low overlap detected for approximate states at indices {poor_indices.tolist()}. "
            f"Overlaps: {max_overlaps[poor_overlaps].tolist()}. "
            f"The approximate states may not be well-represented in the eigenstate basis.",
            UserWarning,
            stacklevel=2,
        )

    return indices.astype(np.int_)


@overload
def find_exact_states(
    states_approx: Sequence[CoupledState],
    QN_construct: Union[Sequence[CoupledBasisState], Sequence[CoupledState]],
    QN_basis: Sequence[CoupledState],
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
    overlap_threshold: float = 0.5,
    use_optimal_assignment: bool = True,
) -> List[CoupledState]: ...


@overload
def find_exact_states(
    states_approx: Sequence[UncoupledState],
    QN_construct: Union[Sequence[UncoupledBasisState], Sequence[UncoupledState]],
    QN_basis: Sequence[UncoupledState],
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
    overlap_threshold: float = 0.5,
    use_optimal_assignment: bool = True,
) -> List[UncoupledState]: ...


def find_exact_states(
    states_approx,
    QN_construct,
    QN_basis,
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
    overlap_threshold: float = 0.5,
    use_optimal_assignment: bool = True,
):
    """Find the closest eigenstates to a set of approximate states.

    This function maps approximate quantum states to their closest representation
    in the eigenstate basis of a given Hamiltonian using optimal assignment.

    Args:
        states_approx (list): List of State objects to find the closest match to
        QN_construct (list): List of BasisState objects from which H was constructed
            (construction basis)
        QN_basis (list): List of State objects defining the eigenstate basis of H
        H (np.ndarray, optional): Hamiltonian matrix. Must be provided if V is None.
        V (np.ndarray, optional): Eigenvectors of H. If None, computed from H.
        V_ref (np.ndarray, optional): Reference eigenvectors for consistent ordering.
        overlap_threshold (float, optional): Minimum overlap threshold for warnings.
            Defaults to 0.5.
        use_optimal_assignment (bool, optional): If True, use linear_sum_assignment
            for optimal one-to-one mapping. Defaults to True.

    Returns:
        list: List of eigenstates from QN_basis that best match states_approx.

    Example:
        >>> # Find exact eigenstates from approximate states
        >>> exact_states = find_exact_states(
        ...     states_approx=ground_states,
        ...     QN_construct=basis_states,
        ...     QN_basis=eigenstates,
        ...     H=hamiltonian
        ... )
    """
    indices = find_exact_states_indices(
        states_approx,
        QN_construct,
        H,
        V,
        V_ref,
        overlap_threshold,
        use_optimal_assignment,
    )
    return [QN_basis[idx] for idx in indices]


def get_indices_quantumnumbers_base(
    qn_selector: QuantumSelector,
    QN: Union[Sequence[CoupledState], Sequence[CoupledBasisState], npt.NDArray[Any]],
    mode: str = "python",
) -> npt.NDArray[np.int_]:
    """Return the indices corresponding to all states in QN that correspond to
    the quantum numbers in QuantumSelector.

    Entries in QuantumSelector quantum numbers can be either single numbers or
    lists/arrays. States with all possible combinations of quantum numbers in
    QuantumSelector are found. Quantum numbers set to None will match all states
    (no filtering on that quantum number).

    Args:
        qn_selector: QuantumSelector class containing the quantum numbers to
            find corresponding indices for
        QN: List or array of CoupledState or CoupledBasisState objects
        mode: Index mode - "python" for 0-based indexing (default),
              "julia" for 1-based indexing

    Returns:
        Array of indices corresponding to the quantum numbers in the
        specified indexing mode.

    Raises:
        TypeError: If QN contains unsupported state types or qn_selector is wrong type
        ValueError: If mode is not "python" or "julia"

    Note:
        - P (parity) quantum number is supported. Note that P can be None, 1, or -1.
        - Ω quantum number in QuantumSelector is currently not supported and will be ignored.
        - For CoupledState objects, P is extracted from the largest component.
          If you need to filter by a callable P selector, states will be matched individually.
    """
    if not isinstance(qn_selector, QuantumSelector):
        raise TypeError(
            f"qn_selector must be a QuantumSelector object, got {type(qn_selector)}"
        )

    if len(QN) == 0:
        return np.array([], dtype=np.int_)

    # Extract quantum numbers based on state type
    first_element = QN[0]
    if isinstance(first_element, CoupledState):
        QN_coupled = cast(Sequence[CoupledState], QN)
        Js: npt.NDArray[np.int_] = np.array([s.largest.J for s in QN_coupled])
        F1s: npt.NDArray[np.floating] = np.array([s.largest.F1 for s in QN_coupled])
        Fs: npt.NDArray[np.int_] = np.array([s.largest.F for s in QN_coupled])
        mFs: npt.NDArray[np.floating] = np.array([s.largest.mF for s in QN_coupled])
        estates: npt.NDArray[Any] = np.array(
            [s.largest.electronic_state for s in QN_coupled]
        )
        Ps: npt.NDArray[Any] = np.array([s.largest.P for s in QN_coupled])
    elif isinstance(first_element, CoupledBasisState):
        QN_basis = cast(Sequence[CoupledBasisState], QN)
        Js = np.array([s.J for s in QN_basis])
        F1s = np.array([s.F1 for s in QN_basis])
        Fs = np.array([s.F for s in QN_basis])
        mFs = np.array([s.mF for s in QN_basis])
        estates = np.array([s.electronic_state for s in QN_basis])
        Ps = np.array([s.P for s in QN_basis])
    else:
        raise TypeError(
            f"get_indices_quantumnumbers_base() only supports CoupledState and "
            f"CoupledBasisState types, got {type(first_element)}"
        )

    # Generate all combinations of quantum numbers to match
    # Convert scalar values to lists for uniform handling
    fields: List[List[Any]] = []

    # Process J
    J_selector = qn_selector.J
    if J_selector is None:
        fields.append([None])
    elif isinstance(J_selector, (list, tuple, np.ndarray)):
        fields.append(list(J_selector))
    else:
        fields.append([J_selector])

    # Process F1
    F1_selector = qn_selector.F1
    if F1_selector is None:
        fields.append([None])
    elif isinstance(F1_selector, (list, tuple, np.ndarray)):
        fields.append(list(F1_selector))
    else:
        fields.append([F1_selector])

    # Process F
    F_selector = qn_selector.F
    if F_selector is None:
        fields.append([None])
    elif isinstance(F_selector, (list, tuple, np.ndarray)):
        fields.append(list(F_selector))
    else:
        fields.append([F_selector])

    # Process mF
    mF_selector = qn_selector.mF
    if mF_selector is None:
        fields.append([None])
    elif isinstance(mF_selector, (list, tuple, np.ndarray)):
        fields.append(list(mF_selector))
    else:
        fields.append([mF_selector])

    # Process electronic
    electronic_selector = qn_selector.electronic
    if electronic_selector is None:
        fields.append([None])
    elif isinstance(electronic_selector, (list, tuple, np.ndarray)):
        fields.append(list(electronic_selector))
    else:
        fields.append([electronic_selector])

    # Process P (parity)
    P_selector = qn_selector.P
    if P_selector is None:
        fields.append([None])
    elif callable(P_selector):
        # If P is a callable, we need to handle it differently
        # Apply the callable to each state individually
        # For now, treat it as selecting all states and filter later
        fields.append([P_selector])
    elif isinstance(P_selector, (list, tuple, np.ndarray)):
        fields.append(list(P_selector))
    else:
        fields.append([P_selector])

    combinations = product(*fields)

    # Build combined mask for all matching states
    mask = np.zeros(len(QN), dtype=bool)
    mask_all = np.ones(len(QN), dtype=bool)

    for J_val, F1_val, F_val, mF_val, estate_val, P_val in combinations:
        # Generate masks for each quantum number
        # If value is None, match all states (no filtering)
        mask_J = (Js == J_val) if J_val is not None else mask_all
        mask_F1 = (F1s == F1_val) if F1_val is not None else mask_all
        mask_F = (Fs == F_val) if F_val is not None else mask_all
        mask_mF = (mFs == mF_val) if mF_val is not None else mask_all
        mask_es = (estates == estate_val) if estate_val is not None else mask_all

        # Handle P (parity) - can be None, a value, or a callable
        if callable(P_val):
            # Apply callable to each state individually
            mask_P = np.array([P_val(s) for s in QN], dtype=bool)
        elif P_val is not None:
            mask_P = Ps == P_val
        else:
            mask_P = mask_all

        # Combine masks: state must match ALL specified quantum numbers
        mask = mask | (mask_J & mask_F1 & mask_F & mask_mF & mask_es & mask_P)

    # Return indices in requested format
    if mode == "python":
        return np.where(mask)[0]
    elif mode == "julia":
        return np.where(mask)[0] + 1
    else:
        raise ValueError(f"mode must be 'python' or 'julia', got '{mode}'")


def get_indices_quantumnumbers(
    qn_selector: Union[QuantumSelector, Sequence[QuantumSelector], npt.NDArray[Any]],
    QN: Union[Sequence[CoupledState], Sequence[CoupledBasisState], npt.NDArray[Any]],
) -> npt.NDArray[np.int_]:
    """return the indices corresponding to all states in QN that correspond to
    the quantum numbers in QuantumSelector or a list of QuantumSelector objects.
    Entries in QuantumSelector quantum numbers can be either single numbers or
    lists/arrays. States with all possible combinations of quantum numbers in
    QuantumSelector are found

    Args:
        qn_selector (Union[QuantumSelector, list, np.ndarray]):
                    QuantumSelector class or list/array of QuantumSelectors
                    containing the quantum numbers to find corresponding indices

        QN (Union[list, np.ndarray]): list or array of states

    Raises:
        AssertionError: only supports State and CoupledBasisState types the States list
        or array

    Returns:
        np.ndarray: indices corresponding to the quantum numbers
    """
    if isinstance(qn_selector, QuantumSelector):
        return get_indices_quantumnumbers_base(qn_selector, QN)
    elif isinstance(qn_selector, (list, np.ndarray)):
        return np.unique(
            np.concatenate(
                [get_indices_quantumnumbers_base(qns, QN) for qns in qn_selector]
            )
        )
    else:
        raise AssertionError(
            "qn_selector required to be of type QuantumSelector, list or np.ndarray"
        )


StateType = TypeVar("StateType")


def get_unique_basisstates_from_basisstates(
    states: Sequence[StateType],
) -> List[StateType]:
    """get a list/array of unique BasisStates in the states list/array

    Args:
        states (Sequence): list/array of BasisStates

    Returns:
        List: list/array of unique BasisStates
    """
    assert isinstance(states[0], BasisState), "Not a sequence of BasisState objects"
    return get_unique_list(states)


def get_unique_basisstates_from_states(
    states: Sequence[CoupledState],
) -> List[BasisState]:
    """
    get a Sequence of unique BasisStates in a sequence of State objects

    Args:
        states (Sequence[State]): Sequence of State objects

    Returns:
        Sequence[BasisState]: Sequence of unique BasisStates that comprise the input
                                States
    """
    assert isinstance(states[0], CoupledState), "Not a sequence of State objects"
    return get_unique_basisstates_from_basisstates(
        [s for S in states for a, s in S.data]
    )
