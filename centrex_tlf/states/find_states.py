"""State finding and quantum number selection utilities.

This module provides tools for:
- Selecting subsets of quantum states using QuantumSelector
- Finding eigenstates that best match approximate reference states
- Optimal assignment of approximate states to exact eigenstates using Hungarian algorithm
- Extracting unique basis states from state collections

The main use cases are:
1. Finding dressed eigenstates from bare states after field interactions
2. Tracking state evolution across parameter sweeps
3. Filtering states by quantum numbers for reduced Hilbert spaces
"""

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
    """Selector for filtering quantum states by quantum numbers.

    Provides flexible state selection by specifying quantum number constraints.
    Each quantum number can be:
    - None: matches all values (no filtering)
    - Single value: matches only that value
    - Sequence: matches any value in the sequence

    Multiple quantum numbers are combined with AND logic (state must match all
    specified criteria). Multiple selectors can be combined with OR logic using
    get_indices_quantumnumbers().

    Attributes:
        J (int | Sequence[int] | None): Rotational quantum number(s). Defaults to None.
        F1 (float | Sequence[float] | None): Intermediate hyperfine quantum number(s)
            for coupled representation. Can be half-integer. Defaults to None.
        F (int | Sequence[int] | None): Total angular momentum quantum number(s).
            Defaults to None.
        mF (int | float | Sequence | None): Magnetic quantum number(s). Can be
            half-integer for half-integer F. Defaults to None.
        electronic (ElectronicState | Sequence[ElectronicState] | None): Electronic
            state(s) (X, B, etc.). Defaults to None.
        P (int | Callable | Sequence[int] | None): Parity quantum number. Can be:
            - None: match all parities
            - 1 or -1: match specific parity
            - Callable: function taking state and returning bool
            - Sequence: match any parity in sequence
            Defaults to None.
        Ω (int | Sequence[int] | None): Projection of electronic angular momentum.
            Currently not supported in get_indices (reserved for future use).
            Defaults to None.

    Example:
        >>> # Select all J=0 states in X electronic state
        >>> selector = QuantumSelector(J=0, electronic=ElectronicState.X)
        >>> indices = selector.get_indices(QN)
        >>>
        >>> # Select J=0 or J=1, with even parity only
        >>> selector = QuantumSelector(J=[0, 1], P=1)
        >>>
        >>> # Use callable for complex selection logic
        >>> selector = QuantumSelector(P=lambda s: s.largest.J % 2 == 0)

    Note:
        For CoupledState objects, quantum numbers are extracted from the largest
        component (s.largest). This works well for states that are predominantly
        one basis state.
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
        """Find indices of states matching the quantum number criteria.

        Args:
            QN (Sequence[CoupledState] | Sequence[CoupledBasisState]): States to
                search through.
            mode (str): Indexing convention. "python" for 0-based (default),
                "julia" for 1-based.

        Returns:
            npt.NDArray[np.int_]: Array of matching state indices.

        Example:
            >>> selector = QuantumSelector(J=0, F=1)
            >>> indices = selector.get_indices(ground_states)
            >>> selected_states = [ground_states[i] for i in indices]
        """
        return get_indices_quantumnumbers_base(self, QN, mode)


def find_state_idx_from_state(
    H: npt.NDArray[np.complex128],
    reference_state: CoupledState,
    QN: Sequence[CoupledState],
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
) -> int:
    """Find eigenstate index with highest overlap to reference state.

    Diagonalizes the Hamiltonian and finds which eigenstate has maximum overlap
    |<reference|eigenstate>|² with the given reference state. Useful for tracking
    a single state across parameter sweeps.

    Args:
        H (npt.NDArray[np.complex128]): Hamiltonian matrix of shape (N, N) where
            N = len(QN).
        reference_state (CoupledState): Reference state to match against eigenstates.
        QN (Sequence[CoupledState]): Basis states used to construct H.
        V_ref (npt.NDArray[np.complex128] | None): Reference eigenvectors for
            consistent phase/ordering across multiple calls. Shape (N, N).
            Defaults to None.

    Returns:
        int: Index of the eigenstate with maximum overlap probability.

    Example:
        >>> # Track ground state across magnetic field sweep
        >>> H_B0 = generate_hamiltonian(B=0)
        >>> ground_idx = find_state_idx_from_state(H_B0, ground_state_approx, QN)
        >>> tracked_state = QN[ground_idx]

    Warning:
        This function uses a greedy argmax approach. If finding multiple states,
        use find_exact_states_indices() which uses optimal assignment to prevent
        multiple states mapping to the same eigenstate.

    Note:
        Overlap probability is computed as |<ψ_ref|ψ_exact>|² where ψ_ref is the
        reference state vector and ψ_exact is an eigenstate.
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
    """Find index of vector in array with maximum overlap to given vector.

    Computes |<state_vec|column_i>| for each column in vector_array and returns
    the index with maximum overlap. Typically used for matching states to eigenvectors.

    Args:
        state_vec (npt.NDArray[np.complex128]): Target state vector of shape (N,).
        vector_array (npt.NDArray[np.complex128]): Array of vectors where each
            column is a candidate vector. Shape (N, M) where M is number of vectors.

    Returns:
        int: Column index (0 to M-1) with highest overlap to state_vec.

    Example:
        >>> # Find which eigenstate matches ground state
        >>> E, V = np.linalg.eigh(H)
        >>> ground_vec = ground_state.state_vector(QN)
        >>> idx = find_closest_vector_idx(ground_vec, V)
        >>> ground_eigenstate = V[:, idx]
        >>> ground_energy = E[idx]

    Note:
        Uses absolute value of overlap, so relative phase between vectors is ignored.
        Overlap is computed as |<state_vec†|column>| without squaring.
    """

    overlaps = np.abs(state_vec.conj().T @ vector_array)
    idx = int(np.argmax(overlaps))

    return idx


def check_approx_state_exact_state(approx: CoupledState, exact: CoupledState) -> None:
    """Validate that exact eigenstate matches expected approximate state.

    Checks that the largest (dominant) component of an exact eigenstate has the same
    quantum numbers as the approximate (bare) state. Used to verify state assignment
    is correct after diagonalization.

    Args:
        approx (CoupledState): Approximate/bare state with expected quantum numbers.
        exact (CoupledState): Exact eigenstate from Hamiltonian diagonalization.

    Raises:
        TypeError: If largest components have mismatched types (e.g., one CoupledBasisState,
            one UncoupledBasisState).
        ValueError: If quantum numbers don't match. Separate error for each QN:
            electronic_state, J, F, F1, or mF.
        NotImplementedError: If state type is not CoupledBasisState.

    Example:
        >>> # Verify state assignment after field is applied
        >>> approx_state = ground_states[0]  # Bare state
        >>> exact_state = find_exact_states([approx_state], ...)[0]
        >>> check_approx_state_exact_state(approx_state, exact_state)  # Raises if mismatch

    Note:
        Only the dominant (largest amplitude) components are compared. For strongly
        mixed states, this check may not be meaningful.
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
    """Find optimal eigenstate indices matching approximate states.

    Uses the Hungarian algorithm (linear sum assignment) to find the best one-to-one
    mapping between approximate (bare) states and exact eigenstates that maximizes
    total overlap. This prevents multiple approximate states from being assigned to
    the same eigenstate, which can occur with greedy matching.

    Args:
        states_approx (Sequence[CoupledState] | Sequence[UncoupledState]): Approximate
            states to match to eigenstates.
        QN_construct (Sequence[BasisState]): Basis states used to construct H.
        H (npt.NDArray[np.complex128] | None): Hamiltonian matrix of shape (N, N).
            Must be provided if V is None. Defaults to None.
        V (npt.NDArray[np.complex128] | None): Pre-computed eigenvectors of H,
            shape (N, N). If None, computed from H. Defaults to None.
        V_ref (npt.NDArray[np.complex128] | None): Reference eigenvectors for
            consistent ordering/phase across multiple calculations. Defaults to None.
        overlap_threshold (float): Minimum overlap probability to warn about poor
            matching. Overlaps below this trigger UserWarning. Defaults to 0.5.
        use_optimal_assignment (bool): If True, use Hungarian algorithm for optimal
            assignment. If False, use greedy argmax (legacy behavior, may produce
            duplicates). Defaults to True.

    Returns:
        npt.NDArray[np.int_]: Array of shape (len(states_approx),) containing
            eigenstate index for each approximate state.

    Raises:
        ValueError: If neither H nor V is provided.
        ValueError: If more approximate states than eigenstates (when using optimal
            assignment).
        ValueError: If greedy assignment produces duplicates (when not using optimal
            assignment).
        UserWarning: If any overlap is below overlap_threshold.

    Example:
        >>> # Find dressed states from bare states
        >>> ground_indices = find_exact_states_indices(
        ...     states_approx=ground_states_bare,
        ...     QN_construct=basis_states,
        ...     H=hamiltonian_with_fields
        ... )
        >>> dressed_states = [eigenstates[i] for i in ground_indices]

    Note:
        - Overlap probability is |<approx|exact>|²
        - Hungarian algorithm minimizes total cost = Σ(1 - overlap_i)
        - For n approximate states and m eigenstates with n ≤ m, guarantees unique assignment
        - V_ref helps maintain consistent eigenstate ordering across parameter sweeps
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
    """Find exact eigenstates corresponding to approximate states.

    Maps approximate (bare/unperturbed) quantum states to their exact eigenstate
    representations in the presence of interactions (fields, couplings). Returns
    the actual State objects rather than just indices.

    Args:
        states_approx (Sequence[CoupledState] | Sequence[UncoupledState]): Approximate
            states to match against eigenstates.
        QN_construct (Sequence[BasisState]): Basis states used to construct H.
        QN_basis (Sequence[CoupledState] | Sequence[UncoupledState]): Complete set of
            eigenstate objects. These are the states to select from.
        H (npt.NDArray[np.complex128] | None): Hamiltonian matrix. Must be provided
            if V is None. Defaults to None.
        V (npt.NDArray[np.complex128] | None): Pre-computed eigenvectors. If None,
            computed from H. Defaults to None.
        V_ref (npt.NDArray[np.complex128] | None): Reference eigenvectors for
            consistent ordering. Defaults to None.
        overlap_threshold (float): Minimum overlap probability for warning. Defaults to 0.5.
        use_optimal_assignment (bool): Use Hungarian algorithm for optimal assignment.
            Defaults to True.

    Returns:
        List[CoupledState] | List[UncoupledState]: List of eigenstates from QN_basis
            that best match states_approx, in the same order as states_approx.

    Example:
        >>> # Find dressed states after applying electric field
        >>> bare_states = generate_coupled_states_X(J_max=2)
        >>> ground_states_bare = bare_states[:4]  # First 4 states
        >>>
        >>> H = generate_hamiltonian_with_field(E_field=1000)
        >>> eigenstates = generate_coupled_states_X(J_max=2)  # Same basis
        >>>
        >>> dressed_states = find_exact_states(
        ...     states_approx=ground_states_bare,
        ...     QN_construct=bare_states,
        ...     QN_basis=eigenstates,
        ...     H=H
        ... )
        >>> # dressed_states[i] is the dressed version of ground_states_bare[i]

    Note:
        This is a convenience wrapper around find_exact_states_indices() that returns
        the actual State objects instead of indices.
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
    """Find state indices matching quantum number selection criteria.

    Core implementation for filtering states by quantum numbers. Generates all
    combinations of specified quantum numbers and returns indices of states matching
    ANY combination (OR logic between combinations, AND logic within each combination).

    Args:
        qn_selector (QuantumSelector): Quantum number selection criteria. Each field
            can be None (match all), single value, or sequence of values.
        QN (Sequence[CoupledState] | Sequence[CoupledBasisState]): States to filter.
        mode (str): Indexing convention. "python" for 0-based (default), "julia"
            for 1-based. Defaults to "python".

    Returns:
        npt.NDArray[np.int_]: Sorted array of matching state indices.

    Raises:
        TypeError: If qn_selector is not QuantumSelector, or if QN contains
            unsupported state types (must be CoupledState or CoupledBasisState).
        ValueError: If mode is not "python" or "julia".

    Example:
        >>> # Find all J=0 and J=1 states with mF=0
        >>> selector = QuantumSelector(J=[0, 1], mF=0)
        >>> indices = get_indices_quantumnumbers_base(selector, QN)
        >>> selected_states = [QN[i] for i in indices]
        >>>
        >>> # Select using callable for complex logic
        >>> selector = QuantumSelector(P=lambda s: s.largest.J % 2 == 0)
        >>> even_J_indices = get_indices_quantumnumbers_base(selector, QN)

    Note:
        - For CoupledState objects, quantum numbers extracted from s.largest component
        - Parity (P) can be: None, 1, -1, sequence [1, -1], or callable
        - Omega (Ω) field in selector is currently ignored (reserved for future)
        - Empty QN returns empty array
        - Multiple selector values create OR logic: J=[0,1] matches J=0 OR J=1
        - Multiple quantum numbers create AND logic: J=0, mF=0 matches J=0 AND mF=0
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
    """Find state indices using single or multiple quantum number selectors.

    Wrapper around get_indices_quantumnumbers_base() that handles both single
    QuantumSelector and sequences of selectors. Multiple selectors are combined
    with OR logic (union of matching states).

    Args:
        qn_selector (QuantumSelector | Sequence[QuantumSelector]): Single selector
            or sequence of selectors to apply.
        QN (Sequence[CoupledState] | Sequence[CoupledBasisState]): States to filter.

    Returns:
        npt.NDArray[np.int_]: Sorted array of unique matching state indices
            (0-based Python indexing).

    Raises:
        AssertionError: If qn_selector is not QuantumSelector, list, or array.

    Example:
        >>> # Single selector
        >>> selector = QuantumSelector(J=0)
        >>> indices = get_indices_quantumnumbers(selector, QN)
        >>>
        >>> # Multiple selectors (OR logic)
        >>> selector1 = QuantumSelector(J=0, mF=0)
        >>> selector2 = QuantumSelector(J=1, mF=1)
        >>> indices = get_indices_quantumnumbers([selector1, selector2], QN)
        >>> # Returns states matching (J=0, mF=0) OR (J=1, mF=1)

    Note:
        When using multiple selectors, results are combined with OR:
        - Single selector: match all criteria (AND within selector)
        - Multiple selectors: match any selector (OR between selectors)
        - Duplicates automatically removed via np.unique()
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
    """Extract unique basis states from a sequence of basis states.

    Removes duplicate basis states while preserving order. Useful for reducing
    redundancy in basis state collections.

    Args:
        states (Sequence[StateType]): Sequence of BasisState objects (can be
            CoupledBasisState, UncoupledBasisState, etc.).

    Returns:
        List[StateType]: List of unique BasisState objects in order of first appearance.

    Raises:
        AssertionError: If states[0] is not a BasisState object.

    Example:
        >>> basis_states = [state1, state2, state1, state3, state2]
        >>> unique_states = get_unique_basisstates_from_basisstates(basis_states)
        >>> # Returns [state1, state2, state3]

    Note:
        Uses equality comparison to determine uniqueness. BasisStates with identical
        quantum numbers are considered duplicates.
    """
    assert isinstance(states[0], BasisState), "Not a sequence of BasisState objects"
    return get_unique_list(states)


def get_unique_basisstates_from_states(
    states: Sequence[CoupledState],
) -> List[BasisState]:
    """Extract all unique basis states from superposition states.

    Decomposes State objects (which are superpositions of BasisStates) into their
    constituent BasisStates and returns the unique set. Useful for finding the
    minimal basis needed to represent a set of states.

    Args:
        states (Sequence[CoupledState]): Sequence of CoupledState objects. Each may
            be a superposition of multiple basis states.

    Returns:
        List[BasisState]: List of unique BasisState objects that appear in any of
            the input states, in order of first appearance.

    Raises:
        AssertionError: If states[0] is not a CoupledState object.

    Example:
        >>> # Each state is superposition: |ψ⟩ = Σ_i c_i |basis_i⟩
        >>> state1 = CoupledState([(0.7, basis_a), (0.3, basis_b)])
        >>> state2 = CoupledState([(0.6, basis_b), (0.4, basis_c)])
        >>> unique_basis = get_unique_basisstates_from_states([state1, state2])
        >>> # Returns [basis_a, basis_b, basis_c]

    Note:
        - Extracts all (amplitude, basis_state) pairs from state.data
        - Keeps only unique basis states (amplitudes discarded)
        - Useful for constructing minimal Hilbert space for calculations
    """
    assert isinstance(states[0], CoupledState), "Not a sequence of State objects"
    return get_unique_basisstates_from_basisstates(
        [s for S in states for a, s in S.data]
    )
