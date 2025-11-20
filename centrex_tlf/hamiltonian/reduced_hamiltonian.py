import warnings
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from scipy import linalg

from centrex_tlf.constants import BConstants, TlFNuclearSpins, XConstants
from centrex_tlf.states import (
    Basis,
    CoupledBasisState,
    CoupledState,
    ElectronicState,
    QuantumSelector,
    UncoupledBasisState,
    find_exact_states,
    generate_uncoupled_states_ground,
    get_unique_basisstates_from_states,
)
from centrex_tlf.states.generate_states import (
    generate_coupled_states_B,
    generate_coupled_states_ground,
    generate_coupled_states_X,
)
from centrex_tlf.transitions import MicrowaveTransition, OpticalTransition

from .basis_transformations import generate_transform_matrix
from .generate_hamiltonian import (
    Hamiltonian,
    generate_coupled_hamiltonian_B,
    generate_coupled_hamiltonian_B_function,
    generate_uncoupled_hamiltonian_X,
    generate_uncoupled_hamiltonian_X_function,
)
from .matrix_elements_electric_dipole import generate_ED_ME_mixed_state
from .utils import matrix_to_states, reduced_basis_hamiltonian, reorder_evecs

__all__ = [
    "generate_diagonalized_hamiltonian",
    "generate_reduced_X_hamiltonian",
    "generate_reduced_B_hamiltonian",
    "compose_reduced_hamiltonian",
    "generate_total_reduced_hamiltonian",
    "generate_reduced_hamiltonian_transitions",
]


@dataclass
class HamiltonianDiagonalized:
    H: npt.NDArray[np.complex128]
    V: npt.NDArray[np.complex128]
    V_ref: Optional[npt.NDArray[np.complex128]] = None


def generate_diagonalized_hamiltonian(
    hamiltonian: npt.NDArray[np.complex128],
    keep_order: bool = True,
    return_V_ref: bool = False,
    rtol: Optional[float] = None,
) -> HamiltonianDiagonalized:
    """Diagonalize a Hamiltonian matrix.

    Diagonalizes the Hamiltonian and optionally maintains state ordering
    using reference eigenvectors.

    Args:
        hamiltonian: Hamiltonian matrix to diagonalize
        keep_order: If True, reorder eigenvectors for consistent state tracking.
            Defaults to True.
        return_V_ref: If True, return reference eigenvectors. Defaults to False.
        rtol: Relative tolerance for zeroing small matrix elements. If provided,
            elements smaller than rtol * max(|H|) are set to zero.

    Returns:
        HamiltonianDiagonalized: Dataclass containing diagonalized Hamiltonian (H),
            eigenvectors (V), and optionally reference eigenvectors (V_ref)
    """
    _ = np.linalg.eigh(hamiltonian)
    D: npt.NDArray[np.complex128] = _[0]
    V: npt.NDArray[np.complex128] = _[1]
    V_ref: Optional[npt.NDArray[np.complex128]] = None
    if keep_order:
        V_ref = np.eye(V.shape[0], dtype=np.complex128)
        D, V = reorder_evecs(V, D, V_ref)

    hamiltonian_diagonalized = V.conj().T @ hamiltonian @ V
    if rtol:
        hamiltonian_diagonalized[
            np.abs(hamiltonian_diagonalized)
            < np.abs(hamiltonian_diagonalized).max() * rtol
        ] = 0
    if not return_V_ref or not keep_order:
        return HamiltonianDiagonalized(hamiltonian_diagonalized, V)
    else:
        return HamiltonianDiagonalized(hamiltonian_diagonalized, V, V_ref)


@dataclass
class ReducedHamiltonian:
    H: npt.NDArray[np.complex128]
    V: npt.NDArray[np.complex128]
    QN_basis: List[CoupledState]
    QN_construct: List[CoupledBasisState]
    hamiltonian: Hamiltonian | None = None
    transform: npt.NDArray[np.complex128] | None = None
    QN_pretransform: Sequence[UncoupledBasisState] | None = None

    def __iter__(
        self,
    ) -> Iterator[Union[List[CoupledState], npt.NDArray[np.complex128]]]:
        """Support for legacy code unpacking: ground_states, H_X_red = X_hamiltonian.

        Yields QN_basis first, then H to allow unpacking as:
            ground_states, H_X_red = X_hamiltonian

        Note: Due to limitations in Python's type system, type checkers cannot infer
        position-specific types from this iterator. For fully type-safe code,
        prefer accessing attributes directly:
            ground_states = X_hamiltonian.QN_basis
            H_X_red = X_hamiltonian.H
        """
        yield self.QN_basis
        yield self.H


def generate_reduced_X_hamiltonian(
    X_states_approx: Sequence[CoupledBasisState],
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-3]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin: Optional[int] = None,
    Jmax: Optional[int] = None,
    constants: XConstants = XConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    transform: Optional[npt.NDArray[np.complex128]] = None,
    H_func: Optional[Callable] = None,
) -> ReducedHamiltonian:
    """Generate reduced X state Hamiltonian for specified ground states.

    Constructs the Hamiltonian for all X states from Jmin to Jmax, then reduces it to
    only the subspace corresponding to X_states_approx. The X state Hamiltonian is
    constructed in the uncoupled basis, transformed to coupled basis, diagonalized,
    and then the eigenstates corresponding to X_states_approx are identified and used
    to build the reduced Hamiltonian.

    Args:
        X_states_approx: Approximate X (ground) states defining the reduced subspace
        E: Electric field in V/cm. Defaults to np.array([0.0, 0.0, 0.0]).
        B: Magnetic field in G. Defaults to np.array([0.0, 0.0, 1e-3]). If smaller
            than 1e-5, some states become degenerate.
        rtol: Remove Hamiltonian components smaller than rtol * max(|H|). Defaults to
            None.
        stol: Remove state components with amplitude smaller than stol. Defaults to
            1e-3.
        Jmin: Minimum J to include in full Hamiltonian construction. Defaults to None
            (uses minimum J from X_states_approx).
        Jmax: Maximum J to include in full Hamiltonian construction. Defaults to None
            (uses maximum J from X_states_approx).
        constants: X state molecular constants. Defaults to XConstants().
        nuclear_spins: TlF nuclear spin values. Defaults to TlFNuclearSpins().
        transform: Transformation matrix from uncoupled to coupled basis for J states
            from Jmin to Jmax. If None, generated automatically. Defaults to None.
        H_func: Function to generate the Hamiltonian as function of (E, B). If None,
            uses default X state Hamiltonian. Defaults to None.

    Returns:
        ReducedHamiltonian: Dataclass containing the reduced Hamiltonian matrix,
            eigenvectors, identified states, and construction information
    """

    # need to generate the other states in case of mixing
    _Jmin = min([gs.J for gs in X_states_approx]) if Jmin is None else Jmin
    _Jmax = max([gs.J for gs in X_states_approx]) if Jmax is None else Jmax

    QN = generate_uncoupled_states_ground(
        Js=np.arange(_Jmin, _Jmax + 1), nuclear_spins=nuclear_spins
    )
    QNc = generate_coupled_states_ground(
        Js=np.arange(_Jmin, _Jmax + 1), nuclear_spins=nuclear_spins
    )
    H_X_uc: Optional[Hamiltonian] = None
    if H_func is None:
        H_X_uc = generate_uncoupled_hamiltonian_X(QN, constants=constants)
        H_X_uc_func = generate_uncoupled_hamiltonian_X_function(H_X_uc)
    else:
        H_X_uc_func = H_func

    if transform is None:
        S_transform = generate_transform_matrix(QN, QNc)
    else:
        assert transform.shape[0] == len(QN), (
            (
                f"shape of transform incorrect; requires {len(QN), len(QN)}, "
                f"not {transform.shape}"
            ),
        )
        S_transform = transform

    H_X = S_transform.conj().T @ H_X_uc_func(E, B) @ S_transform
    if rtol:
        H_X[np.abs(H_X) < np.abs(H_X).max() * rtol] = 0

    # diagonalize the Hamiltonian
    H_diagonalized = generate_diagonalized_hamiltonian(
        H_X, keep_order=True, return_V_ref=True, rtol=rtol
    )

    # new set of quantum numbers:
    QN_diag = matrix_to_states(H_diagonalized.V, list(QNc))

    ground_states = find_exact_states(
        [1 * gs for gs in X_states_approx],
        list(QNc),
        QN_diag,
        V=H_diagonalized.V,
        # V_ref=H_diagonalized.V_ref_,
    )
    ground_states = [gs.remove_small_components(stol) for gs in ground_states]

    H_X_red = reduced_basis_hamiltonian(
        [qn.remove_small_components(stol) for qn in QN_diag],
        H_diagonalized.H,
        ground_states,
    )

    return ReducedHamiltonian(
        H=H_X_red,
        V=H_diagonalized.V,
        QN_basis=ground_states,
        QN_construct=list(QNc),
        hamiltonian=H_X_uc,
        transform=S_transform,
        QN_pretransform=cast(Sequence[UncoupledBasisState], QN),
    )


def generate_reduced_B_hamiltonian(
    B_states_approx: Sequence[CoupledBasisState],
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-5]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin: Optional[int] = None,
    Jmax: Optional[int] = None,
    constants: BConstants = BConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    H_func: Optional[Callable] = None,
) -> ReducedHamiltonian:
    """
    Generate the reduced B state hamiltonian.
    Generates the Hamiltonian for all B states from Jmin to Jmax, if provided. Otherwise
    uses Jmin = 1 and for Jmax the maximum J found in B_states_approx with 2 added.
    Then selects the part of the hamiltonian that corresponds to B_states_approx.
    The Hamiltonian is diagonal and the returned states are the states corresponding to
    B_states_approx in the basis of the Hamiltonian.

    Args:
        B_states_approx (Sequence[CoupledBasisState]): States
        E (npt.NDArray[np.float64], optional): Electric field in V/cm. Defaults to
                                                np.array([0.0, 0.0, 0.0]).
        B (npt.NDArray[np.float64], optional): Magnetic field in G. Defaults to
                                                np.array([0.0, 0.0, 1e-5]).
        rtol (Optional[float], optional): Remove components smaller than rtol in the
                                                        hamiltonian. Defaults to None.
        stol: (float): Remove superpositions with amplitude smaller than stol from each
                        state. Defaults to 1e-3.
        Jmin (Optional[int], optional): Minimum J to include in the Hamiltonian.
                                        Defaults to None.
        Jmax (Optional[int], optional): Maximum J to include in the Hamiltonian.
                                        Defaults to None.
        constants (BConstants, optional): B state constants. Defaults to BConstants().
        nuclear_spins (TlFNuclearSpins, optional): TlF nuclear spins. Defaults to
                                                    TlFNuclearSpins().
        H_func (Optional[Callable], optional): Function to generate the Hamiltonian
                                                depending on E and B. Defaults to None.

    Returns:
        ReducedHamiltonian: States and Hamiltonian
    """
    # need to generate the other states in case of mixing
    _Jmin = 1 if Jmin is None else Jmin
    _Jmax = max([gs.J for gs in B_states_approx]) + 2 if Jmax is None else Jmax

    omega_basis_flag = False

    # check basis
    if all([qn.basis == Basis.CoupledP for qn in B_states_approx]):
        qn_select = QuantumSelector(J=np.arange(_Jmin, _Jmax + 1), P=[-1, 1], Ω=1)
    elif all([qn.basis == Basis.CoupledΩ for qn in B_states_approx]):
        qn_select = QuantumSelector(J=np.arange(_Jmin, _Jmax + 1), P=None, Ω=[-1, 1])
        omega_basis_flag = True
    else:
        raise TypeError(
            "B_states_approx basis invalid, CoupledP or CoupledΩ are allowed, not "
            f"{B_states_approx[0].basis}"
        )
    QN_B = list(generate_coupled_states_B(qn_select, nuclear_spins=nuclear_spins))

    H_B: Optional[Hamiltonian] = None
    if H_func is None:
        H_B = generate_coupled_hamiltonian_B(QN_B, constants=constants)
        H_B_func = generate_coupled_hamiltonian_B_function(H_B)
    else:
        H_B_func = H_func

    H_diagonalized = generate_diagonalized_hamiltonian(
        H_B_func(E, B), keep_order=True, return_V_ref=True, rtol=rtol
    )

    # new set of quantum numbers:
    QN_B_diag = matrix_to_states(H_diagonalized.V, QN_B)

    if omega_basis_flag:
        warnings.warn(
            (
                "generate_reduced_B_hamiltonian called in Ω basis; mapping states to "
                "approximate states not implemented. Hamiltonian is not reduced."
            ),
            SyntaxWarning,
        )
        excited_states = [es.remove_small_components(stol) for es in QN_B_diag]
    else:
        excited_states = find_exact_states(
            [1 * e for e in B_states_approx], QN_B, QN_B_diag, V=H_diagonalized.V
        )
        excited_states = [es.remove_small_components(stol) for es in excited_states]

    H_B_red = reduced_basis_hamiltonian(
        [qn.remove_small_components(stol) for qn in QN_B_diag],
        H_diagonalized.H,
        excited_states,
    )

    return ReducedHamiltonian(
        H=H_B_red,
        V=H_diagonalized.V,
        QN_basis=excited_states,
        QN_construct=list(QN_B),
        hamiltonian=H_B,
    )


def compose_reduced_hamiltonian(
    H_X_red: npt.NDArray[np.complex128],
    H_B_red: npt.NDArray[np.complex128],
    element_limit: float = 0.1,
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Compose total reduced Hamiltonian from X and B state Hamiltonians.

    Creates a block diagonal Hamiltonian combining the X (ground) and B (excited)
    state Hamiltonians. Small matrix elements below element_limit are set to zero
    for numerical stability.

    Args:
        H_X_red: Reduced X state Hamiltonian matrix
        H_B_red: Reduced B state Hamiltonian matrix
        element_limit: Threshold below which matrix elements are set to zero.
            Defaults to 0.1.

    Returns:
        Tuple containing:
            - H_int: Block diagonal Hamiltonian [H_X_red, H_B_red]
            - V_ref_int: Identity matrix for state ordering reference
    """
    H_X_red[np.abs(H_X_red) < element_limit] = 0
    H_B_red[np.abs(H_B_red) < element_limit] = 0

    H_int: npt.NDArray[np.complex128] = linalg.block_diag(H_X_red, H_B_red)
    V_ref_int = np.eye(H_int.shape[0], dtype=np.complex128)

    return H_int, V_ref_int


@dataclass
class ReducedHamiltonianTotal:
    X_states: List[CoupledState]
    B_states: List[CoupledState]
    QN: List[CoupledState]
    H_int: npt.NDArray[np.complex128]
    V_ref_int: npt.NDArray[np.complex128]
    X_states_basis: List[CoupledBasisState]
    B_states_basis: List[CoupledBasisState]
    QN_basis: List[CoupledBasisState]
    X_hamiltonian: ReducedHamiltonian
    B_hamiltonian: ReducedHamiltonian


def generate_total_reduced_hamiltonian(
    X_states_approx: Sequence[CoupledBasisState],
    B_states_approx: Sequence[CoupledBasisState],
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-5]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    X_constants: XConstants = XConstants(),
    B_constants: BConstants = BConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    transform: Optional[npt.NDArray[np.complex128]] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
    use_omega_basis: bool = True,
) -> ReducedHamiltonianTotal:
    """
    Generate the total reduced hamiltonian for all X and B states in X_states_approx and
    B_states_approx, from an X state hamiltonian for all states from Jmin_X to Jmax_X
    and a B state Hamiltonian for all states from Jmin_B to Jmax_B.
    Returns the X_states, B_states, total states, Hamiltonian and V_reference_int which
    keeps track of the state ordering.

    Args:
        X_states_approx (Sequence[CoupledBasisState]): X_states_approx to generate the

        B_states_approx (Sequence[CoupledBasisState]): _description_
        E (npt.NDArray[np.float64], optional): Electric field. Defaults to
                                                np.array([0.0, 0.0, 0.0]).
        B (npt.NDArray[np.float64], optional): Magnetic field. Defaults to
                                                np.array([0.0, 0.0, 1e-5]). If smaller
                                                than 1e-5 some X states become
                                                degenerate again.
        rtol (Optional[float], optional): Tolerance for the Hamiltonian. Defaults to
                                            None.
        stol: (float): Remove superpositions with amplitude smaller than stol from each
                        state. Defaults to 1e-3.
        Jmin_X (Optional[int], optional): Jmin for the X state Hamiltonian. Defaults to
                                            None.
        Jmax_X (Optional[int], optional): Jmax for the X state Hamiltonian. Defaults to
                                            None.
        Jmin_B (Optional[int], optional): Jmin for the B state Hamiltonian. Defaults to
                                            None.
        Jmax_B (Optional[int], optional): Jmax for the B state Hamiltonian. Defaults to
                                            None.
        X_constants (XConstants, optional): X state constants. Defaults to XConstants().
        B_constants (BConstants, optional): B state constants. Defaults to BConstants().
        nuclear_spins (TlFNuclearSpins, optional): TlF nuclear spins. Defaults to
                                                    TlFNuclearSpins().
        transform (npt.NDArray[np.complex128], optional): transformation matrix to
                                                            transform the uncoupled X
                                                            state Hamiltonian to
                                                            coupled. Defaults to None.
        H_func_X (Optional[Callable], optional): Function to generate the X state
                                                    Hamiltonian for E and B. Defaults
                                                    to None.
        H_func_B (Optional[Callable], optional): Function to generate the B state
                                                    Hamiltonian for E and B. Defaults
                                                    to None.
        use_omega_basis (Optional[bool]): Use Ω basis to calculate the B state
                                            hamiltonian and then transform to P basis if
                                            the P state is requested. Ω basis is faster
                                            to calculate. Defaults to True.

    Returns:
        ReducedHamiltonian: Dataclass holding the X states, B states, total states,
                            Hamiltonian and reference eigenvectors
    """

    X_hamiltonian = generate_reduced_X_hamiltonian(
        X_states_approx,
        E=E,
        B=B,
        rtol=rtol,
        stol=stol,
        Jmin=Jmin_X,
        Jmax=Jmax_X,
        constants=X_constants,
        nuclear_spins=nuclear_spins,
        H_func=H_func_X,
        transform=transform,
    )
    ground_states, H_X_red = X_hamiltonian
    ground_states = cast(List[CoupledState], ground_states)
    H_X_red = cast(npt.NDArray[np.complex128], H_X_red)

    if use_omega_basis and B_states_approx[0].basis == Basis.CoupledP:
        _B_states_approx = cast(
            Sequence[CoupledBasisState],
            get_unique_basisstates_from_states(
                [qn.transform_to_omega_basis() for qn in B_states_approx]
            ),
        )
    else:
        _B_states_approx = cast(List[CoupledBasisState], list(B_states_approx))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H_B_red = generate_reduced_B_hamiltonian(
            _B_states_approx,
            E=E,
            B=B,
            rtol=rtol,
            stol=stol,
            Jmin=Jmin_B,
            Jmax=Jmax_B,
            constants=B_constants,
            nuclear_spins=nuclear_spins,
            H_func=H_func_B,
        )

    if use_omega_basis and B_states_approx[0].basis == Basis.CoupledP:
        excited_states = [
            qn.transform_to_parity_basis().remove_small_components(stol)
            for qn in H_B_red.QN_basis
        ]

        state_vecs = np.array(
            [(1 * s).state_vector(excited_states) for s in B_states_approx]
        )
        excited_states_reduced = [
            excited_states[idx] for idx in np.argmax(np.abs(state_vecs), axis=1)
        ]

        H_B_red_matrix = reduced_basis_hamiltonian(
            excited_states, H_B_red.H, excited_states_reduced
        )

        excited_states = excited_states_reduced

        H_int, V_ref_int = compose_reduced_hamiltonian(H_X_red, H_B_red_matrix)
    else:
        excited_states = H_B_red.QN_basis
        H_int, V_ref_int = compose_reduced_hamiltonian(H_X_red, H_B_red.H)

    QN = ground_states.copy()
    QN.extend(excited_states)
    return ReducedHamiltonianTotal(
        X_states=ground_states,
        B_states=excited_states,
        QN=QN,
        H_int=H_int,
        V_ref_int=V_ref_int,
        X_states_basis=list(X_states_approx),
        B_states_basis=list(B_states_approx),
        QN_basis=list(X_states_approx) + list(B_states_approx),
        X_hamiltonian=X_hamiltonian,
        B_hamiltonian=H_B_red,
    )


def generate_reduced_hamiltonian_transitions(
    transitions: Sequence[Union[OpticalTransition, MicrowaveTransition]],
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-5]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    Xconstants: XConstants = XConstants(),
    Bconstants: BConstants = BConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    minimum_amplitude: float = 5e-3,
    minimum_coupling: float = 1e-3,
    use_omega_basis: bool = True,
) -> ReducedHamiltonianTotal:
    """Generate reduced Hamiltonian automatically from transition definitions.

    Analyzes optical and microwave transitions to determine which rotational states
    are coupled and automatically constructs a reduced Hamiltonian containing only
    the relevant states. For optical transitions, calculates electric dipole matrix
    elements to identify strongly coupled states. For microwave transitions, includes
    the specified J levels.

    Args:
        transitions: Sequence of optical or microwave transitions defining the system
        E: Electric field in V/cm. Defaults to np.array([0.0, 0.0, 0.0]).
        B: Magnetic field in G. Defaults to np.array([0.0, 0.0, 1e-5]).
        rtol: Remove Hamiltonian components smaller than rtol * max(|H|). Defaults to
            None.
        stol: Remove state components with amplitude smaller than stol. Defaults to
            1e-3.
        Jmin_X: Minimum J for X state Hamiltonian. Defaults to None.
        Jmax_X: Maximum J for X state Hamiltonian. Defaults to None (determined
            automatically from transitions).
        Jmin_B: Minimum J for B state Hamiltonian. Defaults to None.
        Jmax_B: Maximum J for B state Hamiltonian. Defaults to None (determined
            automatically from transitions).
        Xconstants: X state molecular constants. Defaults to XConstants().
        Bconstants: B state molecular constants. Defaults to BConstants().
        nuclear_spins: TlF nuclear spin values. Defaults to TlFNuclearSpins().
        minimum_amplitude: Minimum amplitude to keep state components. Defaults to
            5e-3.
        minimum_coupling: Minimum electric dipole coupling strength to include ground
            states. Defaults to 1e-3.
        use_omega_basis: Use Ω basis for B state calculation (faster). Defaults to
            True.

    Returns:
        ReducedHamiltonianTotal: Complete reduced Hamiltonian with X and B states
    """
    _J_ground: List[int] = []
    excited_states_selectors = []

    # figure out which rotational levels to include
    for transition in transitions:
        if isinstance(transition, OpticalTransition):
            excited_states_approx_qn_select = transition.qn_select_excited
            excited_states_approx = list(
                generate_coupled_states_B(excited_states_approx_qn_select)
            )
            excited_states, excited_hamiltonian = generate_reduced_B_hamiltonian(
                B_states_approx=excited_states_approx,
                E=E,
                B=B,
                rtol=rtol,
                stol=stol,
                Jmin=Jmin_B,
                Jmax=Jmax_B,
                constants=Bconstants,
                nuclear_spins=nuclear_spins,
            )
            excited_states = cast(List[CoupledState], excited_states)

            # figure out which excited states are involved
            excited_states = [
                s.remove_small_components(minimum_amplitude) for s in excited_states
            ]

            Js_excited: npt.NDArray[np.int_] = np.unique(
                [s.J for es in excited_states for a, s in es]
            )
            if Jmax_X is None:
                _Jmax_X = Js_excited.max() + 2
            else:
                _Jmax_X = Jmax_X
            ground_states = generate_coupled_states_ground(
                Js=np.arange(0, _Jmax_X + 1).astype(int)
            )
            # calculate the coupling between the ground and excited states to determine
            # which states to include
            nonzero_coupling = []
            for gs in ground_states:
                for es in excited_states:
                    if (
                        np.abs(generate_ED_ME_mixed_state(1 * gs, es))
                        >= minimum_coupling
                    ):
                        nonzero_coupling.append(gs)

            Js_ground = list(np.unique([s.J for s in nonzero_coupling]))

            _J_ground.extend(Js_ground)
            excited_states_selectors.append(excited_states_approx_qn_select)

        if isinstance(transition, MicrowaveTransition):
            _J_ground.extend([transition.J_ground, transition.J_excited])

    # removing duplicates
    J_ground: List[int] = list(np.unique(_J_ground))

    ground_states_approx_qn_select = QuantumSelector(
        J=J_ground, electronic=ElectronicState.X, Ω=0
    )
    ground_states_approx = list(
        generate_coupled_states_X(ground_states_approx_qn_select)
    )
    excited_states_approx = list(generate_coupled_states_B(excited_states_selectors))

    return generate_total_reduced_hamiltonian(
        X_states_approx=ground_states_approx,
        B_states_approx=excited_states_approx,
        E=E,
        B=B,
        rtol=rtol,
        stol=stol,
        Jmin_X=Jmin_X,
        Jmax_X=Jmax_X,
        Jmin_B=Jmin_B,
        Jmax_B=Jmax_B,
        X_constants=Xconstants,
        B_constants=Bconstants,
        nuclear_spins=nuclear_spins,
        transform=None,
        H_func_X=None,
        H_func_B=None,
        use_omega_basis=use_omega_basis,
    )
