import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
import sympy as smp

from centrex_tlf import constants, hamiltonian, states
from centrex_tlf import couplings as couplings_tlf
from centrex_tlf.couplings.utils_compact import (
    compact_coupling_field,
    insert_levels_coupling_field,
)
from centrex_tlf.transitions import MicrowaveTransition, OpticalTransition

from . import utils_decay as utils_decay
from .generate_hamiltonian import generate_total_symbolic_hamiltonian
from .generate_system_of_equations import (
    generate_density_matrix,
    generate_dissipator_term,
    generate_system_of_equations_symbolic,
)
from .utils_compact import generate_qn_compact

__all__ = [
    "generate_OBE_system",
    "generate_OBE_system_transitions",
    "setup_OBE_system",
    "setup_OBE_system_transitions",
    "OBESystem",
]


@dataclass
class OBESystem:
    ground: Sequence[states.CoupledState]
    excited: Sequence[states.CoupledState]
    QN: Sequence[states.CoupledState]
    H_int: npt.NDArray[np.complex128]
    V_ref_int: npt.NDArray[np.complex128]
    couplings: List[Any]
    H_symbolic: smp.MutableDenseMatrix
    C_array: npt.NDArray[np.floating]
    system: smp.MutableDenseMatrix | None
    coupling_symbols: Sequence[smp.Symbol]
    polarization_symbols: Sequence[Sequence[smp.Symbol]]
    dissipator: Optional[smp.MutableDenseMatrix] = None
    QN_original: Optional[Sequence[states.CoupledState]] = None
    decay_channels: Optional[Sequence[utils_decay.DecayChannel]] = None
    couplings_original: Optional[List[couplings_tlf.CouplingFields]] = None

    def __repr__(self) -> str:
        ground = [s.largest for s in self.ground]
        ground = list(
            np.unique(
                [
                    f"|{s.electronic_state.name}, J = {s.J}, "  # type: ignore
                    f"P = {'+' if s.P == 1 else '-'}>"  # type: ignore
                    for s in ground
                ]
            )
        )
        ground_str: str = ", ".join(ground)  # type: ignore
        excited = [s.largest for s in self.excited]
        excited = list(
            np.unique(
                [
                    str(
                        f"|{s.electronic_state.name}, J = {s.J}, "  # type: ignore
                        f"F₁ = {smp.S(str(s.F1), rational=True)}, "  # type: ignore
                        f"F = {s.F}, "  # type: ignore
                        f"P = {'+' if s.P == 1 else '-'}>"  # type: ignore
                    )
                    for s in excited
                ]
            )
        )
        excited_str: str = ", ".join(excited)  # type: ignore
        return f"OBESystem(ground=[{ground_str}], excited=[{excited_str}])"


def _normalize_decay_channels(
    decay_channels: Optional[
        Union[Sequence[utils_decay.DecayChannel], utils_decay.DecayChannel]
    ],
) -> Optional[List[utils_decay.DecayChannel]]:
    if decay_channels is None:
        return None
    if isinstance(decay_channels, list):
        return decay_channels
    if not isinstance(decay_channels, (tuple, list, np.ndarray)):
        return [decay_channels]
    if isinstance(decay_channels, (tuple, np.ndarray)):
        return list(decay_channels)
    raise ValueError(
        f"decay_channels is type {type(decay_channels)}; supply a list, tuple"
        " or np.ndarray"
    )


def _generate_couplings(
    transition_selectors: Sequence[couplings_tlf.TransitionSelector],
    QN_basis: Sequence[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    QN: Sequence[states.CoupledState],
    V_ref_int: npt.NDArray[np.complex128],
    normalize_pol: bool,
) -> List[Any]:
    couplings = []
    for ts in transition_selectors:
        if ts.ground_main is not None and ts.excited_main is not None:
            couplings.append(
                couplings_tlf.generate_coupling_field(
                    ts.ground_main,
                    ts.excited_main,
                    ts.ground,
                    ts.excited,
                    QN_basis,
                    H_int,
                    QN,
                    V_ref_int,
                    pol_vecs=ts.polarizations,
                    pol_main=ts.polarizations[0],
                    normalize_pol=normalize_pol,
                )
            )
        else:
            couplings.append(
                couplings_tlf.generate_coupling_field_automatic(
                    ts.ground,
                    ts.excited,
                    QN_basis,
                    H_int,
                    QN,
                    V_ref_int,
                    pol_vecs=ts.polarizations,
                )
            )
    return couplings


def _build_obe_system(
    transition_selectors: Sequence[couplings_tlf.TransitionSelector],
    QN_basis: Sequence[states.CoupledState],
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    QN: Sequence[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    V_ref_int: npt.NDArray[np.complex128],
    qn_compact: Optional[
        Union[states.QuantumSelector, Sequence[states.QuantumSelector]]
    ],
    decay_channels: Optional[
        Union[Sequence[utils_decay.DecayChannel], utils_decay.DecayChannel]
    ],
    Γ: float,
    normalize_pol: bool,
    method: str,
    verbose: bool,
) -> OBESystem:
    logger = logging.getLogger(__name__)

    if verbose:
        logger.info(
            "generate_OBE_system: 2/5 -> "
            "Generating the couplings corresponding to the transitions"
        )
    couplings = _generate_couplings(
        transition_selectors, QN_basis, H_int, QN, V_ref_int, normalize_pol
    )

    if verbose:
        logger.info("generate_OBE_system: 3/5 -> Generating the symbolic Hamiltonian")
    if qn_compact is not None:
        H_symbolic, QN_compact = generate_total_symbolic_hamiltonian(
            QN, H_int, couplings, transition_selectors, qn_compact=qn_compact
        )
        couplings_compact = [
            compact_coupling_field(coupling, QN, qn_compact) for coupling in couplings
        ]
    else:
        H_symbolic = generate_total_symbolic_hamiltonian(
            QN, H_int, couplings, transition_selectors
        )
        QN_compact = None
        couplings_compact = None

    if verbose:
        logger.info("generate_OBE_system: 4/5 -> Generating the collapse matrices")
    C_array = couplings_tlf.collapse_matrices(
        QN, ground_states, excited_states, gamma=Γ, qn_compact=qn_compact
    )

    _decay_channels = _normalize_decay_channels(decay_channels)
    if _decay_channels is not None:
        indices = utils_decay.get_insert_level_indices(
            _decay_channels, QN, excited_states
        )
        couplings = [
            insert_levels_coupling_field(coupling, indices_insert=indices)
            for coupling in couplings
        ]
        QN = utils_decay.add_states_QN(_decay_channels, QN, indices)

        if (
            (qn_compact is not None)
            and (QN_compact is not None)
            and (couplings_compact is not None)
        ):
            indices, H_symbolic = utils_decay.add_levels_symbolic_hamiltonian(
                H_symbolic, _decay_channels, QN_compact, excited_states
            )
            QN_compact = utils_decay.add_states_QN(_decay_channels, QN_compact, indices)
            C_array = utils_decay.add_decays_C_arrays(
                _decay_channels, indices, QN_compact, C_array, Γ
            )
            couplings_compact = [
                insert_levels_coupling_field(coupling, indices_insert=indices)
                for coupling in couplings_compact
            ]
        else:
            indices, H_symbolic = utils_decay.add_levels_symbolic_hamiltonian(
                H_symbolic, _decay_channels, QN, excited_states
            )
            C_array = utils_decay.add_decays_C_arrays(
                _decay_channels, indices, QN, C_array, Γ
            )

    if verbose:
        logger.info(
            "generate_OBE_system: 5/5 -> Transforming the Hamiltonian and collapse "
            "matrices into a symbolic system of equations"
        )
    if method == "expanded":
        hamiltonian_term, dissipator = generate_system_of_equations_symbolic(
            H_symbolic, C_array, fast=True, split_output=True
        )
        system = hamiltonian_term + dissipator
    elif method == "matrix":
        system = None
        density_matrix = generate_density_matrix(H_symbolic.shape[0])
        dissipator = generate_dissipator_term(C_array, density_matrix, fast=True)
    else:
        raise ValueError(f"method {method} not recognised; use 'expanded' or 'matrix'")

    return OBESystem(
        QN=QN_compact if QN_compact is not None else QN,
        ground=ground_states,
        excited=excited_states,
        couplings=couplings if couplings_compact is None else couplings_compact,
        H_symbolic=H_symbolic,
        H_int=H_int,
        V_ref_int=V_ref_int,
        C_array=C_array,
        system=system,
        dissipator=dissipator,
        coupling_symbols=[trans.Ω for trans in transition_selectors],
        polarization_symbols=[
            trans.polarization_symbols for trans in transition_selectors
        ],
        QN_original=None if qn_compact is None else QN,
        decay_channels=_decay_channels,
        couplings_original=None if qn_compact is None else couplings,
    )


def check_transitions_allowed(
    transition_selectors: Sequence[couplings_tlf.TransitionSelector],
) -> None:
    """
    Check if a sequence of TransitionSelectors are all allowed transitions

    Args:
        transition_selectors (Sequence[couplings_TlF.TransitionSelector]): Sequence of
        TransitionSelectors a set of transitions.

    Raises:
        AssertionError: error if any given transition is not allowed.
    """
    for transition_selector in transition_selectors:
        if (
            transition_selector.ground_main is not None
            and transition_selector.excited_main is not None
        ):
            try:
                ΔmF_allowed = couplings_tlf.utils.ΔmF_allowed(
                    transition_selector.polarizations[0]
                )
                couplings_tlf.utils.assert_transition_coupled_allowed(
                    cast(
                        states.CoupledBasisState,
                        transition_selector.ground_main.largest,
                    ),
                    cast(
                        states.CoupledBasisState,
                        transition_selector.excited_main.largest,
                    ),
                    ΔmF_allowed=ΔmF_allowed,
                )
            except AssertionError as err:
                raise AssertionError(
                    f"{transition_selector.description} with polarization "
                    f"{np.round(transition_selector.polarizations[0], 2)} => "
                    f"{err.args[0]}"
                )
        else:
            raise ValueError(
                "Cannot check if transition is allowed; no main states selected"
            )
    return


def generate_OBE_system(
    X_states: Union[states.QuantumSelector, Sequence[states.QuantumSelector]],
    B_states: Union[states.QuantumSelector, Sequence[states.QuantumSelector]],
    transition_selectors: Sequence[couplings_tlf.TransitionSelector],
    qn_compact: Optional[
        Union[states.QuantumSelector, Sequence[states.QuantumSelector]]
    ] = None,
    decay_channels: Optional[
        Union[Sequence[utils_decay.DecayChannel], utils_decay.DecayChannel]
    ] = None,
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-5]),
    X_constants: constants.XConstants = constants.XConstants(),
    B_constants: constants.BConstants = constants.BConstants(),
    nuclear_spins: constants.TlFNuclearSpins = constants.TlFNuclearSpins(),
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    transform: Optional[npt.NDArray[np.complex128]] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
    verbose: bool = False,
    normalize_pol: bool = False,
    Γ: float = hamiltonian.Γ,
    method: str = "expanded",
) -> OBESystem:
    """Convenience function for generating the symbolic OBE system of equations.

    Args:

        X_states : X states to include in the OBE system
        B_states : B states to include in the OBE system
        transition_selectors (list): list of TransitionSelectors defining the
                                    transitions used in the OBE system.
        qn_compact (QuantumSelector): dataclass specifying a subset of states to
                                        select based on the quantum numbers
        decay_channels (DecayChannel): dataclass specifying the decay channel to
                                        add
        verbose (bool, optional): Log progress to INFO. Defaults to False.

    Returns:
        OBESystem: dataclass designed to hold the generated values
                    ground, exxcited, QN, H_int, V_ref_int, couplings, H_symbolic,
                    C_array, system
    """
    if method not in ["expanded", "matrix"]:
        raise ValueError(f"method {method} not recognised; use 'expanded' or 'matrix'")
    # check if transitions are allowed before generating the hamiltonian
    check_transitions_allowed(transition_selectors=transition_selectors)

    QN_X_original = list(states.generate_coupled_states_X(X_states))
    QN_B_original = list(states.generate_coupled_states_B(B_states))
    QN_original = QN_X_original + QN_B_original
    rtol = None
    stol = 1e-3
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info("generate_OBE_system: 1/5 -> Generating the reduced Hamiltonian")
    H_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        X_states_approx=QN_X_original,
        B_states_approx=QN_B_original,
        E=E,
        B=B,
        Jmin_X=Jmin_X,
        Jmax_X=Jmax_X,
        Jmin_B=Jmin_B,
        Jmax_B=Jmax_B,
        rtol=rtol,
        stol=stol,
        X_constants=X_constants,
        B_constants=B_constants,
        nuclear_spins=nuclear_spins,
        transform=transform,
        H_func_X=H_func_X,
        H_func_B=H_func_B,
    )
    ground_states = H_reduced.X_states
    excited_states = H_reduced.B_states
    QN = H_reduced.QN
    H_int = H_reduced.H_int
    V_ref_int = H_reduced.V_ref_int

    return _build_obe_system(
        transition_selectors=transition_selectors,
        QN_basis=QN_original,
        ground_states=ground_states,
        excited_states=excited_states,
        QN=QN,
        H_int=H_int,
        V_ref_int=V_ref_int,
        qn_compact=qn_compact,
        decay_channels=decay_channels,
        Γ=Γ,
        normalize_pol=normalize_pol,
        method=method,
        verbose=verbose,
    )


def generate_OBE_system_transitions(
    transitions: Sequence[Union[OpticalTransition, MicrowaveTransition]],
    transition_selectors: Sequence[couplings_tlf.TransitionSelector],
    qn_compact: Optional[
        Union[states.QuantumSelector, Sequence[states.QuantumSelector], bool]
    ] = None,
    decay_channels: Optional[
        Union[Sequence[utils_decay.DecayChannel], utils_decay.DecayChannel]
    ] = None,
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-5]),
    Γ: float = hamiltonian.Γ,
    X_constants: constants.XConstants = constants.XConstants(),
    B_constants: constants.BConstants = constants.BConstants(),
    nuclear_spins: constants.TlFNuclearSpins = constants.TlFNuclearSpins(),
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    transform: Optional[npt.NDArray[np.complex128]] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
    verbose: bool = False,
    normalize_pol: bool = False,
    method: str = "expanded",
) -> OBESystem:
    """Convenience function for generating the symbolic OBE system of equations.

    Args:
        transitions (list): list of TransitionSelectors defining the transitions
                            used in the OBE system.
        qn_compact (QuantumSelector): dataclass specifying a subset of states to
                                        select based on the quantum numbers
        decay_channels (DecayChannel): dataclass specifying the decay channel to
                                        add
        verbose (bool, optional): Log progress to INFO. Defaults to False.

    Returns:
        OBESystem: dataclass designed to hold the generated values
                    ground, excited, QN, H_int, V_ref_int, couplings, H_symbolic,
                    C_array, system
    """
    rtol = None
    stol = 1e-3

    # check if transitions are allowed before generating the hamiltonian
    check_transitions_allowed(transition_selectors=transition_selectors)

    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info("generate_OBE_system: 1/5 -> Generating the reduced Hamiltonian")
    H_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(
        transitions=transitions,
        E=E,
        B=B,
        rtol=rtol,
        stol=stol,
        Jmin_X=Jmin_X,
        Jmax_X=Jmax_X,
        Jmin_B=Jmin_B,
        Jmax_B=Jmax_B,
        Xconstants=X_constants,
        Bconstants=B_constants,
        nuclear_spins=nuclear_spins,
    )

    if H_reduced.QN_basis is None:
        raise TypeError("H_reduced.QN_basis is None")

    if qn_compact is True:
        _qn_compact = generate_qn_compact(transitions, H_reduced)
    elif qn_compact is False:
        _qn_compact = None
    else:
        _qn_compact = qn_compact

    ground_states = H_reduced.X_states
    excited_states = H_reduced.B_states
    QN = H_reduced.QN
    H_int = H_reduced.H_int
    V_ref_int = H_reduced.V_ref_int

    return _build_obe_system(
        transition_selectors=transition_selectors,
        QN_basis=H_reduced.QN_basis,
        ground_states=ground_states,
        excited_states=excited_states,
        QN=QN,
        H_int=H_int,
        V_ref_int=V_ref_int,
        qn_compact=_qn_compact,
        decay_channels=decay_channels,
        Γ=Γ,
        normalize_pol=normalize_pol,
        method=method,
        verbose=verbose,
    )


def setup_OBE_system(
    X_states: Union[states.QuantumSelector, Sequence[states.QuantumSelector]],
    B_states: Union[states.QuantumSelector, Sequence[states.QuantumSelector]],
    transitions: Sequence[couplings_tlf.TransitionSelector],
    qn_compact: Optional[
        Union[Sequence[states.QuantumSelector], states.QuantumSelector]
    ] = None,
    decay_channels: Optional[
        Union[Sequence[utils_decay.DecayChannel], utils_decay.DecayChannel]
    ] = None,
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-5]),
    X_constants: constants.XConstants = constants.XConstants(),
    B_constants: constants.BConstants = constants.BConstants(),
    nuclear_spins: constants.TlFNuclearSpins = constants.TlFNuclearSpins(),
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    transform: Optional[npt.NDArray[np.complex128]] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
    verbose: bool = False,
    normalize_pol: bool = False,
    method: str = "expanded",
):
    """Convenience function for generating the OBE system

    Args:
        X_states : X states to include in the OBE system
        B_states : B states to include in the OBE system
        ode_parameters (odeParameters): dataclass containing the ode parameters.
                                        e.g. Ω, δ, vz, ..., etc.
        transitions (TransitionSelector): object containing all information
                                            required to generate the coupling
                                            matrices and symbolic matrix for
                                            each transition
        qn_compact (QuantumSelector): dataclass specifying a subset of states to
                                        select based on the quantum numbers
        full_output (bool, optional): Returns all matrices, states etc. if True,
                                        Returns only QN if False.
                                        Defaults to False.
        decay_channels (DecayChannel): dataclass specifying the decay channel to
                                        add
        verbose (bool, optional): Log progress to INFO. Defaults to False.

    Returns:
        full_output == True:
            list: list of states in system
        full_output == False:
            OBESystem: dataclass designed to hold the generated values
                        ground, exxcited, QN, H_int, V_ref_int, couplings,
                        H_symbolic, C_array, system
    """
    obe_system = generate_OBE_system(
        X_states,
        B_states,
        transitions,
        qn_compact=qn_compact,
        decay_channels=decay_channels,
        E=E,
        B=B,
        X_constants=X_constants,
        B_constants=B_constants,
        nuclear_spins=nuclear_spins,
        Jmin_X=Jmin_X,
        Jmax_X=Jmax_X,
        Jmin_B=Jmin_B,
        Jmax_B=Jmax_B,
        transform=transform,
        H_func_X=H_func_X,
        H_func_B=H_func_B,
        verbose=verbose,
        normalize_pol=normalize_pol,
        method=method,
    )
    return obe_system


def setup_OBE_system_transitions(
    transitions: Sequence[Union[OpticalTransition, MicrowaveTransition]],
    transition_selectors: Sequence[couplings_tlf.TransitionSelector],
    qn_compact: Optional[
        Union[Sequence[states.QuantumSelector], states.QuantumSelector, bool]
    ] = None,
    decay_channels: Optional[
        Union[Sequence[utils_decay.DecayChannel], utils_decay.DecayChannel]
    ] = None,
    E: npt.NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.floating] = np.array([0.0, 0.0, 1e-5]),
    X_constants: constants.XConstants = constants.XConstants(),
    B_constants: constants.BConstants = constants.BConstants(),
    nuclear_spins: constants.TlFNuclearSpins = constants.TlFNuclearSpins(),
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    transform: Optional[npt.NDArray[np.complex128]] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
    verbose: bool = False,
    normalize_pol: bool = False,
    Γ: float = hamiltonian.Γ,
    method: str = "expanded",
) -> OBESystem:
    """Convenience function for generating the OBE system

    Args:
        ode_parameters (odeParameters): dataclass containing the ode parameters.
                                        e.g. Ω, δ, vz, ..., etc.
        transitions (Sequence[TransitionSelector]): Sequence containing all transition
                                            information required to generate
                                            the coupling matrices and symbolic matrix
                                            for each transition
        qn_compact (QuantumSelector): dataclass specifying a subset of states to
                                        select based on the quantum numbers
        full_output (bool, optional): Returns all matrices, states etc. if True,
                                        Returns only QN if False.
                                        Defaults to False.
        decay_channels (DecayChannel): dataclass specifying the decay channel to
                                        add
        verbose (bool, optional): Log progress to INFO. Defaults to False.

    Returns:
        full_output == True:
            list: list of states in system
        full_output == False:
            OBESystem: dataclass designed to hold the generated values
                        ground, exxcited, QN, H_int, V_ref_int, couplings,
                        H_symbolic, C_array, system
    """
    obe_system = generate_OBE_system_transitions(
        transitions=transitions,
        transition_selectors=transition_selectors,
        qn_compact=qn_compact,
        decay_channels=decay_channels,
        E=E,
        B=B,
        Γ=Γ,
        X_constants=X_constants,
        B_constants=B_constants,
        nuclear_spins=nuclear_spins,
        Jmin_X=Jmin_X,
        Jmax_X=Jmax_X,
        Jmin_B=Jmin_B,
        Jmax_B=Jmax_B,
        transform=transform,
        H_func_X=H_func_X,
        H_func_B=H_func_B,
        verbose=verbose,
        normalize_pol=normalize_pol,
        method=method,
    )

    return obe_system
