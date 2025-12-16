"""Optical coupling matrix generation for laser-driven transitions.

This module provides tools for computing coupling matrices that describe how laser
fields couple quantum states. It handles multiple polarizations, automatic state
selection based on coupling strength, and generation of coupling field objects for
use in optical Bloch equations.
"""

from dataclasses import dataclass
from typing import Sequence, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from centrex_tlf import hamiltonian, states
from centrex_tlf.states.states import CoupledBasisState

from .utils import ΔmF_allowed, assert_transition_coupled_allowed, select_main_states

try:
    from ..centrex_tlf_rust import (
        generate_coupling_matrix_py as _generate_coupling_matrix_rust,
    )

    HAS_RUST = True
except ImportError:
    _generate_coupling_matrix_rust = None  # type: ignore[assignment]
    HAS_RUST = False

__all__ = [
    "generate_coupling_matrix",
    "generate_coupling_field",
    "generate_coupling_field_automatic",
    "CouplingFields",
    "CouplingField",
    "generate_coupling_dataframe",
]


def _generate_coupling_matrix_python(
    QN: Sequence[states.CoupledState],
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    pol_vec: npt.NDArray[np.complex128] | None = None,
    reduced: bool = False,
    normalize_pol: bool = True,
) -> npt.NDArray[np.complex128]:
    """Generate optical coupling matrix for transitions between quantum states.

    Constructs a Hermitian coupling matrix H where H[i,j] represents the electric dipole
    coupling strength between states i and j. Only non-zero between ground and excited
    state pairs.

    Args:
        QN (Sequence[CoupledState]): Complete list of basis states defining the Hilbert
            space
        ground_states (Sequence[CoupledState]): Ground states that couple to excited
            states
        excited_states (Sequence[CoupledState]): Excited states that couple to ground
            states
        pol_vec (npt.NDArray[np.complex128] | None): Polarization vector [Ex, Ey, Ez]
            in Cartesian basis. Defaults to None, which uses [0, 0, 1] (σ polarization).
        reduced (bool): If True, return only reduced matrix elements (no angular part).
            Defaults to False.
        normalize_pol (bool): If True, normalize the polarization vector. Defaults to
            True.

    Returns:
        npt.NDArray[np.complex128]: Hermitian coupling matrix of shape (n, n) where
            n = len(QN)

    Raises:
        AssertionError: If QN is not a list

    Example:
        >>> H_coupling = generate_coupling_matrix(QN, ground_states, excited_states)
        >>> coupling_strength = np.abs(H_coupling[ground_idx, excited_idx])
    """
    assert isinstance(QN, list), "QN required to be of type list"

    # Initialize default polarization vector if not provided
    if pol_vec is None:
        pol_vec = np.array([0.0, 0.0, 1.0], dtype=np.complex128)

    if normalize_pol:
        pol_vec = pol_vec / np.linalg.norm(pol_vec)

    H = np.zeros((len(QN), len(QN)), dtype=complex)

    idx_mapping_ground = dict(
        [(idg, QN.index(gs)) for idg, gs in enumerate(ground_states)]
    )
    idx_mapping_excited = dict(
        [(ide, QN.index(es)) for ide, es in enumerate(excited_states)]
    )

    # start looping over ground and excited states
    for idg, ground_state in enumerate(ground_states):
        i = idx_mapping_ground[idg]
        for ide, excited_state in enumerate(excited_states):
            j = idx_mapping_excited[ide]

            # calculate matrix element and add it to the Hamiltonian
            H[i, j] = hamiltonian.generate_ED_ME_mixed_state(
                excited_state,
                ground_state,
                pol_vec=pol_vec,
                reduced=reduced,
                normalize_pol=False,
            )
            # # make H hermitian
            if H[i, j] != 0:
                H[j, i] = np.conj(H[i, j])

    return H


def generate_coupling_matrix(
    QN: Sequence[states.CoupledState],
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    pol_vec: npt.NDArray[np.complex128] | None = None,
    reduced: bool = False,
    normalize_pol: bool = True,
) -> npt.NDArray[np.complex128]:
    """Generate optical coupling matrix for transitions between quantum states.

    Constructs a Hermitian coupling matrix H where H[i,j] represents the electric dipole
    coupling strength between states i and j. Only non-zero between ground and excited
    state pairs.

    Args:
        QN (Sequence[CoupledState]): Complete list of basis states defining the Hilbert
            space
        ground_states (Sequence[CoupledState]): Ground states that couple to excited
            states
        excited_states (Sequence[CoupledState]): Excited states that couple to ground
            states
        pol_vec (npt.NDArray[np.complex128] | None): Polarization vector [Ex, Ey, Ez]
            in Cartesian basis. Defaults to None, which uses [0, 0, 1] (σ polarization).
        reduced (bool): If True, return only reduced matrix elements (no angular part).
            Defaults to False.
        normalize_pol (bool): If True, normalize the polarization vector. Defaults to
            True.

    Returns:
        npt.NDArray[np.complex128]: Hermitian coupling matrix of shape (n, n) where
            n = len(QN)

    Raises:
        AssertionError: If QN is not a list

    Example:
        >>> H_coupling = generate_coupling_matrix(QN, ground_states, excited_states)
        >>> coupling_strength = np.abs(H_coupling[ground_idx, excited_idx])
    """
    # Initialize default polarization vector if not provided
    if pol_vec is None:
        pol_vec = np.array([0.0, 0.0, 1.0], dtype=np.complex128)

    if normalize_pol:
        pol_vec = pol_vec / np.linalg.norm(pol_vec)

    if excited_states[0].largest.basis is states.Basis.CoupledP:
        QN = [
            qn.transform_to_omega_basis()
            if qn.largest.basis is states.Basis.CoupledP
            else qn
            for qn in QN
        ]
        excited_states = [qn.transform_to_omega_basis() for qn in excited_states]

    if HAS_RUST and _generate_coupling_matrix_rust is not None:
        return _generate_coupling_matrix_rust(
            QN,
            ground_states,
            excited_states,
            pol_vec,
            reduced,
        )
    else:
        return _generate_coupling_matrix_python(
            QN,
            ground_states,
            excited_states,
            pol_vec,
            reduced,
            normalize_pol=False,
        )


@dataclass
class CouplingField:
    """Represents an optical coupling field for a specific polarization.

    Attributes:
        polarization (npt.NDArray[np.complex128]): Polarization vector [Ex, Ey, Ez]
        field (npt.NDArray[np.complex128]): Coupling matrix for this polarization
    """

    polarization: npt.NDArray[np.complex128]
    field: npt.NDArray[np.complex128]


@dataclass
class CouplingFields:
    """Collection of coupling fields for a transition with multiple polarizations.

    Attributes:
        ground_main (CoupledState): Main ground state for the transition
        excited_main (CoupledState): Main excited state for the transition
        main_coupling (complex): Coupling strength of the main transition
        ground_states (Sequence[CoupledState]): All ground states with significant
            coupling
        excited_states (Sequence[CoupledState]): All excited states with significant
            coupling
        fields (Sequence[CouplingField]): Coupling matrices for each polarization
    """

    ground_main: states.CoupledState
    excited_main: states.CoupledState
    main_coupling: complex
    ground_states: Sequence[states.CoupledState]
    excited_states: Sequence[states.CoupledState]
    fields: Sequence[CouplingField]

    def __repr__(self):
        gs = self.ground_main.largest
        es = self.excited_main.largest
        gs_str = gs.state_string_custom(["electronic", "J", "F1", "F", "mF", "P", "Ω"])
        es_str = es.state_string_custom(["electronic", "J", "F1", "F", "mF", "P", "Ω"])
        return (
            f"CouplingFields(ground_main={gs_str},"
            f" excited_main={es_str},"
            f" main_coupling={self.main_coupling:.2e}"
        )


def _generate_coupling_dataframe(
    field: CouplingField, states_list: Sequence[states.CoupledState]
) -> pd.DataFrame:
    indices = np.nonzero(np.triu(field.field))
    ground_states = []
    excited_states = []
    couplings = []
    for idx, idy in zip(*indices):
        gs = states_list[idx].largest.state_string_custom(
            ["electronic", "J", "F1", "F", "mF"]
        )
        es = states_list[idy].largest.state_string_custom(
            ["electronic", "J", "F1", "F", "mF"]
        )
        ground_states.append(gs)
        excited_states.append(es)
        couplings.append(field.field[idx, idy])

    data = {"ground": ground_states, "excited": excited_states, "couplings": couplings}
    return pd.DataFrame(data)


def generate_coupling_dataframe(
    fields: CouplingFields, states_list: Sequence[states.CoupledState]
) -> Sequence[pd.DataFrame]:
    """
    Generate a list of pandas DataFrames with the non-zero couplings between states
    listed for each separate CouplingField input

    Args:
        fields (CouplingFields): coupling fields for a given transitions, with one for
        each polarization
        states_list (Sequence[states.State]): states involved in the system

    Returns:
        Sequence[pd.DataFrame]: list of DataFrames with non-zero couplings
    """
    dfs = []
    for field in fields.fields:
        dfs.append(_generate_coupling_dataframe(field, states_list))
    return dfs


def generate_coupling_field(
    ground_main_approx: states.CoupledState,
    excited_main_approx: states.CoupledState,
    ground_states_approx: Union[
        Sequence[states.CoupledState], Sequence[states.CoupledBasisState]
    ],
    excited_states_approx: Union[
        Sequence[states.CoupledState], Sequence[states.CoupledBasisState]
    ],
    QN_basis: Union[Sequence[states.CoupledState], Sequence[states.CoupledBasisState]],
    H_rot: npt.NDArray[np.complex128],
    QN: Sequence[states.CoupledState],
    V_ref: npt.NDArray[np.complex128],
    pol_main: npt.NDArray[np.complex128] | None = None,
    pol_vecs: Sequence[npt.NDArray[np.complex128]] | None = None,
    relative_coupling: float = 1e-3,
    absolute_coupling: float = 1e-6,
    normalize_pol: bool = True,
) -> CouplingFields:
    """Generate coupling fields for optical transitions with multiple polarizations.

    Creates CouplingField objects for each polarization that describe the coupling
    between ground and excited states. Automatically determines which states are
    significantly coupled based on relative and absolute thresholds.

    Args:
        ground_main_approx (CoupledState): Main ground state for the transition
        excited_main_approx (CoupledState): Main excited state for the transition
        ground_states_approx (Sequence[CoupledState] | Sequence[CoupledBasisState]):
            Approximate ground states involved in coupling
        excited_states_approx (Sequence[CoupledState] | Sequence[CoupledBasisState]):
            Approximate excited states involved in coupling
        QN_basis (Sequence[CoupledState] | Sequence[CoupledBasisState]): Basis states
            used for Hamiltonian construction
        H_rot (npt.NDArray[np.complex128]): Rotational Hamiltonian matrix
        QN (Sequence[CoupledState]): Complete quantum number basis
        V_ref (npt.NDArray[np.complex128]): Reference eigenvector matrix
        pol_main (npt.NDArray[np.complex128] | None): Main polarization vector
            [Ex, Ey, Ez]. Defaults to None, which uses [0, 0, 1].
        pol_vecs (Sequence[npt.NDArray[np.complex128]] | None): Additional polarization
            vectors to include. Defaults to None (empty list).
        relative_coupling (float): Threshold for coupling relative to main coupling.
            States with |coupling/main_coupling| < relative_coupling are excluded.
            Defaults to 1e-3.
        absolute_coupling (float): Absolute threshold for coupling strength. States
            with |coupling| < absolute_coupling are excluded. Defaults to 1e-6.
        normalize_pol (bool): If True, normalize polarization vectors. Defaults to True.

    Returns:
        CouplingFields: Dataclass containing ground/excited states, couplings for each
            polarization, and the main coupling strength

    Raises:
        AssertionError: If pol_main or pol_vecs are not numpy arrays with correct dtype
    """
    # Initialize default values
    if pol_main is None:
        pol_main = np.array([0, 0, 1], dtype=np.complex128)
    if pol_vecs is None:
        pol_vecs = []

    assert isinstance(pol_main, np.ndarray), (
        "supply a numpy ndarray with dtype np.complex128 for pol_main"
    )
    if len(pol_vecs) > 0:
        assert isinstance(pol_vecs[0], np.ndarray), (
            "supply a Sequence of np.ndarrays with dtype np.complex128 for pol_vecs"
        )
    if not np.issubdtype(pol_main.dtype, np.complex128):
        pol_main.astype(np.complex128)
    if not np.issubdtype(pol_vecs[0].dtype, np.complex128):
        pol_vecs = [pol.astype(np.complex128) for pol in pol_vecs]

    _ground_states_approx: Sequence[states.CoupledState]
    _excited_states_approx: Sequence[states.CoupledState]
    _QN_basis: Sequence[states.CoupledState]

    if isinstance(ground_states_approx[0], CoupledBasisState):
        ground_states_approx = cast(Sequence[CoupledBasisState], ground_states_approx)
        _ground_states_approx = states.states.basisstate_to_state_list(
            ground_states_approx
        )
    else:
        _ground_states_approx = cast(
            Sequence[states.CoupledState], ground_states_approx
        )

    if isinstance(excited_states_approx[0], CoupledBasisState):
        excited_states_approx = cast(Sequence[CoupledBasisState], excited_states_approx)
        _excited_states_approx = states.states.basisstate_to_state_list(
            excited_states_approx
        )
    else:
        _excited_states_approx = cast(
            Sequence[states.CoupledState], excited_states_approx
        )

    if isinstance(QN_basis[0], CoupledBasisState):
        QN_basis = cast(Sequence[CoupledBasisState], QN_basis)
        _QN_basis = states.states.basisstate_to_state_list(QN_basis)
    else:
        _QN_basis = cast(Sequence[states.CoupledState], QN_basis)

    ground_states = states.find_exact_states(
        _ground_states_approx, _QN_basis, QN, H_rot, V_ref=V_ref
    )
    excited_states = states.find_exact_states(
        _excited_states_approx, _QN_basis, QN, H_rot, V_ref=V_ref
    )
    ground_main = states.find_exact_states(
        [ground_main_approx], _QN_basis, QN, H_rot, V_ref=V_ref
    )[0]
    excited_main = states.find_exact_states(
        [excited_main_approx], _QN_basis, QN, H_rot, V_ref=V_ref
    )[0]

    states.check_approx_state_exact_state(ground_main_approx, ground_main)
    states.check_approx_state_exact_state(excited_main_approx, excited_main)
    ME_main = hamiltonian.generate_ED_ME_mixed_state(
        excited_main,
        ground_main,
        pol_vec=np.asarray(pol_main, dtype=np.complex128),
        normalize_pol=normalize_pol,
    )

    assert ME_main != 0, (
        f"main coupling element for {ground_main_approx} -> "
        f"{excited_main_approx} is zero, pol = {pol_main}"
    )

    _ground_main = cast(CoupledBasisState, ground_main.largest)
    _excited_main = cast(CoupledBasisState, excited_main.largest)

    ΔmF_raw = ΔmF_allowed(pol_main)
    assert_transition_coupled_allowed(_ground_main, _excited_main, ΔmF_raw)

    couplings = []
    for pol in pol_vecs:
        coupling = generate_coupling_matrix(
            QN,
            ground_states,
            excited_states,
            pol_vec=pol,
            reduced=False,
            normalize_pol=normalize_pol,
        )
        if normalize_pol:
            pol = pol.copy() / np.linalg.norm(pol)

        coupling[np.abs(coupling) < relative_coupling * np.max(np.abs(coupling))] = 0
        coupling[np.abs(coupling) < absolute_coupling] = 0
        couplings.append(CouplingField(polarization=pol, field=coupling))
    return CouplingFields(
        ground_main, excited_main, ME_main, ground_states, excited_states, couplings
    )


def generate_coupling_field_automatic(
    ground_states_approx: Union[
        Sequence[states.CoupledState],
        Sequence[states.CoupledBasisState],
        Sequence[states.UncoupledBasisState],
    ],
    excited_states_approx: Union[
        Sequence[states.CoupledState],
        Sequence[states.CoupledBasisState],
        Sequence[states.UncoupledBasisState],
    ],
    QN_basis: Union[
        Sequence[states.CoupledState],
        Sequence[states.CoupledBasisState],
        Sequence[states.UncoupledBasisState],
    ],
    H_rot: npt.NDArray[np.complex128],
    QN: Sequence[states.CoupledState],
    V_ref: npt.NDArray[np.complex128],
    pol_vecs: Sequence[npt.NDArray[np.complex128]],
    relative_coupling: float = 1e-3,
    absolute_coupling: float = 1e-6,
    normalize_pol: bool = True,
) -> CouplingFields:
    """Calculate the coupling fields for a transition for one or multiple
    polarizations.

    Args:
        ground_states_approx (list): list of approximate ground states
        excited_states_approx (list): list of approximate excited states
        QN_basis (Sequence[states.State]): Sequence of States the H_rot was constructed
                                            from
        H_rot (np.ndarray): System hamiltonian in the rotational frame
        QN (list): list of states in the system
        V_ref ([type]): [description]
        pol_vec (list): list of polarizations.
        relative_coupling (float): minimum relative coupling, set
                                            smaller coupling to zero.
                                            Defaults to 1e-3.
        absolute_coupling (float): minimum absolute coupling, set
                                            smaller couplings to zero.
                                            Defaults to 1e-6.

    Returns:
        dictionary: CouplingFields dataclass with the coupling information.
                    Attributes:
                        ground_main: main ground state
                        excited_main: main excited state
                        main_coupling: coupling strength between main_ground
                                        and main_excited
                        ground_states: ground states in coupling
                        excited_states: excited_states in coupling
                        fields: list of CouplingField dataclasses, one for each
                                polarization, containing the polarization and coupling
                                field
    """
    assert isinstance(pol_vecs[0], np.ndarray), (
        "supply a Sequence of np.ndarrays with dtype np.floating for pol_vecs"
    )

    _ground_states_approx: Sequence[states.CoupledState]
    _excited_states_approx: Sequence[states.CoupledState]
    _QN_basis: Sequence[states.CoupledState]

    if isinstance(ground_states_approx[0], CoupledBasisState):
        ground_states_approx = cast(Sequence[CoupledBasisState], ground_states_approx)
        _ground_states_approx = states.states.basisstate_to_state_list(
            ground_states_approx
        )
    else:
        _ground_states_approx = cast(
            Sequence[states.CoupledState], ground_states_approx
        )

    if isinstance(excited_states_approx[0], CoupledBasisState):
        excited_states_approx = cast(Sequence[CoupledBasisState], excited_states_approx)
        _excited_states_approx = states.states.basisstate_to_state_list(
            excited_states_approx
        )
    else:
        _excited_states_approx = cast(
            Sequence[states.CoupledState], excited_states_approx
        )

    if isinstance(QN_basis[0], CoupledBasisState):
        QN_basis = cast(Sequence[CoupledBasisState], QN_basis)
        _QN_basis = states.states.basisstate_to_state_list(QN_basis)
    else:
        _QN_basis = cast(Sequence[states.CoupledState], QN_basis)

    pol_main = pol_vecs[0]
    ground_main_approx, excited_main_approx = select_main_states(
        _ground_states_approx, _excited_states_approx, pol_main
    )
    return generate_coupling_field(
        ground_main_approx=ground_main_approx,
        excited_main_approx=excited_main_approx,
        ground_states_approx=_ground_states_approx,
        excited_states_approx=_excited_states_approx,
        QN_basis=_QN_basis,
        H_rot=H_rot,
        QN=QN,
        V_ref=V_ref,
        pol_main=pol_main,
        pol_vecs=pol_vecs,
        relative_coupling=relative_coupling,
        absolute_coupling=absolute_coupling,
        normalize_pol=normalize_pol,
    )
