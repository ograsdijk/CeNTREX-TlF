"""Optical coupling matrix generation for laser-driven transitions.

This module provides tools for computing coupling matrices that describe how laser
fields couple quantum states. It handles multiple polarizations, automatic state
selection based on coupling strength, and generation of coupling field objects for
use in optical Bloch equations.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from centrex_tlf import hamiltonian, states
from centrex_tlf.states.states import CoupledBasisState

from .utils import ΔmF_allowed, check_transition_coupled_allowed_polarization

try:
    from ..centrex_tlf_rust import (
        generate_coupling_matrix_py as _generate_coupling_matrix_rust,
    )

    HAS_RUST = True
except ImportError:
    _generate_coupling_matrix_rust = None  # type: ignore[assignment]
    HAS_RUST = False

# Tolerance for deciding that a main coupling matrix element is *zero*, i.e. that the
# transition is driven neither directly nor through field mixing. Deliberately far below
# `absolute_coupling`, which is a matrix *pruning* threshold: a genuinely weak but real
# field-mixed coupling must stay usable, and is flagged by `weak_main_fraction` instead.
MAIN_COUPLING_ZERO_TOL = 1e-12

__all__ = [
    "generate_coupling_matrix",
    "generate_coupling_field",
    "generate_coupling_field_automatic",
    "select_main_states_indices_coupling",
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
    if not isinstance(QN, list):
        raise TypeError("QN required to be of type list")

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


def _generate_coupling_matrix_python_with_indices(
    QN: Sequence[states.CoupledState],
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    ground_indices: Sequence[int],
    excited_indices: Sequence[int],
    pol_vec: npt.NDArray[np.complex128],
    reduced: bool = False,
) -> npt.NDArray[np.complex128]:
    H = np.zeros((len(QN), len(QN)), dtype=complex)
    for i, ground_state in zip(ground_indices, ground_states):
        for j, excited_state in zip(excited_indices, excited_states):
            H[i, j] = hamiltonian.generate_ED_ME_mixed_state(
                excited_state,
                ground_state,
                pol_vec=pol_vec,
                reduced=reduced,
                normalize_pol=False,
            )
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

    ground_indices: list[int] | None = None
    excited_indices: list[int] | None = None
    if excited_states[0].largest.basis is states.Basis.CoupledP:
        # Preserve the original state indices before transforming to Ω basis. Opposite
        # parity partners can have the same largest Ω-basis component, so an index map
        # built after the transform may collapse distinct retained levels.
        QN_original = list(QN)
        original_index = {id(state): idx for idx, state in enumerate(QN_original)}

        def _original_index(state: states.CoupledState) -> int:
            idx = original_index.get(id(state))
            if idx is None:
                # Fall back to equality search for states not identical by id.
                idx = QN_original.index(state)
            return idx

        ground_indices = [_original_index(gs) for gs in ground_states]
        excited_indices = [_original_index(es) for es in excited_states]
        QN = [
            qn.transform_to_omega_basis()
            if qn.largest.basis is states.Basis.CoupledP
            else qn
            for qn in QN
        ]
        excited_states = [qn.transform_to_omega_basis() for qn in excited_states]

    if HAS_RUST and _generate_coupling_matrix_rust is not None:
        if ground_indices is None or excited_indices is None:
            idx_map = {s.largest: i for i, s in enumerate(QN)}
            ground_indices = [idx_map[gs.largest] for gs in ground_states]
            excited_indices = [idx_map[es.largest] for es in excited_states]
        return _generate_coupling_matrix_rust(
            QN,
            ground_indices,
            excited_indices,
            pol_vec,
            reduced,
        )
    elif ground_indices is not None and excited_indices is not None:
        return _generate_coupling_matrix_python_with_indices(
            QN,
            ground_states,
            excited_states,
            ground_indices,
            excited_indices,
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


def _dress_states(
    states_approx: Sequence[states.CoupledState],
    QN_basis: Sequence[states.CoupledState],
    QN: Sequence[states.CoupledState],
    H_rot: npt.NDArray[np.complex128],
    V_ref: npt.NDArray[np.complex128],
) -> Sequence[states.CoupledState]:
    """Map approximate (bare) states onto the field-dressed eigenstates of H_rot.

    Thin wrapper around :func:`states.find_exact_states` so that
    :func:`generate_coupling_field` and :func:`generate_coupling_field_automatic` share a
    single implementation and the (non-free) assignment is not performed twice.
    """
    return states.find_exact_states(states_approx, QN_basis, QN, H_rot, V_ref=V_ref)


def _explain_zero_main_coupling(
    ground_main: states.CoupledState,
    excited_main: states.CoupledState,
    pol_main: npt.NDArray[np.complex128],
) -> str:
    """Explain why a main coupling matrix element came out zero.

    The E1 selection rules are applied to the *bare* dominant component of each state, so
    they cannot decide whether a field-mixed transition is allowed. They are still the
    best available explanation when the mixed-state matrix element is zero, which is
    exactly the situation where the bare labels are informative.
    """
    try:
        allowed, reason = cast(
            Tuple[bool, str],
            check_transition_coupled_allowed_polarization(
                cast(CoupledBasisState, ground_main.largest),
                cast(CoupledBasisState, excited_main.largest),
                ΔmF_allowed(pol_main),
                return_err=True,
            ),
        )
    except Exception as err:  # noqa: BLE001 - never let a diagnostic mask the real error
        # e.g. an UncoupledBasisState has no F/mF, raising AttributeError here.
        return f"selection rules could not be evaluated ({err})"

    if not allowed:
        return (
            f"{reason}; this holds for the dominant basis component and no field mixing "
            f"lifts it here"
        )
    return (
        "the E1 selection rules are satisfied by the dominant basis components, so this "
        "is most likely a state-identification problem rather than a forbidden "
        "transition"
    )


def select_main_states_indices_coupling(
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    polarization: npt.NDArray[np.complex128],
    absolute_coupling: float = 1e-6,
    normalize_pol: bool = True,
) -> Tuple[int, int]:
    """Pick the main ground/excited pair, falling back to field-mixed couplings.

    The main pair is the Rabi normalization reference: the whole coupling matrix is
    divided by its matrix element. It should therefore be a *strongly* coupled pair
    whenever one exists.

    A bare-allowed pair is always preferred, chosen with the historical heuristic of
    :func:`~centrex_tlf.couplings.utils.select_main_states` (prefer an mF = 0 ground
    state, last match in scan order), so the selected pair is unchanged at any field.
    Only when no bare-allowed pair has a non-vanishing matrix element — the case that
    used to raise outright — does this fall back to the mixed-state matrix elements, and
    then it picks the *strongest* available coupling rather than a positional heuristic.

    Args:
        ground_states (Sequence[CoupledState]): field-dressed ground states.
        excited_states (Sequence[CoupledState]): field-dressed excited states.
        polarization (npt.NDArray[np.complex128]): Jones vector [Ex, Ey, Ez].
        absolute_coupling (float): matrix elements below this count as zero.
        normalize_pol (bool): normalize the polarization vector before evaluating.

    Returns:
        Tuple[int, int]: indices into ``ground_states`` and ``excited_states``.

    Raises:
        ValueError: if no pair has a matrix element above ``absolute_coupling``.
    """
    pol = np.asarray(polarization, dtype=np.complex128)
    ΔmF_raw = ΔmF_allowed(polarization)
    ΔmF_iterable = (
        (int(ΔmF_raw),)
        if isinstance(ΔmF_raw, (int, np.integer))
        else tuple(int(x) for x in np.asarray(ΔmF_raw).tolist())
    )

    def coupling(g_idx: int, e_idx: int) -> float:
        return abs(
            hamiltonian.generate_ED_ME_mixed_state(
                excited_states[e_idx],
                ground_states[g_idx],
                pol_vec=pol,
                normalize_pol=normalize_pol,
            )
        )

    # Pass 1: bare-allowed pairs only, in the historical scan order. Matrix elements are
    # evaluated lazily here so the common case stays cheap.
    allowed: List[Tuple[int, int]] = []
    allowed_mF0: List[Tuple[int, int]] = []
    for e_idx, exc in enumerate(excited_states):
        for g_idx, gnd in enumerate(ground_states):
            gnd_bs = cast(CoupledBasisState, gnd.largest)
            if not check_transition_coupled_allowed_polarization(
                gnd_bs,
                cast(CoupledBasisState, exc.largest),
                ΔmF_iterable,
                return_err=False,
            ):
                continue
            if coupling(g_idx, e_idx) < absolute_coupling:
                continue
            allowed.append((g_idx, e_idx))
            if gnd_bs.mF == 0:
                allowed_mF0.append((g_idx, e_idx))

    if allowed_mF0:
        return allowed_mF0[-1]
    if allowed:
        return allowed[len(allowed) // 2]

    # Pass 2: nothing is allowed by the bare rules, so the transition can only be driven
    # through field mixing. Take the strongest coupling, still preferring mF = 0.
    best: Optional[Tuple[int, int]] = None
    best_me = -np.inf
    best_mF0: Optional[Tuple[int, int]] = None
    best_me_mF0 = -np.inf
    for e_idx in range(len(excited_states)):
        for g_idx, gnd in enumerate(ground_states):
            me = coupling(g_idx, e_idx)
            if me < absolute_coupling:
                continue
            if me > best_me:
                best_me, best = me, (g_idx, e_idx)
            if cast(CoupledBasisState, gnd.largest).mF == 0 and me > best_me_mF0:
                best_me_mF0, best_mF0 = me, (g_idx, e_idx)

    chosen = best_mF0 if best_mF0 is not None else best
    if chosen is None:
        raise ValueError(
            "None of the supplied ground and excited states are coupled: every "
            f"mixed-state dipole matrix element is below absolute_coupling="
            f"{absolute_coupling:.1e} for polarization {polarization}."
        )
    return chosen


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
    weak_main_fraction: float = 1e-2,
    _dressed: Optional[
        Tuple[
            Sequence[states.CoupledState],
            Sequence[states.CoupledState],
            states.CoupledState,
            states.CoupledState,
        ]
    ] = None,
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
        weak_main_fraction (float): Warn when |main_coupling| falls below this fraction of
            the strongest element of the coupling matrix. main_coupling is the Rabi
            normalization reference, so a mixing-only main pair silently inflates every
            Rabi rate. Defaults to 1e-2.
        _dressed (tuple | None): Optionally supply already field-dressed
            (ground_states, excited_states, ground_main, excited_main) to avoid repeating
            the state assignment. Internal use.

    Returns:
        CouplingFields: Dataclass containing ground/excited states, couplings for each
            polarization, and the main coupling strength

    Raises:
        TypeError: If pol_main or pol_vecs are not numpy arrays with correct dtype
        ValueError: If the mixed-state matrix element between the main states vanishes,
            i.e. the transition is driven neither directly nor via field mixing.

    Notes:
        Whether a transition is allowed is decided by the magnitude of the *mixed-state*
        dipole matrix element between the field-dressed main states, not by applying the
        E1 selection rules to their bare labels. In an electric or magnetic field, state
        mixing makes nominally forbidden pairs genuinely driveable; at zero field the
        numeric test reduces exactly to the selection rules, since P, F and mF remain
        good quantum numbers there.
    """
    # Initialize default values
    if pol_main is None:
        pol_main = np.array([0, 0, 1], dtype=np.complex128)
    if pol_vecs is None:
        pol_vecs = []

    if not isinstance(pol_main, np.ndarray):
        raise TypeError("supply a numpy ndarray with dtype np.complex128 for pol_main")
    if len(pol_vecs) > 0 and not isinstance(pol_vecs[0], np.ndarray):
        raise TypeError(
            "supply a Sequence of np.ndarrays with dtype np.complex128 for pol_vecs"
        )
    if not np.issubdtype(pol_main.dtype, np.complex128):
        pol_main = pol_main.astype(np.complex128)
    if len(pol_vecs) > 0 and not np.issubdtype(pol_vecs[0].dtype, np.complex128):
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

    if _dressed is not None:
        ground_states, excited_states, ground_main, excited_main = _dressed
    else:
        ground_states = _dress_states(
            _ground_states_approx, _QN_basis, QN, H_rot, V_ref
        )
        excited_states = _dress_states(
            _excited_states_approx, _QN_basis, QN, H_rot, V_ref
        )
        ground_main = _dress_states(
            [ground_main_approx], _QN_basis, QN, H_rot, V_ref
        )[0]
        excited_main = _dress_states(
            [excited_main_approx], _QN_basis, QN, H_rot, V_ref
        )[0]

    states.check_approx_state_exact_state(ground_main_approx, ground_main)
    states.check_approx_state_exact_state(excited_main_approx, excited_main)
    ME_main = hamiltonian.generate_ED_ME_mixed_state(
        excited_main,
        ground_main,
        pol_vec=np.asarray(pol_main, dtype=np.complex128),
        normalize_pol=normalize_pol,
    )

    # The mixed-state matrix element is the authoritative test: it already accounts for
    # any Stark/Zeeman mixing that makes a bare-forbidden pair driveable. The E1
    # selection rules are consulted only to explain a vanishing element.
    if abs(ME_main) < MAIN_COUPLING_ZERO_TOL:
        raise ValueError(
            f"main coupling element for {ground_main_approx} -> "
            f"{excited_main_approx} is zero (|ME| = {abs(ME_main):.3e}), "
            f"pol = {pol_main}; "
            + _explain_zero_main_coupling(ground_main, excited_main, pol_main)
        )

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

    # main_coupling is the Rabi normalization reference: the whole coupling matrix is
    # divided by it when the symbolic Hamiltonian is built. A main pair that is only
    # weakly allowed (e.g. driveable solely through field mixing) therefore inflates
    # every Rabi rate in the system without any other visible symptom.
    if couplings and weak_main_fraction > 0:
        strongest = max(float(np.abs(c.field).max()) for c in couplings)
        pruned = abs(ME_main) < max(
            absolute_coupling, relative_coupling * strongest
        )
        if strongest > 0 and abs(ME_main) < weak_main_fraction * strongest:
            warnings.warn(
                (
                    "the main coupling element has been pruned from the coupling matrix "
                    "by relative_coupling/absolute_coupling, so the nominal main "
                    "transition is not driven at all. "
                    if pruned
                    else ""
                )
                + f"main coupling {ground_main.largest} -> {excited_main.largest} is weak: "
                f"|main_coupling| = {abs(ME_main):.3e} is only "
                f"{abs(ME_main) / strongest:.2e} of the strongest coupling "
                f"({strongest:.3e}) in the matrix. main_coupling normalizes the Rabi "
                f"rate, so the requested power will map to a much larger Rabi rate than "
                f"intended. Pass a more strongly coupled ground_main/excited_main pair, "
                f"or set weak_main_fraction=0 to silence this.",
                stacklevel=2,
            )

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
    weak_main_fraction: float = 1e-2,
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
    if not isinstance(pol_vecs[0], np.ndarray):
        raise TypeError(
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

    # Dress the states first so the main pair can be chosen on the actual mixed-state
    # matrix elements rather than on bare-label selection rules, which give the wrong
    # answer whenever an electric or magnetic field mixes the eigenstates.
    ground_states = _dress_states(_ground_states_approx, _QN_basis, QN, H_rot, V_ref)
    excited_states = _dress_states(_excited_states_approx, _QN_basis, QN, H_rot, V_ref)

    idg, ide = select_main_states_indices_coupling(
        ground_states,
        excited_states,
        pol_main,
        absolute_coupling=absolute_coupling,
        normalize_pol=normalize_pol,
    )

    return generate_coupling_field(
        ground_main_approx=_ground_states_approx[idg],
        excited_main_approx=_excited_states_approx[ide],
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
        weak_main_fraction=weak_main_fraction,
        _dressed=(
            ground_states,
            excited_states,
            ground_states[idg],
            excited_states[ide],
        ),
    )
