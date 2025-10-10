from __future__ import annotations

import copy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf.constants import TlFNuclearSpins

from .find_states import QuantumSelector, get_unique_basisstates_from_basisstates
from .states import Basis, CoupledBasisState, ElectronicState, UncoupledBasisState
from .utils import parity_X

__all__ = [
    "generate_uncoupled_states_ground",
    "generate_uncoupled_states_excited",
    "generate_coupled_states_ground",
    "generate_coupled_states_excited",
    "generate_coupled_states_X",
    "generate_coupled_states_B",
]


def generate_uncoupled_states_ground(
    Js: Union[List[int], npt.NDArray[np.int_]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    """Generate uncoupled basis states for the X (ground) electronic state.

    Creates all possible uncoupled basis states for specified rotational levels J
    of the ground X electronic state, including all combinations of mJ, m1 (Tl nuclear
    spin projection), and m2 (F nuclear spin projection).

    Args:
        Js (Union[List[int], npt.NDArray[np.int_]]): Rotational quantum numbers J to
            generate states for
        nuclear_spins (TlFNuclearSpins, optional): Nuclear spin values for Tl and F.
            Defaults to TlFNuclearSpins().

    Returns:
        npt.NDArray[Any]: Array of UncoupledBasisState objects for the ground state
    """
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F
    # convert J to int(J); np.int with (-1)**J throws an exception for negative J
    QN = np.array(
        [
            UncoupledBasisState(
                int(J),
                mJ,
                I_Tl,
                float(m1),
                I_F,
                float(m2),
                Omega=0,
                P=parity_X(J),
                electronic_state=ElectronicState.X,
                basis=Basis.Uncoupled,
            )
            for J in Js
            for mJ in range(-J, J + 1)
            for m1 in np.arange(-I_Tl, I_Tl + 1)
            for m2 in np.arange(-I_F, I_F + 1)
        ]
    )
    return QN


def generate_uncoupled_states_excited(
    Js: Union[List[int], npt.NDArray[np.int_]],
    Ωs: List[int] = [-1, 1],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    """Generate uncoupled basis states for the B (excited) electronic state.

    Creates all possible uncoupled basis states for specified rotational levels J
    and Ω values of the excited B electronic state, including all combinations of
    mJ, m1 (Tl nuclear spin projection), and m2 (F nuclear spin projection).

    Args:
        Js (Union[List[int], npt.NDArray[np.int_]]): Rotational quantum numbers J to
            generate states for
        Ωs (List[int], optional): Projection of electronic angular momentum on
            internuclear axis. Defaults to [-1, 1].
        nuclear_spins (TlFNuclearSpins, optional): Nuclear spin values for Tl and F.
            Defaults to TlFNuclearSpins().

    Returns:
        npt.NDArray[Any]: Array of UncoupledBasisState objects for the excited state
    """
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F
    QN = np.array(
        [
            UncoupledBasisState(
                J,
                mJ,
                I_Tl,
                float(m1),
                I_F,
                float(m2),
                Omega=Ω,
                electronic_state=ElectronicState.B,
                basis=Basis.Uncoupled,
            )
            for Ω in Ωs
            for J in Js
            for mJ in range(-J, J + 1)
            for m1 in np.arange(-I_Tl, I_Tl + 1)
            for m2 in np.arange(-I_F, I_F + 1)
        ]
    )
    return QN


def generate_coupled_states_ground(
    Js: Union[List[int], npt.NDArray[np.int_]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    """Generate coupled basis states for the X (ground) electronic state.

    Creates all possible coupled basis states for specified rotational levels J
    of the ground X electronic state. States are coupled in the |F, mF, F1, J⟩ basis
    where F1 = J + I_F and F = F1 + I_Tl (hyperfine structure).

    Args:
        Js (Union[List[int], npt.NDArray[np.int_]]): Rotational quantum numbers J to
            generate states for
        nuclear_spins (TlFNuclearSpins, optional): Nuclear spin values for Tl and F.
            Defaults to TlFNuclearSpins().

    Returns:
        npt.NDArray[Any]: Array of CoupledBasisState objects for the ground state
    """
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F
    QN = np.array(
        [
            CoupledBasisState(
                F,
                mF,
                float(F1),
                J,
                I_F,
                I_Tl,
                electronic_state=ElectronicState.X,
                P=parity_X(J),
                Omega=0,
                basis=Basis.Coupled,
            )
            for J in Js
            for F1 in np.arange(np.abs(J - I_F), J + I_F + 1)
            for F in range(int(abs(F1 - I_Tl)), int(F1 + I_Tl + 1))
            for mF in range(-F, F + 1)
        ]
    )
    return QN


def generate_coupled_states_excited(
    Js: Union[List[int], npt.NDArray[np.int_]],
    Ps: Union[int, List[int], Tuple[int]] = 1,
    Omegas: Union[int, List[int], Tuple[int]] = 1,
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    basis: Optional[Basis] = None,
) -> npt.NDArray[Any]:
    """Generate coupled basis states for the B (excited) electronic state.

    Creates all possible coupled basis states for specified rotational levels J,
    parities P, and Ω values of the excited B electronic state. Can generate states
    in either parity basis (P) or Ω basis.

    Args:
        Js (Union[List[int], npt.NDArray[np.int_]]): Rotational quantum numbers J to
            generate states for
        Ps (Union[int, List[int], Tuple[int]], optional): Parity quantum number(s).
            Defaults to 1.
        Omegas (Union[int, List[int], Tuple[int]], optional): Projection of electronic
            angular momentum on internuclear axis. Defaults to 1.
        nuclear_spins (TlFNuclearSpins, optional): Nuclear spin values for Tl and F.
            Defaults to TlFNuclearSpins().
        basis (Optional[Basis], optional): Basis to use for states. Defaults to None.

    Returns:
        npt.NDArray[Any]: Array of CoupledBasisState objects for the excited state

    Raises:
        ValueError: If both multiple P and multiple Ω values are provided
    """
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F

    if not isinstance(Ps, (list, tuple)):
        _Ps = [Ps]
    else:
        _Ps = list(Ps)
    if not isinstance(Omegas, (list, tuple)):
        _Omegas = [Omegas]
    else:
        _Omegas = list(Omegas)

    # Check for conflicting basis specification
    if len(_Ps) > 1 and len(_Omegas) > 1:
        raise ValueError(
            "Cannot supply both multiple Ω and multiple P values, need to pick a basis"
        )

    # Single comprehension works for both bases - no need to duplicate
    QN = np.array(
        [
            CoupledBasisState(
                F,
                mF,
                float(F1),
                J,
                I_F,
                I_Tl,
                electronic_state=ElectronicState.B,
                P=P,
                Omega=Omega,
                basis=basis,
            )
            for J in Js
            for F1 in np.arange(np.abs(J - I_F), J + I_F + 1)
            for F in range(int(abs(F1 - I_Tl)), int(F1 + I_Tl + 1))
            for mF in range(-F, F + 1)
            for P in _Ps
            for Omega in _Omegas
        ]
    )
    return QN


def generate_coupled_states_base(
    qn_selector: QuantumSelector,
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    basis: Optional[Basis] = None,
) -> List[CoupledBasisState]:
    """Generate CoupledBasisStates for quantum numbers specified by qn_selector.

    Base function that generates all coupled basis states matching the criteria in
    qn_selector. Handles all combinations of quantum numbers and validates that
    physically allowed combinations are generated (e.g., F1 must satisfy angular
    momentum coupling rules).

    Args:
        qn_selector (QuantumSelector): QuantumSelector with quantum numbers to use
            for generation
        nuclear_spins (TlFNuclearSpins, optional): Nuclear spin values for Tl and F.
            Defaults to TlFNuclearSpins().
        basis (Optional[Basis], optional): Basis to use for the states. Defaults to None.

    Returns:
        List[CoupledBasisState]: List of CoupledBasisStates for the specified quantum numbers

    Raises:
        ValueError: If required quantum numbers (P, J, electronic state, or Ω) are not set
    """
    # Validate required quantum numbers
    if (basis is not None) and (basis != Basis.CoupledΩ):
        if qn_selector.P is None:
            raise ValueError("function requires a parity to be set for this basis")

    if qn_selector.J is None:
        raise ValueError("function requires a rotational quantum number J to be set")

    if qn_selector.electronic is None:
        raise ValueError("function requires electronic state to be set")

    if qn_selector.Ω is None:
        raise ValueError("function requires Ω to be set")

    # Generate all combinations
    quantum_numbers = []
    for field_name in ["J", "F1", "F", "mF", "electronic", "P", "Ω"]:
        field_val = getattr(qn_selector, field_name)
        quantum_numbers.append(
            [field_val]
            if not isinstance(field_val, (list, tuple, np.ndarray))
            else list(field_val)
        )

    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F

    QN: List[CoupledBasisState] = []
    # the worst nested loops I've ever created
    Js, F1s, Fs, mFs, estates, Ps, Ωs = quantum_numbers
    for estate in estates:
        for J in Js:
            F1_allowed = np.arange(np.abs(J - I_F), J + I_F + 1)
            F1sl = F1s if F1s[0] is not None else F1_allowed
            for F1 in F1sl:
                if F1 not in F1_allowed:
                    continue
                Fs_allowed = range(int(abs(F1 - I_Tl)), int(F1 + I_Tl + 1))
                Fsl = Fs if Fs[0] is not None else Fs_allowed
                for F in Fsl:
                    if F not in Fs_allowed:
                        continue
                    mF_allowed = range(-F, F + 1)
                    mFsl = mFs if mFs[0] is not None else mF_allowed
                    for mF in mFsl:
                        if mF not in mF_allowed:
                            continue
                        for P_val in Ps:
                            # Handle callable P (e.g., parity_X function)
                            P_resolved: Optional[int] = (
                                P_val(J) if callable(P_val) else P_val
                            )  # type: ignore[assignment]
                            for Ω in Ωs:
                                QN.append(
                                    CoupledBasisState(
                                        F,
                                        mF,
                                        float(F1),
                                        J,
                                        I_F,
                                        I_Tl,
                                        electronic_state=estate,
                                        P=P_resolved,
                                        Ω=Ω,
                                        basis=basis,
                                    )
                                )
    return QN


def generate_coupled_states_X(
    qn_selector: Union[QuantumSelector, Sequence[QuantumSelector], npt.NDArray[Any]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    basis: Basis = Basis.Coupled,
) -> List[CoupledBasisState]:
    """Generate ground X state CoupledBasisStates for quantum numbers in qn_selector.

    Convenience function that automatically sets Ω=0, P=parity_X(J), and electronic
    state to X for the provided quantum selectors, then generates the corresponding
    coupled basis states.

    Args:
        qn_selector (Union[QuantumSelector, Sequence[QuantumSelector], npt.NDArray[Any]]):
            QuantumSelector or sequence of QuantumSelectors specifying quantum numbers
            to generate states for
        nuclear_spins (TlFNuclearSpins, optional): Nuclear spin values for Tl and F.
            Defaults to TlFNuclearSpins().
        basis (Basis, optional): Basis to use for the states. Defaults to Basis.Coupled.

    Returns:
        List[CoupledBasisState]: List of unique CoupledBasisStates for the ground X state

    Raises:
        AssertionError: If qn_selector is not a QuantumSelector, list, or np.ndarray
    """
    if isinstance(qn_selector, QuantumSelector):
        qns = copy.copy(qn_selector)
        qns.Ω = 0
        qns.P = parity_X
        qns.electronic = ElectronicState.X
        return generate_coupled_states_base(qns, nuclear_spins=nuclear_spins)
    elif isinstance(qn_selector, (list, np.ndarray)):
        coupled_states = []
        for qns in qn_selector:
            qns = copy.copy(qns)
            qns.Ω = 0
            qns.P = parity_X
            qns.electronic = ElectronicState.X
            coupled_states.append(
                generate_coupled_states_base(
                    qns, nuclear_spins=nuclear_spins, basis=basis
                )
            )

        return get_unique_basisstates_from_basisstates(
            [item for sublist in coupled_states for item in sublist]
        )
    else:
        raise AssertionError(
            "qn_selector required to be of type QuantumSelector, list or np.ndarray"
        )


def check_B_basis(
    P: Union[int, Callable, Sequence[int], npt.NDArray[np.int_], None],
    Ω: Union[int, Sequence[int], npt.NDArray[np.int_], None],
) -> None:
    """Validate that P and Ω specifications are compatible for B state basis.

    Checks that either P or Ω is specified (but not both with multiple values),
    ensuring a unique basis is chosen for the B state.

    Args:
        P (Union[int, Callable, Sequence[int], npt.NDArray[np.int_], None]): Parity
            quantum number(s) or callable returning parity
        Ω (Union[int, Sequence[int], npt.NDArray[np.int_], None]): Projection of
            electronic angular momentum quantum number(s)

    Raises:
        ValueError: If neither P nor Ω is specified, or if both have multiple values
    """
    if P is None and Ω is None:
        raise ValueError("Need to supply either P or Ω to determine the basis")

    # Check if both have multiple values (conflicting basis specification)
    P_is_multi = isinstance(P, (list, tuple)) and len(P) > 1
    Ω_is_multi = isinstance(Ω, (list, tuple)) and len(Ω) > 1

    if P_is_multi and Ω_is_multi:
        raise ValueError(
            "Cannot supply both multiple P and multiple Ω values, need to pick a basis"
        )


def generate_coupled_states_B(
    qn_selector: Union[QuantumSelector, Sequence[QuantumSelector], npt.NDArray[Any]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    basis: Optional[Basis] = None,
) -> List[CoupledBasisState]:
    """Generate excited B state CoupledBasisStates for quantum numbers in qn_selector.

    Convenience function that automatically sets electronic state to B, defaults Ω to 1
    if not specified, validates basis compatibility, and generates the corresponding
    coupled basis states.

    Args:
        qn_selector (Union[QuantumSelector, Sequence[QuantumSelector], npt.NDArray[Any]]):
            QuantumSelector or sequence of QuantumSelectors specifying quantum numbers
            to generate states for
        nuclear_spins (TlFNuclearSpins, optional): Nuclear spin values for Tl and F.
            Defaults to TlFNuclearSpins().
        basis (Optional[Basis], optional): Basis to use for the states. Defaults to None.

    Returns:
        List[CoupledBasisState]: List of unique CoupledBasisStates for the excited B state

    Raises:
        AssertionError: If qn_selector is not a QuantumSelector, list, or np.ndarray
        ValueError: If P and Ω specifications are incompatible (via check_B_basis)
    """
    if isinstance(qn_selector, QuantumSelector):
        qns = copy.copy(qn_selector)
        qns.Ω = 1 if qns.Ω is None else qns.Ω
        qns.electronic = ElectronicState.B
        check_B_basis(qns.P, qns.Ω)
        return generate_coupled_states_base(qns, nuclear_spins=nuclear_spins)
    elif isinstance(qn_selector, (list, np.ndarray)):
        coupled_states = []
        for qns in qn_selector:
            qns = copy.copy(qns)
            qns.Ω = 1 if qns.Ω is None else qns.Ω
            qns.electronic = ElectronicState.B
            check_B_basis(qns.P, qns.Ω)
            coupled_states.append(
                generate_coupled_states_base(
                    qns, nuclear_spins=nuclear_spins, basis=basis
                )
            )
        return get_unique_basisstates_from_basisstates(
            [item for sublist in coupled_states for item in sublist]
        )
    else:
        raise AssertionError(
            "qn_selector required to be of type QuantumSelector, list or np.ndarray"
        )
