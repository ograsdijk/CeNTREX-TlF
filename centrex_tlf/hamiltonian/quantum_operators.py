import math
from functools import lru_cache
from typing import Callable, overload

from centrex_tlf.constants import HamiltonianConstants
from centrex_tlf.states import (
    CoupledBasisState,
    CoupledState,
    UncoupledBasisState,
    UncoupledState,
)

__all__ = [
    "J2",
    "J4",
    "J6",
    "Jz",
    "I1z",
    "I2z",
    "Jp",
    "Jm",
    "I1p",
    "I1m",
    "I2p",
    "I2m",
    "Jx",
    "Jy",
    "I1x",
    "I1y",
    "I2x",
    "I2y",
    "com",
]

########################################################
# Diagonal operators multiple state by eigenvalue
########################################################


@overload
def J2(psi: UncoupledBasisState, *args) -> UncoupledState: ...


@overload
def J2(psi: CoupledBasisState, *args) -> CoupledState: ...


def J2(psi, *args):
    """J² operator: Square of total angular momentum.

    Returns the eigenvalue J(J+1) times the input state.

    Args:
        psi: Basis state (coupled or uncoupled)
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with amplitude J(J+1)
    """
    if isinstance(psi, CoupledBasisState):
        return CoupledState([(psi.J * (psi.J + 1), psi)])
    else:
        return UncoupledState([(psi.J * (psi.J + 1), psi)])


@overload
def J4(psi: UncoupledBasisState, *args) -> UncoupledState: ...


@overload
def J4(psi: CoupledBasisState, *args) -> CoupledState: ...


def J4(psi, *args):
    """J⁴ operator: Fourth power of total angular momentum.

    Returns the eigenvalue [J(J+1)]² times the input state.

    Args:
        psi: Basis state (coupled or uncoupled)
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with amplitude [J(J+1)]²
    """
    if isinstance(psi, CoupledBasisState):
        return CoupledState([((psi.J * (psi.J + 1)) ** 2, psi)])
    else:
        return UncoupledState([((psi.J * (psi.J + 1)) ** 2, psi)])


@overload
def J6(psi: UncoupledBasisState, *args) -> UncoupledState: ...


@overload
def J6(psi: CoupledBasisState, *args) -> CoupledState: ...


def J6(psi, *args):
    """J⁶ operator: Sixth power of total angular momentum.

    Returns the eigenvalue [J(J+1)]³ times the input state.

    Args:
        psi: Basis state (coupled or uncoupled)
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with amplitude [J(J+1)]³
    """
    if isinstance(psi, CoupledBasisState):
        return CoupledState([((psi.J * (psi.J + 1)) ** 3, psi)])
    else:
        return UncoupledState([((psi.J * (psi.J + 1)) ** 3, psi)])


def Jz(psi: UncoupledBasisState, *args) -> UncoupledState:
    """Jz operator: z-component of total angular momentum.

    Returns the eigenvalue mJ times the input state.

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with amplitude mJ
    """
    return UncoupledState([(psi.mJ, psi)])


def I1z(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I1z operator: z-component of first nuclear spin (Tl).

    Returns the eigenvalue m1 times the input state.
    Only defined for UncoupledBasisState (requires m1 quantum number).

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with amplitude m1
    """
    return UncoupledState([(float(psi.m1), psi)])


def I2z(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I2z operator: z-component of second nuclear spin (F).

    Returns the eigenvalue m2 times the input state.
    Only defined for UncoupledBasisState (requires m2 quantum number).

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with amplitude m2
    """
    return UncoupledState([(float(psi.m2), psi)])


########################################################
# Ladder operators
########################################################


def Jp(psi: UncoupledBasisState, *args) -> UncoupledState:
    """J+ operator: Raising operator for total angular momentum.

    Raises mJ by 1 with amplitude √[J(J+1) - mJ(mJ+1)].

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with mJ increased by 1
    """
    amp = math.sqrt((psi.J - psi.mJ) * (psi.J + psi.mJ + 1))
    ket = UncoupledBasisState(
        psi.J,
        psi.mJ + 1,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp, ket)])


def Jm(psi: UncoupledBasisState, *args) -> UncoupledState:
    """J- operator: Lowering operator for total angular momentum.

    Lowers mJ by 1 with amplitude √[J(J+1) - mJ(mJ-1)].

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with mJ decreased by 1
    """
    amp = math.sqrt((psi.J + psi.mJ) * (psi.J - psi.mJ + 1))
    ket = UncoupledBasisState(
        psi.J,
        psi.mJ - 1,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp, ket)])


def I1p(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I1+ operator: Raising operator for first nuclear spin (Tl).

    Raises m1 by 1 with amplitude √[I1(I1+1) - m1(m1+1)].

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with m1 increased by 1
    """
    amp = math.sqrt((psi.I1 - psi.m1) * (psi.I1 + psi.m1 + 1))
    ket = UncoupledBasisState(
        psi.J,
        psi.mJ,
        psi.I1,
        psi.m1 + 1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp, ket)])


def I1m(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I1- operator: Lowering operator for first nuclear spin (Tl).

    Lowers m1 by 1 with amplitude √[I1(I1+1) - m1(m1-1)].

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with m1 decreased by 1
    """
    amp = math.sqrt((psi.I1 + psi.m1) * (psi.I1 - psi.m1 + 1))
    ket = UncoupledBasisState(
        psi.J,
        psi.mJ,
        psi.I1,
        psi.m1 - 1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp, ket)])


def I2p(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I2+ operator: Raising operator for second nuclear spin (F).

    Raises m2 by 1 with amplitude √[I2(I2+1) - m2(m2+1)].

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with m2 increased by 1
    """
    amp = math.sqrt((psi.I2 - psi.m2) * (psi.I2 + psi.m2 + 1))
    ket = UncoupledBasisState(
        psi.J,
        psi.mJ,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2 + 1,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp, ket)])


def I2m(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I2- operator: Lowering operator for second nuclear spin (F).

    Lowers m2 by 1 with amplitude √[I2(I2+1) - m2(m2-1)].

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        State with m2 decreased by 1
    """
    amp = math.sqrt((psi.I2 + psi.m2) * (psi.I2 - psi.m2 + 1))
    ket = UncoupledBasisState(
        psi.J,
        psi.mJ,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2 - 1,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp, ket)])


########################################################
# Cartesian operators (defined in terms of ladder operators)
########################################################


def Jx(psi: UncoupledBasisState, *args) -> UncoupledState:
    """Jx operator: x-component of total angular momentum.

    Defined as (J+ + J-)/2.

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        Linear combination of states from ladder operators
    """
    return 0.5 * (Jp(psi) + Jm(psi))


def Jy(psi: UncoupledBasisState, *args) -> UncoupledState:
    """Jy operator: y-component of total angular momentum.

    Defined as -i(J+ - J-)/2.

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        Linear combination of states from ladder operators
    """
    return -0.5j * (Jp(psi) - Jm(psi))


def I1x(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I1x operator: x-component of first nuclear spin (Tl).

    Defined as (I1+ + I1-)/2.

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        Linear combination of states from ladder operators
    """
    return 0.5 * (I1p(psi) + I1m(psi))


def I1y(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I1y operator: y-component of first nuclear spin (Tl).

    Defined as -i(I1+ - I1-)/2.

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        Linear combination of states from ladder operators
    """
    return -0.5j * (I1p(psi) - I1m(psi))


def I2x(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I2x operator: x-component of second nuclear spin (F).

    Defined as (I2+ + I2-)/2.

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        Linear combination of states from ladder operators
    """
    return 0.5 * (I2p(psi) + I2m(psi))


def I2y(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I2y operator: y-component of second nuclear spin (F).

    Defined as -i(I2+ - I2-)/2.

    Args:
        psi: Uncoupled basis state
        *args: Additional arguments (unused, for compatibility)

    Returns:
        Linear combination of states from ladder operators
    """
    return -0.5j * (I2p(psi) - I2m(psi))


########################################################
# Composition of operators
########################################################


@lru_cache(maxsize=int(1e6))
def com(
    A: Callable,
    B: Callable,
    psi: UncoupledBasisState,
    coefficients: HamiltonianConstants,
) -> UncoupledState:
    """Compose two quantum operators: Apply A(B|ψ⟩).

    Computes the action of operator A on the result of operator B acting on |ψ⟩.
    Results are cached for performance.

    Args:
        A: First operator to apply
        B: Second operator to apply (applied first)
        psi: Uncoupled basis state
        coefficients: Hamiltonian constants (physical parameters)

    Returns:
        State resulting from A(B|ψ⟩)

    Note:
        This is NOT the commutator [A,B]. For commutator, compute A(B|ψ⟩) - B(A|ψ⟩).
    """
    ABpsi = UncoupledState()
    # operate with A on all components in B|psi>
    for amp, cpt in B(psi, coefficients):
        ABpsi += amp * A(cpt, coefficients)
    return ABpsi
