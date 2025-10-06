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
    if isinstance(psi, CoupledBasisState):
        return CoupledState([(psi.J * (psi.J + 1), psi)])
    else:
        return UncoupledState([(psi.J * (psi.J + 1), psi)])


@overload
def J4(psi: UncoupledBasisState, *args) -> UncoupledState: ...


@overload
def J4(psi: CoupledBasisState, *args) -> CoupledState: ...


def J4(psi, *args):
    if isinstance(psi, CoupledBasisState):
        return CoupledState([((psi.J * (psi.J + 1)) ** 2, psi)])
    else:
        return UncoupledState([((psi.J * (psi.J + 1)) ** 2, psi)])


@overload
def J6(psi: UncoupledBasisState, *args) -> UncoupledState: ...


@overload
def J6(psi: CoupledBasisState, *args) -> CoupledState: ...


def J6(psi, *args):
    if isinstance(psi, CoupledBasisState):
        return CoupledState([((psi.J * (psi.J + 1)) ** 3, psi)])
    else:
        return UncoupledState([((psi.J * (psi.J + 1)) ** 3, psi)])


def Jz(psi: UncoupledBasisState, *args) -> UncoupledState:
    return UncoupledState([(psi.mJ, psi)])


def I1z(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I1z operator - only defined for UncoupledBasisState (requires m1 quantum number)."""
    return UncoupledState([(psi.m1, psi)])


def I2z(psi: UncoupledBasisState, *args) -> UncoupledState:
    """I2z operator - only defined for UncoupledBasisState (requires m2 quantum number)."""
    return UncoupledState([(psi.m2, psi)])


########################################################
# Ladder operators
########################################################


def Jp(psi: UncoupledBasisState, *args) -> UncoupledState:
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
    return 0.5 * (Jp(psi) + Jm(psi))


def Jy(psi: UncoupledBasisState, *args) -> UncoupledState:
    return -0.5j * (Jp(psi) - Jm(psi))


def I1x(psi: UncoupledBasisState, *args) -> UncoupledState:
    return 0.5 * (I1p(psi) + I1m(psi))


def I1y(psi: UncoupledBasisState, *args) -> UncoupledState:
    return -0.5j * (I1p(psi) - I1m(psi))


def I2x(psi: UncoupledBasisState, *args) -> UncoupledState:
    return 0.5 * (I2p(psi) + I2m(psi))


def I2y(psi: UncoupledBasisState, *args) -> UncoupledState:
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
    ABpsi = UncoupledState()
    # operate with A on all components in B|psi>
    for amp, cpt in B(psi, coefficients):
        ABpsi += amp * A(cpt, coefficients)
    return ABpsi
