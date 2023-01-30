from functools import lru_cache
from typing import Callable

import numpy as np

from centrex_tlf.states import BasisState, State, UncoupledBasisState

from .constants import HamiltonianConstants

__all__ = [
    "J2",
    "J4",
    "J6",
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


def J2(psi: BasisState, *args) -> State:
    return State([(psi.J * (psi.J + 1), psi)])


def J4(psi: BasisState, *args) -> State:
    return State([((psi.J * (psi.J + 1)) ** 2, psi)])


def J6(psi: BasisState, *args) -> State:
    return State([((psi.J * (psi.J + 1)) ** 3, psi)])


def Jz(psi: UncoupledBasisState, *args) -> State:
    return State([(psi.mJ, psi)])


def I1z(psi: UncoupledBasisState, *args) -> State:
    return State([(psi.m1, psi)])


def I2z(psi: UncoupledBasisState, *args) -> State:
    return State([(psi.m2, psi)])


########################################################
#
########################################################


def Jp(psi: UncoupledBasisState, *args) -> State:
    amp = np.sqrt((psi.J - psi.mJ) * (psi.J + psi.mJ + 1))
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
    return State([(amp, ket)])


def Jm(psi: UncoupledBasisState, *args) -> State:
    amp = np.sqrt((psi.J + psi.mJ) * (psi.J - psi.mJ + 1))
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
    return State([(amp, ket)])


def I1p(psi: UncoupledBasisState, *args) -> State:
    amp = np.sqrt((psi.I1 - psi.m1) * (psi.I1 + psi.m1 + 1))
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
    return State([(amp, ket)])


def I1m(psi: UncoupledBasisState, *args) -> State:
    amp = np.sqrt((psi.I1 + psi.m1) * (psi.I1 - psi.m1 + 1))
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
    return State([(amp, ket)])


def I2p(psi: UncoupledBasisState, *args) -> State:
    amp = np.sqrt((psi.I2 - psi.m2) * (psi.I2 + psi.m2 + 1))
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
    return State([(amp, ket)])


def I2m(psi: UncoupledBasisState, *args) -> State:
    amp = np.sqrt((psi.I2 + psi.m2) * (psi.I2 - psi.m2 + 1))
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
    return State([(amp, ket)])


########################################################
###
########################################################


def Jx(psi: UncoupledBasisState, *args) -> State:
    return 0.5 * (Jp(psi) + Jm(psi))


def Jy(psi: UncoupledBasisState, *args) -> State:
    return -0.5j * (Jp(psi) - Jm(psi))


def I1x(psi: UncoupledBasisState, *args) -> State:
    return 0.5 * (I1p(psi) + I1m(psi))


def I1y(psi: UncoupledBasisState, *args) -> State:
    return -0.5j * (I1p(psi) - I1m(psi))


def I2x(psi: UncoupledBasisState, *args) -> State:
    return 0.5 * (I2p(psi) + I2m(psi))


def I2y(psi: UncoupledBasisState, *args) -> State:
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
) -> State:
    ABpsi = State()
    # operate with A on all components in B|psi>
    for amp, cpt in B(psi, coefficients):
        ABpsi += amp * A(cpt, coefficients)
    return ABpsi
