from functools import lru_cache

import numpy as np

from centrex_tlf.states import UncoupledBasisState, UncoupledState, parity_X

from .constants import XConstants
from .general_uncoupled import Hrot
from .quantum_operators import (
    I1m,
    I1p,
    I1x,
    I1y,
    I1z,
    I2m,
    I2p,
    I2x,
    I2y,
    I2z,
    Jm,
    Jp,
    Jx,
    Jy,
    Jz,
    com,
)

__all__ = [
    "Hc1",
    "Hc2",
    "Hc4",
    "Hc3a",
    "Hc3b",
    "Hc3c",
    "Hc3",
    "Hff",
    "HZx",
    "HZy",
    "HZz",
    "HSx",
    "HSy",
    "HSz",
    "R1p",
]

########################################################
# Terms with angular momentum dot products
########################################################


def Hc1(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return coefficients.c1 * (
        com(I1z, Jz, psi, coefficients)
        + (1 / 2) * (com(I1p, Jm, psi, coefficients) + com(I1m, Jp, psi, coefficients))
    )


def Hc2(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return coefficients.c2 * (
        com(I2z, Jz, psi, coefficients)
        + (1 / 2) * (com(I2p, Jm, psi, coefficients) + com(I2m, Jp, psi, coefficients))
    )


def Hc4(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return coefficients.c4 * (
        com(I1z, I2z, psi, coefficients)
        + (1 / 2)
        * (com(I1p, I2m, psi, coefficients) + com(I1m, I2p, psi, coefficients))
    )


def Hc3a(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return (
        15
        * coefficients.c3
        / coefficients.c1
        / coefficients.c2
        * com(Hc1, Hc2, psi, coefficients)
        / ((2 * psi.J + 3) * (2 * psi.J - 1))
    )


def Hc3b(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return (
        15
        * coefficients.c3
        / coefficients.c2
        / coefficients.c1
        * com(Hc2, Hc1, psi, coefficients)
        / ((2 * psi.J + 3) * (2 * psi.J - 1))
    )


def Hc3c(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return (
        -10
        * coefficients.c3
        / coefficients.c4
        / coefficients.B_rot
        * com(Hc4, Hrot, psi, coefficients)
        / ((2 * psi.J + 3) * (2 * psi.J - 1))
    )


def Hc3(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return Hc3a(psi, coefficients) + Hc3b(psi, coefficients) + Hc3c(psi, coefficients)


########################################################
# Field free X state Hamiltonian
########################################################


def Hff(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return (
        Hrot(psi, coefficients)
        + Hc1(psi, coefficients)
        + Hc2(psi, coefficients)
        + Hc3a(psi, coefficients)
        + Hc3b(psi, coefficients)
        + Hc3c(psi, coefficients)
        + Hc4(psi, coefficients)
    )


########################################################
# Zeeman X state
########################################################


@lru_cache(maxsize=int(1e6))
def HZx(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    if psi.J != 0:
        return (
            -coefficients.μ_J / psi.J * Jx(psi)
            - coefficients.μ_Tl / psi.I1 * I1x(psi)
            - coefficients.μ_F / psi.I2 * I2x(psi)
        )
    else:
        return -coefficients.μ_Tl / psi.I1 * I1x(psi) - coefficients.μ_F / psi.I2 * I2x(
            psi
        )


@lru_cache(maxsize=int(1e6))
def HZy(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    if psi.J != 0:
        return (
            -coefficients.μ_J / psi.J * Jy(psi)
            - coefficients.μ_Tl / psi.I1 * I1y(psi)
            - coefficients.μ_F / psi.I2 * I2y(psi)
        )
    else:
        return -coefficients.μ_Tl / psi.I1 * I1y(psi) - coefficients.μ_F / psi.I2 * I2y(
            psi
        )


@lru_cache(maxsize=int(1e6))
def HZz(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    if psi.J != 0:
        return (
            -coefficients.μ_J / psi.J * Jz(psi)
            - coefficients.μ_Tl / psi.I1 * I1z(psi)
            - coefficients.μ_F / psi.I2 * I2z(psi)
        )
    else:
        return -coefficients.μ_Tl / psi.I1 * I1z(psi) - coefficients.μ_F / psi.I2 * I2z(
            psi
        )


########################################################
# Stark Hamiltonian
########################################################


@lru_cache(maxsize=int(1e6))
def HSx(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return (
        -coefficients.D_TlF
        * (R1m(psi, coefficients) - R1p(psi, coefficients))
        / np.sqrt(2)
    )


@lru_cache(maxsize=int(1e6))
def HSy(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return (
        -coefficients.D_TlF
        * 1j
        * (R1m(psi, coefficients) + R1p(psi, coefficients))
        / np.sqrt(2)
    )


@lru_cache(maxsize=int(1e6))
def HSz(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return -coefficients.D_TlF * R10(psi, coefficients)


# Old functions from Jakobs original Hamiltonian


def R10(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    amp1 = np.sqrt(2) * np.sqrt(
        (psi.J - psi.mJ) * (psi.J + psi.mJ) / (8 * psi.J**2 - 2)
    )
    ket1 = UncoupledBasisState(
        psi.J - 1,
        psi.mJ,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=parity_X(psi.J - 1),
        electronic_state=psi.electronic_state,
    )
    amp2 = np.sqrt(2) * np.sqrt(
        (psi.J - psi.mJ + 1) * (psi.J + psi.mJ + 1) / (6 + 8 * psi.J * (psi.J + 2))
    )
    ket2 = UncoupledBasisState(
        psi.J + 1,
        psi.mJ,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=parity_X(psi.J + 1),
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp1, ket1), (amp2, ket2)])


def R1m(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    amp1 = (
        -0.5
        * np.sqrt(2)
        * np.sqrt((psi.J + psi.mJ) * (psi.J + psi.mJ - 1) / (4 * psi.J**2 - 1))
    )
    ket1 = UncoupledBasisState(
        psi.J - 1,
        psi.mJ - 1,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=parity_X(psi.J - 1),
        electronic_state=psi.electronic_state,
    )
    amp2 = (
        0.5
        * np.sqrt(2)
        * np.sqrt(
            (psi.J - psi.mJ + 1) * (psi.J - psi.mJ + 2) / (3 + 4 * psi.J * (psi.J + 2))
        )
    )
    ket2 = UncoupledBasisState(
        psi.J + 1,
        psi.mJ - 1,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=parity_X(psi.J + 1),
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp1, ket1), (amp2, ket2)])


def R1p(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    amp1: float = (
        -0.5
        * np.sqrt(2)
        * np.sqrt((psi.J - psi.mJ) * (psi.J - psi.mJ - 1) / (4 * psi.J**2 - 1))
    )
    ket1 = UncoupledBasisState(
        psi.J - 1,
        psi.mJ + 1,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=parity_X(psi.J - 1),
        electronic_state=psi.electronic_state,
    )
    amp2: float = (
        0.5
        * np.sqrt(2)
        * np.sqrt(
            (psi.J + psi.mJ + 1) * (psi.J + psi.mJ + 2) / (3 + 4 * psi.J * (psi.J + 2))
        )
    )
    ket2 = UncoupledBasisState(
        psi.J + 1,
        psi.mJ + 1,
        psi.I1,
        psi.m1,
        psi.I2,
        psi.m2,
        Omega=psi.Omega,
        P=parity_X(psi.J + 1),
        electronic_state=psi.electronic_state,
    )
    return UncoupledState([(amp1, ket1), (amp2, ket2)])


def HI1R(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return com(I1z, R10, psi, coefficients) + (
        com(I1p, R1m, psi, coefficients) - com(I1m, R1p, psi, coefficients)
    ) / np.sqrt(2)


def HI2R(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return com(I2z, R10, psi, coefficients) + (
        com(I2p, R1m, psi, coefficients) - com(I2m, R1p, psi, coefficients)
    ) / np.sqrt(2)


def Hc3_alt(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return 5 * coefficients.c3 / coefficients.c4 * Hc4(
        psi, coefficients
    ) - 15 * coefficients.c3 / 2 * (
        com(HI1R, HI2R, psi, coefficients) + com(HI2R, HI1R, psi, coefficients)
    )


@lru_cache(maxsize=int(1e6))
def Hff_alt(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    return (
        Hrot(psi, coefficients)
        + Hc1(psi, coefficients)
        + Hc2(psi, coefficients)
        + Hc3_alt(psi, coefficients)
        + Hc4(psi, coefficients)
    )
