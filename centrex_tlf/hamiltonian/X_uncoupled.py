"""Hamiltonian operators for the X (ground) electronic state of TlF in the uncoupled basis.

This module implements the various terms of the molecular Hamiltonian including:
- Hyperfine coupling terms (Hc1, Hc2, Hc3, Hc4)
- Field-free Hamiltonian (Hff)
- Zeeman interaction terms (HZx, HZy, HZz)
- Stark interaction terms (HSx, HSy, HSz)
- Spherical tensor operators (R10, R1p, R1m)

All functions operate on uncoupled basis states and return UncoupledState objects.
"""

from functools import lru_cache

import numpy as np

from centrex_tlf.constants import XConstants
from centrex_tlf.states import UncoupledBasisState, UncoupledState, parity_X

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
    # Hyperfine coupling terms
    "Hc1",
    "Hc2",
    "Hc3a",
    "Hc3b",
    "Hc3c",
    "Hc3",
    "Hc4",
    # Field-free Hamiltonian
    "Hff",
    # Zeeman terms
    "HZx",
    "HZy",
    "HZz",
    # Stark terms
    "HSx",
    "HSy",
    "HSz",
    # Spherical tensor operators
    "R10",
    "R1m",
    "R1p",
]

########################################################
# Terms with angular momentum dot products
########################################################


def Hc1(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Thallium spin-rotation coupling term: c1 * I1 · J.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return coefficients.c1 * (
        com(I1z, Jz, psi, coefficients)
        + (1 / 2) * (com(I1p, Jm, psi, coefficients) + com(I1m, Jp, psi, coefficients))
    )


def Hc2(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Fluorine spin-rotation coupling term: c2 * I2 · J.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return coefficients.c2 * (
        com(I2z, Jz, psi, coefficients)
        + (1 / 2) * (com(I2p, Jm, psi, coefficients) + com(I2m, Jp, psi, coefficients))
    )


def Hc4(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Nuclear spin-spin coupling term: c4 * I1 · I2.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return coefficients.c4 * (
        com(I1z, I2z, psi, coefficients)
        + (1 / 2)
        * (com(I1p, I2m, psi, coefficients) + com(I1m, I2p, psi, coefficients))
    )


def Hc3a(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """First component of tensor spin-spin coupling: c3 term with I1·J and I2·J.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return (
        15
        * coefficients.c3
        / coefficients.c1
        / coefficients.c2
        * com(Hc1, Hc2, psi, coefficients)
        / ((2 * psi.J + 3) * (2 * psi.J - 1))
    )


def Hc3b(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Second component of tensor spin-spin coupling: c3 term with I2·J and I1·J.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return (
        15
        * coefficients.c3
        / coefficients.c2
        / coefficients.c1
        * com(Hc2, Hc1, psi, coefficients)
        / ((2 * psi.J + 3) * (2 * psi.J - 1))
    )


def Hc3c(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Third component of tensor spin-spin coupling: c3 term with I1·I2 and rotation.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return (
        -10
        * coefficients.c3
        / coefficients.c4
        / coefficients.B_rot
        * com(Hc4, Hrot, psi, coefficients)
        / ((2 * psi.J + 3) * (2 * psi.J - 1))
    )


def Hc3(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Complete tensor spin-spin coupling term: Hc3 = Hc3a + Hc3b + Hc3c.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return Hc3a(psi, coefficients) + Hc3b(psi, coefficients) + Hc3c(psi, coefficients)


########################################################
# Field free X state Hamiltonian
########################################################


def Hff(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Complete field-free Hamiltonian for X state.

    Includes rotational energy and all hyperfine coupling terms:
    H = H_rot + H_c1 + H_c2 + H_c3a + H_c3b + H_c3c + H_c4

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the complete field-free Hamiltonian
    """
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
    """Zeeman interaction with magnetic field in x-direction.

    H_Zx = -μ_J·J_x - μ_Tl·I1_x - μ_F·I2_x

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants with magnetic moments

    Returns:
        UncoupledState: Resulting state after applying the Zeeman operator

    Note:
        Cached for performance. For J=0 states, only nuclear terms contribute.
    """
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
    """Zeeman interaction with magnetic field in y-direction.

    H_Zy = -μ_J·J_y - μ_Tl·I1_y - μ_F·I2_y

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants with magnetic moments

    Returns:
        UncoupledState: Resulting state after applying the Zeeman operator

    Note:
        Cached for performance. For J=0 states, only nuclear terms contribute.
    """
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
    """Zeeman interaction with magnetic field in z-direction.

    H_Zz = -μ_J·J_z - μ_Tl·I1_z - μ_F·I2_z

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants with magnetic moments

    Returns:
        UncoupledState: Resulting state after applying the Zeeman operator

    Note:
        Cached for performance. For J=0 states, only nuclear terms contribute.
    """
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
    """Stark interaction with electric field in x-direction.

    H_Sx = -D·E_x = -D·(R_{1,-1} - R_{1,+1})/√2

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants with dipole moment D_TlF

    Returns:
        UncoupledState: Resulting state after applying the Stark operator

    Note:
        Cached for performance.
    """
    return (
        -coefficients.D_TlF
        * (R1m(psi, coefficients) - R1p(psi, coefficients))
        / np.sqrt(2)
    )


@lru_cache(maxsize=int(1e6))
def HSy(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Stark interaction with electric field in y-direction.

    H_Sy = -D·E_y = -iD·(R_{1,-1} + R_{1,+1})/√2

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants with dipole moment D_TlF

    Returns:
        UncoupledState: Resulting state after applying the Stark operator

    Note:
        Cached for performance.
    """
    return (
        -coefficients.D_TlF
        * 1j
        * (R1m(psi, coefficients) + R1p(psi, coefficients))
        / np.sqrt(2)
    )


@lru_cache(maxsize=int(1e6))
def HSz(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Stark interaction with electric field in z-direction.

    H_Sz = -D·E_z = -D·R_{1,0}

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants with dipole moment D_TlF

    Returns:
        UncoupledState: Resulting state after applying the Stark operator

    Note:
        Cached for performance.
    """
    return -coefficients.D_TlF * R10(psi, coefficients)


########################################################
# Spherical tensor operators
########################################################


def R10(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Rank-1 spherical tensor operator R_{1,0}.

    Connects states with ΔJ = ±1, Δm_J = 0 (π transition).

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Linear combination of states with J±1 and same m_J
    """
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
    """Rank-1 spherical tensor operator R_{1,-1}.

    Connects states with ΔJ = ±1, Δm_J = -1 (σ⁻ transition).

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Linear combination of states with J±1 and m_J-1
    """
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
    """Rank-1 spherical tensor operator R_{1,+1}.

    Connects states with ΔJ = ±1, Δm_J = +1 (σ⁺ transition).

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Linear combination of states with J±1 and m_J+1
    """
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
    """Nuclear spin-rotation coupling operator I1·R for Tl nucleus.

    Computes the commutator of I1 with the spherical tensor operators R_{1,q}.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return com(I1z, R10, psi, coefficients) + (
        com(I1p, R1m, psi, coefficients) - com(I1m, R1p, psi, coefficients)
    ) / np.sqrt(2)


def HI2R(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Nuclear spin-rotation coupling operator I2·R for F nucleus.

    Computes the commutator of I2 with the spherical tensor operators R_{1,q}.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return com(I2z, R10, psi, coefficients) + (
        com(I2p, R1m, psi, coefficients) - com(I2m, R1p, psi, coefficients)
    ) / np.sqrt(2)


def Hc3_alt(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Alternative formulation of tensor spin-spin coupling term.

    Uses nuclear spin-rotation coupling operators HI1R and HI2R to compute
    the c3 coupling term in an alternative form.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the operator
    """
    return 5 * coefficients.c3 / coefficients.c4 * Hc4(
        psi, coefficients
    ) - 15 * coefficients.c3 / 2 * (
        com(HI1R, HI2R, psi, coefficients) + com(HI2R, HI1R, psi, coefficients)
    )


@lru_cache(maxsize=int(1e6))
def Hff_alt(psi: UncoupledBasisState, coefficients: XConstants) -> UncoupledState:
    """Alternative formulation of field-free Hamiltonian for X state.

    Uses Hc3_alt instead of Hc3a + Hc3b + Hc3c to compute the tensor coupling.
    Includes: H = H_rot + H_c1 + H_c2 + H_c3_alt + H_c4

    Args:
        psi (UncoupledBasisState): Uncoupled basis state
        coefficients (XConstants): X state molecular constants

    Returns:
        UncoupledState: Resulting state after applying the complete field-free Hamiltonian

    Note:
        Cached for performance. Equivalent to Hff but uses alternative c3 formulation.
    """
    return (
        Hrot(psi, coefficients)
        + Hc1(psi, coefficients)
        + Hc2(psi, coefficients)
        + Hc3_alt(psi, coefficients)
        + Hc4(psi, coefficients)
    )
