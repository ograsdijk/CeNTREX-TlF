"""Magnetic hyperfine Hamiltonian operators for B state in coupled Omega basis.

This module implements magnetic hyperfine interactions for both nuclear spins (Tl and F)
in the B electronic state using the coupled |J,F₁,F,mF,Ω⟩ basis with fixed Omega.

The magnetic hyperfine interaction arises from the coupling between nuclear magnetic
moments and the magnetic field produced by unpaired electrons. For molecules in Π
electronic states with Omega ≠ 0, the interaction includes an additional Omega-dependent
factor.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from centrex_tlf.constants import BConstants
from centrex_tlf.states import CoupledBasisState, CoupledState

from ..wigner import sixj_f, threej_f


@lru_cache(maxsize=int(1e6))
def H_mhf_Tl(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Magnetic hyperfine operator for Tl nucleus in B state (coupled Omega basis).

    Calculates the magnetic hyperfine interaction for the thallium (Tl, ²⁰⁵Tl) nucleus
    with nuclear spin I₁=1/2. This interaction couples the Tl nuclear magnetic moment
    to the magnetic field produced by the unpaired electron.

    The operator is off-diagonal in J (ΔJ = 0, ±1) while preserving I₁, I₂, F₁, F,
    mF, and Omega. The Omega-dependent factor reflects the projection of electronic
    angular momentum along the internuclear axis.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state, requires h1_Tl parameter
            (Tl magnetic hyperfine constant).

    Returns:
        CoupledState containing superposition over states with J' = J-1, J, J+1.
        Each term includes Wigner 6-j and 3-j symbols, phase factors, and angular
        momentum factors. Zero-amplitude terms are excluded.

    Notes:
        - Selection rules: ΔJ = 0, ±1; ΔI₁ = 0; ΔF₁ = 0; ΔF = 0; ΔmF = 0; ΔΩ = 0
        - For TlF B state: h1(Tl) ≈ 28.789 GHz
        - Tl has two isotopes: ²⁰⁵Tl (I=1/2, 70.5%) and ²⁰³Tl (I=1/2, 29.5%)
        - The Omega factor makes this interaction state-dependent in Π states

    References:
        Brown, J. M., & Carrington, A. (2003). "Rotational Spectroscopy of Diatomic
        Molecules." Cambridge University Press. Eq. 8.324 and following.

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(h1_Tl=28789000000.0)
        >>> result = H_mhf_Tl(state, constants)
        >>> # Returns superposition with J=0,1,2
    """
    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    # I1, I2, F1, F and mF are the same for both states
    I1 = I1p
    I2 = I2p
    F1 = F1p
    F = Fp
    mF = mFp

    # Omega also doesn't change
    Omega = Omegap

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix element can
    # be non-zero
    # Need J = Jp+1 ... |Jp-1|
    for J in np.arange(np.abs(Jp - 1), Jp + 2):
        # Calculate matrix element
        amp = (
            Omega
            * constants.h1_Tl
            * (-1) ** (J + Jp + F1 + I1 - Omega)
            * sixj_f(I1, Jp, F1, J, I1, 1)
            * threej_f(J, 1, Jp, -Omega, 0, Omegap)
            * np.sqrt((2 * J + 1) * (2 * Jp + 1) * I1 * (I1 + 1) * (2 * I1 + 1))
        )

        basis_state = CoupledBasisState(
            F,
            mF,
            F1,
            J,
            I1,
            I2,
            Omega=Omega,
            electronic_state=psi.electronic_state,
            P=psi.P,
        )
        if amp != 0:
            data.append((amp, basis_state))

    return CoupledState(data)


@lru_cache(maxsize=int(1e6))
def H_mhf_F(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Magnetic hyperfine operator for F nucleus in B state (coupled Omega basis).

    Calculates the magnetic hyperfine interaction for the fluorine (¹⁹F) nucleus
    with nuclear spin I₂=1/2. This interaction couples the F nuclear magnetic moment
    to the magnetic field produced by the unpaired electron.

    The operator is off-diagonal in both J (ΔJ = 0, ±1) and F₁ (ΔF₁ = 0, ±1) while
    preserving I₁, I₂, F, mF, and Omega. This double off-diagonality arises from
    the recoupling scheme where F₁ = J + I₁, then F = F₁ + I₂.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state, requires h1_F parameter
            (F magnetic hyperfine constant).

    Returns:
        CoupledState containing superposition over states with J' = J-1, J, J+1 and
        F₁' = F₁-1, F₁, F₁+1 (constrained by |J-I₁| ≤ F₁ ≤ J+I₁). Each term includes
        two Wigner 6-j symbols (for F₁ and F recoupling), one 3-j symbol (for J),
        phase factors, and angular momentum factors. Zero-amplitude terms are excluded.

    Notes:
        - Selection rules: ΔJ = 0, ±1; ΔF₁ = 0, ±1; ΔI₂ = 0; ΔF = 0; ΔmF = 0; ΔΩ = 0
        - For TlF B state: h1(F) ≈ 0.861 GHz (much smaller than h1(Tl))
        - ¹⁹F is the only stable fluorine isotope (100% abundance, I=1/2)
        - The smaller value reflects the larger distance of F nucleus from the
          unpaired electron compared to Tl
        - F₁ range: |J - I₁| to J + I₁ in steps of 1 (typically J-1/2 to J+1/2)

    References:
        Brown, J. M., & Carrington, A. (2003). "Rotational Spectroscopy of Diatomic
        Molecules." Cambridge University Press. Eq. 8.324 and following.

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(h1_F=861000000.0)
        >>> result = H_mhf_F(state, constants)
        >>> # Returns superposition with varying J and F1
    """
    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    # I1, I2, F and mF are the same for both states
    I1 = I1p
    I2 = I2p
    F = Fp
    mF = mFp

    # Omega also doesn't change
    Omega = Omegap

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix element can
    # be non-zero
    # Need J = Jp+1 ... |Jp-1|
    for J in np.arange(np.abs(Jp - 1), Jp + 2):
        # F1 can be J +/- 1/2
        for F1 in np.arange(np.abs(J - 1 / 2), J + 3 / 2):
            # Calculate matrix element
            amp = (
                Omega
                * constants.h1_F
                * (-1) ** (2 * F1p + F + 2 * J + I1 + I2 - Omega + 1)
                * sixj_f(I2, F1p, F, F1, I2, 1)
                * sixj_f(Jp, F1p, I1, F1, J, 1)
                * threej_f(J, 1, Jp, -Omega, 0, Omegap)
                * np.sqrt(
                    (2 * F1 + 1)
                    * (2 * F1p + 1)
                    * (2 * J + 1)
                    * (2 * Jp + 1)
                    * I2
                    * (I2 + 1)
                    * (2 * I2 + 1)
                )
            )

            basis_state = CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I1,
                I2,
                Omega=Omega,
                electronic_state=psi.electronic_state,
                P=psi.P,
            )
            if amp != 0:
                data.append((amp, basis_state))

    return CoupledState(data)
