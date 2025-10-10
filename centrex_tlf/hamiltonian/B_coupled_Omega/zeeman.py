"""Zeeman effect Hamiltonian operators for B state in coupled Omega basis.

This module implements the interaction between molecular magnetic moment and external
magnetic fields (Zeeman effect). The electronic magnetic moment in the B state
(²Π₁/₂) couples to magnetic fields through the orbital angular momentum projection Ω.

The Zeeman interaction is important for understanding magnetic field effects on
molecular energy levels and for experiments using magnetic fields for state manipulation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from centrex_tlf.constants import BConstants
from centrex_tlf.states import CoupledBasisState, CoupledState

from ..wigner import sixj_f, threej_f


@lru_cache(maxsize=int(1e6))
def mu_p(psi: CoupledBasisState, p: int, constants: BConstants) -> CoupledState:
    """Magnetic dipole operator (pth spherical tensor component) for B state.

    Applies the pth component (p = -1, 0, +1) of the magnetic dipole operator in
    spherical tensor form. For molecules in ²Π states, the electronic magnetic moment
    is primarily due to orbital angular momentum, with g_L ≈ 1.

    The operator couples rotational states through the electronic magnetic moment,
    which is proportional to the projection Ω of electronic angular momentum along
    the internuclear axis. This Omega-dependence is characteristic of molecules
    with non-zero electronic angular momentum.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        p: Spherical tensor component index. Must be -1, 0, or +1:
            - p = +1: σ⁺ component (increases mF by 1)
            - p =  0: π component (preserves mF)
            - p = -1: σ⁻ component (decreases mF by 1)
        constants: Molecular constants for TlF B state, requires μ_B parameter
            (Bohr magneton in Hz/Gauss, ≈1.4 MHz/Gauss).

    Returns:
        CoupledState containing superposition over all allowed final states with
        J' = J-1, J, J+1; F₁' varying according to |J'-I₁| ≤ F₁' ≤ J'+I₁;
        F' varying according to |F₁'-I₂| ≤ F' ≤ F₁'+I₂; and mF' = mF + p.
        Zero-amplitude terms are excluded.

    Notes:
        - Selection rules: ΔJ = 0, ±1; ΔF₁ = 0, ±1; ΔF = 0, ±1; ΔmF = p; ΔΩ = 0
        - g_L = 1 (orbital g-factor) appropriate for ²Π₁/₂ state
        - Amplitude proportional to Ω (vanishes for Σ states with Ω=0)
        - For TlF B state: μ_B ≈ 1.4 MHz/Gauss
        - Spin contribution neglected (g_S terms small for ²Π₁/₂)
        - Matrix elements similar to electric dipole but with magnetic coupling

    References:
        Brown, J. M., & Carrington, A. (2003). "Rotational Spectroscopy of Diatomic
        Molecules." Cambridge University Press. Chapter 8 on Zeeman effect.
        Ramsey, N. F. (1956). "Molecular Beams." Oxford University Press.

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_B=1400000.0)
        >>> result_pi = mu_p(state, 0, constants)      # π component
        >>> result_sp = mu_p(state, +1, constants)     # σ⁺ component
        >>> result_sm = mu_p(state, -1, constants)     # σ⁻ component
    """
    # Electronic orbital g-factor (appropriate for ²Π₁/₂ state)
    gL = 1.0

    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    # I1, I2 are the same for both states
    I1 = I1p
    I2 = I2p

    # Value of mF changes by p
    mF = mFp + p

    # Omega doesn't change
    Omega = Omegap

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over possible values of Jprime
    for J in np.arange(np.abs(Jp - 1), Jp + 2):
        # Loop over possible values of F1
        for F1 in np.arange(np.abs(J - I1), J + I1 + 1):
            # Loop over possible values of F
            for F in np.arange(np.abs(F1 - I2), F1 + I2 + 1):
                amp = (
                    gL
                    * Omega
                    * constants.μ_B
                    * (-1) ** (F + Fp + F1 + F1p + I1 + I2 - Omega - mF)
                    * np.sqrt(
                        (2 * F + 1)
                        * (2 * Fp + 1)
                        * (2 * F1 + 1)
                        * (2 * F1p + 1)
                        * (2 * J + 1)
                        * (2 * Jp + 1)
                    )
                    * threej_f(F, 1, Fp, -mF, p, mFp)
                    * threej_f(J, 1, Jp, -Omega, 0, Omegap)
                    * sixj_f(F1p, Fp, I2, F, F1, 1)
                    * sixj_f(Jp, F1p, I1, F1, J, 1)
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
def HZx(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Zeeman Hamiltonian for magnetic field along x-axis (laboratory frame).

    Calculates the interaction energy with a magnetic field pointing along the
    laboratory x-direction. Constructed from spherical tensor components as:
        H_Zx = -(μ₋₁ - μ₊₁)/√2

    This operator mixes states with ΔmF = ±1, corresponding to transverse magnetic
    field interactions. Important for experiments with rotating or transverse fields.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state (requires μ_B).

    Returns:
        CoupledState containing superposition of states mixed by x-component of
        magnetic field.

    Notes:
        - Couples states with ΔmF = ±1
        - For quantization along z, this represents transverse field
        - The factor 1/√2 normalizes the Cartesian-to-spherical transformation
        - Used in rotating frame calculations and RF/microwave transitions

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_B=1400000.0)
        >>> result = HZx(state, constants)
    """
    return -(mu_p(psi, -1, constants) - mu_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HZy(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Zeeman Hamiltonian for magnetic field along y-axis (laboratory frame).

    Calculates the interaction energy with a magnetic field pointing along the
    laboratory y-direction. Constructed from spherical tensor components as:
        H_Zy = -i(μ₋₁ + μ₊₁)/√2

    This operator mixes states with ΔmF = ±1, corresponding to transverse magnetic
    field interactions. The factor of i reflects the phase relationship between
    Cartesian and spherical components.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state (requires μ_B).

    Returns:
        CoupledState containing superposition of states mixed by y-component of
        magnetic field. The amplitudes are complex due to the phase factor i.

    Notes:
        - Couples states with ΔmF = ±1
        - Returns complex amplitudes (imaginary factor i)
        - For quantization along z, this represents transverse field
        - The factor i and 1/√2 normalize the Cartesian-to-spherical transformation
        - Used in rotating frame calculations and RF/microwave transitions

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_B=1400000.0)
        >>> result = HZy(state, constants)
        >>> # Result has complex amplitudes
    """
    return -1j * (mu_p(psi, -1, constants) + mu_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HZz(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Zeeman Hamiltonian for magnetic field along z-axis (quantization axis).

    Calculates the interaction energy with a magnetic field pointing along the
    laboratory z-direction (quantization axis). This is simply the μ₀ component:
        H_Zz = -μ₀

    This operator preserves mF (ΔmF = 0), corresponding to the energy shift in
    a longitudinal magnetic field. This is the most commonly used component for
    Zeeman shift calculations in aligned DC magnetic fields.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state (requires μ_B).

    Returns:
        CoupledState containing superposition of states mixed by z-component of
        magnetic field. All coupled states have the same mF as input.

    Notes:
        - Preserves mF (ΔmF = 0)
        - For quantization along z, this represents longitudinal field
        - Creates linear Zeeman shift: E_Zeeman = -μ_z·B_z
        - Most important component for DC magnetic field experiments
        - Energy shifts proportional to mF for weak fields
        - Used for magnetic substate selection and manipulation

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_B=1400000.0)
        >>> result = HZz(state, constants)
        >>> # All coupled states have mF=0
    """
    return -mu_p(psi, 0, constants)
