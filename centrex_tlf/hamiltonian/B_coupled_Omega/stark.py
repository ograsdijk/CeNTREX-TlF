"""Stark effect Hamiltonian operators for B state in coupled Omega basis.

This module implements the interaction between the permanent electric dipole moment
of TlF and external electric fields. The Stark effect causes energy shifts and mixing
of rotational states in the presence of electric fields.

The operators are implemented using spherical tensor components of the electric dipole
operator, which couple states with different J, F₁, F, and mF quantum numbers.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from centrex_tlf.constants import BConstants
from centrex_tlf.states import CoupledBasisState, CoupledState

from ..wigner import sixj_f, threej_f


@lru_cache(maxsize=int(1e6))
def d_p(psi: CoupledBasisState, p: int, constants: BConstants) -> CoupledState:
    """Electric dipole operator (pth spherical tensor component) for B state.

    Applies the pth component (p = -1, 0, +1) of the electric dipole operator in
    spherical tensor form. This operator couples rotational states through the
    permanent electric dipole moment of TlF.

    The operator is off-diagonal in J (ΔJ = 0, ±1), F₁ (ΔF₁ = 0, ±1), F (ΔF = 0, ±1),
    and mF (ΔmF = p), while Omega and nuclear spins I₁, I₂ are conserved. The matrix
    elements involve Wigner 3-j and 6-j symbols for angular momentum coupling.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        p: Spherical tensor component index. Must be -1, 0, or +1:
            - p = +1: σ⁺ polarization (increases mF by 1)
            - p =  0: π polarization (preserves mF)
            - p = -1: σ⁻ polarization (decreases mF by 1)
        constants: Molecular constants for TlF B state, requires μ_E parameter
            (electric dipole moment in Hz/(V/cm)).

    Returns:
        CoupledState containing superposition over all allowed final states with
        J' = J-1, J, J+1; F₁' varying according to |J'-I₁| ≤ F₁' ≤ J'+I₁;
        F' varying according to |F₁'-I₂| ≤ F' ≤ F₁'+I₂; and mF' = mF + p.
        Zero-amplitude terms are excluded.

    Notes:
        - Selection rules: ΔJ = 0, ±1; ΔF₁ = 0, ±1; ΔF = 0, ±1; ΔmF = p; ΔΩ = 0
        - For TlF B state: μ_E ≈ 1.15 MHz/(V/cm) = 4.22 Debye
        - The operator conserves parity for ΔJ = ±1, changes parity for ΔJ = 0
        - Matrix elements scale as √[J(J+1)] for transitions between adjacent J

    References:
        Zare, R. N. (1988). "Angular Momentum." Wiley, New York.
        Brown, J. M., & Carrington, A. (2003). "Rotational Spectroscopy of Diatomic
        Molecules." Cambridge University Press. Section 9.5.

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_E=1147778.856512753)
        >>> result_pi = d_p(state, 0, constants)      # π polarization
        >>> result_sp = d_p(state, +1, constants)     # σ⁺ polarization
        >>> result_sm = d_p(state, -1, constants)     # σ⁻ polarization
    """
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
                    constants.μ_E
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
def HSx(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Stark Hamiltonian for electric field along x-axis (laboratory frame).

    Calculates the interaction energy with an electric field pointing along the
    laboratory x-direction. Constructed from spherical tensor components as:
        H_Sx = -(d₋₁ - d₊₁)/√2

    This operator mixes states with ΔmF = ±1, corresponding to the selection rules
    for x-polarized light or transverse electric fields.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state (requires μ_E).

    Returns:
        CoupledState containing superposition of states mixed by x-polarized
        electric field interaction.

    Notes:
        - Couples states with ΔmF = ±1
        - For quantization along z, this represents transverse field interaction
        - The factor 1/√2 normalizes the Cartesian-to-spherical transformation

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_E=1147778.856512753)
        >>> result = HSx(state, constants)
    """
    return -(d_p(psi, -1, constants) - d_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HSy(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Stark Hamiltonian for electric field along y-axis (laboratory frame).

    Calculates the interaction energy with an electric field pointing along the
    laboratory y-direction. Constructed from spherical tensor components as:
        H_Sy = -i(d₋₁ + d₊₁)/√2

    This operator mixes states with ΔmF = ±1, corresponding to the selection rules
    for y-polarized light or transverse electric fields. The factor of i reflects
    the phase relationship between Cartesian and spherical components.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state (requires μ_E).

    Returns:
        CoupledState containing superposition of states mixed by y-polarized
        electric field interaction. The amplitudes are complex due to the
        phase factor i.

    Notes:
        - Couples states with ΔmF = ±1
        - Returns complex amplitudes (imaginary factor i)
        - For quantization along z, this represents transverse field interaction
        - The factor i and 1/√2 normalize the Cartesian-to-spherical transformation

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_E=1147778.856512753)
        >>> result = HSy(state, constants)
        >>> # Result has complex amplitudes
    """
    return -1j * (d_p(psi, -1, constants) + d_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HSz(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Stark Hamiltonian for electric field along z-axis (quantization axis).

    Calculates the interaction energy with an electric field pointing along the
    laboratory z-direction (quantization axis). This is simply the d₀ component:
        H_Sz = -d₀

    This operator preserves mF (ΔmF = 0), corresponding to π-polarized transitions
    or longitudinal electric fields. This is the most commonly used component for
    Stark shift calculations in aligned experiments.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state (requires μ_E).

    Returns:
        CoupledState containing superposition of states mixed by z-polarized
        electric field interaction. All coupled states have the same mF as input.

    Notes:
        - Preserves mF (ΔmF = 0)
        - For quantization along z, this represents longitudinal field interaction
        - Most important component for DC Stark effect in aligned fields
        - Creates quadratic Stark shift for non-degenerate states
        - First-order Stark effect requires linear superposition of opposite parity

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(μ_E=1147778.856512753)
        >>> result = HSz(state, constants)
        >>> # All coupled states have mF=0
    """
    return -d_p(psi, 0, constants)
