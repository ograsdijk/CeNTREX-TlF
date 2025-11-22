"""Physical constants for TlF molecule.

This module defines the molecular constants for thallium fluoride (TlF) including
rotational, hyperfine, and Stark shift parameters for both the X (ground) and B (excited)
electronic states.

References:
    Rotational constants from "Microwave Spectral tables: Diatomic molecules"
    by Lovas & Tiemann (1974).
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "HamiltonianConstants",
    "XConstants",
    "BConstants",
    "Γ",
    "ED_XtB",
    "TlFNuclearSpins",
]

a0 = 0.529177210903e-10  # m

# Unit conversion constants
Debye = 3.333333333333333e-30  # Coulomb meter (C·m)
Debye_Hz_V_cm = 503411.7791722602  # Conversion factor: Hz/(V/cm)

# Rotational constants from "Microwave Spectral tables: Diatomic molecules"
# by Lovas & Tiemann (1974).
# Note: B_rot differs from the value given by Ramsey by about 30 MHz.
B_ϵ = 6.689873e9  # Hz
α = 45.0843e6  # Hz


@dataclass
class HamiltonianConstants:
    """Base class for molecular Hamiltonian constants.

    Attributes:
        B_rot: Rotational constant in Hz
    """

    B_rot: float


@dataclass(unsafe_hash=True)
class XConstants(HamiltonianConstants):
    """Constants for the X (ground) electronic state of TlF.

    Attributes:
        B_rot: Rotational constant (Hz)
        c1: Spin-rotation coupling constant (Hz)
        c2: Tensor spin-spin coupling constant (Hz)
        c3: Scalar spin-spin coupling constant (Hz)
        c4: Nuclear spin-rotation coupling constant (Hz)
        μ_J: Rotational magnetic moment (Hz/G)
        μ_Tl: Thallium nuclear magnetic moment (Hz/G)
        μ_F: Fluorine nuclear magnetic moment (Hz/G)
        D_TlF: Electric dipole moment in Hz/(V/cm) units
        D: Electric dipole moment in SI units (C·m)
    """

    B_rot: float = B_ϵ - α / 2  # Hz
    c1: float = 126030.0  # Hz
    c2: float = 17890.0  # Hz
    c3: float = 700.0  # Hz
    c4: float = -13300.0  # Hz
    μ_J: float = 35.0  # Hz/G
    μ_Tl: float = 1240.5  # Hz/G
    μ_F: float = 2003.63  # Hz/G
    D_TlF: float = 4.2282 * Debye_Hz_V_cm  # Hz/(V/cm)
    D: float = 4.2282 * Debye  # C·m


@dataclass(unsafe_hash=True)
class BConstants(HamiltonianConstants):
    """Constants for the B (excited) electronic state of TlF.

    All constants are in Hz unless otherwise noted.

    Attributes:
        B_rot: Rotational constant (Hz)
        D_rot: Centrifugal distortion constant (Hz)
        H_const: Higher-order centrifugal distortion constant (Hz)
        h1_Tl: Thallium hyperfine coupling constant (Hz)
        h1_F: Fluorine hyperfine coupling constant (Hz)
        q: Electric quadrupole coupling constant (Hz)
        c_Tl: Thallium spin-rotation coupling constant (Hz)
        c1p_Tl: Thallium tensor coupling constant (Hz)
        μ_B: Magnetic moment of B state (Hz/G)
        gL: Orbital g-factor (dimensionless)
        gS: Spin g-factor (dimensionless)
        μ_E: Electric dipole moment in Hz/(V/cm) units
        Γ: Natural linewidth (rad/s = 2π Hz)
    """

    B_rot: float = 6687.879e6  # Hz
    D_rot: float = 0.010869e6  # Hz
    H_const: float = -8.1e-2  # Hz
    h1_Tl: float = 28789e6  # Hz
    h1_F: float = 861e6  # Hz
    q: float = 2.423e6  # Hz
    c_Tl: float = -7.83e6  # Hz
    c1p_Tl: float = 11.17e6  # Hz
    μ_B: float = 1.4e6  # Hz/G
    gL: float = 1  # dimensionless
    gS: float = 2  # dimensionless
    μ_E: float = 2.28 * Debye_Hz_V_cm  # Hz/(V/cm)
    Γ: float = 2 * np.pi * 1.56e6  # rad/s


# Transition dipole moment from X to B state
ED_XtB = 0.80026518 * Debye  # C·m
EQ_XtB = ED_XtB * a0  # C·m^2


# Natural linewidth of B state (convenience constant)
Γ = 2 * np.pi * 1.56e6  # rad/s (same as BConstants.Γ)


@dataclass
class TlFNuclearSpins:
    """Nuclear spin quantum numbers for TlF isotopes.

    Attributes:
        I_F: Nuclear spin of ¹⁹F (fluorine-19)
        I_Tl: Nuclear spin of ²⁰⁵Tl (thallium-205) and ²⁰³Tl (thallium-203)

    Note:
        Both ²⁰⁵Tl and ²⁰³Tl have nuclear spin I = 1/2.
    """

    I_F: float = 1 / 2
    I_Tl: float = 1 / 2
