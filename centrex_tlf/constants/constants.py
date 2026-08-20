"""Physical constants and molecular constants for TlF.

This module defines molecular constants for thallium fluoride (TlF), including
rotational, hyperfine, Zeeman, and Stark parameters for the X (ground) and
B (excited) electronic states.

References:
    X-state rotational constants:
        NIST Diatomic Spectral Database, TlF table (Dunham coefficients Y_kl
        compiled from the spectroscopy of Lovas & Tiemann):
        https://physics.nist.gov/PhysRefData/MolSpec/Diatomic/Html/Tables/TlF.html

    X-state permanent dipole moment:
        NIST Diatomic Spectral Database, TlF table (vibrationally resolved
        mu_v values):
        https://physics.nist.gov/PhysRefData/MolSpec/Diatomic/Html/Tables/TlF.html

    X-state hyperfine constants (c1, c2, c4):
        Values used for the CeNTREX TlF X-state effective Hamiltonian, consistent
        with the NIST TlF table's C_Tl = 126.03(12) kHz, C_F = 17.89(15) kHz, and
        c4 = -13.30(72) kHz (Tl-F scalar spin-spin constant).

    X-state tensor spin-spin constant (c3):
        NIST tabulates the tensor spin-spin constant as 3.50(15) kHz, in a
        convention where the operator is normalized differently than the one
        implemented here. The CeNTREX Hamiltonian's Hc3/Hc3_alt operators
        correspond to the NIST value divided by 5, i.e. c3 = 0.70 kHz. This is
        a convention difference, not a discrepant measurement: do NOT "correct"
        c3 = 700 Hz to 3.5 kHz without also converting the operator normalization.

    X-state Zeeman constants (μ_J, μ_Tl, μ_F):
        Rotational and nuclear g-factors for the CeNTREX TlF X-state effective
        Hamiltonian (uncoupled-basis Zeeman terms HZx/HZy/HZz in
        X_uncoupled.py), consistent with values historically used for TlF
        Zeeman spectroscopy (see e.g. Ramsey, "Molecular Beams", Oxford
        University Press (1956), for the general formalism).

    B-state constants:
        Meijer & Sartakov, arXiv:1911.10734, Table I, based on a refit of the
        spectroscopy in Norrgard et al., Phys. Rev. A 95, 062506 (2017),
        arXiv:1702.02548. The sign of H follows the original Norrgard convention.

    B-state lifetime and decay rate Γ:
        Hunter et al., Phys. Rev. A 85, 012511 (2012), arXiv:1110.3748:
        measured B³Π₁(v'=0) lifetime τ = 99(9) ns. The population decay rate
        is Γ = 1/τ (units s⁻¹); the equivalent spectroscopic linewidth is
        Γ/(2π) ≈ 1.608 MHz.

    Fundamental physical constants:
        scipy.constants (CODATA data distributed with the installed SciPy version).
"""

from dataclasses import dataclass

import scipy.constants as cst

__all__ = [
    "B0_X",
    "B_LIFETIME",
    "Bohr_magneton_Hz_G",
    "D0_X",
    "Debye",
    "Debye_Hz_V_cm",
    "MU_X_E_D",
    "MU_X_V0_D",
    "MU_X_V1_D",
    "MU_X_V2_D",
    "Y01_X",
    "Y02_X",
    "Y11_X",
    "Y21_X",
    "Γ",
    "BConstants",
    "ED_XtB",
    "HamiltonianConstants",
    "TlFNuclearSpins",
    "XConstants",
    "a0",
]


# ---------------------------------------------------------------------------
# Fundamental physical constants and unit conversions
# ---------------------------------------------------------------------------

# Bohr radius.
a0 = cst.value("Bohr radius")  # m

# Debye in SI units.
#
# The debye is defined by
#     1 D = 10^-18 statC cm = 10^-21 / c C m,
# where c is the speed of light in m/s.
Debye = 1e-21 / cst.c  # C·m

# Conversion from dipole moment in debye and electric field in V/cm to
# Stark interaction frequency in Hz:
#
#     (1 D)(1 V/cm) / h.
Debye_Hz_V_cm = Debye * 100 / cst.h  # Hz/(V/cm)

# Bohr magneton in frequency units.
Bohr_magneton_Hz_G = cst.value("Bohr magneton in Hz/T") * 1e-4  # Hz/G


# ---------------------------------------------------------------------------
# X-state rotational constants
# ---------------------------------------------------------------------------

# X-state Dunham coefficients Y_kl for 205Tl19F, from the NIST Diatomic
# Spectral Database TlF table:
#     https://physics.nist.gov/PhysRefData/MolSpec/Diatomic/Html/Tables/TlF.html
# based on the spectroscopy compiled by Lovas & Tiemann, "Microwave Spectral
# Tables I. Diatomic Molecules" (1974).
#
# The Dunham expansion gives the rovibrational term energy as
#
#     E(v, J) / h = Σ_{k,l} Y_kl (v + 1/2)^k [J(J + 1)]^l
#
# Stored explicitly (rather than only the derived v=0 constants below) since
# these coefficients may later be used to add vibrationally excited states to
# CeNTREX-TlF.
Y01_X = 6689.8736e6  # Hz, Y_01 (rotational constant B_e)
Y11_X = -45.0843e6  # Hz, Y_11 (vibration-rotation coupling, alpha_e)
Y21_X = 0.0942e6  # Hz, Y_21 (second-order vibration-rotation coupling, gamma_e)
Y02_X = -5.84e3  # Hz, Y_02 (centrifugal distortion, -D_e)

# v=0 rotational constants derived from the Dunham coefficients above:
#
#     B_0 = Y01 + Y11/2 + Y21/4
#     D_0 = -Y02
#
# Do not hard-code these separately from the Dunham coefficients; they must
# stay derived so that a future vibrational-state extension only needs to
# update Y_kl.
B0_X = Y01_X + Y11_X / 2 + Y21_X / 4  # Hz; ~6.667355e9
D0_X = -Y02_X  # Hz; ~5.84e3


# ---------------------------------------------------------------------------
# X-state permanent electric dipole moment (vibrational dependence)
# ---------------------------------------------------------------------------

# Measured X-state body-fixed permanent electric dipole moment, resolved by
# vibrational level v, from the NIST Diatomic Spectral Database TlF table:
#     https://physics.nist.gov/PhysRefData/MolSpec/Diatomic/Html/Tables/TlF.html
#
# Reported values and uncertainties (debye):
#     mu_e = 4.1939(10) D   (equilibrium value)
#     mu_0 = 4.2282(8)  D   (v=0)
#     mu_1 = 4.2972(10) D   (v=1)
#     mu_2 = 4.3665(11) D   (v=2)
#
# Stored explicitly, even though only v=0 is used by the current X-state
# Hamiltonian, so the source data is available if vibrational states are
# added to CeNTREX-TlF later. Do not add vibrational states or a
# vibrational Hamiltonian based on this alone.
MU_X_E_D = 4.1939  # D, equilibrium dipole moment mu_e
MU_X_V0_D = 4.2282  # D, v=0 dipole moment mu_0
MU_X_V1_D = 4.2972  # D, v=1 dipole moment mu_1
MU_X_V2_D = 4.3665  # D, v=2 dipole moment mu_2


# ---------------------------------------------------------------------------
# B-state lifetime and decay rate
# ---------------------------------------------------------------------------

# Hunter et al., Phys. Rev. A 85, 012511 (2012), arXiv:1110.3748:
# B³Π₁(v'=0) lifetime τ = 99(9) ns.
B_LIFETIME = 99e-9  # s

# The population decay rate is derived directly from the measured lifetime:
#
#     Γ = 1 / B_LIFETIME
#
# Γ has units s⁻¹ (it is a rate, not a plain frequency in Hz); the equivalent
# spectroscopic linewidth is Γ/(2π) ≈ 1.608 MHz. Do not hard-code that
# linewidth separately — it must stay derived from B_LIFETIME.
#
# Set as the default of `BConstants.Γ` below; the module-level `Γ` alias
# (kept for backward compatibility with `from centrex_tlf.constants import
# Γ`) is assigned right after the class.


@dataclass
class HamiltonianConstants:
    """Base class for molecular Hamiltonian constants.

    Both the X and B electronic states are modeled with a quartic
    centrifugal-distortion correction to the rigid rotor, so `D_rot` is part
    of the common interface rather than an optional/duck-typed attribute:
    the rotational Hamiltonian is

        H_rot / h = B_rot * J(J+1) - D_rot * [J(J+1)]^2

    Attributes:
        B_rot: Rotational constant in Hz.
        D_rot: Quartic centrifugal-distortion constant in Hz.
    """

    B_rot: float
    D_rot: float


@dataclass(unsafe_hash=True)
class XConstants(HamiltonianConstants):
    """Constants for the X¹Σ⁺ ground electronic state of TlF.

    Attributes:
        B_rot:
            v=0 rotational constant B_0 (Hz), derived from the Dunham
            coefficients as B0_X = Y01_X + Y11_X/2 + Y21_X/4.

        D_rot:
            Quartic centrifugal-distortion constant for X(v=0) (Hz), derived
            from the Dunham coefficients as D0_X = -Y02_X. The rotational
            Hamiltonian uses B_rot J(J+1) - D_rot [J(J+1)]^2.

        c1:
            Tl nuclear spin-rotation coupling constant (Hz),
            multiplying I_Tl · J. NIST reports C_Tl = 126.03(12) kHz.

        c2:
            F nuclear spin-rotation coupling constant (Hz),
            multiplying I_F · J. NIST reports C_F = 17.89(15) kHz.

        c3:
            Tensor nuclear spin-spin coupling constant (Hz), in the operator
            normalization used by the X-state Hamiltonian implemented here
            (`Hc3`/`Hc3_alt`). NIST tabulates the tensor spin-spin constant
            as 3.50(15) kHz in a *different* operator normalization; the
            value used here, c3 = 0.70 kHz, is the NIST value divided by 5
            under this package's convention. This is a convention
            difference, not a discrepant measurement — do not "correct"
            700 Hz to 3.5 kHz without also converting the normalization.

        c4:
            Scalar nuclear spin-spin coupling constant (Hz),
            multiplying I_Tl · I_F. NIST reports c4 = -13.30(72) kHz.

        μ_J:
            Rotational magnetic moment in frequency units (Hz/G), used in the
            uncoupled-basis Zeeman terms (HZx/HZy/HZz in X_uncoupled.py).

        μ_Tl:
            ²⁰⁵Tl nuclear magnetic moment in frequency units (Hz/G), used in
            the uncoupled-basis Zeeman terms.

        μ_F:
            ¹⁹F nuclear magnetic moment in frequency units (Hz/G), used in
            the uncoupled-basis Zeeman terms.

        D_TlF:
            Body-fixed permanent electric dipole moment expressed as a Stark
            frequency per electric field, in Hz/(V/cm). Uses the v=0 value
            MU_X_V0_D = 4.2282(8) D; see the X-state dipole moment section
            above for the vibrationally resolved NIST data.

        D:
            Body-fixed permanent electric dipole moment in SI units (C·m),
            also from MU_X_V0_D.
    """

    B_rot: float = B0_X  # Hz; v=0 rotational constant
    D_rot: float = D0_X  # Hz; v=0 quartic centrifugal distortion

    # Hyperfine constants. See the "X-state hyperfine constants" and
    # "X-state tensor spin-spin constant" references above for provenance
    # and, for c3, the operator-normalization convention used here.
    c1: float = 126030.0  # Hz, Tl nuclear spin-rotation; NIST C_Tl = 126.03(12) kHz
    c2: float = 17890.0  # Hz, F nuclear spin-rotation; NIST C_F = 17.89(15) kHz
    c3: float = 700.0  # Hz, tensor nuclear spin-spin; NIST 3.50(15) kHz / 5 (see above)
    c4: float = -13300.0  # Hz, scalar nuclear spin-spin; NIST c4 = -13.30(72) kHz

    # Zeeman constants. See the "X-state Zeeman constants" reference above.
    μ_J: float = 35.0  # Hz/G
    μ_Tl: float = 1240.5  # Hz/G
    μ_F: float = 2003.63  # Hz/G

    # Permanent electric dipole moment. Uses the v=0 value MU_X_V0_D; see the
    # X-state dipole moment section above for the vibrationally resolved data.
    D_TlF: float = MU_X_V0_D * Debye_Hz_V_cm  # Hz/(V/cm)
    D: float = MU_X_V0_D * Debye  # C·m


@dataclass(unsafe_hash=True)
class BConstants(HamiltonianConstants):
    """Constants for the B³Π₁ excited electronic state of TlF.

    Spectroscopic constants are in Hz unless otherwise noted.

    Attributes:
        B_rot:
            Rotational constant B (Hz).

        D_rot:
            Quartic centrifugal-distortion constant D (Hz). The rotational
            Hamiltonian uses B J² - D J⁴ + H J⁶.

        H_const:
            Sextic centrifugal-distortion constant H (Hz). The negative sign
            follows Norrgard et al.'s original fitted value and Hamiltonian
            convention.

        h1_Tl:
            Tl magnetic hyperfine constant h1(Tl) (Hz).

        h1_F:
            F magnetic hyperfine constant h1(F) (Hz).

        q:
            Λ-doubling parameter q (Hz). This is not an electric-quadrupole
            interaction: both ²⁰⁵Tl and ¹⁹F have I=1/2 and therefore no nuclear
            electric quadrupole moment.

        c_Tl:
            Tl nuclear spin-rotation constant C_I(Tl) (Hz).

        c1p_Tl:
            Λ-doubling contribution C'_I(Tl) to the Tl nuclear spin-rotation
            interaction (Hz).

        μ_B:
            Bohr magneton divided by h, expressed in Hz/G. This is the magnetic
            moment unit entering the electronic Zeeman Hamiltonian, not a fitted
            B-state magnetic moment.

        gL:
            Orbital electronic g-factor used by the model (dimensionless).

        gS:
            Spin electronic g-factor used by the uncoupled-basis model
            (dimensionless). The value 2 is the idealized electronic-spin value
            used by this Hamiltonian rather than the CODATA free-electron g-factor.

        μ_E:
            Body-fixed permanent electric dipole moment of the B state expressed
            as a Stark frequency per electric field, in Hz/(V/cm).

        Γ:
            Excited-state population decay rate (s⁻¹), derived directly from
            the measured B-state lifetime as Γ = 1/B_LIFETIME (τ = 99(9) ns,
            Hunter et al., Phys. Rev. A 85, 012511 (2012)). The equivalent
            spectroscopic linewidth is Γ/(2π) ≈ 1.608 MHz. See the "B-state
            lifetime and decay rate" reference above.
    """

    # Rotation
    B_rot: float = 6687.879e6  # Hz
    D_rot: float = 0.010869e6  # Hz
    H_const: float = -8.1e-2  # Hz

    # Hyperfine and Λ-doubling
    h1_Tl: float = 28789e6  # Hz
    h1_F: float = 861e6  # Hz
    q: float = 2.423e6  # Hz
    c_Tl: float = -7.83e6  # Hz, C_I(Tl)
    c1p_Tl: float = 11.17e6  # Hz, C'_I(Tl)

    # Zeeman
    μ_B: float = Bohr_magneton_Hz_G  # Hz/G; μ_B / h
    gL: float = 1.0  # dimensionless
    gS: float = 2.0  # dimensionless

    # Permanent electric dipole moment
    μ_E: float = 2.28 * Debye_Hz_V_cm  # Hz/(V/cm)

    # Spontaneous decay
    Γ: float = 1 / B_LIFETIME  # s⁻¹; Γ/(2π) ≈ 1.608 MHz


# Convenience module-level decay rate (s⁻¹), for backward compatibility with
# `from centrex_tlf.constants import Γ`. Kept in sync with `BConstants.Γ` by
# reading it off the class rather than duplicating the derivation.
Γ = BConstants.Γ


# ---------------------------------------------------------------------------
# X ↔ B transition moments
# ---------------------------------------------------------------------------

# Electric-dipole transition moment used for the X ↔ B optical transition.
#
# X ↔ B electronic transition dipole moment inferred from the measured
# B-state lifetime: 0.315(14) a.u. ≈ 0.80065(356) D.
ED_XtB = 0.315 * cst.e * a0  # C·m


@dataclass
class TlFNuclearSpins:
    """Nuclear spin quantum numbers for the TlF isotopologues used here.

    Attributes:
        I_F:
            Nuclear spin of ¹⁹F.

        I_Tl:
            Nuclear spin of ²⁰⁵Tl or ²⁰³Tl.

    Notes:
        ¹⁹F, ²⁰⁵Tl, and ²⁰³Tl all have nuclear spin I = 1/2.
    """

    I_F: float = 1 / 2
    I_Tl: float = 1 / 2
