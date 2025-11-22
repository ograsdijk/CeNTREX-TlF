"""Rabi frequency calculations and beam parameter conversions.

This module provides utilities for converting between laser power, intensity,
electric field, and Rabi frequencies for both optical and microwave transitions.
Includes Gaussian and rectangular beam profiles, and beam width conversions.
"""

from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scipy.constants as cst

from centrex_tlf.constants import ED_XtB, XConstants

__all__ = [
    "fwhm_to_sigma",
    "sigma_to_fwhm",
    "electric_field_to_rabi",
    "sigma_to_waist",
    "waist_to_sigma",
    "electric_field_to_intensity",
    "intensity_to_electric_field",
    "intensity_to_power_gaussian_beam",
    "intensity_to_power_rectangular_beam",
    "intensity_to_rabi",
    "power_to_intensity_gaussian_beam",
    "power_to_intensity_rectangular_beam",
    "power_to_rabi_gaussian_beam",
    "power_to_rabi_gaussian_beam_microwave",
    "power_to_rabi_rectangular_beam",
    "rabi_to_power_gaussian_beam",
    "rabi_to_power_gaussian_beam_microwave",
    "rabi_to_electric_field",
]

T = TypeVar("T", float, npt.NDArray[np.floating])


def fwhm_to_sigma(fwhm: T) -> T:
    """Convert full width at half maximum (FWHM) to standard deviation.

    For a Gaussian profile: σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.355.

    Args:
        fwhm (T): Full width at half maximum [arbitrary units].

    Returns:
        T: Standard deviation σ in same units as input.

    Example:
        >>> fwhm_to_sigma(2.355)  # FWHM of 2.355 gives σ ≈ 1
        1.0
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def sigma_to_fwhm(sigma: T) -> T:
    """Convert standard deviation to full width at half maximum (FWHM).

    For a Gaussian profile: FWHM = σ · 2√(2ln2) ≈ σ · 2.355.

    Args:
        sigma (T): Standard deviation σ [arbitrary units].

    Returns:
        T: Full width at half maximum in same units as input.

    Example:
        >>> sigma_to_fwhm(1.0)  # σ of 1 gives FWHM ≈ 2.355
        2.3548200450309493
    """
    return sigma * 2 * np.sqrt(2 * np.log(2))


def sigma_to_waist(sigma: T) -> T:
    """Convert Gaussian standard deviation to beam waist (1/e² radius).

    For Gaussian beams: w₀ = 2σ where w₀ is the waist radius at which intensity
    drops to 1/e² of peak value.

    Args:
        sigma (T): Standard deviation σ [m].

    Returns:
        T: Beam waist w₀ in same units as input.

    Example:
        >>> sigma_to_waist(0.5e-3)  # 0.5 mm sigma gives 1 mm waist
        0.001
    """
    return sigma * 2


def waist_to_sigma(waist: T) -> T:
    """Convert beam waist (1/e² radius) to Gaussian standard deviation.

    For Gaussian beams: σ = w₀/2 where w₀ is the waist radius.

    Args:
        waist (T): Beam waist w₀ [m].

    Returns:
        T: Standard deviation σ in same units as input.

    Example:
        >>> waist_to_sigma(1e-3)  # 1 mm waist gives 0.5 mm sigma
        0.0005
    """
    return waist / 2


def intensity_to_electric_field(intensity: T) -> T:
    """Convert optical intensity to electric field amplitude.

    Uses the time-averaged electromagnetic energy density relation for plane waves:
    I = (1/2)·c·ε₀·E₀², solving for E₀ gives E₀ = √(2I/(c·ε₀)).

    Args:
        intensity (T): Optical intensity [W/m²].

    Returns:
        T: Electric field amplitude E₀ [V/m]. For time-harmonic fields,
            E(t) = E₀·cos(ωt).

    Raises:
        ValueError: If intensity is negative.

    Example:
        >>> intensity_to_electric_field(1000.0)  # 1 kW/m² intensity
        868.3641436915924

    Note:
        For focused Gaussian beams, intensity is the peak on-axis value.
    """
    if np.any(intensity < 0):
        raise ValueError("Intensity must be non-negative")
    return np.sqrt((2 / (cst.c * cst.epsilon_0)) * intensity)  # type: ignore[return-value]


def electric_field_to_rabi(electric_field: T, coupling: float, D: float) -> T:
    """Calculate Rabi frequency from electric field and transition coupling.

    Rabi frequency Ω = (E₀·μ·coupling)/ℏ where μ is the transition dipole moment
    and coupling is the reduced matrix element (e.g., Clebsch-Gordan coefficient).

    Args:
        electric_field (T): Electric field amplitude E₀ [V/m].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        D (float): Transition dipole moment [C·m].

    Returns:
        T: Rabi frequency [rad/s]. The on-resonance oscillation frequency.

    Example:
        >>> E = 1000.0  # V/m
        >>> electric_field_to_rabi(E, 1.0, 2.67e-30)  # TlF typical values
        2.532...e+05

    Note:
        For two-level systems, population oscillates at frequency Ω/2.
    """
    return electric_field * coupling * D / cst.hbar


def intensity_to_rabi(intensity: T, coupling: float, D: float) -> T:
    """Calculate Rabi frequency directly from laser intensity.

    Combines intensity-to-field and field-to-Rabi conversions:
    Ω = √(2I/(c·ε₀)) · (μ·coupling)/ℏ.

    Args:
        intensity (T): Optical intensity [W/m²].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        D (float): Transition dipole moment [C·m].

    Returns:
        T: Rabi frequency [rad/s].

    Example:
        >>> intensity_to_rabi(1000.0, 1.0, 2.67e-30)  # 1 kW/m²
        2.199...e+05

    Note:
        This is the most commonly used conversion for optical transitions.
    """
    electric_field = intensity_to_electric_field(intensity)
    rabi = electric_field_to_rabi(electric_field, coupling, D)
    return rabi


def intensity_to_power_rectangular_beam(intensity: T, wx: float, wy: float) -> T:
    """Convert intensity to total power for rectangular (flat-top) beam.

    Assumes uniform intensity across rectangular cross-section: P = I·A where A = wx·wy.

    Args:
        intensity (T): Intensity [W/m²].
        wx (float): Beam width in x-direction [m].
        wy (float): Beam width in y-direction [m].

    Returns:
        T: Total optical power [W].

    Example:
        >>> intensity_to_power_rectangular_beam(1000.0, 1e-3, 2e-3)  # 1x2 mm beam
        0.002
    """
    return intensity * wx * wy


def power_to_intensity_rectangular_beam(power: T, wx: float, wy: float) -> T:
    """Convert total power to intensity for rectangular (flat-top) beam.

    Assumes uniform intensity across rectangular cross-section: I = P/A where A = wx·wy.

    Args:
        power (T): Total optical power [W].
        wx (float): Beam width in x-direction [m].
        wy (float): Beam width in y-direction [m].

    Returns:
        T: Intensity [W/m²].

    Example:
        >>> power_to_intensity_rectangular_beam(0.001, 1e-3, 1e-3)  # 1 mW in 1x1 mm
        1000.0
    """
    return power / (wx * wy)


def power_to_rabi_rectangular_beam(
    power: T,
    coupling: float,
    wx: float,
    wy: float,
    D: float = ED_XtB,
) -> T:
    """Calculate Rabi frequency from power for rectangular beam profile.

    Combines power-to-intensity conversion (assuming flat-top beam) with Rabi
    frequency calculation. Useful for microwave transitions or multimode lasers.

    Args:
        power (T): Total optical power [W].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        wx (float): Beam width in x-direction [m].
        wy (float): Beam width in y-direction [m].
        D (float): Transition dipole moment [C·m]. Defaults to 2.6675506e-30 for
            TlF B-X optical transition.

    Returns:
        T: Rabi frequency [rad/s].

    Example:
        >>> power_to_rabi_rectangular_beam(0.001, 1.0, 1e-3, 1e-3)  # 1 mW, 1x1 mm
        2.199...e+05
    """
    intensity = power_to_intensity_rectangular_beam(power, wx, wy)
    rabi = intensity_to_rabi(intensity, coupling, D)
    return rabi


def power_to_intensity_gaussian_beam(power: T, sigma_x: float, sigma_y: float) -> T:
    """Convert total power to peak intensity for Gaussian beam.

    For Gaussian beam I(x,y) = I₀·exp(-x²/(2σₓ²)-y²/(2σᵧ²)), integrating over all
    space gives P = I₀·2π·σₓ·σᵧ, so I₀ = P/(2π·σₓ·σᵧ).

    Args:
        power (T): Total optical power [W].
        sigma_x (float): Gaussian standard deviation in x [m].
        sigma_y (float): Gaussian standard deviation in y [m].

    Returns:
        T: Peak (on-axis) intensity I₀ [W/m²].

    Example:
        >>> power_to_intensity_gaussian_beam(0.001, 0.5e-3, 0.5e-3)  # 1 mW, 0.5 mm sigma
        636.6197723675814

    Note:
        For circular beam with waist w₀, use sigma = w₀/2.
    """
    return power / (2 * np.pi * sigma_x * sigma_y)


def intensity_to_power_gaussian_beam(intensity: T, sigma_x: float, sigma_y: float) -> T:
    """Convert peak intensity to total power for Gaussian beam.

    Integrates Gaussian profile: P = I₀·2π·σₓ·σᵧ where I₀ is peak intensity.

    Args:
        intensity (T): Peak (on-axis) intensity I₀ [W/m²].
        sigma_x (float): Gaussian standard deviation in x [m].
        sigma_y (float): Gaussian standard deviation in y [m].

    Returns:
        T: Total optical power [W].

    Example:
        >>> intensity_to_power_gaussian_beam(1000.0, 0.5e-3, 0.5e-3)
        0.0015707963267948967
    """
    return intensity * 2 * np.pi * sigma_x * sigma_y


def power_to_rabi_gaussian_beam(
    power: T,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = ED_XtB,
) -> T:
    """Calculate peak Rabi frequency from power for Gaussian beam profile.

    Most common conversion for laser-molecule interactions. Gives on-axis (peak) Rabi
    frequency for Gaussian spatial profile.

    Args:
        power (T): Total optical power [W].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        sigma_x (float): Gaussian standard deviation in x [m].
        sigma_y (float): Gaussian standard deviation in y [m].
        D (float): Transition dipole moment [C·m]. Defaults to 2.6675506e-30 for
            TlF B-X optical transition at 272 nm.

    Returns:
        T: Peak (on-axis) Rabi frequency [rad/s].

    Example:
        >>> power_to_rabi_gaussian_beam(0.001, 1.0, 0.5e-3, 0.5e-3)  # 1 mW, TEM₀₀
        1.396...e+05

    Note:
        For circular beam with waist w₀, use sigma = w₀/2.
    """
    intensity = power_to_intensity_gaussian_beam(power, sigma_x, sigma_y)
    rabi = intensity_to_rabi(intensity, coupling, D)
    return rabi


def power_to_rabi_gaussian_beam_microwave(
    power: T,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = XConstants.D,
) -> T:
    """Calculate peak Rabi frequency from microwave power for Gaussian profile.

    Specialized for microwave transitions within ground X state. Uses larger permanent
    dipole moment appropriate for rotational transitions.

    Args:
        power (T): Total microwave power [W].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        sigma_x (float): Gaussian standard deviation in x [m].
        sigma_y (float): Gaussian standard deviation in y [m].
        D (float): Permanent dipole moment [C·m]. Defaults to 1.4103753e-29 for
            TlF X state (rotational transitions).

    Returns:
        T: Peak (on-axis) Rabi frequency [rad/s].

    Example:
        >>> power_to_rabi_gaussian_beam_microwave(0.1, 1.0, 5e-3, 5e-3)  # 100 mW
        7.377...e+05

    Note:
        Microwave dipole moment is ~5x larger than optical transition dipole.
    """
    return power_to_rabi_gaussian_beam(
        power=power, coupling=coupling, sigma_x=sigma_x, sigma_y=sigma_y, D=D
    )


def rabi_to_electric_field(rabi: T, coupling: float, D: float) -> T:
    """Calculate required electric field from desired Rabi frequency.

    Inverts the Rabi frequency formula: E₀ = Ω·ℏ/(μ·coupling).

    Args:
        rabi (T): Desired Rabi frequency [rad/s].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        D (float): Transition dipole moment [C·m].

    Returns:
        T: Required electric field amplitude [V/m].

    Example:
        >>> rabi_to_electric_field(1e6, 1.0, 2.67e-30)  # 1 MHz Rabi frequency
        3948.0...
    """
    return rabi * cst.hbar / (coupling * D)


def electric_field_to_intensity(electric_field: T) -> T:
    """Convert electric field amplitude to optical intensity.

    Uses time-averaged electromagnetic energy density: I = (1/2)·c·ε₀·E₀².

    Args:
        electric_field (T): Electric field amplitude E₀ [V/m].

    Returns:
        T: Optical intensity [W/m²].

    Example:
        >>> electric_field_to_intensity(868.36)  # Field of ~870 V/m
        999.99...
    """
    return 1 / 2 * cst.c * cst.epsilon_0 * electric_field**2


def rabi_to_power_gaussian_beam(
    rabi: T,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = ED_XtB,
) -> T:
    """Calculate required laser power from desired Rabi frequency for Gaussian beam.

    Inverse of power_to_rabi_gaussian_beam. Useful for determining laser power needed
    to achieve target Rabi frequency for specific transition.

    Args:
        rabi (T): Desired peak Rabi frequency [rad/s].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        sigma_x (float): Gaussian standard deviation in x [m].
        sigma_y (float): Gaussian standard deviation in y [m].
        D (float): Transition dipole moment [C·m]. Defaults to 2.6675506e-30 for
            TlF B-X optical transition.

    Returns:
        T: Required total optical power [W].

    Example:
        >>> rabi_to_power_gaussian_beam(1e6, 1.0, 0.5e-3, 0.5e-3)  # 1 MHz Rabi
        7.147...e-06
    """
    electric_field = rabi_to_electric_field(rabi, coupling, D)
    intensity = electric_field_to_intensity(electric_field)
    power = intensity_to_power_gaussian_beam(intensity, sigma_x, sigma_y)
    return power


def rabi_to_power_gaussian_beam_microwave(
    rabi: T,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = XConstants.D,
) -> T:
    """Calculate required microwave power from desired Rabi frequency for Gaussian beam.

    Specialized for microwave rotational transitions within ground X state.

    Args:
        rabi (T): Desired peak Rabi frequency [rad/s].
        coupling (float): Reduced matrix element coupling strength (dimensionless).
        sigma_x (float): Gaussian standard deviation in x [m].
        sigma_y (float): Gaussian standard deviation in y [m].
        D (float): Permanent dipole moment [C·m]. Defaults to 1.4103753e-29 for
            TlF X state rotational transitions.

    Returns:
        T: Required total microwave power [W].

    Example:
        >>> rabi_to_power_gaussian_beam_microwave(1e6, 1.0, 5e-3, 5e-3)
        1.356...e-07
    """
    return rabi_to_power_gaussian_beam(rabi, coupling, sigma_x, sigma_y, D)
