"""Multipass beam geometry and Gaussian beam intensity profiles.

This module provides functions for calculating multipass optical arrangements,
Gaussian beam profiles, and their spatial intensity/Rabi frequency distributions.
Commonly used for laser cooling and optical pumping experiments.
"""
from typing import List, Sequence, cast, overload

import numpy as np
import numpy.typing as npt

from .rabi import intensity_to_rabi

__all__ = [
    "multipass_prism_order",
    "gaussian",
    "gaussian_amp",
    "gaussian_2d",
    "gaussian_2d_amp",
    "generate_2d_multipass_gaussian_intensity",
    "generate_2d_multipass_gaussian_rabi",
]


def multipass_prism_order(passes: int) -> List[int]:
    """Determine spatial ordering of passes in a two-prism multipass arrangement.

    In a multipass cell using two prisms, the beam reflects between prisms creating
    spatially separated passes. This function calculates the physical order in which
    the passes appear spatially.

    Args:
        passes (int): Total number of passes through the cell (must be odd).

    Returns:
        List[int]: Spatial ordering of passes (1-indexed).

    Raises:
        ValueError: If passes is even (two-prism multipass requires odd number).

    Example:
        >>> multipass_prism_order(5)
        [1, 4, 3, 2, 5]
        >>> multipass_prism_order(7)
        [1, 6, 5, 4, 3, 2, 7]
        
    Note:
        The pattern alternates: center pass, outer passes work inward, final center pass.
    """
    if passes % 2 == 0:
        raise ValueError("Number of passes must be odd for two-prism multipass")

    npass = [1]
    for p in range(1, passes):
        if p % 2 == 0:
            npass.append(p + 1)
        else:
            npass.append(passes - p)
    return npass


@overload
def gaussian(
    x: npt.NDArray[np.floating], mean: float, sigma: float
) -> npt.NDArray[np.floating]: ...


@overload
def gaussian(x: float, mean: float, sigma: float) -> float: ...


def gaussian(x, mean, sigma):
    """Non-normalized 1D Gaussian function.

    Evaluates exp(-(x-μ)²/(2σ²)) without the normalization factor 1/√(2πσ²).
    Peak value is 1.0 at x = mean.

    Args:
        x (float | npt.NDArray[np.floating]): Position(s) to evaluate Gaussian.
        mean (float): Center position (μ) of the Gaussian.
        sigma (float): Standard deviation (σ) controlling width.

    Returns:
        float | npt.NDArray[np.floating]: Gaussian evaluated at x. Peak value is 1.0.

    Example:
        >>> gaussian(0, 0, 1)  # Peak at center
        1.0
        >>> gaussian(1, 0, 1)  # One sigma away
        0.6065306597126334
    """
    return np.exp(-((x - mean) ** 2) / (2 * sigma**2))


@overload
def gaussian_amp(
    x: npt.NDArray[np.floating], a: float, mean: float, sigma: float
) -> npt.NDArray[np.floating]: ...


@overload
def gaussian_amp(x: float, a: float, mean: float, sigma: float) -> float: ...


def gaussian_amp(x, a, mean, sigma):
    """Non-normalized 1D Gaussian function with custom amplitude.

    Evaluates a·exp(-(x-μ)²/(2σ²)) where a sets the peak height.

    Args:
        x (float | npt.NDArray[np.floating]): Position(s) to evaluate Gaussian.
        a (float): Peak amplitude at x = mean.
        mean (float): Center position (μ) of the Gaussian.
        sigma (float): Standard deviation (σ) controlling width.

    Returns:
        float | npt.NDArray[np.floating]: Gaussian evaluated at x. Peak value is a.

    Example:
        >>> gaussian_amp(0, 2.5, 0, 1)  # Peak amplitude of 2.5
        2.5
    """
    return a * gaussian(x, mean, sigma)


@overload
def gaussian_2d(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    mean_x: float,
    mean_y: float,
    sigma_x: float,
    sigma_y: float,
) -> npt.NDArray[np.floating]: ...


@overload
def gaussian_2d(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    mean_x: npt.NDArray[np.floating],
    mean_y: npt.NDArray[np.floating],
    sigma_x: float,
    sigma_y: float,
) -> npt.NDArray[np.floating]: ...


@overload
def gaussian_2d(
    x: float,
    y: float,
    mean_x: float,
    mean_y: float,
    sigma_x: float,
    sigma_y: float,
) -> float: ...


def gaussian_2d(x, y, mean_x, mean_y, sigma_x, sigma_y):
    """Non-normalized 2D Gaussian function.

    Evaluates exp(-[(x-μₓ)²/(2σₓ²) + (y-μᵧ)²/(2σᵧ²)]) without normalization.
    Peak value is 1.0 at (mean_x, mean_y).

    Args:
        x (float | npt.NDArray[np.floating]): x-coordinate(s).
        y (float | npt.NDArray[np.floating]): y-coordinate(s).
        mean_x (float | npt.NDArray[np.floating]): Center x-position (μₓ).
        mean_y (float | npt.NDArray[np.floating]): Center y-position (μᵧ).
        sigma_x (float): Standard deviation in x (σₓ).
        sigma_y (float): Standard deviation in y (σᵧ).

    Returns:
        float | npt.NDArray[np.floating]: 2D Gaussian evaluated at (x, y).

    Example:
        >>> gaussian_2d(0, 0, 0, 0, 1, 1)  # Peak at center
        1.0
        >>> gaussian_2d(1, 1, 0, 0, 1, 1)  # At (σₓ, σᵧ)
        0.36787944117144233
    """
    a = (x - mean_x) ** 2 / (2 * sigma_x**2)
    b = (y - mean_y) ** 2 / (2 * sigma_y**2)
    return np.exp(-(a + b))


@overload
def gaussian_2d_amp(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    a: float,
    mean_x: float,
    mean_y: float,
    sigma_x: float,
    sigma_y: float,
) -> npt.NDArray[np.floating]: ...


@overload
def gaussian_2d_amp(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    a: npt.NDArray[np.floating],
    mean_x: npt.NDArray[np.floating],
    mean_y: npt.NDArray[np.floating],
    sigma_x: float,
    sigma_y: float,
) -> npt.NDArray[np.floating]: ...


@overload
def gaussian_2d_amp(
    x: float,
    y: float,
    a: float,
    mean_x: float,
    mean_y: float,
    sigma_x: float,
    sigma_y: float,
) -> float: ...


def gaussian_2d_amp(
    x,
    y,
    a,
    mean_x,
    mean_y,
    sigma_x,
    sigma_y,
):
    """Non-normalized 2D Gaussian function with custom amplitude.

    Evaluates a·exp(-[(x-μₓ)²/(2σₓ²) + (y-μᵧ)²/(2σᵧ²)]) where a sets peak height.

    Args:
        x (float | npt.NDArray[np.floating]): x-coordinate(s).
        y (float | npt.NDArray[np.floating]): y-coordinate(s).
        a (float | npt.NDArray[np.floating]): Peak amplitude.
        mean_x (float | npt.NDArray[np.floating]): Center x-position (μₓ).
        mean_y (float | npt.NDArray[np.floating]): Center y-position (μᵧ).
        sigma_x (float): Standard deviation in x (σₓ).
        sigma_y (float): Standard deviation in y (σᵧ).

    Returns:
        float | npt.NDArray[np.floating]: 2D Gaussian evaluated at (x, y).

    Example:
        >>> gaussian_2d_amp(0, 0, 5.0, 0, 0, 1, 1)  # Peak amplitude of 5.0
        5.0
    """
    return a * gaussian_2d(x, y, mean_x, mean_y, sigma_x, sigma_y)


def generate_2d_multipass_gaussian_intensity(
    X: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    locations_x: Sequence[float],
    locations_y: Sequence[float],
    intensities: Sequence[float],
    sigma_x: float,
    sigma_y: float,
) -> npt.NDArray[np.floating]:
    """Generate 2D spatial intensity distribution for a multipass beam arrangement.

    Creates a composite intensity map by superimposing multiple Gaussian beams at
    different spatial locations, each with specified intensity. Useful for modeling
    multipass laser cooling or optical pumping geometries.

    Args:
        X (npt.NDArray[np.floating]): 2D meshgrid of x-coordinates [m].
        Y (npt.NDArray[np.floating]): 2D meshgrid of y-coordinates [m]. Must match X shape.
        locations_x (Sequence[float]): x-positions of beam centers [m], one per pass.
        locations_y (Sequence[float]): y-positions of beam centers [m], one per pass.
        intensities (Sequence[float]): Peak intensities [W/m²], one per pass.
        sigma_x (float): Gaussian standard deviation in x [m] (same for all passes).
        sigma_y (float): Gaussian standard deviation in y [m] (same for all passes).

    Returns:
        npt.NDArray[np.floating]: 2D array of total intensity [W/m²] at each (X, Y) point.
            Shape matches X and Y.

    Raises:
        ValueError: If locations_x, locations_y, and intensities have different lengths.

    Example:
        >>> X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        >>> intensity = generate_2d_multipass_gaussian_intensity(
        ...     X, Y, [0, 2], [0, 0], [1000, 800], 0.5, 0.5
        ... )
    """
    if len(locations_x) != len(locations_y) or len(locations_x) != len(intensities):
        raise ValueError(
            f"Lengths of locations_x ({len(locations_x)}), locations_y ({len(locations_y)}), "
            f"and intensities ({len(intensities)}) must be equal"
        )
    _intensities = cast(npt.NDArray[np.floating], np.asarray(intensities))
    _locations_x = cast(npt.NDArray[np.floating], np.asarray(locations_x))
    _locations_y = cast(npt.NDArray[np.floating], np.asarray(locations_y))
    multipass = gaussian_2d_amp(
        x=X[:, :, np.newaxis],
        y=Y[:, :, np.newaxis],
        a=_intensities[np.newaxis, :],
        mean_x=_locations_x[np.newaxis, :],
        mean_y=_locations_y[np.newaxis, :],
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    ).sum(axis=2)
    return multipass


def generate_2d_multipass_gaussian_rabi(
    X: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    locations_x: Sequence[float],
    locations_y: Sequence[float],
    intensities: Sequence[float],
    sigma_x: float,
    sigma_y: float,
    coupling: float,
    D: float = 2.6675506e-30,
) -> npt.NDArray[np.floating]:
    """Generate 2D spatial Rabi frequency distribution for a multipass beam arrangement.

    Combines multipass intensity calculation with conversion to Rabi frequencies using
    the transition dipole moment. Useful for simulating optical Bloch equations or
    molecular dynamics in multipass beam geometries.

    Args:
        X (npt.NDArray[np.floating]): 2D meshgrid of x-coordinates [m].
        Y (npt.NDArray[np.floating]): 2D meshgrid of y-coordinates [m].
        locations_x (Sequence[float]): x-positions of beam centers [m], one per pass.
        locations_y (Sequence[float]): y-positions of beam centers [m], one per pass.
        intensities (Sequence[float]): Peak intensities [W/m²], one per pass.
        sigma_x (float): Gaussian standard deviation in x [m].
        sigma_y (float): Gaussian standard deviation in y [m].
        coupling (float): Reduced matrix element coupling strength [C·m].
        D (float): Transition dipole moment [C·m]. Defaults to 2.6675506e-30 for
            TlF B-X transition.

    Returns:
        npt.NDArray[np.floating]: 2D array of Rabi frequency [rad/s] at each (X, Y) point.

    Example:
        >>> X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        >>> rabi = generate_2d_multipass_gaussian_rabi(
        ...     X, Y, [0, 2], [0, 0], [1000, 800], 0.5, 0.5, 1e-30
        ... )
    """
    multipass = generate_2d_multipass_gaussian_intensity(
        X, Y, locations_x, locations_y, intensities, sigma_x, sigma_y
    )
    rabi = intensity_to_rabi(intensity=multipass, coupling=coupling, D=D)
    return rabi
