"""Multipass beam geometry and Gaussian beam intensity profiles.

This module provides functions for calculating multipass optical arrangements,
Gaussian beam profiles, and their spatial intensity/Rabi frequency distributions.
Commonly used for laser cooling and optical pumping experiments.
"""
from typing import List, Sequence, cast, overload

import numpy as np
import numpy.typing as npt

from centrex_tlf.constants import ED_XtB

from .rabi import intensity_to_rabi

__all__ = [
    "multipass_prism_order",
    "trace_prism_multipass",
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
        passes (int): Total number of passes through the cell (must be positive and odd).

    Returns:
        List[int]: Spatial ordering of passes (1-indexed).

    Raises:
        ValueError: If passes is nonpositive or even.

    Example:
        >>> multipass_prism_order(5)
        [1, 4, 3, 2, 5]
        >>> multipass_prism_order(7)
        [1, 6, 3, 4, 5, 2, 7]
        
    Note:
        From left to right, odd pass labels increase and even pass labels decrease,
        interleaved starting with pass 1. Actual ordering depends on entry and prism
        offset; use trace_prism_multipass for a specified geometry.
    """
    if passes <= 0:
        raise ValueError("Number of passes must be positive")
    if passes % 2 == 0:
        raise ValueError("Number of passes must be odd for two-prism multipass")

    npass = [1]
    for p in range(1, passes):
        if p % 2 == 0:
            npass.append(p + 1)
        else:
            npass.append(passes - p)
    return npass


PRISM_LEG_LENGTH_M = 0.040


def trace_prism_multipass(
    prism_center_offset_m: float,
    first_entry_m: float,
) -> tuple[List[int], npt.NDArray[np.float64]]:
    """Trace a normal-incidence ray through two parallel multipass prisms.

    The prisms have right-isosceles 40 mm legs and parallel hypotenuse faces.
    Coordinates run along the front faces, with zero at an endpoint of the
    first prism's front face. ``first_entry_m`` is the incoming ray coordinate
    on that face. The second prism center is displaced by
    ``prism_center_offset_m`` along the face coordinate.

    At each prism, the internal reflections map the incoming face coordinate
    ``x`` to ``2 * center - x``. The function alternates this reflection
    between prisms and stops after the outgoing ray misses the opposite front
    face. The incoming entry segment is pass 1; each subsequent outgoing gap
    crossing adds another pass.

    Face endpoints are inclusive. Outgoing coordinates within eight machine
    epsilons of an endpoint (scaled by the face length and endpoint coordinates)
    are snapped to it before recording the pass and continuing the trace.

    Returns ``(order, locations_m)``. ``locations_m`` contains the traced pass
    coordinates sorted along the front face, and ``order[i]`` is the 1-indexed
    chronological pass label at ``locations_m[i]``. This function returns
    geometry only; it does not calculate intensity, attenuation, or optical
    losses.

    Args:
        prism_center_offset_m: Signed center-to-center offset along the front
            faces [m].
        first_entry_m: Incoming ray coordinate measured from an endpoint of
            the first prism front face [m].

    Returns:
        A tuple containing spatially sorted chronological pass labels and
        their matching pass locations along the front faces [m].

    Raises:
        ValueError: If either input is non-finite, the entry lies outside the
            first face, zero prism offset would make the ray circulate
            indefinitely, or the trace exceeds the iteration safety limit.
    """
    if not np.isfinite(prism_center_offset_m):
        raise ValueError("Prism center offset must be finite")
    if not np.isfinite(first_entry_m):
        raise ValueError("First entry coordinate must be finite")
    if prism_center_offset_m == 0:
        raise ValueError("Prism center offset must be nonzero for a finite multipass trace")

    front_face_length_m = PRISM_LEG_LENGTH_M * np.sqrt(2.0)
    if not 0 <= first_entry_m <= front_face_length_m:
        raise ValueError(
            "First entry coordinate must lie on the first prism front face "
            f"[0, {front_face_length_m:g}] m"
        )

    centers_m = (front_face_length_m / 2, front_face_length_m / 2 + prism_center_offset_m)
    faces_m = (
        (0.0, front_face_length_m),
        (prism_center_offset_m, prism_center_offset_m + front_face_length_m),
    )
    prism_index = 0
    incoming_m = first_entry_m
    seen_states = {(prism_index, incoming_m)}
    chronological_locations_m: list[float] = [first_entry_m]

    for _ in range(100_000):
        outgoing_m = 2 * centers_m[prism_index] - incoming_m
        next_prism_index = 1 - prism_index
        next_face_min_m, next_face_max_m = faces_m[next_prism_index]
        endpoint_tolerance_m = 8 * np.finfo(np.float64).eps * max(
            front_face_length_m, abs(next_face_min_m), abs(next_face_max_m)
        )
        if abs(outgoing_m - next_face_min_m) <= endpoint_tolerance_m:
            outgoing_m = next_face_min_m
        elif abs(outgoing_m - next_face_max_m) <= endpoint_tolerance_m:
            outgoing_m = next_face_max_m
        chronological_locations_m.append(outgoing_m)
        if not next_face_min_m <= outgoing_m <= next_face_max_m:
            break
        incoming_m = outgoing_m
        prism_index = next_prism_index
        state = (prism_index, incoming_m)
        if state in seen_states:
            raise ValueError(
                "Prism trace stopped progressing at floating-point precision"
            )
        seen_states.add(state)
    else:
        raise ValueError(
            "Prism trace exceeded 100,000 passes; the center offset may be "
            "too small for a finite practical trace"
        )

    chronological_locations = np.asarray(chronological_locations_m, dtype=np.float64)
    spatial_indices = np.argsort(chronological_locations, kind="stable")
    locations_m = chronological_locations[spatial_indices]
    order = (spatial_indices + 1).tolist()
    return order, locations_m


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
    D: float = ED_XtB,
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
        coupling (float): Dimensionless transition matrix element coupling strength.
        D (float): Transition dipole moment [C·m]. Defaults to ED_XtB for
            TlF B-X transition.

    Returns:
        npt.NDArray[np.floating]: 2D array of Rabi frequency [rad/s] at each (X, Y) point.

    Example:
        >>> X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        >>> rabi = generate_2d_multipass_gaussian_rabi(
        ...     X, Y, [0, 2], [0, 0], [1000, 800], 0.5, 0.5, 1.0
        ... )
    """
    multipass = generate_2d_multipass_gaussian_intensity(
        X, Y, locations_x, locations_y, intensities, sigma_x, sigma_y
    )
    rabi = intensity_to_rabi(intensity=multipass, coupling=coupling, D=D)
    return rabi
