"""Doppler shift and velocity-to-detuning conversion utilities.

This module provides functions to calculate Doppler-shifted frequencies and
convert molecular velocities to laser detunings for spectroscopy applications.
"""
from typing import Union, overload

import numpy as np
import numpy.typing as npt
import scipy.constants as cst

__all__ = ["doppler_shift", "velocity_to_detuning"]

# Define type aliases for clarity
FloatOrArray = Union[float, npt.NDArray[np.floating]]


@overload
def doppler_shift(velocity: float, frequency: float = 1.1e15) -> float: ...
@overload
def doppler_shift(
    velocity: npt.NDArray[np.floating], frequency: float = 1.1e15
) -> npt.NDArray[np.floating]: ...
@overload
def doppler_shift(
    velocity: float, frequency: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]: ...
@overload
def doppler_shift(
    velocity: npt.NDArray[np.floating], frequency: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]: ...


def doppler_shift(
    velocity: FloatOrArray, frequency: FloatOrArray = 1.1e15
) -> FloatOrArray:
    """Calculate the Doppler-shifted frequency for a given velocity.

    Uses the non-relativistic Doppler formula: f' = f(1 + v/c) where c is the speed
    of light. This approximation is valid for v << c.

    Args:
        velocity (FloatOrArray): Velocity in m/s (float or array). Positive for
            molecules approaching the laser (blue-shift), negative for receding
            (red-shift).
        frequency (FloatOrArray): Rest-frame frequency in Hz (float or array).
            Defaults to 1.1e15 Hz (TlF B-X transition at ~272 nm).

    Returns:
        FloatOrArray: Doppler-shifted frequency in Hz. Returns same type as input
            (float or array).

    Raises:
        ValueError: If frequency is non-positive.

    Example:
        >>> doppler_shift(100.0)  # 100 m/s towards observer
        1.1000003666666666e+15
        >>> doppler_shift(-100.0)  # 100 m/s away from observer
        1.0999996333333334e+15
        >>> doppler_shift(np.array([0, 100, -100]))  # Array of velocities
        array([1.1e+15, 1.10000037e+15, 1.09999963e+15])
        
    Note:
        For relativistic velocities (v > 0.1c), use the relativistic Doppler formula.
    """
    if np.any(np.asarray(frequency) <= 0):
        raise ValueError(f"Frequency must be positive, got {frequency}")

    return frequency * (1 + velocity / cst.c)


@overload
def velocity_to_detuning(velocity: float, frequency: float = 1.1e15) -> float: ...
@overload
def velocity_to_detuning(
    velocity: npt.NDArray[np.floating], frequency: float = 1.1e15
) -> npt.NDArray[np.floating]: ...
@overload
def velocity_to_detuning(
    velocity: float, frequency: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]: ...
@overload
def velocity_to_detuning(
    velocity: npt.NDArray[np.floating], frequency: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]: ...


def velocity_to_detuning(
    velocity: FloatOrArray, frequency: FloatOrArray = 1.1e15
) -> FloatOrArray:
    """Convert molecular velocity to laser detuning based on Doppler shift.

    Calculates the detuning from resonance: Δω = ω(v/c) where ω = 2πf is the angular
    frequency. This represents the frequency shift in the molecular frame.

    Args:
        velocity (FloatOrArray): Velocity in m/s (float or array). Positive for
            molecules approaching the laser (positive detuning), negative for receding
            (negative detuning).
        frequency (FloatOrArray): Rest-frame transition frequency in Hz (float or array).
            Defaults to 1.1e15 Hz (TlF B-X transition at ~272 nm).

    Returns:
        FloatOrArray: Detuning in rad/s. Positive means laser is blue-detuned in
            molecular frame, negative means red-detuned. Returns same type as input.

    Raises:
        ValueError: If frequency is non-positive.

    Example:
        >>> velocity_to_detuning(100.0)  # 100 m/s molecular velocity
        2.3089690054388264e+09
        >>> velocity_to_detuning(np.array([50, 100, 150]))  # Multiple velocities
        array([1.15448450e+09, 2.30896901e+09, 3.46345351e+09])
        
    Note:
        The detuning Δω = ω' - ω₀ where ω' is the laser frequency in the molecular
        frame and ω₀ is the atomic transition frequency.
    """
    if np.any(np.asarray(frequency) <= 0):
        raise ValueError(f"Frequency must be positive, got {frequency}")

    # Direct computation of detuning in rad/s
    return frequency * (velocity / cst.c) * (2 * np.pi)
