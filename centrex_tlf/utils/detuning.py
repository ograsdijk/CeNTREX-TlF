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
    """
    Calculate the Doppler-shifted frequency for a given velocity.

    Uses the non-relativistic Doppler formula: f' = f(1 + v/c)
    Positive velocity means moving towards the observer (blue-shift).
    Negative velocity means moving away from the observer (red-shift).

    Args:
        velocity (FloatOrArray): Velocity in m/s (float or array).
                                 Positive for approaching, negative for receding.
        frequency (FloatOrArray, optional): Rest-frame frequency in Hz (float or array).
                                            Defaults to 1.1e15 Hz (TlF B-X transition).

    Returns:
        FloatOrArray: Doppler-shifted frequency in Hz (float or array).

    Raises:
        ValueError: If frequency is non-positive.

    Example:
        >>> doppler_shift(100.0)  # 100 m/s towards observer
        >>> doppler_shift(-100.0)  # 100 m/s away from observer
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
    """
    Convert a velocity to a detuning based on the Doppler shift.

    Calculates the detuning: Δω = ω(v/c) where ω = 2πf
    Positive velocity means positive detuning (blue-shift).
    Negative velocity means negative detuning (red-shift).

    Args:
        velocity (FloatOrArray): Velocity in m/s (float or array).
                                 Positive for approaching, negative for receding.
        frequency (FloatOrArray, optional): Rest-frame frequency in Hz (float or array).
                                            Defaults to 1.1e15 Hz (TlF B-X transition).

    Returns:
        FloatOrArray: Detuning in rad/s (float or array).

    Raises:
        ValueError: If frequency is non-positive.

    Example:
        >>> velocity_to_detuning(100.0)  # Returns detuning in rad/s
        >>> velocity_to_detuning(np.array([50, 100, 150]))  # Array of velocities
    """
    if np.any(np.asarray(frequency) <= 0):
        raise ValueError(f"Frequency must be positive, got {frequency}")

    # Direct computation of detuning in rad/s
    return frequency * (velocity / cst.c) * (2 * np.pi)
