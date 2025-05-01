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

    Args:
        velocity (FloatOrArray): Velocity in m/s (float or array).
        frequency (FloatOrArray, optional): Frequency in Hz (float or array).
                                            Defaults to 1.1e15 Hz.

    Returns:
        FloatOrArray: Doppler-shifted frequency in Hz (float or array).
    """
    # Input validation can be adapted for arrays if needed, e.g., using np.any
    if np.any(np.asarray(velocity) < 0):
        raise ValueError("Velocity must be non-negative.")
    if np.any(np.asarray(frequency) <= 0):
        raise ValueError("Frequency must be positive.")

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

    Args:
        velocity (FloatOrArray): Velocity in m/s (float or array).
        frequency (FloatOrArray, optional): Frequency in Hz (float or array).
                                            Defaults to 1.1e15 Hz.

    Returns:
        FloatOrArray: Detuning frequency in rad/s (2π ⋅ Hz) (float or array).
    """
    # Input validation can be adapted for arrays if needed
    if np.any(np.asarray(velocity) < 0):
        raise ValueError("Velocity must be non-negative.")
    if np.any(np.asarray(frequency) <= 0):
        raise ValueError("Frequency must be positive.")

    # Direct computation of detuning
    return frequency * (velocity / cst.c) * (2 * np.pi)
