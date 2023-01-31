import numpy as np
import scipy.constants as cst

__all__ = ["doppler_shift", "velocity_to_detuning"]


def doppler_shift(velocity: float, frequency: float = 1.1e15) -> float:
    """
    Doppler a velocity shift

    Args:
        velocity (float): velocity in m/s
        frequency (float, optional): frequency in Hz. Defaults to 1.1e15 Hz, the UV
                                    frequency of the X to B TlF transition.

    Returns:
        float: doppler shifted frequency in Hz
    """
    return frequency * (1 + velocity / cst.c)


def velocity_to_detuning(velocity: float, frequency: float = 1.1e15) -> float:
    """
    Convert a velocity to a detuning based on the doppler shift.

    Args:
        velocity (float): velocity in m/s
        frequency (float, optional): frequency in Hz. Defaults to 1.1e15 Hz, the UV
                                    frequency of the X to B TlF transition.

    Returns:
        float: detuning frequency in radial frequency 2π ⋅ Hz
    """
    return (doppler_shift(velocity, frequency) - frequency) * 2 * np.pi
