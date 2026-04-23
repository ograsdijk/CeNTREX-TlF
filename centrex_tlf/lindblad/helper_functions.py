from __future__ import annotations

from enum import IntEnum
import math
from typing import Callable, Iterable, Mapping

__all__ = [
    "HelperFunctionId",
    "HELPER_FUNCTION_NAMES",
    "HELPER_FUNCTION_IDS",
    "HELPER_FUNCTIONS",
    "gaussian_1d",
    "gaussian_2d",
    "gaussian_2d_rotated",
    "phase_modulation",
    "square_wave",
    "resonant_polarization_modulation",
    "sawtooth_wave",
    "variable_on_off",
    "variable_on_off_duty",
    "variable_on_off_duty_invT",
    "multipass_2d_intensity",
    "rabi_from_intensity",
    "multipass_2d_rabi",
    "gaussian_beam_rabi",
    "alternating_sign",
    "linear_interp",
    "pchip_interp",
]


class HelperFunctionId(IntEnum):
    GAUSSIAN_2D = 1
    GAUSSIAN_2D_ROTATED = 2
    PHASE_MODULATION = 3
    SQUARE_WAVE = 4
    RESONANT_POLARIZATION_MODULATION = 5
    SAWTOOTH_WAVE = 6
    VARIABLE_ON_OFF = 7
    MULTIPASS_2D_INTENSITY = 8
    RABI_FROM_INTENSITY = 9
    MULTIPASS_2D_RABI = 10
    GAUSSIAN_BEAM_RABI = 11
    VARIABLE_ON_OFF_DUTY = 12
    ALTERNATING_SIGN = 13
    LINEAR_INTERP = 14
    GAUSSIAN_1D = 15
    PCHIP_INTERP = 16


def gaussian_2d(
    x: float,
    y: float,
    amplitude: float,
    mean_x: float,
    mean_y: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    dx = x - mean_x
    dy = y - mean_y
    return amplitude * math.exp(
        -((dx * dx) / (2.0 * sigma_x * sigma_x) + (dy * dy) / (2.0 * sigma_y * sigma_y))
    )


def gaussian_2d_rotated(
    x: float,
    y: float,
    amplitude: float,
    mean_x: float,
    mean_y: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
) -> float:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    a = cos_theta * cos_theta / (2.0 * sigma_x * sigma_x) + sin_theta * sin_theta / (
        2.0 * sigma_y * sigma_y
    )
    b = math.sin(2.0 * theta) / (2.0 * sigma_x * sigma_x) - math.sin(2.0 * theta) / (
        2.0 * sigma_y * sigma_y
    )
    c = sin_theta * sin_theta / (2.0 * sigma_x * sigma_x) + cos_theta * cos_theta / (
        2.0 * sigma_y * sigma_y
    )
    dx = x - mean_x
    dy = y - mean_y
    return amplitude * math.exp(-(a * dx * dx + b * dx * dy + c * dy * dy))


def phase_modulation(t: float, beta: float, omega: float) -> complex:
    phi = beta * math.sin(omega * t)
    return complex(math.cos(phi), math.sin(phi))


def square_wave(t: float, omega: float, phase: float) -> float:
    return 0.5 * (1.0 + (1.0 if math.sin(omega * t + phase) >= 0.0 else -1.0))


def resonant_polarization_modulation(t: float, gamma: float, omega: float) -> complex:
    theta = 0.5 * gamma * math.sin(omega * t)
    a = 0.5 * (math.cos(theta) + math.sin(theta))
    return complex(a, a)


def sawtooth_wave(t: float, omega: float, phase: float) -> float:
    frac = ((omega * t + phase - math.pi) / (2.0 * math.pi)) % 1.0
    return frac


def variable_on_off(t: float, ton: float, toff: float, phase: float) -> float:
    period = ton + toff
    duty = ton / period
    frac = ((t / period) + phase / (2.0 * math.pi)) % 1.0
    return 1.0 if frac < duty else 0.0


def variable_on_off_duty(t: float, duty: float, inv_period: float, phase: float) -> float:
    frac = (t * inv_period + phase / (2.0 * math.pi)) % 1.0
    if frac <= 0.0:
        frac += 1.0
    return 1.0 if frac < duty else 0.0


def variable_on_off_duty_invT(
    t: float, duty: float, inv_period: float, phase: float
) -> float:
    return variable_on_off_duty(t, duty, inv_period, phase)


def multipass_2d_intensity(
    x: float,
    y: float,
    amplitudes: Iterable[float],
    xlocs: Iterable[float],
    ylocs: Iterable[float],
    sigma_x: float,
    sigma_y: float,
) -> float:
    intensity = 0.0
    for amplitude, xloc, yloc in zip(amplitudes, xlocs, ylocs):
        intensity += gaussian_2d(x, y, amplitude, xloc, yloc, sigma_x, sigma_y)
    return intensity


def rabi_from_intensity(
    intensity: float,
    coupling: float,
    dipole_moment: float = 2.6675506e-30,
) -> float:
    hbar = 1.0545718176461565e-34
    c = 299792458.0
    eps0 = 8.8541878128e-12
    electric_field = math.sqrt(intensity * 2.0 / (c * eps0))
    return electric_field * coupling * dipole_moment / hbar


def multipass_2d_rabi(
    x: float,
    y: float,
    intensities: Iterable[float],
    xlocs: Iterable[float],
    ylocs: Iterable[float],
    sigma_x: float,
    sigma_y: float,
    main_coupling: float,
    dipole_moment: float = 2.6675506e-30,
) -> float:
    intensity = multipass_2d_intensity(x, y, intensities, xlocs, ylocs, sigma_x, sigma_y)
    return rabi_from_intensity(intensity, main_coupling, dipole_moment)


def gaussian_beam_rabi(
    x: float,
    y: float,
    intensity: float,
    xloc: float,
    yloc: float,
    sigma_x: float,
    sigma_y: float,
    main_coupling: float,
    dipole_moment: float = 2.6675506e-30,
) -> float:
    beam_intensity = gaussian_2d(x, y, intensity, xloc, yloc, sigma_x, sigma_y)
    return rabi_from_intensity(beam_intensity, main_coupling, dipole_moment)


def alternating_sign(x: float, x0: float, width: float) -> float:
    n = math.floor((x - x0) / width)
    return 1.0 if n % 2 == 0 else -1.0


def linear_interp(x: float, grid: Iterable[float], values: Iterable[float]) -> float:
    grid_values = tuple(float(item) for item in grid)
    y_values = tuple(float(item) for item in values)
    if len(grid_values) != len(y_values):
        raise ValueError("linear_interp grid and values must have the same length")
    if len(grid_values) == 0:
        raise ValueError("linear_interp grid must contain at least one point")
    if len(grid_values) == 1:
        return y_values[0]
    if any(right <= left for left, right in zip(grid_values, grid_values[1:])):
        raise ValueError("linear_interp grid must be strictly increasing")
    if x <= grid_values[0]:
        return y_values[0]
    if x >= grid_values[-1]:
        return y_values[-1]
    for left_idx, (left, right) in enumerate(zip(grid_values, grid_values[1:])):
        if left <= x <= right:
            frac = (x - left) / (right - left)
            return y_values[left_idx] + frac * (y_values[left_idx + 1] - y_values[left_idx])
    return y_values[-1]


def gaussian_1d(x: float, center: float, sigma: float) -> float:
    dx = x - center
    return math.exp(-(dx * dx) / (2.0 * sigma * sigma))


def pchip_interp(x: float, grid: Iterable[float], values: Iterable[float]) -> float:
    from scipy.interpolate import PchipInterpolator

    grid_arr = [float(g) for g in grid]
    values_arr = [float(v) for v in values]
    interp = PchipInterpolator(grid_arr, values_arr, extrapolate=True)
    return float(interp(x))


HELPER_FUNCTIONS: Mapping[str, Callable[..., complex | float]] = {
    "gaussian_1d": gaussian_1d,
    "gaussian_2d": gaussian_2d,
    "gaussian_2d_rotated": gaussian_2d_rotated,
    "phase_modulation": phase_modulation,
    "square_wave": square_wave,
    "resonant_polarization_modulation": resonant_polarization_modulation,
    "sawtooth_wave": sawtooth_wave,
    "variable_on_off": variable_on_off,
    "multipass_2d_intensity": multipass_2d_intensity,
    "rabi_from_intensity": rabi_from_intensity,
    "multipass_2d_rabi": multipass_2d_rabi,
    "gaussian_beam_rabi": gaussian_beam_rabi,
    "variable_on_off_duty": variable_on_off_duty,
    "variable_on_off_duty_invT": variable_on_off_duty_invT,
    "alternating_sign": alternating_sign,
    "linear_interp": linear_interp,
    "pchip_interp": pchip_interp,
}

HELPER_FUNCTION_IDS: Mapping[str, HelperFunctionId] = {
    "gaussian_1d": HelperFunctionId.GAUSSIAN_1D,
    "gaussian_2d": HelperFunctionId.GAUSSIAN_2D,
    "gaussian_2d_rotated": HelperFunctionId.GAUSSIAN_2D_ROTATED,
    "phase_modulation": HelperFunctionId.PHASE_MODULATION,
    "square_wave": HelperFunctionId.SQUARE_WAVE,
    "resonant_polarization_modulation": HelperFunctionId.RESONANT_POLARIZATION_MODULATION,
    "sawtooth_wave": HelperFunctionId.SAWTOOTH_WAVE,
    "variable_on_off": HelperFunctionId.VARIABLE_ON_OFF,
    "multipass_2d_intensity": HelperFunctionId.MULTIPASS_2D_INTENSITY,
    "rabi_from_intensity": HelperFunctionId.RABI_FROM_INTENSITY,
    "multipass_2d_rabi": HelperFunctionId.MULTIPASS_2D_RABI,
    "gaussian_beam_rabi": HelperFunctionId.GAUSSIAN_BEAM_RABI,
    "variable_on_off_duty": HelperFunctionId.VARIABLE_ON_OFF_DUTY,
    "variable_on_off_duty_invT": HelperFunctionId.VARIABLE_ON_OFF_DUTY,
    "alternating_sign": HelperFunctionId.ALTERNATING_SIGN,
    "linear_interp": HelperFunctionId.LINEAR_INTERP,
    "pchip_interp": HelperFunctionId.PCHIP_INTERP,
}

HELPER_FUNCTION_NAMES: Mapping[int, str] = {
    int(helper_id): name for name, helper_id in HELPER_FUNCTION_IDS.items()
}
