from typing import TypeVar, overload

import numpy as np
import numpy.typing as npt
import scipy.constants as cst

from centrex_tlf.constants import D_XtB, XConstants

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
    """
    Full width at half maximum to standard deviation

    Args:
        fwhm (float): full width at half maximum

    Returns:
        float: standard deviation
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def sigma_to_fwhm(sigma: T) -> T:
    """
    Standard deviation to full width at half maximum

    Args:
        sigma (float): standard deviation

    Returns:
        float: full width at half maximum
    """
    return sigma * 2 * np.sqrt(2 * np.log(2))


def sigma_to_waist(sigma: T) -> T:
    """
    Standard deviation to waist

    Args:
        sigma (float): standard deviation

    Returns:
        float: waist
    """
    return sigma * 2


def waist_to_sigma(waist: T) -> T:
    """
    Waist to standard deviation

    Args:
        waist (float): waist

    Returns:
        float: standard deviation
    """
    return waist / 2


def intensity_to_electric_field(intensity: T) -> T:
    """
    Intensity in W/m^2 to electric field

    Args:
        intensity (float): intensity [W/m^2]

    Returns:
        float: electric field
    """
    return np.sqrt((2 / (cst.c * cst.epsilon_0)) * intensity)


def electric_field_to_rabi(electric_field: T, coupling: float, D: float) -> T:
    """
    Rabi rate from an electric field an coupling strength, with the dipole moment D
    default value set to the X to B transition.

    Args:
        electric_field (float): electric field
        coupling (float): coupling strength in Coulomb meter
        D (float, optional): Dipole moment.

    Returns:
        float: Rabi rate in rotational frequency [2π ⋅ Hz]
    """
    return electric_field * coupling * D / cst.hbar


def intensity_to_rabi(intensity: T, coupling: float, D: float) -> T:
    """
    Rabi rate from an intensity

    Args:
        intensity (float): intensity [W/m^2]
        coupling (float): coupling strength in Coulomb meter
        D (float): dipole moment

    Returns:
        float: rabi rate
    """
    electric_field = intensity_to_electric_field(intensity)
    rabi = electric_field_to_rabi(electric_field, coupling, D)
    return rabi


def intensity_to_power_rectangular_beam(intensity: T, wx: float, wy: float) -> T:
    """
    Power in W from a given intensity and beam width

    Args:
        intensity (float): intensity [W/m^2]
        wx (float): x width [m]
        wy (float): y width [m]

    Returns:
        float: power [W]
    """
    return intensity * wx * wy


def power_to_intensity_rectangular_beam(power: T, wx: float, wy: float) -> T:
    """
    Intensity in W/m^2 from a given power and beam width

    Args:
        power (float): power [W]
        wx (float): x width [m]
        wy (float): y width [m]

    Returns:
        float: intensity [W/m^2]
    """
    return power / (wx * wy)


def power_to_rabi_rectangular_beam(
    power: T,
    coupling: float,
    wx: float,
    wy: float,
    D: float = D_XtB,
) -> T:
    """
    Rabi rate from laser power and coupling strength for the X to B TLF transition.

    Args:
        power (float): power [W]
        coupling (float): coupling strength
        wx (float): x width [m]
        wy (float): y width [m]
        D (float, optional): Dipole moment. Defaults to 2.6675506e-30 Coulomb meter for
                            the X to B TLF transition.

    Returns:
        float: Rabi rate in rotational frequency [2π ⋅ Hz]
    """
    intensity = power_to_intensity_rectangular_beam(power, wx, wy)

    rabi = intensity_to_rabi(intensity, coupling, D)
    return rabi


def power_to_intensity_gaussian_beam(power: T, sigma_x: float, sigma_y: float) -> T:
    """
    Intensity in W/m^2 from a given power and beam width

    Args:
        power (float): power [W]
        sigma_x (float): x standard deviation [m]
        sigma_y (float): y standard deviation [m]

    Returns:
        float: intensity [W/m^2]
    """
    return power / (2 * np.pi * sigma_x * sigma_y)


def intensity_to_power_gaussian_beam(intensity: T, sigma_x: float, sigma_y: float) -> T:
    """
    Power in W from a given intensity and beam width

    Args:
        intensity (float): intensity [W/m^2]
        sigma_x (float): x standard deviation [m]
        sigma_y (float): y standard deviation [m]

    Returns:
        float: power [W]
    """
    return intensity * 2 * np.pi * sigma_x * sigma_y


def power_to_rabi_gaussian_beam(
    power: T,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = D_XtB,
) -> T:
    """
    Rabi rate from laser power and coupling strength for the X to B TlF transition.

    Args:
        power (float): power [W]
        coupling (float): coupling strength
        sigma_x (float): x standard deviation [m]
        sigma_y (float): y standard deviation [m]
        D (float, optional): Dipole moment. Defaults to 2.6675506e-30 Coulomb meter for
                            the X to B TLF transition.

    Returns:
        float: Rabi rate in rotational frequency [2π ⋅ Hz]
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
    """
    Rabi rate from microwave power and coupling strength for the X to X microwave
    transition

    Args:
        power (float): power [W]
        coupling (float): coupling strength
        sigma_x (float): x standard deviation
        sigma_y (float): y standard deviation
        D (float, optional): Dipole moment. Defaults to 1.4103753e-29 Coulomb meter for
                            the X to X TlF transition.

    Returns:
        float: Rabi rate in rotational frequency [2π ⋅ Hz]
    """
    return power_to_rabi_gaussian_beam(
        power=power, coupling=coupling, sigma_x=sigma_x, sigma_y=sigma_y, D=D
    )


def rabi_to_electric_field(rabi: T, coupling: float, D: float) -> T:
    """
    Electric field from a given rabi rate and coupling strength

    Args:
        rabi (float): rabi rate in rotational frequency [2π ⋅ Hz]
        coupling (float): coupling strength
        D (float): Dipole moment [Coulomb meter]

    Returns:
        float: electric field [V/m]
    """
    return rabi * cst.hbar / (coupling * D)


def electric_field_to_intensity(electric_field: T) -> T:
    """
    Intensity in W/m^2 from a given electric field

    Args:
        electric_field (float): electric field [V/m]

    Returns:
        float: intensity [W/m^2]
    """
    return 1 / 2 * cst.c * cst.epsilon_0 * electric_field**2


def rabi_to_power_gaussian_beam(
    rabi: T,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = D_XtB,
) -> T:
    """
    power in W given a rabi rate and couling strength

    Args:
        rabi (float): rabi rate in rotational frequency [2π ⋅ Hz]
        coupling (float): coupling strength
        D (float, optional): dipole moment. Defaults to 2.6675506e-30 Coulomb meter for
                            the X to B TlF transition.

    Returns:
        float: power [W]
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
    """
    power in W given a rabi rate and coupling strength

    Args:
        rabi (float): rabi rate in rotational frequency [2π ⋅ Hz]
        coupling (float): coupling strength
        D (float, optional): dipole moment. Defaults to 1.4103753e-29 Coulomb meter for
                            the X to X TlF transition.

    Returns:
        float: power [W]
    """
    return rabi_to_power_gaussian_beam(rabi, coupling, sigma_x, sigma_y, D)
