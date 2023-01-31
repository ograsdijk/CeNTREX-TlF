import numpy as np
import scipy.constants as cst

__all__ = [
    "intensity_to_electric_field",
    "electric_field_to_rabi",
    "power_to_rabi_gaussian_beam",
    "power_to_rabi_gaussian_beam_microwave",
    "rabi_to_electric_field",
    "electric_field_to_intensity",
    "rabi_to_power_gaussian_beam",
    "rabi_to_power_gaussian_beam_microwave",
]


def intensity_to_electric_field(intensity: float) -> float:
    """
    Intensity in W/m^2 to electric field

    Args:
        intensity (float): intensity [W/m^2]

    Returns:
        float: electric field
    """
    return np.sqrt((2 / (cst.c * cst.epsilon_0)) * intensity)


def electric_field_to_rabi(electric_field: float, coupling: float, D: float) -> float:
    """
    Rabi rate from an electric field an coupling strength, with the dipole moment D
    default value set to the X to B transition.

    Args:
        electric_field (float): electric field
        coupling (float): coupling strength
        D (float, optional): Dipole moment.

    Returns:
        float: Rabi rate in rotational frequency [2π ⋅ Hz]
    """
    return electric_field * coupling * D / cst.hbar


def power_to_rabi_gaussian_beam(
    power: float,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = 2.6675506e-30,
) -> float:
    intensity = power / (2 * np.pi * sigma_x * sigma_y)

    electric_field = intensity_to_electric_field(intensity)
    rabi_rate = electric_field_to_rabi(
        electric_field=electric_field, coupling=coupling, D=D
    )
    return rabi_rate


def power_to_rabi_gaussian_beam_microwave(
    power: float,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = 1.4103753e-29,
) -> float:
    """
    Rabi rate from an electric field and coupling strength for the X to X microwave
    transition

    Args:
        electric_field (float): electric field
        coupling (float): coupling strength
        D (float, optional): Dipole moment. Defaults to 1.4103753e-29 for the X to X
                            TlF transition.

    Returns:
        float: Rabi rate in rotational frequency [2π ⋅ Hz]
    """
    return power_to_rabi_gaussian_beam(
        power=power, coupling=coupling, sigma_x=sigma_x, sigma_y=sigma_y, D=D
    )


def rabi_to_electric_field(rabi: float, coupling: float, D: float) -> float:
    """
    Electric field from a given rabi rate and coupling strength

    Args:
        rabi (float): rabi rate in rotational frequency [2π ⋅ Hz]
        coupling (float): coupling strength
        D (float): Dipole moment

    Returns:
        float: electric field
    """
    return rabi * cst.hbar / (coupling * D)


def electric_field_to_intensity(electric_field: float) -> float:
    """
    Intensity in W/m^2 from a given electric field

    Args:
        electric_field (float): electric field

    Returns:
        float: intensity [W/m^2]
    """
    return 1 / 2 * cst.c * cst.epsilon_0 * electric_field**2


def rabi_to_power_gaussian_beam(
    rabi: float,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = 2.6675506e-30,
) -> float:
    """
    power in W given a rabi rate and couling strength

    Args:
        rabi (float): rabi rate in rotational frequency [2π ⋅ Hz]
        coupling (float): coupling strength
        D (float, optional): dipole moment. Defaults to 2.6675506e-30 for the X to B TlF
                            transition..

    Returns:
        float: power [W]
    """
    electric_field = rabi_to_electric_field(rabi, coupling, D)
    intensity = electric_field_to_intensity(electric_field)
    power = intensity * 2 * np.pi * sigma_x * sigma_y
    return power


def rabi_to_power_gaussian_beam_microwave(
    rabi: float,
    coupling: float,
    sigma_x: float,
    sigma_y: float,
    D: float = 1.4103753e-29,
) -> float:
    """
    power in W given a rabi rate and couling strength

    Args:
        rabi (float): rabi rate in rotational frequency [2π ⋅ Hz]
        coupling (float): coupling strength
        D (float, optional): dipole moment. Defaults to 1.4103753e-29 for the X to X TlF
                            transition.

    Returns:
        float: power [W]
    """
    return rabi_to_power_gaussian_beam(rabi, coupling, sigma_x, sigma_y, D)
