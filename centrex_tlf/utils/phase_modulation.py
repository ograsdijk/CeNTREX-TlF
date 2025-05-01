from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import special


def sideband_spectrum(
    β: float, ω: float, kmax: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate the sideband spectrum of a phase modulation EOM.

    The spectrum is generated using:
        J₀(β) + Σ Jₖ(β) + Σ (-1)ᵏ Jₖ(β),
    summing over k from 0 to kmax, where the first sum is for peaks at positive detuning
    and the second sum is for peaks at negative detuning.

    Args:
        β (float): Modulation index.
        ω (float): Frequency of the modulation (rad/s).
        kmax (int): Maximum number of sidebands to compute.

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            - Array of frequencies (rad/s).
            - Array of sideband amplitudes.
    """
    # Input validation
    if kmax < 0:
        raise ValueError("kmax must be a non-negative integer.")
    if β < 0 or ω < 0:
        raise ValueError("β and ω must be non-negative.")

    # Generate sideband indices and frequencies
    ks = np.arange(-kmax, kmax + 1, 1)
    ωs = ks * ω

    # Compute sideband amplitudes
    negative_indices = ks < 0
    ks_abs = np.abs(ks)
    sidebands = special.jv(ks_abs, β)
    sidebands[negative_indices] *= (-1) ** ks[negative_indices]

    return ωs, sidebands.astype(np.float64)
