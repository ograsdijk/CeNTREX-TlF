from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import special

__all__ = ["sideband_spectrum"]


def sideband_spectrum(
    β: float, ω: float, kmax: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate the sideband spectrum of a phase modulation EOM.

    The spectrum is generated using Bessel functions of the first kind:
        Amplitude(k·ω) = Jₖ(β) for k ≥ 0
        Amplitude(-k·ω) = (-1)ᵏ Jₖ(β) for k > 0

    where Jₖ is the Bessel function of the first kind of order k.

    Args:
        β (float): Modulation index (dimensionless).
        ω (float): Angular frequency of the modulation (rad/s).
        kmax (int): Maximum sideband order to compute (non-negative integer).

    Returns:
        Tuple containing:
            - frequencies: Array of sideband frequencies (rad/s), shape (2*kmax + 1,)
            - amplitudes: Array of corresponding sideband amplitudes, shape (2*kmax + 1,)

    Raises:
        ValueError: If kmax is negative, or if β or ω are negative.
        TypeError: If kmax is not an integer.
    """
    # Input validation
    if not isinstance(kmax, (int, np.integer)):
        raise TypeError(f"kmax must be an integer, got {type(kmax).__name__}")
    if kmax < 0:
        raise ValueError(f"kmax must be non-negative, got {kmax}")
    if β < 0:
        raise ValueError(f"β must be non-negative, got {β}")
    if ω < 0:
        raise ValueError(f"ω must be non-negative, got {ω}")

    # Generate sideband indices from -kmax to +kmax
    ks = np.arange(-kmax, kmax + 1, dtype=np.int64)

    # Calculate frequencies for each sideband
    frequencies = (ks * ω).astype(np.float64)

    # Compute sideband amplitudes using Bessel functions
    # For negative orders: J_{-k}(β) = (-1)^k J_k(β)
    ks_abs = np.abs(ks)
    amplitudes = special.jv(ks_abs, β).astype(np.float64)

    # Apply phase factor for negative sidebands
    negative_mask = ks < 0
    amplitudes[negative_mask] *= (-1) ** ks_abs[negative_mask]

    return frequencies, amplitudes
