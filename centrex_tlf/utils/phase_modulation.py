from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import special


def sideband_spectrum(
    β: float, ω: float, kmax: int
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Generate the sideband spectrum of a phase modulation eom, generated with:
    J0(β) + ΣJₖ(β) + Σ(-1)ᵏJₖ(β)
    summing over k from 0 to kmax, where the first sum is for peaks at positive detuning
    and the second sum for peaks at negative detuning.

    Args:
        kmax (int): maximum number of sidebands to plot

    Returns:
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]: ndarray with frequencies
        and ndarray with sideband amplitudes
    """
    ks = np.arange(-kmax, kmax + 1, 1)
    ωs = ks * ω

    m = ks < 0
    ks[m] *= -1

    sidebands = special.jv(ks, β)
    sidebands[m] = sidebands[m] * (-1) ** ks[m]

    return ωs, sidebands
