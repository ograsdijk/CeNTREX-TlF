from dataclasses import dataclass

import numpy as np

__all__ = ["HamiltonianConstants", "XConstants", "BConstants", "Γ"]

Γ = 2 * np.pi * 1.56e6

# Values for rotational constant are from "Microwave Spectral tables:
# Diatomic molecules" by Lovas & Tiemann (1974).
# Note that Brot differs from the one given by Ramsey by about 30 MHz.

B_ϵ = 6.689873e9
α = 45.0843e6


@dataclass
class HamiltonianConstants:
    B_rot: float


@dataclass(unsafe_hash=True)
class XConstants(HamiltonianConstants):
    B_rot: float = B_ϵ - α / 2
    c1: float = 126030.0
    c2: float = 17890.0
    c3: float = 700.0
    c4: float = -13300.0
    μ_J: float = 35.0  # Hz/G
    μ_Tl: float = 1240.5  # Hz/G
    μ_F: float = 2003.63  # Hz/G
    D_TlF: float = 4.2282 * 503411.7791722602  # Convert Debye to Hz/(V/cm)


@dataclass(unsafe_hash=True)
class BConstants(HamiltonianConstants):
    # Constants in Hz
    B_rot: float = 6687.879e6
    D_rot: float = 0.010869e6
    H_const: float = -8.1e-2
    h1_Tl: float = 28789e6
    h1_F: float = 861e6
    q: float = 2.423e6
    c_Tl: float = -7.83e6
    c1p_Tl: float = 11.17e6
    μ_B: float = 1.4e6
    gL: float = 1
    gS: float = 2
    μ_E = 2.28 * 503411.7791722602  # Convert Debye to Hz/(V/cm)
