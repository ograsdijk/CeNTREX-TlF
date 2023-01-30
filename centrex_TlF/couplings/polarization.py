from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "polarization_X",
    "polarization_Y",
    "polarization_Z",
    "polarization_σp",
    "polarization_σm",
]


@dataclass
class Polarization:
    vector: npt.NDArray[np.complex_]
    name: str

    def __repr__(self) -> str:
        return f"Polarization({self.name})"


polarization_X = Polarization(np.array([1, 0, 0], dtype=np.complex128), "X")
polarization_Y = Polarization(np.array([0, 1, 0], dtype=np.complex128), "Y")
polarization_Z = Polarization(np.array([0, 0, 1], dtype=np.complex128), "Z")
polarization_σp = Polarization(
    np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0], dtype=np.complex128), "σp"
)
polarization_σm = Polarization(
    np.array([1 / np.sqrt(2), -1j / np.sqrt(2), 0], dtype=np.complex128), "σm"
)
