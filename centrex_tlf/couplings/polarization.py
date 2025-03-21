from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

__all__ = [
    "polarization_X",
    "polarization_Y",
    "polarization_Z",
    "polarization_σp",
    "polarization_σm",
]


def format_value(val: complex) -> str:
    # If there is no imaginary part, show only the real component.
    if val.imag == 0:
        return f"{val.real}"
    return f"{val}"


@dataclass
class Polarization:
    vector: npt.NDArray[np.complex128]
    name: str

    def __repr__(self) -> str:
        return f"Polarization({self.name})"

    def __mul__(self, value: int | float | complex) -> Self:
        assert isinstance(value, (int, float, complex))
        vec_new = self.vector * value
        name_new = self._decompose(vec_new)
        return self.__class__(vector=vec_new, name=name_new)

    def __rmul__(self, value: int | float | complex) -> Self:
        return self.__mul__(value)

    def __add__(self, other: Self) -> Self:
        vec_new = self.vector + other.vector
        name_new = self._decompose(vec_new)
        return self.__class__(vector=vec_new, name=name_new)

    def normalize(self) -> Self:
        vec_new = self.vector / np.linalg.norm(self.vector)
        name_new = self._decompose(vec_new)
        return self.__class__(vector=vec_new, name=name_new)

    def _decompose(self, vector: npt.NDArray[np.complex128] | None) -> str:
        if vector is None:
            vector = self.vector
        x, y, z = vector
        x_str = format_value(x)
        y_str = format_value(y)
        z_str = format_value(z)

        name = f"{x_str} X + {y_str} Y + {z_str} Z"
        return name


polarization_X = Polarization(np.array([1, 0, 0], dtype=np.complex128), "X")
polarization_Y = Polarization(np.array([0, 1, 0], dtype=np.complex128), "Y")
polarization_Z = Polarization(np.array([0, 0, 1], dtype=np.complex128), "Z")
polarization_σp = Polarization(
    np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0], dtype=np.complex128), "σp"
)
polarization_σm = Polarization(
    np.array([1 / np.sqrt(2), -1j / np.sqrt(2), 0], dtype=np.complex128), "σm"
)
