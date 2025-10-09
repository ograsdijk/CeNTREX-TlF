from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

__all__ = [
    "Polarization",
    "polarization_X",
    "polarization_Y",
    "polarization_Z",
    "polarization_σp",
    "polarization_σm",
]


def format_value(val: complex) -> str:
    """Format a complex number for display, showing only real part if imaginary is zero.

    Args:
        val: Complex number to format

    Returns:
        String representation of the value
    """
    if val.imag == 0:
        return f"{val.real}"
    return f"{val}"


def decompose(vector: npt.NDArray[np.complex128]) -> str:
    """Decompose a polarization vector into X, Y, Z components.

    Args:
        vector: 3-element array representing polarization [X, Y, Z]

    Returns:
        String representation like "1.0 X + 0.0 Y + 0.0 Z"
    """
    x, y, z = vector

    x_str = format_value(x)
    y_str = format_value(y)
    z_str = format_value(z)

    return f"{x_str} X + {y_str} Y + {z_str} Z"


@dataclass
class Polarization:
    """Represents a light polarization state.

    Attributes:
        vector: 3-element complex array representing polarization in [X, Y, Z] basis
        name: Human-readable name for the polarization state

    Examples:
        >>> pol_x = Polarization(np.array([1, 0, 0], dtype=np.complex128), "X")
        >>> pol_circular = Polarization(
        ...     np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0], dtype=np.complex128), "σ+"
        ... )
    """

    vector: npt.NDArray[np.complex128]
    name: str

    def __post_init__(self) -> None:
        """Validate that vector has the correct shape."""
        if self.vector.shape != (3,):
            raise ValueError(
                f"Polarization vector must have shape (3,), got {self.vector.shape}"
            )

    def __repr__(self) -> str:
        return f"Polarization({self.name})"

    def __mul__(self, value: int | float | complex) -> Self:
        """Multiply polarization vector by a scalar.

        Args:
            value: Scalar to multiply by

        Returns:
            New Polarization with scaled vector

        Raises:
            TypeError: If value is not a number
        """
        if not isinstance(value, (int, float, complex)):
            raise TypeError(
                f"Cannot multiply Polarization by {type(value).__name__}, "
                f"expected int, float, or complex"
            )
        vec_new = self.vector * value
        new_name = decompose(vec_new)
        return self.__class__(vector=vec_new, name=new_name)

    def __rmul__(self, value: int | float | complex) -> Self:
        """Right multiplication: value * polarization."""
        return self.__mul__(value)

    def __add__(self, other: Self) -> Self:
        """Add two polarization vectors.

        Args:
            other: Another Polarization to add

        Returns:
            New Polarization with summed vector
        """
        vec_new = self.vector + other.vector
        new_name = decompose(vec_new)
        return self.__class__(vector=vec_new, name=new_name)

    def normalize(self) -> Self:
        """Normalize the polarization vector to unit length.

        Returns:
            New Polarization with normalized vector

        Raises:
            ValueError: If the vector has zero norm
        """
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            raise ValueError("Cannot normalize zero-length polarization vector")
        vec_new = self.vector / norm
        new_name = decompose(vec_new)
        return self.__class__(vector=vec_new, name=new_name)


polarization_X = Polarization(np.array([1, 0, 0], dtype=np.complex128), "X")
polarization_Y = Polarization(np.array([0, 1, 0], dtype=np.complex128), "Y")
polarization_Z = Polarization(np.array([0, 0, 1], dtype=np.complex128), "Z")
polarization_σp = Polarization(
    np.array([-1 / np.sqrt(2), 1j / np.sqrt(2), 0], dtype=np.complex128), "σp"
)

polarization_σm = Polarization(
    np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0], dtype=np.complex128), "σm"
)
