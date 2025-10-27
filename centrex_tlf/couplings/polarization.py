"""Polarization vector representations for optical transitions.

This module defines polarization states for light fields and provides utilities for
manipulating them. Common polarizations (X, Y, Z, σ⁺, σ⁻) are predefined.

The polarization vector is represented in the [Ex, Ey, Ez] basis, where:
    - X, Y, Z are linear polarizations along the respective axes
    - σ⁺ (sigma plus) corresponds to left circular polarization (ΔmF = +1)
    - σ⁻ (sigma minus) corresponds to right circular polarization (ΔmF = -1)

Examples
--------
>>> from centrex_tlf.couplings import polarization_X, polarization_Y
>>> # Combine polarizations
>>> diagonal = (polarization_X + polarization_Y).normalize()
>>> # Scale polarizations
>>> half_x = 0.5 * polarization_X
"""

from dataclasses import dataclass
from fractions import Fraction
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
    "polarization_unpolarized",
]


def _approx_equal(a: float, b: float, tol: float) -> bool:
    """Check if two floats are approximately equal within tolerance.

    Args:
        a: First value
        b: Second value
        tol: Tolerance for comparison

    Returns:
        True if |a - b| <= tol
    """
    return abs(a - b) <= tol


def _rational_str(x: float, tol: float, max_den: int) -> str | None:
    """Return a rational string for x if within tol, else None.

    Args:
        x: Value to convert to rational string
        tol: Tolerance for rational approximation
        max_den: Maximum denominator for fraction

    Returns:
        Rational string representation or None if approximation not close enough
    """
    frac = Fraction(x).limit_denominator(max_den)
    if abs(float(frac) - x) <= tol:
        if frac.denominator == 1:
            return f"{frac.numerator}"
        return f"{frac.numerator}/{frac.denominator}"
    return None


def _known_const_str(x: float, tol: float) -> str | None:
    """Snap to a few handy constants if close.

    Args:
        x: Value to check against known constants
        tol: Tolerance for constant matching

    Returns:
        String representation of known constant or None
    """
    # Note: Only include irrational constants that can't be represented as simple rationals.
    # Values like 0.5 (1/2) or 0.75 (3/4) will be handled by _rational_str,
    # which is checked after this function.
    roots = {
        # Common square roots and their reciprocals
        "1/√2": 1 / np.sqrt(2),
        "1/√3": 1 / np.sqrt(3),
        "1/√5": 1 / np.sqrt(5),
        "1/√6": 1 / np.sqrt(6),
        "√2": np.sqrt(2),
        "√3": np.sqrt(3),
        "√5": np.sqrt(5),
        "√6": np.sqrt(6),
        # Common multiples of square roots
        "2/√3": 2 / np.sqrt(3),
        "√3/2": np.sqrt(3) / 2,
        "√2/3": np.sqrt(2) / 3,
        "2/√5": 2 / np.sqrt(5),
        "√5/2": np.sqrt(5) / 2,
        # Useful fractions of square roots
        "1/√8": 1 / np.sqrt(8),
        "1/√12": 1 / np.sqrt(12),
        "√2/4": np.sqrt(2) / 4,
        "√3/6": np.sqrt(3) / 6,
        # Spherical harmonics and Clebsch-Gordan related
        "√(2/3)": np.sqrt(2 / 3),
        "√(1/3)": np.sqrt(1 / 3),
        "√(3/4)": np.sqrt(3 / 4),
        "√(3/8)": np.sqrt(3 / 8),
        "√(5/8)": np.sqrt(5 / 8),
        # Pi-related constants
        "√π": np.sqrt(np.pi),
        "1/√π": 1 / np.sqrt(np.pi),
    }
    for label, val in roots.items():
        if _approx_equal(abs(x), val, tol):
            return f"-{label}" if x < 0 else label
    return None


def format_value(
    value: complex,
    *,
    zero_tol: float = 1e-12,
    rational_tol: float = 1e-9,
    max_den: int = 32,
    use_known_constants: bool = True,
) -> str:
    """Format a (possibly complex) number with optional rational/constant snapping.

    For complex values, prints as 'a + i b' or 'a - i b'. Signs are included.

    Args:
        value: Complex number to format
        zero_tol: Values with absolute value below this are treated as zero
        rational_tol: Closeness needed to snap to rational representation
        max_den: Maximum denominator complexity for rational representation
        use_known_constants: Enable snapping to 1/√2, 1/√3, etc.

    Returns:
        Formatted string representation of the value

    Examples:
        >>> format_value(1/np.sqrt(2))
        '1/√2'
        >>> format_value(0.5 + 0.5j)
        '1/2 + i 1/2'
    """
    # Treat near-zero as exactly 0
    r = value.real if abs(value.real) > zero_tol else 0.0
    im = value.imag if abs(value.imag) > zero_tol else 0.0

    def _fmt_real(x: float) -> str:
        # Try known constants
        if use_known_constants:
            s = _known_const_str(x, rational_tol)
            if s is not None:
                return s
        # Try rationals
        s = _rational_str(x, rational_tol, max_den)
        if s is not None:
            return s
        # Fall back to a trimmed float
        return f"{x:.6g}"  # short, scientific if needed

    if im == 0.0:
        # Pure real
        return _fmt_real(r)

    if r == 0.0:
        # Pure imaginary → 'i b' with sign inside b
        b = _fmt_real(im)
        # prepend 'i ' but preserve sign of b
        if b.startswith("-"):
            return f"- i {b[1:]}"
        return f"i {b}"

    # General complex: 'a ± i b'
    a = _fmt_real(r)
    b = _fmt_real(abs(im))
    sign = "-" if im < 0 else "+"
    # Ensure signs appear only once (a already has its sign)
    return f"{a} {sign} i {b}"


def decompose(
    vector: npt.NDArray[np.complex128],
    *,
    zero_tol: float = 1e-12,
    rational_tol: float = 1e-9,
    max_den: int = 32,
    use_known_constants: bool = True,
) -> str:
    """Decompose a polarization vector into X, Y, Z components with clean signs and rationalization.

    Args:
        vector: 3-element complex array [Ex, Ey, Ez]
        zero_tol: Values with absolute value below this are treated as zero
        rational_tol: Closeness needed to snap to rational representation
        max_den: Maximum denominator complexity for rational representation
        use_known_constants: Enable snapping to 1/√2, 1/√3, etc.

    Returns:
        Human-readable string decomposition like "1/√2 X + 1/√2 Y"

    Examples:
        >>> vec = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=np.complex128)
        >>> decompose(vec)
        '1/√2 X + 1/√2 Y'
    """
    x, y, z = vector

    def fmt_component(value: complex, label: str, first: bool = False) -> str:
        s = format_value(
            value,
            zero_tol=zero_tol,
            rational_tol=rational_tol,
            max_den=max_den,
            use_known_constants=use_known_constants,
        )
        # Decide sign prefix for non-first components by looking at the leading sign
        if s.startswith("-"):
            return f"- {s[1:]} {label}"
        return f"{s} {label}" if first else f"+ {s} {label}"

    parts = [
        fmt_component(x, "X", first=True),
        fmt_component(y, "Y"),
        fmt_component(z, "Z"),
    ]
    return " ".join(parts)


@dataclass
class Polarization:
    """Represents a light polarization state.

    The polarization is represented as a 3D complex vector in the [Ex, Ey, Ez] basis,
    where Ex, Ey, Ez are the electric field components along the X, Y, Z axes.

    Attributes:
        vector: 3-element complex array representing polarization in [Ex, Ey, Ez] basis
        name: Human-readable name for the polarization state

    Examples:
        >>> import numpy as np
        >>> # Linear X polarization
        >>> pol_x = Polarization(np.array([1, 0, 0], dtype=np.complex128), "X")
        >>>
        >>> # Circular σ+ polarization (ΔmF = +1)
        >>> pol_circular = Polarization(
        ...     np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0], dtype=np.complex128), "σ+"
        ... )
        >>>
        >>> # Combine and normalize polarizations
        >>> diagonal = (pol_x + Polarization(np.array([0, 1, 0], dtype=np.complex128), "Y")).normalize()
        >>>
        >>> # Scale polarization
        >>> half_x = pol_x / 2
        >>> double_x = 2 * pol_x
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
        """Return repr string showing the polarization name."""
        return f"Polarization({self.name})"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"Polarization: {self.name}"

    @property
    def norm(self) -> float:
        """Return the norm (magnitude) of the polarization vector.

        Returns:
            Euclidean norm of the polarization vector
        """
        return float(np.linalg.norm(self.vector))

    @property
    def is_normalized(self) -> bool:
        """Check if the polarization vector is normalized (unit length).

        Returns:
            True if norm is approximately 1, False otherwise
        """
        return _approx_equal(self.norm, 1.0, 1e-10)

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

        Raises:
            TypeError: If other is not a Polarization instance
        """
        if not isinstance(other, Polarization):
            raise TypeError(f"Cannot add {type(other).__name__} to Polarization")
        vec_new = self.vector + other.vector
        new_name = decompose(vec_new)
        return self.__class__(vector=vec_new, name=new_name)

    def __sub__(self, other: Self) -> Self:
        """Subtract two polarization vectors.

        Args:
            other: Another Polarization to subtract

        Returns:
            New Polarization with difference vector
        """
        if not isinstance(other, Polarization):
            raise TypeError(f"Cannot subtract {type(other).__name__} from Polarization")

        vec_new = self.vector - other.vector
        new_name = decompose(vec_new)
        return self.__class__(vector=vec_new, name=new_name)

    def __neg__(self) -> Self:
        """Negate the polarization vector.

        Returns:
            New Polarization with negated vector
        """
        return self.__mul__(-1)

    def __truediv__(self, value: int | float | complex) -> Self:
        """Divide polarization vector by a scalar.

        Args:
            value: Scalar to divide by

        Returns:
            New Polarization with scaled vector

        Raises:
            TypeError: If value is not a number
            ZeroDivisionError: If value is zero
        """
        if not isinstance(value, (int, float, complex)):
            raise TypeError(
                f"Cannot divide Polarization by {type(value).__name__}, "
                f"expected int, float, or complex"
            )
        if value == 0:
            raise ZeroDivisionError("Cannot divide polarization by zero")
        return self.__mul__(1 / value)

    def normalize(self) -> Self:
        """Normalize the polarization vector to unit length.

        Returns:
            New Polarization with normalized vector

        Raises:
            ValueError: If the vector has zero norm
        """
        norm = np.linalg.norm(self.vector)
        # Use small tolerance to avoid floating point precision issues
        if norm < 1e-15:
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
# used in branching ratio calculations, averaging over q=-1,0,+1 polarizations
polarization_unpolarized = (
    np.sqrt(2 / 3) * polarization_X + np.sqrt(1 / 3) * polarization_Z
)
