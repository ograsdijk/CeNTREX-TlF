"""Cached wrappers for Wigner 3j and 6j symbols.

Provides efficient cached float-valued wrappers around sympy's Wigner symbol functions.
Wigner symbols are used extensively in angular momentum coupling calculations.

References:
    - Wigner 3j symbols: Angular momentum coupling coefficients for three angular momenta
    - Wigner 6j symbols: Recoupling coefficients for three angular momenta
"""

from functools import lru_cache

from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

__all__ = ["threej_f", "sixj_f"]


@lru_cache(maxsize=int(1e6))
def threej_f(
    j1: float | int,
    j2: float | int,
    j3: float | int,
    m1: float | int,
    m2: float | int,
    m3: float | int,
) -> complex:
    """Compute Wigner 3j symbol with caching.

    The Wigner 3j symbol is related to Clebsch-Gordan coefficients and represents
    the coupling of three angular momenta.

            ⎛ j1  j2  j3 ⎞
            ⎜            ⎟
            ⎝ m1  m2  m3 ⎠

    Args:
        j1 (float | int): First angular momentum quantum number
        j2 (float | int): Second angular momentum quantum number
        j3 (float | int): Third angular momentum quantum number
        m1 (float | int): First magnetic quantum number
        m2 (float | int): Second magnetic quantum number
        m3 (float | int): Third magnetic quantum number

    Returns:
        complex: Wigner 3j symbol value (though mathematically real)

    Note:
        Results are cached for performance. The function accepts half-integer
        values as floats (e.g., 0.5 for 1/2). Returns complex type for
        compatibility with complex arithmetic in the codebase.
    """
    return complex(
        wigner_3j(
            Rational(j1),
            Rational(j2),
            Rational(j3),
            Rational(m1),
            Rational(m2),
            Rational(m3),
        )
    )


@lru_cache(maxsize=int(1e6))
def sixj_f(
    j1: float | int,
    j2: float | int,
    j3: float | int,
    j4: float | int,
    j5: float | int,
    j6: float | int,
) -> complex:
    """Compute Wigner 6j symbol with caching.

    The Wigner 6j symbol represents the recoupling of three angular momenta
    and is used in transformations between different coupling schemes.

            ⎧ j1  j2  j3 ⎫
            ⎨            ⎬
            ⎩ j4  j5  j6 ⎭

    Args:
        j1 (float | int): First angular momentum quantum number
        j2 (float | int): Second angular momentum quantum number
        j3 (float | int): Third angular momentum quantum number
        j4 (float | int): Fourth angular momentum quantum number
        j5 (float | int): Fifth angular momentum quantum number
        j6 (float | int): Sixth angular momentum quantum number

    Returns:
        complex: Wigner 6j symbol value (though mathematically real)

    Note:
        Results are cached for performance. The function accepts half-integer
        values as floats (e.g., 0.5 for 1/2). Returns complex type for
        compatibility with complex arithmetic in the codebase.
    """
    return complex(
        wigner_6j(
            Rational(j1),
            Rational(j2),
            Rational(j3),
            Rational(j4),
            Rational(j5),
            Rational(j6),
        )
    )
