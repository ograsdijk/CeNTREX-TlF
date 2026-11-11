"""Cached wrappers for Wigner 3j and 6j symbols.

Provides cached float-valued Wigner symbols for the angular momentum coupling
calculations. A symbol is evaluated once per process and then served from the cache.
The first evaluation uses the Rust extension when it is available and every angular
momentum is at most `RUST_MAX_J`; otherwise it uses sympy's exact arithmetic.

Why the cut-off: for the symbols this package needs (one small angular momentum, e.g.
(J, 1, J')) the Rust values agree with sympy to 5e-16 up to J = 20 and 5e-15 at J = 40,
and then degrade slowly (about 1e-12 near J = 100). Symbols with three large angular
momenta lose accuracy faster and the Rust code refuses them from about J = 50. Below the
cut-off Rust is about 100x faster per first evaluation (1.6 vs 156 microseconds), which
is what a fresh process pays for every distinct symbol.

References:
    - Wigner 3j symbols: Angular momentum coupling coefficients for three angular momenta
    - Wigner 6j symbols: Recoupling coefficients for three angular momenta
"""

from functools import lru_cache

from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

try:
    from ..centrex_tlf_rust import wigner_3j_py as _wigner_3j_rust
    from ..centrex_tlf_rust import wigner_6j_py as _wigner_6j_rust
except ImportError:
    _wigner_3j_rust = None  # type: ignore[assignment]
    _wigner_6j_rust = None  # type: ignore[assignment]

__all__ = ["threej_f", "sixj_f"]

#: Largest angular momentum for which the Rust implementation is used.
RUST_MAX_J = 40


def _threej_sympy(j1, j2, j3, m1, m2, m3) -> complex:
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


def _sixj_sympy(j1, j2, j3, j4, j5, j6) -> complex:
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


def _rust_value(function, js, args) -> complex | None:
    """Rust value of a symbol, or None when sympy should evaluate it.

    None for: no extension, an angular momentum above `RUST_MAX_J`, arguments Rust
    rejects (not multiples of 1/2) and symbols it cannot evaluate accurately. Sympy then
    either returns the exact value or raises its own error, as it always did.
    """
    if function is None:
        return None
    try:
        if max(js) > RUST_MAX_J:
            return None
        return complex(function(*args))
    except (TypeError, ValueError):
        return None


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
        compatibility with complex arithmetic in the codebase. Evaluated in Rust up
        to `RUST_MAX_J`, with sympy beyond that; see the module docstring.
    """
    value = _rust_value(_wigner_3j_rust, (j1, j2, j3), (j1, j2, j3, m1, m2, m3))
    if value is not None:
        return value
    return _threej_sympy(j1, j2, j3, m1, m2, m3)


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
        compatibility with complex arithmetic in the codebase. Evaluated in Rust up
        to `RUST_MAX_J`, with sympy beyond that; see the module docstring.
    """
    value = _rust_value(
        _wigner_6j_rust, (j1, j2, j3, j4, j5, j6), (j1, j2, j3, j4, j5, j6)
    )
    if value is not None:
        return value
    return _sixj_sympy(j1, j2, j3, j4, j5, j6)
