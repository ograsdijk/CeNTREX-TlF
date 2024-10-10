from functools import lru_cache
from typing import Union

from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

__all__ = ["threej_f", "sixj_f"]


@lru_cache(maxsize=int(1e6))
def threej_f(
    j1: Union[float, int],
    j2: Union[float, int],
    j3: Union[float, int],
    m1: Union[float, int],
    m2: Union[float, int],
    m3: Union[float, int],
) -> complex:
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
    j1: Union[float, int],
    j2: Union[float, int],
    j3: Union[float, int],
    j4: Union[float, int],
    j5: Union[float, int],
    j6: Union[float, int],
) -> complex:
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
