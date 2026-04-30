"""Build the X-state uncoupled basis and a field-evaluating H(E, B) callable.

Wraps centrex_tlf APIs. The returned H callable returns a complex Hermitian
matrix in **rad/s** — propagate with `exp(-1j * D * dt)` directly (no hbar).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from centrex_tlf.hamiltonian import (
    generate_uncoupled_hamiltonian_X,
    generate_uncoupled_hamiltonian_X_function,
)
from centrex_tlf.states import generate_uncoupled_states_ground

HFunc = Callable[
    [Sequence[float] | npt.NDArray[np.float64], Sequence[float] | npt.NDArray[np.float64]],
    npt.NDArray[np.complex128],
]


def build_basis(Jmax: int = 6) -> npt.NDArray:
    """Enumerate the X-state uncoupled basis up to and including J=Jmax."""
    if Jmax < 0:
        raise ValueError("Jmax must be >= 0")
    return generate_uncoupled_states_ground(list(range(Jmax + 1)))


def build_H_func(QN: npt.NDArray) -> HFunc:
    """Return H(E, B) -> ndarray in rad/s for the given basis QN.

    E is a 3-vector in V/cm; B is a 3-vector in Gauss.
    """
    H_uncoupled = generate_uncoupled_hamiltonian_X(QN)
    return generate_uncoupled_hamiltonian_X_function(H_uncoupled)
