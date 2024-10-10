import math
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from centrex_tlf import hamiltonian, states

__all__ = ["calculate_ED_ME_mixed_state", "ED_ME_coupled"]


def calculate_ED_ME_mixed_state(
    bra: states.CoupledState,
    ket: states.CoupledState,
    pol_vec: npt.NDArray[np.complex128] = np.array(
        [1.0, 1.0, 1.0], dtype=np.complex128
    ),
    reduced: bool = False,
    normalize_pol: bool = True,
) -> complex:
    ME = 0j

    # Transform to Omega basis if required. For the X state the basis is Coupled and
    # doesn't require to be transformed to the Omega basis, since Omega is 0.
    if bra.largest.basis is states.Basis.CoupledP:
        bra = bra.transform_to_omega_basis()
    if ket.largest.basis is states.Basis.CoupledP:
        ket = ket.transform_to_omega_basis()

    if normalize_pol:
        pol_vec /= np.linalg.norm(pol_vec)

    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += (
                amp_bra.conjugate()
                * amp_ket
                * ED_ME_coupled(
                    basis_bra, basis_ket, pol_vec=tuple(pol_vec), rme_only=reduced
                )
            )

    return ME


@lru_cache(maxsize=int(1e6))
def ED_ME_coupled(
    bra: states.CoupledBasisState,
    ket: states.CoupledBasisState,
    pol_vec: Tuple[complex, complex, complex] = (1.0 + 0j, 1.0 + 0j, 1.0 + 0j),
    rme_only: bool = False,
) -> complex:
    """calculate electric dipole matrix elements between coupled basis states

    Args:
        bra (CoupledBasisState): coupled basis state object
        ket (CoupledBasisState): coupled basis state object
        pol_vec (Tuple[float, float, float]): polarization vector.
                                        Defaults to np.array([1,1,1]).
        rme_only (bool, optional): set True to return only reduced matrix
                                    element, otherwise angular component is
                                    included. Defaults to False.

    Returns:
        complex: electric dipole matrix element between bra en ket
    """

    # find quantum numbers for ground state
    F = bra.F
    mF = bra.mF
    J = bra.J
    F1 = bra.F1
    I1 = bra.I1
    I2 = bra.I2
    Omega = bra.Omega

    # find quantum numbers for excited state
    Fp = ket.F
    mFp = ket.mF
    Jp = ket.J
    F1p = ket.F1
    Omegap = ket.Omega

    # calculate the reduced matrix element
    q = Omega - Omegap
    ME: complex = (
        (-1) ** (F1 + J + Fp + F1p + I1 + I2)
        * math.sqrt((2 * F + 1) * (2 * Fp + 1) * (2 * F1p + 1) * (2 * F1 + 1))
        * hamiltonian.sixj_f(F1p, Fp, I2, F, F1, 1)
        * hamiltonian.sixj_f(Jp, F1p, I1, F1, J, 1)
        * (-1) ** (J - Omega)
        * math.sqrt((2 * J + 1) * (2 * Jp + 1))
        * hamiltonian.threej_f(J, 1, Jp, -Omega, q, Omegap)
        * float(np.abs(q) < 2)
    )

    # if we want the complete matrix element, calculate angular part
    if not rme_only:
        # calculate elements of the polarization vector in spherical basis
        p_vec: Dict[int, complex] = {}
        p_vec[-1] = -1 / math.sqrt(2) * (pol_vec[0] + 1j * pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1 / math.sqrt(2) * (pol_vec[0] - 1j * pol_vec[1])

        # calculate the value of p that connects the states
        p = mF - mFp
        p = p * int(np.abs(p) <= 1)
        # multiply RME by the angular part
        ME = (
            ME
            * (-1) ** (F - mF)
            * hamiltonian.threej_f(F, 1, Fp, -mF, p, mFp)
            * p_vec[p]
            * int(np.abs(p) <= 1)
        )

    # return the matrix element
    return ME
