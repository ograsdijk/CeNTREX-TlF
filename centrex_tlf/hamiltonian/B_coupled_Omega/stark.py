from functools import lru_cache

import numpy as np

from centrex_tlf.states import CoupledBasisState, CoupledState

from ..constants import BConstants
from ..wigner import sixj_f, threej_f


@lru_cache(maxsize=int(1e6))
def d_p(psi: CoupledBasisState, p: int, constants: BConstants) -> CoupledState:
    """
    Operates on psi using the pth spherical tensor component of the
    dipole operator.
    """
    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    # I1, I2 are the same for both states
    I1 = I1p
    I2 = I2p

    # Value of mF changes by p
    mF = mFp + p

    # Omega doesn't change
    Omega = Omegap

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over possible values of Jprime
    for J in np.arange(np.abs(Jp - 1), Jp + 2):
        # Loop over possible values of F1
        for F1 in np.arange(np.abs(J - I1), J + I1 + 1):
            # Loop over possible values of F
            for F in np.arange(np.abs(F1 - I2), F1 + I2 + 1):
                amp = (
                    constants.μ_E
                    * (-1) ** (F + Fp + F1 + F1p + I1 + I2 - Omega - mF)
                    * np.sqrt(
                        (2 * F + 1)
                        * (2 * Fp + 1)
                        * (2 * F1 + 1)
                        * (2 * F1p + 1)
                        * (2 * J + 1)
                        * (2 * Jp + 1)
                    )
                    * threej_f(F, 1, Fp, -mF, p, mFp)
                    * threej_f(J, 1, Jp, -Omega, 0, Omegap)
                    * sixj_f(F1p, Fp, I2, F, F1, 1)
                    * sixj_f(Jp, F1p, I1, F1, J, 1)
                )

                basis_state = CoupledBasisState(
                    F,
                    mF,
                    F1,
                    J,
                    I1,
                    I2,
                    Omega=Omega,
                    electronic_state=psi.electronic_state,
                    P=psi.P,
                )
                if amp != 0:
                    data.append((amp, basis_state))

    return CoupledState(data)


@lru_cache(maxsize=int(1e6))
def HSx(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Stark Hamiltonian operator for x-component of electric field
    """
    return -(d_p(psi, -1, constants) - d_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HSy(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Stark Hamiltonian operator for y-component of electric field
    """
    return -1j * (d_p(psi, -1, constants) + d_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HSz(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Stark Hamiltonian for z-component of electric field
    """
    return -d_p(psi, 0, constants)
