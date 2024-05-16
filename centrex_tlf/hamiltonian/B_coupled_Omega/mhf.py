from functools import lru_cache

import numpy as np

from centrex_tlf.states import CoupledBasisState, CoupledState

from ..constants import BConstants
from ..wigner import sixj_f, threej_f


@lru_cache(maxsize=int(1e6))
def H_mhf_Tl(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Operator for magnetic hyperfine term for Tl nucleus.
    """
    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    # I1, I2, F1, F and mF are the same for both states
    I1 = I1p
    I2 = I2p
    F1 = F1p
    F = Fp
    mF = mFp

    # Omega also doesn't change
    Omega = Omegap

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix element can
    # be non-zero
    # Need J = Jp+1 ... |Jp-1|
    for J in np.arange(np.abs(Jp - 1), Jp + 2):
        # Calculate matrix element
        amp = (
            Omega
            * constants.h1_Tl
            * (-1) ** (J + Jp + F1 + I1 - Omega)
            * sixj_f(I1, Jp, F1, J, I1, 1)
            * threej_f(J, 1, Jp, -Omega, 0, Omegap)
            * np.sqrt((2 * J + 1) * (2 * Jp + 1) * I1 * (I1 + 1) * (2 * I1 + 1))
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
def H_mhf_F(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Operator for magnetic hyperfine term for F nucleus.
    """
    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    # I1, I2, F and mF are the same for both states
    I1 = I1p
    I2 = I2p
    F = Fp
    mF = mFp

    # Omega also doesn't change
    Omega = Omegap

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix element can
    # be non-zero
    # Need J = Jp+1 ... |Jp-1|
    for J in np.arange(np.abs(Jp - 1), Jp + 2):
        # F1 can be J +/- 1/2
        for F1 in np.arange(np.abs(J - 1 / 2), J + 3 / 2):
            # Calculate matrix element
            amp = (
                Omega
                * constants.h1_F
                * (-1) ** (2 * F1p + F + 2 * J + I1 + I2 - Omega + 1)
                * sixj_f(I2, F1p, F, F1, I2, 1)
                * sixj_f(Jp, F1p, I1, F1, J, 1)
                * threej_f(J, 1, Jp, -Omega, 0, Omegap)
                * np.sqrt(
                    (2 * F1 + 1)
                    * (2 * F1p + 1)
                    * (2 * J + 1)
                    * (2 * Jp + 1)
                    * I2
                    * (I2 + 1)
                    * (2 * I2 + 1)
                )
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
