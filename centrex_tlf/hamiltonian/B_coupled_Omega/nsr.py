from functools import lru_cache

import numpy as np

from centrex_tlf.states import CoupledBasisState, CoupledState

from ..constants import BConstants
from ..wigner import sixj_f


@lru_cache(maxsize=int(1e6))
def H_c_Tl(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Calculates the effect of the c1 term on the input basis state
    """
    # Find the quantum numbers of the input state
    Jp = psi.J
    I1p = psi.I1
    I2p = psi.I2
    F1p = psi.F1
    Fp = psi.F
    mFp = psi.mF
    Omegap = psi.Omega

    # J, I1, I2, F and mF are the same for both states
    J = Jp
    I1 = I1p
    I2 = I2p
    F = Fp
    F1 = F1p
    mF = mFp

    # Omega also doesn't change
    Omega = Omegap

    # Calculate matrix element
    amp = (
        constants.c_Tl
        * (-1) ** (I1 + F1 + J)
        * sixj_f(Jp, I1, F1, I1, J, 1)
        * np.sqrt(I1 * (I1 + 1) * (2 * I1 + 1) * J * (J + 1) * (2 * J + 1))
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

    return CoupledState([(amp, basis_state)])
