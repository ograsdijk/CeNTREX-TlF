from functools import lru_cache

import numpy as np

from centrex_tlf.states import CoupledBasisState, State

from ..constants import BConstants
from ..wigner import sixj_f, threej_f


@lru_cache(maxsize=int(1e6))
def H_LD(psi: CoupledBasisState, constants: BConstants) -> State:
    """
    Calculates the "q-term" that couples states with opposite Omega
    shifting e-parity up and f-parity down in energy
    """
    # All quantum numbers the same, except Omega inverts sign
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    Omega = -psi.Omega

    amp = constants.q * J * (J + 1) / 2
    ket = CoupledBasisState(
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

    return State([(amp, ket)])


@lru_cache(maxsize=int(1e6))
def H_cp1_Tl(psi: CoupledBasisState, constants: BConstants) -> State:
    """
    Calculates the lambda-doubling nuclear spin - rotation term
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
    F1 = F1p
    mF = mFp

    # Omegas are opposite
    Omega = -Omegap

    # Calculate the value of q
    q = Omega

    data = []

    # Loop over possible values of J
    # Need J = |Jp-1| ... |Jp+1|
    for J in np.arange(np.abs(Jp - 1), Jp + 2):
        # Calculate matrix element
        amp = (
            -constants.c1p_Tl
            / 2
            * (-1) ** (J + Jp + F1 + I1 - Omegap)
            * sixj_f(I1, Jp, F1, J, I1, 1)
            * np.sqrt(I1 * (I1 + 1) * (2 * I1 + 1) * (2 * J + 1) * (2 * Jp + 1))
            * (
                (-1) ** (J)
                * np.sqrt(J * (J + 1) * (2 * J + 1))
                * threej_f(J, 1, Jp, 0, q, Omegap)
                * threej_f(J, 1, J, -Omega, q, 0)
                + (-1) ** (Jp)
                * np.sqrt(Jp * (Jp + 1) * (2 * Jp + 1))
                * threej_f(Jp, 1, Jp, 0, q, Omegap)
                * threej_f(J, 1, Jp, -Omega, q, 0)
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

        data.append((amp, basis_state))

    return State(data)
