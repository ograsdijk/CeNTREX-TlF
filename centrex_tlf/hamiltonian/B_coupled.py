from functools import lru_cache

import numpy as np

from centrex_tlf.constants import BConstants
from centrex_tlf.states import CoupledBasisState, CoupledState

from .quantum_operators import J2, J4, J6
from .wigner import sixj_f, threej_f

__all__ = [
    "Hrot",
    "H_LD",
    "H_mhf_Tl",
    "H_mhf_F",
    "H_c_Tl",
    "H_cp1_Tl",
    "HZx",
    "HZy",
    "HZz",
    "HSx",
    "HSy",
    "HSz",
]


def Hrot(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    return (
        constants.B_rot * J2(psi)
        - constants.D_rot * J4(psi)
        + constants.H_const * J6(psi)
    )


###################################################
# Λ doubling term
###################################################


def H_LD(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    P = psi.P
    S = 0

    data = []

    def ME(J, Jprime, Omega, Omegaprime):
        amp = (
            constants.q
            * (-1) ** (J - Omegaprime)
            / (2 * np.sqrt(6))
            * threej_f(J, 2, J, -Omegaprime, Omegaprime - Omega, Omega)
            * np.sqrt((2 * J - 1) * 2 * J * (2 * J + 1) * (2 * J + 2) * (2 * J + 3))
        )

        return amp

    for Pprime in [-1, 1]:
        amp = (
            P * (-1) ** (J - S) * ME(J, J, 1, -1)
            + Pprime * (-1) ** (J - S) * ME(J, J, -1, 1)
        ) / 2
        ket = CoupledBasisState(
            F,
            mF,
            F1,
            J,
            I1,
            I2,
            Omega=psi.Omega,
            P=Pprime,
            electronic_state=psi.electronic_state,
        )

        # If matrix element is non-zero, add to list
        if amp != 0:
            data.append((amp, ket))

    return CoupledState(data)


###################################################
# Electron Magnetic Hyperfine Operator
###################################################


@lru_cache(maxsize=int(1e6))
def H_mhf_Tl(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    Omega = psi.Omega
    P = psi.P

    # I1, I2, F1 and F and mF are the same for both states
    I1prime = I1
    I2prime = I2
    F1prime = F1
    mFprime = mF
    Fprime = F

    # Container for the states and amplitudes
    data = []

    # Loop over possible values of Jprime
    for Jprime in np.arange(np.abs(J - 1), J + 2):
        # Check that the Jprime and Fprime values are physical
        if np.abs(Fprime - Jprime) <= (I1 + I2):
            # Calculate matrix element
            try:
                amp = constants.h1_Tl * (
                    (-1) ** (J + Jprime + F1 + I1 - Omega)
                    * sixj_f(I1, Jprime, F1, J, I1, 1)
                    * threej_f(J, 1, Jprime, -Omega, 0, Omega)
                    * np.sqrt(
                        (2 * J + 1) * (2 * Jprime + 1) * I1 * (I1 + 1) * (2 * I1 + 1)
                    )
                )

            except ValueError:
                amp = 0

            basis_state = CoupledBasisState(
                Fprime,
                mFprime,
                F1prime,
                Jprime,
                I1prime,
                I2prime,
                Omega=psi.Omega,
                P=P,
                electronic_state=psi.electronic_state,
            )

            # If matrix element is non-zero, add to list
            if amp != 0:
                data.append((amp, basis_state))

    return CoupledState(data)


@lru_cache(maxsize=int(1e6))
def H_mhf_F(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    Omega = psi.Omega
    P = psi.P

    # I1, I2, F and mF are the same for both states
    I1prime = I1
    I2prime = I2
    Fprime = F
    mFprime = mF

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix
    # element can be non-zero
    # Need Jprime = J+1 ... |J-1|
    for Jprime in np.arange(np.abs(J - 1), J + 2):
        # Loop over possible values of F1prime
        for F1prime in np.arange(np.abs(Jprime - I1), Jprime + I1 + 1):
            try:
                amp = constants.h1_F * (
                    (-1) ** (2 * F1prime + F + 2 * J + 1 + I1 + I2 - Omega)
                    * sixj_f(I2, F1prime, F, F1, I2, 1)
                    * sixj_f(Jprime, F1prime, I1, F1, J, 1)
                    * threej_f(J, 1, Jprime, -Omega, 0, Omega)
                    * np.sqrt(
                        (2 * F1 + 1)
                        * (2 * F1prime + 1)
                        * (2 * J + 1)
                        * (2 * Jprime + 1)
                        * I2
                        * (I2 + 1)
                        * (2 * I2 + 1)
                    )
                )

            except ValueError:
                amp = 0

            basis_state = CoupledBasisState(
                Fprime,
                mFprime,
                F1prime,
                Jprime,
                I1prime,
                I2prime,
                P=P,
                Omega=psi.Omega,
                electronic_state=psi.electronic_state,
            )

            # If matrix element is non-zero, add to list
            if amp != 0:
                data.append((amp, basis_state))

    return CoupledState(data)


###################################################
# C(Tl) - term
###################################################


def H_c_Tl(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF

    # I1, I2, F and mF are the same for both states
    Jprime = J
    I1prime = I1
    I2prime = I2
    Fprime = F
    F1prime = F1
    mFprime = mF

    # Initialize container for storing states and matrix elements
    data = []

    # Calculate matrix element
    amp = (
        constants.c_Tl
        * (-1) ** (J + F1 + I1)
        * sixj_f(I1, J, F1, J, I1, 1)
        * np.sqrt(J * (J + 1) * (2 * J + 1) * I1 * (I1 + 1) * (2 * I1 + 1))
    )

    basis_state = CoupledBasisState(
        Fprime,
        mFprime,
        F1prime,
        Jprime,
        I1prime,
        I2prime,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )

    # If matrix element is non-zero, add to list
    if amp != 0:
        data.append((amp, basis_state))

    return CoupledState(data)


@lru_cache(maxsize=int(1e6))
def H_cp1_Tl(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    # Omega = psi.Omega
    P = psi.P

    # I1, I2, F and mF are the same for both states
    I1prime = I1
    I2prime = I2
    Fprime = F
    F1prime = F1
    mFprime = mF

    # Total spin is 1
    S = 0

    # Omegaprime is negative of Omega
    # Omegaprime = -Omega

    # Calculate the correct value of q
    # q = Omegaprime

    # Initialize container for storing states and matrix elements
    data = []

    def ME(J, Jprime, Omega, Omegaprime):
        q = Omegaprime
        amp = (
            -0.5
            * constants.c1p_Tl
            * (-1) ** (-J + Jprime - Omegaprime + F1 + I1)
            * np.sqrt((2 * Jprime + 1) * (2 * J + 1) * I1 * (I1 + 1) * (2 * I1 + 1))
            * sixj_f(I1, J, F1, Jprime, I1, 1)
            * (
                (-1) ** (J)
                * threej_f(Jprime, 1, J, -Omegaprime, q, 0)
                * threej_f(J, 1, J, 0, q, Omega)
                * np.sqrt(J * (J + 1) * (2 * J + 1))
                + (
                    (-1) ** (Jprime)
                    * threej_f(Jprime, 1, Jprime, -Omegaprime, q, 0)
                    * threej_f(Jprime, 1, J, 0, q, Omega)
                    * np.sqrt(Jprime * (Jprime + 1) * (2 * Jprime + 1))
                )
            )
        )

        return amp

    # Loop over the possible values of quantum numbers for which the matrix
    # element can be non-zero
    # Need Jprime = J+1 ... |J-1|
    for Jprime in range(np.abs(J - 1), J + 2):
        for Pprime in [-1, 1]:
            amp = (
                (
                    P * (-1) ** (J - S) * ME(J, Jprime, 1, -1)
                    + Pprime * (-1) ** (Jprime - S) * ME(J, Jprime, -1, 1)
                )
                * (-1) ** float((J - Jprime) != 0)
                / 2
            )

            ket = CoupledBasisState(
                Fprime,
                mFprime,
                F1prime,
                Jprime,
                I1prime,
                I2prime,
                Omega=psi.Omega,
                P=Pprime,
                electronic_state=psi.electronic_state,
            )

            # If matrix element is non-zero, add to list
            if amp != 0:
                data.append((amp, ket))

    return CoupledState(data)


@lru_cache(maxsize=int(1e6))
def mu_p(psi: CoupledBasisState, p: int, constants: BConstants) -> CoupledState:
    """
    Operates on psi using the pth spherical tensor component of the magnetic
    dipole operator.
    """
    # Some constants
    gL = 1

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
    for J in np.arange(np.abs(Jp - 1), Jp + 2).tolist():
        # Loop over possible values of F1
        for F1 in np.arange(np.abs(J - I1), J + I1 + 1).tolist():
            # Loop over possible values of F
            for F in np.arange(np.abs(F1 - I2), F1 + I2 + 1).tolist():
                amp = (
                    gL
                    * Omega
                    * constants.μ_B
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
def HZx(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Zeeman Hamiltonian operator for x-component of magnetic field
    """
    return -(mu_p(psi, -1, constants) - mu_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HZy(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Zeeman Hamiltonian operator for y-component of magnetic field
    """
    return -1j * (mu_p(psi, -1, constants) + mu_p(psi, +1, constants)) / np.sqrt(2)


@lru_cache(maxsize=int(1e6))
def HZz(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Zeeman Hamiltonian for z-component of magnetic field
    """
    return -mu_p(psi, 0, constants)


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
    for J in np.arange(np.abs(Jp - 1), Jp + 2).tolist():
        # Loop over possible values of F1
        for F1 in np.arange(np.abs(J - I1), J + I1 + 1).tolist():
            # Loop over possible values of F
            for F in np.arange(np.abs(F1 - I2), F1 + I2 + 1).tolist():
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
