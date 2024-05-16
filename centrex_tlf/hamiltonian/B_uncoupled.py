import numpy as np

from centrex_tlf.states import UncoupledBasisState, UncoupledState

from .constants import BConstants
from .wigner import threej_f

__all__ = ["H_LD", "H_c1p", "H_mhf_Tl", "H_mhf_F", "HZx", "HZy", "HZz"]


########################################################
# Λ doubling term
########################################################


def H_LD(psi: UncoupledBasisState, coefficients: BConstants) -> UncoupledState:
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    if Omega is None:
        raise TypeError("can only have Omega be int (not None)")
    Omegaprime = -Omega

    amp = (
        coefficients.q
        * (-1) ** (J - Omegaprime)
        / (2 * np.sqrt(6))
        * threej_f(J, 2, J, -Omegaprime, Omegaprime - Omega, Omega)
        * np.sqrt((2 * J - 1) * 2 * J * (2 * J + 1) * (2 * J + 2) * (2 * J + 3))
    )
    ket = UncoupledBasisState(
        J, mJ, I1, m1, I2, m2, Omegaprime, electronic_state=psi.electronic_state
    )

    return UncoupledState([(amp, ket)])


########################################################
# C'(Tl) - term (Brown 1978 "A determination of fundamental Zeeman parameters
# for the OH radical", eqn A12)
########################################################


def H_c1p(psi: UncoupledBasisState, coefficients: BConstants) -> UncoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    if Omega is None:
        raise TypeError("can only have Omega be int (not None)")

    # I1, I2 and m2 must be the same for non-zero matrix element
    I1prime = I1
    m2prime = m2
    I2prime = I2

    # To have non-zero matrix element need OmegaPrime = -Omega
    Omegaprime = -Omega

    # q is chosen such that q == Omegaprime
    q = Omega

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix element can
    # be non-zero
    # Need Jprime = J+1 ... |J-1|
    for Jprime in range(np.abs(J - 1), J + 2):
        # Loop over possible values of mJprime and m1prime
        for mJprime in np.arange(-Jprime, Jprime + 1):
            # Must have mJ+m1 = mJprime + m1prime
            m1prime = mJ + m1 - mJprime
            if np.abs(m1prime <= I1):
                # Evaluate the matrix element

                # Matrix element for T(J)T(I)
                term1 = (
                    (-1) ** (Jprime - Omegaprime + I1 - m1 - q + mJprime)
                    * np.sqrt(
                        Jprime
                        * (Jprime + 1)
                        * (2 * Jprime + 1) ** 2
                        * (2 * J + 1)
                        * I1
                        * (I1 + 1)
                        * (2 * I1 + 1)
                    )
                    * (threej_f(Jprime, 1, J, -mJprime, mJprime - mJ, mJ))
                    * (threej_f(I1, 1, I1, -m1prime, m1prime - m1, m1))
                    * (threej_f(Jprime, 1, J, 0, -q, Omega))
                    * (threej_f(Jprime, 1, Jprime, -Omegaprime, -q, 0))
                )

                # Matrix element for T(I)T(J)
                term2 = (
                    (-1) ** (mJprime + J - Omegaprime + I1 - m1 - q)
                    * np.sqrt(
                        J
                        * (J + 1)
                        * (2 * J + 1) ** 2
                        * (2 * Jprime + 1)
                        * I1
                        * (I1 + 1)
                        * (2 * I1 + 1)
                    )
                    * (threej_f(Jprime, 1, J, -mJprime, mJprime - mJ, mJ))
                    * (threej_f(Jprime, 1, J, -Omegaprime, -q, 0))
                    * (threej_f(J, 1, J, 0, -q, Omega))
                    * (threej_f(I1, 1, I1, -m1prime, m1prime - m1, m1))
                )

                amp = coefficients.c_Tl * 0.5 * (term1 + term2)

                basis_state = UncoupledBasisState(
                    Jprime,
                    mJprime,
                    I1prime,
                    m1prime,
                    I2prime,
                    m2prime,
                    Omegaprime,
                    P=psi.P,
                    electronic_state=psi.electronic_state,
                )

                if amp != 0:
                    data.append((amp, basis_state))

    return UncoupledState(data)


########################################################
# Electron magnetic hyperfine operator
########################################################


def H_mhf_Tl(psi: UncoupledBasisState, coefficients: BConstants) -> UncoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    if Omega is None:
        raise TypeError("can only have Omega be int (not None)")

    # I1, I2 and m2 must be the same for non-zero matrix element
    I2prime = I2
    m2prime = m2
    I1prime = I1

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix element can
    # be non-zero
    # Need Jprime = J+1 ... |J-1|
    for Jprime in np.arange(np.abs(J - 1), J + 2):
        # Evaluate the part of the matrix element that is common for all p
        common_coefficient = (
            coefficients.h1_Tl
            * threej_f(J, 1, Jprime, -Omega, 0, Omega)
            * np.sqrt((2 * J + 1) * (2 * Jprime + 1) * I1 * (I1 + 1) * (2 * I1 + 1))
        )

        # Loop over the spherical tensor components of I1:
        for p in np.arange(-1, 2):
            # To have non-zero matrix element need mJ-p = mJprime
            mJprime = mJ + p

            # Also need m2 - p = m2prime
            m1prime = m1 - p

            # Check that mJprime and m2prime are physical
            if np.abs(mJprime) <= Jprime and np.abs(m1prime) <= I1prime:
                # Calculate rest of matrix element
                p_factor = (
                    (-1) ** (p - mJ + I1 - m1 - Omega)
                    * threej_f(J, 1, Jprime, -mJ, -p, mJprime)
                    * threej_f(I1, 1, I1prime, -m1, p, m1prime)
                )

                amp = Omega * common_coefficient * p_factor
                basis_state = UncoupledBasisState(
                    Jprime, mJprime, I1prime, m1prime, I2prime, m2prime, psi.Omega
                )
                if amp != 0:
                    data.append((amp, basis_state))

    return UncoupledState(data)


def H_mhf_F(psi: UncoupledBasisState, coefficients: BConstants) -> UncoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    if Omega is None:
        raise TypeError("can only have Omega be int (not None)")

    # I1, I2 and m1 must be the same for non-zero matrix element
    I1prime = I1
    m1prime = m1
    I2prime = I2

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix element can
    # be non-zero
    # Need Jprime = J+1 ... |J-1|
    for Jprime in np.arange(np.abs(J - 1), J + 2):
        # Evaluate the part of the matrix element that is common for all p
        common_coefficient = (
            coefficients.h1_F
            * threej_f(J, 1, Jprime, -Omega, 0, Omega)
            * np.sqrt((2 * J + 1) * (2 * Jprime + 1) * I2 * (I2 + 1) * (2 * I2 + 1))
        )

        # Loop over the spherical tensor components of I2:
        for p in np.arange(-1, 2):
            # To have non-zero matrix element need mJ-p = mJprime
            mJprime = mJ + p

            # Also need m2 - p = m2prime
            m2prime = m2 - p

            # Check that mJprime and m2prime are physical
            if np.abs(mJprime) <= Jprime and np.abs(m2prime) <= I2prime:
                # Calculate rest of matrix element
                p_factor = (
                    (-1) ** (p - mJ + I2 - m2 - Omega)
                    * threej_f(J, 1, Jprime, -mJ, -p, mJprime)
                    * threej_f(I2, 1, I2prime, -m2, p, m2prime)
                )

                amp = Omega * common_coefficient * p_factor
                basis_state = UncoupledBasisState(
                    Jprime, mJprime, I1prime, m1prime, I2prime, m2prime, psi.Omega
                )
                if amp != 0:
                    data.append((amp, basis_state))

    return UncoupledState(data)


########################################################
# Zeeman B state Hamiltonian
########################################################


def HZx(psi: UncoupledBasisState) -> UncoupledState:
    # TODO
    return UncoupledState([(1.0, psi)])


def HZy(psi: UncoupledBasisState) -> UncoupledState:
    # TODO
    return UncoupledState([(1.0, psi)])


def HZz(psi: UncoupledBasisState, coefficients: BConstants) -> UncoupledState:
    # Find the quantum numbers of the input state
    J = psi.J
    mJ = psi.mJ
    I1 = psi.I1
    m1 = psi.m1
    I2 = psi.I2
    m2 = psi.m2
    Omega = psi.Omega
    if Omega is None:
        raise TypeError("can only have Omega be int (not None)")
    S = 1

    # The other state must have the same value for I1,m1,I2,m2,mJ and Omega
    I1prime = I1
    m1prime = m1
    I2prime = I2
    m2prime = m2
    Omegaprime = Omega
    mJprime = mJ

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over possible values of Jprime
    for Jprime in range(np.abs(J - 1), J + 2):
        # Electron orbital angular momentum term
        L_term = (
            coefficients.gL
            * Omega
            * np.sqrt((2 * J + 1) * (2 * Jprime + 1))
            * (-1) ** (mJprime - Omegaprime)
            * (threej_f(Jprime, 1, J, -mJprime, 0, mJ))
            * (threej_f(Jprime, 1, J, -Omegaprime, 0, Omega))
        )

        # Electron spin term
        S_term = (
            coefficients.gS
            * np.sqrt((2 * J + 1) * (2 * Jprime + 1))
            * (-1) ** (mJprime - Omegaprime)
            * (threej_f(Jprime, 1, J, -mJprime, 0, mJ))
            * (threej_f(Jprime, 1, J, -Omegaprime, 0, Omega))
            * (-1) ** (S)
            * (threej_f(S, 1, S, 0, 0, 0))
            * np.sqrt(S * (S + 1) * (2 * S + 1))
        )

        amp = L_term + S_term
        basis_state = UncoupledBasisState(
            Jprime, mJprime, I1prime, m1prime, I2prime, m2prime, Omegaprime
        )

        if amp != 0:
            data.append((amp, basis_state))

    return UncoupledState(data)
