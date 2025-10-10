"""Lambda-doubling Hamiltonian operators for B state in coupled Omega basis.

This module implements lambda-doubling interactions in the B electronic state,
including the q-term (Λ-doubling) and nuclear spin-rotation coupling that couples
states with opposite Omega quantum numbers.

The lambda-doubling effect arises from the interaction between electronic and
rotational motion in molecules with non-zero electronic angular momentum.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from centrex_tlf.constants import BConstants
from centrex_tlf.states import CoupledBasisState, CoupledState

from ..wigner import sixj_f, threej_f


@lru_cache(maxsize=int(1e6))
def H_LD(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Lambda-doubling q-term operator for B state in coupled Omega basis.

    Calculates the lambda-doubling "q-term" that couples states with opposite Omega
    values, shifting e-parity levels up and f-parity levels down in energy. This term
    arises from the interaction between electronic orbital motion and molecular rotation.

    The matrix element is diagonal in all quantum numbers except Omega, which changes
    sign: |J,F₁,F,mF,Ω⟩ → |J,F₁,F,mF,-Ω⟩

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state, requires q parameter.

    Returns:
        CoupledState containing the lambda-doubling contribution with amplitude
        q·J(J+1)/2 and basis state with inverted Omega.

    Notes:
        - The operator is off-diagonal in Omega (changes sign)
        - The energy shift is proportional to J(J+1)
        - e-parity levels shift up, f-parity levels shift down

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(q=2423000.0)
        >>> result = H_LD(state, constants)
        >>> # Returns state with Omega=-1 and amplitude q·J(J+1)/2
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

    return CoupledState([(amp, ket)])


@lru_cache(maxsize=int(1e6))
def H_cp1_Tl(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Lambda-doubling nuclear spin-rotation operator for Tl nucleus in B state.

    Calculates the lambda-doubling nuclear spin-rotation interaction term (c'₁) for
    the thallium (Tl) nucleus. This term couples states with different J values and
    opposite Omega values, arising from the interaction between the Tl nuclear spin
    and the molecular rotation in the presence of lambda-doubling.

    This operator is off-diagonal in both J (ΔJ = 0, ±1) and Omega (changes sign).
    The quantum numbers I₁, I₂, F₁, F, and mF remain unchanged.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state, requires c1p_Tl parameter
            (Tl lambda-doubling nuclear spin-rotation constant).

    Returns:
        CoupledState containing sum over all allowed transitions with ΔJ = 0, ±1
        and Omega → -Omega. Each term includes appropriate Wigner 6-j and 3-j
        symbols and angular momentum factors.

    Notes:
        - Couples states with J' = J-1, J, J+1 and Omega' = -Omega
        - The matrix element involves two terms (T(J)T(J) and T(Jp)T(Jp))
        - Typical values for c1p_Tl in TlF B state: ~11 MHz
        - Selection rules: ΔJ = 0, ±1; ΔΩ = 0 but Omega changes sign

    References:
        Brown, J. M., & Carrington, A. (2003). "Rotational Spectroscopy of Diatomic
        Molecules." Cambridge University Press. Chapter on lambda-doubling.

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(c1p_Tl=11170000.0)
        >>> result = H_cp1_Tl(state, constants)
        >>> # Returns superposition of states with J=0,1,2 and Omega=-1
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

    return CoupledState(data)
