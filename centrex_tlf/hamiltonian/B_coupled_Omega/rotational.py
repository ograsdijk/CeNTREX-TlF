"""Rotational Hamiltonian operators for B state in coupled Omega basis.

This module implements the rotational energy terms for the B electronic state,
including the rigid rotor term and centrifugal distortion corrections up to
sixth order in J.

The rotational energy of a molecule arises from its rotation about axes perpendicular
to the internuclear axis. For a non-rigid rotor, centrifugal distortion introduces
correction terms proportional to higher powers of J.
"""

from __future__ import annotations

from functools import lru_cache

from centrex_tlf.constants import BConstants
from centrex_tlf.states import CoupledBasisState, CoupledState

from ..quantum_operators import J2, J4, J6


@lru_cache(maxsize=int(1e6))
def Hrot(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """Rotational Hamiltonian for B electronic state including centrifugal distortion.

    Calculates the rotational energy of the molecule including rigid rotor (B·J²)
    and centrifugal distortion terms (D·J⁴ and H·J⁶). The operator is diagonal in
    the |J,F₁,F,mF,Ω⟩ basis with eigenvalues depending on J(J+1).

    The Hamiltonian is:
        H_rot = B·J² - D·J⁴ + H·J⁶

    where J² = J(J+1), and higher powers include additional angular momentum factors.

    Args:
        psi: Input coupled basis state |J,F₁,F,mF,I₁,I₂,Ω⟩ in B electronic state.
        constants: Molecular constants for TlF B state containing:
            - B_rot: Rotational constant (rigid rotor), ~6.688 GHz for TlF B state
            - D_rot: Quartic centrifugal distortion constant, ~10.9 kHz
            - H_const: Sextic centrifugal distortion constant, ~-0.081 Hz

    Returns:
        CoupledState containing the rotational energy contribution. Since J², J⁴,
        and J⁶ are diagonal operators, the returned state is proportional to the
        input state with amplitude given by the eigenvalue.

    Notes:
        - All terms are diagonal in J (ΔJ = 0)
        - The rotational energy increases with J but is reduced by centrifugal
          distortion at high J
        - For TlF B state, D_rot/B_rot ≈ 1.6×10⁻⁶ (small correction)
        - H_const is needed only for very high J states (J > 10)
        - The negative sign on D_rot reflects that centrifugal forces stretch
          the bond, reducing the effective rotational constant

    References:
        Herzberg, G. (1950). "Molecular Spectra and Molecular Structure I. Spectra
        of Diatomic Molecules." Van Nostrand, New York.

    Example:
        >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
        >>> constants = BConstants(B_rot=6687879000.0, D_rot=10869.0, H_const=-0.081)
        >>> result = Hrot(state, constants)
        >>> # Returns energy B·J(J+1) - D·[J(J+1)]² + H·[J(J+1)]³
    """
    return (
        constants.B_rot * J2(psi)
        - constants.D_rot * J4(psi)
        + constants.H_const * J6(psi)
    )
