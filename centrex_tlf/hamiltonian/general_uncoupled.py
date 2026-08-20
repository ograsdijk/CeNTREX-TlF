from centrex_tlf.constants import HamiltonianConstants
from centrex_tlf.states import UncoupledBasisState, UncoupledState

from .quantum_operators import J2, J4

__all__ = ["Hrot", "Hrot_rigid"]

########################################################
# Rotational Term
########################################################


def Hrot_rigid(psi: UncoupledBasisState, coefficients: HamiltonianConstants) -> UncoupledState:
    """Rigid-rotor rotational Hamiltonian in uncoupled basis.

    H_rot,rigid = B·J²

    This is the plain rigid-rotor operator, without any centrifugal
    distortion correction. It exists separately from `Hrot` because the
    algebraic tensor-spin-spin identity `Hc3c` in `X_uncoupled.py` depends
    on this exact rigid-rotor factor and must not pick up the distortion
    term if `Hrot` is extended.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state |J,mJ,I₁,m₁,I₂,m₂⟩
        coefficients (HamiltonianConstants): Molecular constants (B_rot)

    Returns:
        UncoupledState: Rigid-rotor rotational energy contribution
    """
    return coefficients.B_rot * J2(psi)


def Hrot(psi: UncoupledBasisState, coefficients: HamiltonianConstants) -> UncoupledState:
    """Rotational Hamiltonian in uncoupled basis.

    H_rot = B·J² - D·[J(J+1)]²

    `D_rot` is part of the common `HamiltonianConstants` interface (both the
    X and B states carry a quartic centrifugal-distortion constant), so it is
    always applied here.

    Args:
        psi (UncoupledBasisState): Uncoupled basis state |J,mJ,I₁,m₁,I₂,m₂⟩
        coefficients (HamiltonianConstants): Molecular constants (B_rot, D_rot)

    Returns:
        UncoupledState: Rotational energy contribution
    """
    return coefficients.B_rot * J2(psi) - coefficients.D_rot * J4(psi)
