from centrex_tlf.constants import HamiltonianConstants
from centrex_tlf.states import UncoupledBasisState, UncoupledState

from .quantum_operators import J2

__all__ = ["Hrot"]

########################################################
# Rotational Term
########################################################


def Hrot(
    psi: UncoupledBasisState, coefficients: HamiltonianConstants
) -> UncoupledState:
    """Rotational Hamiltonian in uncoupled basis.
    
    H_rot = B·J²
    
    Args:
        psi (UncoupledBasisState): Uncoupled basis state |J,mJ,I₁,m₁,I₂,m₂⟩
        coefficients (HamiltonianConstants): Molecular constants (B_rot)
    
    Returns:
        UncoupledState: Rotational energy contribution
    """
    return coefficients.B_rot * J2(psi)
