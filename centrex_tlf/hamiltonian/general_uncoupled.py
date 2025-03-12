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
    return coefficients.B_rot * J2(psi)
