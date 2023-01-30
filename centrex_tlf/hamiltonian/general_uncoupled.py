from centrex_tlf.states import State, UncoupledBasisState

from .constants import HamiltonianConstants
from .quantum_operators import J2

__all__ = ["Hrot"]

########################################################
# Rotational Term
########################################################


def Hrot(psi: UncoupledBasisState, coefficients: HamiltonianConstants) -> State:
    return coefficients.B_rot * J2(psi)
