from functools import lru_cache

from centrex_tlf.constants import BConstants
from centrex_tlf.states import CoupledBasisState, CoupledState

from ..quantum_operators import J2, J4, J6


@lru_cache(maxsize=int(1e6))
def Hrot(psi: CoupledBasisState, constants: BConstants) -> CoupledState:
    """
    Rotational Hamiltonian for the B-state.
    """
    return (
        constants.B_rot * J2(psi)
        + constants.D_rot * J4(psi)
        + constants.H_const * J6(psi)
    )
