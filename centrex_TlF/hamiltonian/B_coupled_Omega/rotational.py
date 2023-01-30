from functools import lru_cache

from centrex_tlf.states import CoupledBasisState, State

from ..constants import BConstants
from ..quantum_operators import J2, J4, J6


@lru_cache(maxsize=int(1e6))
def Hrot(psi: CoupledBasisState, constants: BConstants) -> State:
    """
    Rotational Hamiltonian for the B-state.
    """
    return (
        constants.B_rot * J2(psi)
        + constants.D_rot * J4(psi)
        + constants.H_const * J6(psi)
    )
