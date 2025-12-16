from typing import Any, Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf.constants import BConstants, XConstants
from centrex_tlf.hamiltonian import HamiltonianCoupledBOmega, HamiltonianUncoupledX
from centrex_tlf.states import BasisState, CoupledBasisState, UncoupledBasisState

def generate_uncoupled_hamiltonian_X_py(
    states: Sequence[UncoupledBasisState], constants: XConstants
) -> HamiltonianUncoupledX:
    """
    Generate the uncoupled X state Hamiltonian for the supplied basis states using Rust.

    Args:
        states (Sequence[UncoupledBasisState]): Array of uncoupled basis states.
        constants (XConstants): X state molecular constants.

    Returns:
        HamiltonianUncoupledX: Dataclass containing all X state Hamiltonian matrix terms.
    """
    ...

def generate_coupled_hamiltonian_B_py(
    states: Sequence[CoupledBasisState], constants: BConstants
) -> HamiltonianCoupledBOmega:
    """
    Generate the coupled B state Hamiltonian for the supplied basis states using Rust.

    Args:
        states (Sequence[CoupledBasisState]): Array of coupled basis states.
        constants (BConstants): B state molecular constants.

    Returns:
        HamiltonianCoupledBOmega: Dataclass containing all B state Hamiltonian matrix terms.
    """
    ...

def wigner_3j_py(
    j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
) -> float:
    """
    Calculate the Wigner 3j symbol using Rust.

    Args:
        j1 (float): Angular momentum 1.
        j2 (float): Angular momentum 2.
        j3 (float): Angular momentum 3.
        m1 (float): Projection 1.
        m2 (float): Projection 2.
        m3 (float): Projection 3.

    Returns:
        float: The value of the Wigner 3j symbol.
    """
    ...

def wigner_6j_py(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
) -> float:
    """
    Calculate the Wigner 6j symbol using Rust.

    Args:
        j1 (float): Angular momentum 1.
        j2 (float): Angular momentum 2.
        j3 (float): Angular momentum 3.
        j4 (float): Angular momentum 4.
        j5 (float): Angular momentum 5.
        j6 (float): Angular momentum 6.

    Returns:
        float: The value of the Wigner 6j symbol.
    """
    ...

def generate_transform_matrix_py(
    basis1: Union[Sequence[BasisState], npt.NDArray[Any]],
    basis2: Union[Sequence[BasisState], npt.NDArray[Any]],
) -> npt.NDArray[np.complex128]:
    """
    Generate transformation matrix between two quantum state bases using Rust.

    Computes the transformation matrix S where S[i,j] = <basis1[i]|basis2[j]>.

    Args:
        basis1 (Sequence[BasisState] | npt.NDArray): First basis.
        basis2 (Sequence[BasisState] | npt.NDArray): Second basis.

    Returns:
        npt.NDArray[np.complex128]: Transformation matrix S.
    """
    ...

def generate_coupling_matrix_py(
    QN: Sequence[CoupledBasisState],
    ground_states: Sequence[CoupledBasisState],
    excited_states: Sequence[CoupledBasisState],
    pol_vec: Union[npt.NDArray[np.complex128], Sequence[complex]],
    reduced: bool = False,
) -> npt.NDArray[np.complex128]:
    """
    Generate optical coupling matrix for transitions between quantum states (Rust binding).

    Args:
        QN (Sequence[CoupledBasisState]): Complete list of basis states defining the Hilbert space.
        ground_states (Sequence[CoupledBasisState]): Ground states that couple to excited states.
        excited_states (Sequence[CoupledBasisState]): Excited states that couple to ground states.
        pol_vec (np.ndarray | Sequence[complex]): Polarization vector [Ex, Ey, Ez] (complex).
        reduced (bool): If True, return only reduced matrix elements (no angular part).

    Returns:
        npt.NDArray[np.complex128]: Hermitian coupling matrix of shape (n, n) where n = len(QN).
    """
    ...

