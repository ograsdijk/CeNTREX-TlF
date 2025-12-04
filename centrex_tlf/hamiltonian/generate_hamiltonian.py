from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf.constants import BConstants, HamiltonianConstants, XConstants
from centrex_tlf.states import (
    Basis,
    CoupledBasisState,
    CoupledState,
    UncoupledBasisState,
)

from . import B_coupled_Omega, X_uncoupled

try:
    from ..centrex_tlf_rust import (
        generate_coupled_hamiltonian_B_py as _generate_coupled_hamiltonian_B_rust,
    )
    from ..centrex_tlf_rust import (
        generate_uncoupled_hamiltonian_X_py as _generate_uncoupled_hamiltonian_X_rust,
    )

    HAS_RUST = True
except ImportError:
    _generate_coupled_hamiltonian_B_rust = None  # type: ignore[assignment]
    _generate_uncoupled_hamiltonian_X_rust = None  # type: ignore[assignment]
    HAS_RUST = False

__all__ = [
    "Hamiltonian",
    "HamiltonianUncoupledX",
    "HamiltonianCoupledBP",
    "HamiltonianCoupledBOmega",
    "HMatElems",
    "generate_uncoupled_hamiltonian_X",
    "generate_coupled_hamiltonian_B",
    "generate_uncoupled_hamiltonian_X_function",
    "generate_coupled_hamiltonian_B_function",
]


def HMatElems(
    H: Callable,
    QN: Union[
        Sequence[UncoupledBasisState], Sequence[CoupledBasisState], npt.NDArray[Any]
    ],
    constants: HamiltonianConstants,
) -> npt.NDArray[np.complex128]:
    """Calculate Hamiltonian matrix elements in basis QN.

    Computes ⟨i|H|j⟩ for all basis states i,j in QN using the Hamiltonian operator H.

    Args:
        H (Callable): Hamiltonian operator function H(state, constants) -> State
        QN (Sequence[BasisState] | npt.NDArray): Basis states
        constants (HamiltonianConstants): Molecular constants

    Returns:
        npt.NDArray[np.complex128]: Hamiltonian matrix with elements ⟨i|H|j⟩
    """
    result = np.zeros((len(QN), len(QN)), dtype=complex)

    # Pre-compute the Hamiltonian acting on each basis state
    H_QN = [H(qn, constants) for qn in QN]

    for i, a in enumerate(QN):
        for j in range(i, len(QN)):
            val = (1 * a) @ H_QN[j]
            result[i, j] = val
            if i != j:
                result[j, i] = np.conjugate(val)
    return result


def HMatElemsBCoupledP(
    H: Callable,
    QN: Union[Sequence[CoupledState], npt.NDArray[Any]],
    constants: HamiltonianConstants,
) -> npt.NDArray[np.complex128]:
    """Calculate Hamiltonian matrix elements for mixed (superposition) states.

    Computes ⟨ψᵢ|H|ψⱼ⟩ where ψᵢ and ψⱼ are superpositions of basis states.
    Used for B state in parity basis when computing in Omega representation.

    Args:
        H (Callable): Hamiltonian operator function H(state, constants) -> State
        QN (Sequence[CoupledState] | npt.NDArray): Mixed/superposition states
        constants (HamiltonianConstants): Molecular constants

    Returns:
        npt.NDArray[np.complex128]: Hamiltonian matrix with elements ⟨ψᵢ|H|ψⱼ⟩
    """
    result = np.zeros((len(QN), len(QN)), dtype=complex)

    # Pre-compute the Hamiltonian acting on each superposition state
    H_QN = []
    for state in QN:
        h_state = CoupledState()
        for amp, basis_state in state:
            h_state += amp * H(basis_state, constants)
        H_QN.append(h_state)

    for i, a in enumerate(QN):
        for j in range(i, len(QN)):
            val = a @ H_QN[j]
            result[i, j] = val
            if i != j:
                result[j, i] = np.conjugate(val)
    return result


@dataclass
class Hamiltonian:
    None


@dataclass
class HamiltonianUncoupledX(Hamiltonian):
    Hff: npt.NDArray[np.complex128]
    HSx: npt.NDArray[np.complex128]
    HSy: npt.NDArray[np.complex128]
    HSz: npt.NDArray[np.complex128]
    HZx: npt.NDArray[np.complex128]
    HZy: npt.NDArray[np.complex128]
    HZz: npt.NDArray[np.complex128]


@dataclass
class HamiltonianCoupledBP(Hamiltonian):
    Hrot: npt.NDArray[np.complex128]
    H_mhf_Tl: npt.NDArray[np.complex128]
    H_mhf_F: npt.NDArray[np.complex128]
    H_LD: npt.NDArray[np.complex128]
    H_cp1_Tl: npt.NDArray[np.complex128]
    H_c_Tl: npt.NDArray[np.complex128]
    HSx: npt.NDArray[np.complex128]
    HSy: npt.NDArray[np.complex128]
    HSz: npt.NDArray[np.complex128]
    HZx: npt.NDArray[np.complex128]
    HZy: npt.NDArray[np.complex128]
    HZz: npt.NDArray[np.complex128]


@dataclass
class HamiltonianCoupledBOmega(Hamiltonian):
    Hrot: npt.NDArray[np.complex128]
    H_mhf_Tl: npt.NDArray[np.complex128]
    H_mhf_F: npt.NDArray[np.complex128]
    H_LD: npt.NDArray[np.complex128]
    H_cp1_Tl: npt.NDArray[np.complex128]
    H_c_Tl: npt.NDArray[np.complex128]
    HSx: npt.NDArray[np.complex128]
    HSy: npt.NDArray[np.complex128]
    HSz: npt.NDArray[np.complex128]
    HZx: npt.NDArray[np.complex128]
    HZy: npt.NDArray[np.complex128]
    HZz: npt.NDArray[np.complex128]


def _generate_uncoupled_hamiltonian_X_python(
    QN: Sequence[UncoupledBasisState],
    constants: XConstants = XConstants(),
) -> HamiltonianUncoupledX:
    """Generate the uncoupled X state Hamiltonian for the supplied basis states.

    Constructs all Hamiltonian terms (field-free, Stark, Zeeman) in the uncoupled
    basis for the X (ground) electronic state.

    Args:
        QN (Sequence[UncoupledBasisState] | npt.NDArray): Array of uncoupled basis
            states |J,mJ,I₁,m₁,I₂,m₂⟩
        constants (XConstants): X state molecular constants. Defaults to XConstants().

    Returns:
        HamiltonianUncoupledX: Dataclass containing all X state Hamiltonian
            matrix terms (Hff, HSx, HSy, HSz, HZx, HZy, HZz)
    """
    for qn in QN:
        assert qn.isUncoupled, "supply list with UncoupledBasisStates"

    return HamiltonianUncoupledX(
        HMatElems(X_uncoupled.Hff_alt, QN, constants),
        HMatElems(X_uncoupled.HSx, QN, constants),
        HMatElems(X_uncoupled.HSy, QN, constants),
        HMatElems(X_uncoupled.HSz, QN, constants),
        HMatElems(X_uncoupled.HZx, QN, constants),
        HMatElems(X_uncoupled.HZy, QN, constants),
        HMatElems(X_uncoupled.HZz, QN, constants),
    )


def generate_uncoupled_hamiltonian_X(
    QN: Sequence[UncoupledBasisState],
    constants: XConstants = XConstants(),
) -> HamiltonianUncoupledX:
    """Generate the uncoupled X state Hamiltonian for the supplied basis states.

    Constructs all Hamiltonian terms (field-free, Stark, Zeeman) in the uncoupled
    basis for the X (ground) electronic state.

    Args:
        QN (Sequence[UncoupledBasisState] | npt.NDArray): Array of uncoupled basis
            states |J,mJ,I₁,m₁,I₂,m₂⟩
        constants (XConstants): X state molecular constants. Defaults to XConstants().

    Returns:
        HamiltonianUncoupledX: Dataclass containing all X state Hamiltonian
            matrix terms (Hff, HSx, HSy, HSz, HZx, HZy, HZz)

    Note:
        This function uses a Rust implementation if available for better performance.
    """
    for qn in QN:
        assert qn.isUncoupled, "supply list with UncoupledBasisStates"

    if HAS_RUST and _generate_uncoupled_hamiltonian_X_rust is not None:
        return _generate_uncoupled_hamiltonian_X_rust(QN, constants)
    else:
        return _generate_uncoupled_hamiltonian_X_python(QN, constants)


def _generate_coupled_hamiltonian_B_python(
    QN: Union[Sequence[CoupledBasisState], npt.NDArray[Any]],
    constants: BConstants = BConstants(),
) -> Union[HamiltonianCoupledBP, HamiltonianCoupledBOmega]:
    """Generate the coupled B state Hamiltonian for the supplied basis states.

    Constructs all Hamiltonian terms (rotational, hyperfine, lambda-doubling, Stark,
    Zeeman) in the coupled basis for the B (excited) electronic state. Supports both
    parity (P) and Omega (Ω) basis representations.

    Args:
        QN (Sequence[CoupledBasisState] | npt.NDArray): Array of coupled basis
            states. Can be in parity basis |J,F,F₁,mF,P⟩ or Omega basis
            |J,F,F₁,mF,Ω⟩
        constants (BConstants): B state molecular constants. Defaults to BConstants().

    Returns:
        HamiltonianCoupledBP | HamiltonianCoupledBOmega: Dataclass containing all
            B state Hamiltonian matrix terms. Returns HamiltonianCoupledBP for
            parity basis or HamiltonianCoupledBOmega for Omega basis.

    """
    return HamiltonianCoupledBOmega(
        HMatElems(B_coupled_Omega.rotational.Hrot, QN, constants),
        HMatElems(B_coupled_Omega.mhf.H_mhf_Tl, QN, constants),
        HMatElems(B_coupled_Omega.mhf.H_mhf_F, QN, constants),
        HMatElems(B_coupled_Omega.ld.H_LD, QN, constants),
        HMatElems(B_coupled_Omega.ld.H_cp1_Tl, QN, constants),
        HMatElems(B_coupled_Omega.nsr.H_c_Tl, QN, constants),
        HMatElems(B_coupled_Omega.stark.HSx, QN, constants),
        HMatElems(B_coupled_Omega.stark.HSy, QN, constants),
        HMatElems(B_coupled_Omega.stark.HSz, QN, constants),
        HMatElems(B_coupled_Omega.zeeman.HZx, QN, constants),
        HMatElems(B_coupled_Omega.zeeman.HZy, QN, constants),
        HMatElems(B_coupled_Omega.zeeman.HZz, QN, constants),
    )


def generate_coupled_hamiltonian_B(
    QN: Union[Sequence[CoupledBasisState], npt.NDArray[Any]],
    constants: BConstants = BConstants(),
) -> Union[HamiltonianCoupledBP, HamiltonianCoupledBOmega]:
    """Generate the coupled B state Hamiltonian for the supplied basis states.

    Constructs all Hamiltonian terms (rotational, hyperfine, lambda-doubling, Stark,
    Zeeman) in the coupled basis for the B (excited) electronic state. Supports both
    parity (P) and Omega (Ω) basis representations.

    Args:
        QN (Sequence[CoupledBasisState] | npt.NDArray): Array of coupled basis
            states. Can be in parity basis |J,F,F₁,mF,P⟩ or Omega basis
            |J,F,F₁,mF,Ω⟩
        constants (BConstants): B state molecular constants. Defaults to BConstants().

    Returns:
        HamiltonianCoupledBP | HamiltonianCoupledBOmega: Dataclass containing all
            B state Hamiltonian matrix terms. Returns HamiltonianCoupledBP for
            parity basis or HamiltonianCoupledBOmega for Omega basis.

    Note:
        This function uses a Rust implementation if available for better performance
        when using the Omega basis.
    """
    for qn in QN:
        assert qn.isCoupled, "supply list withCoupledBasisStates"
    if all([qn.basis == Basis.CoupledP for qn in QN]):
        # raise NotImplementedError(
        #     "Generating the hamiltonian in the CoupledP basis is not yet implemented."
        # )
        QN_omega = [s.transform_to_omega_basis() for s in QN]

        return HamiltonianCoupledBOmega(
            HMatElemsBCoupledP(B_coupled_Omega.rotational.Hrot, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.mhf.H_mhf_Tl, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.mhf.H_mhf_F, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.ld.H_LD, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.ld.H_cp1_Tl, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.nsr.H_c_Tl, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.stark.HSx, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.stark.HSy, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.stark.HSz, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.zeeman.HZx, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.zeeman.HZy, QN_omega, constants),
            HMatElemsBCoupledP(B_coupled_Omega.zeeman.HZz, QN_omega, constants),
        )
    elif all([qn.basis == Basis.CoupledΩ for qn in QN]):
        if HAS_RUST and _generate_coupled_hamiltonian_B_rust is not None:
            return _generate_coupled_hamiltonian_B_rust(QN, constants)
        else:
            return _generate_coupled_hamiltonian_B_python(QN, constants)
    else:
        raise AssertionError("QN basis not supported")


def _uncoupled_ham_func_X(
    E: Union[List[float], npt.NDArray[np.float64]],
    B: Union[List[float], npt.NDArray[np.float64]],
    H: HamiltonianUncoupledX,
) -> npt.NDArray[np.complex128]:
    return (
        2
        * np.pi
        * (
            H.Hff
            + E[0] * H.HSx
            + E[1] * H.HSy
            + E[2] * H.HSz
            + B[0] * H.HZx
            + B[1] * H.HZy
            + B[2] * H.HZz
        )
    )


def generate_uncoupled_hamiltonian_X_function(
    H: HamiltonianUncoupledX,
) -> Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.complex128]
]:
    """Create function for X state Hamiltonian that depends on E and B fields.

    Returns a function H(E, B) that computes the total X state Hamiltonian for
    given electric and magnetic field vectors.

    Args:
        H (HamiltonianUncoupledX): Pre-computed X state Hamiltonian terms

    Returns:
        Callable: Function H(E, B) -> Hamiltonian matrix
    """
    return partial(_uncoupled_ham_func_X, H=H)


def _coupled_ham_func_B(
    E: Union[List[float], npt.NDArray[np.float64]],
    B: Union[List[float], npt.NDArray[np.float64]],
    H: Union[HamiltonianCoupledBP, HamiltonianCoupledBOmega],
) -> npt.NDArray[np.complex128]:
    return (
        2
        * np.pi
        * (
            H.Hrot
            + H.H_mhf_Tl
            + H.H_mhf_F
            + H.H_LD
            + H.H_cp1_Tl
            + H.H_c_Tl
            + E[0] * H.HSx
            + E[1] * H.HSy
            + E[2] * H.HSz
            + B[0] * H.HZx
            + B[1] * H.HZy
            + B[2] * H.HZz
        )
    )


def generate_coupled_hamiltonian_B_function(
    H: Union[HamiltonianCoupledBP, HamiltonianCoupledBOmega],
) -> Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.complex128]
]:
    """Create function for B state Hamiltonian that depends on E and B fields.

    Returns a function H(E, B) that computes the total B state Hamiltonian for
    given electric and magnetic field vectors.

    Args:
        H (HamiltonianCoupledBP | HamiltonianCoupledBOmega): Pre-computed B state
            Hamiltonian terms

    Returns:
        Callable: Function H(E, B) -> Hamiltonian matrix
    """
    return partial(_coupled_ham_func_B, H=H)
