from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Sequence, Union

import numpy as np
import numpy.typing as npt

from centrex_tlf.states import (
    Basis,
    CoupledBasisState,
    CoupledState,
    UncoupledBasisState,
)

from . import B_coupled_Omega, X_uncoupled
from .constants import BConstants, HamiltonianConstants, XConstants

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
    result = np.zeros((len(QN), len(QN)), dtype=complex)
    for i, a in enumerate(QN):
        for j in range(i, len(QN)):
            b = QN[j]
            val = (1 * a) @ H(b, constants)
            result[i, j] = val
            if i != j:
                result[j, i] = np.conjugate(val)
    return result


def HMatElemsBCoupledP(
    H: Callable,
    QN: Union[Sequence[CoupledState], npt.NDArray[Any]],
    constants: HamiltonianConstants,
) -> npt.NDArray[np.complex128]:
    result = np.zeros((len(QN), len(QN)), dtype=complex)
    for i, a in enumerate(QN):
        for j in range(i, len(QN)):
            val = 0j
            b = QN[j]
            for ampa, ai in a:
                for ampb, bi in b:
                    val += ampa * ampb * (1 * ai) @ H(bi, constants)
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


def generate_uncoupled_hamiltonian_X(
    QN: Union[
        Sequence[UncoupledBasisState], Sequence[CoupledBasisState], npt.NDArray[Any]
    ],
    constants: XConstants = XConstants(),
) -> HamiltonianUncoupledX:
    """
    Generate the uncoupled X state hamiltonian for the supplied set of
    basis states.

    Args:
        QN (array): array of UncoupledBasisStates

    Returns:
        HamiltonianUncoupledX: dataclass to hold uncoupled X hamiltonian terms
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


def generate_coupled_hamiltonian_B(
    QN: Union[Sequence[CoupledBasisState], npt.NDArray[Any]],
    constants: BConstants = BConstants(),
) -> Union[HamiltonianCoupledBP, HamiltonianCoupledBOmega]:
    """Calculate the coupled B state hamiltonian for the supplied set of
    basis states.

    Args:
        QN (array): array of CoupledBasisStates

    Returns:
        HamiltonianCoupledB: dataclass to hold coupled B hamiltonian terms
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
    else:
        raise AssertionError("QN basis not supported")


def _uncoupled_ham_func_X(
    E: Union[List[float], npt.NDArray[np.float64]],
    B: Union[List[float], npt.NDArray[np.float64]],
    H: HamiltonianUncoupledX,
):
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


def generate_uncoupled_hamiltonian_X_function(H: HamiltonianUncoupledX) -> Callable:
    return partial(_uncoupled_ham_func_X, H=H)


def _coupled_ham_func_B(
    E: Union[List[float], npt.NDArray[np.float64]],
    B: Union[List[float], npt.NDArray[np.float64]],
    H: Union[HamiltonianCoupledBP, HamiltonianCoupledBOmega],
):
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
) -> Callable:
    return partial(_coupled_ham_func_B, H=H)
