from typing import Literal, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import sympy as smp

from .utils import generate_density_matrix_symbolic

__all__ = ["generate_system_of_equations_symbolic"]


@overload
def generate_system_of_equations_symbolic(
    hamiltonian: smp.Matrix,
    C_array: npt.NDArray[np.floating | np.complexfloating],  # 3D array
    fast: bool,
    split_output: Literal[False],
) -> smp.Matrix: ...


@overload
def generate_system_of_equations_symbolic(
    hamiltonian: smp.Matrix,
    C_array: npt.NDArray[np.floating | np.complexfloating],  # 3D array
    fast: bool,
) -> smp.Matrix: ...


@overload
def generate_system_of_equations_symbolic(
    hamiltonian: smp.Matrix,
    C_array: npt.NDArray[np.floating | np.complexfloating],  # 3D array
    fast: bool,
    split_output: Literal[True],
) -> Tuple[smp.Matrix, smp.Matrix]: ...


def generate_system_of_equations_symbolic(
    hamiltonian: smp.Matrix,
    C_array: npt.NDArray[np.floating | np.complexfloating],  # 3D array
    fast: bool = False,
    split_output: bool = False,
) -> Union[smp.Matrix, Tuple[smp.Matrix, smp.Matrix]]:
    n_states: int = hamiltonian.shape[0]  # Explicitly annotate n_states
    density_matrix: smp.Matrix = generate_density_matrix_symbolic(n_states)

    # Ensure C_array is complex for consistency
    if not np.iscomplexobj(C_array):
        C_array = C_array.astype(np.complex128)

    C_conj_array: npt.NDArray[np.floating | np.complexfloating] = np.einsum(
        "ijk->ikj",
        C_array.conj(),  # type: ignore[arg-type]
    )

    # Initialize the matrix for the summation
    matrix_mult_sum: smp.Matrix = smp.zeros(n_states, n_states)

    if fast:
        # Sparse matrix multiplication optimization
        for C, C_conj in zip(C_array, C_conj_array):
            nonzero_C = np.nonzero(C)
            nonzero_C_conj = np.nonzero(C_conj)
            value = (
                C[nonzero_C][0]
                * C_conj[nonzero_C_conj][0]
                * density_matrix[nonzero_C[-1], nonzero_C_conj[0]][0]
            )
            matrix_mult_sum[nonzero_C[0][0], nonzero_C_conj[-1][0]] += value
    else:
        # Full matrix multiplication
        for idx in range(C_array.shape[0]):
            matrix_mult_sum += C_array[idx] @ density_matrix @ C_conj_array[idx]

    # Precompute terms for Lindblad operators
    C_precalc: npt.NDArray[np.floating | np.complexfloating] = np.einsum(
        "ijk,ikl",
        C_conj_array,  # type: ignore[arg-type]
        C_array,  # type: ignore[arg-type]
    )
    lindblad_term: smp.Matrix = -0.5 * (
        C_precalc @ density_matrix + density_matrix @ C_precalc
    )

    # Compute Hamiltonian contribution
    hamiltonian_term: smp.Matrix = -1j * (
        hamiltonian @ density_matrix - density_matrix @ hamiltonian
    )

    if split_output:
        return hamiltonian_term, matrix_mult_sum + lindblad_term
    else:
        system: smp.Matrix = smp.zeros(n_states, n_states)
        system += matrix_mult_sum + lindblad_term + hamiltonian_term
        return system
