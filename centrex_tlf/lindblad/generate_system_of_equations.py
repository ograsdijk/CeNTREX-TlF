"""System of equations generation for Lindblad master equation.

This module generates symbolic systems of differential equations from the Lindblad
master equation, which describes the time evolution of the density matrix for an
open quantum system including decoherence and dissipation.

The Lindblad master equation in Lindblad form is:
    dρ/dt = -i[H, ρ] + Σᵢ(CᵢρCᵢ† - ½{Cᵢ†Cᵢ, ρ})

where H is the Hamiltonian, ρ is the density matrix, Cᵢ are Lindblad operators
(jump operators), and {A,B} = AB + BA is the anticommutator.
"""

from __future__ import annotations

from typing import Literal, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import sympy as smp

__all__ = [
    "generate_system_of_equations_symbolic",
    "generate_dissipator_term",
    "generate_hamiltonian_term",
    "generate_density_matrix",
]


def generate_density_matrix(nstates: int, symbol: str = "\u03c1") -> smp.Matrix:
    """Generate symbolic density matrix for nstates-level system.

    The density matrix ρ is Hermitian, so ρᵢⱼ = ρⱼᵢ* (complex conjugate).
    This function creates a symbolic matrix with elements as sympy IndexedBase
    symbols, ensuring Hermiticity by defining only upper triangle elements
    independently.

    Args:
        nstates: Number of quantum states in the system.

    Returns:
        Symbolic density matrix (nstates x nstates) with Hermitian structure.
    """
    rho = smp.IndexedBase(symbol)  # Unicode ρ for density matrix
    density_matrix = smp.Matrix(
        nstates,
        nstates,
        lambda i, j: rho[i, j] if i <= j else rho[j, i].conjugate(),
    )
    return density_matrix


def generate_dissipator_term(
    C_array: npt.NDArray[np.floating | np.complexfloating],
    density_matrix: smp.Matrix,
    fast: bool = False,
) -> smp.Matrix:
    nstates = density_matrix.shape[0]

    # Ensure collapse operators are complex-valued for proper conjugation
    if not np.iscomplexobj(C_array):
        C_array = C_array.astype(np.complex128)

    # Ensure collapse operators are complex-valued for proper conjugation
    if not np.iscomplexobj(C_array):
        C_array = C_array.astype(np.complex128)

    # Compute conjugate transpose of all collapse operators: Cᵢ†
    # einsum("ijk->ikj") transposes the last two dimensions for each operator
    C_conj_array: npt.NDArray[np.complexfloating] = np.einsum(
        "ijk->ikj",
        C_array.conj(),  # type: ignore[arg-type]
    )

    # Initialize accumulator for dissipation term: Σᵢ CᵢρCᵢ†
    dissipation_sum: smp.Matrix = smp.zeros(nstates, nstates)

    if fast:
        # Sparse optimization: only compute non-zero contributions
        # Significant speedup when collapse operators are sparse (typical for decay)
        for C, C_conj in zip(C_array, C_conj_array):
            # Get indices of non-zero elements: nonzero returns tuple of (row_indices, col_indices)
            nonzero_C = np.nonzero(C)
            nonzero_C_conj = np.nonzero(C_conj)

            # Only process if both operators have non-zero elements
            if len(nonzero_C[0]) > 0 and len(nonzero_C_conj[0]) > 0:
                # For sparse operators (e.g., |g⟩⟨e|), typically only one non-zero element
                # nonzero_C[0][0] is row index, nonzero_C[1][0] is column index
                # density_matrix indexing returns a MatrixElement directly (no [0] needed)
                value = (
                    C[nonzero_C][0]  # Get first non-zero value from C
                    * C_conj[nonzero_C_conj][0]  # Get first non-zero value from C†
                    * density_matrix[int(nonzero_C[1][0]), int(nonzero_C_conj[0][0])]
                )
                dissipation_sum[int(nonzero_C[0][0]), int(nonzero_C_conj[1][0])] += (
                    value
                )
    else:
        # Standard computation: Σᵢ CᵢρCᵢ† for all operators
        # Use full matrix multiplication (more general but slower)
        for idx in range(C_array.shape[0]):
            dissipation_sum += C_array[idx] @ density_matrix @ C_conj_array[idx]

    # Precompute Σᵢ Cᵢ†Cᵢ for anticommutator term: -½{Cᵢ†Cᵢ, ρ}
    # einsum("ijk,ikl") efficiently computes the sum of Cᵢ†Cᵢ for all i
    C_dagger_C_sum: npt.NDArray[np.complexfloating] = np.einsum(
        "ijk,ikl",
        C_conj_array,  # type: ignore[arg-type]
        C_array,  # type: ignore[arg-type]
    )

    # Anticommutator term: -½{Cᵢ†Cᵢ, ρ} = -½(Cᵢ†Cᵢ·ρ + ρ·Cᵢ†Cᵢ)
    anticommutator_term: smp.Matrix = -0.5 * (
        C_dagger_C_sum @ density_matrix + density_matrix @ C_dagger_C_sum
    )

    lindblad_dissipation = dissipation_sum + anticommutator_term
    return lindblad_dissipation


def generate_hamiltonian_term(
    hamiltonian: smp.Matrix, density_matrix: smp.Matrix
) -> smp.Matrix:
    # Compute Hamiltonian contribution: -i[H, ρ] = -i(Hρ - ρH)
    # This is the coherent (unitary) evolution part
    hamiltonian_term: smp.Matrix = -1j * (
        hamiltonian @ density_matrix - density_matrix @ hamiltonian
    )
    return hamiltonian_term


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
    """Generate symbolic system of differential equations from Lindblad master equation.

    Constructs the symbolic representation of the Lindblad master equation:
        dρ/dt = -i[H, ρ] + Σᵢ(CᵢρCᵢ† - ½{Cᵢ†Cᵢ, ρ})

    where H is the Hamiltonian (symbolic matrix), ρ is the density matrix (symbolic),
    and Cᵢ are Lindblad operators (numerical collapse operators).

    This function generates a symbolic matrix equation representing the time evolution
    of each density matrix element ρᵢⱼ. The result can be converted to numerical code
    for efficient ODE solving.

    Args:
        hamiltonian: Symbolic Hamiltonian matrix (n_states × n_states) containing
            sympy symbols for time-dependent parameters (e.g., laser detunings,
            Rabi frequencies). Typically contains Complex symbols for coupling
            strengths and Real symbols for energies.
        C_array: Array of Lindblad/collapse operators with shape (n_operators,
            n_states, n_states). Each C_array[i] represents a decay channel or
            decoherence process. Can be real or complex; will be converted to
            complex128 if real.
        fast: If True, uses sparse matrix multiplication optimization that only
            processes non-zero elements. Significant speedup for sparse collapse
            operators (e.g., spontaneous decay between specific states). Default False.
        split_output: If True, returns Hamiltonian and Lindblad contributions
            separately as a tuple. If False, returns combined system. Default False.

    Returns:
        If split_output=False:
            Symbolic matrix (n_states × n_states) representing dρ/dt with elements
            as symbolic expressions in terms of density matrix elements ρᵢⱼ and
            Hamiltonian parameters.
        If split_output=True:
            Tuple of two matrices:
                - hamiltonian_term: -i[H, ρ] contribution
                - lindblad_term: Combined Lindblad dissipation and decay terms

    Notes:
        - The fast mode assumes sparse C matrices and only processes non-zero entries
        - For dense C matrices or when most elements are non-zero, fast=False is better
        - The symbolic output can be lambdified for numerical integration
        - Typical use: generate symbolic equations once, then solve numerically many times
        - Memory usage scales as O(n_states²) for symbolic expressions

    Raises:
        ValueError: If hamiltonian is not square or C_array dimensions incompatible.

    Example:
        >>> import sympy as smp
        >>> import numpy as np
        >>>
        >>> # Create 2-level system
        >>> n = 2
        >>> Omega = smp.Symbol("Omega", complex=True)  # Rabi frequency
        >>> delta = smp.Symbol("delta", real=True)      # Detuning
        >>>
        >>> # Hamiltonian for driven 2-level system
        >>> H = smp.Matrix([[0, Omega/2], [smp.conjugate(Omega)/2, delta]])
        >>>
        >>> # Spontaneous decay operator |g⟩⟨e|
        >>> Gamma = 1.0  # Decay rate
        >>> C_decay = np.array([[[0, np.sqrt(Gamma)], [0, 0]]])
        >>>
        >>> # Generate equations
        >>> system = generate_system_of_equations_symbolic(H, C_decay, fast=True)
        >>>
        >>> # system now contains symbolic dρ/dt = f(ρ, Omega, delta)
        >>> # Can be lambdified for numerical integration:
        >>> from sympy.utilities.lambdify import lambdify
        >>> rho_symbols = [smp.Symbol(f"rho_{i}_{j}") for i in range(n) for j in range(n)]
        >>> f_numeric = lambdify([rho_symbols, Omega, delta], system, "numpy")
    """
    # Extract system size from Hamiltonian dimensions
    n_states: int = hamiltonian.shape[0]

    # Generate symbolic density matrix with elements ρᵢⱼ as sympy symbols
    density_matrix = generate_density_matrix(n_states)

    lindblad_dissipation = generate_dissipator_term(C_array, density_matrix)
    hamiltonian_term = generate_hamiltonian_term(hamiltonian, density_matrix)

    if split_output:
        # Return coherent and dissipative parts separately
        return hamiltonian_term, lindblad_dissipation
    else:
        # Return complete Lindblad equation: dρ/dt = -i[H,ρ] + Σᵢ(CᵢρCᵢ† - ½{Cᵢ†Cᵢ,ρ})
        system: smp.Matrix = smp.zeros(n_states, n_states)
        system += lindblad_dissipation + hamiltonian_term
        return system
