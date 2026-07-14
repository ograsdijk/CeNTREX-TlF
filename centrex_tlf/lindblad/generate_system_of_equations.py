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


def _sympy_number(value: complex) -> smp.Expr:
    """Convert a python/numpy complex to a sympy number, dropping a zero imaginary part."""
    if value.imag == 0.0:
        return smp.Float(value.real)
    return smp.Float(value.real) + smp.Float(value.imag) * smp.I


def generate_dissipator_term(
    C_array: npt.NDArray[np.floating | np.complexfloating],
    density_matrix: smp.Matrix,
    fast: bool = False,
) -> smp.Matrix:
    """Build the symbolic Lindblad dissipator entrywise from operator sparsity.

    Both terms are assembled from nonzero structure instead of dense sympy
    matrix products (which multiply mostly zeros and dominate OBE setup time;
    see IMPLEMENTATION_AUDIT.md "Performance Review (2026-07-11)"):

    - Jump term Σᵢ CᵢρCᵢ†: iterates nonzero entries of each Cᵢ, so a
      single-jump operator contributes exactly one term. Handles multi-entry
      operators as well (all nonzero pairs).
    - Anticommutator -½{Cᵢ†Cᵢ, ρ}: Σᵢ Cᵢ†Cᵢ is a plain numeric matrix,
      computed with numpy; only its nonzero entries generate symbolic terms.
      For single-jump operators it is diagonal, giving -(γᵢ+γⱼ)/2 · ρᵢⱼ.

    Args:
        C_array: Collapse operators, shape (n_operators, n, n).
        density_matrix: Symbolic density matrix from generate_density_matrix.
        fast: Deprecated, ignored. The sparse construction is always used and
            is both faster and more general than the old fast/dense paths.
    """
    nstates = density_matrix.shape[0]

    if not np.iscomplexobj(C_array):
        C_array = C_array.astype(np.complex128)

    # Cache symbolic rho entries; Matrix __getitem__ is comparatively slow.
    rho = [[density_matrix[i, j] for j in range(nstates)] for i in range(nstates)]
    terms: list[list[list[smp.Expr]]] = [
        [[] for _ in range(nstates)] for _ in range(nstates)
    ]

    # Jump term: (C ρ C†)[i, j] = Σ_{α,β} C[i,α] ρ[α,β] conj(C[j,β])
    for C in C_array:
        nonzero = np.argwhere(C != 0)
        for i, alpha in nonzero:
            c_ia = complex(C[i, alpha])
            for j, beta in nonzero:
                coefficient = _sympy_number(c_ia * complex(C[j, beta]).conjugate())
                terms[i][j].append(coefficient * rho[alpha][beta])

    # Anticommutator term from the numeric Σᵢ Cᵢ†Cᵢ
    C_conj_array: npt.NDArray[np.complexfloating] = np.einsum(
        "ijk->ikj",
        C_array.conj(),  # type: ignore[arg-type]
    )
    C_dagger_C_sum: npt.NDArray[np.complexfloating] = np.einsum(
        "ijk,ikl",
        C_conj_array,  # type: ignore[arg-type]
        C_array,  # type: ignore[arg-type]
    )
    nonzero = np.argwhere(C_dagger_C_sum != 0)
    # -½ (C†C)·ρ : row i of C†C hits every column j of ρ
    for i, k in nonzero:
        coefficient = _sympy_number(-0.5 * complex(C_dagger_C_sum[i, k]))
        for j in range(nstates):
            terms[i][j].append(coefficient * rho[k][j])
    # -½ ρ·(C†C) : column j of C†C hits every row i of ρ
    for k, j in nonzero:
        coefficient = _sympy_number(-0.5 * complex(C_dagger_C_sum[k, j]))
        for i in range(nstates):
            terms[i][j].append(coefficient * rho[i][k])

    return smp.Matrix(
        nstates, nstates, lambda i, j: smp.Add(*terms[i][j])
    )


def generate_hamiltonian_term(
    hamiltonian: smp.Matrix, density_matrix: smp.Matrix
) -> smp.Matrix:
    """Build the coherent term -i[H, ρ] entrywise from H's nonzero structure.

    Equivalent to -1j * (H @ rho - rho @ H) but O(nnz(H)·n) sympy operations
    instead of O(n³) dense symbolic matrix products.
    """
    nstates = hamiltonian.shape[0]
    rho = [[density_matrix[i, j] for j in range(nstates)] for i in range(nstates)]
    terms: list[list[list[smp.Expr]]] = [
        [[] for _ in range(nstates)] for _ in range(nstates)
    ]

    for a in range(nstates):
        for b in range(nstates):
            h_ab = hamiltonian[a, b]
            if h_ab == 0:
                continue
            # -i (H ρ)[a, j] contribution: H[a,b] ρ[b,j]
            coefficient = -1j * h_ab
            for j in range(nstates):
                terms[a][j].append(coefficient * rho[b][j])
            # +i (ρ H)[i, b] contribution: ρ[i,a] H[a,b]
            coefficient = 1j * h_ab
            for i in range(nstates):
                terms[i][b].append(coefficient * rho[i][a])

    return smp.Matrix(
        nstates, nstates, lambda i, j: smp.Add(*terms[i][j])
    )


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
