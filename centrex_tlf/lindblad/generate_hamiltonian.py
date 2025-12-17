import copy
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import sympy as smp

from centrex_tlf import couplings as couplings_tlf
from centrex_tlf import states

from .utils import has_off_diagonal_elements, strip_float_ones
from .utils_compact import compact_symbolic_hamiltonian_indices

__all__ = [
    "symbolic_hamiltonian_to_rotating_frame",
    "generate_rwa_symbolic_hamiltonian",
    "generate_total_symbolic_hamiltonian",
    "generate_unitary_transformation_matrix",
    "generate_symbolic_hamiltonian",
]


def generate_unitary_transformation_matrix(
    hamiltonian: smp.MutableDenseMatrix,
) -> smp.MutableDenseMatrix:
    """
    Generate the unitary transformation matrix to move to the rotating frame.
    This function computes a unitary transformation matrix based on the couplings
    in the given Hamiltonian. It identifies coupled states, solves the equations
    to determine the frequency differences between states, and constructs the
    transformation matrix to transition to the rotating frame.
    Args:
        hamiltonian (smp.MutableDenseMatrix): The Hamiltonian matrix
            representing the system, where off-diagonal elements correspond to
            couplings between states.
    Returns:
        smp.MutableDenseMatrix: The unitary transformation matrix
            for transitioning to the rotating frame.
    """
    n_states = np.shape(hamiltonian)[0]

    # generate t symbol for non-rotating frame
    t = smp.Symbol("t", real=True)

    coupled_states = []
    for i, j in zip(*np.nonzero(hamiltonian)):
        if i < j:
            syms = hamiltonian[i, j].free_symbols
            syms = [s for s in syms if str(s)[0] == "ω"]
            assert len(syms) == 1, f"Too many/few couplings, syms = {syms}"
            coupled_states.append((i, j, syms[0]))

    # solve equations to generate unitary transformation to rotating frame
    A = smp.symbols(f"a:{n_states}")
    Eqns = []
    # generate equations
    for i, j, ω in coupled_states:
        Eqns.append(ω - (A[i] - A[j]))
    # solve system of equations
    sol = smp.solve(Eqns, A)
    # set free parameters to zero in the solution
    free_params = [value for value in A if value not in list(sol.keys())]
    for free_param in free_params:
        for key, val in sol.items():
            sol[key] = val.subs(free_param, 0)

    # generate unitary transformation matrix
    T = smp.eye(n_states, n_states)
    for var in sol.keys():
        ida = int(str(var)[1:])
        T[ida, ida] = smp.exp(1j * sol[var] * t)
    return T


def symbolic_hamiltonian_to_rotating_frame(
    hamiltonian: smp.MutableDenseMatrix,
    QN: List[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    couplings: Sequence[couplings_tlf.CouplingFields],
    δs: Sequence[smp.Symbol],
) -> smp.MutableDenseMatrix:
    """Transform a symbolic hamiltonian to the rotating frame. Exponential terms
    with the transition frequencies are required to be present in the
    hamiltonian matrix, as well as symbolic energies on the diagonal.

    Args:
        hamiltonian (sympy.Matrix): symbolic hamiltonian
        QN (list/array): list/array of states in the system
        H_int (np.ndarray): numerical hamiltonian, energies only
        couplings (list): list of couplings in system

    Returns:
        sympy.Matrix: symbolic hamiltonian in the rotating frame
    """
    energies = np.diag(hamiltonian)

    # generate t symbol for non-rotating frame
    t = smp.Symbol("t", real=True)

    T = generate_unitary_transformation_matrix(hamiltonian)

    # use unitary matrix to transform to rotating frame
    transformed = T.adjoint() @ hamiltonian @ T - 1j * T.adjoint() @ smp.diff(T, t)
    # expand to get rid of exponentials that cancel out
    for i in range(transformed.shape[0]):
        for j in range(i + 1, transformed.shape[0]):
            if transformed[i, j] != 0:
                transformed[i, j] = smp.expand(transformed[i, j])
                transformed[j, i] = smp.conjugate(transformed[i, j])
    # transformed = smp.expand(transformed)

    transformed = smp.Matrix(transformed)

    for idc, (δ, coupling) in enumerate(zip(δs, couplings)):
        # generate transition frequency symbol
        ω = smp.Symbol(f"ω{idc}", real=True)
        # get indices of ground and excited states
        idg = QN.index(coupling.ground_main)
        ide = QN.index(coupling.excited_main)

        # transform to δ instead of ω and E
        if idg < ide:
            transformed = transformed.subs(ω, energies[ide] - energies[idg] + δ)
        elif idg > ide:
            transformed = transformed.subs(ω, energies[idg] - energies[ide] + δ)

    # remove excited state energy from all diagonal entries
    for idc, (δ, coupling) in enumerate(zip(δs, couplings)):
        idg = QN.index(coupling.ground_main)
        expr = transformed[idg, idg]
        for d in δs:
            expr = expr.subs(d, 0)

        for idx in range(transformed.shape[0]):
            transformed[idx, idx] -= expr

    # remove extraneous 1.0 from expressions like 1.0*δ
    for idx in range(transformed.shape[0]):
        transformed[idx, idx] = strip_float_ones(transformed[idx, idx])

    # substitute level energies for symbolic values
    transformed = transformed.subs(
        [(E, val) for E, val in zip(energies, np.diag(H_int))]
    )

    return transformed


def generate_symbolic_hamiltonian(
    H_int: npt.NDArray[np.complex128],
    couplings: Sequence[couplings_tlf.CouplingFields],
    Ωs: Sequence[smp.Symbol],
    pols: List[Optional[Sequence[smp.Symbol]]],
) -> smp.MutableDenseMatrix:
    """Generate a symbolic hamiltonian without transforming to the rotating frame.
    This function generates a symbolic Hamiltonian matrix based on the provided
    quantum numbers, internal Hamiltonian, couplings, and transition frequencies.
    It constructs the Hamiltonian by iterating over the couplings and fields,
    incorporating the effects of polarizations and transition frequencies.

    Args:
        QN (List[states.CoupledState]): List of coupled states in the system.
        H_int (npt.NDArray[np.complex128]): Internal Hamiltonian matrix.
        couplings (Sequence[couplings_tlf.CouplingFields]): List of coupling fields.
        Ωs (Sequence[smp.Symbol]): Transition frequency symbols.
        δs (Sequence[smp.Symbol]): Frequency difference symbols.
        pols (List[Optional[Sequence[smp.Symbol]]]): Polarization symbols for each coupling.

    Returns:
        smp.MutableDenseMatrix: Symbolic Hamiltonian matrix.
    """
    assert not has_off_diagonal_elements(H_int), (
        "Hamiltonian should not have off-diagonal elements"
    )
    n_states = H_int.shape[0]
    # initialize empty hamiltonian
    hamiltonian = smp.zeros(*H_int.shape)
    energies = smp.symbols(f"E:{n_states}")
    hamiltonian += smp.eye(n_states) * np.asarray(energies)

    # generate t symbol for non-rotating frame
    t = smp.Symbol("t", real=True)

    # iterate over couplings
    for idc, (Ω, coupling) in enumerate(zip(Ωs, couplings)):
        # generate transition frequency symbol
        ω = smp.Symbol(f"ω{idc}", real=True)
        # main coupling matrix element
        main_coupling = coupling.main_coupling
        # iterate over fields (polarizations) in the coupling
        for idf, field in enumerate(coupling.fields):
            if pols:
                P = pols[idc]
                if P:
                    _P = P[idf]
                    val = (_P * Ω / main_coupling) / 2
                    for i, j in zip(*np.nonzero(field.field)):
                        if i < j:
                            hamiltonian[i, j] += (
                                val * field.field[i, j] * smp.exp(1j * ω * t)
                            )
                            hamiltonian[j, i] += (
                                smp.conjugate(val)
                                * field.field[j, i]
                                * smp.exp(-1j * ω * t)
                            )
                else:
                    val = (Ω / main_coupling) / 2
                    for i, j in zip(*np.nonzero(field.field)):
                        if i < j:
                            hamiltonian[i, j] += (
                                val * field.field[i, j] * smp.exp(1j * ω * t)
                            )
                            hamiltonian[j, i] += (
                                smp.conjugate(val)
                                * field.field[j, i]
                                * smp.exp(-1j * ω * t)
                            )
            else:
                val = (Ω / main_coupling) / 2
                for i, j in zip(*np.nonzero(field.field)):
                    if i < j:
                        hamiltonian[i, j] += (
                            val * field.field[i, j] * smp.exp(1j * ω * t)
                        )
                        hamiltonian[j, i] += (
                            smp.conjugate(val)
                            * field.field[j, i]
                            * smp.exp(-1j * ω * t)
                        )

    return hamiltonian


def generate_rwa_symbolic_hamiltonian(
    QN: List[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    couplings: Sequence[couplings_tlf.CouplingFields],
    Ωs: Sequence[smp.Symbol],
    δs: Sequence[smp.Symbol],
    pols: List[Optional[Sequence[smp.Symbol]]],
) -> smp.MutableDenseMatrix:
    hamiltonian = generate_symbolic_hamiltonian(H_int, couplings, Ωs, pols)
    transformed = symbolic_hamiltonian_to_rotating_frame(
        hamiltonian, QN, H_int, couplings, δs
    )
    transformed = smp.Matrix(transformed)
    return transformed


@overload
def generate_total_symbolic_hamiltonian(
    QN: List[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    couplings: List[couplings_tlf.CouplingFields],
    transitions: Sequence[Any],
    qn_compact: Literal[None],
) -> smp.MutableDenseMatrix: ...


@overload
def generate_total_symbolic_hamiltonian(
    QN: List[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    couplings: List[couplings_tlf.CouplingFields],
    transitions: Sequence[Any],
) -> smp.MutableDenseMatrix: ...


@overload
def generate_total_symbolic_hamiltonian(
    QN: List[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    couplings: Sequence[couplings_tlf.CouplingFields],
    transitions: Sequence[Any],
    qn_compact: Union[Sequence[states.QuantumSelector], states.QuantumSelector],
) -> Tuple[smp.MutableDenseMatrix, List[states.CoupledState]]: ...


def generate_total_symbolic_hamiltonian(
    QN: List[states.CoupledState],
    H_int: npt.NDArray[np.complex128],
    couplings: Sequence[Any],
    transitions: Sequence[Any],
    qn_compact: Optional[
        Union[Sequence[states.QuantumSelector], states.QuantumSelector]
    ] = None,
) -> Union[
    Tuple[smp.MutableDenseMatrix, List[states.CoupledState]],
    smp.MutableDenseMatrix,
]:
    """Generate the total symbolic hamiltonian for the given system

    Args:
        QN (Sequence[states.State]): states
        H_int (np.ndarray): internal hamiltonian
        couplings (Sequence[states.State]): list of dictionaries with all couplings of
                                            the system
        transitions (Sequence[states.State]): list of dictionaries with all transitions
                                                of the system
        qn_compact (Sequence[states.State], optional): list of QuantumSelectors or lists
                                                        of QuantumSelectors with each
                                                        QuantumSelector containing the
                                                        quantum numbers to compact into
                                                        a single state.
                                                        Defaults to None.

    Returns:
        sympy matrix: symbolic hamiltonian
        if qn_compact is provided, also returns the states corresponding to the
        compacted hamiltonian, i.e. ham, QN_compact
    """
    assert not has_off_diagonal_elements(H_int), (
        "Hamiltonian should not have off-diagonal elements"
    )
    Ωs = [t.Ω for t in transitions]
    Δs = [t.δ for t in transitions]
    pols: List[Optional[Sequence[smp.Symbol]]] = []
    for transition in transitions:
        if not transition.polarization_symbols:
            pols.append(None)
        else:
            pols.append(transition.polarization_symbols)

    H_symbolic = generate_rwa_symbolic_hamiltonian(QN, H_int, couplings, Ωs, Δs, pols)
    if qn_compact is not None:
        if isinstance(qn_compact, states.QuantumSelector):
            qn_compact = [qn_compact]
        QN_compact = copy.deepcopy(QN)
        for qnc in qn_compact:
            indices_compact = states.get_indices_quantumnumbers(qnc, QN_compact)
            QN_compact = states.compact_QN_coupled_indices(QN_compact, indices_compact)
            H_symbolic = compact_symbolic_hamiltonian_indices(
                H_symbolic, indices_compact
            )
        return H_symbolic, QN_compact

    return H_symbolic
