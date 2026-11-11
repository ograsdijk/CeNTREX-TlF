import copy
from collections import defaultdict
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import sympy as smp

from centrex_tlf import couplings as couplings_tlf
from centrex_tlf import states

from .utils import has_off_diagonal_elements, strip_float_ones
from .utils_compact import compact_symbolic_hamiltonian_indices


def _build_manifold_indices(
    QN: List[states.CoupledState],
) -> List[List[int]]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, qn in enumerate(QN):
        largest = qn.largest
        key = (largest.electronic_state, int(largest.J))
        groups[key].append(i)
    return list(groups.values())

__all__ = [
    "symbolic_hamiltonian_to_rotating_frame",
    "generate_rwa_symbolic_hamiltonian",
    "generate_total_symbolic_hamiltonian",
    "generate_unitary_transformation_matrix",
    "generate_symbolic_hamiltonian",
    "apply_per_level_rotating_frame",
]


def _rotation_frequencies_from_graph(
    coupled_states: Sequence[Tuple[int, int, smp.Symbol]],
    n_states: int,
) -> Optional[dict]:
    """Solve ``ω = a_i - a_j`` for every coupling by walking the coupling graph.

    Reproduces what ``sympy.solve`` followed by "set the free parameters to
    zero" returns for this system. Each connected set of coupled states has one
    undetermined frequency; the linear solve leaves the highest-index state of
    the set free, so that state is the zero of the set here too and every other
    state follows by adding or subtracting ω along the couplings. The free
    states themselves are left out of the result, as they are in the solve.

    Returns None when two paths to the same state disagree (a closed loop of
    couplings whose frequencies do not cancel), so the caller can fall back to
    the general solver.
    """
    neighbours: dict = defaultdict(list)
    for i, j, ω in coupled_states:
        i, j = int(i), int(j)
        neighbours[i].append((j, ω))  # a_j = a_i - ω
        neighbours[j].append((i, -ω))  # a_i = a_j + ω

    frequencies: dict = {}
    roots = set()
    for root in sorted(neighbours, reverse=True):
        if root in frequencies:
            continue
        roots.add(root)
        frequencies[root] = smp.Integer(0)
        stack = [root]
        while stack:
            node = stack.pop()
            for other, shift in neighbours[node]:
                value = frequencies[node] - shift
                if other not in frequencies:
                    frequencies[other] = value
                    stack.append(other)
                elif frequencies[other] != value:
                    return None
    # the root of every connected set is the free parameter: not part of the solution
    return {idx: frequencies[idx] for idx in sorted(frequencies) if idx not in roots}


def generate_unitary_transformation_matrix(
    hamiltonian: smp.MutableDenseMatrix,
    manifold_indices: Optional[List[List[int]]] = None,
) -> smp.MutableDenseMatrix:
    """
    Generate the unitary transformation matrix to move to the rotating frame.
    This function computes a unitary transformation matrix based on the couplings
    in the given Hamiltonian. It identifies coupled states, solves the equations
    to determine the frequency differences between states, and constructs the
    transformation matrix to transition to the rotating frame.

    When manifold_indices is provided, states within each manifold group share
    the same rotation frequency (taken from whichever member was directly
    coupled). This ensures all states in a (electronic_state, J) manifold rotate
    together, removing GHz-scale absolute energies while preserving hyperfine
    splittings within the manifold.

    Args:
        hamiltonian (smp.MutableDenseMatrix): The Hamiltonian matrix
            representing the system, where off-diagonal elements correspond to
            couplings between states.
        manifold_indices (list of list of int, optional): Groups of state indices
            that should share the same rotation frequency. Typically grouped by
            (electronic_state, J). If None, only directly coupled states are
            rotated (original behavior).
    Returns:
        smp.MutableDenseMatrix: The unitary transformation matrix
            for transitioning to the rotating frame.
    """
    n_states = np.shape(hamiltonian)[0]

    # generate t symbol for non-rotating frame
    t = smp.Symbol("t", real=True)

    coupled_states = []
    # nonzero entries in row-major order, without converting the matrix to NumPy
    for (i, j), element in sorted(hamiltonian.todok().items()):
        if i < j:
            syms = element.free_symbols
            syms = [s for s in syms if str(s)[0] == "ω"]
            if len(syms) != 1:
                raise ValueError(f"Too many/few couplings, syms = {syms}")
            coupled_states.append((i, j, syms[0]))

    # solve equations to generate unitary transformation to rotating frame
    A = smp.symbols(f"a:{n_states}")
    frequencies = _rotation_frequencies_from_graph(coupled_states, n_states)
    if frequencies is not None:
        sol = {A[idx]: frequency for idx, frequency in frequencies.items()}
    else:
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

    # propagate rotation frequencies within manifolds
    if manifold_indices is not None:
        for group in manifold_indices:
            group_freq = smp.Integer(0)
            for idx in group:
                if A[idx] in sol and sol[A[idx]] != 0:
                    group_freq = sol[A[idx]]
                    break
            for idx in group:
                sol[A[idx]] = group_freq

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

    manifold_indices = _build_manifold_indices(QN)
    T = generate_unitary_transformation_matrix(hamiltonian, manifold_indices=manifold_indices)

    # use unitary matrix to transform to rotating frame
    transformed = T.adjoint() @ hamiltonian @ T - 1j * T.adjoint() @ smp.diff(T, t)
    # expand to get rid of exponentials that cancel out
    for (i, j), element in sorted(transformed.todok().items()):
        if i < j:
            transformed[i, j] = smp.expand(element)
            transformed[j, i] = smp.conjugate(transformed[i, j])
    # transformed = smp.expand(transformed)

    transformed = smp.Matrix(transformed)

    qn_index = {id(state): idx for idx, state in enumerate(QN)}

    def _qn_index(state: states.CoupledState) -> int:
        idx = qn_index.get(id(state))
        if idx is None:
            # Coupling main states are often distinct objects from QN entries;
            # fall back to equality search in that case.
            idx = QN.index(state)
        return idx

    for idc, (δ, coupling) in enumerate(zip(δs, couplings)):
        # generate transition frequency symbol
        ω = smp.Symbol(f"ω{idc}", real=True)
        # get indices of ground and excited states
        idg = _qn_index(coupling.ground_main)
        ide = _qn_index(coupling.excited_main)

        # transform to δ instead of ω and E
        if idg < ide:
            _subs_where_present(
                transformed, [(ω, energies[ide] - energies[idg] + δ)]
            )
        elif idg > ide:
            _subs_where_present(
                transformed, [(ω, energies[idg] - energies[ide] + δ)]
            )

    # remove excited state energy from all diagonal entries
    for idc, (δ, coupling) in enumerate(zip(δs, couplings)):
        idg = _qn_index(coupling.ground_main)
        expr = transformed[idg, idg]
        for d in δs:
            expr = expr.subs(d, 0)

        for idx in range(transformed.shape[0]):
            transformed[idx, idx] -= expr

    # remove extraneous 1.0 from expressions like 1.0*δ
    for idx in range(transformed.shape[0]):
        transformed[idx, idx] = strip_float_ones(transformed[idx, idx])

    # substitute level energies for symbolic values
    _subs_where_present(transformed, list(zip(energies, np.diag(H_int))))

    return transformed


def _subs_where_present(
    matrix: smp.MutableDenseMatrix,
    replacements: Sequence[Tuple[Any, Any]],
) -> None:
    """Substitute symbols in place, visiting only the entries that contain them.

    Gives the same entries as ``matrix.subs(replacements)`` for symbol → value
    pairs, but ``Matrix.subs`` converts every replacement to SymPy and walks the
    expression tree once per pair for *every* nonzero entry. Here the pairs are
    converted once and each entry is only offered the symbols it actually
    holds, in the original order, so the sequential-substitution result is
    unchanged.
    """
    pairs = [(smp.sympify(old), smp.sympify(new)) for old, new in replacements]
    symbols = {old for old, _ in pairs}
    for (i, j), entry in sorted(matrix.todok().items()):
        present = entry.free_symbols & symbols
        if not present:
            continue
        matrix[i, j] = entry.subs([pair for pair in pairs if pair[0] in present])


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
    n_states = H_int.shape[0]
    if has_off_diagonal_elements(H_int):
        raise ValueError("Hamiltonian should not have off-diagonal elements")

    hamiltonian = smp.zeros(*H_int.shape)
    energies = smp.symbols(f"E:{n_states}")
    hamiltonian += smp.eye(n_states) * np.asarray(energies)

    t = smp.Symbol("t", real=True)

    def _add_coupling_terms(hamiltonian, field, val, ω, t):
        for i, j in zip(*np.nonzero(field.field)):
            if i < j:
                hamiltonian[i, j] += val * field.field[i, j] * smp.exp(1j * ω * t)
                hamiltonian[j, i] += (
                    smp.conjugate(val) * field.field[j, i] * smp.exp(-1j * ω * t)
                )

    for idc, (Ω, coupling) in enumerate(zip(Ωs, couplings)):
        ω = smp.Symbol(f"ω{idc}", real=True)
        main_coupling = coupling.main_coupling
        for idf, field in enumerate(coupling.fields):
            if pols and pols[idc]:
                val = (pols[idc][idf] * Ω / main_coupling) / 2
            else:
                val = (Ω / main_coupling) / 2
            _add_coupling_terms(hamiltonian, field, val, ω, t)

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
    if has_off_diagonal_elements(H_int):
        raise ValueError("Hamiltonian should not have off-diagonal elements")
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


def apply_per_level_rotating_frame(obe_system: Any) -> Any:
    """Move an ``OBESystem`` into a per-level co-rotating frame.

    This removes the *numeric* static part of every diagonal Hamiltonian
    entry analytically, by transforming with the diagonal unitary
    ``T = diag(exp(-i * E_i * t))``, where ``E_i`` is the purely numeric
    residual energy of state ``i`` (all remaining free symbols -- e.g.
    symbolic detunings -- are set to 0 before evaluating ``E_i``, so they are
    *not* removed and parameter scans over detuning etc. still work
    unchanged in the rotated frame). Under ``rho' = T^dagger rho T``, the
    transformed Hamiltonian is

        H'[i, i] = H[i, i] - E_i
        H'[i, j] = H[i, j] * exp(i * (E_i - E_j) * t)   for i != j

    (the multiplication by the oscillating phase is skipped whenever
    ``E_i == E_j`` exactly, to avoid inserting a spurious ``exp(i*0*t)``
    factor). The Lindblad dissipator is built from single-jump collapse
    operators ``C``; a diagonal phase rotation sends ``C -> phase * C``, and
    ``C rho C^dagger`` (population transfer) and ``C^dagger C`` (used in the
    anticommutator) are invariant under an overall phase on ``C``, so
    ``C_array`` is reused unchanged -- it is *not* touched by this function.

    Only ``H_symbolic`` is replaced; the returned ``OBESystem`` is otherwise
    a shallow copy of ``obe_system``, with ``system``/``dissipator`` left
    unset so they lazily rebuild from the new (now explicitly
    time-dependent) ``H_symbolic`` on first access.

    Frame-invariance notes for consumers:

    - Populations (diagonal density-matrix elements) and any
      weighted-population integral/rate (``output in {"populations",
      "weighted_integral", "photon_integral", "excited_population", ...}``)
      are IDENTICAL in the original and rotated frames, because
      ``rho'_ii = rho_ii`` for a diagonal transformation.
    - Individual COHERENCES (``output="selected"`` off-diagonal entries) are
      reported in the ROTATED frame and differ from the original frame by
      the known phase ``exp(i * (E_i - E_j) * t)`` -- multiply the rotated
      coherence by ``exp(-i * (E_i - E_j) * t)`` to recover the original
      frame's value.
    - The initial condition is unchanged: ``T(0) = I``, so ``rho'(0) =
      rho(0)``.

    Args:
        obe_system: an ``OBESystem`` (as returned by
            ``generate_OBE_system``/``generate_OBE_system_transitions``).

    Returns:
        A new ``OBESystem`` with the per-level co-rotating-frame
        ``H_symbolic`` and all other fields copied from ``obe_system``.
    """
    # Local import: utils_setup imports from this module at module scope, so
    # importing OBESystem at module scope here would create a circular import.
    from .utils_setup import OBESystem

    H = obe_system.H_symbolic
    n = int(H.shape[0])
    t = smp.Symbol("t", real=True)

    energies: list[float] = []
    for i in range(n):
        entry = H[i, i]
        zero_subs = {s: 0 for s in entry.free_symbols}
        energies.append(complex(entry.subs(zero_subs)).real)

    H_new = smp.zeros(n, n)
    for i in range(n):
        H_new[i, i] = H[i, i] - energies[i]
        for j in range(n):
            if i == j:
                continue
            entry = H[i, j]
            if entry == 0:
                continue
            delta = energies[i] - energies[j]
            if delta != 0.0:
                H_new[i, j] = entry * smp.exp(smp.I * smp.Float(delta) * t)
            else:
                H_new[i, j] = entry

    return OBESystem(
        ground=obe_system.ground,
        excited=obe_system.excited,
        QN=obe_system.QN,
        H_int=obe_system.H_int,
        V_ref_int=obe_system.V_ref_int,
        couplings=obe_system.couplings,
        H_symbolic=smp.Matrix(H_new),
        C_array=obe_system.C_array,
        coupling_symbols=obe_system.coupling_symbols,
        polarization_symbols=obe_system.polarization_symbols,
        QN_original=obe_system.QN_original,
        decay_channels=obe_system.decay_channels,
        couplings_original=obe_system.couplings_original,
    )
