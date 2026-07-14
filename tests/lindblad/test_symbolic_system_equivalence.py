"""Regression test pinning the CURRENT symbolic OBE construction.

`IMPLEMENTATION_AUDIT.md` (section "Performance Review (2026-07-11)") plans to
turn `OBESystem.system`/`OBESystem.dissipator` into lazily built, sparse
entrywise constructors instead of the current eager
`generate_system_of_equations_symbolic(H_symbolic, C_array, fast=True,
split_output=True)` call (see
`centrex_tlf.lindblad.utils_setup._build_obe_system`, `method="expanded"`
branch). This test numerically pins the *current* construction so that
refactor is required to reproduce it bit-for-bit (up to floating point
tolerance): it builds a small `OBESystem` with `method="expanded"`, assigns
random numeric values to every symbol involved (Hamiltonian parameters and
density-matrix entries), and checks that both the combined `system` matrix
and the standalone `dissipator` matrix agree with a plain-numpy evaluation of
the Lindblad master equation

    drho/dt = -i[H, rho] + sum_k(C_k rho C_k^dagger
                                  - 0.5 * {C_k^dagger C_k, rho})

built directly from the numeric Hamiltonian and `C_array` -- no sympy
involved in the "expected" side at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as smp

from centrex_tlf import couplings, lindblad, transitions
from centrex_tlf.lindblad.utils_setup import OBESystem


@pytest.fixture(scope="module")
def small_obe_system() -> OBESystem:
    """OBESystem built through the real production path at default Jmax.

    Uses `transitions.R0_F1_1o2_F0` (X J=0 -> B J=1, F1'=1/2, F'=0) with the
    DEFAULT `Jmax_X`/`Jmax_B`. The discovery pass in
    `generate_reduced_hamiltonian_transitions` deliberately builds a wider J
    range than the driven manifolds so the dressed eigenstates incorporate
    rotational mixing (hyperfine/Stark mixing in X, J-mixing in B); the
    reduced-Hamiltonian helpers default to `Jmax = max(J) + 2` for exactly
    this reason. Do NOT restrict Jmax to just the driven manifolds here --
    that amputates the mixing and produces a system that is not what the
    production pipeline generates.

    The price is that undriven spectator manifolds (e.g. X J=2) keep huge
    (~1e11 rad/s) absolute rotational energies on the diagonal, so
    evaluating the symbolic and numpy sides in different operation orders
    leaves float64 cancellation residue of order eps * max|H|. That is a
    tolerance question, not a correctness question -- the assertions below
    use a magnitude-aware atol instead of a fixed absolute one.

    `method="expanded"` is required so `_build_obe_system` populates both
    `system` (= hamiltonian_term + dissipator) and `dissipator`.
    """
    trans = transitions.R0_F1_1o2_F0
    transition_selectors = couplings.generate_transition_selectors(
        [trans], [[couplings.polarization_Z]]
    )
    return lindblad.generate_OBE_system_transitions(
        [trans],
        transition_selectors,
        method="expanded",
    )


def _random_hermitian_rho(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[smp.Basic, complex]]:
    """Draw a random Hermitian density matrix plus its symbolic substitution dict.

    `generate_density_matrix(n)` (see
    `centrex_tlf/lindblad/generate_system_of_equations.py`) represents rho
    with independent `IndexedBase("ρ")[i, j]` symbols for the upper
    triangle (i <= j); lower-triangle entries are `conjugate(rho[j, i])` of
    the *same* symbols, not independent symbols. So only upper-triangle
    (including diagonal) symbols need values in the substitution dict --
    substituting a numeric value into `rho[j, i]` automatically resolves
    `conjugate(rho[j, i])` to the complex conjugate of that same numeric
    value once sympy evaluates the substituted expression.
    """
    density_matrix_sym = lindblad.generate_density_matrix(n)
    rho_num = np.zeros((n, n), dtype=complex)
    subs: dict[smp.Basic, complex] = {}
    for i in range(n):
        for j in range(i, n):
            if i == j:
                value = complex(rng.uniform(-1.0, 1.0), 0.0)
            else:
                value = complex(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
            rho_num[i, j] = value
            rho_num[j, i] = np.conj(value)
            subs[density_matrix_sym[i, j]] = value
    return rho_num, subs


def _lindblad_rhs(
    H: np.ndarray, C_array: np.ndarray, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Plain-numpy Lindblad master equation RHS (no sympy).

    Returns (full_rhs, dissipator_only).
    """
    C_array = np.asarray(C_array, dtype=complex)
    C_dagger = np.conj(np.transpose(C_array, (0, 2, 1)))
    commutator = H @ rho - rho @ H
    jump_sum = np.zeros_like(rho)
    for C, C_dag in zip(C_array, C_dagger):
        jump_sum += C @ rho @ C_dag
    C_dagger_C_sum = np.einsum("kij,kjl->il", C_dagger, C_array)
    anticommutator = C_dagger_C_sum @ rho + rho @ C_dagger_C_sum
    dissipator = jump_sum - 0.5 * anticommutator
    return -1j * commutator + dissipator, dissipator


def _cancellation_atol(*operands: np.ndarray) -> float:
    """Magnitude-aware absolute tolerance for comparing two float64 evaluations.

    The symbolic and numpy sides compute the same sums/differences in
    different orders, so entries where large terms cancel carry an absolute
    residue of order eps * (largest cancelling operand). With ~1e11 rad/s
    rotational energies on the Hamiltonian diagonal that floor is ~1e-5 --
    a fixed atol like 1e-10 would reject a correct construction. A real
    construction bug (wrong/missing/mis-signed term) produces errors on the
    scale of the term magnitudes themselves, many orders above this atol,
    so no bug-catching power is lost.

    The 64x headroom covers accumulation over the ~n-term sums per entry.
    """
    scale = max(float(np.abs(op).max()) for op in operands)
    return 64.0 * np.finfo(np.float64).eps * max(scale, 1.0)


def _substitute_to_numpy(matrix: smp.Matrix, subs: dict[smp.Basic, complex], n: int) -> np.ndarray:
    """Substitute numeric values into a symbolic (n, n) matrix and densify to numpy.

    Uses `.subs()` (not `lambdify`) so that `conjugate()` wrappers around
    lower-triangle rho entries are resolved automatically by sympy's
    evaluation of the substituted (fully numeric) expression tree, then
    `complex()` per element to get a plain numpy array.
    """
    substituted = matrix.subs(subs)
    return np.array(
        [[complex(substituted[i, j]) for j in range(n)] for i in range(n)],
        dtype=complex,
    )


def test_symbolic_system_matches_lindblad_rhs(small_obe_system: OBESystem) -> None:
    system = small_obe_system
    n = len(system.QN)
    rng = np.random.default_rng(20260711)

    h_symbols = sorted(system.H_symbolic.free_symbols, key=str)
    h_subs: dict[smp.Basic, complex] = {
        symbol: complex(rng.uniform(-5.0, 5.0), 0.0) for symbol in h_symbols
    }

    rho_num, rho_subs = _random_hermitian_rho(n, rng)

    H_num = _substitute_to_numpy(system.H_symbolic, h_subs, n)
    expected_rhs, _ = _lindblad_rhs(H_num, system.C_array, rho_num)

    full_subs = {**h_subs, **rho_subs}
    system_num = _substitute_to_numpy(system.system, full_subs, n)

    # Cancelling operands are the H@rho / rho@H products.
    atol = _cancellation_atol(np.abs(H_num) * np.abs(rho_num).max())
    np.testing.assert_allclose(system_num, expected_rhs, atol=atol, rtol=1e-12)


def test_symbolic_dissipator_matches_numpy_dissipator(small_obe_system: OBESystem) -> None:
    system = small_obe_system
    n = len(system.QN)
    rng = np.random.default_rng(20260712)

    rho_num, rho_subs = _random_hermitian_rho(n, rng)
    zero_H = np.zeros((n, n), dtype=complex)
    _, expected_dissipator = _lindblad_rhs(zero_H, system.C_array, rho_num)

    dissipator_num = _substitute_to_numpy(system.dissipator, rho_subs, n)

    # Cancelling operands are the (C^dagger C)@rho anticommutator products.
    C_arr = np.asarray(system.C_array, dtype=complex)
    cdagger_c = np.einsum("kij,kjl->il", np.conj(np.transpose(C_arr, (0, 2, 1))), C_arr)
    atol = _cancellation_atol(np.abs(cdagger_c) * np.abs(rho_num).max())
    np.testing.assert_allclose(dissipator_num, expected_dissipator, atol=atol, rtol=1e-12)
