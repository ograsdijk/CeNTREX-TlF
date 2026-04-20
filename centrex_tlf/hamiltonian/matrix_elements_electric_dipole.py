"""Electric dipole matrix element calculations for optical transitions.

This module provides functions for computing electric dipole matrix elements between
quantum states, including both reduced matrix elements and full matrix elements with
polarization-dependent angular factors.
"""

import math
from functools import lru_cache
from typing import Tuple

import numpy as np
import numpy.typing as npt

from centrex_tlf import couplings, states

from .wigner import sixj_f, threej_f

__all__ = [
    "generate_ED_ME_mixed_state",
    "generate_ED_ME_mixed_state_uncoupled",
    "ED_ME_coupled",
    "ED_ME_uncoupled",
    "angular_part",
]


def generate_ED_ME_mixed_state(
    bra: states.CoupledState,
    ket: states.CoupledState,
    pol_vec: npt.NDArray[np.complex128] | None = None,
    reduced: bool = False,
    normalize_pol: bool = True,
) -> complex:
    """Calculate electric dipole matrix element between mixed (superposition) states.

    Computes ⟨bra|D̂·ε|ket⟩ where bra and ket are mixed states (superpositions of basis
    states) and ε is the polarization vector. Automatically transforms states to Omega
    basis when needed (parity basis states are converted).

    Args:
        bra (CoupledState): Bra state (superposition of coupled basis states)
        ket (CoupledState): Ket state (superposition of coupled basis states)
        pol_vec (npt.NDArray[np.complex128] | None): Polarization vector [Ex, Ey, Ez]
            in Cartesian basis. Defaults to None, which uses [√(2/3), 0, √(1/3)], e.g.
            averaging over all q=-1,0,+1 polarizations.
        reduced (bool): If True, return only reduced matrix element (no angular part).
            Defaults to False.
        normalize_pol (bool): If True, normalize the polarization vector. Defaults to
            True.

    Returns:
        complex: Electric dipole matrix element

    Note:
        For X state (Ω=0), the coupled basis already represents the Omega basis,
        so no transformation is needed.

    Example:
        >>> ME = generate_ED_ME_mixed_state(ground_state, excited_state)
        >>> coupling_strength = np.abs(ME)
    """
    # Initialize default polarization vector if not provided
    if pol_vec is None:
        pol_vec = couplings.polarization_unpolarized.vector

    pol_vec = np.asarray(pol_vec, dtype=np.complex128)

    ME = 0j

    # Transform to Omega basis if required. For the X state the basis is Coupled and
    # doesn't require to be transformed to the Omega basis, since Omega is 0.
    if bra.largest.basis is states.Basis.CoupledP:
        bra = bra.transform_to_omega_basis()
    if ket.largest.basis is states.Basis.CoupledP:
        ket = ket.transform_to_omega_basis()

    if normalize_pol:
        pol_vec = pol_vec / np.linalg.norm(pol_vec)

    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            if abs(basis_bra.J - basis_ket.J) > 1:
                continue
            ME += (
                amp_bra.conjugate()
                * amp_ket
                * ED_ME_coupled(
                    basis_bra, basis_ket, pol_vec=tuple(pol_vec), rme_only=reduced
                )
            )

    return ME


def generate_ED_ME_mixed_state_uncoupled(
    bra: states.UncoupledState,
    ket: states.UncoupledState,
    pol_vec: npt.NDArray[np.complex128] | None = None,
    reduced: bool = False,
    normalize_pol: bool = True,
) -> complex:
    """Calculate electric dipole matrix element between uncoupled mixed states.

    Mirrors `generate_ED_ME_mixed_state`, but for `UncoupledState`. This stays in the
    uncoupled basis and evaluates the ME by summing `ED_ME_uncoupled` over uncoupled
    basis-state components.
    """
    if pol_vec is None:
        pol_vec = couplings.polarization_unpolarized.vector

    pol_vec = np.asarray(pol_vec, dtype=np.complex128)

    if normalize_pol:
        pol_vec = pol_vec / np.linalg.norm(pol_vec)

    ME = 0j
    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            if abs(basis_bra.J - basis_ket.J) > 1:
                continue
            ME += (
                amp_bra.conjugate()
                * amp_ket
                * ED_ME_uncoupled(
                    basis_bra, basis_ket, pol_vec=tuple(pol_vec), rme_only=reduced
                )
            )

    return ME


@lru_cache(maxsize=int(1e6))
def ED_ME_uncoupled(
    bra: states.UncoupledBasisState,
    ket: states.UncoupledBasisState,
    pol_vec: Tuple[complex, complex, complex] = (
        (1.0 + 0j) / math.sqrt(2),
        0j,
        (1.0 + 0j) / math.sqrt(2),
    ),
    rme_only: bool = False,
) -> complex:
    """Calculate electric dipole matrix element between uncoupled basis states.

    Implements the standard uncoupled-basis (|J mJ I1 m1 I2 m2; Ω⟩) rank-1 spherical
    tensor matrix element, closely following the structure of the user-provided
    reference implementation.

    Note:
        This intentionally does not include any physical dipole moment scaling.
    """
    ME = 0j
    for amp_bra, basis_bra in _expand_uncoupled_parity_to_omega_components(bra):
        for amp_ket, basis_ket in _expand_uncoupled_parity_to_omega_components(ket):
            ME += amp_bra.conjugate() * amp_ket * _ED_ME_uncoupled_omega(
                basis_bra,
                basis_ket,
                pol_vec=pol_vec,
                rme_only=rme_only,
            )
    return ME


def _expand_uncoupled_parity_to_omega_components(
    state: states.UncoupledBasisState,
) -> list[tuple[complex, states.UncoupledBasisState]]:
    """Expand a parity-basis B-state |J,|Ω|,P⟩ into signed-Ω components.

    This mirrors `states.CoupledBasisState.transform_to_omega_basis`'s convention:
        |J Ω P⟩ = (|J +Ω⟩ + P(-1)^J |J -Ω⟩)/√2

    For X (Ω=0) and for Ω=0 states, returns the state itself.
    """
    if state.electronic_state == states.ElectronicState.B and state.P is not None and state.Omega != 0:
        Omega_abs = abs(state.Omega)
        amp_plus = 1 / math.sqrt(2)
        amp_minus = state.P * ((-1) ** state.J) / math.sqrt(2)

        state_plus = states.UncoupledBasisState(
            J=state.J,
            mJ=state.mJ,
            I1=state.I1,
            m1=state.m1,
            I2=state.I2,
            m2=state.m2,
            Omega=Omega_abs,
            P=state.P,
            electronic_state=state.electronic_state,
            basis=state.basis,
            v=state.v,
        )
        state_minus = states.UncoupledBasisState(
            J=state.J,
            mJ=state.mJ,
            I1=state.I1,
            m1=state.m1,
            I2=state.I2,
            m2=state.m2,
            Omega=-Omega_abs,
            P=state.P,
            electronic_state=state.electronic_state,
            basis=state.basis,
            v=state.v,
        )

        return [(complex(amp_plus), state_plus), (complex(amp_minus), state_minus)]

    return [(1.0 + 0j, state)]


@lru_cache(maxsize=int(1e6))
def _ED_ME_uncoupled_omega(
    bra: states.UncoupledBasisState,
    ket: states.UncoupledBasisState,
    pol_vec: Tuple[complex, complex, complex],
    rme_only: bool,
) -> complex:
    """Uncoupled-basis ED ME assuming signed-Ω components (no parity mixing)."""
    # Spectator nuclear spins
    if bra.I1 != ket.I1 or bra.I2 != ket.I2:
        return 0j
    if bra.m1 != ket.m1 or bra.m2 != ket.m2:
        return 0j

    # Rotational selection rules
    if abs(bra.J - ket.J) > 1:
        return 0j
    if not rme_only and abs(bra.mJ - ket.mJ) > 1:
        return 0j

    J = float(bra.J)
    mJ = float(bra.mJ)
    Omega = float(bra.Omega)

    Jp = float(ket.J)
    mJp = float(ket.mJ)
    Omegap = float(ket.Omega)

    q = Omega - Omegap
    if abs(q) > 1:
        return 0j

    M_r = (
        (-1) ** (J - Omega)
        * math.sqrt((2 * J + 1) * (2 * Jp + 1))
        * threej_f(J, 1, Jp, -Omega, q, Omegap)
    )
    if M_r == 0.0:
        return 0j

    if rme_only:
        return complex(M_r)

    angular = _angular_part_uncoupled(pol_vec, J, mJ, Jp, mJp)
    if angular == 0.0:
        return 0j

    return complex(M_r) * angular


@lru_cache(maxsize=int(1e6))
def _angular_part_uncoupled(
    pol_vec: Tuple[complex, complex, complex],
    J: float,
    mJ: float,
    Jp: float,
    mJp: float,
) -> complex:
    """Polarization-dependent angular factor for uncoupled |J mJ⟩ states."""
    q_s = mJ - mJp
    if abs(q_s) > 1:
        return 0.0

    # Cartesian → spherical-basis components
    if q_s == 1:
        eps_q = -1 / math.sqrt(2) * (pol_vec[0] + 1j * pol_vec[1])  # σ+
    elif q_s == -1:
        eps_q = 1 / math.sqrt(2) * (pol_vec[0] - 1j * pol_vec[1])  # σ-
    elif q_s == 0:
        eps_q = pol_vec[2]
    else:
        return 0.0

    angular = (-1) ** (J - mJ) * threej_f(J, 1, Jp, -mJ, q_s, mJp) * eps_q
    return angular


@lru_cache(maxsize=int(1e6))
def ED_ME_coupled(
    bra: states.CoupledBasisState,
    ket: states.CoupledBasisState,
    pol_vec: Tuple[complex, complex, complex] = (
        (1.0 + 0j) / math.sqrt(2),
        0j,
        (1.0 + 0j) / math.sqrt(2),
    ),
    rme_only: bool = False,
) -> complex:
    """Calculate electric dipole matrix element between coupled basis states.

    Computes the electric dipole matrix element ⟨bra|D̂·ε|ket⟩ for transitions between
    molecular eigenstates. The calculation follows the formula in Oskari Timgren's
    thesis (page 131), using Wigner 3-j and 6-j symbols.

    Args:
        bra (CoupledBasisState): Coupled basis state object (typically ground state)
        ket (CoupledBasisState): Coupled basis state object (typically excited state)
        pol_vec (Tuple[complex, complex, complex]): Polarization vector in Cartesian
            basis [Ex, Ey, Ez]. Defaults to ((1+0j)/√2, 0j, (1+0j)/√2).
        rme_only (bool): If True, return only reduced matrix element without angular/
            polarization dependence. Defaults to False.

    Returns:
        complex: Electric dipole matrix element ⟨bra|D̂·ε|ket⟩

    Note:
        Cached for performance. Selection rules: |ΔΩ| < 2, |Δm_F| ≤ 1

    Example:
        >>> ME = ED_ME_coupled(ground_basis_state, excited_basis_state)
        >>> rme = ED_ME_coupled(ground_basis_state, excited_basis_state, rme_only=True)
    """
    # Safe early exits (apply to both full and reduced MEs)
    if abs(bra.Omega - ket.Omega) > 1:
        return 0j
    if abs(bra.J - ket.J) > 1:
        return 0j
    if abs(bra.F - ket.F) > 1:
        return 0j
    # ΔmF only matters for the full (non-reduced) ME
    if not rme_only and abs(bra.mF - ket.mF) > 1:
        return 0j

    # find quantum numbers for ground state
    F = bra.F
    mF = bra.mF
    J = bra.J
    F1 = bra.F1
    I1 = bra.I1
    I2 = bra.I2
    Omega = bra.Omega

    # find quantum numbers for excited state
    Fp = ket.F
    mFp = ket.mF
    Jp = ket.J
    F1p = ket.F1
    Omegap = ket.Omega

    q = Omega - Omegap
    if abs(q) > 1:
        return 0.0j

    # calculate the reduced matrix element
    # see Oskari Timgren's Thesis, page 131
    phase = (-1) ** (F1p + F1 + Fp + I1 + I2 - Omega)
    prefactor = math.sqrt(
        (2 * J + 1)
        * (2 * Jp + 1)
        * (2 * F1 + 1)
        * (2 * F1p + 1)
        * (2 * F + 1)
        * (2 * Fp + 1)
    )

    ME: complex = (
        phase
        * prefactor
        * sixj_f(F1p, Fp, I2, F, F1, 1)
        * sixj_f(Jp, F1p, I1, F1, J, 1)
        * threej_f(J, 1, Jp, -Omega, q, Omegap)
    )

    if ME == 0.0:
        return 0.0

    if not rme_only:
        ME *= angular_part(pol_vec, F, mF, Fp, mFp)

    return ME


@lru_cache(maxsize=int(1e6))
def angular_part(
    pol_vec: Tuple[complex, complex, complex],
    F: int,
    mF: int,
    Fp: int,
    mFp: int,
) -> complex:
    """Calculate polarization-dependent angular factor for electric dipole transitions.

    Computes the angular part of the matrix element including polarization dependence:
        (-1)^(F-m_F) * ⟨F, 1, F'; -m_F, q, m_F'⟩ * ε_q
    where q = m_F - m_F' and ε_q are spherical components of the polarization.

    Args:
        pol_vec (Tuple[complex, complex, complex]): Polarization vector in Cartesian
            basis [Ex, Ey, Ez]
        F (int): Total angular momentum quantum number of initial state
        mF (int): Projection of total angular momentum of initial state
        Fp (int): Total angular momentum quantum number of final state
        mFp (int): Projection of total angular momentum of final state

    Returns:
        complex: Angular factor with polarization. Returns 0 if |q| > 1 (selection rule)

    Note:
        Cached for performance. Spherical polarization components:
        - q=+1 (σ⁺): -(E_x + iE_y)/√2
        - q=0 (π): E_z
        - q=-1 (σ⁻): (E_x - iE_y)/√2
    """
    # q that connects the two Zeeman sub-levels
    q = mF - mFp
    if abs(q) > 1:
        return 0.0

    # Cartesian → spherical-basis components
    if q == 1:
        p_q = -1 / math.sqrt(2) * (pol_vec[0] + 1j * pol_vec[1])  # σ+
    elif q == -1:
        p_q = 1 / math.sqrt(2) * (pol_vec[0] - 1j * pol_vec[1])  # σ-
    elif q == 0:
        p_q = pol_vec[2]
    else:
        raise ValueError(f"Invalid q value: {q}")

    angular = (-1) ** (F - mF) * threej_f(F, 1, Fp, -mF, q, mFp) * p_q
    return angular
