import math
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from centrex_tlf import states

from .wigner import sixj_f, threej_f

# You already have threej_f, sixj_f somewhere
# from your.wigner.module import threej_f, sixj_f

__all__ = [
    "cartesian_to_spherical_rank1",
    "couple_two_rank1_to_rank2",
    "angular_part_rank2",
    "EQ_ME_coupled",
    "generate_EQ_ME_mixed_state",
]


def cartesian_to_spherical_rank1(
    vec: Tuple[complex, complex, complex],
) -> Dict[int, complex]:
    """Convert Cartesian (x, y, z) to spherical rank-1 components v_q, q=-1,0,+1.

    Convention:
        v_{+1} = - (v_x + i v_y)/sqrt(2)
        v_{ 0} = v_z
        v_{-1} =   (v_x - i v_y)/sqrt(2)
    """
    vx, vy, vz = vec
    return {
        +1: -1.0 / math.sqrt(2.0) * (vx + 1j * vy),
        0: vz,
        -1: 1.0 / math.sqrt(2.0) * (vx - 1j * vy),
    }


def couple_two_rank1_to_rank2(
    e_vec: Tuple[complex, complex, complex],
    k_vec: Tuple[complex, complex, complex],
) -> Tuple[complex, complex, complex, complex, complex]:
    """Build rank-2 spherical tensor T^{(2)}_Q from two vectors e and k.

    T^{(2)}_Q = sum_{q1,q2} ⟨1,q1; 1,q2 | 2,Q⟩ e^{(1)}_{q1} k^{(1)}_{q2}

    Args:
        e_vec: Cartesian polarization vector (E_x, E_y, E_z)
        k_vec: Cartesian propagation (or geometry) vector (k_x, k_y, k_z)

    Returns:
        Tuple (T_{-2}, T_{-1}, T_0, T_{+1}, T_{+2})
    """
    e_sph = cartesian_to_spherical_rank1(e_vec)
    k_sph = cartesian_to_spherical_rank1(k_vec)

    T = {Q: 0.0 + 0.0j for Q in range(-2, 3)}

    for q1 in [-1, 0, +1]:
        for q2 in [-1, 0, +1]:
            for Q in range(-2, 3):
                # CG coefficient  ⟨1,q1;1,q2|2,Q⟩
                # In Wigner 3-j notation:
                #   ⟨1,q1;1,q2|2,Q⟩
                #   = (-1)^{1-1+Q} * sqrt(2*2+1) * (1 1 2; q1 q2 -Q)
                cg = (
                    (-1) ** (1 - 1 + Q)
                    * math.sqrt(2 * 2 + 1)
                    * threej_f(1, 1, 2, q1, q2, -Q)
                )
                T[Q] += cg * e_sph[q1] * k_sph[q2]

    # Return in order (-2, -1, 0, +1, +2)
    return (T[-2], T[-1], T[0], T[+1], T[+2])


@lru_cache(maxsize=int(1e6))
def angular_part_rank2(
    field_tensor: Tuple[complex, complex, complex, complex, complex],
    F: int,
    mF: int,
    Fp: int,
    mFp: int,
) -> complex:
    """Angular factor for rank-2 tensor in F, mF basis.

    Computes:
        (-1)^(F-m_F) * ( F  2  F'
                        -m_F p m_F') * T^{(2)}_p

    where p = m_F - m_F', and T^{(2)}_p are the spherical rank-2 components
    of the effective field tensor (p = -2..+2).
    """
    p = mF - mFp
    if abs(p) > 2:
        return 0.0

    # index mapping: p -> 0..4
    idx = p + 2
    T_p = field_tensor[idx]

    return (-1) ** (F - mF) * threej_f(F, 2, Fp, -mF, p, mFp) * T_p


@lru_cache(maxsize=int(1e6))
def EQ_ME_coupled(
    bra: states.CoupledBasisState,
    ket: states.CoupledBasisState,
    pol_vec: Tuple[complex, complex, complex] | None = None,
    k_vec: Tuple[complex, complex, complex] | None = None,
    rme_only: bool = False,
) -> complex:
    """Electric quadrupole (rank-2) matrix element between coupled basis states.

    This mirrors ED_ME_coupled but with tensor rank k=2. It includes Ω-coupling
    via a J,Ω three-j symbol, so ΔJ can be 0, ±1, ±2 and |ΔΩ| ≤ 2.

    IMPORTANT:
        • This function expects *basis states in the Ω basis* (Basis.Coupled),
          NOT in the parity basis (Basis.CoupledP).
        • If your states are in the parity basis, first transform them to the
          Ω basis (or use generate_EQ_ME_mixed_state, which does that for you).

    Args:
        bra: Initial coupled basis state (e.g. X or B state) in Ω basis.
        ket: Final coupled basis state in Ω basis.
        pol_vec: Ignored if rme_only=True. If rme_only=False, Cartesian
            polarization vector (E_x, E_y, E_z).
        k_vec: Ignored if rme_only=True. If rme_only=False, Cartesian
            propagation/geometry vector (k_x, k_y, k_z).
        rme_only: If True, return only the reduced matrix element (no m_F or
            field-geometry dependence).

    Returns:
        complex: Quadrupole matrix element ⟨bra|Q̂^{(2)}·T^{(2)}|ket⟩, or the
        reduced matrix element if rme_only=True.

    Raises:
        ValueError:
            • If bra or ket are in the parity basis (Basis.CoupledP).
            • If required quantum numbers (J, F, Ω) are not set.
            • If pol_vec / k_vec are missing when rme_only=False.
    """
    # 0. Safety: require Ω-basis, not parity basis
    if (
        getattr(bra, "basis", None) is states.Basis.CoupledP
        or getattr(ket, "basis", None) is states.Basis.CoupledP
    ):
        raise ValueError(
            "EQ_ME_coupled expects CoupledBasisState in the Ω basis "
            "(Basis.Coupled), not in the parity basis (Basis.CoupledP). "
            "Use generate_EQ_ME_mixed_state on CoupledState objects, or "
            "transform basis states to the Ω basis before calling."
        )

    # 1. Extract quantum numbers and check they exist
    F = bra.F
    mF = bra.mF
    J = bra.J
    F1 = bra.F1
    I1 = bra.I1
    I2 = bra.I2
    Omega = bra.Omega

    Fp = ket.F
    mFp = ket.mF
    Jp = ket.J
    F1p = ket.F1
    Omegap = ket.Omega

    if J is None or Jp is None:
        raise ValueError("Both states must have total J set for EQ_ME_coupled")
    if F is None or Fp is None:
        raise ValueError("Both states must have total F set for EQ_ME_coupled")
    if Omega is None or Omegap is None:
        raise ValueError("Both states must have Ω set for EQ_ME_coupled")

    k_rank = 2  # tensor rank for E2

    # 2. Ω-coupling selection rule: |ΔΩ| ≤ 2
    q_Omega = Omega - Omegap
    if abs(q_Omega) > k_rank:
        return 0.0

    # 3. Reduced matrix element in hyperfine + rotational angular momentum
    ME: complex = (
        (-1) ** (F1p + F1 + Fp + I1 + I2 - Omega)
        * math.sqrt(
            (2 * J + 1)
            * (2 * Jp + 1)
            * (2 * F1 + 1)
            * (2 * F1p + 1)
            * (2 * F + 1)
            * (2 * Fp + 1)
        )
        * sixj_f(F1p, Fp, I2, F, F1, k_rank)
        * sixj_f(Jp, F1p, I1, F1, J, k_rank)
        * threej_f(J, k_rank, Jp, -Omega, q_Omega, Omegap)
    )

    if rme_only:
        return ME

    # 4. Full matrix element: need polarization / geometry
    if pol_vec is None or k_vec is None:
        raise ValueError("pol_vec and k_vec are required unless rme_only=True")

    field_tensor = couple_two_rank1_to_rank2(pol_vec, k_vec)
    ME *= angular_part_rank2(field_tensor, F, mF, Fp, mFp)
    return ME


def generate_EQ_ME_mixed_state(
    bra: states.CoupledState,
    ket: states.CoupledState,
    pol_vec: npt.NDArray[np.complex128] | None = None,
    k_vec: npt.NDArray[np.complex128] | None = None,
    reduced: bool = False,
    normalize_vectors: bool = True,
) -> complex:
    """Electric quadrupole matrix element between mixed (superposition) states.

    Computes ⟨bra|Q̂^{(2)}·T^{(2)}|ket⟩ where T^{(2)} is built from pol_vec
    and k_vec as a coupled rank-2 spherical tensor.

    Args:
        bra, ket: Mixed CoupledState objects (superpositions)
        pol_vec: Polarization vector [E_x, E_y, E_z] in Cartesian basis.
        k_vec: Propagation/geometry vector [k_x, k_y, k_z] in Cartesian basis.
        reduced: If True, return only reduced matrix element, no m_F or
            field-geometry dependence.
        normalize_vectors: If True, normalize pol_vec and k_vec.

    Returns:
        complex: Electric quadrupole matrix element.
    """
    if pol_vec is None:
        # some default, for example linear polarization along x
        pol_vec = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    if k_vec is None:
        # some default, for example propagation along z
        k_vec = np.array([0.0 + 0j, 0.0 + 0j, 1.0 + 0j], dtype=np.complex128)

    if normalize_vectors:
        pol_norm = np.linalg.norm(pol_vec)
        if pol_norm != 0:
            pol_vec = pol_vec / pol_norm
        k_norm = np.linalg.norm(k_vec)
        if k_norm != 0:
            k_vec = k_vec / k_norm

    # parity → Omega basis if needed, same as in your ED code
    if bra.largest.basis is states.Basis.CoupledP:
        bra = bra.transform_to_omega_basis()
    if ket.largest.basis is states.Basis.CoupledP:
        ket = ket.transform_to_omega_basis()

    ME = 0j
    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += (
                amp_bra.conjugate()
                * amp_ket
                * EQ_ME_coupled(
                    basis_bra,
                    basis_ket,
                    pol_vec=tuple(pol_vec),
                    k_vec=tuple(k_vec),
                    rme_only=reduced,
                )
            )

    return ME
