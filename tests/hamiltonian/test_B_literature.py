"""B-state Hamiltonian against published numbers.

Zero-field structure and g-factors of B(v=0), J~ = 1 and 2, compared with

- Meijer & Sartakov, Phys. Rev. A 101, 042506 (2020), Table 2: Lambda-doublet
  splittings E_e - E_f and g_F for an isolated Omega = 1 state;
- Clayburn et al., Phys. Rev. A 102, 052802 (2020), Table I: measured J~ = 1
  hyperfine and Lambda-doublet splittings.

The g-factor test also pins the *sign* of the Zeeman term: the electronic
orbital moment is antiparallel to L, so a level with g_F > 0 and mF > 0 moves up
in a field along +z. Rust-versus-Python comparisons cannot catch a sign error
shared by both implementations.
"""

import numpy as np
import pytest

from centrex_tlf import hamiltonian, states
from centrex_tlf.constants import BConstants
from centrex_tlf.hamiltonian.generate_hamiltonian import (
    _generate_coupled_hamiltonian_B_python,
)

MHZ = 1e6


def _parity_states():
    return list(
        states.generate_coupled_states_B(
            states.QuantumSelector(
                J=[1, 2, 3, 4],
                F1=None,
                F=None,
                mF=None,
                electronic=states.ElectronicState.B,
                P=[-1, 1],
                Ω=1,
            )
        )
    )


def _levels(QN, H, B_z=0.0):
    """Eigenvalues (Hz) of the stretched mF = F levels, keyed by (J, F1, F, P)."""
    H0 = H.Hrot + H.H_mhf_Tl + H.H_mhf_F + H.H_LD + H.H_cp1_Tl + H.H_c_Tl
    w, v = np.linalg.eigh(H0 + B_z * H.HZz)
    levels = {}
    for k in range(len(w)):
        s = QN[int(np.argmax(np.abs(v[:, k])))]
        if s.mF == s.F:
            levels[(s.J, s.F1, s.F, s.P)] = w[k]
    return levels


@pytest.fixture(scope="module")
def QN():
    return _parity_states()


@pytest.fixture(scope="module")
def H(QN):
    return hamiltonian.generate_coupled_hamiltonian_B(QN)


@pytest.fixture(scope="module")
def zero_field(QN, H):
    return _levels(QN, H)


# (J, F1, F): (E_e - E_f in MHz, quoted standard deviation), e is P = (-1)^J.
MEIJER_SARTAKOV_DOUBLETS = {
    (1, 0.5, 0): (16, 1),
    (1, 0.5, 1): (16, 1),
    (1, 1.5, 1): (-18, 1),
    (1, 1.5, 2): (-18, 1),
    (2, 2.5, 2): (-14, 2),
    (2, 2.5, 3): (-14, 2),
    (2, 1.5, 1): (15, 4),
    (2, 1.5, 2): (14, 4),
}

MEIJER_SARTAKOV_G_F = {
    (1, 0.5, 1): 0.32,
    (1, 1.5, 1): 0.25,
    (1, 1.5, 2): 0.14,
    (2, 2.5, 2): 0.11,
    (2, 2.5, 3): 0.08,
    (2, 1.5, 1): 0.43,
    (2, 1.5, 2): 0.25,
}


@pytest.mark.parametrize("level", list(MEIJER_SARTAKOV_DOUBLETS))
def test_lambda_doublet_splitting_matches_meijer_sartakov(zero_field, level):
    J, F1, F = level
    expected, sd = MEIJER_SARTAKOV_DOUBLETS[level]
    e = zero_field[(J, F1, F, (-1) ** J)]
    f = zero_field[(J, F1, F, -((-1) ** J))]
    assert (e - f) / MHZ == pytest.approx(expected, abs=sd)


def test_J1_F0_doublet_is_2q_plus_c1p(zero_field):
    # Meijer & Sartakov Eq. (4), with C'_I(F) = 0.
    c = BConstants()
    split = zero_field[(1, 0.5, 0, -1)] - zero_field[(1, 0.5, 0, 1)]
    assert split == pytest.approx(2 * c.q + c.c1p_Tl, rel=1e-6)


def test_J1_hyperfine_splittings_match_clayburn(zero_field):
    """Clayburn Table I, measured; tolerance 2.5 of their standard deviations."""
    E = {k: v / MHZ for k, v in zero_field.items()}
    # a: F=1 - F=0 within F1=1/2; c: F=2 - F=1 within F1=3/2;
    # b: (F1=3/2, F=1) - (F1=1/2, F=1). Superscript is the parity.
    measured = {
        "a+": (E[(1, 0.5, 1, 1)] - E[(1, 0.5, 0, 1)], 560, 2),
        "a-": (E[(1, 0.5, 1, -1)] - E[(1, 0.5, 0, -1)], 562, 2),
        "c+": (E[(1, 1.5, 2, 1)] - E[(1, 1.5, 1, 1)], 315, 2),
        "c-": (E[(1, 1.5, 2, -1)] - E[(1, 1.5, 1, -1)], 315, 2),
        "b+": (E[(1, 1.5, 1, 1)] - E[(1, 0.5, 1, 1)], 13532, 3),
        "b-": (E[(1, 1.5, 1, -1)] - E[(1, 0.5, 1, -1)], 13500, 3),
        "w1": (E[(1, 0.5, 1, -1)] - E[(1, 0.5, 1, 1)], 14.4, 0.9),
        "w2": (E[(1, 1.5, 1, 1)] - E[(1, 1.5, 1, -1)], 17.4, 1.1),
    }
    for label, (value, expected, sd) in measured.items():
        assert value == pytest.approx(expected, abs=2.5 * sd), label


@pytest.mark.parametrize("backend", ["default", "python"])
@pytest.mark.parametrize("level", list(MEIJER_SARTAKOV_G_F))
def test_g_factor_sign_and_magnitude(QN, H, zero_field, level, backend):
    """E(mF) = +g_F mu_B mF B_z with the positive g_F of Meijer & Sartakov."""
    if backend == "python":
        H = _generate_coupled_hamiltonian_B_python(QN, BConstants())
        base = _levels(QN, H)
    else:
        base = zero_field
    J, F1, F = level
    key = (J, F1, F, -((-1) ** J))  # f parity
    dB = 1e-3  # Gauss
    shifted = _levels(QN, H, B_z=dB)
    g_F = (shifted[key] - base[key]) / (dB * BConstants().μ_B * F)
    assert g_F == pytest.approx(MEIJER_SARTAKOV_G_F[level], abs=0.006)


def test_stretched_state_zeeman_shift_is_positive():
    """Pure |J, Omega, mJ = J> with both spins up: <H_Zz> = +mu_B mJ / (J(J+1))."""
    from centrex_tlf.hamiltonian.B_coupled_Omega import zeeman
    from centrex_tlf.states import CoupledBasisState, ElectronicState

    c = BConstants()
    for Omega in (1, -1):
        psi = CoupledBasisState(
            F=3, mF=3, F1=2.5, J=2, I1=0.5, I2=0.5, Omega=Omega, P=None,
            electronic_state=ElectronicState.B,
        )
        diagonal = [a for a, s in zeeman.HZz(psi, c).data if s == psi]
        assert len(diagonal) == 1
        assert np.real(diagonal[0]) == pytest.approx(c.μ_B * 2 / 6, rel=1e-12)
