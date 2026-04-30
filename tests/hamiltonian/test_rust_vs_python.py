"""Tests comparing Rust and Python backends for Hamiltonian generation.

These tests ensure the Rust implementations produce identical results to the
Python reference implementations for:
- X state Hamiltonian (uncoupled basis)
- B state Hamiltonian (coupled Omega basis)
- Basis transformation matrices
- Wigner 3j and 6j symbols
"""

import numpy as np
import pytest

from centrex_tlf import hamiltonian, states
from centrex_tlf.constants import BConstants, XConstants

rust = pytest.importorskip("centrex_tlf.centrex_tlf_rust")

from centrex_tlf.hamiltonian.generate_hamiltonian import (
    _generate_coupled_hamiltonian_B_python,
    _generate_uncoupled_hamiltonian_X_python,
)
from centrex_tlf.hamiltonian.basis_transformations import (
    _generate_transform_matrix_python,
)


class TestXStateHamiltonian:
    @pytest.fixture
    def x_states_j01(self):
        return states.generate_uncoupled_states_ground(Js=[0, 1])

    @pytest.fixture
    def x_states_j012(self):
        return states.generate_uncoupled_states_ground(Js=[0, 1, 2])

    def test_x_hamiltonian_matches_python_j01(self, x_states_j01):
        constants = XConstants()
        h_rust = rust.generate_uncoupled_hamiltonian_X_py(x_states_j01, constants)
        h_python = _generate_uncoupled_hamiltonian_X_python(x_states_j01, constants)

        np.testing.assert_allclose(h_rust.Hff, h_python.Hff, atol=1e-6)
        np.testing.assert_allclose(h_rust.HSx, h_python.HSx, atol=1e-6)
        np.testing.assert_allclose(h_rust.HSy, h_python.HSy, atol=1e-6)
        np.testing.assert_allclose(h_rust.HSz, h_python.HSz, atol=1e-6)
        np.testing.assert_allclose(h_rust.HZx, h_python.HZx, atol=1e-6)
        np.testing.assert_allclose(h_rust.HZy, h_python.HZy, atol=1e-6)
        np.testing.assert_allclose(h_rust.HZz, h_python.HZz, atol=1e-6)

    def test_x_hamiltonian_matches_python_j012(self, x_states_j012):
        constants = XConstants()
        h_rust = rust.generate_uncoupled_hamiltonian_X_py(x_states_j012, constants)
        h_python = _generate_uncoupled_hamiltonian_X_python(x_states_j012, constants)

        np.testing.assert_allclose(h_rust.Hff, h_python.Hff, atol=1e-6)
        np.testing.assert_allclose(h_rust.HSx, h_python.HSx, atol=1e-6)
        np.testing.assert_allclose(h_rust.HSy, h_python.HSy, atol=1e-6)
        np.testing.assert_allclose(h_rust.HSz, h_python.HSz, atol=1e-6)
        np.testing.assert_allclose(h_rust.HZx, h_python.HZx, atol=1e-6)
        np.testing.assert_allclose(h_rust.HZy, h_python.HZy, atol=1e-6)
        np.testing.assert_allclose(h_rust.HZz, h_python.HZz, atol=1e-6)

    def test_x_hamiltonian_hermitian(self, x_states_j01):
        constants = XConstants()
        h = rust.generate_uncoupled_hamiltonian_X_py(x_states_j01, constants)
        for matrix in [h.Hff, h.HSx, h.HSy, h.HSz, h.HZx, h.HZy, h.HZz]:
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)

    def test_x_field_free_eigenvalues(self):
        qn = states.generate_uncoupled_states_ground(Js=[0, 1])
        constants = XConstants()
        h = rust.generate_uncoupled_hamiltonian_X_py(qn, constants)
        evals = np.linalg.eigvalsh(2 * np.pi * h.Hff)
        assert evals.shape[0] == len(qn)
        assert np.all(np.isfinite(evals))


class TestBStateHamiltonian:
    @pytest.fixture
    def b_states_j1(self):
        return states.generate_coupled_states_excited(
            Js=[1], Ps=None, Omegas=[-1, 1]
        )

    @pytest.fixture
    def b_states_j12(self):
        return states.generate_coupled_states_excited(
            Js=[1, 2], Ps=None, Omegas=[-1, 1]
        )

    def test_b_hamiltonian_matches_python_j1(self, b_states_j1):
        constants = BConstants()
        h_rust = rust.generate_coupled_hamiltonian_B_py(b_states_j1, constants)
        h_python = _generate_coupled_hamiltonian_B_python(b_states_j1, constants)

        np.testing.assert_allclose(h_rust.Hrot, h_python.Hrot, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_mhf_Tl, h_python.H_mhf_Tl, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_mhf_F, h_python.H_mhf_F, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_LD, h_python.H_LD, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_cp1_Tl, h_python.H_cp1_Tl, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_c_Tl, h_python.H_c_Tl, atol=1e-4)
        np.testing.assert_allclose(h_rust.HSx, h_python.HSx, atol=1e-4)
        np.testing.assert_allclose(h_rust.HSy, h_python.HSy, atol=1e-4)
        np.testing.assert_allclose(h_rust.HSz, h_python.HSz, atol=1e-4)
        np.testing.assert_allclose(h_rust.HZx, h_python.HZx, atol=1e-4)
        np.testing.assert_allclose(h_rust.HZy, h_python.HZy, atol=1e-4)
        np.testing.assert_allclose(h_rust.HZz, h_python.HZz, atol=1e-4)

    def test_b_hamiltonian_matches_python_j12(self, b_states_j12):
        constants = BConstants()
        h_rust = rust.generate_coupled_hamiltonian_B_py(b_states_j12, constants)
        h_python = _generate_coupled_hamiltonian_B_python(b_states_j12, constants)

        np.testing.assert_allclose(h_rust.Hrot, h_python.Hrot, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_mhf_Tl, h_python.H_mhf_Tl, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_mhf_F, h_python.H_mhf_F, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_LD, h_python.H_LD, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_cp1_Tl, h_python.H_cp1_Tl, atol=1e-4)
        np.testing.assert_allclose(h_rust.H_c_Tl, h_python.H_c_Tl, atol=1e-4)
        np.testing.assert_allclose(h_rust.HSx, h_python.HSx, atol=1e-4)
        np.testing.assert_allclose(h_rust.HSy, h_python.HSy, atol=1e-4)
        np.testing.assert_allclose(h_rust.HSz, h_python.HSz, atol=1e-4)
        np.testing.assert_allclose(h_rust.HZx, h_python.HZx, atol=1e-4)
        np.testing.assert_allclose(h_rust.HZy, h_python.HZy, atol=1e-4)
        np.testing.assert_allclose(h_rust.HZz, h_python.HZz, atol=1e-4)

    def test_b_hamiltonian_hermitian(self, b_states_j1):
        constants = BConstants()
        h = rust.generate_coupled_hamiltonian_B_py(b_states_j1, constants)
        for matrix in [
            h.Hrot, h.H_mhf_Tl, h.H_mhf_F, h.H_LD, h.H_cp1_Tl, h.H_c_Tl,
            h.HSx, h.HSy, h.HSz, h.HZx, h.HZy, h.HZz,
        ]:
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)

    def test_b_field_free_eigenvalues(self, b_states_j1):
        constants = BConstants()
        h = rust.generate_coupled_hamiltonian_B_py(b_states_j1, constants)
        h_ff = h.Hrot + h.H_mhf_Tl + h.H_mhf_F + h.H_LD + h.H_cp1_Tl + h.H_c_Tl
        evals = np.linalg.eigvalsh(2 * np.pi * h_ff)
        assert evals.shape[0] == len(b_states_j1)
        assert np.all(np.isfinite(evals))

    def test_b_hamiltonian_parity_basis(self):
        b_states = states.generate_coupled_states_excited(
            Js=[1, 2], Ps=[-1, 1], Omegas=1
        )
        constants = BConstants()
        h_rust = rust.generate_coupled_hamiltonian_B_py(b_states, constants)
        h_python = _generate_coupled_hamiltonian_B_python(b_states, constants)
        np.testing.assert_allclose(h_rust.Hrot, h_python.Hrot, atol=1e-4)
        np.testing.assert_allclose(h_rust.HZz, h_python.HZz, atol=1e-4)


class TestTransformMatrix:
    def test_coupled_to_uncoupled_matches_python(self):
        coupled = states.generate_coupled_states_ground(Js=[0, 1])
        uncoupled = states.generate_uncoupled_states_ground(Js=[0, 1])
        s_rust = rust.generate_transform_matrix_py(coupled, uncoupled)
        s_python = _generate_transform_matrix_python(coupled, uncoupled)
        np.testing.assert_allclose(s_rust, s_python, atol=1e-12)

    def test_uncoupled_to_coupled_matches_python(self):
        coupled = states.generate_coupled_states_ground(Js=[0, 1])
        uncoupled = states.generate_uncoupled_states_ground(Js=[0, 1])
        s_rust = rust.generate_transform_matrix_py(uncoupled, coupled)
        s_python = _generate_transform_matrix_python(uncoupled, coupled)
        np.testing.assert_allclose(s_rust, s_python, atol=1e-12)

    def test_transform_unitarity(self):
        coupled = states.generate_coupled_states_ground(Js=[0, 1])
        uncoupled = states.generate_uncoupled_states_ground(Js=[0, 1])
        s = rust.generate_transform_matrix_py(coupled, uncoupled)
        product = s.conj().T @ s
        np.testing.assert_allclose(product, np.eye(len(uncoupled)), atol=1e-12)

    def test_transform_roundtrip(self):
        coupled = states.generate_coupled_states_ground(Js=[0, 1])
        uncoupled = states.generate_uncoupled_states_ground(Js=[0, 1])
        s = rust.generate_transform_matrix_py(coupled, uncoupled)
        s_inv = rust.generate_transform_matrix_py(uncoupled, coupled)
        product = s @ s_inv
        np.testing.assert_allclose(product, np.eye(len(coupled)), atol=1e-12)

    def test_transform_larger_basis(self):
        coupled = states.generate_coupled_states_ground(Js=[0, 1, 2])
        uncoupled = states.generate_uncoupled_states_ground(Js=[0, 1, 2])
        s_rust = rust.generate_transform_matrix_py(coupled, uncoupled)
        s_python = _generate_transform_matrix_python(coupled, uncoupled)
        np.testing.assert_allclose(s_rust, s_python, atol=1e-12)


class TestCouplingMatrix:
    @pytest.fixture
    def coupling_setup_omega(self):
        """Setup with Omega basis states (no parity superposition ambiguity)."""
        qn_select = states.QuantumSelector(J=1, Ω=0, electronic=states.ElectronicState.X)
        ground = states.generate_coupled_states_X(qn_select)

        qn_select = states.QuantumSelector(
            J=1, F1=1 / 2, F=1, Ω=1, electronic=states.ElectronicState.B
        )
        excited = states.generate_coupled_states_B(qn_select, basis=states.Basis.CoupledΩ)

        QN = [1 * s for s in ground + excited]
        ground = [1 * s for s in ground]
        excited = [1 * s for s in excited]
        ground_indices = [QN.index(gs) for gs in ground]
        excited_indices = [QN.index(es) for es in excited]
        return QN, ground, excited, ground_indices, excited_indices

    def test_coupling_matrix_matches_python(self, coupling_setup_omega):
        from centrex_tlf.couplings.coupling_matrix import _generate_coupling_matrix_python

        QN, ground, excited, ground_idx, excited_idx = coupling_setup_omega
        pol_vec = np.array([0.0, 0.0, 1.0])

        h_rust = rust.generate_coupling_matrix_py(QN, ground_idx, excited_idx, pol_vec, False)
        h_python = _generate_coupling_matrix_python(
            QN, ground, excited, pol_vec, False, normalize_pol=False
        )
        np.testing.assert_allclose(h_rust, h_python, atol=1e-12)

    def test_coupling_matrix_hermitian(self, coupling_setup_omega):
        QN, ground, excited, ground_idx, excited_idx = coupling_setup_omega
        pol_vec = np.array([0.0, 0.0, 1.0])
        h = rust.generate_coupling_matrix_py(QN, ground_idx, excited_idx, pol_vec, False)
        np.testing.assert_allclose(h, h.conj().T, atol=1e-12)

    @pytest.mark.parametrize("pol_vec", [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1j, 0.0]) / np.sqrt(2),
    ])
    def test_coupling_matrix_polarizations(self, coupling_setup_omega, pol_vec):
        from centrex_tlf.couplings.coupling_matrix import _generate_coupling_matrix_python

        QN, ground, excited, ground_idx, excited_idx = coupling_setup_omega
        h_rust = rust.generate_coupling_matrix_py(QN, ground_idx, excited_idx, pol_vec, False)
        h_python = _generate_coupling_matrix_python(
            QN, ground, excited, pol_vec, False, normalize_pol=False
        )
        np.testing.assert_allclose(h_rust, h_python, atol=1e-12)

    def test_coupling_matrix_reduced(self, coupling_setup_omega):
        from centrex_tlf.couplings.coupling_matrix import _generate_coupling_matrix_python

        QN, ground, excited, ground_idx, excited_idx = coupling_setup_omega
        pol_vec = np.array([0.0, 0.0, 1.0])
        h_rust = rust.generate_coupling_matrix_py(QN, ground_idx, excited_idx, pol_vec, True)
        h_python = _generate_coupling_matrix_python(
            QN, ground, excited, pol_vec, True, normalize_pol=False
        )
        np.testing.assert_allclose(h_rust, h_python, atol=1e-12)


class TestWignerSymbols:
    @pytest.mark.parametrize("j1,j2,j3,m1,m2,m3,expected", [
        (1, 1, 0, 0, 0, 0, -1 / np.sqrt(3)),
        (1, 1, 1, 1, 0, -1, -1 / np.sqrt(6)),
        (1, 1, 2, 0, 0, 0, 1 / np.sqrt(5) * np.sqrt(2 / 3)),
        (2, 1, 1, 0, 0, 0, 1 / np.sqrt(5) * np.sqrt(2 / 3)),
        (0.5, 0.5, 0, 0.5, -0.5, 0, 1 / np.sqrt(2)),
        (0.5, 0.5, 1, 0.5, -0.5, 0, 1 / np.sqrt(6)),
        (0.5, 0.5, 1, 0.5, 0.5, -1, -1 / np.sqrt(3)),
        (1, 1, 0, 1, -1, 0, 1 / np.sqrt(3)),
        (1, 0, 1, 0, 0, 0, -1 / np.sqrt(3)),
        (2, 2, 0, 0, 0, 0, 1 / np.sqrt(5)),
    ])
    def test_wigner_3j(self, j1, j2, j3, m1, m2, m3, expected):
        result = rust.wigner_3j_py(j1, j2, j3, m1, m2, m3)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    @pytest.mark.parametrize("j1,j2,j3,m1,m2,m3", [
        (1, 1, 0, 1, 0, 0),
        (1, 1, 3, 0, 0, 0),
        (1, 1, 1, 2, 0, 0),
    ])
    def test_wigner_3j_selection_rules(self, j1, j2, j3, m1, m2, m3):
        result = rust.wigner_3j_py(j1, j2, j3, m1, m2, m3)
        assert result == 0.0

    @pytest.mark.parametrize("j1,j2,j3,j4,j5,j6,expected", [
        (1, 1, 1, 1, 1, 1, 1 / 6),
        (0.5, 0.5, 1, 0.5, 0.5, 0, 1 / np.sqrt(4)),
        (1, 1, 0, 1, 1, 1, -1 / 3),
        (2, 1, 1, 1, 2, 2, 0.15275252316519466),
    ])
    def test_wigner_6j(self, j1, j2, j3, j4, j5, j6, expected):
        result = rust.wigner_6j_py(j1, j2, j3, j4, j5, j6)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_wigner_3j_symmetry(self):
        j1, j2, j3 = 1.0, 1.0, 2.0
        m1, m2, m3 = 1.0, 0.0, -1.0
        val = rust.wigner_3j_py(j1, j2, j3, m1, m2, m3)
        phase = (-1) ** (j1 + j2 + j3)
        val_swap_cols = rust.wigner_3j_py(j2, j1, j3, m2, m1, m3)
        np.testing.assert_allclose(val_swap_cols, phase * val, atol=1e-14)
        val_negate_m = rust.wigner_3j_py(j1, j2, j3, -m1, -m2, -m3)
        np.testing.assert_allclose(val_negate_m, phase * val, atol=1e-14)

    def test_wigner_6j_symmetry(self):
        j1, j2, j3, j4, j5, j6 = 1.0, 2.0, 2.0, 1.0, 1.0, 2.0
        val = rust.wigner_6j_py(j1, j2, j3, j4, j5, j6)
        val_swap_cols = rust.wigner_6j_py(j2, j1, j3, j5, j4, j6)
        np.testing.assert_allclose(val_swap_cols, val, atol=1e-14)
        val_swap_rows = rust.wigner_6j_py(j4, j5, j3, j1, j2, j6)
        np.testing.assert_allclose(val_swap_rows, val, atol=1e-14)

    def test_wigner_3j_matches_python(self):
        from centrex_tlf.hamiltonian.wigner import threej_f
        cases = [
            (1, 1, 0, 0, 0, 0),
            (1, 1, 2, 1, -1, 0),
            (2, 1, 1, -1, 0, 1),
            (0.5, 0.5, 1, 0.5, -0.5, 0),
            (1, 1, 1, 1, 0, -1),
            (3, 2, 1, -2, 1, 1),
        ]
        for j1, j2, j3, m1, m2, m3 in cases:
            rust_val = rust.wigner_3j_py(j1, j2, j3, m1, m2, m3)
            python_val = threej_f(j1, j2, j3, m1, m2, m3)
            np.testing.assert_allclose(
                rust_val, python_val, atol=1e-14,
                err_msg=f"3j mismatch for ({j1},{j2},{j3},{m1},{m2},{m3})"
            )

    def test_wigner_6j_matches_python(self):
        from centrex_tlf.hamiltonian.wigner import sixj_f
        cases = [
            (1, 1, 1, 1, 1, 1),
            (0.5, 0.5, 1, 0.5, 0.5, 0),
            (1, 1, 0, 1, 1, 1),
            (2, 1, 1, 1, 2, 2),
            (2, 2, 1, 1, 1, 2),
        ]
        for j1, j2, j3, j4, j5, j6 in cases:
            rust_val = rust.wigner_6j_py(j1, j2, j3, j4, j5, j6)
            python_val = sixj_f(j1, j2, j3, j4, j5, j6)
            np.testing.assert_allclose(
                rust_val, python_val, atol=1e-14,
                err_msg=f"6j mismatch for ({j1},{j2},{j3},{j4},{j5},{j6})"
            )
