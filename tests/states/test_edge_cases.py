import numpy as np
import pytest

from centrex_tlf import states
from centrex_tlf.states import (
    CoupledBasisState,
    CoupledState,
    ElectronicState,
    UncoupledBasisState,
    UncoupledState,
)


def _make_coupled_basis_state(**overrides):
    defaults = dict(
        F=1, mF=0, F1=0.5, J=0, I1=0.5, I2=0.5,
        Omega=0, P=1, electronic_state=ElectronicState.X,
    )
    defaults.update(overrides)
    return CoupledBasisState(**defaults)


class TestCoupledBasisStateEdgeCases:
    def test_all_none_decay_placeholder(self):
        s = CoupledBasisState(None, None, None, None, None, None, v="decay")
        assert s.F is None
        assert s.v == "decay"

    def test_equality_different_types(self):
        s = _make_coupled_basis_state()
        assert s != "not a state"
        assert s != 42

    def test_matmul_wrong_type(self):
        s = _make_coupled_basis_state()
        with pytest.raises(TypeError):
            s @ "invalid"

    def test_matmul_self_is_one(self):
        s = _make_coupled_basis_state()
        assert s @ s == 1

    def test_matmul_different_is_zero(self):
        s1 = _make_coupled_basis_state(F=1, mF=0)
        s2 = _make_coupled_basis_state(F=1, mF=1)
        assert s1 @ s2 == 0

    def test_omega_alias_kwarg(self):
        s = CoupledBasisState(
            F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5,
            P=1, electronic_state=ElectronicState.B, Ω=1,
        )
        assert s.Omega == 1
        assert s.Ω == 1

    def test_frozen_immutability(self):
        s = _make_coupled_basis_state()
        with pytest.raises(AttributeError):
            s.F = 99


class TestUncoupledBasisStateEdgeCases:
    def test_missing_parity_raises(self):
        with pytest.raises(ValueError):
            UncoupledBasisState(
                J=1, mJ=0, I1=0.5, m1=0.5, I2=0.5, m2=0.5,
                Omega=0, electronic_state=ElectronicState.X,
            )

    def test_missing_electronic_state_raises(self):
        with pytest.raises(ValueError):
            UncoupledBasisState(
                J=1, mJ=0, I1=0.5, m1=0.5, I2=0.5, m2=0.5, Omega=0, P=1,
            )

    def test_matmul_self_is_one(self):
        s = UncoupledBasisState(
            J=1, mJ=0, I1=0.5, m1=0.5, I2=0.5, m2=0.5,
            Omega=0, P=1, electronic_state=ElectronicState.X,
        )
        assert s @ s == 1


class TestStateVectorEdgeCases:
    def test_state_vector_empty_basis(self):
        s = _make_coupled_basis_state()
        state = 1 * s
        vec = state.state_vector([])
        assert len(vec) == 0

    def test_state_vector_no_overlap(self):
        s1 = _make_coupled_basis_state(F=1, mF=0)
        s2 = _make_coupled_basis_state(F=1, mF=1)
        state = 1 * s1
        vec = state.state_vector([s2])
        assert vec[0] == 0

    def test_state_vector_with_overlap(self):
        s = _make_coupled_basis_state()
        state = 1 * s
        vec = state.state_vector([s])
        assert vec[0] == pytest.approx(1.0)


class TestCoupledStateEdgeCases:
    def test_zero_amplitude_components(self):
        s1 = _make_coupled_basis_state(F=1, mF=0)
        s2 = _make_coupled_basis_state(F=1, mF=1)
        state = 0 * s1 + 1 * s2
        assert len(state.data) == 1

    def test_add_states(self):
        s1 = _make_coupled_basis_state(F=1, mF=0)
        s2 = _make_coupled_basis_state(F=1, mF=1)
        state = 1 * s1 + 2 * s2
        assert len(state.data) == 2

    def test_is_coupled_property(self):
        s = _make_coupled_basis_state()
        assert s.isCoupled is True
        assert s.isUncoupled is False

    def test_is_uncoupled_property(self):
        s = UncoupledBasisState(
            J=1, mJ=0, I1=0.5, m1=0.5, I2=0.5, m2=0.5,
            Omega=0, P=1, electronic_state=ElectronicState.X,
        )
        assert s.isUncoupled is True
        assert s.isCoupled is False
