import pytest

from centrex_tlf import states, transitions
from centrex_tlf.transitions import (
    MicrowaveTransition,
    OpticalTransition,
    OpticalTransitionType,
)


def test_optical_transition_type_values():
    assert OpticalTransitionType.P == -1
    assert OpticalTransitionType.Q == 0
    assert OpticalTransitionType.R == 1


def test_optical_transition_r0():
    t = OpticalTransition(
        t=OpticalTransitionType.R,
        J_ground=0,
        F1_excited=0.5,
        F_excited=1,
    )
    assert t.J_excited == 1
    assert t.name == "R(0) F1'=1/2 F'=1"
    assert t.Ω_ground == 0
    assert t.Ω_excited == 1
    assert t.P_ground == 1
    assert t.P_excited == -1


def test_optical_transition_ground_excited_states():
    t = transitions.R0_F1_1o2_F1
    gs = t.ground_states
    es = t.excited_states
    assert len(gs) > 0
    assert len(es) > 0
    for s in gs:
        assert isinstance(s, states.CoupledBasisState)
    for s in es:
        assert isinstance(s, states.CoupledBasisState)


def test_optical_transition_invalid_j():
    with pytest.raises(ValueError):
        OpticalTransition(
            t=OpticalTransitionType.R,
            J_ground=-1,
            F1_excited=0.5,
            F_excited=1,
        )


def test_microwave_transition():
    t = MicrowaveTransition(J_ground=0, J_excited=1)
    assert t.name == "J=0→J=1"
    assert t.Ω_ground == 0
    assert t.Ω_excited == 0
    assert t.P_ground == 1
    assert t.P_excited == -1


def test_microwave_transition_invalid_j():
    with pytest.raises(ValueError):
        MicrowaveTransition(J_ground=-1, J_excited=1)


def test_microwave_transition_selectors():
    t = MicrowaveTransition(J_ground=0, J_excited=1)
    gs = t.qn_select_ground
    es = t.qn_select_excited
    assert gs.J == 0
    assert es.J == 1


def test_predefined_transitions_exist():
    assert hasattr(transitions, "R0_F1_1o2_F0")
    assert hasattr(transitions, "R0_F1_1o2_F1")
    assert hasattr(transitions, "Q1_F1_1o2_F0")
    assert isinstance(transitions.R0_F1_1o2_F1, OpticalTransition)
