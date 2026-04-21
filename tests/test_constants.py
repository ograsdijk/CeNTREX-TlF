import numpy as np

from centrex_tlf.constants import BConstants, TlFNuclearSpins, XConstants


def test_xconstants_defaults():
    x = XConstants()
    assert x.B_rot > 0
    assert x.c1 == 126030.0
    assert x.D_TlF > 0


def test_bconstants_defaults():
    b = BConstants()
    assert b.B_rot > 0
    assert b.Γ > 0
    assert b.gL == 1
    assert b.gS == 2


def test_nuclear_spins_defaults():
    ns = TlFNuclearSpins()
    assert ns.I_F == 0.5
    assert ns.I_Tl == 0.5


def test_xconstants_custom():
    x = XConstants(B_rot=1e9)
    assert x.B_rot == 1e9
    assert x.c1 == 126030.0


def test_bconstants_custom():
    b = BConstants(B_rot=1e9)
    assert b.B_rot == 1e9
    assert b.Γ == BConstants().Γ


def test_nuclear_spins_custom():
    ns = TlFNuclearSpins(I_F=1.5, I_Tl=0.5)
    assert ns.I_F == 1.5


def test_xconstants_hashable():
    x1 = XConstants()
    x2 = XConstants()
    assert hash(x1) == hash(x2)
    assert {x1, x2} == {x1}


def test_bconstants_hashable():
    b1 = BConstants()
    b2 = BConstants()
    assert hash(b1) == hash(b2)
