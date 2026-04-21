import numpy as np
import pytest

from centrex_tlf.couplings.polarization import (
    Polarization,
    polarization_X,
    polarization_Y,
    polarization_Z,
    polarization_σm,
    polarization_σp,
)


def test_polarization_creation():
    p = Polarization(vector=np.array([1, 0, 0], dtype=complex), name="test")
    assert p.name == "test"
    assert p.norm == pytest.approx(1.0)


def test_polarization_invalid_shape():
    with pytest.raises(ValueError):
        Polarization(vector=np.array([1, 0], dtype=complex), name="bad")


def test_polarization_is_normalized():
    assert polarization_X.is_normalized
    assert polarization_Y.is_normalized
    assert polarization_Z.is_normalized
    assert polarization_σp.is_normalized
    assert polarization_σm.is_normalized


def test_polarization_mul():
    p = polarization_X * 2
    assert p.norm == pytest.approx(2.0)


def test_polarization_rmul():
    p = 0.5 * polarization_Z
    assert p.norm == pytest.approx(0.5)


def test_polarization_add():
    p = polarization_X + polarization_Y
    assert p.norm == pytest.approx(np.sqrt(2))


def test_polarization_sub():
    p = polarization_X - polarization_X
    assert p.norm == pytest.approx(0.0)


def test_polarization_neg():
    p = -polarization_Z
    np.testing.assert_allclose(p.vector, -polarization_Z.vector)


def test_polarization_div():
    p = polarization_X / 2.0
    assert p.norm == pytest.approx(0.5)


def test_polarization_div_zero():
    with pytest.raises(ZeroDivisionError):
        polarization_X / 0.0


def test_polarization_normalize():
    p = Polarization(vector=np.array([3, 0, 0], dtype=complex), name="test")
    pn = p.normalize()
    assert pn.is_normalized
    assert pn.norm == pytest.approx(1.0)


def test_polarization_normalize_zero():
    p = Polarization(vector=np.array([0, 0, 0], dtype=complex), name="zero")
    with pytest.raises(ValueError):
        p.normalize()


def test_sigma_plus_minus_orthogonal():
    dot = np.vdot(polarization_σp.vector, polarization_σm.vector)
    assert abs(dot) < 1e-12
