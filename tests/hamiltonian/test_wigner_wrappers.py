"""The cached Wigner wrappers: Rust below the cut-off, sympy above it or without Rust."""

import numpy as np
import pytest

from centrex_tlf.hamiltonian import wigner


def _package_like_symbols(J):
    """3j and 6j symbols of the shapes the package evaluates, around a given J."""
    threej, sixj = [], []
    for Jp in (J - 1, J, J + 1):
        if Jp < 0:
            continue
        for a, q, b in ((-1, 0, 1), (0, 0, 0), (0, 1, -1), (1, -1, 0)):
            if abs(a) <= J and abs(b) <= Jp:
                threej.append((J, 1, Jp, a, q, b))
        for F1 in (J - 0.5, J + 0.5):
            for F1p in (Jp - 0.5, Jp + 0.5):
                if F1 > 0 and F1p > 0:
                    sixj.append((Jp, F1p, 0.5, F1, J, 1))
    return threej, sixj


@pytest.mark.parametrize("J", [0, 1, 2, 5, 11, 20, 39])
def test_wrappers_agree_with_sympy_below_the_cutoff(J):
    threej, sixj = _package_like_symbols(J)
    for args in threej:
        value = wigner.threej_f.__wrapped__(*args)
        assert isinstance(value, complex)
        np.testing.assert_allclose(value, wigner._threej_sympy(*args), rtol=1e-13, atol=1e-15)
    for args in sixj:
        value = wigner.sixj_f.__wrapped__(*args)
        assert isinstance(value, complex)
        np.testing.assert_allclose(value, wigner._sixj_sympy(*args), rtol=1e-13, atol=1e-15)


def test_rust_is_used_below_the_cutoff_and_sympy_above(monkeypatch):
    pytest.importorskip("centrex_tlf.centrex_tlf_rust")
    calls = []
    real = wigner._wigner_3j_rust

    def counting(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(wigner, "_wigner_3j_rust", counting)
    J = wigner.RUST_MAX_J
    wigner.threej_f.__wrapped__(J, 1, J - 1, 0, 0, 0)
    assert len(calls) == 1
    above = wigner.threej_f.__wrapped__(J + 1, 1, J, 0, 0, 0)
    assert len(calls) == 1
    assert above == wigner._threej_sympy(J + 1, 1, J, 0, 0, 0)


def test_wrappers_fall_back_to_sympy_without_the_extension(monkeypatch):
    monkeypatch.setattr(wigner, "_wigner_3j_rust", None)
    monkeypatch.setattr(wigner, "_wigner_6j_rust", None)
    assert wigner.threej_f.__wrapped__(2, 1, 1, 0, 0, 0) == wigner._threej_sympy(2, 1, 1, 0, 0, 0)
    assert wigner.sixj_f.__wrapped__(1, 1.5, 0.5, 2.5, 2, 1) == wigner._sixj_sympy(
        1, 1.5, 0.5, 2.5, 2, 1
    )


def test_symbols_rust_refuses_are_evaluated_by_sympy(monkeypatch):
    pytest.importorskip("centrex_tlf.centrex_tlf_rust")
    # three large angular momenta: Rust raises instead of returning an inaccurate value
    monkeypatch.setattr(wigner, "RUST_MAX_J", 100)
    args = (60, 60, 60, 1, -1, 0)
    assert wigner.threej_f.__wrapped__(*args) == wigner._threej_sympy(*args)


def test_forbidden_symbols_are_exactly_zero():
    assert wigner.threej_f.__wrapped__(1, 1, 3, 0, 0, 0) == 0
    assert wigner.threej_f.__wrapped__(1, 1, 1, 1, 1, 0) == 0
    assert wigner.sixj_f.__wrapped__(1, 1, 3, 1, 1, 1) == 0


def test_numpy_scalar_arguments_are_accepted():
    value = wigner.threej_f.__wrapped__(np.int64(2), 1, np.float64(1.0), 0, 0, 0)
    np.testing.assert_allclose(value, wigner._threej_sympy(2, 1, 1, 0, 0, 0), rtol=1e-13)
