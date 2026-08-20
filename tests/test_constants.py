import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.constants as cst

from centrex_tlf.constants import (
    B0_X,
    B_LIFETIME,
    D0_X,
    MU_X_E_D,
    MU_X_V0_D,
    MU_X_V1_D,
    MU_X_V2_D,
    Y01_X,
    Y02_X,
    Y11_X,
    Y21_X,
    Γ,
    BConstants,
    ED_XtB,
    HamiltonianConstants,
    TlFNuclearSpins,
    XConstants,
)
from centrex_tlf.constants.constants import (
    Bohr_magneton_Hz_G,
    Debye,
    Debye_Hz_V_cm,
    a0,
)

RUST_SRC = Path(__file__).resolve().parents[1] / "rust" / "src"


def test_xconstants_defaults():
    x = XConstants()
    assert x.B_rot > 0
    assert x.c1 == 126030.0
    assert x.D_TlF > 0


def test_x_dunham_coefficients():
    assert Y01_X == 6689.8736e6
    assert Y11_X == -45.0843e6
    assert Y21_X == 0.0942e6
    assert Y02_X == -5.84e3


def test_x_b0_derived_from_dunham_coefficients():
    assert B0_X == Y01_X + Y11_X / 2 + Y21_X / 4
    assert np.isclose(B0_X, 6.667355000e9)


def test_x_d0_derived_from_dunham_coefficients():
    assert D0_X == -Y02_X
    assert np.isclose(D0_X, 5.84e3)


def test_xconstants_rotational_defaults_match_derived_values():
    x = XConstants()
    assert x.B_rot == B0_X
    assert x.D_rot == D0_X


def test_x_rotational_energy_levels():
    x = XConstants()
    for J in (0, 1, 2, 3):
        JJ1 = J * (J + 1)
        expected = B0_X * JJ1 - D0_X * JJ1**2
        E_J = x.B_rot * JJ1 - x.D_rot * JJ1**2
        assert expected == E_J


def test_x_rotational_energy_differs_from_old_rigid_rotor_model():
    # Old model: B_rot = B_epsilon - alpha_e/2, no centrifugal distortion.
    old_B_rot = 6.689873e9 - 45.0843e6 / 2
    for J in (1, 2, 3):
        JJ1 = J * (J + 1)
        old_E_J = old_B_rot * JJ1
        new_E_J = B0_X * JJ1 - D0_X * JJ1**2
        # The correction is dominated by the ~24.15 kHz shift in B0 (times
        # J(J+1)) with a small additional centrifugal-distortion term.
        expected_correction = (B0_X - old_B_rot) * JJ1 - D0_X * JJ1**2
        assert np.isclose(new_E_J - old_E_J, expected_correction)
        assert new_E_J != old_E_J


def test_x_vibrational_dipole_moments():
    assert MU_X_E_D == 4.1939
    assert MU_X_V0_D == 4.2282
    assert MU_X_V1_D == 4.2972
    assert MU_X_V2_D == 4.3665


def test_xconstants_dipole_matches_v0():
    x = XConstants()
    assert MU_X_V0_D * Debye == x.D
    assert x.D_TlF == MU_X_V0_D * Debye_Hz_V_cm


def test_hamiltonian_constants_declares_d_rot_explicitly():
    field_names = [f.name for f in dataclasses.fields(HamiltonianConstants)]
    assert field_names == ["B_rot", "D_rot"]
    # Both fields are required (no default) on the common base class, so
    # D_rot is part of the interface rather than an optional/duck-typed
    # attribute that Hrot would need to getattr() defensively.
    for field in dataclasses.fields(HamiltonianConstants):
        assert field.default is dataclasses.MISSING


def test_xconstants_and_bconstants_expose_d_rot_first_two_fields():
    # Both subclasses keep B_rot/D_rot as the first two fields, inherited
    # in that order from HamiltonianConstants.
    x_fields = [f.name for f in dataclasses.fields(XConstants)]
    b_fields = [f.name for f in dataclasses.fields(BConstants)]
    assert x_fields[:2] == ["B_rot", "D_rot"]
    assert b_fields[:2] == ["B_rot", "D_rot"]


def test_general_uncoupled_hrot_uses_common_d_rot_field():
    from centrex_tlf.hamiltonian.general_uncoupled import Hrot
    from centrex_tlf.states import ElectronicState, UncoupledBasisState

    psi = UncoupledBasisState(
        J=2,
        mJ=0,
        I1=0.5,
        m1=0.5,
        I2=0.5,
        m2=0.5,
        Omega=0,
        P=1,
        electronic_state=ElectronicState.X,
    )
    x = XConstants()
    result = Hrot(psi, x)
    JJ1 = 2 * 3
    expected = x.B_rot * JJ1 - x.D_rot * JJ1**2
    (amp, state) = result.data[0]
    assert state == psi
    assert np.isclose(amp, expected)


def test_bconstants_defaults():
    b = BConstants()
    assert b.B_rot > 0
    assert b.Γ > 0
    assert b.gL == 1
    assert b.gS == 2


def test_b_lifetime_value():
    assert B_LIFETIME == 99e-9


def test_bconstants_gamma_derived_from_lifetime():
    assert BConstants().Γ == 1 / B_LIFETIME


def test_module_level_gamma_matches_bconstants():
    assert Γ == BConstants().Γ


def test_bconstants_gamma_equivalent_linewidth():
    assert np.isclose(BConstants().Γ / (2 * np.pi), 1.6076e6, rtol=1e-4)


def test_bconstants_gamma_matches_rust_default():
    """Python and Rust must derive the same Γ from the same B_LIFETIME.

    Rust's `BConstants::default().gamma` is not exposed to Python (Rust-side
    Hamiltonian builders always take Γ from the Python `BConstants` instance
    passed in, via `parse_b_constants`), so there is no live binding to check
    at runtime. Both sides compute `1.0 / 99e-9` in f64/float64, so they are
    bit-identical by construction; `rust/src/constants.rs::tests` pins the
    same numeric checks on the Rust side (`test_bconstants_gamma_derived_from_lifetime`,
    `test_b_lifetime_value`, `test_bconstants_gamma_equivalent_linewidth`).
    """
    assert B_LIFETIME == 99e-9
    assert BConstants().Γ == 1.0 / 99e-9


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


def test_debye():
    assert Debye == 1e-21 / cst.c


def test_debye_hz_v_cm():
    assert Debye_Hz_V_cm == Debye * 100 / cst.h


def test_bconstants_bohr_magneton():
    b = BConstants()
    assert np.isclose(b.μ_B, cst.value("Bohr magneton in Hz/T") * 1e-4)


def test_ed_xtb_value():
    assert ED_XtB == 0.315 * cst.e * a0


def test_ed_xtb_in_debye():
    assert np.isclose(ED_XtB / Debye, 0.80065, atol=1e-4)


def _parse_rust_f64_consts(source: str) -> dict[str, float]:
    """Pull the plain `pub const NAME: f64 = <literal>;` values out of Rust source."""
    pattern = re.compile(
        r"^pub const (\w+): f64 = (-?[\d._]+(?:[eE][-+]?\d+)?);", re.MULTILINE
    )
    return {
        name: float(literal.replace("_", ""))
        for name, literal in pattern.findall(source)
    }


def test_rust_constants_match_scipy_derived_python():
    """The frozen Rust literals must equal the scipy-derived Python values.

    Python computes `Debye`, `Debye_Hz_V_cm`, `Bohr_magneton_Hz_G` and `ED_XtB`
    from `scipy.constants` at import time, while `rust/src/constants.rs` hard-codes
    the same numbers. They are bit-identical today, but nothing forces them to stay
    that way: a CODATA revision in a newer SciPy, or a user pinning a different
    SciPy, would silently give the Python and Rust backends different
    `XConstants`/`BConstants` defaults. This test makes that failure loud.

    `ED_XTB` is checked through its factors rather than as a literal, since Rust
    derives it as `0.315 * ELEMENTARY_CHARGE * BOHR_RADIUS` — the same
    multiplication order Python uses, so the f64 results agree exactly.
    """
    constants_rs = RUST_SRC / "constants.rs"
    if not constants_rs.is_file():
        pytest.skip("Rust source tree not available (installed wheel)")

    rust = _parse_rust_f64_consts(constants_rs.read_text(encoding="utf-8"))
    expected = {
        "DEBYE": Debye,
        "DEBYE_HZ_V_CM": Debye_Hz_V_cm,
        "BOHR_MAGNETON_HZ_G": Bohr_magneton_Hz_G,
        "ELEMENTARY_CHARGE": cst.e,
        "BOHR_RADIUS": a0,
    }
    for name, python_value in expected.items():
        assert name in rust, f"{name} missing from rust/src/constants.rs"
        assert rust[name] == python_value, (
            f"{name}: Rust has {rust[name]!r}, Python derives {python_value!r}. "
            "Update rust/src/constants.rs to match the installed scipy.constants."
        )

    assert 0.315 * rust["ELEMENTARY_CHARGE"] * rust["BOHR_RADIUS"] == ED_XtB


def test_rust_does_not_inline_the_transition_dipole():
    """`ED_XTB` must be referenced by name so the drift test above covers it."""
    eval_rs = RUST_SRC / "lindblad" / "eval.rs"
    if not eval_rs.is_file():
        pytest.skip("Rust source tree not available (installed wheel)")

    source = eval_rs.read_text(encoding="utf-8")
    assert repr(ED_XtB) not in source
    assert "ED_XTB" in source


def test_unit_conversions_are_star_exported():
    """`from centrex_tlf.constants import *` must still provide the conversions.

    `__all__` used to be misspelled `_all__`, so the star import exported every
    module-level name. Now that it is real, the unit-conversion factors have to be
    listed explicitly or downstream scripts break with a bare `NameError`.
    """
    import centrex_tlf.constants as constants_module

    for name in ("a0", "Debye", "Debye_Hz_V_cm", "Bohr_magneton_Hz_G"):
        assert name in constants_module.__all__
        assert hasattr(constants_module, name)


def test_eq_xtb_not_exported():
    import centrex_tlf.constants as constants_module

    assert "EQ_XtB" not in constants_module.__all__
    assert not hasattr(constants_module, "EQ_XtB")


def test_helper_functions_use_canonical_ed_xtb():
    import inspect

    from centrex_tlf.lindblad import helper_functions
    from centrex_tlf.utils import multipass, rabi

    dipole_arg_funcs = (
        (helper_functions.rabi_from_intensity, "dipole_moment"),
        (helper_functions.multipass_2d_rabi, "dipole_moment"),
        (helper_functions.gaussian_beam_rabi, "dipole_moment"),
        (rabi.power_to_rabi_rectangular_beam, "D"),
        (rabi.power_to_rabi_gaussian_beam, "D"),
        (rabi.rabi_to_power_gaussian_beam, "D"),
        (multipass.generate_2d_multipass_gaussian_rabi, "D"),
    )
    for func, param_name in dipole_arg_funcs:
        default = inspect.signature(func).parameters[param_name].default
        assert default == ED_XtB
