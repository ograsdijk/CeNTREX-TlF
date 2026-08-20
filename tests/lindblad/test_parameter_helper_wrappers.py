from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import sympy as smp

import centrex_tlf.lindblad as lindblad
from centrex_tlf.lindblad import helper_functions, parameters
from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py, fill_hamiltonian_py
from centrex_tlf.lindblad.parameters import (
    LindbladParameters,
    Parameter,
    RuntimeExpression,
    Time,
)
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.utils_setup import OBESystem

rust = pytest.importorskip("centrex_tlf.centrex_tlf_rust")


NumericCase = tuple[str, tuple[Any, ...], dict[str, Any], str, tuple[Any, ...]]


NUMERIC_CASES: list[NumericCase] = [
    ("gaussian_1d", (0.2, 0.0, 0.5), {}, "gaussian_1d", (0.2, 0.0, 0.5)),
    (
        "gaussian_2d",
        (0.2, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7),
        {},
        "gaussian_2d",
        (0.2, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7),
    ),
    (
        "gaussian_2d",
        (0.2, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7),
        {"theta": 0.3},
        "gaussian_2d_rotated",
        (0.2, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7, 0.3),
    ),
    (
        "gaussian_2d_rotated",
        (0.2, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7, 0.3),
        {},
        "gaussian_2d_rotated",
        (0.2, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7, 0.3),
    ),
    ("phase_modulation", (0.2, 3.8, 1.2), {}, "phase_modulation", (0.2, 3.8, 1.2)),
    ("square_wave", (0.2, 1.2, 0.3), {}, "square_wave", (0.2, 1.2, 0.3)),
    (
        "resonant_polarization_modulation",
        (0.2, 0.4, 1.2),
        {},
        "resonant_polarization_modulation",
        (0.2, 0.4, 1.2),
    ),
    ("sawtooth_wave", (0.2, 1.2, 0.3), {}, "sawtooth_wave", (0.2, 1.2, 0.3)),
    ("variable_on_off", (0.2, 0.1, 0.3, 0.0), {}, "variable_on_off", (0.2, 0.1, 0.3, 0.0)),
    (
        "variable_on_off_duty",
        (0.2, 0.25, 2.5, 0.1),
        {},
        "variable_on_off_duty",
        (0.2, 0.25, 2.5, 0.1),
    ),
    (
        "variable_on_off_duty_invT",
        (0.2, 0.25, 2.5, 0.1),
        {},
        "variable_on_off_duty_invT",
        (0.2, 0.25, 2.5, 0.1),
    ),
    (
        "multipass_2d_intensity",
        (0.1, -0.2, (1.0, 0.5), (-0.3, 0.4), (0.25, -0.1), 0.7, 0.9),
        {},
        "multipass_2d_intensity",
        (0.1, -0.2, (1.0, 0.5), (-0.3, 0.4), (0.25, -0.1), 0.7, 0.9),
    ),
    ("rabi_from_intensity", (2.0, 0.4), {}, "rabi_from_intensity", (2.0, 0.4)),
    (
        "multipass_2d_rabi",
        (0.1, -0.2, (1.0, 0.5), (-0.3, 0.4), (0.25, -0.1), 0.7, 0.9, 0.35),
        {},
        "multipass_2d_rabi",
        (0.1, -0.2, (1.0, 0.5), (-0.3, 0.4), (0.25, -0.1), 0.7, 0.9, 0.35),
    ),
    (
        "gaussian_beam_rabi",
        (0.1, -0.2, 2.0, -0.3, 0.4, 0.7, 0.9, 0.35),
        {},
        "gaussian_beam_rabi",
        (0.1, -0.2, 2.0, -0.3, 0.4, 0.7, 0.9, 0.35),
    ),
    ("alternating_sign", (1.2, 0.0, 0.5), {}, "alternating_sign", (1.2, 0.0, 0.5)),
    ("linear_interp", (0.25, (0.0, 1.0), (2.0, 4.0)), {}, "linear_interp", (0.25, (0.0, 1.0), (2.0, 4.0))),
    (
        "pchip_interp",
        (0.25, (0.0, 0.5, 1.0), (2.0, 3.0, 4.0)),
        {},
        "pchip_interp",
        (0.25, (0.0, 0.5, 1.0), (2.0, 3.0, 4.0)),
    ),
]


@pytest.mark.parametrize(("name", "args", "kwargs", "helper_name", "helper_args"), NUMERIC_CASES)
def test_polymorphic_helpers_match_numeric_helpers(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helper_name: str,
    helper_args: tuple[Any, ...],
) -> None:
    wrapper = getattr(parameters, name)
    helper = getattr(helper_functions, helper_name)

    np.testing.assert_allclose(wrapper(*args, **kwargs), helper(*helper_args))
    np.testing.assert_allclose(getattr(lindblad, name)(*args, **kwargs), helper(*helper_args))


# ---------------------------------------------------------------------------
# Waveform conventions vs the Julia extension.
#
# The rest of this module compares the Python wrappers against the Python
# helpers, and the Rust lowering against the Python plan. That is mutual
# agreement, not correctness: `sawtooth_wave` was once offset by half a period
# in BOTH backends and every test still passed. These tests pin the waveforms
# against the Julia definitions they are ports of, transcribed from
# Waveforms.jl and CeNTREX-TlF-julia-extension/.../julia_common.jl.
# ---------------------------------------------------------------------------


def _julia_sawtoothwave(x: float) -> float:
    """Waveforms.jl: ``sawtoothwave(x) = rem2pi(x, RoundNearest) / pi``.

    ``rem2pi(_, RoundNearest)`` is the zero-centred remainder, range (-pi, pi],
    which Python's ``round`` (banker's rounding, like Julia's) reproduces.
    """
    return (x - 2.0 * math.pi * round(x / (2.0 * math.pi))) / math.pi


def _julia_sawtooth_wave(t: float, omega: float, phase: float) -> float:
    """julia_common.jl: ``0.5*(1 + sawtoothwave(omega*t + phase - pi))``."""
    return 0.5 * (1.0 + _julia_sawtoothwave(omega * t + phase - math.pi))


def _julia_square_wave(t: float, omega: float, phase: float) -> float:
    """julia_common.jl: ``0.5*(1 + squarewave(omega*t + phase))``.

    Waveforms.jl: ``squarewave(x) = ifelse(mod2pi(x) < pi, 1.0, -1.0)``.
    """
    return 1.0 if (omega * t + phase) % (2.0 * math.pi) < math.pi else 0.0


# Spans many periods in both directions, so the floor-based `% 1.0` used in the
# port is exercised against Julia's round-to-nearest `rem2pi` across wraps and
# for negative arguments -- not just inside the first period.
_WAVEFORM_TIMES = [i * 0.137 for i in range(-200, 201)]
_WAVEFORM_OMEGA = 1.2
_WAVEFORM_PHASE = 0.3


def test_sawtooth_wave_matches_julia_reference() -> None:
    for t in _WAVEFORM_TIMES:
        expected = _julia_sawtooth_wave(t, _WAVEFORM_OMEGA, _WAVEFORM_PHASE)
        actual = helper_functions.sawtooth_wave(t, _WAVEFORM_OMEGA, _WAVEFORM_PHASE)
        assert actual == pytest.approx(expected, abs=1e-12), f"t={t}"


def test_square_wave_matches_julia_reference() -> None:
    """Agreement away from the switching points.

    Julia switches on ``mod2pi(x) < pi``; the port switches on
    ``sin(x) >= 0``. These agree everywhere except exactly at ``mod2pi(x)`` of
    0 or pi, where the choice is arbitrary and floating-point ``sin`` is
    unreliable besides. Points within ``_SWITCH_GUARD`` of a switch are skipped
    rather than asserted on -- that boundary difference is a known, accepted
    convention difference, not a defect.
    """
    switch_guard = 1e-9
    checked = 0
    for t in _WAVEFORM_TIMES:
        x = (_WAVEFORM_OMEGA * t + _WAVEFORM_PHASE) % (2.0 * math.pi)
        if min(x, abs(x - math.pi), abs(x - 2.0 * math.pi)) < switch_guard:
            continue
        expected = _julia_square_wave(t, _WAVEFORM_OMEGA, _WAVEFORM_PHASE)
        actual = helper_functions.square_wave(t, _WAVEFORM_OMEGA, _WAVEFORM_PHASE)
        assert actual == pytest.approx(expected, abs=1e-12), f"t={t}"
        checked += 1
    assert checked > 0.9 * len(_WAVEFORM_TIMES)


@pytest.mark.parametrize(
    ("wrapper_name", "reference"),
    [
        ("sawtooth_wave", _julia_sawtooth_wave),
        ("square_wave", _julia_square_wave),
    ],
)
def test_waveforms_match_julia_through_runtime_expression(
    wrapper_name: str, reference: Callable[[float, float, float], float]
) -> None:
    """The evaluated RuntimeExpression path must agree with Julia too.

    Guards against the two backends drifting back into agreeing with each
    other while both disagree with the reference.
    """
    wrapper = getattr(parameters, wrapper_name)
    expression = wrapper(Time(), _WAVEFORM_OMEGA, _WAVEFORM_PHASE)
    for t in _WAVEFORM_TIMES[::7]:
        x = (_WAVEFORM_OMEGA * t + _WAVEFORM_PHASE) % (2.0 * math.pi)
        if min(x, abs(x - math.pi), abs(x - 2.0 * math.pi)) < 1e-9:
            continue
        expected = reference(t, _WAVEFORM_OMEGA, _WAVEFORM_PHASE)
        actual = complex(expression.evaluate(t=t)).real
        assert actual == pytest.approx(expected, abs=1e-12), f"t={t}"


def _julia_variable_on_off_duty_invT(
    t: float, duty: float, inv_period: float, phase: float
) -> float:
    """julia_common.jl: `mod1(t*invT + phase*INV_2PI, 1.0)`, then `frac < duty`.

    `mod1(x, 1.0)` returns the half-open range (0, 1], so an exact 0 becomes
    1.0. The Python and Rust ports spell that as `% 1.0` plus a `<= 0.0`
    correction.
    """
    frac = (t * inv_period + phase / (2.0 * math.pi)) % 1.0
    if frac <= 0.0:
        frac += 1.0
    return 1.0 if frac < duty else 0.0


_DUTY_CASES = [(0.25, 0.4, 0.0), (0.5, 1.3, 0.7), (0.9, 0.31, -1.1)]


def test_variable_on_off_duty_matches_julia_reference() -> None:
    for duty, inv_period, phase in _DUTY_CASES:
        for t in _WAVEFORM_TIMES:
            expected = _julia_variable_on_off_duty_invT(t, duty, inv_period, phase)
            assert helper_functions.variable_on_off_duty(t, duty, inv_period, phase) == expected


def test_variable_on_off_duty_invT_is_an_alias_not_a_second_implementation() -> None:
    """`_invT` exists only for Julia name parity; it must not drift.

    The Julia backend spells this gate `variable_on_off_duty_invT`
    (`julia_common.jl`), Python and Rust spell it `variable_on_off_duty`.
    Both names are exported so an expression written against either
    vocabulary lowers unchanged.
    """
    assert (
        helper_functions.HELPER_FUNCTION_IDS["variable_on_off_duty_invT"]
        is helper_functions.HELPER_FUNCTION_IDS["variable_on_off_duty"]
    )
    for duty, inv_period, phase in _DUTY_CASES:
        for t in _WAVEFORM_TIMES[::3]:
            assert helper_functions.variable_on_off_duty_invT(
                t, duty, inv_period, phase
            ) == helper_functions.variable_on_off_duty(t, duty, inv_period, phase)


def test_helper_function_names_resolves_aliases_to_the_canonical_name() -> None:
    """A plain inversion of HELPER_FUNCTION_IDS let the alias claim the id.

    Both names map to the same HelperFunctionId, so inverting the dict made
    whichever was declared last win -- previously `variable_on_off_duty_invT`,
    and silently swappable by reordering. HELPER_FUNCTION_ALIASES now makes
    the canonical choice explicit.
    """
    ids = helper_functions.HELPER_FUNCTION_IDS
    names = helper_functions.HELPER_FUNCTION_NAMES
    aliases = helper_functions.HELPER_FUNCTION_ALIASES

    assert names[int(ids["variable_on_off_duty_invT"])] == "variable_on_off_duty"

    for alias, canonical in aliases.items():
        assert ids[alias] is ids[canonical]
        assert canonical not in aliases, f"{canonical} is itself an alias"
        assert names[int(ids[alias])] == canonical

    # Every non-alias name must still round-trip, i.e. aliases are the only
    # id collisions.
    non_aliases = [name for name in ids if name not in aliases]
    assert len(names) == len(non_aliases)
    for name in non_aliases:
        assert names[int(ids[name])] == name


def test_variable_on_off_duty_alias_lowers_identically_through_rust() -> None:
    duty, inv_period, phase = 0.25, 1.3, 0.7
    canonical = parameters.variable_on_off_duty(Time(), duty, inv_period, phase)
    alias = parameters.variable_on_off_duty_invT(Time(), duty, inv_period, phase)
    for t in _WAVEFORM_TIMES[::7]:
        expected = _julia_variable_on_off_duty_invT(t, duty, inv_period, phase)
        assert complex(canonical.evaluate(t=t)).real == pytest.approx(expected, abs=1e-12)
        assert complex(alias.evaluate(t=t)).real == pytest.approx(expected, abs=1e-12)


def _function_name(expression: RuntimeExpression) -> str:
    return getattr(expression.expr.func, "__name__", str(expression.expr.func))


def test_public_exports_use_polymorphic_wrappers() -> None:
    assert lindblad.phase_modulation is parameters.phase_modulation
    assert lindblad.resonant_polarization_modulation is parameters.resonant_polarization_modulation
    assert lindblad.gaussian_2d is parameters.gaussian_2d
    assert helper_functions.phase_modulation is not parameters.phase_modulation


@pytest.mark.parametrize(
    ("factory", "expected_name"),
    [
        (lambda t: parameters.gaussian_1d(t, 0.0, 0.5), "gaussian_1d"),
        (lambda t: parameters.gaussian_2d(t, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7), "gaussian_2d"),
        (
            lambda t: parameters.gaussian_2d(t, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7, theta=0.3),
            "gaussian_2d_rotated",
        ),
        (
            lambda t: parameters.gaussian_2d_rotated(t, -0.1, 1.3, 0.0, 0.1, 0.5, 0.7, 0.3),
            "gaussian_2d_rotated",
        ),
        (lambda t: parameters.phase_modulation(t, 3.8, 1.2), "phase_modulation"),
        (lambda t: parameters.square_wave(t, 1.2, 0.3), "square_wave"),
        (
            lambda t: parameters.resonant_polarization_modulation(t, 0.4, 1.2),
            "resonant_polarization_modulation",
        ),
        (lambda t: parameters.sawtooth_wave(t, 1.2, 0.3), "sawtooth_wave"),
        (lambda t: parameters.variable_on_off(t, 0.1, 0.3, 0.0), "variable_on_off"),
        (
            lambda t: parameters.variable_on_off_duty(t, 0.25, 2.5, 0.1),
            "variable_on_off_duty",
        ),
        (
            lambda t: parameters.variable_on_off_duty_invT(t, 0.25, 2.5, 0.1),
            "variable_on_off_duty_invT",
        ),
        (lambda t: parameters.rabi_from_intensity(t, 0.4), "rabi_from_intensity"),
        (
            lambda t: parameters.gaussian_beam_rabi(t, -0.2, 2.0, -0.3, 0.4, 0.7, 0.9, 0.35),
            "gaussian_beam_rabi",
        ),
        (lambda t: parameters.alternating_sign(t, 0.0, 0.5), "alternating_sign"),
    ],
)
def test_scalar_helpers_build_runtime_expressions(
    factory: Callable[[RuntimeExpression], RuntimeExpression],
    expected_name: str,
) -> None:
    expression = factory(Time())
    assert isinstance(expression, RuntimeExpression)
    assert _function_name(expression) == expected_name


def test_tuple_helpers_build_runtime_expressions() -> None:
    x = Time()
    amplitudes = Parameter("amplitudes", (1.0, 0.5))
    xlocs = Parameter("xlocs", (-0.3, 0.4))
    ylocs = Parameter("ylocs", (0.25, -0.1))
    grid = Parameter("grid", (0.0, 0.5, 1.0))
    values = Parameter("values", (2.0, 3.0, 4.0))

    expressions = [
        (parameters.multipass_2d_intensity(x, -0.2, amplitudes, xlocs, ylocs, 0.7, 0.9), "multipass_2d_intensity"),
        (parameters.multipass_2d_rabi(x, -0.2, amplitudes, xlocs, ylocs, 0.7, 0.9, 0.35), "multipass_2d_rabi"),
        (parameters.linear_interp(x, grid, values), "linear_interp"),
        (parameters.pchip_interp(x, grid, values), "pchip_interp"),
    ]
    for expression, expected_name in expressions:
        assert isinstance(expression, RuntimeExpression)
        assert _function_name(expression) == expected_name


def _two_level_system() -> OBESystem:
    omega, delta = smp.symbols("Ω δ", real=True)
    hamiltonian = smp.Matrix(
        [
            [0, omega / 2],
            [smp.conjugate(omega) / 2, -delta],
        ]
    )
    zeros = np.zeros((2, 2), dtype=np.complex128)
    c_array = np.zeros((0, 2, 2), dtype=np.complex128)
    return OBESystem(
        ground=[],
        excited=[],
        QN=[],
        H_int=zeros,
        V_ref_int=zeros,
        couplings=[],
        H_symbolic=hamiltonian,
        C_array=c_array,
        system=None,
        coupling_symbols=[omega, delta],
        polarization_symbols=[],
    )


@pytest.mark.parametrize(
    "expression",
    [
        lambda p: parameters.phase_modulation(p.time(), p.real("beta", 0.4), p.real("omega_m", 1.2)),
        lambda p: parameters.resonant_polarization_modulation(p.time(), p.real("gamma", 0.3), p.real("omega_m", 1.2)),
        lambda p: parameters.gaussian_2d(
            p.real("x", 0.1),
            p.real("y", -0.2),
            p.real("amp", 1.0),
            p.real("x0", -0.1),
            p.real("y0", 0.2),
            p.real("sigma_x", 0.7),
            p.real("sigma_y", 0.9),
        ),
        lambda p: parameters.gaussian_2d(
            p.real("x", 0.1),
            p.real("y", -0.2),
            p.real("amp", 1.0),
            p.real("x0", -0.1),
            p.real("y0", 0.2),
            p.real("sigma_x", 0.7),
            p.real("sigma_y", 0.9),
            theta=p.real("theta", 0.3),
        ),
        lambda p: parameters.multipass_2d_rabi(
            p.real("x", 0.1),
            p.real("y", -0.2),
            p.real("amps", (1.0, 0.5)),
            p.real("xlocs", (-0.3, 0.4)),
            p.real("ylocs", (0.25, -0.1)),
            p.real("sigma_x", 0.7),
            p.real("sigma_y", 0.9),
            p.real("coupling", 0.35),
        ),
        lambda p: parameters.linear_interp(
            p.real("x", 0.25),
            p.real("grid", (0.0, 0.5, 1.0)),
            p.real("values", (2.0, 3.0, 4.0)),
        ),
        lambda p: parameters.pchip_interp(
            p.real("x", 0.25),
            p.real("grid", (0.0, 0.5, 1.0)),
            p.real("values", (2.0, 3.0, 4.0)),
        ),
    ],
)
def test_representative_polymorphic_helpers_lower_to_rust(expression: Any) -> None:
    system = _two_level_system()
    omega_symbol, delta_symbol = system.coupling_symbols
    params = LindbladParameters()
    params.bind(omega_symbol, expression(params), finalize=False)
    params.bind(delta_symbol, 0.0)

    prepared = prepare_lindblad_problem(system, params, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    time = 0.37
    slots = evaluate_parameter_graph_py(prepared.parameter_graph, time)
    h_python = fill_hamiltonian_py(prepared.hamiltonian_plan, slots, time)
    h_rust = np.asarray(
        rust.evaluate_lindblad_hamiltonian_py(rust_plan, time),
        dtype=np.complex128,
    )
    np.testing.assert_allclose(h_rust, h_python)
