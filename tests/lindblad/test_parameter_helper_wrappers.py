from __future__ import annotations

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
