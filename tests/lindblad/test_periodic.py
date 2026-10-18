"""Periodic-map comparisons against direct segmented complex evolution."""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import sympy as smp
from scipy.integrate import solve_ivp

from centrex_tlf import lindblad
from centrex_tlf.lindblad import CudaSolverOptions
from centrex_tlf.lindblad.periodic import _cycles


def model(kind="periodic"):
    t = smp.Symbol("t", real=True)
    r = smp.Symbol("r")
    H = smp.Matrix([[0, r, 0], [smp.conjugate(r), -0.7, 0], [0, 0, 0]])
    C = np.zeros((2, 3, 3), complex)
    C[0, 0, 1], C[1, 2, 1] = np.sqrt(0.3), np.sqrt(0.2)
    params = lindblad.LindbladParameters()
    amplitude = params.real("amplitude", 1.2)
    frequency = params.real("frequency", 2 * np.pi)
    drive = (
        amplitude
        * lindblad.phase_modulation(t, 0.7, frequency)
        * lindblad.square_wave(t, frequency, 0.23)
    )
    if kind == "gaussian":
        drive *= lindblad.gaussian(t, 0.2, 10.0)
    if kind == "ramp":
        drive *= 1 + t
    if kind == "disabled":
        drive = 0 * drive * lindblad.gaussian(t, 0.2, 10.0)
    params.bind(r, drive)
    prepared = lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=C), params)
    initial = np.diag([1.0, 0.0, 0.0])[None].astype(complex)
    return prepared, initial, C


@pytest.fixture
def gpu():
    try:
        return lindblad.cuda._cupy()
    except (RuntimeError, ImportError) as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("kind", ["gaussian", "ramp"])
def test_nonperiodic_envelope_rejected_before_gpu(kind, monkeypatch):
    p, initial, _ = model(kind)
    monkeypatch.setattr(
        "centrex_tlf.lindblad.periodic._cupy", lambda: pytest.fail("GPU must not load")
    )
    with pytest.raises(ValueError, match="periodicity"):
        lindblad.prepare_periodic_lindblad_propagator(p, 1.0, initial, dt=0.01)


def test_bad_period_and_budget_rejected_without_gpu(monkeypatch):
    p, initial, _ = model()
    monkeypatch.setattr(
        "centrex_tlf.lindblad.periodic._cupy", lambda: pytest.fail("GPU must not load")
    )
    for period in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="period"):
            lindblad.prepare_periodic_lindblad_propagator(p, period, initial, dt=0.01)
    with pytest.raises(ValueError, match="periodicity"):
        lindblad.prepare_periodic_lindblad_propagator(p, 0.91, initial, dt=0.01)
    with pytest.raises(ValueError, match="bytes"):
        lindblad.prepare_periodic_lindblad_propagator(p, 1.0, initial, dt=0.01, max_map_bytes=1)
    with pytest.raises(ValueError, match="periodicity"):
        lindblad.prepare_periodic_lindblad_propagator(
            p, 1.0, initial, dt=0.01, parameter_values={"frequency": 2 * np.pi * 1.01}
        )
    with pytest.raises(ValueError, match="step_schedule"):
        lindblad.prepare_periodic_lindblad_propagator(
            p, 1.0, initial, dt=0.01, cuda_options=CudaSolverOptions(step_schedule=[(0, 0.001)])
        )


def independent(C, initial, t0, target, amplitude=1.2):
    def rhs(t, y):
        drive = (
            amplitude
            * np.exp(1j * 0.7 * np.sin(2 * np.pi * t))
            * (np.sin(2 * np.pi * t + 0.23) >= 0)
        )
        H = np.array([[0, drive, 0], [drive.conjugate(), -0.7, 0], [0, 0, 0]], complex)
        rho = y.reshape(3, 3)
        value = -1j * (H @ rho - rho @ H)
        for jump in C:
            D = jump.conj().T @ jump
            value += jump @ rho @ jump.conj().T - 0.5 * (D @ rho + rho @ D)
        return value.ravel()

    points = [
        (k * np.pi - 0.23) / (2 * np.pi)
        for k in range(int(np.floor(2 * t0)) - 2, int(np.ceil(2 * target)) + 3)
    ]
    points = [t0, *[x for x in points if t0 < x < target], target]
    y = initial.ravel()
    for a, b in pairwise(points):
        if b > a:
            # Avoid the right-hand gate value at a boundary derivative.
            def interior(t, y, a=a, b=b):
                return rhs(min(max(t, a + 1e-12), b - 1e-12), y)

            y = solve_ivp(interior, (a, b), y, method="DOP853", rtol=2e-12, atol=2e-14).y[:, -1]
    return y.reshape(3, 3)


@pytest.mark.parametrize("solver", ["fixed_tsit5", "fixed_ck54"])
def test_periodic_map_initial_states_velocities_entry_phases_and_remainders(gpu, solver):
    p, initial, C = model()
    prop = lindblad.prepare_periodic_lindblad_propagator(
        p,
        1.0,
        initial,
        dt=0.004,
        solver=solver,
        phase_origin=0.17,
        parameter_values={"amplitude": 0.9},
        max_map_bytes=400,
        cuda_options=CudaSolverOptions(batch_size=2, density_reduction=True),
    )
    assert prop.stats["retained_coordinates"] == 5
    assert prop.stats["map_bytes"] == 200
    states = np.repeat(initial, 2, axis=0)
    states[1] = np.array([[0.5, 0.1 + 0.2j, 0], [0.1 - 0.2j, 0.5, 0], [0, 0, 0]])
    for start in (0.17, 0.41, -0.2):
        times = start + np.array([2.31, 0.0, 0.1, 1.0, 0.83, 2.31, 8.0])
        got = prop.density_matrices(states, times, t0=start)
        expected = np.array([[independent(C, rho, start, t, 0.9) for t in times] for rho in states])
        np.testing.assert_allclose(got, expected, atol=3e-9)
        np.testing.assert_allclose(np.trace(got, axis1=-2, axis2=-1), 1.0, atol=2e-12)
        np.testing.assert_allclose(
            prop.evaluate(states, times, t0=start, output="populations"),
            np.diagonal(got, axis1=-2, axis2=-1).real,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            prop.evaluate(
                states, times, t0=start, output="selected", output_indices=[(1, 0), (2, 0)]
            ),
            got[..., [1, 2], [0, 0]],
            atol=1e-13,
        )
    assert len(prop._powers) <= 2
    prop.clear_cuda_cache()
    assert not prop._powers
    np.testing.assert_allclose(
        prop.density_matrices(states, [1.17]),
        [[independent(C, rho, 0.17, 1.17, 0.9)] for rho in states],
        atol=3e-9,
    )
    # Any new support outside the closure is rejected, rather than discarded.
    states[0, 0, 2] = 0.02j
    states[0, 2, 0] = -0.02j
    with pytest.raises(ValueError, match="outside"):
        prop.evaluate(states, [1.17])
    p.parameter_graph["base_values"][0]["re"] += 0.1
    with pytest.raises(ValueError, match="changed"):
        prop.evaluate(initial, [1.17])


def test_unreduced_static_map_and_empty_times(gpu):
    H = smp.Matrix([[0, 0.7], [0.7, 1.0]])
    p = lindblad.prepare_lindblad_problem(
        SimpleNamespace(H_symbolic=H, C_array=np.zeros((0, 2, 2), complex)), {}
    )
    prop = lindblad.prepare_periodic_lindblad_propagator(p, 0.3, dt=0.003, reduce=False)
    rho = np.array([[[0.4, 0.1j], [-0.1j, 0.6]]])
    assert prop.evaluate(rho, []).shape == (1, 0, 4)
    t = 0.93
    U = la.expm(-1j * np.array(H, complex) * t)
    np.testing.assert_allclose(
        prop.density_matrices(rho, [t])[0, 0], U @ rho[0] @ U.conj().T, atol=2e-11
    )
    with pytest.raises(ValueError, match="at or after"):
        prop.evaluate(rho, [-1.0])


def test_cycle_roundoff_and_limits():
    assert _cycles(0.3, 0.1) == (3, 0.0)
    assert _cycles(0.0, 0.1) == (0, 0.0)
    with pytest.raises(ValueError, match="too many"):
        _cycles(1e30, 0.1)
