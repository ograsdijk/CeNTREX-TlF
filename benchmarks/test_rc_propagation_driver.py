from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_rc_propagation as driver


def test_two_switches_have_exact_quarter_period_offset() -> None:
    edges = driver.switch_boundaries(0.0, 1e-6)
    np.testing.assert_allclose(edges, [0.0, 0.25e-6, 0.5e-6, 0.75e-6, 1.0e-6], atol=1e-21)
    expected_states = [(1, 1), (1, 0), (0, 0), (0, 1)]
    actual_states = [driver.switch_values(float((lo + hi) / 2)) for lo, hi in pairwise(edges)]
    assert actual_states == expected_states
    assert np.isclose(driver.SWITCH_PHASE_OFFSET_RAD, np.pi / 2, rtol=0.0, atol=5e-16)


def test_solver_native_photon_integrals_sum_across_segments() -> None:
    stats = [
        {"photon_integral": 0.125},
        {"photon_integral": 0.25},
        {"photon_integral": 0.5},
    ]
    assert driver.photon_yield(stats) == 0.875


class _MockPackedLayout:
    def unpack(self, packed: np.ndarray) -> np.ndarray:
        return np.asarray(packed, dtype=np.complex128).reshape(2, 2)


def test_segmented_batch_keeps_initial_and_sample_axes(monkeypatch) -> None:
    excited = SimpleNamespace(
        largest=SimpleNamespace(electronic_state=driver.states.ElectronicState.B)
    )
    prepared = SimpleNamespace(layout=_MockPackedLayout())
    case = SimpleNamespace(
        n_states=2,
        prepared_segmented=prepared,
        switch_override_slots=("ω1", "φ1", "ω2", "φ2"),
        system=SimpleNamespace(QN=[excited]),
    )
    initial = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.complex128,
    )
    calls: list[dict[str, object]] = []

    def fake_solve(prepared_arg, rho_batch, t_span, **kwargs):
        assert prepared_arg is prepared
        calls.append(
            {
                "span": t_span,
                "output": kwargs["output"],
                "parameter_batch": np.asarray(kwargs["parameter_batch"]).copy(),
            }
        )
        n_initial = len(rho_batch)
        lo, hi = t_span
        if kwargs["output"] == "full":
            sample_times = np.asarray(kwargs["saveat"], dtype=float)
            values = np.empty((n_initial, len(sample_times), 4), dtype=float)
            for trajectory in range(n_initial):
                for sample_idx, sample_time in enumerate(sample_times, start=1):
                    evolved = np.asarray(rho_batch[trajectory], dtype=np.complex128).copy()
                    fraction = (sample_time - lo) / (hi - lo)
                    evolved[0, 0] += 0.01 * (trajectory + 1) * fraction
                    values[trajectory, sample_idx - 1] = evolved.real.reshape(-1)
            return SimpleNamespace(
                t=sample_times,
                values=values,
                solver_stats={"rhs_calls": 3 * n_initial},
            )
        assert kwargs["output"] == "photon_integral"
        values = (np.arange(1, n_initial + 1, dtype=float) * (hi - lo))[:, None]
        return SimpleNamespace(values=values)

    monkeypatch.setattr(driver, "solve_lindblad_batch", fake_solve)
    times, densities, stats = driver.propagate_segmented_batch(
        case,
        initial,
        (0.0, 0.5e-6),
        save_step_s=0.1e-6,
        parallel=False,
    )

    np.testing.assert_allclose(times, [0, 0.1e-6, 0.2e-6, 0.25e-6, 0.35e-6, 0.45e-6, 0.5e-6])
    assert densities.shape == (2, 7, 2, 2)
    expected = np.empty_like(densities)
    expected[:, 0] = initial
    for trajectory in range(2):
        cursor = 1
        carry = initial[trajectory].copy()
        for segment_idx, (lo, hi) in enumerate(((0.0, 0.25e-6), (0.25e-6, 0.5e-6))):
            start = carry.copy()
            segment_times = times[1 + segment_idx * 3 : 1 + (segment_idx + 1) * 3]
            for sample_time in segment_times:
                carry = start.copy()
                fraction = (sample_time - lo) / (hi - lo)
                carry[0, 0] += 0.01 * (trajectory + 1) * fraction
                expected[trajectory, cursor] = carry
                cursor += 1
    np.testing.assert_allclose(densities, expected, atol=1e-15)
    assert [call["output"] for call in calls] == ["full", "photon_integral"] * 2
    np.testing.assert_allclose(calls[0]["parameter_batch"], [[0, np.pi / 2, 0, np.pi / 2]] * 2)
    np.testing.assert_allclose(calls[2]["parameter_batch"], [[0, np.pi / 2, 0, -np.pi / 2]] * 2)
    np.testing.assert_allclose(stats[0]["photon_integral"], [0.25e-6, 0.5e-6])
    np.testing.assert_allclose(stats[1]["photon_integral"], [0.25e-6, 0.5e-6])


def test_segmented_batch_rejects_empty_initial_state_batch(monkeypatch) -> None:
    case = SimpleNamespace(
        n_states=2,
        prepared_segmented=object(),
        switch_override_slots=("ω1", "φ1", "ω2", "φ2"),
    )

    def solver_must_not_be_called(*_args, **_kwargs):
        raise AssertionError("empty batch reached the solver")

    monkeypatch.setattr(driver, "solve_lindblad_batch", solver_must_not_be_called)
    with pytest.raises(ValueError, match="at least one initial state"):
        driver.propagate_segmented_batch(case, np.empty((0, 2, 2)), (0.0, 0.5e-6))


@pytest.mark.parametrize("t_span", [(1.0, 0.0), (0.0, 0.0), (0.0, np.inf), (np.nan, 1.0)])
def test_segmented_batch_rejects_invalid_time_span(monkeypatch, t_span) -> None:
    case = SimpleNamespace(
        n_states=2,
        prepared_segmented=object(),
        switch_override_slots=("ω1", "φ1", "ω2", "φ2"),
    )

    def solver_must_not_be_called(*_args, **_kwargs):
        raise AssertionError("invalid time span reached the solver")

    monkeypatch.setattr(driver, "solve_lindblad_batch", solver_must_not_be_called)
    with pytest.raises(ValueError, match="finite endpoints with t1 > t0"):
        driver.propagate_segmented_batch(
            case,
            np.zeros((1, 2, 2), dtype=np.complex128),
            t_span,
        )
