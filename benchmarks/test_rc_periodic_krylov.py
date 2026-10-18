"""Small exact linear systems for the matrix-free periodic Arnoldi helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rc_periodic_krylov import arnoldi_repeated_action


def test_augmented_photon_accumulator_and_nonnormal_action() -> None:
    # First two coordinates are a nonnormal density-space surrogate; the final
    # row is an integrated observable and the final column preserves it.
    period = np.array(
        [
            [0.8, 0.25, 0.0],
            [0.0, 0.6, 0.0],
            [0.3, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    initial = np.array([0.4, -0.7, 0.13])
    cycles = 7
    result = arnoldi_repeated_action(
        lambda value: period @ value,
        initial,
        cycles,
        checkpoints=(1, 2, 3),
        max_dimension=3,
    )

    expected = np.linalg.matrix_power(period, cycles) @ initial
    assert result.action_count == 3
    assert result.happy_breakdown
    assert result.breakdown_dimension == 3
    np.testing.assert_allclose(result.checkpoints[3].value, expected, rtol=2e-13, atol=2e-13)
    # The accumulator is not accidentally reset when the period action is
    # applied repeatedly.
    assert result.checkpoints[3].value[-1] != pytest.approx(initial[-1])


def test_happy_breakdown_for_invariant_signed_vector() -> None:
    period = np.diag([0.5, 0.8, 1.0])
    initial = np.array([0.0, -2.0, 0.0])
    result = arnoldi_repeated_action(
        lambda value: period @ value,
        initial,
        5,
        checkpoints=(1, 4),
        max_dimension=4,
    )
    expected = np.array([0.0, -2.0 * 0.8**5, 0.0])
    assert result.happy_breakdown
    assert result.breakdown_dimension == 1
    assert result.checkpoints[4].actual_dimension == 1
    np.testing.assert_allclose(result.checkpoints[4].value, expected, atol=1e-14)


def test_checkpoint_diagnostics_are_dimension_local() -> None:
    period = np.array([[0.5, 0.2, 0.0], [0.0, 0.8, 0.1], [0.0, 0.0, 1.0]])
    result = arnoldi_repeated_action(
        lambda value: period @ value,
        np.array([1.0, -0.5, 0.25]),
        3,
        checkpoints=(1, 2, 3),
        max_dimension=3,
    )
    first, second, final = (result.checkpoints[q] for q in (1, 2, 3))
    assert not first.happy_breakdown
    assert not second.happy_breakdown
    assert final.happy_breakdown
    assert first.residual_indicator != 0.0
    assert second.residual_indicator != 0.0
    assert first.next_arnoldi_coefficient != second.next_arnoldi_coefficient


def test_action_budget_stop_preserves_completed_checkpoints() -> None:
    period = np.diag([0.2, 0.4, 0.6, 0.8, 1.0])
    calls = 0

    def action(value):
        nonlocal calls
        calls += 1
        return period @ value

    result = arnoldi_repeated_action(
        action,
        np.ones(5),
        10,
        checkpoints=(2, 4),
        max_dimension=4,
        stop_requested=lambda: calls >= 3,
    )
    assert result.stopped_early
    assert result.action_count == 3
    assert set(result.checkpoints) == {2}
    assert result.checkpoints[2].actual_dimension == 2


def test_zero_vector_and_zero_cycles_are_exact() -> None:
    result = arnoldi_repeated_action(
        lambda value: np.array([value[1], 0.0]),
        np.array([1.0, 2.0]),
        0,
        checkpoints=(1, 2),
        max_dimension=2,
    )
    assert result.action_count == 0
    np.testing.assert_array_equal(result.checkpoints[2].value, [1.0, 2.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cycles": -1},
        {"cycles": 1, "checkpoints": (0,)},
        {"cycles": 1, "checkpoints": (True,)},
        {"cycles": 1, "max_dimension": 2, "checkpoints": (3,)},
    ],
)
def test_invalid_dimension_or_cycle_arguments_are_rejected(kwargs) -> None:
    parameters = {"cycles": 1, "checkpoints": (1,), "max_dimension": 1}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        arnoldi_repeated_action(lambda value: value, np.array([1.0]), **parameters)


def test_action_shape_and_finiteness_are_checked() -> None:
    with pytest.raises(ValueError, match="shape"):
        arnoldi_repeated_action(lambda value: value[:1], np.ones(2), 1, checkpoints=(1,), max_dimension=1)
    with pytest.raises(ValueError, match="non-finite"):
        arnoldi_repeated_action(
            lambda value: np.full_like(value, np.nan),
            np.ones(2),
            1,
            checkpoints=(1,),
            max_dimension=1,
        )
    with pytest.raises(ValueError, match="real"):
        arnoldi_repeated_action(
            lambda value: value.astype(np.complex128),
            np.ones(2),
            1,
            checkpoints=(1,),
            max_dimension=1,
        )
    with pytest.raises(ValueError, match="real"):
        arnoldi_repeated_action(
            lambda value: value,
            np.ones(2, dtype=np.complex128),
            1,
            checkpoints=(1,),
            max_dimension=1,
        )
