"""Validation and geometry checks for multipass beams."""

from fractions import Fraction

import numpy as np
import pytest

from centrex_tlf.utils.multipass import (
    PRISM_LEG_LENGTH_M,
    multipass_prism_order,
    trace_prism_multipass,
)


@pytest.mark.parametrize("passes", [-7, -3, -1, 0, 2, 4])
def test_prism_order_rejects_nonpositive_or_even_counts(passes):
    with pytest.raises(ValueError):
        multipass_prism_order(passes)


def test_single_pass():
    assert multipass_prism_order(1) == [1]


def test_seven_pass_order_matches_prism_reflections():
    face_length = PRISM_LEG_LENGTH_M * np.sqrt(2)
    step = face_length / 7.25
    order, locations = trace_prism_multipass(-step, face_length - 0.5 * step)

    # Odd passes advance by 2*step; even passes travel in the other direction.
    chronological_locations = locations[np.argsort(order)]
    np.testing.assert_allclose(
        chronological_locations / step, [0.5, 4.75, 2.5, 2.75, 4.5, 0.75, 6.5]
    )
    assert order == [1, 6, 3, 4, 5, 2, 7]
    assert multipass_prism_order(7) == order


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("ulps", [-256, -1, 0, 1, 256])
def test_endpoint_hits_and_clear_misses(sign, ulps):
    face_length = PRISM_LEG_LENGTH_M * np.sqrt(2)
    offset = sign * face_length / 4
    entry = (0.25 if sign > 0 else 0.75) * face_length
    entry += sign * ulps * np.spacing(face_length)
    order, locations = trace_prism_multipass(offset, entry)

    # Positive perturbations miss the next face; one-ULP misses are snapped.
    assert len(order) == (3 if ulps == 256 else 4)
    if abs(ulps) <= 1:
        endpoint = offset if sign > 0 else offset + face_length
        assert locations[np.argsort(order)][2] == endpoint


@pytest.mark.parametrize("denominator", [*range(2, 21), 100, 1000])
@pytest.mark.parametrize("sign", [-1, 1])
def test_boundary_traces_against_exact_arithmetic(denominator, sign):
    face_length = PRISM_LEG_LENGTH_M * np.sqrt(2)
    offset = Fraction(sign, denominator)
    for entry in map(Fraction, [0, 0.25, 0.5, 0.75, 1]):
        point, side = entry, 0
        expected = []
        while True:
            point = 1 + 2 * side * offset - point
            expected.append(float(point) * face_length)
            side = 1 - side
            if not side * offset <= point <= side * offset + 1:
                break
        order, locations = trace_prism_multipass(
            float(offset) * face_length, float(entry) * face_length
        )
        actual = locations[np.argsort(order)]
        assert len(actual) == len(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)


def _triangle_exit(entry, face_length):
    """Independent ray/segment intersections and vector specular reflections."""
    vertices = np.array(
        [[0.0, 0.0], [face_length, 0.0], [face_length / 2, face_length / 2]]
    )
    point, direction = np.array([entry, 0.0]), np.array([0.0, 1.0])
    for bounce in range(3):
        candidates = []
        for index in range(3):
            start = vertices[index]
            segment = vertices[(index + 1) % 3] - start
            matrix = np.column_stack((direction, -segment))
            if abs(np.linalg.det(matrix)) < 1e-14:
                continue
            distance, fraction = np.linalg.solve(matrix, start - point)
            if distance > 1e-12 and 0 <= fraction <= 1:
                candidates.append((distance, index, segment))
        assert candidates
        distance, index, segment = min(candidates, key=lambda item: item[0])
        point = point + distance * direction
        if index == 0:
            assert bounce == 2
            np.testing.assert_allclose(direction, [0, -1], atol=1e-14)
            return point[0]
        normal = np.array([-segment[1], segment[0]]) / np.linalg.norm(segment)
        direction = direction - 2 * np.dot(direction, normal) * normal
    pytest.fail("Ray did not exit after two internal reflections")


def test_tracer_against_independent_triangle_intersections():
    face_length = PRISM_LEG_LENGTH_M * np.sqrt(2)
    rng = np.random.default_rng(734)
    for _ in range(300):
        offset = rng.choice([-1, 1]) * rng.uniform(0.005, 1.5) * face_length
        entry = rng.uniform(0.001, 0.999) * face_length
        point = entry
        side, expected = 0, []
        for _ in range(1000):
            shift = side * offset
            point = _triangle_exit(point - shift, face_length) + shift
            expected.append(point)
            side = 1 - side
            if not side * offset <= point <= side * offset + face_length:
                break
        else:
            pytest.fail("Reference trace did not terminate")
        order, locations = trace_prism_multipass(offset, entry)
        actual = locations[np.argsort(order)]
        assert len(actual) == len(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-13)


@pytest.mark.parametrize(
    "offset, entry",
    [(0.0, 0.01), (np.nan, 0.01), (0.01, np.inf), (0.01, -0.01)],
)
def test_tracer_rejects_invalid_inputs(offset, entry):
    with pytest.raises(ValueError):
        trace_prism_multipass(offset, entry)


def test_tracer_detects_unresolvable_offset():
    with pytest.raises(ValueError, match="stopped progressing"):
        trace_prism_multipass(np.nextafter(0.0, 1.0), 0.01)
