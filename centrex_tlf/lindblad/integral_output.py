from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

INTEGRAL_OUTPUTS = {"weighted_integral", "photon_integral", "excited_population"}
INTEGRAL_RATE_OUTPUTS = {
    "weighted_integral": "weighted_rate",
    "photon_integral": "photon_rate",
    "excited_population": "excited_population_rate",
}


def normalize_integral_saveat(
    integral_saveat: None | float | Sequence[float] | npt.NDArray[np.floating],
    t_span: tuple[float, float],
) -> np.ndarray:
    """Normalize the explicit quadrature grid and include both interval endpoints."""
    if integral_saveat is None:
        raise ValueError("integral_method='sampled' requires explicit integral_saveat values")
    if isinstance(integral_saveat, float | int | np.floating | np.integer):
        step = float(integral_saveat)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("integral_saveat step must be positive and finite")
        values = np.arange(t_span[0], t_span[1], step, dtype=np.float64)
    else:
        values = np.asarray(integral_saveat, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("integral_saveat must be a one-dimensional sequence")
        if not np.all(np.isfinite(values)):
            raise ValueError("integral_saveat values must be finite")
        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError("integral_saveat values must be strictly increasing")

    t0, t1 = t_span
    if np.any(values < t0) or np.any(values > t1):
        raise ValueError("integral_saveat values must lie inside t_span")
    return np.unique(np.concatenate(([t0], values, [t1]))).astype(np.float64, copy=False)


def cumulative_trapezoid_at(
    sample_times: npt.NDArray[np.floating],
    sample_rates: npt.NDArray[np.floating],
    query_times: npt.NDArray[np.floating],
) -> np.ndarray:
    """Evaluate a piecewise-linear-rate trapezoid integral at query times.

    The returned samples do not become additional quadrature knots, so output
    sampling cannot change the integral grid.
    """
    times = np.asarray(sample_times, dtype=np.float64)
    rates = np.asarray(sample_rates, dtype=np.float64)
    queries = np.asarray(query_times, dtype=np.float64)
    if times.ndim != 1 or queries.ndim != 1 or rates.shape[-1] != times.size:
        raise ValueError("sample times, rates, and query times have incompatible shapes")
    if times.size == 0:
        raise ValueError("the integral sampling grid produced no rate samples")
    if times.size == 1:
        return np.zeros((*rates.shape[:-1], queries.size), dtype=np.float64)

    widths = np.diff(times)
    if np.any(widths <= 0.0):
        raise ValueError("integral sampling times must be strictly increasing")
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(times[0]), abs(times[-1]))
    if np.any(queries < times[0] - tolerance) or np.any(queries > times[-1] + tolerance):
        raise ValueError("requested integral output times must lie inside the integration interval")
    queries = np.clip(queries, times[0], times[-1])

    interval_areas = 0.5 * (rates[..., :-1] + rates[..., 1:]) * widths
    cumulative = np.concatenate(
        [np.zeros((*rates.shape[:-1], 1), dtype=np.float64), np.cumsum(interval_areas, axis=-1)],
        axis=-1,
    )
    left = np.clip(np.searchsorted(times, queries, side="right") - 1, 0, times.size - 2)
    partial_widths = queries - times[left]
    left_rates = rates[..., left]
    right_rates = rates[..., left + 1]
    query_rates = left_rates + (right_rates - left_rates) * (partial_widths / widths[left])
    partial_areas = 0.5 * (left_rates + query_rates) * partial_widths
    return cumulative[..., left] + partial_areas
