import numpy as np
import pytest
from run_rc_tolerance_sweep import _stats_summary, compare_trajectory


def test_compare_trajectory_checks_full_saved_grid_and_endpoint():
    times = np.array([0.0, 1.0])
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, 0, 0] = 1.0
    result = compare_trajectory(times, density, 0.0, times, density.copy(), 0.0)

    assert result["candidate_finite"]
    assert result["common_time_grid_matches"]
    assert result["all_saved"]["passes"]
    assert result["endpoint"]["passes"]
    assert result["passes"]


def test_compare_trajectory_retains_nonfinite_failure_without_eigensolve():
    times = np.array([0.0, 1.0])
    density = np.zeros((2, 2, 2), dtype=np.complex128)
    density[:, 0, 0] = 1.0
    candidate = density.copy()
    candidate[1, 0, 0] = np.nan
    result = compare_trajectory(times, candidate, 0.0, times, density, 0.0)

    assert not result["candidate_finite"]
    assert not result["passes"]
    assert result["all_saved"] is None
    assert result["failure"]


def test_solver_stats_separate_density_and_photon_call_costs():
    stats = _stats_summary(
        [
            {
                "accepted_steps": 3,
                "rejected_steps": 1,
                "rhs_calls": 25,
                "full_density_wall_seconds": 0.2,
                "photon_integral_wall_seconds": 0.1,
                "combined_wall_seconds": 0.3,
            },
            {
                "accepted_steps": 4,
                "rejected_steps": 0,
                "rhs_calls": 30,
                "full_density_wall_seconds": 0.4,
                "photon_integral_wall_seconds": 0.15,
                "combined_wall_seconds": 0.55,
            },
        ]
    )

    assert stats["full_density_accepted_steps"] == 7
    assert stats["full_density_rejected_steps"] == 1
    assert stats["full_density_rhs_calls"] == 55
    assert stats["full_density_solver_wall_seconds"] == pytest.approx(0.6)
    assert stats["photon_integral_solver_wall_seconds"] == pytest.approx(0.25)
