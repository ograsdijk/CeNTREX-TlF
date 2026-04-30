import numpy as np
import pytest

from centrex_tlf import states
from centrex_tlf.states.population import (
    J_levels,
    generate_population_states,
    thermal_population,
)


def test_thermal_population_sums_to_one():
    total = sum(thermal_population(J, T=4.0) for J in range(20))
    assert abs(total - 1.0) < 1e-6


def test_thermal_population_j0_dominates_at_low_T():
    p0 = thermal_population(0, T=0.1)
    p1 = thermal_population(1, T=0.1)
    assert p0 > p1


def test_thermal_population_invalid_temperature():
    with pytest.raises(ValueError):
        thermal_population(0, T=0.0)
    with pytest.raises(ValueError):
        thermal_population(0, T=-1.0)


def test_j_levels():
    assert J_levels(0) == 4
    assert J_levels(1) == 12
    assert J_levels(2) == 20


def test_generate_population_states_trace_one():
    rho = generate_population_states([0, 1], levels=4)
    assert rho.shape == (4, 4)
    assert abs(np.trace(rho) - 1.0) < 1e-12


def test_generate_population_states_uniform():
    rho = generate_population_states([0, 1, 2], levels=4)
    assert abs(rho[0, 0] - 1 / 3) < 1e-12
    assert abs(rho[1, 1] - 1 / 3) < 1e-12
    assert abs(rho[2, 2] - 1 / 3) < 1e-12
    assert rho[3, 3] == 0


def test_generate_population_states_invalid_index():
    with pytest.raises((ValueError, IndexError)):
        generate_population_states([10], levels=4)
