"""Experimental TlF carrier, Doppler units and explicit frequency overrides."""

import numpy as np
import pytest
from scipy.constants import c

from centrex_tlf.constants import (
    X_B_REFERENCE_FREQUENCY_HZ,
    X_B_REFERENCE_WAVELENGTH_M,
)
from centrex_tlf.utils.detuning import doppler_shift, velocity_to_detuning


def test_experimental_optical_reference():
    assert X_B_REFERENCE_FREQUENCY_HZ == 1_103_435_082_708_000.0
    assert X_B_REFERENCE_WAVELENGTH_M == c / X_B_REFERENCE_FREQUENCY_HZ
    assert doppler_shift(0.0) == X_B_REFERENCE_FREQUENCY_HZ


def test_default_doppler_scale_and_sign():
    velocities = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        velocity_to_detuning(velocities) / (2 * np.pi * 1e6),
        [-3.6806632497339207, 0.0, 3.6806632497339207],
    )
    np.testing.assert_allclose(
        doppler_shift(velocities), X_B_REFERENCE_FREQUENCY_HZ * (1 + velocities / c)
    )


def test_explicit_frequency_overrides():
    frequencies = np.array([1e9, 2e9])
    np.testing.assert_allclose(doppler_shift(10.0, frequencies), frequencies * (1 + 10 / c))
    np.testing.assert_allclose(
        velocity_to_detuning(10.0, frequencies), 2 * np.pi * frequencies * 10 / c
    )
    assert doppler_shift(0.0, 1e9) == 1e9


@pytest.mark.parametrize("function", [doppler_shift, velocity_to_detuning])
@pytest.mark.parametrize("frequency", [0.0, -1.0, np.array([1e9, 0.0])])
def test_nonpositive_frequency_rejected(function, frequency):
    with pytest.raises(ValueError, match="Frequency must be positive"):
        function(1.0, frequency)
