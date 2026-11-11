"""Utility functions for TlF molecular spectroscopy calculations.

This module provides utilities for:
- Doppler shifts and velocity-to-detuning conversions
- Multipass beam geometry and intensity profiles
- Thermal and uniform population distributions
- Rabi frequency and beam power calculations
"""

from importlib import import_module
from typing import Any

from . import detuning, multipass, population, rabi

# `plotting` pulls in matplotlib, so it is imported on first use rather than with
# the package. Names must match `plotting.__all__` (checked in the tests).
_PLOTTING_NAMES = (
    "plot_level_diagram",
    "filter_levels_with_decay_or_coupling",
    "combine_decay_only_states",
    "DressedLevel",
    "TransitionLevelStructure",
    "TransitionLevelDiagram",
    "calculate_transition_level_structure",
    "plot_transition_level_diagram",
)

__all__ = detuning.__all__
__all__ += multipass.__all__
__all__ += population.__all__
__all__ += rabi.__all__
__all__ += list(_PLOTTING_NAMES)


def __getattr__(name: str) -> Any:
    if name == "plotting" or name in _PLOTTING_NAMES:
        module = import_module(".plotting", __name__)
        return module if name == "plotting" else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
