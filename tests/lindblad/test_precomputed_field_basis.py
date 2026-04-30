from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from centrex_tlf import couplings, transitions

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

try:
    from effective_downfolding import build_transition_ground_basis_state
    from precomputed_field_basis import (
        operator_bundle,
        prepare_precomputed_adiabatic_obe_model,
        project_common_state_to_initial_adiabatic_basis,
        solve_precomputed_adiabatic_model,
    )
except ImportError:
    pytestmark = pytest.mark.skip(
        reason="effective_downfolding / precomputed_field_basis not available"
    )


def electric_reference_vector(axis: str, value: float) -> np.ndarray:
    vector = np.zeros(3, dtype=float)
    vector[{"X": 0, "Y": 1, "Z": 2}[axis.upper()]] = value
    return vector


def test_prepare_precomputed_adiabatic_obe_model_uses_shared_basis_across_field_grid():
    reference_transition = transitions.Q1_F1_1o2_F0
    common_ground_basis = build_transition_ground_basis_state(
        reference_transition,
        ground_F1=1 / 2,
        ground_F=1,
        ground_mF=0,
    )
    ground_main_state = 1.0 * common_ground_basis
    optical_polarization = couplings.polarization_Z
    excited_candidates = [1.0 * state for state in reference_transition.excited_states]
    excited_main_state = next(
        state for state in excited_candidates if int(state.largest.mF) == 0
    )
    transition_selectors = couplings.generate_transition_selectors(
        transitions=[reference_transition],
        polarizations=[[optical_polarization]],
        ground_mains=[ground_main_state],
        excited_mains=[excited_main_state],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Low overlap detected for approximate states.*",
        )
        prepared = prepare_precomputed_adiabatic_obe_model(
            transitions=[reference_transition],
            transition_selectors=transition_selectors,
            field_grid=np.array([0.0, 100.0], dtype=float),
            electric_field_vector_fn=lambda value: electric_reference_vector("X", value),
            magnetic_field=np.array([0.0, 0.0, 1e-5], dtype=float),
            include_connection=True,
        )

    assert prepared.transform_grid.shape == (2, len(prepared.full_states), 37)
    assert prepared.h_diag_grid.shape == (2, 37, 37)
    assert prepared.c_array_grid[0].shape == prepared.c_array_grid[1].shape
    assert len(prepared.tracked_states) == 37
    assert len(prepared.full_states) == 43

    bundle = operator_bundle(
        prepared,
        electric_field=50.0,
        electric_field_derivative=0.0,
        rabi_rate=0.0,
    )
    assert bundle["c_array"].shape == prepared.c_array_grid[0].shape

    psi0, _ = project_common_state_to_initial_adiabatic_basis(prepared, common_ground_basis)
    rho0 = np.outer(psi0, np.conjugate(psi0))
    solution = solve_precomputed_adiabatic_model(
        prepared,
        t_span=(0.0, 1e-7),
        rho0=rho0,
        electric_field_fn=lambda t: 100.0 * float(t / 1e-7),
        electric_field_derivative_fn=lambda t: 1e9,
        rabi_rate=0.0,
        t_eval=np.linspace(0.0, 1e-7, 5),
        method="RK4-fixed",
    )
    assert solution.success
    assert solution.y.shape == (37 * 37, 5)
