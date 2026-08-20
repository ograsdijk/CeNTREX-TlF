"""Pinning tests for the interpolated effective-model preparation path.

`prepare_interpolated_effective_model` and
`prepare_lindblad_safe_compact_interpolated_model` were previously uncovered --
the only test in this directory (`test_grid_diagnostics.py`) builds synthetic
2x2 bundles by hand and exercises the diagnostics helper, not the prep path.
These tests pin the structure and the numerics that the prep path produces so
that refactors of it (hoisting field-independent work out of the per-point
loop, reusing already-built systems, parallelising the loop) can be checked.

Rather than golden magic numbers, these pin *invariants* and *equivalences*.
Bundle entries depend on eigenvectors, which for near-degenerate +-mF pairs are
BLAS-build sensitive (see AGENTS.md), so the magnetic field here is 1e-3 G --
large enough to lift the degeneracy properly -- and the assertions avoid
individual off-diagonal amplitudes.

The builds are slow (~0.6 s per field point per build), so both models are
built once at module scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from centrex_tlf import couplings, transitions
from centrex_tlf.effective_hamiltonian.compact_reference import (
    build_compact_reference_decomposed_bundle,
)
from centrex_tlf.effective_hamiltonian.operator_bundle import (
    _compact_transition_frequency,
)
from centrex_tlf.effective_hamiltonian.preparation import (
    prepare_interpolated_effective_model,
    prepare_lindblad_safe_compact_interpolated_model,
)

TRANSITION = transitions.Q1_F1_3o2_F2
FIELD_POINTS = (100.0, 150.0, 200.0)
MASTER_FIELD = 150.0
# Not the 1e-5 G placeholder: at that field the +-mF eigenvectors are only
# determined to ~1e-4..1e-2 and vary with the BLAS build.
MAGNETIC_FIELD = (0.0, 0.0, 1e-3)


@pytest.fixture(scope="module")
def polarization() -> couplings.Polarization:
    return couplings.Polarization(
        np.array([0.0, 0.0, 1.0], dtype=np.complex128), name="Z"
    )


@pytest.fixture(scope="module")
def base_model(polarization):
    return prepare_interpolated_effective_model(
        field_points=FIELD_POINTS,
        transition=TRANSITION,
        optical_polarization=polarization,
        magnetic_field=MAGNETIC_FIELD,
        master_field=MASTER_FIELD,
    )


@pytest.fixture(scope="module")
def safe_model(polarization):
    return prepare_lindblad_safe_compact_interpolated_model(
        field_points=FIELD_POINTS,
        transition=TRANSITION,
        optical_polarization=polarization,
        magnetic_field=MAGNETIC_FIELD,
        master_field=MASTER_FIELD,
    )


def test_base_model_grid_layout(base_model):
    np.testing.assert_allclose(base_model.field_points, np.array(FIELD_POINTS))
    assert base_model.master_field == MASTER_FIELD
    assert len(base_model.patches) == len(FIELD_POINTS)
    for patch, field_z in zip(base_model.patches, FIELD_POINTS):
        np.testing.assert_allclose(patch.electric_field, [0.0, 0.0, field_z])


def test_base_model_index_sets_are_disjoint_and_in_range(base_model):
    n = base_model.n_effective_states
    ground = base_model.ground_indices
    sink = base_model.sink_indices
    excited = base_model.excited_indices
    for name, indices in (("ground", ground), ("sink", sink), ("excited", excited)):
        assert indices.size > 0, f"{name} index set is empty"
        assert indices.min() >= 0 and indices.max() < n, f"{name} index out of range"
    combined = np.concatenate([ground, sink, excited])
    assert combined.size == np.unique(combined).size, "index sets overlap"
    assert base_model.ground_main_index in ground.tolist()


def test_every_patch_shares_one_operator_shape(base_model):
    n = base_model.n_effective_states
    n_parent = len(base_model.parent_basis_qn)
    for patch in base_model.patches:
        bundle = patch.bundle
        assert bundle.h_internal.shape == (n, n)
        assert bundle.h_opt.shape == (n, n)
        assert bundle.h_det.shape == (n, n)
        # Columns are the effective states, rows the parent basis they are
        # expressed in, so this is (n_parent, n) and not square.
        assert patch.aligned_basis_vectors.shape == (n_parent, n)


def test_patch_operators_are_hermitian(base_model):
    for patch in base_model.patches:
        bundle = patch.bundle
        for name in ("h_internal", "h_opt", "h_det"):
            operator = np.asarray(getattr(bundle, name), dtype=np.complex128)
            np.testing.assert_allclose(
                operator, operator.conj().T, atol=1e-8, err_msg=f"{name} not Hermitian"
            )


def test_decay_rates_are_non_negative(base_model):
    for patch in base_model.patches:
        bundle = patch.bundle
        assert np.all(np.asarray(bundle.excited_to_ground_rates_hz) >= -1e-9)
        assert np.all(np.asarray(bundle.excited_to_sink_rates_hz) >= -1e-9)


def test_patch_transition_frequencies_match_independent_rebuild(
    safe_model, polarization
):
    """Pin the frequencies against a from-scratch rebuild of each patch.

    `prepare_lindblad_safe_compact_interpolated_model` used to obtain these by
    rebuilding every patch a second time, after
    `prepare_interpolated_effective_model` had already built them. This test
    pins the values against that independent rebuild so the duplicate build can
    be removed without changing results.
    """
    rebuilt = []
    for field_z in np.asarray(safe_model.field_points, dtype=np.float64).tolist():
        system, _ = build_compact_reference_decomposed_bundle(
            transition=TRANSITION,
            optical_polarization=polarization,
            electric_field=np.array([0.0, 0.0, float(field_z)], dtype=np.float64),
            magnetic_field=np.asarray(MAGNETIC_FIELD, dtype=np.float64),
        )
        rebuilt.append(
            _compact_transition_frequency(
                system, transition=TRANSITION, optical_polarization=polarization
            )
        )
    np.testing.assert_allclose(
        np.asarray(safe_model.patch_transition_frequencies, dtype=np.float64),
        np.asarray(rebuilt, dtype=np.float64),
        rtol=1e-12,
        atol=0.0,
    )


def test_common_omega_reference_is_the_master_point(safe_model):
    field_points = np.asarray(safe_model.field_points, dtype=np.float64)
    master_index = int(np.argmin(np.abs(field_points - safe_model.master_field)))
    assert safe_model.common_omega_reference == pytest.approx(
        float(safe_model.patch_transition_frequencies[master_index]), rel=1e-12
    )


def test_safe_model_matches_base_model_layout(safe_model, base_model):
    np.testing.assert_allclose(safe_model.field_points, base_model.field_points)
    np.testing.assert_array_equal(
        safe_model.ground_indices, base_model.ground_indices
    )
    np.testing.assert_array_equal(safe_model.sink_indices, base_model.sink_indices)
    np.testing.assert_array_equal(
        safe_model.excited_indices, base_model.excited_indices
    )
    assert safe_model.ground_main_index == base_model.ground_main_index
    assert safe_model.n_effective_states == base_model.n_effective_states
    np.testing.assert_array_equal(
        safe_model.target_indices,
        np.concatenate([base_model.ground_indices, base_model.sink_indices]),
    )
