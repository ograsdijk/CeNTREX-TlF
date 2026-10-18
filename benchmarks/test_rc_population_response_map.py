"""Analytical unit tests for population-mixture response-map semantics."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rc_population_response_map import (
    build_diagonal_response,
    canonical_cache_key,
    load_response_map,
    make_response_map,
    save_response_map,
    validate_mixture_response,
)

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout


def test_two_level_amplitude_damping_response_map_is_linear() -> None:
    layout = PackedHermitianLayout(2)
    decay_probability = 0.35
    ground = np.diag([1.0, 0.0]).astype(np.complex128)
    excited_final = np.diag([1.0 - decay_probability, decay_probability]).astype(
        np.complex128
    )
    responses = np.column_stack(
        [
            np.r_[layout.pack(ground), 0.0],
            np.r_[layout.pack(excited_final), 1.0 - decay_probability],
        ]
    )
    response_map = make_response_map(2, [0, 1], responses, {"rate": 0.35})

    weights = np.array([0.2, 0.8])
    result = response_map.apply(weights, selected_indices=[1])
    expected = np.diag([1.0 - 0.8 * decay_probability, 0.8 * decay_probability])
    np.testing.assert_allclose(result.density, expected, atol=1e-15)
    np.testing.assert_allclose(result.populations, np.diag(expected).real, atol=1e-15)
    np.testing.assert_allclose(result.selected_populations, [expected[1, 1].real], atol=1e-15)
    assert result.photon_yield == pytest.approx(0.8 * (1.0 - decay_probability))


def test_population_mixture_reconstructs_full_complex_coherences() -> None:
    layout = PackedHermitianLayout(2)
    rho0 = np.array([[0.8, 0.1 + 0.3j], [0.1 - 0.3j, 0.2]], dtype=np.complex128)
    rho1 = np.array([[0.3, -0.2 + 0.1j], [-0.2 - 0.1j, 0.7]], dtype=np.complex128)
    responses = np.column_stack([np.r_[layout.pack(rho0), 0.4], np.r_[layout.pack(rho1), 0.9]])
    response_map = make_response_map(2, [1, 0], responses, {"entry_phase_s": 0.0})

    # Response columns correspond to coverage order [1, 0].
    result = response_map.apply(np.array([0.25, 0.75]))
    # The vector has full-layout order [state 0, state 1]; map columns are
    # coverage order [state 1, state 0].
    expected = 0.75 * rho0 + 0.25 * rho1
    np.testing.assert_allclose(result.density, expected, atol=1e-15)
    assert result.density[0, 1].imag == pytest.approx(expected[0, 1].imag)
    assert result.photon_yield == pytest.approx(0.75 * 0.4 + 0.25 * 0.9)
    weight_rows = np.array([[1.0, 0.0], [0.25, 0.75]])
    density_rows = response_map.density(weight_rows)
    photon_rows = response_map.photons(weight_rows)
    population_rows = response_map.populations(weight_rows, selected_indices=[0, 1])
    np.testing.assert_allclose(density_rows[0], rho1, atol=1e-15)
    np.testing.assert_allclose(density_rows[1], expected, atol=1e-15)
    np.testing.assert_allclose(photon_rows, [0.9, 0.525], atol=1e-15)
    np.testing.assert_allclose(
        population_rows,
        np.diagonal(density_rows, axis1=1, axis2=2).real.T,
    )


def test_subset_map_rejects_uncovered_states_and_invalid_mixtures() -> None:
    layout = PackedHermitianLayout(3)
    ground0 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    ground1 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    responses = np.column_stack(
        [np.r_[layout.pack(ground0), 0.0], np.r_[layout.pack(ground1), 0.0]]
    )
    response_map = make_response_map(3, [0, 1], responses, {"coverage": "X"})

    with pytest.raises(ValueError, match="outside this map's coverage"):
        response_map.apply(np.array([0.5, 0.5, 1e-3]))
    with pytest.raises(ValueError, match="nonnegative"):
        response_map.apply(np.array([1.1, -0.1]))
    with pytest.raises(ValueError, match="sum to one"):
        response_map.apply(np.array([0.4, 0.4]))
    result = response_map.apply(np.array([0.4, 0.6]))
    np.testing.assert_allclose(result.populations, [0.4, 0.6, 0.0])
    with pytest.raises(ValueError, match="real-valued"):
        response_map.apply(np.array([0.4 + 0.0j, 0.6 + 0.0j]))
    with pytest.raises(ValueError, match="integer"):
        make_response_map(3, [0.5, 1.5], responses, {"coverage": "bad"})


def test_cache_key_changes_when_runtime_or_phase_metadata_changes() -> None:
    baseline = {"duration_s": 2e-4, "entry_phase_s": 0.0, "rabi": 1.0}
    assert canonical_cache_key(baseline) == canonical_cache_key(dict(reversed(list(baseline.items()))))
    assert canonical_cache_key(baseline) != canonical_cache_key({**baseline, "entry_phase_s": 1e-9})
    assert canonical_cache_key(baseline) != canonical_cache_key({**baseline, "rabi": 1.01})


def test_cache_roundtrip_and_invalidation_metadata() -> None:
    layout = PackedHermitianLayout(2)
    density = np.diag([0.6, 0.4]).astype(np.complex128)
    columns = np.column_stack([np.r_[layout.pack(density), 0.2]])
    metadata = {
        "duration_s": 2e-4,
        "native_sha256": "abc",
        "entry_phase_s": 0.0,
        "polarization_vector": [0.5 + 0.25j],
    }
    response_map = make_response_map(2, [1], columns, metadata)
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        data_path = root / "response.npz"
        metadata_path = root / "response.json"
        saved = save_response_map(response_map, data_path, metadata_path)
        loaded = load_response_map(data_path, metadata_path, expected_metadata=metadata)
        assert saved["cache_key"] == canonical_cache_key(metadata)
        np.testing.assert_array_equal(loaded.response_columns, columns)
        with pytest.raises(ValueError, match="cache key"):
            load_response_map(
                data_path,
                metadata_path,
                expected_metadata={**metadata, "entry_phase_s": 10e-9},
            )


def test_diagonal_response_builder_batches_segments_and_native_integral(monkeypatch) -> None:
    import bench_rc_propagation as driver
    import rc_population_response_map as response_module

    layout = PackedHermitianLayout(2)
    prepared = SimpleNamespace(layout=layout)
    case = SimpleNamespace(
        n_states=2,
        prepared_segmented=prepared,
        switch_override_slots=("a", "b", "c", "d"),
    )
    monkeypatch.setattr(driver, "switch_boundaries", lambda lo, hi: np.array([lo, hi]))
    monkeypatch.setattr(driver, "switch_values", lambda _time: (1, 0))
    observed: dict[str, object] = {}

    def fake_batch(_prepared, rho_batch, t_span, **kwargs):
        observed["rho_batch"] = rho_batch.copy()
        observed["t_span"] = t_span
        observed["kwargs"] = kwargs
        values = np.zeros((rho_batch.shape[0], layout.packed_len + 1))
        for row, rho in zip(values, rho_batch, strict=True):
            row[:-1] = rho
            row[-1] = 0.25
        return SimpleNamespace(values=values, solver_stats={"rhs_calls": 12})

    monkeypatch.setattr(response_module, "solve_lindblad_batch", fake_batch)
    response_map, stats = build_diagonal_response(
        case,
        [0, 1],
        (0.0, 5e-6),
        metadata={"runtime": "mock"},
        batch_size=2,
        parallel=True,
        threads=8,
        integral_weights=[(1, 1.0)],
    )
    np.testing.assert_array_equal(response_map.covered_indices, [0, 1])
    np.testing.assert_allclose(response_map.response_columns[-1], [0.25, 0.25])
    assert observed["t_span"] == (0.0, 5e-6)
    assert observed["kwargs"]["output"] == "full_and_integral"
    assert observed["kwargs"]["parameter_batch"].shape == (2, 4)
    assert observed["kwargs"]["threads"] == 8
    assert observed["rho_batch"].shape == (2, layout.packed_len)
    assert observed["rho_batch"][0, 0] == pytest.approx(1.0)
    assert stats.n_solve_calls == 1
    assert stats.per_column_allocated_wall_seconds[0]["allocation_method"].startswith(
        "batch wall divided equally"
    )


def test_mixture_validation_compares_full_density_and_photons() -> None:
    layout = PackedHermitianLayout(2)
    rho_a = np.array([[0.75, 0.1j], [-0.1j, 0.25]], dtype=np.complex128)
    rho_b = np.array([[0.2, 0.05 + 0.1j], [0.05 - 0.1j, 0.8]], dtype=np.complex128)
    response_map = make_response_map(
        2,
        [0, 1],
        np.column_stack([np.r_[layout.pack(rho_a), 0.3], np.r_[layout.pack(rho_b), 0.7]]),
        {"case": "mock"},
    )
    weights = np.array([0.4, 0.6])
    expected_density = 0.4 * rho_a + 0.6 * rho_b
    validation = validate_mixture_response(
        response_map,
        weights,
        {"endpoint_density": expected_density, "photon_yield": 0.54},
    )
    assert validation.density_matches
    assert validation.photons_match
    assert validation.density_max_abs_error == pytest.approx(0.0)
    assert validation.photon_abs_error == pytest.approx(0.0)
