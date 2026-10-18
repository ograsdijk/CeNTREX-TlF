from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
suite = importlib.import_module("run_rc_comparison")


def _valid_density() -> np.ndarray:
    return np.asarray([np.diag([0.7, 0.3]), np.diag([0.65, 0.35])], dtype=np.complex128)


def _fake_solver(monkeypatch, *, fail_refinement: bool = False, mismatch_tsit: bool = False):
    calls = []

    def fake_ode_result(_case, _rho, _span, *, solver, reltol, execution_mode, dt_s, **_kwargs):
        calls.append((solver, reltol, dt_s, execution_mode))
        times = np.asarray([0.0, 1.0])
        density = _valid_density()
        if fail_refinement and dt_s == 5e-11:
            density = density.copy()
            density[:, 0, 0] += 2e-6
            density[:, 1, 1] -= 2e-6
        if mismatch_tsit and solver == "tsit5":
            times[-1] = 1.1
        return times, density, 0.25, []

    monkeypatch.setattr(suite, "ode_result", fake_ode_result)
    return calls


def test_validated_reference_reduces_tolerance_and_initial_step(monkeypatch) -> None:
    calls = _fake_solver(monkeypatch)
    result, metadata = suite.validated_reference(object(), np.zeros((2, 2)), (0.0, 1.0), 3e-11)

    assert metadata["validated"]
    assert result[0].tolist() == [0.0, 1.0]
    assert calls == [
        ("dopri5", 1e-10, 1e-10, "expanded_sparse"),
        ("dopri5", 3e-11, 5e-11, "expanded_sparse"),
        ("dopri5", 1e-11, 2.5e-11, "expanded_sparse"),
        ("tsit5", 1e-11, 2.5e-11, "expanded_sparse"),
    ]


def test_validated_reference_rejects_unconverged_refinement(monkeypatch) -> None:
    _fake_solver(monkeypatch, fail_refinement=True)
    _, metadata = suite.validated_reference(object(), np.zeros((2, 2)), (0.0, 1.0), 3e-11)
    assert not metadata["validated"]
    assert not metadata["checks"]["primary_to_refined"]["passes_reference_tolerance"]


def test_validated_reference_rejects_cross_solver_time_grid_mismatch(monkeypatch) -> None:
    _fake_solver(monkeypatch, mismatch_tsit=True)
    with pytest.raises(AssertionError):
        suite.validated_reference(object(), np.zeros((2, 2)), (0.0, 1.0), 3e-11)


@pytest.mark.parametrize("tolerance", [1e-10, 0.0, -1e-11, np.inf, np.nan])
def test_validated_reference_requires_tolerance_below_primary(monkeypatch, tolerance) -> None:
    calls = _fake_solver(monkeypatch)
    with pytest.raises(ValueError, match="less than 1e-10"):
        suite.validated_reference(object(), np.zeros((2, 2)), (0.0, 1.0), tolerance)
    assert not calls


@pytest.mark.parametrize("artifact_stage", ["ode", "batch"])
def test_reference_reuse_rejects_changed_physics_and_loads_full_density(tmp_path, artifact_stage):
    provenance = {
        key: key
        for key in (
            "config",
            "model",
            "amplitude_envelope",
            "driver_sha256",
            "rust_extension_sha256",
        )
    }
    provenance["model"] = {"polarizations": [(1.0, 0.0), (0.0, 1.0)]}
    payload = {"provenance": provenance, "helper_sha256": "helper", "batch_api_sha256": "batch"}
    saved = copy.deepcopy(payload)
    saved["arguments"] = {"stage": artifact_stage}
    saved["initial_labels"] = ["state"]
    saved.update(
        status="complete",
        timings_accepted=True,
        reference_checks=[
            {
                "initial_label": "state",
                "validated": True,
                "checks": {
                    "tight_reference_physical": {"photon_yield": 0.25},
                    "primary_to_refined": {},
                    "refined_to_tight": {},
                    "tsit5_crosscheck": {},
                },
            }
        ],
    )
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    arrays = (
        {"times": [0.0, 1.0], "reference": _valid_density()[None]}
        if artifact_stage == "batch"
        else {"state_times": [0.0, 1.0], "state_reference": _valid_density()}
    )
    np.savez(tmp_path / "ref_densities.npz", **arrays)
    result, check = suite.load_reference(path, payload, "state")
    np.testing.assert_array_equal(result[1], _valid_density())
    assert result[2] == 0.25 and check["validated"]
    payload["provenance"]["config"] = "different_detuning"
    with pytest.raises(ValueError, match="provenance mismatch: config"):
        suite.load_reference(path, payload, "state")
