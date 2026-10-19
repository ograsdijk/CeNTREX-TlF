"""Lightweight contract tests for the density-only period-action adapter."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rc_periodic_krylov as runner


def test_period_action_carries_full_signed_coherent_density(monkeypatch) -> None:
    from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

    density_layout = PackedHermitianLayout(2)
    rho = np.array([[0.4, 0.2 + 0.3j], [0.2 - 0.3j, -0.1]], dtype=np.complex128)
    packed = density_layout.pack(rho)
    case = SimpleNamespace(
        prepared_dynamic=SimpleNamespace(layout=density_layout),
        n_states=2,
    )
    expected = np.array([[0.1, -0.25 + 0.05j], [-0.25 - 0.05j, 0.8]])
    observed: dict[str, object] = {}

    def fake_propagate(case_arg, rho_arg, span, **kwargs):
        observed["rho"] = rho_arg.copy()
        observed["span"] = span
        observed["kwargs"] = kwargs
        return {
            "endpoint_density": expected,
            "segment_solver_wall_seconds_sum": 0.125,
        }

    monkeypatch.setattr(runner.observables, "propagate_endpoint", fake_propagate)
    result, segment_wall, details = runner._run_action(
        case,
        packed,
        solver="tsit5",
        tolerance=1e-9,
        execution_mode="structured",
    )

    np.testing.assert_array_equal(observed["rho"], rho)
    assert observed["span"] == (0.0, runner.PERIOD_S)
    assert observed["kwargs"]["output_kind"] == "full_density"
    np.testing.assert_array_equal(density_layout.unpack(result), expected)
    assert segment_wall == 0.125
    assert details["endpoint_density"] is expected


def test_endpoint_loader_checks_dopri_reference_and_tsit_candidate_roles(monkeypatch) -> None:
    config = runner.driver.BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 200e-6, 0.0)

    def make_metadata(solver: str, tolerance: float) -> dict:
        return {
            "status": "complete",
            "timings_accepted": True,
            "arguments": {
                "config": "nominal",
                "duration_us": 200.0,
                "entry_ns": 0.0,
                "detuning_mhz": None,
            },
            "provenance": {
                "config": asdict(config),
                "amplitude_envelope": "constant",
                "model": {
                    "n_states": 2,
                    "E_V_cm": runner.driver.E_FIELD_V_CM,
                    "B_G": runner.driver.B_FIELD_G,
                    "retain_opposite_parity_levels": True,
                    "qn_compact": False,
                    "Jmax_X": 5,
                    "Jmax_B": 3,
                },
            },
            "rows": [
                {
                    "initial_label": runner.INITIAL_LABEL,
                    "method": solver,
                    "execution_mode": "structured",
                    "phase_frame": False,
                    "tolerance": tolerance,
                }
            ],
        }

    metadata_by_name = {
        "reference.json": make_metadata("dopri5", 1e-11),
        "candidate.json": make_metadata("tsit5", 1e-9),
    }
    density = np.zeros((2, 2, 2), dtype=np.complex128)

    class FakeArchive:
        def __enter__(self):
            return {
                "times": np.array([0.0, config.duration_s]),
                "densities": density,
            }

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        runner,
        "_artifact_paths",
        lambda path: (Path(f"{path.stem}_densities.npz"), metadata_by_name[path.name]),
    )
    monkeypatch.setattr(runner.np, "load", lambda _path: FakeArchive())
    monkeypatch.setattr(runner, "_sha256", lambda _path: "test-hash")

    reference, _ = runner.load_endpoint_artifact(
        Path("reference.json"), config, expected_n_states=2, expected_solver="dopri5"
    )
    candidate, _ = runner.load_endpoint_artifact(
        Path("candidate.json"), config, expected_n_states=2, expected_solver="tsit5"
    )
    np.testing.assert_array_equal(reference, candidate)
    with pytest.raises(ValueError, match="solver_configuration"):
        runner.load_endpoint_artifact(
            Path("candidate.json"), config, expected_n_states=2, expected_solver="dopri5"
        )
