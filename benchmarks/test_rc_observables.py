from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import run_rc_observables as observable
from bench_rc_propagation import BenchmarkConfig

from centrex_tlf import states


class _Layout:
    packed_len = 8

    def pack(self, rho):
        flat = np.asarray(rho, dtype=np.complex128).reshape(-1)
        return np.concatenate((flat.real, flat.imag))

    def unpack(self, packed):
        packed = np.asarray(packed, dtype=float)
        flat = packed[:4] + 1j * packed[4:8]
        return flat.reshape(2, 2)


def _case():
    layout = _Layout()
    return SimpleNamespace(
        n_states=2,
        prepared_dynamic=SimpleNamespace(layout=layout),
        prepared_segmented=object(),
        switch_override_slots=("ω1", "φ1", "ω2", "φ2"),
        state_labels={"X_J2_F1_5o2_F2_mF0": 0},
        system=SimpleNamespace(QN=[object(), object()]),
    )


def _groups(_case):
    return {"X_J0": [], "X_J1": [1], "X_J2": [0], "X_J3": [], "B_total": []}


def _rotate(rho, theta):
    unitary = np.diag([1.0, np.exp(1j * theta)])
    return unitary @ rho @ unitary.conj().T


def test_segmented_full_output_carries_coherence_across_switch(monkeypatch):
    case = _case()
    monkeypatch.setattr(observable.driver, "_prepare_segmented_plan", lambda _case: None)
    monkeypatch.setattr(observable.driver, "switch_boundaries", lambda _a, _b: [0.0, 1.0, 2.0])
    monkeypatch.setattr(observable.driver, "switch_values", lambda _t: (1, 0))
    monkeypatch.setattr(observable, "_group_indices", _groups)
    calls = []

    def fake_solve(_prepared, batch, span, **kwargs):
        rho = batch[0]
        calls.append((rho.copy(), span, kwargs["output"]))
        result = _rotate(rho, 0.3)
        return SimpleNamespace(values=case.prepared_dynamic.layout.pack(result)[None, :], solver_stats={})

    monkeypatch.setattr(observable, "solve_lindblad_batch", fake_solve)
    rho0 = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
    result = observable.propagate_endpoint(
        case,
        rho0,
        (0.0, 2.0),
        output_kind="full_density",
        solver="tsit5",
        reltol=1e-9,
        execution_mode="structured",
    )

    np.testing.assert_allclose(calls[1][0], _rotate(rho0, 0.3), atol=1e-14)
    np.testing.assert_allclose(result["endpoint_density"], _rotate(rho0, 0.6), atol=1e-14)
    assert abs(result["endpoint_density"][0, 1]) > 0.0


def test_selected_outputs_request_distinct_native_diagonals(monkeypatch):
    case = _case()
    monkeypatch.setattr(observable.driver, "_prepare_segmented_plan", lambda _case: None)
    monkeypatch.setattr(observable.driver, "switch_boundaries", lambda _a, _b: [0.0, 1.0])
    monkeypatch.setattr(observable.driver, "switch_values", lambda _t: (1, 1))
    monkeypatch.setattr(observable, "_group_indices", _groups)
    seen = []

    def fake_solve(_prepared, batch, _span, **kwargs):
        output = kwargs["output"]
        if output == "selected":
            indices = kwargs["output_indices"]
            seen.append(indices)
            return SimpleNamespace(values=np.asarray([[0.2 + 0j] * len(indices)]), solver_stats={})
        return SimpleNamespace(values=case.prepared_dynamic.layout.pack(batch[0])[None, :], solver_stats={})

    monkeypatch.setattr(observable, "solve_lindblad_batch", fake_solve)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    common = dict(solver="tsit5", reltol=1e-9, execution_mode="structured")
    observable.propagate_endpoint(
        case, rho0, (0.0, 1.0), output_kind="selected_j1", selected_indices=[1], **common
    )
    observable.propagate_endpoint(
        case, rho0, (0.0, 1.0), output_kind="selected_single", selected_indices=[0], **common
    )
    assert seen == [[(1, 1)], [(0, 0)]]


def test_photon_mixed_output_uses_one_solve_per_segment_and_accumulates(monkeypatch):
    case = _case()
    case.system.QN = [
        SimpleNamespace(largest=SimpleNamespace(electronic_state=states.ElectronicState.X)),
        SimpleNamespace(largest=SimpleNamespace(electronic_state=states.ElectronicState.B)),
    ]
    monkeypatch.setattr(observable.driver, "_prepare_segmented_plan", lambda _case: None)
    monkeypatch.setattr(observable.driver, "switch_boundaries", lambda _a, _b: [0.0, 1.0, 2.0])
    monkeypatch.setattr(observable.driver, "switch_values", lambda _t: (1, 0))
    monkeypatch.setattr(observable, "_group_indices", _groups)
    calls = []

    def fake_solve(_prepared, batch, _span, **kwargs):
        calls.append(kwargs["output"])
        next_rho = _rotate(batch[0], 0.1)
        packed = case.prepared_dynamic.layout.pack(next_rho)
        return SimpleNamespace(values=np.concatenate((packed, [len(calls)]))[None, :], solver_stats={})

    monkeypatch.setattr(observable, "solve_lindblad_batch", fake_solve)
    monkeypatch.setattr(observable.driver, "GAMMA", 1.0)
    result = observable.propagate_endpoint(
        case,
        np.diag([1.0, 0.0]).astype(np.complex128),
        (0.0, 2.0),
        output_kind="photons",
        solver="tsit5",
        reltol=1e-9,
        execution_mode="structured",
    )
    assert calls == ["full_and_integral", "full_and_integral"]
    assert result["photon_yield"] == 3.0


def test_reference_validation_rejects_same_duration_wrong_entry_phase():
    config = BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 200e-6, 0.0)
    metadata = {
        "status": "complete",
        "timings_accepted": True,
        "arguments": {"config": "nominal", "duration_us": 200.0, "entry_ns": 1.0, "detuning_mhz": None},
        "provenance": {
            "config": {
                "name": "nominal",
                "optical_rabi_over_gamma": 1.0,
                "microwave_rabi_over_gamma": 1.0,
                "detuning_mhz": 0.0,
                "duration_s": 200e-6,
                "entry_phase_s": 0.0,
            },
            "model": {
                "n_states": 70,
                "E_V_cm": 200.0,
                "B_G": 0.01,
                "retain_opposite_parity_levels": True,
                "qn_compact": False,
                "Jmax_X": 5,
                "Jmax_B": 3,
            },
            "amplitude_envelope": "constant",
        },
        "rows": [{"method": "dopri5", "execution_mode": "structured", "initial_label": "X_J2_F1_5o2_F2_mF0"}],
    }
    with pytest.raises(ValueError, match="entry_ns"):
        observable.validate_reference_metadata(
            metadata,
            config,
            initial_label="X_J2_F1_5o2_F2_mF0",
            expected_n_states=70,
        )


def test_reference_validation_accepts_explicit_candidate_solver():
    config = BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 200e-6, 0.0)
    metadata = {
        "status": "complete",
        "timings_accepted": True,
        "arguments": {"config": "nominal", "duration_us": 200.0, "entry_ns": 0.0, "detuning_mhz": None},
        "provenance": {
            "config": {
                "name": "nominal",
                "optical_rabi_over_gamma": 1.0,
                "microwave_rabi_over_gamma": 1.0,
                "detuning_mhz": 0.0,
                "duration_s": 200e-6,
                "entry_phase_s": 0.0,
            },
            "model": {
                "n_states": 70,
                "E_V_cm": 200.0,
                "B_G": 0.01,
                "retain_opposite_parity_levels": True,
                "qn_compact": False,
                "Jmax_X": 5,
                "Jmax_B": 3,
            },
            "amplitude_envelope": "constant",
        },
        "rows": [{
            "method": "tsit5",
            "execution_mode": "structured",
            "phase_frame": False,
            "tolerance": 1e-9,
            "initial_label": "X_J2_F1_5o2_F2_mF0",
        }],
    }
    result = observable.validate_reference_metadata(
        metadata,
        config,
        initial_label="X_J2_F1_5o2_F2_mF0",
        expected_n_states=70,
        expected_solver="tsit5",
    )
    assert result["checks"]["solver_configuration"]
