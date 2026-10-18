from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

sys.path.insert(0, str(Path(__file__).resolve().parent))
pilot = importlib.import_module("rc_periodic_pilot")


def test_periodic_basis_probe_preserves_linear_coherence_inputs_and_cost_gate(monkeypatch):
    layout = PackedHermitianLayout(4)
    qn = [object() for _ in range(4)]
    case = SimpleNamespace(
        prepared_dynamic=SimpleNamespace(layout=layout),
        amplitude_envelope="constant",
        state_labels={"X_J1_F1_3o2_F1_mF0": 0, "X_J2_F1_5o2_F2_mF0": 1, "X_J3_F1_7o2_F3_mF0": 2},
        system=SimpleNamespace(QN=qn, couplings=[SimpleNamespace(excited_main=qn[3])]),
    )
    calls = []

    def propagate(_case, columns, *, reltol, threads):
        matrices = np.stack([layout.unpack(column) for column in columns])
        np.testing.assert_allclose(matrices, matrices.swapaxes(-1, -2).conj())
        assert len(set(np.flatnonzero(columns).tolist())) == 8
        assert np.linalg.eigvalsh(matrices[4]).min() < 0
        calls.append((reltol, threads))
        stats = [{"full_density_wall_seconds": 60.0, "photon_integral_wall_seconds": 40.0}]
        return np.array([0.0, 5e-6]), matrices, np.zeros(8), stats, 100.0

    monkeypatch.setattr(pilot.driver, "_prepare_segmented_plan", lambda _case: None)
    monkeypatch.setattr(pilot, "_propagate_chunk", propagate)
    result = pilot.run_periodic_pilot(case, budget_seconds=10, refinement_probe_limit_seconds=0)
    assert calls == [(1e-9, 8)]
    assert result["no_map_constructed"]
    assert not result["accuracy_claim"]
    assert not result["full_layout_gate"]["feasible_within_projected_action_budget"]
    assert not result["tentative_common_support"]["applicable_dimension_match"]
