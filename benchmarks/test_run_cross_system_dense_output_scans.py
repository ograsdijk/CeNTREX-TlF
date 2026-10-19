import json
from queue import Queue
from types import SimpleNamespace

import numpy as np

from benchmarks import run_cross_system_dense_output_scans as scans
from benchmarks.run_cross_system_batch_amortization import PRODUCTS


def test_persistent_worker_prepares_once_and_accepts_one_product_per_task(monkeypatch):
    prepared, evaluated = [], []
    monkeypatch.setattr(scans, "build_fixture", lambda _name: SimpleNamespace(prepared=object()))

    class Propagator:
        def __init__(self):
            self.stats = {"factorization_seconds": 0.1}

    def fake_prepare(_prepared, _rho, **kwargs):
        prepared.append(kwargs["parameter_values"])
        return Propagator()

    def fake_eval(_fixture, _propagator, product, rho0_batch=None):
        evaluated.append(product)
        if product == "full_and_integral":
            return {"full": np.zeros((len(rho0_batch), 2, 2)),
                    "photon_integral": np.zeros((len(rho0_batch), 1))}, 0.01
        shape = (len(rho0_batch), 2, 2) if product == "full" else (len(rho0_batch), 2)
        return np.zeros(shape), 0.01

    monkeypatch.setattr(scans, "prepare_dense_lindblad_propagator", fake_prepare)
    monkeypatch.setattr(scans, "evaluate_static_dense", fake_eval)
    task_queue, result_queue = Queue(), Queue()
    rho = np.zeros((3, 2, 2))
    task_queue.put({"kind": "prepare", "point_index": 0,
                    "point": {"rabi": 1.5, "detuning": -7.0}})
    for product in PRODUCTS:
        task_queue.put({"kind": "output", "point_index": 0, "product": product})
    task_queue.put({"kind": "stop"})
    scans._output_worker("r2_f4", rho, task_queue, result_queue)
    assert len(prepared) == 1
    assert prepared[0] == {"rabi": 1.5 + 0j, "detuning": -7 + 0j}
    assert evaluated == list(PRODUCTS)
    assert [result_queue.get()["kind"] for _ in range(len(PRODUCTS) + 2)] == [
        "ready", "prepared", *("output" for _ in PRODUCTS)]


def test_reads_frozen_native_product_scan_rows_with_provenance(tmp_path):
    points = scans._points(False)
    metadata = {"state_count": 40, "packed_length": 820,
                "main_ground_label": "g", "main_excited_label": "e",
                "initial_label": "all", "native_extension_sha256": "native",
                "fixture_builder_sha256": "fixture"}
    rows = [{"batch": "K40_all_model_states", "method": "native_forward",
             "workers": 3, "product": product, "repeat": repeat,
             "point_index": point_index,
             "wall_seconds": 2.0 + repeat, "accuracy_passed": True}
            for product in PRODUCTS for repeat in range(3)
            for point_index in range(len(points))]
    (tmp_path / "result.json").write_text(json.dumps({
        "complete": True, "kernel": scans.KERNEL, "solver": scans.SOLVER,
        "rtol": scans.RTOL, "atol": scans.ATOL, "points": points,
        "fixture_metadata": metadata, "rows": rows, "runner_sha256": "runner"}),
        encoding="utf-8")
    extracted, provenance = scans._native_rows(
        tmp_path, "K40_all_model_states", 3, metadata, points)
    assert provenance["available"]
    assert [item["scan_seconds"] for item in extracted["full"]] == [2.0, 3.0, 4.0]
    assert extracted["full"][0]["unique_timing_count"] == 1


def test_native_scan_rejects_incomplete_points_or_inconsistent_duplicated_times(tmp_path):
    points = scans._points(False)
    metadata = {"state_count": 40, "packed_length": 820,
                "main_ground_label": "g", "main_excited_label": "e",
                "initial_label": "all", "native_extension_sha256": "native",
                "fixture_builder_sha256": "fixture"}
    rows = [{"batch": "K40_all_model_states", "method": "native_forward",
             "workers": 3, "product": "full", "repeat": 0,
             "point_index": i, "wall_seconds": 2.0 + (i == 7),
             "accuracy_passed": True} for i in range(len(points) - 1)]
    rows.append({"batch": "K40_all_model_states", "method": "native_forward",
                 "workers": 3, "product": "full", "repeat": 0,
                 "point_index": 7, "wall_seconds": 3.0, "accuracy_passed": True})
    (tmp_path / "result.json").write_text(json.dumps({
        "complete": True, "kernel": scans.KERNEL, "solver": scans.SOLVER,
        "rtol": scans.RTOL, "atol": scans.ATOL, "points": points,
        "fixture_metadata": metadata, "rows": rows}), encoding="utf-8")
    extracted, _ = scans._native_rows(tmp_path, "K40_all_model_states", 3, metadata, points)
    row = extracted["full"][0]
    assert row["available"] is False
    assert row["scan_seconds"] is None
    assert row["scan_timing_consistent"] is False


def test_joint_outputs_pack_as_numeric_npz_members():
    packed = scans._pack_output_values({
        2: {"rho": np.ones((3, 4, 4), dtype=np.complex128),
            "photon_integral": np.ones((3, 1))},
    })
    assert set(packed) == {"p2_rho", "p2_photon_integral"}
    assert all(array.dtype != object for array in packed.values())
