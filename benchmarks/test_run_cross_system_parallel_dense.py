from queue import Queue
from types import SimpleNamespace

import numpy as np

from benchmarks import run_cross_system_parallel_dense as dense
from benchmarks.run_cross_system_parallel_dense import (
    TreeMemorySampler,
    _as_product,
    _expected_points,
)


def test_expected_products_cover_populations_and_joint_observables():
    rho = np.zeros((2, 1, 3, 3), dtype=np.complex128)
    rho[:, 0, 0, 0] = [0.2, 0.4]
    rho[:, 0, 1, 1] = [0.3, 0.2]
    rho[:, 0, 2, 2] = [0.5, 0.4]
    fixture = SimpleNamespace(
        selected_indices_1=(0,), selected_indices_4=(0, 1, 2), group_indices=(1, 2))
    products = _expected_points(fixture, {"full": rho, "photon": np.array([[0.1], [0.2]])})
    np.testing.assert_array_equal(products["populations"], np.real(np.diagonal(rho, axis1=2, axis2=3)))
    np.testing.assert_allclose(products["group_sum"][:, 0, 0], [0.8, 0.6])
    np.testing.assert_array_equal(products["full_and_integral"]["photon_integral"], [[[0.1]], [[0.2]]])


def test_tree_memory_sampler_reports_start_peak_and_interval():
    with TreeMemorySampler(interval_s=0.01) as sampler:
        _ = bytearray(1024 * 32)
    summary = sampler.summary()
    assert summary["sample_interval_seconds"] == 0.01
    assert summary["sample_count"] >= 1
    assert summary["tree_rss_start_bytes"] is not None
    assert summary["tree_rss_peak_bytes"] >= summary["tree_rss_start_bytes"]


def test_dense_point_preparation_passes_explicit_runtime_overrides(monkeypatch):
    calls = {}

    class FakePropagator:
        def __init__(self):
            self.stats = {"factorization_seconds": 0.25}

    def fake_prepare(prepared, rho_batch, **kwargs):
        calls.update(prepared=prepared, rho_shape=rho_batch.shape, kwargs=kwargs)
        return FakePropagator()

    monkeypatch.setattr(dense, "prepare_dense_lindblad_propagator", fake_prepare)
    prepared = object()
    fixture = SimpleNamespace(prepared=prepared)
    rho = np.zeros((2, 3, 3))
    point = {"rabi": 4.0, "detuning": -5.0}
    propagator, setup = dense._prepare_point(fixture, rho, point)
    assert propagator is not None
    assert calls["kwargs"]["parameter_values"] == {"rabi": 4 + 0j, "detuning": -5 + 0j}
    assert setup["initial_count_prepared"] == 2


def test_forward_full_and_joint_outputs_unpack_and_preserve_point_batch_axes():
    layout = SimpleNamespace(packed_len=4, unpack=lambda row: np.asarray(row).reshape(2, 2))
    fixture = SimpleNamespace(prepared=SimpleNamespace(layout=layout))
    packed = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    full = _as_product("full", packed, fixture)
    assert full.shape == (2, 3, 2, 2)
    joint_packed = np.concatenate((packed, np.ones((2, 3, 1))), axis=-1)
    joint = _as_product("full_and_integral", joint_packed, fixture)
    assert joint["rho"].shape == (2, 3, 2, 2)
    assert joint["photon_integral"].shape == (2, 3, 1)


def test_all_products_cover_every_parameter_point_at_both_power_corners(monkeypatch, tmp_path):
    points = dense._points(False)
    calls = []
    n, packed_len, k = 3, 9, 2
    layout = SimpleNamespace(n=n, packed_len=packed_len,
                             unpack=lambda row: np.asarray(row).reshape(n, n))
    fixture = SimpleNamespace(prepared=SimpleNamespace(layout=layout))
    rho_batch = np.zeros((k, n, n), dtype=np.complex128)

    def fake_forward(_fixture, _rho, sent_points, product, *_args, **_kwargs):
        calls.append((product, [p["detuning_mhz"] for p in sent_points],
                      [p["optical_rabi_over_gamma"] for p in sent_points]))
        width = packed_len + 1 if product == "full_and_integral" else packed_len if product == "full" else n
        return np.zeros((len(sent_points), k, width)), 0.01, {}

    monkeypatch.setattr(dense, "_forward_points", fake_forward)
    result = dense._forward_sweeps(fixture, rho_batch, points, "synthetic", 3, tmp_path, [])
    assert len(calls) == 3 * len(dense.PRODUCTS)
    assert {product for product, *_ in calls} == set(dense.PRODUCTS)
    assert any(delta != 0.0 for _product, deltas, _powers in calls for delta in deltas)
    assert any(power == 0.3 for _product, _deltas, powers in calls for power in powers)
    assert any(power == 3.0 for _product, _deltas, powers in calls for power in powers)
    assert result["outputs"]["full"][0].shape == (len(points), k, n, n)
    assert result["outputs"]["full_and_integral"][0]["rho"].shape == (len(points), k, n, n)
    assert result["outputs"]["full_and_integral"][0]["photon_integral"].shape == (len(points), k, 1)


def test_dense_worker_reuses_one_parameter_factorization_for_all_products(monkeypatch):
    prepare_calls, evaluate_calls = [], []

    class FakePropagator:
        stats = None

        def __init__(self):
            self.stats = {"factorization_seconds": 0.02}

    monkeypatch.setattr(dense, "build_fixture", lambda _name: SimpleNamespace(prepared=object()))

    def fake_prepare(prepared, rho_batch, **kwargs):
        prepare_calls.append(kwargs["parameter_values"])
        return FakePropagator()

    def fake_evaluate(_fixture, _propagator, product, rho0_batch=None):
        evaluate_calls.append(product)
        if product == "full_and_integral":
            return {"full": np.zeros((len(rho0_batch), 2, 2)),
                    "photon_integral": np.zeros((len(rho0_batch), 1))}, 0.01
        shape = (len(rho0_batch), 2, 2) if product == "full" else (len(rho0_batch), 2)
        return np.zeros(shape), 0.01

    monkeypatch.setattr(dense, "prepare_dense_lindblad_propagator", fake_prepare)
    monkeypatch.setattr(dense, "evaluate_static_dense", fake_evaluate)
    task_q, result_q = Queue(), Queue()
    rho = np.zeros((2, 2, 2))
    point = {"rabi": 2.0, "detuning": -1.0}
    task_q.put({"kind": "cold", "point_index": 5, "point": point})
    task_q.put({"kind": "warm", "point_index": 5, "repeat": 1})
    task_q.put({"kind": "stop"})
    dense._dense_worker("r2_f4", rho, task_q, result_q)
    messages = [result_q.get() for _ in range(3)]
    assert len(prepare_calls) == 1
    assert prepare_calls[0] == {"rabi": 2 + 0j, "detuning": -1 + 0j}
    assert evaluate_calls == list(dense.PRODUCTS) * 2
    assert [message["kind"] for message in messages] == ["ready", "cold", "warm"]
