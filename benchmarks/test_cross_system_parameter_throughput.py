import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import run_cross_system_parameter_throughput as throughput


def _toy_initializer(queue):
    import os
    queue.put(os.getpid())


def _toy_ping(_):
    import os
    return os.getpid()


def test_parameter_graph_override_isolated_and_base_only():
    source = SimpleNamespace(parameter_graph={
        "slot_names": ["rabi", "detuning", "combined"],
        "n_base": 2,
        "compounds": [],
        "base_values": [
            {"kind": "scalar", "re": 1.0, "im": 0.0},
            {"kind": "scalar", "re": 0.0, "im": 0.0},
        ],
    })
    changed = throughput._prepared_with_overrides(source, {"rabi": 2.0, "detuning": -3.0})
    assert source.parameter_graph["base_values"][0]["re"] == 1.0
    assert changed.parameter_graph["base_values"][0]["re"] == 2.0
    assert changed.parameter_graph["base_values"][1]["re"] == -3.0
    from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py
    assert evaluate_parameter_graph_py(changed.parameter_graph, 0.0)[:2] == [2.0 + 0j, -3.0 + 0j]
    with pytest.raises(ValueError, match="not a base slot"):
        throughput._prepared_with_overrides(source, {"combined": 4.0})


def test_rc_override_uses_zero_omega_and_explicit_switch_phase():
    row = throughput._rc_parameter_row(
        {"rabi": 2.0, "detuning": -3.0, "microwave_rabi": 4.0},
        True, False, 1.25,
    )
    np.testing.assert_array_equal(row, [0.0, 1.25, 0.0, -1.25, 2.0, -3.0, 4.0, 4.0])


def test_forward_batch_flattens_point_initial_axes_and_sets_joint_weights(monkeypatch):
    calls = {}

    def fake_batch(prepared, rho_batch, _span, **kwargs):
        calls.update(rho_shape=rho_batch.shape, kwargs=kwargs)
        return SimpleNamespace(values=np.zeros((len(rho_batch), 5)), solver_stats={})

    monkeypatch.setattr(throughput, "solve_lindblad_batch", fake_batch)
    fixture = SimpleNamespace(
        rc_case=None,
        prepared=SimpleNamespace(layout=SimpleNamespace(packed_len=4)),
        spec=SimpleNamespace(t_span=(0.0, 1e-6), dt=1e-9),
        excited_indices=(3, 4),
    )
    rho = np.zeros((2, 5, 5), dtype=np.complex128)
    points = [{"rabi": 1.0, "detuning": 0.0, "microwave_rabi": 0.0}] * 8
    values, _wall, _stats = throughput._forward_points(
        fixture, rho, points, "full_and_integral", "structured", "tsit5", 3,
    )
    assert calls["rho_shape"] == (16, 5, 5)
    assert calls["kwargs"]["parameter_batch"].shape == (16, 2)
    assert calls["kwargs"]["integral_weights"] == [(3, throughput.GAMMA), (4, throughput.GAMMA)]
    assert values.shape == (8, 2, 5)


def test_spawn_pool_readiness_uses_initializer_records():
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with ProcessPoolExecutor(max_workers=2, mp_context=context,
                             initializer=_toy_initializer, initargs=(queue,)) as pool:
        list(pool.map(_toy_ping, range(8), chunksize=1))
        ready = {queue.get(timeout=30) for _ in range(2)}
    queue.close()
    assert len(ready) == 2
