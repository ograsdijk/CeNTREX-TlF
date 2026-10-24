"""End-to-end tests for the Rust effective-Lindblad scans.

A one-state synthetic plan keeps these fast and exact: the optical operator is
the identity on the split-real state, so ``dy/dt = (Ω/2) y`` and the final
population is ``exp(∫ Ω/2 dt)``.

``rust_plan.grid_scan`` / ``parameter_scan`` used to pass their arguments to
Rust positionally, one slot off, and fail with a TypeError. Behind that, the
Rust RHS froze the Rabi rate and detuning at their base values, so every row of
a scan returned the same result, and a Rabi rate that depended on time only
through a chain of compound parameters was frozen at t = 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from centrex_tlf.effective_hamiltonian import rust_plan
from centrex_tlf.lindblad.ir import lower_parameter_graph
from centrex_tlf.lindblad.parameters import LindbladParameters

rust = pytest.importorskip("centrex_tlf.centrex_tlf_rust")


def _identity_operator(value: float) -> dict:
    # dim-2 split-real state (one complex amplitude), diagonal, constant over the grid
    return {
        "row_ptrs": np.array([0, 1, 2], dtype=np.int64),
        "col_indices": np.array([0, 1], dtype=np.int64),
        "pchip_coeffs": np.array([value, 0, 0, 0, value, 0, 0, 0], dtype=np.float64),
        "grid": np.array([0.0, 1.0]),
        "nnz": 2,
    }


def _synthetic_plan(params: LindbladParameters):
    graph = lower_parameter_graph(params)
    slots = graph["slot_names"]
    return rust.prepare_effective_lindblad_plan_py(
        {
            "n_states": 1,
            "real_dim": 2,
            "n_grid": 2,
            "field_grid": np.array([0.0, 1.0]),
            "sparse_combined": _identity_operator(0.0),
            "sparse_opt": _identity_operator(1.0),
            "sparse_det": _identity_operator(0.0),
            "excited_indices": np.array([], dtype=np.int64),
            "ground_indices": np.array([0], dtype=np.int64),
            "sink_indices": np.array([], dtype=np.int64),
            "field_coordinate_slot": slots.index("Ez"),
            "rabi_rate_slot": slots.index("Ω0"),
            "detuning_slot": slots.index("δ0"),
            "is_time_dependent": True,
            "operator_interpolation": "linear",
            "parameter_graph": graph,
        }
    )


def _base_parameters() -> LindbladParameters:
    params = LindbladParameters()
    params.real("Ez", 0.5)
    params.real("Ω0", 1.0)
    params.real("δ0", 0.0)
    params._finalize()
    return params


RHO0 = np.array([[1.0 + 0j]])
RABI = np.array([0.0, 1.0, 2.0])


def test_grid_scan_applies_each_rabi_rate() -> None:
    plan = _synthetic_plan(_base_parameters())
    result = rust_plan.grid_scan(
        plan, RHO0, (0.0, 1.0), scan={"Ω0": RABI}, abstol=1e-12, reltol=1e-10, dt=1e-3
    )
    final = np.asarray(result.values).reshape(RABI.size, -1)[:, 0]
    np.testing.assert_allclose(final, np.exp(0.5 * RABI), rtol=1e-8)


def test_parameter_scan_matches_single_solves() -> None:
    plan = _synthetic_plan(_base_parameters())
    result = rust_plan.parameter_scan(
        plan,
        RHO0,
        (0.0, 1.0),
        parameter_slots=["Ω0"],
        parameter_batch=RABI.reshape(-1, 1),
        abstol=1e-12,
        reltol=1e-10,
        dt=1e-3,
    )
    final = np.asarray(result.values).reshape(RABI.size, -1)[:, 0]
    for rabi, value in zip(RABI, final):
        single = _synthetic_plan(
            LindbladParameters.from_kwargs(Ez=0.5, **{"Ω0": float(rabi), "δ0": 0.0})
        )
        t, y = _solve_single(single)
        assert value == pytest.approx(y, rel=1e-8)


def _solve_single(plan) -> tuple[float, float]:
    result = rust_plan.solve_effective_lindblad(
        plan, RHO0, (0.0, 1.0), saveat=np.array([1.0]), abstol=1e-12, reltol=1e-10, dt=1e-3
    )
    return float(result.t[-1]), float(result.populations()[-1, 0])


def test_rabi_rate_through_compound_chain_follows_time() -> None:
    # Ω0 = A, A = 2 B, B = t: Ω0 depends on time only through two compounds.
    params = LindbladParameters.from_kwargs(Ez=0.5, **{"δ0": 0.0, "B": "t", "A": "2*B", "Ω0": "A"})
    plan = _synthetic_plan(params)
    _, population = _solve_single(plan)
    # dy/dt = (2t / 2) y  ->  y(1) = exp(1/2)
    assert population == pytest.approx(np.exp(0.5), rel=1e-8)


def test_malformed_sparse_operator_raises_value_error() -> None:
    params = _base_parameters()
    graph = lower_parameter_graph(params)
    bad = _identity_operator(1.0)
    bad["col_indices"] = np.array([0, 5], dtype=np.int64)
    payload = {
        "n_states": 1,
        "real_dim": 2,
        "n_grid": 2,
        "field_grid": np.array([0.0, 1.0]),
        "sparse_combined": _identity_operator(0.0),
        "sparse_opt": bad,
        "sparse_det": _identity_operator(0.0),
        "excited_indices": np.array([], dtype=np.int64),
        "ground_indices": np.array([0], dtype=np.int64),
        "sink_indices": np.array([], dtype=np.int64),
        "field_coordinate_slot": graph["slot_names"].index("Ez"),
        "rabi_rate_slot": graph["slot_names"].index("Ω0"),
        "detuning_slot": graph["slot_names"].index("δ0"),
        "is_time_dependent": False,
        "operator_interpolation": "linear",
        "parameter_graph": graph,
    }
    with pytest.raises(ValueError, match="column"):
        rust.prepare_effective_lindblad_plan_py(payload)
    payload["sparse_opt"] = _identity_operator(1.0)
    payload["ground_indices"] = np.array([-1], dtype=np.int64)
    with pytest.raises(ValueError, match="negative"):
        rust.prepare_effective_lindblad_plan_py(payload)
