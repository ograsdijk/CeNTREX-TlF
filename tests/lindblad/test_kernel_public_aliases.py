from __future__ import annotations

import inspect

import numpy as np
import pytest
import sympy as smp

from centrex_tlf.lindblad.batch import grid_scan, solve_lindblad_batch
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import (
    STRUCTURED_SPARSE_DENSE_MODES,
    PreparedLindbladProblem,
    _lower_structured_jumps,
    prepare_lindblad_problem,
)
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.utils_setup import OBESystem

rust = pytest.importorskip("centrex_tlf.centrex_tlf_rust")

EXPERIMENTAL_MODES = {
    "structured_sparse_dense": "experimental_structured_sparse_dense",
    "structured_sparse_dense_diag": "experimental_structured_sparse_dense_diag",
    "structured_sparse_dense_soa": "experimental_structured_sparse_dense_soa",
    "structured_sparse_dense_single_unpack": "experimental_structured_sparse_dense_single_unpack",
    "structured_sparse_dense_diag_soa": "experimental_structured_sparse_dense_diag_soa",
    "structured_sparse_dense_diag_single_unpack": "experimental_structured_sparse_dense_diag_single_unpack",
    "structured_sparse_dense_soa_single_unpack": "experimental_structured_sparse_dense_soa_single_unpack",
    "structured_sparse_dense_diag_soa_single_unpack": "experimental_structured_sparse_dense_diag_soa_single_unpack",
}


def _tiny_prepared(representation):
    omega_symbol, delta_symbol = smp.symbols("Ω δ", real=True)
    hamiltonian = smp.Matrix([[0, omega_symbol / 2], [omega_symbol / 2, -delta_symbol]])
    collapse = np.zeros((1, 2, 2), dtype=np.complex128)
    collapse[0, 0, 1] = np.sqrt(0.3)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    system = OBESystem([], [], [], zeros, zeros, [], hamiltonian, collapse, None,
                       [omega_symbol, delta_symbol], [])
    parameters = LindbladParameters()
    omega = parameters.real("omega0", 0.65)
    delta = parameters.real("delta0", 0.12)
    parameters.bind(omega_symbol, omega)
    parameters.bind(delta_symbol, delta)
    prepared = prepare_lindblad_problem(
        system, parameters, backend="rust", hamiltonian_representation=representation
    )
    return prepared, omega, delta


@pytest.mark.parametrize("representation", ["entrywise", "decomposed"])
@pytest.mark.parametrize(("alias", "experimental"), sorted(EXPERIMENTAL_MODES.items()))
def test_public_alias_rhs_matches_its_experimental_kernel(alias, experimental, representation):
    prepared, _omega, _delta = _tiny_prepared(representation)
    rho = np.array([[0.63, 0.12 + 0.09j], [0.12 - 0.09j, 0.37]], dtype=np.complex128)
    packed = prepared.layout.pack(rho)
    native_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    alias_rhs = np.asarray(
        rust.create_lindblad_rhs_evaluator_py(native_plan, alias).rhs_packed_py(packed, 0.13),
        dtype=np.float64,
    )
    experimental_rhs = np.asarray(
        rust.create_lindblad_rhs_evaluator_py(native_plan, experimental).rhs_packed_py(packed, 0.13),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(alias_rhs, experimental_rhs)


def test_public_dsu_alias_single_batch_and_grid_surfaces_match_experimental():
    alias = "structured_sparse_dense_diag_soa_single_unpack"
    experimental = EXPERIMENTAL_MODES[alias]
    prepared, omega, delta = _tiny_prepared("decomposed")
    rho0 = np.array([[0.7, 0.08 + 0.06j], [0.08 - 0.06j, 0.3]], dtype=np.complex128)
    rho1 = np.array([[0.25, 0.03 - 0.02j], [0.03 + 0.02j, 0.75]], dtype=np.complex128)
    common = dict(solver="dopri5", dt=1e-3, reltol=1e-9, abstol=1e-11)

    single_alias = solve_lindblad(prepared, rho0, (0.0, 0.02), execution_mode=alias, **common)
    single_experimental = solve_lindblad(
        prepared, rho0, (0.0, 0.02), execution_mode=experimental, **common
    )
    np.testing.assert_array_equal(single_alias.packed_y, single_experimental.packed_y)

    batch_args = dict(
        prepared=prepared, rho0_batch=np.stack((rho0, rho1)), t_span=(0.0, 0.02),
        execution_mode=alias, output="full", output_when="final", parallel=False, **common,
    )
    batch_alias = solve_lindblad_batch(**batch_args)
    batch_args["execution_mode"] = experimental
    batch_experimental = solve_lindblad_batch(**batch_args)
    np.testing.assert_array_equal(batch_alias.values, batch_experimental.values)

    grid_args = dict(
        prepared=prepared, rho0=rho0, t_span=(0.0, 0.02),
        scan={omega: np.array([0.55, 0.7]), delta: np.array([0.12])},
        execution_mode=alias, output="populations", output_when="final",
        parallel=False, **common,
    )
    grid_alias = grid_scan(**grid_args)
    grid_args["execution_mode"] = experimental
    grid_experimental = grid_scan(**grid_args)
    np.testing.assert_array_equal(grid_alias.values, grid_experimental.values)


def test_solver_and_tolerance_defaults():
    assert inspect.signature(solve_lindblad).parameters["execution_mode"].default is None
    assert inspect.signature(solve_lindblad_batch).parameters["execution_mode"].default is None
    assert inspect.signature(solve_lindblad).parameters["solver"].default is None
    assert inspect.signature(solve_lindblad_batch).parameters["solver"].default == "dopri5"
    for function in (solve_lindblad, solve_lindblad_batch):
        defaults = inspect.signature(function).parameters
        assert defaults["reltol"].default == 1e-4
        assert defaults["abstol"].default == 1e-7


def test_cpu_ode_default_execution_mode_is_dsu():
    from centrex_tlf.lindblad.plan_static import DEFAULT_CPU_EXECUTION_MODE

    assert DEFAULT_CPU_EXECUTION_MODE == "structured_sparse_dense_diag_soa_single_unpack"
    prepared, omega, delta = _tiny_prepared("decomposed")
    rho0 = np.array([[0.7, 0.08 + 0.06j], [0.08 - 0.06j, 0.3]], dtype=np.complex128)
    common = dict(solver="dopri5", dt=1e-3, reltol=1e-9, abstol=1e-11)

    default_single = solve_lindblad(prepared, rho0, (0.0, 0.02), **common)
    dsu_single = solve_lindblad(
        prepared, rho0, (0.0, 0.02),
        execution_mode=DEFAULT_CPU_EXECUTION_MODE, **common,
    )
    np.testing.assert_array_equal(default_single.packed_y, dsu_single.packed_y)

    batch_kwargs = dict(
        prepared=prepared, rho0_batch=np.stack((rho0, rho0)), t_span=(0.0, 0.02),
        output="full", output_when="final", parallel=False, **common,
    )
    default_batch = solve_lindblad_batch(**batch_kwargs)
    dsu_batch = solve_lindblad_batch(
        **batch_kwargs, execution_mode=DEFAULT_CPU_EXECUTION_MODE,
    )
    np.testing.assert_array_equal(default_batch.values, dsu_batch.values)

    grid_kwargs = dict(
        prepared=prepared, rho0=rho0, t_span=(0.0, 0.02),
        scan={omega: np.array([0.55, 0.7]), delta: np.array([0.12])},
        output="populations", output_when="final", parallel=False, **common,
    )
    default_grid = grid_scan(**grid_kwargs)
    dsu_grid = grid_scan(**grid_kwargs, execution_mode=DEFAULT_CPU_EXECUTION_MODE)
    np.testing.assert_array_equal(default_grid.values, dsu_grid.values)


def test_cuda_ode_default_remains_expanded_sparse(monkeypatch):
    from centrex_tlf.lindblad import cuda

    prepared, _omega, _delta = _tiny_prepared("decomposed")
    monkeypatch.setattr(
        cuda,
        "_solve_cuda_single",
        lambda *args, **kwargs: kwargs["execution_mode"],
    )
    mode = solve_lindblad(
        prepared,
        np.eye(2, dtype=np.complex128) / 2,
        (0.0, 0.02),
        solver="fixed_tsit5",
        evolution_device="cuda",
    )
    assert mode == "expanded_sparse"


@pytest.mark.parametrize("mode", sorted(STRUCTURED_SPARSE_DENSE_MODES))
@pytest.mark.parametrize("hamiltonian_kind", ["entrywise", "decomposed"])
def test_public_sparse_dense_aliases_accept_supported_plan_kinds(mode, hamiltonian_kind):
    prepared = object.__new__(PreparedLindbladProblem)
    prepared.expanded_rhs_plan = None
    prepared.hamiltonian_plan = {"kind": hamiltonian_kind}
    prepared.check_execution_mode(mode)


def test_public_sparse_dense_aliases_are_the_eight_factorial_modes():
    assert {
        "structured_sparse_dense",
        "structured_sparse_dense_diag",
        "structured_sparse_dense_soa",
        "structured_sparse_dense_single_unpack",
        "structured_sparse_dense_diag_soa",
        "structured_sparse_dense_diag_single_unpack",
        "structured_sparse_dense_soa_single_unpack",
        "structured_sparse_dense_diag_soa_single_unpack",
    } == STRUCTURED_SPARSE_DENSE_MODES


def test_structured_preparation_rejects_general_collapse_operators():
    collapse = np.zeros((1, 2, 2), dtype=np.complex128)
    collapse[0, 0, 1] = 1.0
    collapse[0, 1, 0] = 1.0
    with pytest.raises(AssertionError, match="not a single-jump operator"):
        _lower_structured_jumps(collapse)
