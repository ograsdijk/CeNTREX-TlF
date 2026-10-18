"""Prepared upper updates must preserve complex drives and runtime changes."""

import copy

import numpy as np
import pytest
import sympy as sp

from centrex_tlf.lindblad.batch import grid_scan, solve_lindblad_batch
from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py, fill_hamiltonian_py
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.reference_dense import reference_rhs
from centrex_tlf.lindblad.utils_setup import OBESystem

rust = pytest.importorskip("centrex_tlf.centrex_tlf_rust")


def _make_system():
    t = sp.Symbol("t", real=True)
    a, b = sp.symbols("a b", real=True)
    wave = sp.sin(a + t) * sp.exp(sp.I * b * t)
    h = sp.Matrix(
        [
            [1.2, wave + 0.2j, 2 * wave],
            [sp.conjugate(wave) - 0.2j, -0.7, sp.conjugate(wave) + wave],
            [2 * sp.conjugate(wave), sp.conjugate(wave) + wave, 0.4],
        ]
    )
    c = np.zeros((1, 3, 3), complex)
    c[0, 0, 2] = np.sqrt(0.3)
    return OBESystem(
        ground=[],
        excited=[],
        QN=[],
        H_int=np.zeros((3, 3), complex),
        V_ref_int=np.eye(3),
        couplings=[],
        H_symbolic=h,
        C_array=c,
        system=None,
        coupling_symbols=[a, b],
        polarization_symbols=[],
    )


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize(
    "assembly,dag", [(False, False), (False, True), (True, False), (True, True)]
)
def test_prepared_complex_updates_match_dense_reference(segmented, assembly, dag):
    system = _make_system()
    rho = np.array([[0.4, 0.03 + 0.02j, -0.02j], [0.03 - 0.02j, 0.35, 0.01], [0.02j, 0.01, 0.25]])
    for av, bv in [(0.0, 0.0), (0.7, -2.1), (-0.2, 3.8)]:
        prepared = prepare_lindblad_problem(
            system,
            {"a": av, "b": bv},
            backend="python",
            hamiltonian_representation="decomposed",
            prepared_hamiltonian_assembly=assembly,
            shared_coefficient_dag=dag,
        )
        payload = copy.deepcopy(prepared.to_payload())
        assert payload["hamiltonian_plan"]["prepared_hamiltonian_assembly"] is assembly
        assert payload["hamiltonian_plan"]["shared_coefficient_dag"] is dag
        # Exercise both supported representations of basis update data.
        if not segmented:
            for coefficient in payload["hamiltonian_plan"]["coefficients"]:
                coefficient.pop("basis_row_segments", None)
        plan = rust.prepare_lindblad_problem_py(payload)
        packed = prepared.layout.pack(rho)
        for time in [0.0, 0.17, 0.83, 0.17]:
            expected_h = fill_hamiltonian_py(
                prepared.hamiltonian_plan,
                evaluate_parameter_graph_py(prepared.parameter_graph, time),
                time,
            )
            np.testing.assert_allclose(
                rust.evaluate_lindblad_hamiltonian_py(plan, time),
                expected_h,
                atol=2e-14,
                rtol=2e-14,
            )
            expected_rhs = reference_rhs(prepared, packed, time)
            for mode in [
                "reference",
                "structured_upper",
                "expanded_sparse",
                "experimental_structured_sparse_dense_diag_soa_single_unpack",
            ]:
                np.testing.assert_allclose(
                    rust.lindblad_rhs_py(plan, packed, time, mode),
                    expected_rhs,
                    atol=2e-13,
                    rtol=2e-13,
                )


@pytest.mark.parametrize(
    "assembly,dag", [(False, False), (False, True), (True, False), (True, True)]
)
def test_optimization_switches_preserve_parallel_grid_density(assembly, dag):
    system = _make_system()
    rho = np.array([[0.4, 0.03 + 0.02j, -0.02j], [0.03 - 0.02j, 0.35, 0.01], [0.02j, 0.01, 0.25]])
    options = dict(
        solver="tsit5",
        execution_mode="structured_sparse_dense_diag_soa_single_unpack",
        output="populations",
        output_when="final",
        dt=1e-3,
        reltol=1e-9,
        abstol=1e-11,
    )
    reference = prepare_lindblad_problem(
        system,
        {"a": 0.1, "b": 0.3},
        prepared_hamiltonian_assembly=False,
        shared_coefficient_dag=False,
    )
    candidate = prepare_lindblad_problem(
        system,
        {"a": 0.1, "b": 0.3},
        prepared_hamiltonian_assembly=assembly,
        shared_coefficient_dag=dag,
    )
    scan = {"a": np.array([0.0, 0.7]), "b": np.array([-0.3, 1.2])}
    before = grid_scan(reference, rho, (0.0, 0.5), scan=scan, parallel=False, **options)
    after = grid_scan(candidate, rho, (0.0, 0.5), scan=scan, parallel=True, threads=2, **options)
    np.testing.assert_allclose(after.values, before.values, atol=1e-11, rtol=1e-9)
    # Full-density batch output also checks complex coherences with overrides.
    options["output"] = "full"
    batch_options = dict(
        parameter_slots=["a", "b"],
        parameter_batch=np.array([[0.0, -0.3], [0.7, 1.2]], dtype=complex),
    )
    initial = np.array([rho, rho.conj()])
    before = solve_lindblad_batch(
        reference, initial, (0.0, 0.5), parallel=False, **batch_options, **options
    )
    after = solve_lindblad_batch(
        candidate, initial, (0.0, 0.5), parallel=True, threads=2, **batch_options, **options
    )
    np.testing.assert_allclose(after.values, before.values, atol=1e-11, rtol=1e-9)


def test_missing_payload_switches_keep_enabled_defaults():
    prepared = prepare_lindblad_problem(_make_system(), {"a": 0.2, "b": 0.3}, backend="python")
    payload = copy.deepcopy(prepared.to_payload())
    assert payload["hamiltonian_plan"].pop("prepared_hamiltonian_assembly") is True
    assert payload["hamiltonian_plan"].pop("shared_coefficient_dag") is True
    plan = rust.prepare_lindblad_problem_py(payload)
    rho = np.eye(3, dtype=complex) / 3
    np.testing.assert_allclose(
        rust.lindblad_rhs_py(plan, prepared.layout.pack(rho), 0.2, "structured_upper"),
        reference_rhs(prepared, prepared.layout.pack(rho), 0.2),
        atol=1e-13,
    )


@pytest.mark.parametrize("option", ["prepared_hamiltonian_assembly", "shared_coefficient_dag"])
def test_optimization_switch_rejects_string(option):
    with pytest.raises(TypeError, match=option + " must be a bool"):
        prepare_lindblad_problem(object(), {}, **{option: "False"})
