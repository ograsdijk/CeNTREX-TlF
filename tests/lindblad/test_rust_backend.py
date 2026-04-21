from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
import sympy as smp

from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py, fill_hamiltonian_py
from centrex_tlf.lindblad.parameters import LindbladParameters, adapt_lindblad_parameters
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.reference_dense import (
    apply_dense_dissipator_reference,
    apply_structured_dissipator_reference,
    apply_structured_dissipator_reference_legacy,
    reference_rhs,
    structured_rhs,
)
from centrex_tlf.lindblad.solve import LindbladMatrixResult, LindbladResult, solve_lindblad
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout
from centrex_tlf.lindblad.utils_setup import OBESystem

rust = pytest.importorskip("centrex_tlf.centrex_tlf_rust")


def _make_two_level_system() -> OBESystem:
    Ω, δ = smp.symbols("Ω δ", real=True)
    hamiltonian = smp.Matrix(
        [
            [0, Ω / 2],
            [smp.conjugate(Ω) / 2, -δ],
        ]
    )
    c_array = np.zeros((1, 2, 2), dtype=np.complex128)
    c_array[0, 0, 1] = np.sqrt(0.3)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    return OBESystem(
        ground=[],
        excited=[],
        QN=[],
        H_int=zeros,
        V_ref_int=zeros,
        couplings=[],
        H_symbolic=hamiltonian,
        C_array=c_array,
        system=None,
        coupling_symbols=[Ω, δ],
        polarization_symbols=[],
    )


def _ground_state_density() -> np.ndarray:
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1.0
    return rho0


def test_packed_layout_roundtrip() -> None:
    layout = PackedHermitianLayout(3)
    rho = np.array(
        [
            [0.2 + 0j, 0.1 + 0.3j, -0.4j],
            [0.1 - 0.3j, 0.5 + 0j, 0.2 - 0.1j],
            [0.4j, 0.2 + 0.1j, 0.3 + 0j],
        ],
        dtype=np.complex128,
    )
    packed = layout.pack(rho)
    recovered = layout.unpack(packed)
    np.testing.assert_allclose(recovered, rho)


def test_lindblad_parameters_order_and_adapter() -> None:
    class LegacyParameters:
        _parameters = ["Ω0", "β", "ωphase", "δ"]
        _compound_vars = ["Ω"]

        Ω0 = 1.2
        β = 0.4
        ωphase = 1.7
        δ = 0.2
        Ω = "Ω0*phase_modulation(t, β, ωphase)"

    adapted = adapt_lindblad_parameters(LegacyParameters())
    assert isinstance(adapted, LindbladParameters)
    assert list(adapted.base_parameters) == ["Ω0", "β", "ωphase", "δ"]
    assert list(adapted.compound_parameters) == ["Ω"]


def test_lowered_hamiltonian_matches_python_evaluation() -> None:
    system = _make_two_level_system()
    parameters = LindbladParameters.from_kwargs(
        Ω0=1.1,
        β=0.3,
        ωphase=2.0,
        δ=0.15,
        Ω="Ω0*phase_modulation(t, β, ωphase)",
    )
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    slots = evaluate_parameter_graph_py(prepared.parameter_graph, 0.37)
    h_python = fill_hamiltonian_py(prepared.hamiltonian_plan, slots, 0.37)
    h_rust = np.asarray(
        rust.evaluate_lindblad_hamiltonian_py(
            rust.prepare_lindblad_problem_py(prepared.to_payload()),
            0.37,
        ),
        dtype=np.complex128,
    )
    np.testing.assert_allclose(h_rust, h_python)


@pytest.mark.parametrize("representation", ["entrywise", "decomposed"])
def test_hamiltonian_representations_match(representation: str) -> None:
    system = _make_two_level_system()
    parameters = LindbladParameters.from_kwargs(
        Ω0=1.1,
        β=0.3,
        ωphase=2.0,
        δ=0.15,
        Ω="Ω0*phase_modulation(t, β, ωphase)",
    )
    entrywise = prepare_lindblad_problem(
        system,
        parameters,
        backend="python",
        hamiltonian_representation="entrywise",
    )
    other = prepare_lindblad_problem(
        system,
        parameters,
        backend="python",
        hamiltonian_representation=representation,
    )
    for time in (0.0, 0.37, 0.8):
        entrywise_h = fill_hamiltonian_py(
            entrywise.hamiltonian_plan,
            evaluate_parameter_graph_py(entrywise.parameter_graph, time),
            time,
        )
        other_h = fill_hamiltonian_py(
            other.hamiltonian_plan,
            evaluate_parameter_graph_py(other.parameter_graph, time),
            time,
        )
        np.testing.assert_allclose(other_h, entrywise_h)


def test_decomposed_hamiltonian_diagnostics_present() -> None:
    system = _make_two_level_system()
    prepared = prepare_lindblad_problem(
        system,
        {"Ω": 0.9, "δ": 0.2},
        backend="python",
        hamiltonian_representation="decomposed",
    )
    diagnostics = prepared.hamiltonian_plan["diagnostics"]
    assert diagnostics["representation"] == "decomposed"
    assert diagnostics["coefficient_count"] >= 1
    assert diagnostics["basis_term_count"] >= 1
    assert "compression_ratio" in diagnostics


def test_structured_dissipator_matches_dense_reference() -> None:
    system = _make_two_level_system()
    prepared = prepare_lindblad_problem(system, {"Ω": 0.8, "δ": 0.0}, backend="python")
    rho = np.array([[0.6, 0.1 - 0.2j], [0.1 + 0.2j, 0.4]], dtype=np.complex128)
    dense = apply_dense_dissipator_reference(prepared.dense_c_array, rho)
    legacy = apply_structured_dissipator_reference_legacy(
        prepared.structured_jumps,
        prepared.source_decay_rates,
        rho,
    )
    structured = apply_structured_dissipator_reference(
        prepared.structured_jumps,
        prepared.source_decay_rates,
        rho,
    )
    np.testing.assert_allclose(legacy, dense)
    np.testing.assert_allclose(structured, dense)


@pytest.mark.parametrize(
    ("parameters", "time"),
    [
        ({"Ω": 0.9, "δ": 0.2}, 0.0),
        (
            {
                "Ω0": 1.0,
                "β": 0.5,
                "ωphase": 1.9,
                "δ": -0.1,
                "Ω": "Ω0*phase_modulation(t, β, ωphase)",
            },
            0.41,
        ),
    ],
)
def test_rust_rhs_matches_python_reference(parameters: dict[str, object], time: float) -> None:
    system = _make_two_level_system()
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    packed = prepared.layout.pack(np.array([[0.8, 0.05 + 0.04j], [0.05 - 0.04j, 0.2]], dtype=np.complex128))
    rhs_python = reference_rhs(prepared, packed, time)
    rhs_reference = np.asarray(rust.lindblad_rhs_py(rust_plan, packed, time, "reference"), dtype=np.float64)
    rhs_structured = np.asarray(rust.lindblad_rhs_py(rust_plan, packed, time, "structured"), dtype=np.float64)
    np.testing.assert_allclose(rhs_reference, rhs_python, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(rhs_structured, rhs_python, atol=1e-11, rtol=1e-11)


def test_python_structured_rhs_matches_dense_reference() -> None:
    system = _make_two_level_system()
    prepared = prepare_lindblad_problem(
        system,
        {
            "Ω0": 1.0,
            "β": 0.25,
            "ωphase": 1.7,
            "δ": 0.05,
            "Ω": "Ω0*phase_modulation(t, β, ωphase)",
        },
        backend="python",
    )
    packed = prepared.layout.pack(
        np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], dtype=np.complex128)
    )
    rhs_dense = reference_rhs(prepared, packed, 0.31)
    rhs_structured = structured_rhs(prepared, packed, 0.31)
    np.testing.assert_allclose(rhs_structured, rhs_dense, atol=1e-11, rtol=1e-11)


def test_rust_matrix_rhs_evaluator_matches_packed_rhs() -> None:
    system = _make_two_level_system()
    prepared = prepare_lindblad_problem(
        system,
        {
            "Ω0": 1.0,
            "β": 0.25,
            "ωphase": 1.7,
            "δ": 0.05,
            "Ω": "Ω0*phase_modulation(t, β, ωphase)",
        },
        backend="python",
    )
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    evaluator = rust.create_lindblad_rhs_evaluator_py(rust_plan, "structured")
    rho = np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], dtype=np.complex128)
    packed = prepared.layout.pack(rho)
    rhs_packed = np.asarray(rust.lindblad_rhs_py(rust_plan, packed, 0.31, "structured"), dtype=np.float64)
    rhs_packed_upper = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, 0.31, "structured_upper"), dtype=np.float64
    )
    rhs_matrix = np.asarray(evaluator.rhs_matrix_py(rho.reshape(-1), 0.31), dtype=np.complex128).reshape(2, 2)
    np.testing.assert_allclose(rhs_packed_upper, rhs_packed, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(prepared.layout.pack(rhs_matrix), rhs_packed, atol=1e-11, rtol=1e-11)


def test_rust_rhs_evaluator_profile_summary_tracks_calls() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.8, str(system.coupling_symbols[1]): 0.05}
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    evaluator = rust.create_lindblad_rhs_evaluator_py(rust_plan, "structured")
    evaluator.reset_profile_py()
    evaluator.enable_profile_py(True)
    rho = np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], dtype=np.complex128)
    packed = prepared.layout.pack(rho)
    evaluator.rhs_matrix_py(rho.reshape(-1), 0.1)
    evaluator.rhs_packed_py(packed, 0.2)
    summary = evaluator.profile_summary_py()
    assert summary["enabled"] is True
    assert summary["calls"] == 2
    assert summary["total_seconds"] >= 0.0
    assert summary["parameter_eval_seconds"] >= 0.0
    assert summary["hamiltonian_fill_seconds"] >= 0.0
    assert summary["commutator_seconds"] >= 0.0
    assert summary["dissipator_seconds"] >= 0.0
    assert summary["unpack_seconds"] >= 0.0
    assert summary["pack_seconds"] >= 0.0


def test_rust_split_rhs_and_jacobian_match_matrix_rhs() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.8, str(system.coupling_symbols[1]): 0.05}
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    evaluator = rust.create_lindblad_rhs_evaluator_py(rust_plan, "structured")
    rho = np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], dtype=np.complex128)
    flat = rho.reshape(-1)
    split = np.concatenate((flat.real, flat.imag))

    rhs_matrix = np.asarray(evaluator.rhs_matrix_py(flat, 0.2), dtype=np.complex128)
    rhs_split = np.asarray(evaluator.rhs_split_py(split, 0.2), dtype=np.float64)
    rhs_from_split = rhs_split[: flat.size] + 1j * rhs_split[flat.size :]
    np.testing.assert_allclose(rhs_from_split, rhs_matrix, atol=1e-11, rtol=1e-11)

    rows, cols, values = evaluator.jacobian_split_sparse_py(0.2)
    jac = scipy.sparse.csc_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=(2 * flat.size, 2 * flat.size),
    )
    basis = np.zeros(2 * flat.size, dtype=np.float64)
    basis[1] = 1.0
    jv = np.asarray(jac @ basis).reshape(-1)
    expected = np.asarray(evaluator.rhs_split_py(basis, 0.2), dtype=np.float64)
    np.testing.assert_allclose(jv, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize(
    "parameters",
    [
        {"Ω": 0.6, "δ": 0.0},
        {
            "Ω0": 0.75,
            "β": 0.4,
            "ωphase": 2.5,
            "δ": 0.1,
            "Ω": "Ω0*phase_modulation(t, β, ωphase)",
        },
    ],
)
def test_rust_solver_matches_python_reference(parameters: dict[str, object]) -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    saveat = np.linspace(0.0, 0.5, 11)

    python_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="python",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    rust_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )

    np.testing.assert_allclose(rust_result.t, python_result.t, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(rust_result.packed_y, python_result.packed_y, atol=5e-7, rtol=5e-6)
    populations = rust_result.populations()
    np.testing.assert_allclose(np.sum(populations, axis=1), 1.0, atol=2e-6)


def test_rust_dopri5_solver_matches_python_reference() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    saveat = np.linspace(0.0, 0.5, 11)
    parameters = {
        "Ω0": 0.75,
        "β": 0.4,
        "ωphase": 2.5,
        "δ": 0.1,
        "Ω": "Ω0*phase_modulation(t, β, ωphase)",
    }

    python_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="python",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    rust_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )

    np.testing.assert_allclose(rust_result.t, python_result.t, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(rust_result.packed_y, python_result.packed_y, atol=1e-7, rtol=2e-6)
    populations = rust_result.populations()
    np.testing.assert_allclose(np.sum(populations, axis=1), 1.0, atol=2e-6)


def test_rust_scipy_solver_matches_python_structured_reference() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    saveat = np.linspace(0.0, 0.5, 11)
    parameters = {
        "Ω0": 0.75,
        "β": 0.4,
        "ωphase": 2.5,
        "δ": 0.1,
        "Ω": "Ω0*phase_modulation(t, β, ωphase)",
    }

    python_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="python",
        execution_mode="structured",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    rust_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="scipy",
        execution_mode="structured",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )

    assert isinstance(rust_result, LindbladMatrixResult)
    np.testing.assert_allclose(rust_result.t, python_result.t, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        rust_result.density_matrices(),
        python_result.density_matrices(),
        atol=5e-7,
        rtol=5e-6,
    )
    np.testing.assert_allclose(rust_result.packed_y, python_result.packed_y, atol=5e-7, rtol=5e-6)


def test_rust_scipy_bdf_solver_matches_python_reference() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    saveat = np.linspace(0.0, 0.5, 11)
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}

    python_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="python",
        execution_mode="structured",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    rust_result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="scipy_bdf",
        execution_mode="structured",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )

    assert isinstance(rust_result, LindbladResult)
    np.testing.assert_allclose(rust_result.t, python_result.t, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        rust_result.density_matrices(),
        python_result.density_matrices(),
        atol=5e-7,
        rtol=5e-6,
    )


def test_tuple_helper_expression_matches_python_reference() -> None:
    system = _make_two_level_system()
    parameters = {
        "x": 0.15,
        "y": -0.2,
        "amplitudes": (1.0, 0.6),
        "xlocs": (-0.3, 0.4),
        "ylocs": (0.25, -0.1),
        "sigma_x": 0.7,
        "sigma_y": 0.9,
        "coupling": 0.35,
        "δ": 0.15,
        "Ω": "multipass_2d_rabi(x, y, amplitudes, xlocs, ylocs, sigma_x, sigma_y, coupling)",
    }
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())

    slots = evaluate_parameter_graph_py(prepared.parameter_graph, 0.0)
    h_python = fill_hamiltonian_py(prepared.hamiltonian_plan, slots, 0.0)
    h_rust = np.asarray(
        rust.evaluate_lindblad_hamiltonian_py(
            rust_plan,
            0.0,
        ),
        dtype=np.complex128,
    )

    np.testing.assert_allclose(h_rust, h_python)
