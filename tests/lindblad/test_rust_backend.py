from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
import sympy as smp

from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py, fill_hamiltonian_py
from centrex_tlf.lindblad.batch import grid_scan, solve_lindblad_batch
from centrex_tlf.lindblad.parameters import (
    LindbladParameters,
    Time,
    adapt_lindblad_parameters,
    gaussian,
    sine,
    tabulated,
)
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.reference_dense import (
    apply_dense_dissipator_reference,
    apply_structured_dissipator_reference,
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


def test_typed_lindblad_parameters_lower_and_scan() -> None:
    system = _make_two_level_system()
    omega_symbol, delta_symbol = system.coupling_symbols
    params = LindbladParameters()
    omega = params.real("omega0", 0.6)
    delta = params.real("delta0", 0.0)
    params.bind(omega_symbol, omega)
    params.bind(delta_symbol, delta)

    prepared = prepare_lindblad_problem(
        system,
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    assert prepared.parameter_graph["slot_names"][:2] == ["omega0", "delta0"]

    batch = grid_scan(
        prepared,
        _ground_state_density(),
        (0.0, 0.5),
        scan={
            omega: np.array([0.4, 0.7]),
            delta: np.array([-0.1, 0.2]),
        },
        solver="dopri5_fast",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    assert batch.parameter_slots == ["omega0", "delta0"]
    assert batch.metadata["grid_shape"] == (2, 2)
    assert set(batch.metadata["grid_axes"]) == {"omega0", "delta0"}


def test_typed_runtime_expression_helpers_match_python_evaluation() -> None:
    system = _make_two_level_system()
    omega_symbol, delta_symbol = system.coupling_symbols
    params = LindbladParameters()
    omega0 = params.real("omega0", 0.9)
    z0 = params.real("z0", -0.1)
    vz = params.real("vz", 0.8)
    sigma_z = params.real("sigma_z", 0.4)
    detuning_offset = params.real("detuning_offset", 0.05)
    detuning_mod = params.real("detuning_mod", 0.02)
    detuning_omega = params.real("detuning_omega", 1.7)
    field_grid = params.real("field_grid", [-1.0, 0.0, 1.0])
    field_values = params.real("field_values", [0.5, 1.0, 0.25])
    t = Time()
    z = z0 + vz * t
    rabi_profile = (
        omega0
        * gaussian(z, center=0.0, sigma=sigma_z)
        * tabulated(z, field_grid, field_values)
    )
    detuning = sine(
        t,
        offset=detuning_offset,
        amplitude=detuning_mod,
        angular_frequency=detuning_omega,
    )
    params.bind(omega_symbol, rabi_profile)
    params.bind(delta_symbol, detuning)

    prepared = prepare_lindblad_problem(system, params, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    time = 0.37
    slots = evaluate_parameter_graph_py(prepared.parameter_graph, time)
    h_python = fill_hamiltonian_py(prepared.hamiltonian_plan, slots, time)
    h_rust = np.asarray(
        rust.evaluate_lindblad_hamiltonian_py(rust_plan, time),
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
    structured = apply_structured_dissipator_reference(
        prepared.structured_jumps,
        prepared.source_decay_rates,
        rho,
    )
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


@pytest.mark.parametrize(("time_dependent", "time"), [(False, 0.0), (True, 0.41)])
def test_expanded_sparse_rhs_matches_structured_upper(
    time_dependent: bool,
    time: float,
) -> None:
    system = _make_two_level_system()
    if time_dependent:
        parameters: dict[str, object] = {
            "omega0": 1.0,
            "modulation_depth": 0.5,
            "modulation_frequency": 1.9,
            str(system.coupling_symbols[1]): -0.1,
            str(system.coupling_symbols[0]): (
                "omega0*phase_modulation(t, modulation_depth, modulation_frequency)"
            ),
        }
    else:
        parameters = {
            str(system.coupling_symbols[0]): 0.9,
            str(system.coupling_symbols[1]): 0.2,
        }
    prepared = prepare_lindblad_problem(
        system,
        parameters,
        backend="python",
        hamiltonian_representation="decomposed",
    )
    assert prepared.expanded_rhs_plan is not None
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    packed = prepared.layout.pack(
        np.array([[0.8, 0.05 + 0.04j], [0.05 - 0.04j, 0.2]], dtype=np.complex128)
    )
    rhs_upper = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, time, "structured_upper"),
        dtype=np.float64,
    )
    rhs_expanded = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, time, "expanded_sparse"),
        dtype=np.float64,
    )
    np.testing.assert_allclose(rhs_expanded, rhs_upper, atol=1e-11, rtol=1e-11)


def test_expanded_sparse_matrix_evaluator_matches_packed_rhs() -> None:
    system = _make_two_level_system()
    parameters = {
        "omega0": 1.0,
        "modulation_depth": 0.25,
        "modulation_frequency": 1.7,
        str(system.coupling_symbols[1]): 0.05,
        str(system.coupling_symbols[0]): (
            "omega0*phase_modulation(t, modulation_depth, modulation_frequency)"
        ),
    }
    prepared = prepare_lindblad_problem(
        system,
        parameters,
        backend="python",
        hamiltonian_representation="decomposed",
    )
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    evaluator = rust.create_lindblad_rhs_evaluator_py(rust_plan, "expanded_sparse")
    rho = np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], dtype=np.complex128)
    packed = prepared.layout.pack(rho)
    rhs_packed = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, 0.31, "expanded_sparse"),
        dtype=np.float64,
    )
    rhs_matrix = np.asarray(
        evaluator.rhs_matrix_py(rho.reshape(-1), 0.31),
        dtype=np.complex128,
    ).reshape(2, 2)
    np.testing.assert_allclose(prepared.layout.pack(rhs_matrix), rhs_packed, atol=1e-11, rtol=1e-11)


def test_expanded_sparse_dopri5_solver_matches_structured_upper() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    saveat = np.linspace(0.0, 0.5, 11)
    parameters = {
        "omega0": 0.75,
        "modulation_depth": 0.4,
        "modulation_frequency": 2.5,
        str(system.coupling_symbols[1]): 0.1,
        str(system.coupling_symbols[0]): (
            "omega0*phase_modulation(t, modulation_depth, modulation_frequency)"
        ),
    }
    prepared = prepare_lindblad_problem(
        system,
        parameters,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    upper_result = solve_lindblad(
        prepared,
        rho0,
        (0.0, 0.5),
        solver="dopri5",
        execution_mode="structured_upper",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    expanded_result = solve_lindblad(
        prepared,
        rho0,
        (0.0, 0.5),
        solver="dopri5",
        execution_mode="expanded_sparse",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    np.testing.assert_allclose(expanded_result.t, upper_result.t, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        expanded_result.packed_y,
        upper_result.packed_y,
        atol=1e-10,
        rtol=1e-8,
    )


def test_rust_dopri5_solver_stats_are_reported() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        execution_mode="structured_upper",
        saveat=np.linspace(0.0, 0.5, 5),
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        collect_stats=True,
    )
    assert result.solver_stats is not None
    stats = result.solver_stats
    assert stats["solver"] == "dopri5"
    assert stats["rhs_calls"] > 0
    assert stats["function_evaluations"] >= stats["rhs_calls"]
    assert stats["accepted_steps"] > 0
    assert stats["rejected_steps"] >= 0
    assert stats["saved_points"] == result.t.size
    assert stats["rhs_seconds"] > 0.0
    assert stats["total_seconds"] >= stats["rhs_seconds"]


@pytest.mark.parametrize(
    "saveat",
    [
        np.linspace(0.0, 0.5, 9),
        np.array([0.0, 0.03, 0.11, 0.2, 0.37, 0.5], dtype=np.float64),
    ],
)
@pytest.mark.parametrize("execution_mode", ["structured_upper", "expanded_sparse"])
def test_rust_dopri5_fast_matches_dopri5(saveat: np.ndarray, execution_mode: str) -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    reference = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        execution_mode=execution_mode,
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    fast = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5_fast",
        execution_mode=execution_mode,
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        collect_stats=True,
    )
    np.testing.assert_allclose(fast.t, reference.t, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(fast.packed_y, reference.packed_y, atol=5e-10, rtol=5e-8)
    assert fast.solver_stats is not None
    assert fast.solver_stats["solver"] == "dopri5_fast"
    assert fast.solver_stats["rhs_calls"] > 0


def test_rust_dopri5_fast_population_outputs_match_full() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    saveat = np.linspace(0.0, 0.5, 7)
    full = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5_fast",
        execution_mode="expanded_sparse",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    populations = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5_fast",
        execution_mode="expanded_sparse",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        output="populations",
        collect_stats=True,
    )
    np.testing.assert_allclose(populations.t, full.t, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(populations.values, full.populations(), atol=1e-12, rtol=1e-10)
    assert populations.solver_stats is not None

    final = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5_fast",
        execution_mode="expanded_sparse",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        output="populations",
        output_when="final",
        dense_output=False,
    )
    np.testing.assert_allclose(final.t, [0.5], atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(final.values, full.populations()[-1], atol=1e-12, rtol=1e-10)


def test_rust_dopri5_fast_selected_outputs_match_full() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    saveat = np.array([0.0, 0.03, 0.11, 0.2, 0.37, 0.5], dtype=np.float64)
    full = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5_fast",
        execution_mode="structured_upper",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    selected_indices = [(0, 0), (0, 1), (1, 0)]
    selected = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5_fast",
        execution_mode="structured_upper",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        output="selected",
        output_indices=selected_indices,
    )
    matrices = full.density_matrices()
    expected = np.array([[matrix[i, j] for i, j in selected_indices] for matrix in matrices])
    np.testing.assert_allclose(selected.t, full.t, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(selected.values, expected, atol=1e-12, rtol=1e-10)

    final = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5_fast",
        execution_mode="structured_upper",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        output="selected",
        output_indices=selected_indices,
        output_when="final",
        dense_output=False,
    )
    np.testing.assert_allclose(final.values, expected[-1], atol=1e-12, rtol=1e-10)


def test_rust_dopri5_fast_dense_output_false_rejects_interior_saveat() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    with pytest.raises(ValueError, match="dense_output=False"):
        solve_lindblad(
            system,
            rho0,
            (0.0, 0.5),
            parameters=parameters,
            backend="rust",
            solver="dopri5_fast",
            execution_mode="expanded_sparse",
            saveat=np.linspace(0.0, 0.5, 7),
            dt=1e-3,
            reltol=1e-8,
            abstol=1e-10,
            dense_output=False,
        )


@pytest.mark.parametrize(
    "saveat",
    [
        np.linspace(0.0, 0.5, 9),
        np.array([0.0, 0.03, 0.11, 0.2, 0.37, 0.5], dtype=np.float64),
    ],
)
@pytest.mark.parametrize("execution_mode", ["structured_upper", "expanded_sparse"])
def test_rust_tsit5_fast_matches_dopri5(saveat: np.ndarray, execution_mode: str) -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    reference = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        execution_mode=execution_mode,
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    fast = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="tsit5_fast",
        execution_mode=execution_mode,
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        collect_stats=True,
    )
    np.testing.assert_allclose(fast.t, reference.t, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(fast.packed_y, reference.packed_y, atol=5e-10, rtol=5e-8)
    assert fast.solver_stats is not None
    assert fast.solver_stats["solver"] == "tsit5_fast"
    assert fast.solver_stats["rhs_calls"] > 0


def test_rust_tsit5_fast_reduced_outputs_match_full() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    saveat = np.array([0.0, 0.03, 0.11, 0.2, 0.37, 0.5], dtype=np.float64)
    full = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="tsit5_fast",
        execution_mode="expanded_sparse",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    populations = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="tsit5_fast",
        execution_mode="expanded_sparse",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        output="populations",
    )
    np.testing.assert_allclose(populations.t, full.t, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(populations.values, full.populations(), atol=1e-12, rtol=1e-10)

    final = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="tsit5_fast",
        execution_mode="expanded_sparse",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        output="selected",
        output_indices=[(0, 0), (0, 1), (1, 0)],
        output_when="final",
        dense_output=False,
    )
    expected = np.array([full.density_matrices()[-1][0, 0], full.density_matrices()[-1][0, 1], full.density_matrices()[-1][1, 0]])
    np.testing.assert_allclose(final.values, expected, atol=1e-12, rtol=1e-10)


def test_rust_batch_initial_conditions_match_repeated_solves() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    prepared = prepare_lindblad_problem(system, parameters, backend="rust")
    rho0_a = _ground_state_density()
    rho0_b = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128)
    batch = solve_lindblad_batch(
        prepared,
        np.stack([rho0_a, rho0_b]),
        (0.0, 0.5),
        solver="dopri5_fast",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
        collect_stats=True,
    )
    expected = []
    for rho0 in (rho0_a, rho0_b):
        result = solve_lindblad(
            prepared,
            rho0,
            (0.0, 0.5),
            solver="dopri5_fast",
            execution_mode="expanded_sparse",
            output="populations",
            output_when="final",
            dense_output=False,
            dt=1e-3,
            reltol=1e-8,
            abstol=1e-10,
        )
        expected.append(result.values)
    np.testing.assert_allclose(batch.values, np.asarray(expected), atol=1e-12, rtol=1e-10)
    assert batch.solver_stats is not None
    assert batch.solver_stats["solver"] == "dopri5_fast_batch"

    parallel = solve_lindblad_batch(
        prepared,
        np.stack([rho0_a, rho0_b]),
        (0.0, 0.5),
        solver="dopri5_fast",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=True,
        threads=2,
    )
    np.testing.assert_allclose(parallel.values, batch.values, atol=1e-12, rtol=1e-10)


def test_rust_batch_selected_saveat_matches_repeated_solves() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    prepared = prepare_lindblad_problem(system, parameters, backend="rust")
    rho0_a = _ground_state_density()
    rho0_b = np.array([[0.5, 0.1j], [-0.1j, 0.5]], dtype=np.complex128)
    saveat = np.linspace(0.0, 0.5, 6)
    selected_indices = [(0, 0), (0, 1), (1, 0)]
    batch = solve_lindblad_batch(
        prepared,
        np.stack([rho0_a, rho0_b]),
        (0.0, 0.5),
        solver="tsit5_fast",
        execution_mode="expanded_sparse",
        output="selected",
        output_indices=selected_indices,
        output_when="saveat",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    expected = []
    for rho0 in (rho0_a, rho0_b):
        result = solve_lindblad(
            prepared,
            rho0,
            (0.0, 0.5),
            solver="tsit5_fast",
            execution_mode="expanded_sparse",
            output="selected",
            output_indices=selected_indices,
            saveat=saveat,
            dt=1e-3,
            reltol=1e-8,
            abstol=1e-10,
        )
        expected.append(result.values)
    np.testing.assert_allclose(batch.t, saveat, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(batch.values, np.asarray(expected), atol=1e-12, rtol=1e-10)


def test_rust_batch_parameter_grid_matches_repeated_solves() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(system, {omega: 0.6, delta: 0.0}, backend="rust")
    rho0 = _ground_state_density()
    scan = {
        omega: np.array([0.4, 0.7]),
        delta: np.array([-0.1, 0.2]),
    }
    batch = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan=scan,
        solver="dopri5_fast",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    expected = []
    for omega_value in scan[omega]:
        for delta_value in scan[delta]:
            result = solve_lindblad(
                system,
                rho0,
                (0.0, 0.5),
                parameters={omega: omega_value, delta: delta_value},
                backend="rust",
                solver="dopri5_fast",
                execution_mode="expanded_sparse",
                output="populations",
                output_when="final",
                dense_output=False,
                dt=1e-3,
                reltol=1e-8,
                abstol=1e-10,
            )
            expected.append(result.values)
    np.testing.assert_allclose(batch.values, np.asarray(expected), atol=1e-12, rtol=1e-10)
    assert batch.metadata["scan_kind"] == "grid"
    assert batch.metadata["grid_shape"] == (2, 2)


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
