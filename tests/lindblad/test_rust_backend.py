from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import scipy.sparse
import sympy as smp

from centrex_tlf.lindblad.batch import grid_scan, solve_lindblad_batch
from centrex_tlf.lindblad.events import population
from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py, fill_hamiltonian_py
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
        _parameters: ClassVar[list[str]] = ["Ω0", "β", "ωphase", "δ"]
        _compound_vars: ClassVar[list[str]] = ["Ω"]

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
        solver="dopri5",
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


def test_entrywise_plan_rejects_expanded_sparse_at_solve_entry() -> None:
    """An entrywise plan cannot serve expanded_sparse; fail before integrating.

    `lower_expanded_sparse_rhs` returns None for a non-decomposed plan, and the
    Rust RHS used to be the first thing to notice, raising on its first call --
    after the solve had started, and inside a parallel scan far from the cause.
    """
    system = _make_two_level_system()
    parameters = LindbladParameters.from_kwargs(Ω=1.1, δ=0.15)
    prepared = prepare_lindblad_problem(
        system,
        parameters,
        backend="rust",
        hamiltonian_representation="entrywise",
    )
    assert prepared.expanded_rhs_plan is None

    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1.0
    with pytest.raises(ValueError, match="requires a decomposed Hamiltonian plan"):
        solve_lindblad(
            prepared,
            rho0,
            (0.0, 1e-6),
            solver="dopri5",
            execution_mode="expanded_sparse",
        )

    rho0_batch = rho0[None, ...]
    with pytest.raises(ValueError, match="requires a decomposed Hamiltonian plan"):
        solve_lindblad_batch(
            prepared,
            rho0_batch,
            (0.0, 1e-6),
            solver="dopri5",
            execution_mode="expanded_sparse",
        )

    with pytest.raises(ValueError, match="requires a decomposed Hamiltonian plan"):
        grid_scan(
            prepared,
            rho0,
            (0.0, 1e-6),
            scan={"δ": np.array([0.1, 0.2])},
            solver="dopri5",
            execution_mode="expanded_sparse",
        )


@pytest.mark.parametrize(
    "mode",
    [
        "expanded_sparse",
        "experimental_expanded_sparse_split_inputs",
        "experimental_expanded_sparse_baseline_packed",
    ],
)
def test_entrywise_plan_rejects_every_expanded_sparse_variant(mode: str) -> None:
    """All `experimental_expanded_sparse_*` modes read the same missing plan.

    `ExecutionMode::is_expanded_sparse_like` (rust/src/lindblad/rhs.rs) covers
    six names; the Python-side check keys on the shared `expanded_sparse`
    substring, so it must reject the experimental variants too.
    """
    system = _make_two_level_system()
    parameters = LindbladParameters.from_kwargs(Ω=1.1, δ=0.15)
    prepared = prepare_lindblad_problem(
        system,
        parameters,
        backend="rust",
        hamiltonian_representation="entrywise",
    )
    with pytest.raises(ValueError, match="requires a decomposed Hamiltonian plan"):
        prepared.check_execution_mode(mode)


def test_execution_mode_check_passes_for_decomposed_and_structured() -> None:
    system = _make_two_level_system()
    parameters = LindbladParameters.from_kwargs(Ω=1.1, δ=0.15)
    decomposed = prepare_lindblad_problem(
        system,
        parameters,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    assert decomposed.expanded_rhs_plan is not None
    for mode in ("reference", "structured", "structured_upper", "expanded_sparse"):
        decomposed.check_execution_mode(mode)

    entrywise = prepare_lindblad_problem(
        system,
        parameters,
        backend="rust",
        hamiltonian_representation="entrywise",
    )
    # Only the expanded-sparse family needs the extra plan.
    for mode in ("reference", "structured", "structured_upper"):
        entrywise.check_execution_mode(mode)


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
    for mode in (
        "expanded_sparse",
        "experimental_expanded_sparse_current_split_inputs",
        "experimental_expanded_sparse_baseline_packed",
    ):
        rhs_expanded = np.asarray(
            rust.lindblad_rhs_py(rust_plan, packed, time, mode),
            dtype=np.float64,
        )
        np.testing.assert_allclose(rhs_expanded, rhs_upper, atol=1e-11, rtol=1e-11)


def _make_chain_system(n_states: int) -> OBESystem:
    """Nearest-neighbor coupled chain with n_states levels.

    Used to exercise the `expanded_sparse` RHS kernel on both sides of the
    partitioned/split-inputs selection gate in `rust/src/lindblad/rhs.rs`
    (`PARTITIONED_PACKED_MAX_STATES`), not just the 2-level systems used
    elsewhere in this file.
    """
    omega, delta = smp.symbols("Omega delta", real=True)
    hamiltonian = smp.zeros(n_states, n_states)
    for i in range(n_states):
        hamiltonian[i, i] = delta * i
    for i in range(n_states - 1):
        hamiltonian[i, i + 1] = omega / 2
        hamiltonian[i + 1, i] = smp.conjugate(omega) / 2
    c_array = np.zeros((n_states - 1, n_states, n_states), dtype=np.complex128)
    for i in range(n_states - 1):
        c_array[i, i, i + 1] = np.sqrt(0.1)
    zeros = np.zeros((n_states, n_states), dtype=np.complex128)
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
        coupling_symbols=[omega, delta],
        polarization_symbols=[],
    )


def _chain_packed_rho(n_states: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    real = rng.standard_normal((n_states, n_states))
    imag = rng.standard_normal((n_states, n_states))
    matrix = real + 1j * imag
    hermitian = matrix + matrix.conj().T
    return hermitian.astype(np.complex128)


@pytest.mark.parametrize(
    "n_states",
    [3, 45],
    ids=["below_partitioned_gate", "above_partitioned_gate"],
)
def test_expanded_sparse_rhs_matches_reference_across_partition_gate(n_states: int) -> None:
    # n_states=3 stays below and n_states=45 stays above the
    # `PARTITIONED_PACKED_MAX_STATES` (=40) gate in rust/src/lindblad/rhs.rs, so this
    # covers both the partitioned kernel and the split-inputs fallback with a single
    # parametrized check.
    system = _make_chain_system(n_states)
    omega_symbol, delta_symbol = system.coupling_symbols
    parameters = {
        "omega0": 1.0,
        "modulation_depth": 0.5,
        "modulation_frequency": 1.9,
        str(delta_symbol): 0.2,
        str(omega_symbol): "omega0*phase_modulation(t, modulation_depth, modulation_frequency)",
    }
    prepared = prepare_lindblad_problem(
        system,
        parameters,
        backend="python",
        hamiltonian_representation="decomposed",
    )
    assert prepared.expanded_rhs_plan is not None
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    packed = prepared.layout.pack(_chain_packed_rho(n_states))
    rhs_reference = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, 0.41, "reference"),
        dtype=np.float64,
    )
    rhs_expanded = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, 0.41, "expanded_sparse"),
        dtype=np.float64,
    )
    np.testing.assert_allclose(rhs_expanded, rhs_reference, atol=1e-12, rtol=1e-9)


def test_expanded_sparse_indirect_time_dependency_updates_reused_workspace() -> None:
    system = _make_two_level_system()
    parameters = {
        "omega0": 1.0,
        "modulation_depth": 0.5,
        "modulation_frequency": 1.9,
        str(system.coupling_symbols[1]): -0.1,
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
    expanded = rust.create_lindblad_rhs_evaluator_py(rust_plan, "expanded_sparse")
    structured = rust.create_lindblad_rhs_evaluator_py(rust_plan, "structured_upper")
    packed = prepared.layout.pack(
        np.array([[0.8, 0.05 + 0.04j], [0.05 - 0.04j, 0.2]], dtype=np.complex128)
    )
    first = np.asarray(expanded.rhs_packed_py(packed, 0.0), dtype=np.float64)
    second = np.asarray(expanded.rhs_packed_py(packed, 0.41), dtype=np.float64)
    expected = np.asarray(structured.rhs_packed_py(packed, 0.41), dtype=np.float64)
    assert not np.allclose(first, second)
    np.testing.assert_allclose(second, expected, atol=1e-11, rtol=1e-11)


def test_expanded_sparse_split_input_flag_matches_default_rhs() -> None:
    system = _make_two_level_system()
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
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    packed = prepared.layout.pack(
        np.array([[0.8, 0.05 + 0.04j], [0.05 - 0.04j, 0.2]], dtype=np.complex128)
    )

    rhs_default = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, 0.31, "expanded_sparse"),
        dtype=np.float64,
    )
    rhs_without_split_inputs = np.asarray(
        rust.lindblad_rhs_py(rust_plan, packed, 0.31, "expanded_sparse", False),
        dtype=np.float64,
    )
    np.testing.assert_allclose(rhs_without_split_inputs, rhs_default, atol=1e-11, rtol=1e-11)


def test_expanded_sparse_split_input_flag_single_batch_and_grid() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(
        system,
        {omega: 0.6, delta: 0.0},
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho0 = _ground_state_density()
    saveat = np.linspace(0.0, 0.5, 6)
    solve_kwargs = dict(
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="saveat",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )

    single_default = solve_lindblad(prepared, rho0, (0.0, 0.5), **solve_kwargs)
    single_without_split_inputs = solve_lindblad(
        prepared,
        rho0,
        (0.0, 0.5),
        use_split_input_rhs=False,
        **solve_kwargs,
    )
    np.testing.assert_allclose(
        single_without_split_inputs.values,
        single_default.values,
        atol=1e-11,
        rtol=1e-11,
    )

    rho0_batch = np.repeat(prepared.layout.pack(rho0).reshape(1, -1), 2, axis=0)
    batch_default = solve_lindblad_batch(
        prepared,
        rho0_batch,
        (0.0, 0.5),
        parallel=False,
        **solve_kwargs,
    )
    batch_without_split_inputs = solve_lindblad_batch(
        prepared,
        rho0_batch,
        (0.0, 0.5),
        use_split_input_rhs=False,
        parallel=False,
        **solve_kwargs,
    )
    np.testing.assert_allclose(
        batch_without_split_inputs.values,
        batch_default.values,
        atol=1e-11,
        rtol=1e-11,
    )

    grid_default = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan={omega: np.array([0.4, 0.7]), delta: np.array([-0.1, 0.2])},
        parallel=False,
        **solve_kwargs,
    )
    grid_without_split_inputs = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan={omega: np.array([0.4, 0.7]), delta: np.array([-0.1, 0.2])},
        use_split_input_rhs=False,
        parallel=False,
        **solve_kwargs,
    )
    np.testing.assert_allclose(
        grid_without_split_inputs.values,
        grid_default.values,
        atol=1e-11,
        rtol=1e-11,
    )


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
def test_rust_dopri5_matches_dopri5(saveat: np.ndarray, execution_mode: str) -> None:
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
        solver="dopri5",
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
    assert fast.solver_stats["solver"] == "dopri5"
    assert fast.solver_stats["rhs_calls"] > 0


def test_rust_dopri5_population_outputs_match_full() -> None:
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
        solver="dopri5",
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
        solver="dopri5",
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
        solver="dopri5",
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


@pytest.mark.parametrize("solver", ["dopri5", "tsit5"])
def test_native_terminal_runtime_event_appends_event_time(solver: str) -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    event_time = 0.23
    result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver=solver,
        execution_mode="expanded_sparse",
        saveat=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=Time() - event_time,
        collect_stats=True,
    )
    assert result.t[-1] == pytest.approx(event_time, abs=1e-10)
    assert np.all(result.t <= event_time + 1e-12)
    assert result.solver_stats is not None
    assert result.solver_stats["event_triggered"] is True
    assert result.solver_stats["event_time"] == pytest.approx(event_time, abs=1e-10)


def test_native_terminal_runtime_helper_event() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    stop_event = gaussian(Time(), center=0.2, sigma=0.05) - 0.5
    result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        execution_mode="expanded_sparse",
        saveat=np.linspace(0.0, 0.5, 8),
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=stop_event,
        collect_stats=True,
    )
    expected = 0.2 - 0.05 * np.sqrt(2.0 * np.log(2.0))
    assert result.t[-1] == pytest.approx(expected, abs=1e-7)
    assert result.solver_stats is not None
    assert result.solver_stats["event_triggered"] is True


def test_native_population_threshold_events() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 2.0, str(system.coupling_symbols[1]): 0.0}
    single = solve_lindblad(
        system,
        rho0,
        (0.0, 0.8),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        execution_mode="expanded_sparse",
        saveat=np.linspace(0.0, 0.8, 9),
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=population(1, threshold=0.05),
    )
    assert single.t[-1] < 0.8
    assert single.populations()[-1, 1] == pytest.approx(0.05, abs=2e-5)

    multi = solve_lindblad(
        system,
        rho0,
        (0.0, 0.8),
        parameters=parameters,
        backend="rust",
        solver="tsit5",
        execution_mode="expanded_sparse",
        saveat=np.linspace(0.0, 0.8, 9),
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=population([0, 1], threshold=1.0),
    )
    assert multi.t[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("solver", ["python_rk45", "scipy_rk45", "scipy_bdf", "scipy_radau"])
def test_scipy_and_python_terminal_runtime_event(solver: str) -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    backend = "python" if solver == "python_rk45" else "rust"
    result = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend=backend,
        solver=solver,
        execution_mode="structured_upper",
        saveat=np.array([0.0, 0.1, 0.3, 0.5]),
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=Time() - 0.2,
    )
    assert result.t[-1] == pytest.approx(0.2, abs=1e-10)
    assert result.solver_stats is not None
    assert result.solver_stats["event_triggered"] is True


def test_terminal_event_final_output_returns_event_state() -> None:
    system = _make_two_level_system()
    rho0 = _ground_state_density()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    event_time = 0.2
    final = solve_lindblad(
        system,
        rho0,
        (0.0, 0.5),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=Time() - event_time,
    )
    full = solve_lindblad(
        system,
        rho0,
        (0.0, event_time),
        parameters=parameters,
        backend="rust",
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
    )
    np.testing.assert_allclose(final.t, [event_time], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(final.values, full.values, atol=2e-10, rtol=2e-8)


def test_rust_dopri5_selected_outputs_match_full() -> None:
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
        solver="dopri5",
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
        solver="dopri5",
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
        solver="dopri5",
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


def test_rust_dopri5_dense_output_false_rejects_interior_saveat() -> None:
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
            solver="dopri5",
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
def test_rust_tsit5_matches_dopri5(saveat: np.ndarray, execution_mode: str) -> None:
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
        solver="tsit5",
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
    assert fast.solver_stats["solver"] == "tsit5"
    assert fast.solver_stats["rhs_calls"] > 0


def test_rust_tsit5_reduced_outputs_match_full() -> None:
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
        solver="tsit5",
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
        solver="tsit5",
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
        solver="tsit5",
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


def test_rust_fixed_rk4_matches_adaptive_for_single_trajectory() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    prepared = prepare_lindblad_problem(system, parameters, backend="rust")
    rho0 = _ground_state_density()
    adaptive = solve_lindblad(
        prepared,
        rho0,
        (0.0, 0.5),
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-10,
        abstol=1e-12,
    )
    fixed = solve_lindblad(
        prepared,
        rho0,
        (0.0, 0.5),
        solver="fixed_rk4",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        collect_stats=True,
    )
    np.testing.assert_allclose(fixed.values, adaptive.values, atol=2e-8, rtol=2e-8)
    assert fixed.solver_stats is not None
    assert fixed.solver_stats["solver"] == "fixed_rk4"
    assert fixed.solver_stats["rejected_steps"] == 0
    assert fixed.solver_stats["rhs_calls"] == 4 * fixed.solver_stats["accepted_steps"]


def test_rust_fixed_dopri5_matches_adaptive_for_single_trajectory() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    prepared = prepare_lindblad_problem(system, parameters, backend="rust")
    rho0 = _ground_state_density()
    adaptive = solve_lindblad(
        prepared,
        rho0,
        (0.0, 0.5),
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-10,
        abstol=1e-12,
    )
    fixed = solve_lindblad(
        prepared,
        rho0,
        (0.0, 0.5),
        solver="fixed_dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        collect_stats=True,
    )
    np.testing.assert_allclose(fixed.values, adaptive.values, atol=2e-8, rtol=2e-8)
    assert fixed.solver_stats is not None
    assert fixed.solver_stats["solver"] == "fixed_dopri5"
    assert fixed.solver_stats["rejected_steps"] == 0
    assert fixed.solver_stats["rhs_calls"] == 6 * fixed.solver_stats["accepted_steps"]


def test_rust_fixed_rk2_runs_with_expected_stage_count() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    prepared = prepare_lindblad_problem(system, parameters, backend="rust")
    result = solve_lindblad(
        prepared,
        _ground_state_density(),
        (0.0, 0.5),
        solver="fixed_rk2",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=5e-4,
        collect_stats=True,
    )
    assert result.solver_stats is not None
    assert result.solver_stats["solver"] == "fixed_rk2"
    assert result.solver_stats["rhs_calls"] == 2 * result.solver_stats["accepted_steps"]
    assert np.isfinite(result.values).all()


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
        solver="dopri5",
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
            solver="dopri5",
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
    assert batch.solver_stats["solver"] == "dopri5"

    parallel = solve_lindblad_batch(
        prepared,
        np.stack([rho0_a, rho0_b]),
        (0.0, 0.5),
        solver="dopri5",
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
        solver="tsit5",
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
            solver="tsit5",
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
        solver="dopri5",
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
                solver="dopri5",
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


def test_rust_fixed_rk4_grid_matches_repeated_solves() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(system, {omega: 0.6, delta: 0.0}, backend="rust")
    rho0 = _ground_state_density()
    scan = {
        omega: np.array([0.4, 0.7]),
        delta: np.array([-0.1, 0.2]),
    }
    grid = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan=scan,
        solver="fixed_rk4",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        parallel=True,
        threads=2,
        collect_stats=True,
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
                solver="fixed_rk4",
                execution_mode="expanded_sparse",
                output="populations",
                output_when="final",
                dense_output=False,
                dt=1e-3,
            )
            expected.append(result.values)
    np.testing.assert_allclose(grid.values, np.asarray(expected), atol=1e-12, rtol=1e-10)
    assert grid.solver_stats is not None
    assert grid.solver_stats["solver"] == "fixed_rk4"
    assert grid.solver_stats["rejected_steps"] == 0


def test_batch_and_grid_integral_final_allow_saveat_none() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(system, {omega: 0.6, delta: 0.0}, backend="rust")
    weights = [(1, 0.3)]
    rho0_a = _ground_state_density()
    rho0_b = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128)

    batch = solve_lindblad_batch(
        prepared,
        np.stack([rho0_a, rho0_b]),
        (0.0, 0.5),
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="photon_integral",
        integral_weights=weights,
        output_when="final",
        saveat=None,
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    assert batch.values.shape == (2, 1)
    assert batch.t.shape == (1,)
    assert np.all(batch.values >= 0.0)

    grid = grid_scan(
        prepared,
        rho0_a,
        (0.0, 0.5),
        scan={omega: np.array([0.4, 0.7]), delta: np.array([-0.1, 0.2])},
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="photon_integral",
        integral_weights=weights,
        output_when="final",
        saveat=None,
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    assert grid.values.shape == (4, 1)
    assert grid.t.shape == (1,)
    assert np.all(grid.values >= 0.0)


def test_batch_integral_trace_and_rate_trace_match_populations() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    prepared = prepare_lindblad_problem(system, parameters, backend="rust")
    weights = [(1, 0.3)]
    saveat = np.linspace(0.0, 0.5, 11)
    rho0_batch = np.stack(
        [
            _ground_state_density(),
            np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128),
        ]
    )

    populations = solve_lindblad_batch(
        prepared,
        rho0_batch,
        (0.0, 0.5),
        solver="tsit5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="saveat",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    rate = solve_lindblad_batch(
        prepared,
        rho0_batch,
        (0.0, 0.5),
        solver="tsit5",
        execution_mode="expanded_sparse",
        output="photon_rate",
        integral_weights=weights,
        output_when="saveat",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    trace = solve_lindblad_batch(
        prepared,
        rho0_batch,
        (0.0, 0.5),
        solver="tsit5",
        execution_mode="expanded_sparse",
        output="photon_integral",
        integral_weights=weights,
        output_when="saveat",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    final_from_same_grid = solve_lindblad_batch(
        prepared,
        rho0_batch,
        (0.0, 0.5),
        solver="tsit5",
        execution_mode="expanded_sparse",
        output="photon_integral",
        integral_weights=weights,
        output_when="final",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )

    np.testing.assert_allclose(rate.t, saveat, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(trace.t, saveat, atol=1e-13, rtol=0.0)
    assert rate.values.shape == (2, saveat.size, 1)
    assert trace.values.shape == (2, saveat.size, 1)
    np.testing.assert_allclose(rate.values[..., 0], 0.3 * populations.values[..., 1])
    np.testing.assert_allclose(trace.values[:, -1, :], final_from_same_grid.values)


def test_grid_integral_trace_shape_and_rate_validation() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(system, {omega: 0.6, delta: 0.0}, backend="rust")
    saveat = np.linspace(0.0, 0.5, 6)
    weights = [(1, 0.3)]

    trace = grid_scan(
        prepared,
        _ground_state_density(),
        (0.0, 0.5),
        scan={omega: np.array([0.4, 0.7]), delta: np.array([-0.1, 0.2])},
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="weighted_integral",
        integral_weights=weights,
        output_when="saveat",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    assert trace.values.shape == (4, saveat.size, 1)
    np.testing.assert_allclose(trace.t, saveat, atol=1e-13, rtol=0.0)

    with pytest.raises(ValueError, match="requires output_when='saveat'"):
        grid_scan(
            prepared,
            _ground_state_density(),
            (0.0, 0.5),
            scan={omega: np.array([0.4])},
            output="weighted_rate",
            integral_weights=weights,
            output_when="final",
        )
    with pytest.raises(ValueError, match="requires explicit saveat"):
        solve_lindblad_batch(
            prepared,
            np.stack([_ground_state_density()]),
            (0.0, 0.5),
            output="weighted_rate",
            integral_weights=weights,
            output_when="saveat",
        )


def test_grid_direct_collation_parallel_matches_serial_for_final_and_saveat() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(system, {omega: 0.6, delta: 0.0}, backend="rust")
    rho0 = _ground_state_density()
    scan = {
        omega: np.array([0.35, 0.6, 0.85]),
        delta: np.array([-0.15, 0.0, 0.25]),
    }

    serial_final = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan=scan,
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=False,
    )
    parallel_final = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan=scan,
        solver="dopri5",
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
    assert serial_final.values.shape == (9, 2)
    np.testing.assert_allclose(parallel_final.t, serial_final.t, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(
        parallel_final.values,
        serial_final.values,
        atol=1e-12,
        rtol=1e-10,
    )

    saveat = np.array([0.0, 0.07, 0.2, 0.5], dtype=np.float64)
    selected_indices = [(0, 0), (0, 1), (1, 0)]
    serial_selected = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan=scan,
        solver="tsit5",
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
    parallel_selected = grid_scan(
        prepared,
        rho0,
        (0.0, 0.5),
        scan=scan,
        solver="tsit5",
        execution_mode="expanded_sparse",
        output="selected",
        output_indices=selected_indices,
        output_when="saveat",
        saveat=saveat,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        parallel=True,
        threads=2,
    )
    assert serial_selected.values.shape == (9, saveat.size, len(selected_indices))
    np.testing.assert_allclose(parallel_selected.t, saveat, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(
        parallel_selected.values,
        serial_selected.values,
        atol=1e-12,
        rtol=1e-10,
    )


def test_batch_terminal_event_final_only_times() -> None:
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.6, str(system.coupling_symbols[1]): 0.0}
    prepared = prepare_lindblad_problem(system, parameters, backend="rust")
    rho0_a = _ground_state_density()
    rho0_b = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128)
    batch = solve_lindblad_batch(
        prepared,
        np.stack([rho0_a, rho0_b]),
        (0.0, 0.5),
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=Time() - 0.2,
        parallel=False,
        collect_stats=True,
    )
    np.testing.assert_allclose(batch.t, [0.2, 0.2], atol=1e-10, rtol=0.0)
    np.testing.assert_array_equal(batch.metadata["event_triggered"], [True, True])
    np.testing.assert_allclose(batch.metadata["event_times"], [0.2, 0.2], atol=1e-10, rtol=0.0)
    assert batch.solver_stats is not None
    assert batch.solver_stats["event_count"] == 2


def test_grid_terminal_event_final_only_times() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(system, {omega: 0.6, delta: 0.0}, backend="rust")
    batch = grid_scan(
        prepared,
        _ground_state_density(),
        (0.0, 0.5),
        scan={omega: np.array([0.4, 0.7]), delta: np.array([-0.1, 0.2])},
        solver="tsit5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        dt=1e-3,
        reltol=1e-8,
        abstol=1e-10,
        stop_event=Time() - 0.2,
        parallel=False,
        collect_stats=True,
    )
    np.testing.assert_allclose(batch.t, np.full(4, 0.2), atol=1e-10, rtol=0.0)
    np.testing.assert_array_equal(batch.metadata["event_triggered"], np.full(4, True))
    assert batch.solver_stats is not None
    assert batch.solver_stats["event_count"] == 4


def test_batch_and_grid_reject_saveat_terminal_events() -> None:
    system = _make_two_level_system()
    omega = str(system.coupling_symbols[0])
    delta = str(system.coupling_symbols[1])
    prepared = prepare_lindblad_problem(system, {omega: 0.6, delta: 0.0}, backend="rust")
    with pytest.raises(ValueError, match="stop_event is only supported"):
        solve_lindblad_batch(
            prepared,
            np.stack([_ground_state_density(), _ground_state_density()]),
            (0.0, 0.5),
            output_when="saveat",
            saveat=np.linspace(0.0, 0.5, 4),
            stop_event=Time() - 0.2,
        )
    with pytest.raises(ValueError, match="stop_event is only supported"):
        grid_scan(
            prepared,
            _ground_state_density(),
            (0.0, 0.5),
            scan={omega: np.array([0.4, 0.7])},
            output_when="saveat",
            saveat=np.linspace(0.0, 0.5, 4),
            stop_event=Time() - 0.2,
        )
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


def _packed_jacobian_dense(evaluator, t: float, dim: int, method: str) -> np.ndarray:
    rows, cols, values = evaluator.jacobian_packed_sparse_py(t, 0.0, method)
    return (
        scipy.sparse.csc_matrix(
            (
                np.asarray(values, dtype=np.float64),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=(dim, dim),
        )
        .toarray()
    )


@pytest.mark.parametrize("mode", ["structured", "expanded_sparse"])
def test_analytic_packed_jacobian_matches_probe_bitwise(mode: str) -> None:
    """The analytic Jacobian must reproduce the probe exactly, not merely closely.

    The Lindblad RHS is linear in rho and the packed encoding is real-linear, so the
    basis-vector probe is already an exact derivative. Transcribing the same terms
    out of the expanded sparse plan sums the identical floating-point products, so
    the two agree bit for bit -- anything less means the transcription reassociated
    something.
    """
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.8, str(system.coupling_symbols[1]): 0.05}
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    evaluator = rust.create_lindblad_rhs_evaluator_py(rust_plan, mode)
    dim = 2 * 2

    probed = _packed_jacobian_dense(evaluator, 0.2, dim, "probe")
    analytic = _packed_jacobian_dense(evaluator, 0.2, dim, "analytic")
    np.testing.assert_array_equal(analytic, probed)
    assert np.abs(probed).max() > 0.0

    # "auto" must select the analytic path when the plan supports it.
    auto = _packed_jacobian_dense(evaluator, 0.2, dim, "auto")
    np.testing.assert_array_equal(auto, probed)


def test_analytic_packed_jacobian_reproduces_the_rhs() -> None:
    """J @ x must equal rhs(x) for arbitrary x, not just for basis vectors."""
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.8, str(system.coupling_symbols[1]): 0.05}
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    evaluator = rust.create_lindblad_rhs_evaluator_py(rust_plan, "expanded_sparse")
    dim = 2 * 2

    analytic = _packed_jacobian_dense(evaluator, 0.2, dim, "analytic")
    rng = np.random.default_rng(20260820)
    for _ in range(5):
        x = rng.standard_normal(dim)
        expected = np.asarray(evaluator.rhs_packed_py(x, 0.2), dtype=np.float64)
        np.testing.assert_allclose(analytic @ x, expected, atol=1e-12, rtol=1e-12)


def test_analytic_packed_jacobian_requires_a_decomposed_plan() -> None:
    """An entrywise plan has no expanded terms, so 'auto' must fall back to probing."""
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.8, str(system.coupling_symbols[1]): 0.05}
    prepared = prepare_lindblad_problem(
        system, parameters, backend="python", hamiltonian_representation="entrywise"
    )
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    evaluator = rust.create_lindblad_rhs_evaluator_py(rust_plan, "structured")
    dim = 2 * 2

    with pytest.raises(ValueError, match="decomposed"):
        evaluator.jacobian_packed_sparse_py(0.2, 0.0, "analytic")

    probed = _packed_jacobian_dense(evaluator, 0.2, dim, "probe")
    auto = _packed_jacobian_dense(evaluator, 0.2, dim, "auto")
    np.testing.assert_array_equal(auto, probed)


def test_hamiltonian_cache_is_not_shared_between_rhs_flavours() -> None:
    """Regression: one validity flag used to guard three disjoint Hamiltonian caches.

    The complex-matrix path fills `expanded_term_values`; the packed path fills
    `expanded_term_values_re`/`_im`. Both used to set the same `hamiltonian_valid`
    flag, so whichever ran second skipped filling its own cache and then read it
    empty. Only reachable for time-independent plans, where the cache is reused.
    Both interleavings must now work and must agree with a fresh evaluator.
    """
    system = _make_two_level_system()
    parameters = {str(system.coupling_symbols[0]): 0.8, str(system.coupling_symbols[1]): 0.05}
    prepared = prepare_lindblad_problem(system, parameters, backend="python")
    rust_plan = rust.prepare_lindblad_problem_py(prepared.to_payload())
    flat = np.array([0.7, 0.1 + 0.05j, 0.1 - 0.05j, 0.3], dtype=np.complex128)
    packed = np.array([0.7, 0.3, 0.1, 0.05], dtype=np.float64)

    def fresh():
        return rust.create_lindblad_rhs_evaluator_py(rust_plan, "expanded_sparse")

    reference_matrix = np.asarray(fresh().rhs_matrix_py(flat, 0.2), dtype=np.complex128)
    reference_packed = np.asarray(fresh().rhs_packed_py(packed, 0.2), dtype=np.float64)

    # matrix first, then packed, then matrix again
    evaluator = fresh()
    np.testing.assert_allclose(
        np.asarray(evaluator.rhs_matrix_py(flat, 0.2), dtype=np.complex128),
        reference_matrix,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.rhs_packed_py(packed, 0.2), dtype=np.float64),
        reference_packed,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.rhs_matrix_py(flat, 0.2), dtype=np.complex128),
        reference_matrix,
        atol=1e-12,
        rtol=1e-12,
    )

    # packed first, then the complex-matrix path via the split Jacobian
    evaluator = fresh()
    evaluator.rhs_packed_py(packed, 0.2)
    rows, cols, values = evaluator.jacobian_split_sparse_py(0.2)
    assert len(values) > 0


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


def test_rust_scipy_rk45_solver_matches_python_structured_reference() -> None:
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
        solver="scipy_rk45",
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
