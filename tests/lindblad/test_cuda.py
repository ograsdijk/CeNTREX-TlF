"""CUDA solver checks against independent complex Liouvillian evolution."""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import sympy as smp
from scipy.integrate import solve_ivp

from centrex_tlf import lindblad
from centrex_tlf.lindblad import CudaSolverOptions
from centrex_tlf.lindblad import cuda as cuda_module


def model():
    omega, delta, phase = smp.symbols("omega delta phase", real=True)
    H = smp.zeros(3)
    H[0, 1] = omega * (1 + smp.I * phase) / 2
    H[1, 0] = smp.conjugate(H[0, 1])
    H[1, 1] = -delta
    C = np.zeros((2, 3, 3), complex)
    C[0, 0, 1] = np.sqrt(0.3)
    C[1, 2, 1] = np.sqrt(0.2)
    p = lindblad.prepare_lindblad_problem(
        SimpleNamespace(H_symbolic=H, C_array=C), dict(omega=1.2, delta=0.4, phase=0.3)
    )
    a = np.diag([1.0, 0.0, 0.0]).astype(complex)
    b = np.array([[0.3, 0.1 + 0.12j, 0.02j], [0.1 - 0.12j, 0.5, 0.03], [-0.02j, 0.03, 0.2]])
    return p, H, C, np.stack((a, b))


@pytest.mark.parametrize("solver", ["fixed_tsit5", "fixed_ck54"])
def test_exact_density_reduction_all_outputs_and_new_initial_support(gpu, solver):
    p, H, C, initial = model()
    initial[1, 0, 2] = initial[1, 2, 0] = 0
    initial[1, 1, 2] = initial[1, 2, 1] = 0
    times = np.array([0.1, 0.137, 0.407])
    h = np.array(
        H.subs({s: dict(omega=1.2, delta=0.4, phase=0.3)[str(s)] for s in H.free_symbols}), complex
    )
    L = generator(h, C)
    expected = np.array(
        [
            [la.expm(L * (t - 0.1)) @ np.r_[rho.reshape(-1, order="F"), 0.0] for t in times]
            for rho in initial
        ]
    )
    for output in (
        "full",
        "populations",
        "selected",
        "photon_integral",
        "full_and_integral",
        "photon_rate",
    ):
        kwargs = {}
        if output in {"photon_integral", "full_and_integral", "photon_rate"}:
            kwargs["integral_weights"] = [(1, 0.5)]
        if output == "selected":
            kwargs["output_indices"] = [(2, 0), (1, 0), (2, 2)]
        answer = lindblad.solve_lindblad_batch(
            p,
            initial,
            (0.1, 0.407),
            solver=solver,
            evolution_device="cuda",
            dt=0.003,
            output=output,
            output_when="saveat",
            saveat=times,
            collect_stats=True,
            cuda_options=CudaSolverOptions(
                batch_size=1, density_reduction=True, density_saveat=times
            ),
            **kwargs,
        )
        rho = expected[:, :, :-1].reshape(2, 3, 3, 3).transpose(0, 1, 3, 2)
        snapshots = p.layout.unpack_batch(answer.metadata["density_values"])
        np.testing.assert_allclose(snapshots, rho, atol=3e-11)
        assert all(c["retained_density_pairs"] == 4 for c in answer.solver_stats["chunks"])
        assert answer.metadata["density_reduction"] is True
        if output == "full":
            want = p.layout.pack_batch(rho)
        elif output == "populations":
            want = np.diagonal(rho, axis1=-2, axis2=-1).real
        elif output == "selected":
            want = rho[:, :, [2, 1, 2], [0, 0, 2]]
        elif output == "photon_integral":
            want = expected[:, :, -1:].real
        elif output == "photon_rate":
            want = (0.5 * rho[:, :, 1, 1].real)[..., None]
        else:
            want = np.concatenate((p.layout.pack_batch(rho), expected[:, :, -1:].real), axis=-1)
        np.testing.assert_allclose(answer.values, want, atol=3e-11)
    # A new initial coherence changes the closure; it must never be dropped.
    _, _, _, original = model()
    answer = lindblad.solve_lindblad_batch(
        p,
        original,
        (0, 0.1),
        solver=solver,
        evolution_device="cuda",
        dt=0.003,
        output="full",
        collect_stats=True,
        cuda_options=CudaSolverOptions(batch_size=1, density_reduction=True),
    )
    assert [c["retained_density_pairs"] for c in answer.solver_stats["chunks"]] == [4, 6]


def test_density_reduction_omitted_populations_and_zero_support(gpu):
    H = smp.Matrix([[0, 1.0, 0, 0], [1.0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    C = np.zeros((1, 4, 4), complex)
    C[0, 2, 1] = np.sqrt(0.3)
    p = lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=C), {})
    initial = np.zeros((2, 4, 4), complex)
    initial[0, 0, 0] = 1.0
    result = lindblad.solve_lindblad_batch(
        p,
        initial,
        (0, 0.1),
        evolution_device="cuda",
        solver="fixed_tsit5",
        dt=0.003,
        output="full_and_integral",
        integral_weights=[(3, 1.0)],
        collect_stats=True,
        cuda_options=CudaSolverOptions(batch_size=1, density_reduction=True),
    )
    assert [c["retained_density_pairs"] for c in result.solver_stats["chunks"]] == [4, 0]
    np.testing.assert_array_equal(result.values[:, -1], 0)
    np.testing.assert_array_equal(result.values[1], 0)
    rho = p.layout.unpack(result.values[0, :-1])
    np.testing.assert_array_equal(rho[3], 0)
    np.testing.assert_array_equal(rho[:, 3], 0)
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-13)


def test_grid_prepacks_distinct_initial_states_before_parameter_expansion(monkeypatch):
    from centrex_tlf.lindblad import batch
    from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

    p, _, _, initial = model()
    expected = np.tile(p.layout.pack_batch(initial), (4, 1))
    packed_shapes = []
    original = PackedHermitianLayout.pack_batch

    def counted_pack(self, matrices):
        packed_shapes.append(matrices.shape)
        return original(self, matrices)

    def capture(prepared, packed, span, **kwargs):
        np.testing.assert_array_equal(packed, expected)
        assert packed.shape == (8, p.layout.packed_len)
        np.testing.assert_array_equal(
            kwargs["parameter_batch"],
            [
                [0.2, 0.1],
                [0.2, 0.1],
                [0.2, 0.3],
                [0.2, 0.3],
                [0.4, 0.1],
                [0.4, 0.1],
                [0.4, 0.3],
                [0.4, 0.3],
            ],
        )
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(PackedHermitianLayout, "pack_batch", counted_pack)
    monkeypatch.setattr(batch, "solve_lindblad_batch", capture)
    result = cuda_module._cuda_grid_scan(
        p, None, (0.0, 1.0), {"omega": [0.2, 0.4], "delta": [0.1, 0.3]}, {"rho0_batch": initial}
    )
    assert packed_shapes == [(2, 3, 3)]
    assert result.metadata["initial_condition_count"] == 2


@pytest.mark.parametrize("bad", ["nonhermitian", "nonfinite"])
def test_grid_validates_matrix_inputs_before_prepacking(bad):
    p, _, _, initial = model()
    if bad == "nonhermitian":
        initial[0, 0, 1] = 1j
    else:
        initial[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Hermitian|finite"):
        cuda_module._cuda_grid_scan(
            p, None, (0.0, 1.0), {"omega": [0.2, 0.4]}, {"rho0_batch": initial}
        )


def generator(H, C):
    n = len(H)
    eye = np.eye(n)
    L = -1j * (np.kron(eye, H) - np.kron(H.T, eye))
    for jump in C:
        K = jump.conj().T @ jump
        L += np.kron(jump.conj(), jump) - 0.5 * (np.kron(eye, K) + np.kron(K.T, eye))
    augmented = np.zeros((n * n + 1, n * n + 1), complex)
    augmented[:-1, :-1] = L
    augmented[-1, n + 1] = 0.5
    return augmented


@pytest.fixture
def gpu():
    import importlib.util

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy unavailable")
    try:
        return cuda_module._cupy()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("solver", ["fixed_tsit5", "fixed_ck54"])
@pytest.mark.parametrize(
    "path,reduced",
    [
        ("direct", False),
        ("direct", True),
        ("component", False),
        ("component", True),
        ("product", False),
        ("tiled", False),
    ],
)
def test_generic_fma_paths_against_complex_liouvillian(gpu, solver, path, reduced):
    p, H, C, initial = model()
    # Six rows amortize the two shared real/imaginary components' storage.
    initial = np.repeat(initial, 3, axis=0)
    drives = [1.2, 0.7, 0.3, 0.0, 0.5, 1.0]
    matrices = [
        np.asarray(
            H.subs({s: dict(omega=drive, delta=0.4, phase=0.3)[str(s)] for s in H.free_symbols}),
            complex,
        )
        for drive in drives
    ]
    times = np.array([0.137, 0.407])
    expected = np.array(
        [
            [
                (la.expm(generator(h, C) * t) @ np.r_[rho.reshape(-1, order="F"), 0.0])[
                    :-1
                ].reshape(3, 3, order="F")
                for t in times
            ]
            for rho, h in zip(initial, matrices, strict=True)
        ]
    )
    result = lindblad.solve_lindblad_batch(
        p,
        initial,
        (0.0, times[-1]),
        solver=solver,
        evolution_device="cuda",
        dt=0.003,
        output="full",
        output_when="saveat",
        saveat=times,
        collect_stats=True,
        parameter_slots=["omega"],
        parameter_batch=np.array(drives)[:, None],
        cuda_options=CudaSolverOptions(
            rhs_specialization="off",
            density_reduction=reduced,
            rhs="direct" if path in {"direct", "component"} else "tiled",
            tile_size=0 if path == "product" else 32,
            hamiltonian_sharing="on" if path == "component" else "off",
        ),
    )
    actual = p.layout.unpack_batch(result.values)
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=1e-11)
    np.testing.assert_allclose(np.trace(actual, axis1=-2, axis2=-1), 1.0, atol=1e-12)
    chunk = result.solver_stats["chunks"][0]
    assert chunk["rhs_specialization"]["kernel"] == "generic"
    assert (chunk["shared_hamiltonian_components"] > 0) == (path == "component")
    assert chunk["tile_size"] == (32 if path == "tiled" else 0)


def sharing_model(nonseparable=False, complex_factor=False):
    """Independent drives with overlapping H support and an exact multipass factor."""
    a, b, d = smp.symbols("a b d")
    H = smp.Matrix(
        [[0, a + 0.2 * b, b], [smp.conjugate(a + 0.2 * b), d, 0], [smp.conjugate(b), 0, 0.3]]
    )
    C = np.zeros((2, 3, 3), complex)
    C[0, 2, 1], C[1, 0, 2] = np.sqrt(0.3), np.sqrt(0.2)
    params = lindblad.LindbladParameters()
    t = params.time()
    y = params.real("height", 0.0)
    pa, pb = params.real("power_a", 1.0), params.real("power_b", 1.0)
    freq = params.real("frequency", 3.0)
    amps = params.real("intensities", (1.0, 0.7))
    xs = params.real("centers", (0.02, 0.08))
    ys = params.real("heights", (0.0, 0.0))
    envelope = lindblad.multipass_2d_intensity(t, y, amps, xs, ys, 0.02, 0.3)
    gate = lindblad.square_wave(2 * t, 2 * np.pi * (freq if nonseparable else 3.0), 0.0)
    params.bind(a, pa**0.5 * envelope * gate, finalize=False)
    drive = pb**0.5 * envelope * (1 + 0.2j) * (1 + 0.1 * lindblad.sine(t, angular_frequency=2.0))
    if complex_factor:
        drive = drive * lindblad.phase_modulation(1.0, y, 1.0)
    params.bind(b, drive, finalize=False)
    params.bind(d, 0.1 + y, finalize=False)
    params._finalize()
    p = lindblad.prepare_lindblad_problem(
        SimpleNamespace(H_symbolic=H, C_array=C),
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    rho = np.array([[0.6, 0.1 + 0.03j, 0.02j], [0.1 - 0.03j, 0.3, 0.01], [-0.02j, 0.01, 0.1]])
    pars = np.array(
        [[0, 0, 0.3, 3], [0.2, 0.5, 1.0, 4.1], [0.1, 1.0, 0, 3.6], [0.3, 0.2, 0.7, 2.8]]
    )
    return p, rho, pars, ["height", "power_a", "power_b", "frequency"]


@pytest.mark.parametrize("solver", ["fixed_tsit5", "fixed_ck54"])
@pytest.mark.parametrize(
    "coeff,hshare", [("off", "off"), ("auto", "off"), ("off", "auto"), ("auto", "auto")]
)
@pytest.mark.parametrize("precompute", [False, True])
def test_coefficient_and_component_sharing(gpu, solver, coeff, hshare, precompute):
    p, rho, pars, slots = sharing_model()
    kwargs = dict(
        parameter_slots=slots,
        parameter_batch=pars,
        solver=solver,
        evolution_device="cuda",
        dt=0.001,
        output="full_and_integral",
        integral_weights=[(1, 0.3)],
        output_when="saveat",
        saveat=[0, 0.051, 0.103],
        collect_stats=True,
    )
    control = lindblad.parameter_scan(
        p,
        rho,
        (0.0, 0.103),
        cuda_options=CudaSolverOptions(
            coefficient_sharing="off", hamiltonian_sharing="off", batch_size=2, graph_steps=8
        ),
        **kwargs,
    )
    got = lindblad.parameter_scan(
        p,
        rho,
        (0.0, 0.103),
        cuda_options=CudaSolverOptions(
            coefficient_sharing=coeff,
            hamiltonian_sharing=hshare,
            precompute=precompute,
            density_reduction=True,
            density_saveat=[0, 0.051, 0.103],
            batch_size=2,
            graph_steps=8,
        ),
        **kwargs,
    )
    np.testing.assert_allclose(got.values, control.values, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(
        got.metadata["density_values"], control.values[..., :-1], rtol=2e-12, atol=2e-14
    )
    assert len(got.solver_stats["chunks"]) == 2


@pytest.mark.parametrize(
    "nonseparable,complex_factor", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("hshare", ["auto", "on"])
def test_sharing_components_and_fallback_diagnostics(gpu, nonseparable, complex_factor, hshare):
    p, rho, pars, slots = sharing_model(nonseparable, complex_factor)
    options = CudaSolverOptions(
        coefficient_sharing="auto", hamiltonian_sharing=hshare, rhs="direct", batch_size=4
    )
    got = lindblad.parameter_scan(
        p,
        rho,
        (0.0, 0.103),
        parameter_slots=slots,
        parameter_batch=pars,
        evolution_device="cuda",
        solver="fixed_tsit5",
        dt=0.001,
        output="full",
        collect_stats=True,
        cuda_options=options,
    )
    old = lindblad.parameter_scan(
        p,
        rho,
        (0.0, 0.103),
        parameter_slots=slots,
        parameter_batch=pars,
        evolution_device="cuda",
        solver="fixed_tsit5",
        dt=0.001,
        output="full",
        cuda_options=CudaSolverOptions(
            rhs="direct", coefficient_sharing="off", hamiltonian_sharing="off"
        ),
    )
    np.testing.assert_allclose(got.values, old.values, rtol=2e-12, atol=2e-14)
    chunk = got.solver_stats["chunks"][0]
    assert chunk["static_subexpressions"] > 0
    if complex_factor:
        assert chunk["shared_hamiltonian_components"] == 0
    elif nonseparable:
        assert chunk["shared_hamiltonian_components"] == 1
        assert chunk["per_trajectory_hamiltonian_edges"] > 0
    else:
        assert chunk["shared_hamiltonian_components"] == 2


@pytest.mark.parametrize("name", ["coefficient_sharing", "hamiltonian_sharing"])
def test_sharing_option_validation(name):
    with pytest.raises(ValueError, match=name):
        CudaSolverOptions(**{name: "invalid"})


def test_sharing_plan_budget_without_gpu():
    from centrex_tlf.lindblad._cuda_lowering import TIME, ScalarCompiler
    from centrex_tlf.lindblad._cuda_sharing import CoefficientProgram

    p, _, batch, slots = sharing_model()
    graph = p.parameter_graph
    parameters = np.zeros((len(batch), len(graph["slot_names"])), complex)
    for k, value in enumerate(graph["base_values"]):
        if value["kind"] == "scalar":
            parameters[:, k] = complex(value["re"], value["im"])
    parameters[:, [graph["slot_names"].index(s) for s in slots]] = batch
    compiler = ScalarCompiler(graph, [c["expression"] for c in p.hamiltonian_plan["coefficients"]])
    varying = {
        k for k in range(parameters.shape[1]) if np.any(parameters[:, k] != parameters[0, k])
    }
    varied = np.asarray([bool(c.dependencies & varying) for c in compiler.coefficients])
    program = CoefficientProgram(
        p,
        compiler,
        parameters,
        varied,
        cuda_module._basis(p),
        np.zeros((101, 2)),
        CudaSolverOptions(
            rhs="direct", coefficient_sharing="auto", hamiltonian_sharing="auto", precompute_bytes=1
        ),
        cuda_module.TS_C,
    )
    assert not program.precompute
    assert program.shared_count == 0
    assert all(not n.symbolic.has(TIME) for n in program.static)
    assert len(program.amplitudes) == 2
    assert all(not (program.h_roots[k].dependencies & varying) for k in program.h_indices)


def test_shared_components_zero_drive_budget_and_outputs(gpu):
    p, rho, pars, slots = sharing_model()
    pars[:, 1:3] = 0
    kwargs = dict(
        parameter_slots=slots,
        parameter_batch=pars,
        evolution_device="cuda",
        solver="fixed_tsit5",
        dt=0.001,
        output_when="saveat",
        saveat=[0, 0.051, 0.103],
    )
    for output in ["populations", "selected", "photon_integral", "photon_rate"]:
        extra = {"integral_weights": [(1, 0.3)]} if output.startswith("photon") else {}
        if output == "selected":
            extra["output_indices"] = [(0, 1), (1, 1)]
        old = lindblad.parameter_scan(
            p,
            rho,
            (0, 0.103),
            output=output,
            **kwargs,
            **extra,
            cuda_options=CudaSolverOptions(
                rhs="direct", coefficient_sharing="off", hamiltonian_sharing="off"
            ),
        )
        new = lindblad.parameter_scan(
            p,
            rho,
            (0, 0.103),
            output=output,
            **kwargs,
            **extra,
            cuda_options=CudaSolverOptions(
                rhs="direct",
                coefficient_sharing="auto",
                hamiltonian_sharing="auto",
                precompute_bytes=1,
                density_saveat=[0.051],
                batch_size=3,
            ),
        )
        np.testing.assert_allclose(new.values, old.values, rtol=2e-12, atol=2e-14)
        assert new.metadata["density_values"].shape == (4, 1, 9)


def test_sharing_preserves_real_geometry_argument_errors(gpu):
    p, rho, pars, slots = sharing_model()
    # Isolate the helper check: a complex height must not also make detuning
    # complex and race with the separate real-diagonal validation.
    d = next(s for s in p.obe_system.H_symbolic.free_symbols if str(s) == "d")
    p.parameters.bind(d, 0.1)
    p = lindblad.prepare_lindblad_problem(
        p.obe_system, p.parameters, hamiltonian_representation="decomposed"
    )
    pars = pars.astype(complex)
    pars[1, 0] = 0.2j
    for coefficient, hamiltonian in [("off", "off"), ("auto", "off"), ("auto", "auto")]:
        with pytest.raises(RuntimeError, match="complex argument"):
            lindblad.parameter_scan(
                p,
                rho,
                (0, 0.01),
                parameter_slots=slots,
                parameter_batch=pars,
                evolution_device="cuda",
                solver="fixed_tsit5",
                dt=0.001,
                cuda_options=CudaSolverOptions(
                    rhs="direct", coefficient_sharing=coefficient, hamiltonian_sharing=hamiltonian
                ),
            )


def test_sharing_complex_overrides_keep_conjugation(gpu):
    from centrex_tlf.lindblad.parameters import RuntimeExpression

    a, b = smp.symbols("a b")
    H = smp.Matrix(
        [[0, a + 0.2 * b, b], [smp.conjugate(a + 0.2 * b), 0.1, 0], [smp.conjugate(b), 0, 0.3]]
    )
    params = lindblad.LindbladParameters()
    u, v = params.complex("u", 0.0), params.complex("v", 0.0)
    params.bind(a, RuntimeExpression(smp.conjugate(u.symbol), {u.name: u}) + v, finalize=False)
    params.bind(b, u + v, finalize=False)
    params._finalize()
    C = np.zeros((1, 3, 3))
    C[0, 0, 1] = np.sqrt(0.2)
    p = lindblad.prepare_lindblad_problem(
        SimpleNamespace(H_symbolic=H, C_array=C),
        params,
        hamiltonian_representation="decomposed",
    )
    rho = np.diag([1.0, 0.0, 0.0]).astype(complex)
    batch = np.array([[1j, -1j], [1j, 1j], [-1j, 1j]])
    for sharing in ["off", "auto", "on"]:
        a = lindblad.parameter_scan(
            p,
            rho,
            (0, 0.103),
            parameter_slots=["u", "v"],
            parameter_batch=batch,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=0.001,
            output="full",
            collect_stats=True,
            cuda_options=CudaSolverOptions(
                rhs="direct",
                coefficient_sharing="off" if sharing == "off" else "auto",
                hamiltonian_sharing=sharing,
            ),
        )
        if sharing == "off":
            reference = a.values.copy()
        np.testing.assert_allclose(a.values, reference, rtol=1e-12, atol=1e-14)
        if sharing != "off":
            assert (
                a.solver_stats["chunks"][0]["coefficient_sharing"] == "complex_parameter_override"
            )


def test_validation_and_lazy_optional_import(monkeypatch):
    p, _, _, initial = model()

    def unavailable():
        raise ImportError("CuPy missing deliberately")

    monkeypatch.setattr(cuda_module, "_cupy", unavailable)
    with pytest.raises(ValueError, match="fixed_tsit5"):
        lindblad.solve_lindblad_batch(p, initial, (0, 0.2), evolution_device="cuda", solver="tsit5")
    with pytest.raises(ValueError, match="maxiters"):
        lindblad.solve_lindblad_batch(
            p, initial, (0, 0.2), evolution_device="cuda", solver="fixed_tsit5", dt=1e-9
        )
    with pytest.raises(ImportError, match="deliberately"):
        lindblad.solve_lindblad_batch(
            p, initial, (0, 0.2), evolution_device="cuda", solver="fixed_tsit5", dt=0.01
        )
    with pytest.raises(ValueError, match="cuda_options"):
        lindblad.solve_lindblad_batch(p, initial, (0, 0.2), cuda_options=CudaSolverOptions())


@pytest.mark.parametrize("solver", ["fixed_tsit5", "fixed_ck54"])
@pytest.mark.parametrize("rhs,precompute", [("tiled", True), ("direct", False)])
def test_coherent_parameter_batches_full_integral_and_chunking(gpu, solver, rhs, precompute):
    p, H, C, initial = model()
    initial = initial[np.arange(5) % 2]
    parameters = np.array(
        [[1.2, 0.4, 0.3], [0.0, -0.7, 0.2], [0.7, 0.0, -0.4], [1.3, 0.5, 0.1], [-0.2, -0.4, 0.8]]
    )
    times = np.array([0.1, 0.137, 0.238, 0.407])
    answer = lindblad.solve_lindblad_batch(
        p,
        initial,
        (0.1, 0.407),
        solver=solver,
        evolution_device="cuda",
        dt=0.003,
        parameter_slots=["omega", "delta", "phase"],
        parameter_batch=parameters,
        output="full_and_integral",
        integral_weights=[(1, 0.5)],
        output_when="saveat",
        saveat=times,
        collect_stats=True,
        cuda_options=CudaSolverOptions(batch_size=2, rhs=rhs, precompute=precompute),
    )
    assert answer.values.shape == (5, 4, 10)
    for k, row in enumerate(parameters):
        substitutions = {
            s: dict(zip(["omega", "delta", "phase"], row, strict=False))[str(s)]
            for s in H.free_symbols
        }
        L = generator(np.array(H.subs(substitutions), complex), C)
        expected = np.stack(
            [la.expm(L * (t - 0.1)) @ np.r_[initial[k].reshape(-1, order="F"), 0] for t in times]
        )
        for j in range(len(times)):
            np.testing.assert_allclose(
                p.layout.unpack(answer.values[k, j, :-1]),
                expected[j, :-1].reshape((3, 3), order="F"),
                atol=3e-11,
            )
        np.testing.assert_allclose(answer.values[k, :, -1], expected[:, -1].real, atol=3e-11)
    assert len(answer.solver_stats["chunks"]) == 3
    assert answer.solver_stats["fixed_step"] and not answer.solver_stats["error_control"]


@pytest.mark.parametrize("solver", ["fixed_tsit5", "fixed_ck54"])
@pytest.mark.parametrize("precompute", [False, True])
def test_eom_gaussian_and_square_switches_at_output_boundaries(gpu, solver, precompute):
    t = smp.Symbol("t", real=True)
    r = smp.Symbol("drive")
    H = smp.Matrix([[0, r / 2], [smp.conjugate(r) / 2, -3]])
    C = np.zeros((1, 2, 2), complex)
    C[0, 0, 1] = np.sqrt(0.5)
    params = lindblad.LindbladParameters()
    omega = params.real("switch_omega", 2 * np.pi * 5)
    phase = params.real("switch_phase", 0.37)
    envelope = lindblad.gaussian(t, 0.2, 0.08)
    params.bind(
        r,
        12
        * envelope
        * lindblad.phase_modulation(t, 1.7, 2 * np.pi * 3)
        * lindblad.square_wave(t, omega, phase),
    )
    p = lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=C), params)
    initial = np.array([[0.4, 0.12 + 0.08j], [0.12 - 0.08j, 0.6]])
    jumps = np.array([(k * np.pi - 0.37) / (2 * np.pi * 5) for k in range(1, 6)])
    jumps = jumps[(jumps > 0) & (jumps < 0.4)]
    times = np.unique(np.r_[0, 0.031, jumps, jumps + 1e-10, 0.327, 0.4])
    answer = lindblad.solve_lindblad(
        p,
        initial,
        (0, 0.4),
        solver=solver,
        evolution_device="cuda",
        dt=0.0007,
        output="full_and_integral",
        integral_weights=[(1, 0.5)],
        saveat=times,
        cuda_options=CudaSolverOptions(precompute=precompute, density_saveat=[0.031, 0.4]),
    )
    # Independent segmented DOP853: hold only the discontinuous gate fixed.
    y = np.r_[initial.reshape(-1, order="F"), 0]
    expected = {0.0: y.copy()}
    boundaries = np.unique(np.r_[times, jumps])
    for a, b in pairwise(boundaries):
        gate = float(np.sin(2 * np.pi * 5 * ((a + b) / 2) + 0.37) >= 0)

        def rhs(time, y, gate=gate):
            drive = (
                12
                * np.exp(-0.5 * ((time - 0.2) / 0.08) ** 2)
                * np.exp(1j * 1.7 * np.sin(2 * np.pi * 3 * time))
                * gate
            )
            h = np.array([[0, drive / 2], [drive.conjugate() / 2, -3]])
            return generator(h, C) @ y

        y = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
        expected[float(b)] = y.copy()
    for k, time in enumerate(times):
        np.testing.assert_allclose(
            p.layout.unpack(answer.values[k, :-1]),
            expected[float(time)][:-1].reshape((2, 2), order="F"),
            atol=5e-9,
        )
        np.testing.assert_allclose(answer.values[k, -1], expected[float(time)][-1].real, atol=5e-9)


def test_grid_order_selected_and_sparse_density_capture(gpu):
    p, H, C, initial = model()
    times = [0.017, 0.11, 0.3]
    result = lindblad.grid_scan(
        p,
        None,
        (0, 0.3),
        rho0_batch=initial,
        scan={"delta": [-0.3, 0.9], "omega": [0.0, 1.2]},
        solver="fixed_tsit5",
        evolution_device="cuda",
        dt=0.002,
        output="selected",
        output_indices=[(1, 0), (2, 2)],
        output_when="saveat",
        saveat=times,
        cuda_options=CudaSolverOptions(batch_size=3, density_saveat=[0, 0.3]),
    )
    assert result.values.shape == (8, 3, 2)
    assert result.metadata["initial_condition_count"] == 2
    assert result.metadata["trajectory_order"] == "parameter_point_then_initial_state"
    assert result.metadata["density_values"].shape == (8, 2, 9)
    k = 0
    for delta in [-0.3, 0.9]:
        for omega in [0.0, 1.2]:
            substitutions = {
                s: {"delta": delta, "omega": omega, "phase": 0.3}[str(s)] for s in H.free_symbols
            }
            L = generator(np.array(H.subs(substitutions), complex), C)
            for rho in initial:
                expected = np.stack(
                    [
                        (la.expm(L * t) @ np.r_[rho.reshape(-1, order="F"), 0])[:-1].reshape(
                            (3, 3), order="F"
                        )
                        for t in times
                    ]
                )
                np.testing.assert_allclose(
                    result.values[k], expected[:, [1, 2], [0, 2]], atol=2e-11
                )
                np.testing.assert_allclose(
                    p.layout.unpack(result.metadata["density_values"][k, 0]), rho, atol=1e-15
                )
                k += 1


def test_static_cuda_full_reconstruction_and_residency():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p, _, _, initial = model()
    propagator = lindblad.prepare_dense_lindblad_propagator(p, rho0_batch=initial)
    times = np.linspace(0, 0.7, 41)
    cpu = propagator.evaluate(initial, times)
    gpu = propagator.evaluate(initial, times, evolution_device="cuda")
    np.testing.assert_allclose(gpu, cpu, atol=2e-13)
    cache = propagator._cuda_full_cache[torch.cuda.current_device()]
    matrices = propagator.density_matrices(initial, times, evolution_device="cuda")
    assert propagator._cuda_full_cache[torch.cuda.current_device()] is cache
    np.testing.assert_allclose(matrices, propagator.density_matrices(initial, times), atol=2e-13)
    result = lindblad.grid_scan(
        p,
        None,
        (0, 0.7),
        rho0_batch=initial,
        scan={"delta": [0.4, -0.3]},
        solver="dense_eig",
        output="full",
        output_when="saveat",
        saveat=[0, 0.7],
        parallel=False,
        evolution_device="cuda",
        collect_stats=True,
    )
    assert result.solver_stats["full_reconstruction_device"] == "cuda"
    np.testing.assert_allclose(result.values[:2], cpu[:, [0, -1]], atol=2e-13)
    propagator.clear_cuda_cache()
    assert not propagator._cuda_full_cache


def test_cuda_scalar_helpers_match_python(gpu):
    from centrex_tlf.lindblad import helper_functions as helpers
    from centrex_tlf.lindblad._cuda_kernels import HEADER
    from centrex_tlf.lindblad._cuda_lowering import ScalarCompiler

    specifications = [
        (1, (0.031, 0.02, 3.0, 0.1, 0.2, 0.4, 0.3)),
        (2, (0.031, 0.02, 3.0, 0.1, 0.2, 0.4, 0.3, 0.7)),
        (3, (0.031, 1.7, 23.0)),
        (4, (0.031, 10.0, 0.3)),
        (5, (0.031, 1.7, 23.0)),
        (6, (0.031, 10.0, 0.3)),
        (7, (0.031, 0.03, 0.07, 0.2)),
        (8, (0.031, 0.02, (1.0, 2.0), (0.0, 0.1), (0.02, 0.04), 0.2, 0.1)),
        (9, (1.2, 0.3, 1e-34)),
        (10, (0.031, 0.02, (1.0, 2.0), (0.0, 0.1), (0.02, 0.04), 0.2, 0.1, 0.3, 1e-34)),
        (11, (0.031, 0.02, 3.0, 0.1, 0.2, 0.4, 0.3, 0.3, 1e-34)),
        (12, (0.031, 0.3, 10.0, 0.2)),
        (13, (0.031, 0.02, 0.04)),
        (14, (0.031, (0.0, 0.02, 0.1), (0.2, 0.7, 0.3))),
        (15, (0.031, 0.1, 0.2)),
        (16, (0.031, (0.0, 0.02, 0.1), (0.2, 0.7, 0.3))),
    ]
    graph = dict(slot_names=[], base_values=[], compounds=[])
    expressions = []
    for fid, arguments in specifications:
        instructions = []
        for argument in arguments:
            if isinstance(argument, tuple):
                slot = len(graph["base_values"])
                graph["slot_names"].append(str(slot))
                graph["base_values"].append(
                    dict(kind="tuple", items=[dict(re=x, im=0.0) for x in argument])
                )
                instructions.append(dict(op=2, index=slot))
            else:
                instructions.append(dict(op=1, re=argument, im=0.0))
        instructions.append(dict(op=13, function=fid, argc=len(arguments)))
        expressions.append(dict(instructions=instructions))
    compiler = ScalarCompiler(graph, expressions)
    function = compiler.function("evaluate", list(range(1, 17)))
    source = (
        HEADER
        + "\n".join(compiler.helpers)
        + function
        + """
    extern "C" __global__ void probe(C* out,int* error){
        evaluate(nullptr,1,0,0.,0.,out,1,error);
    }"""
    )
    kernel = gpu.RawKernel(source, "probe", options=("--std=c++17",))
    output = gpu.empty(17, gpu.complex128)
    error = gpu.zeros(1, gpu.int32)
    kernel((1,), (1,), (output, error))
    expected = [
        getattr(helpers, helpers.HELPER_FUNCTION_NAMES[fid])(*args) for fid, args in specifications
    ]
    np.testing.assert_allclose(output.get()[1:], expected, rtol=2e-13, atol=1e-13)
    assert error.get()[0] == 0


@pytest.mark.parametrize("solver", ["fixed_tsit5", "fixed_ck54"])
def test_photon_decay_zero_span_and_schedule(gpu, solver):
    C = np.zeros((1, 2, 2), complex)
    C[0, 0, 1] = np.sqrt(0.5)
    p = lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=smp.zeros(2), C_array=C), {})
    initial = np.diag([0.0, 1.0])
    options = CudaSolverOptions(step_schedule=[(0.2, 0.002)], density_saveat=[0.0, 0.4])
    result = lindblad.solve_lindblad(
        p,
        initial,
        (0.0, 0.4),
        solver=solver,
        evolution_device="cuda",
        dt=0.003,
        output="photon_integral",
        integral_weights=[(1, 0.5)],
        saveat=[0.0, 0.017, 0.4],
        cuda_options=options,
    )
    np.testing.assert_allclose(result.values[:, 0], -np.expm1(-0.5 * result.t), atol=2e-12)
    np.testing.assert_allclose(
        p.layout.unpack(result.metadata["density_values"][0, -1]),
        np.diag([1 - np.exp(-0.2), np.exp(-0.2)]),
        atol=2e-12,
    )
    zero = lindblad.solve_lindblad(
        p,
        initial,
        (0.3, 0.3),
        solver=solver,
        evolution_device="cuda",
        dt=0.003,
        output="full",
        output_when="final",
    )
    np.testing.assert_array_equal(zero.density_matrices()[0], initial)


def test_scanned_switch_frequencies_and_phases(gpu):
    t = smp.Symbol("t", real=True)
    drive = smp.Symbol("drive", real=True)
    params = lindblad.LindbladParameters()
    omega = params.real("omega", 10.0)
    phase = params.real("phase", 0.1)
    params.bind(drive, 3 * lindblad.square_wave(t, omega, phase))
    p = lindblad.prepare_lindblad_problem(
        SimpleNamespace(
            H_symbolic=smp.Matrix([[0, drive / 2], [drive / 2, -1]]), C_array=np.zeros((0, 2, 2))
        ),
        params,
    )
    initial = np.diag([1.0, 0.0])
    rows = np.array([[10.0, 0.1], [21.0, -0.3], [-13.0, 0.7]])
    result = lindblad.solve_lindblad_batch(
        p,
        np.repeat(initial[None], 3, axis=0),
        (0.0, 0.4),
        parameter_slots=["omega", "phase"],
        parameter_batch=rows,
        solver="fixed_tsit5",
        evolution_device="cuda",
        dt=0.003,
        output="full",
        cuda_options=CudaSolverOptions(batch_size=2),
    )
    for k, (w, phi) in enumerate(rows):
        jumps = [(j * np.pi - phi) / w for j in range(-5, 6)]
        edges = sorted([0.0, 0.4] + [x for x in jumps if 0 < x < 0.4])
        rho = initial.astype(complex)
        for a, b in pairwise(edges):
            amplitude = 3 * float(np.sin(w * (a + b) / 2 + phi) >= 0)
            U = la.expm(-1j * np.array([[0, amplitude / 2], [amplitude / 2, -1]]) * (b - a))
            rho = U @ rho @ U.conj().T
        np.testing.assert_allclose(p.layout.unpack(result.values[k]), rho, atol=1e-11)


def test_static_cuda_large_cartesian_batch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p, _, _, initial = model()
    prop = lindblad.prepare_dense_lindblad_propagator(p, rho0_batch=initial)
    initial = initial[np.arange(23) % 2]
    times = np.linspace(0.0, 1.0, 103)
    np.testing.assert_allclose(
        prop.density_matrices(initial, times, evolution_device="cuda"),
        prop.density_matrices(initial, times),
        atol=3e-13,
    )


def test_reject_invalid_initial_states_and_output_policy(monkeypatch):
    p, _, _, initial = model()

    def unavailable():
        raise AssertionError("validation must precede CUDA loading")

    monkeypatch.setattr(cuda_module, "_cupy", unavailable)
    broken = initial.copy()
    broken[0, 0, 1] = 1j
    with pytest.raises(ValueError, match="Hermitian"):
        lindblad.solve_lindblad(
            p, broken[0], (0.0, 0.1), solver="fixed_tsit5", evolution_device="cuda", dt=0.01
        )
    with pytest.raises(ValueError, match="Hermitian"):
        lindblad.grid_scan(
            p,
            None,
            (0.0, 0.1),
            rho0_batch=broken,
            scan={"omega": [1.0]},
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=0.01,
        )
    with pytest.raises(ValueError, match="output_when"):
        lindblad.solve_lindblad_batch(
            p,
            initial,
            (0.0, 0.1),
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=0.01,
            output_when="sometimes",
        )
    with pytest.raises(ValueError, match="terminal events"):
        lindblad.solve_lindblad_batch(
            p,
            initial,
            (0.0, 0.1),
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=0.01,
            stop_event=object(),
        )


def test_nonlinear_switches_require_declared_jumps():
    from centrex_tlf.lindblad._cuda_lowering import ScalarCompiler

    graph = dict(slot_names=[], base_values=[], compounds=[])
    # sin(t**2): an affine-frequency boundary detector cannot infer its roots.
    expression = dict(
        instructions=[
            dict(op=4),
            dict(op=1, re=2.0, im=0.0),
            dict(op=9),
            dict(op=1, re=1.0, im=0.0),
            dict(op=1, re=0.0, im=0.0),
            dict(op=13, function=4, argc=3),
        ]
    )
    compiler = ScalarCompiler(graph, [expression])
    with pytest.raises(ValueError, match="nonlinear"):
        compiler.discontinuities(np.empty((1, 0)), (0.0, 4.0), None)
    np.testing.assert_allclose(
        compiler.discontinuities(np.empty((1, 0)), (0.0, 4.0), [np.sqrt(np.pi)]), [np.sqrt(np.pi)]
    )


@pytest.mark.parametrize("reduce", [True, False])
def test_static_cuda_dark_states_and_empty_times(reduce):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = lindblad.prepare_lindblad_problem(
        SimpleNamespace(H_symbolic=smp.zeros(2), C_array=np.zeros((0, 2, 2))), {}
    )
    rho = np.diag([0.3, 0.7]).astype(complex)
    if not reduce:
        rho[0, 1], rho[1, 0] = 0.1j, -0.1j
    prop = lindblad.prepare_dense_lindblad_propagator(p, rho0_batch=rho[None], reduce=reduce)
    actual = prop.density_matrices(rho[None], [0.2, 0.8], t0=0.2, evolution_device="cuda")
    np.testing.assert_allclose(actual, np.broadcast_to(rho, actual.shape), atol=1e-15)
    assert prop.density_matrices(rho[None], [], evolution_device="cuda").shape == (1, 0, 2, 2)


def test_static_cuda_full_persistent_worker_session():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p, _, _, initial = model()
    options = dict(
        rho0_batch=initial,
        scan={"omega": [0.7, 1.2]},
        solver="dense_eig",
        output="full",
        output_when="saveat",
        saveat=[0.0, 0.07, 0.4],
    )
    cpu = lindblad.grid_scan(p, None, (0.0, 0.4), parallel=False, **options)
    with lindblad.DenseLindbladSession(p, threads=2) as session:
        for _ in range(2):
            gpu = lindblad.grid_scan(
                p, None, (0.0, 0.4), evolution_device="cuda", dense_session=session, **options
            )
            np.testing.assert_allclose(gpu.values, cpu.values, atol=3e-13)


@pytest.mark.parametrize("batch,mode", [(512, "auto"), (6, "auto"), (6, "on")])
def test_specialized_fma_matches_generic_with_fsal_and_output_boundaries(gpu, batch, mode):
    a = smp.Symbol("a", real=True)
    H = smp.Matrix([[0, a * (1 + 0.3j), 0], [a * (1 - 0.3j), 0.4, 0], [0, 0, 0.2]])
    C = np.zeros((2, 3, 3), complex)
    C[0, 0, 1], C[1, 2, 1] = np.sqrt(0.3), np.sqrt(0.2)
    params = lindblad.LindbladParameters()
    t, scale = params.time(), params.real("scale", 1.0)
    params.bind(a, scale * (1 + 0.2 * lindblad.sine(t, angular_frequency=2.0)))
    p = lindblad.prepare_lindblad_problem(
        SimpleNamespace(H_symbolic=H, C_array=C),
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    v = np.array([1, 0.2j, 0.3], complex)
    v /= np.linalg.norm(v)
    initial = np.broadcast_to(np.outer(v, v.conj()), (batch, 3, 3)).copy()
    options = dict(
        parameter_slots=["scale"],
        parameter_batch=np.linspace(0.0, 1.2, batch)[:, None],
        solver="fixed_tsit5",
        evolution_device="cuda",
        dt=0.001,
        output="full",
        output_when="saveat",
        saveat=[0.0, 0.007, 0.023, 0.103],
        collect_stats=True,
    )

    def solve(specialization):
        return lindblad.solve_lindblad_batch(
            p,
            initial,
            (0.0, 0.103),
            **options,
            cuda_options=CudaSolverOptions(
                batch_size=512,
                rhs="direct",
                hamiltonian_sharing="on",
                density_reduction=True,
                rhs_specialization=specialization,
            ),
        )

    reference = solve("off")
    actual = solve(mode)
    np.testing.assert_allclose(actual.values, reference.values, rtol=1e-11, atol=1e-13)
    chunk = actual.solver_stats["chunks"][0]
    specialized = mode == "on" or batch == 512
    assert chunk["rhs_specialization"]["kernel"] == (
        "specialized_fma" if specialized else "generic"
    )
    assert chunk["diagonal_stage_reuse"] == specialized
    if specialized:
        assert chunk["block_size"] == (128 if batch == 512 else 64)


@pytest.mark.parametrize(
    "name,value",
    [
        ("rhs_specialization", "invalid"),
        ("specialized_block_size", 32),
        ("specialized_block_size", True),
    ],
)
def test_specialized_option_validation(name, value):
    with pytest.raises(ValueError, match=name):
        CudaSolverOptions(**{name: value})


def test_forced_specialization_rejects_multiple_components(gpu):
    p, rho, pars, slots = sharing_model()
    with pytest.raises(ValueError, match="requires_single_shared_component"):
        lindblad.parameter_scan(
            p,
            rho,
            (0.0, 0.01),
            parameter_slots=slots,
            parameter_batch=pars,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=0.001,
            cuda_options=CudaSolverOptions(
                rhs="direct", hamiltonian_sharing="on", rhs_specialization="on"
            ),
        )
