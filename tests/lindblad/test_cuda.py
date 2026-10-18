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
