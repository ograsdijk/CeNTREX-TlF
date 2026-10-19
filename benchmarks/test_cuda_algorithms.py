"""Independent structural and integrator checks for the isolated prototypes."""

from types import SimpleNamespace

import check_cuda_algorithms as experiment
import numpy as np
import sympy as smp

from centrex_tlf import lindblad


def toy():
    t = smp.Symbol("t", real=True)
    drive = smp.Symbol("drive")
    # Complex, noncommuting drive plus single-source decay. Third level is a sink.
    H = smp.Matrix([[0, drive, 0], [smp.conjugate(drive), 2e6, 0], [0, 0, 0]])
    C = np.zeros((2, 3, 3), complex)
    C[0, 0, 1], C[1, 2, 1] = np.sqrt(4e5), np.sqrt(1e5)
    params = lindblad.LindbladParameters()
    params.bind(drive, 3e5 * lindblad.phase_modulation(t, 1.2, 2 * np.pi * 1e6))
    prepared = lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=H, C_array=C), params)
    return prepared


def test_packed_generator_matches_independent_complex_commutator():
    p = toy()
    components, labels = experiment.matrices(p)
    rng = np.random.default_rng(631)
    Z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = Z + Z.conj().T
    c = 3e5 * np.exp(0.7j)
    coeff = [1.0, c]
    packed_derivative = sum(
        A @ p.layout.pack(rho) * (complex(coeff[k]).imag if imag else complex(coeff[k]).real)
        for A, (k, imag) in zip(components, labels, strict=True)
    )
    H = np.array([[0, c, 0], [c.conjugate(), 2e6, 0], [0, 0, 0]], complex)
    expected = -1j * (H @ rho - rho @ H)
    for target, rate in [(0, 4e5), (2, 1e5)]:
        C = np.zeros((3, 3), complex)
        C[target, 1] = np.sqrt(rate)
        D = C.conj().T @ C
        expected += C @ rho @ C.conj().T - 0.5 * (D @ rho + rho @ D)
    np.testing.assert_allclose(packed_derivative, p.layout.pack(expected), rtol=1e-13, atol=2e-9)


def test_closure_retains_sink_populations_but_not_sink_coherences():
    p = toy()
    components, _ = experiment.matrices(p)
    initial = np.zeros((1, 9))
    initial[0, 0] = 1.0
    keep = experiment.closure(components, initial)
    assert set(keep) == {0, 1, 2, 3, 4}
    omitted = np.setdiff1d(np.arange(9), keep)
    assert all(A[omitted][:, keep].nnz == 0 for A in components)


def test_cf4_preserves_full_density_and_converges():
    p = toy()
    packed = np.zeros((1, 9))
    packed[0, 0] = 1.0
    pars = np.empty((1, 0))
    span = (0.0, 1e-6)
    components, labels = experiment.matrices(p)
    keep = experiment.closure(components, packed)
    ref = lindblad.solve_lindblad_batch(
        p, packed, span, solver="fixed_tsit5", evolution_device="cuda", dt=1e-9, output="full"
    )
    errors = []
    for h in (80e-9, 40e-9):
        g, src = experiment.allocate(p, packed, pars, [], span, h)
        experiment.exponential(g, src, components, labels, keep, packed, span, 18, 1)
        host, _ = experiment.bench.previous.replay(g, span)
        result = p.layout.pack_batch(experiment.bench.internal_density(g, host, p.layout))
        errors.append(float(np.max(abs(result - ref.values))))
        assert abs(result[0, :3].sum() - 1) < 1e-12
    assert errors[1] < errors[0] / 8
    assert errors[1] < 1e-5


def test_compact_structured_kernel_reconstructs_coherences_and_sink():
    p = toy()
    state = np.array([np.sqrt(0.5), 1j * np.sqrt(0.5), 0.0])
    packed = p.layout.pack(np.outer(state, state.conj()))[None]
    pars = np.empty((1, 0))
    span = (0.0, 1e-6)
    components, _ = experiment.matrices(p)
    keep = experiment.closure(components, packed)
    reference = lindblad.solve_lindblad_batch(
        p, packed, span, solver="fixed_tsit5", evolution_device="cuda", dt=1e-9, output="full"
    )
    g, src = experiment.allocate(p, packed, pars, [], span, 1e-9)
    experiment.compact_structured(g, src, p, keep, packed, span)
    host, _ = experiment.bench.previous.replay(g, span)
    i, j = g.compact_pairs
    z = (host[: g.P] + 1j * host[g.P :]).reshape(-1, g.B).T
    rho = np.zeros((1, 3, 3), complex)
    rho[:, i, j] = z
    rho[:, j, i] = z.conj()
    np.testing.assert_allclose(p.layout.pack_batch(rho), reference.values, atol=1e-13, rtol=1e-12)
