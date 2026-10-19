"""Small algebra/duality tests for the benchmark-only RC prototypes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from centrex_tlf.lindblad.reference_dense import apply_structured_dissipator_reference
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

from rc_block_adjoint_prototypes import (
    adjoint_rhs_from_hamiltonian,
    block_rhs_from_hamiltonian,
    pack_density_block,
    unpack_density_block,
)


def _toy_prepared() -> SimpleNamespace:
    layout = PackedHermitianLayout(3)
    jumps = [
        {"target": 0, "source": 2, "rate": 0.7},
        {"target": 1, "source": 2, "rate": 0.3},
    ]
    return SimpleNamespace(
        layout=layout,
        structured_jumps=jumps,
        source_decay_rates=np.asarray([0.0, 0.0, 1.0]),
    )


def _hermitian_rng(rng: np.random.Generator, n: int) -> np.ndarray:
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (matrix + matrix.conjugate().T) / 2.0


def test_packed_block_roundtrip_and_shared_h_rhs_matches_single_state_rhs() -> None:
    rng = np.random.default_rng(1357)
    prepared = _toy_prepared()
    n = prepared.layout.n
    states = []
    for _ in range(4):
        factor = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        rho = factor @ factor.conjugate().T
        rho = (rho + rho.conjugate().T) / 2.0
        rho /= np.trace(rho).real
        states.append(rho)
    batch = np.asarray(states)
    packed = pack_density_block(batch, prepared.layout)
    restored = unpack_density_block(packed, prepared.layout)
    np.testing.assert_allclose(restored, batch, rtol=0.0, atol=1e-16)

    hamiltonian = _hermitian_rng(rng, n)
    block_rhs = block_rhs_from_hamiltonian(prepared, packed, sp.csr_matrix(hamiltonian))
    block_derivative = unpack_density_block(block_rhs, prepared.layout)
    for idx, rho in enumerate(batch):
        expected = -1j * (hamiltonian @ rho - rho @ hamiltonian)
        expected += apply_structured_dissipator_reference(
            prepared.structured_jumps, prepared.source_decay_rates, rho
        )
        np.testing.assert_allclose(block_derivative[idx], expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(block_derivative[idx], block_derivative[idx].conjugate().T)


def test_backward_adjoint_satisfies_duality_with_complex_h_and_decay() -> None:
    prepared = _toy_prepared()
    layout = prepared.layout
    rng = np.random.default_rng(8642)
    n = layout.n
    h0 = _hermitian_rng(rng, n)
    h1 = _hermitian_rng(rng, n)

    def hamiltonian(t: float) -> np.ndarray:
        return h0 + 0.37 * np.sin(2.3 * t) * h1

    factor = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho0 = factor @ factor.conjugate().T
    rho0 /= np.trace(rho0).real
    terminal = np.zeros((n, n), dtype=np.complex128)
    terminal[1, 1] = 1.0
    weight = np.diag([0.2, 0.0, 0.4]).astype(np.complex128)
    t0, t1 = 0.0, 0.8

    def forward_rhs(t: float, y: np.ndarray) -> np.ndarray:
        rho = y[:-1].reshape(n, n)
        h = hamiltonian(t)
        drho = -1j * (h @ rho - rho @ h)
        drho += apply_structured_dissipator_reference(
            prepared.structured_jumps, prepared.source_decay_rates, rho
        )
        running = np.trace(weight @ rho)
        return np.concatenate((drho.reshape(-1), np.asarray([running])))

    def forward_objective(initial: np.ndarray) -> tuple[np.ndarray, float]:
        forward0 = np.concatenate((initial.reshape(-1), np.asarray([0.0 + 0.0j])))
        forward = solve_ivp(
            forward_rhs,
            (t0, t1),
            forward0,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
            first_step=1e-3,
        )
        assert forward.success
        return forward.y[:-1, -1].reshape(n, n), float(forward.y[-1, -1].real)

    rho1, integrated = forward_objective(rho0)

    def adjoint_rhs(t: float, packed_lambda: np.ndarray) -> np.ndarray:
        return adjoint_rhs_from_hamiltonian(
            prepared, packed_lambda, hamiltonian(t), weight
        )

    backward = solve_ivp(
        adjoint_rhs,
        (t1, t0),
        layout.pack(terminal),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        first_step=1e-3,
        t_eval=[t0],
    )
    assert backward.success
    lambda0 = layout.unpack(backward.y[:, -1])

    adjoint_value = np.trace(lambda0 @ rho0).real
    forward_value = np.trace(terminal @ rho1).real + integrated
    np.testing.assert_allclose(adjoint_value, forward_value, rtol=2e-10, atol=2e-11)

    onehot_responses = np.diag(lambda0).real
    for idx in range(n):
        onehot = np.zeros((n, n), dtype=np.complex128)
        onehot[idx, idx] = 1.0
        onehot_final, onehot_integral = forward_objective(onehot)
        direct = np.trace(terminal @ onehot_final).real + onehot_integral
        np.testing.assert_allclose(onehot_responses[idx], direct, rtol=2e-10, atol=2e-11)

    mixture_weights = np.asarray([0.2, 0.5, 0.3])
    mixture = np.diag(mixture_weights).astype(np.complex128)
    mixture_final, mixture_integral = forward_objective(mixture)
    mixture_direct = np.trace(terminal @ mixture_final).real + mixture_integral
    mixture_adjoint = float(np.dot(mixture_weights, onehot_responses))
    np.testing.assert_allclose(mixture_adjoint, mixture_direct, rtol=2e-10, atol=2e-11)


def test_adjoint_rhs_sign_matches_running_integral_convention() -> None:
    prepared = _toy_prepared()
    hamiltonian = np.asarray(
        [[0.3, 0.2 + 0.1j, 0.0], [0.2 - 0.1j, -0.1, 0.05j], [0.0, -0.05j, 0.4]],
        dtype=np.complex128,
    )
    lam = np.diag([0.2, 0.4, 0.7]).astype(np.complex128)
    weight = np.diag([0.1, 0.0, 0.3]).astype(np.complex128)
    derivative = prepared.layout.unpack(
        adjoint_rhs_from_hamiltonian(prepared, prepared.layout.pack(lam), hamiltonian, weight)
    )
    epsilon = 1e-7
    rho = np.diag([0.2, 0.3, 0.5]).astype(np.complex128)
    forward = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    forward += apply_structured_dissipator_reference(
        prepared.structured_jumps, prepared.source_decay_rates, rho
    )
    left = np.trace((lam + epsilon * derivative) @ (rho + epsilon * forward)).real
    right = np.trace(lam @ rho).real - epsilon * np.trace(weight @ rho).real
    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-8)
