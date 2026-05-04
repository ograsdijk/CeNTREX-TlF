"""V4: sparse Krylov midpoint propagation.

This variant keeps the full basis but replaces dense eigendecomposition on
short active steps with `scipy.sparse.linalg.expm_multiply`. Long inactive
steps remain on the exact dense `eigh` path because Krylov over millisecond
constant-H spans is inefficient for GHz-scale Hamiltonians.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import expm_multiply

from .propagator import PropagationResult, _diagonalize


@dataclass
class KrylovHybridStats:
    """Step accounting for the hybrid Krylov/eigh propagator."""

    n_krylov_steps: int
    n_eigh_steps: int
    krylov_dt_max: float


@dataclass
class KrylovPropagationResult(PropagationResult):
    stats: KrylovHybridStats


def _assemble_sparse(
    components: Sequence[csr_matrix],
    coeffs: npt.NDArray[np.float64],
) -> csr_matrix:
    H = components[0] * complex(coeffs[0])
    for i in range(1, len(components)):
        w = coeffs[i]
        if w != 0.0:
            H = H + components[i] * complex(w)
    return H.tocsr()


def _step_exact_from_sparse(
    Psi: npt.NDArray[np.complex128],
    H_sparse: csr_matrix,
    dt: float,
) -> npt.NDArray[np.complex128]:
    D, V = _diagonalize(H_sparse.toarray())
    tmp = V.conj().T @ Psi
    tmp *= np.exp(-1j * D * dt)[:, None]
    return V @ tmp


def _step_krylov_shifted(
    Psi: npt.NDArray[np.complex128],
    H_sparse: csr_matrix,
    dt: float,
) -> npt.NDArray[np.complex128]:
    n = H_sparse.shape[0]
    trace_shift = complex(H_sparse.diagonal().sum() / n)
    H_shifted = H_sparse - trace_shift * identity(n, format="csr", dtype=np.complex128)
    out = expm_multiply((-1j * dt) * H_shifted, Psi, traceA=0.0)
    return np.exp(-1j * trace_shift * dt) * out


def propagate_midpoint_krylov_hybrid(
    Psi0: npt.NDArray[np.complex128],
    t_grid: npt.NDArray[np.float64],
    components: Sequence[npt.NDArray[np.complex128]],
    coeffs_at_t: Callable[[float], npt.NDArray[np.float64]],
    *,
    krylov_dt_max: Optional[float] = None,
    store_norm: bool = True,
    store_snapshots: bool = False,
) -> KrylovPropagationResult:
    """Propagate using sparse Krylov on fine steps and exact eigh on coarse steps.

    `components` and `coeffs_at_t` represent `H(t) = sum_i coeff_i(t) H_i` in
    rad/s. If `krylov_dt_max` is not supplied, the smallest timestep in
    `t_grid` is treated as the active-step size and any step up to twice that
    value uses Krylov.
    """
    Psi = np.asarray(Psi0, dtype=np.complex128).copy()
    if Psi.ndim != 2:
        raise ValueError(f"Psi0 must be 2-D (N, K); got shape {Psi.shape}")
    if t_grid.shape[0] < 2:
        raise ValueError("t_grid must have at least 2 points")
    if len(components) == 0:
        raise ValueError("components must not be empty")

    sparse_components = [
        csr_matrix(np.asarray(comp, dtype=np.complex128)) for comp in components
    ]
    n_components = len(sparse_components)

    dt_values = np.diff(t_grid)
    if krylov_dt_max is None:
        krylov_dt_max = 2.0 * float(dt_values.min())

    n_times = t_grid.shape[0]
    norm_trace: Optional[npt.NDArray[np.float64]] = None
    snapshots: Optional[npt.NDArray[np.complex128]] = None
    if store_norm:
        norm_trace = np.empty((n_times, Psi.shape[1]), dtype=np.float64)
        norm_trace[0, :] = np.sum(np.abs(Psi) ** 2, axis=0)
    if store_snapshots:
        snapshots = np.empty((n_times, Psi.shape[0], Psi.shape[1]), dtype=np.complex128)
        snapshots[0] = Psi

    n_krylov = 0
    n_eigh = 0
    for k in range(n_times - 1):
        t_mid = 0.5 * (float(t_grid[k]) + float(t_grid[k + 1]))
        dt = float(t_grid[k + 1] - t_grid[k])
        coeffs = np.asarray(coeffs_at_t(t_mid), dtype=np.float64)
        if coeffs.shape != (n_components,):
            raise ValueError(
                f"coeffs_at_t returned shape {coeffs.shape}; expected {(n_components,)}"
            )
        H_sparse = _assemble_sparse(sparse_components, coeffs)
        if dt <= krylov_dt_max:
            Psi = _step_krylov_shifted(Psi, H_sparse, dt)
            n_krylov += 1
        else:
            Psi = _step_exact_from_sparse(Psi, H_sparse, dt)
            n_eigh += 1

        if norm_trace is not None:
            norm_trace[k + 1, :] = np.sum(np.abs(Psi) ** 2, axis=0)
        if snapshots is not None:
            snapshots[k + 1] = Psi

    return KrylovPropagationResult(
        Psi_final=Psi,
        norm_trace=norm_trace,
        snapshots=snapshots,
        t_grid=np.asarray(t_grid, dtype=np.float64),
        stats=KrylovHybridStats(
            n_krylov_steps=n_krylov,
            n_eigh_steps=n_eigh,
            krylov_dt_max=float(krylov_dt_max),
        ),
    )
