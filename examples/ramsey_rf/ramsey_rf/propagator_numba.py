"""V5: dense midpoint propagation with a Numba-compiled rotate step.

The Hamiltonian diagonalization still uses SciPy LAPACK. Numba only replaces
the Python/NumPy sequence

    V.conj().T @ Psi -> phase multiply -> V @ tmp

with a compiled loop. This is intentionally optional; import this module only
when `numba` is available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - optional benchmark dependency
    raise ImportError(
        "V5 requires numba. Run with `uv run --with numba ...` or install numba."
    ) from exc

from .propagator import HMidFn, PropagationResult, _diagonalize


@njit(cache=True)
def _rotate_eigenbasis_numba(
    psi: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    dt: float,
) -> np.ndarray:
    n, n_cols = psi.shape
    tmp = np.empty((n, n_cols), dtype=np.complex128)
    out = np.empty((n, n_cols), dtype=np.complex128)

    for a in range(n):
        phase = np.exp((-1j) * eigvals[a] * dt)
        for col in range(n_cols):
            acc = 0.0 + 0.0j
            for i in range(n):
                acc += np.conj(eigvecs[i, a]) * psi[i, col]
            tmp[a, col] = phase * acc

    for i in range(n):
        for col in range(n_cols):
            acc = 0.0 + 0.0j
            for a in range(n):
                acc += eigvecs[i, a] * tmp[a, col]
            out[i, col] = acc

    return out


def warm_up_numba_rotate(n: int = 4, n_cols: int = 1) -> None:
    """Compile the rotate kernel before timing a benchmark."""
    psi = np.zeros((n, n_cols), dtype=np.complex128)
    psi[0, :] = 1.0
    eigvals = np.arange(n, dtype=np.float64)
    eigvecs = np.eye(n, dtype=np.complex128)
    _rotate_eigenbasis_numba(psi, eigvals, eigvecs, 1e-9)


def step_eigh_numba(
    Psi: npt.NDArray[np.complex128],
    t_k: float,
    t_kp1: float,
    H_mid_fn: HMidFn,
) -> npt.NDArray[np.complex128]:
    t_mid = 0.5 * (t_k + t_kp1)
    dt = t_kp1 - t_k
    H_mid = H_mid_fn(t_mid)
    D, V = _diagonalize(H_mid)
    return _rotate_eigenbasis_numba(Psi, D, V, dt)


def propagate_midpoint_numba(
    Psi0: npt.NDArray[np.complex128],
    t_grid: npt.NDArray[np.float64],
    H_at_t: HMidFn,
    *,
    store_norm: bool = True,
    store_snapshots: bool = False,
) -> PropagationResult:
    """Propagate with SciPy `eigh` plus a Numba rotate/apply kernel."""
    Psi = np.asarray(Psi0, dtype=np.complex128).copy()
    if Psi.ndim != 2:
        raise ValueError(f"Psi0 must be 2-D (N, K); got shape {Psi.shape}")
    n_times = t_grid.shape[0]
    if n_times < 2:
        raise ValueError("t_grid must have at least 2 points")

    warm_up_numba_rotate(Psi.shape[0], Psi.shape[1])

    norm_trace: Optional[npt.NDArray[np.float64]] = None
    snapshots: Optional[npt.NDArray[np.complex128]] = None
    if store_norm:
        norm_trace = np.empty((n_times, Psi.shape[1]), dtype=np.float64)
        norm_trace[0, :] = np.sum(np.abs(Psi) ** 2, axis=0)
    if store_snapshots:
        snapshots = np.empty((n_times, Psi.shape[0], Psi.shape[1]), dtype=np.complex128)
        snapshots[0] = Psi

    for k in range(n_times - 1):
        Psi = step_eigh_numba(Psi, float(t_grid[k]), float(t_grid[k + 1]), H_at_t)
        if norm_trace is not None:
            norm_trace[k + 1, :] = np.sum(np.abs(Psi) ** 2, axis=0)
        if snapshots is not None:
            snapshots[k + 1] = Psi

    return PropagationResult(
        Psi_final=Psi,
        norm_trace=norm_trace,
        snapshots=snapshots,
        t_grid=np.asarray(t_grid, dtype=np.float64),
    )
