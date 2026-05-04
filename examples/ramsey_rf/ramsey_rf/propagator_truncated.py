"""V3a: midpoint piecewise-constant unitary propagation in a TRUNCATED subspace.

For our RF Ramsey problem the dynamics are confined to the J=1 dressed manifold
(plus its Stark-mixed J=0/2/3 admixtures at high E), which spans only K~16-32
of the N=196 basis states at Jmax=6. Working in that K-dim subspace drops the
per-step eigh cost from O(N³) to O(K³) — for K=32, N=196: ~230× per eigh.

The truncation matrix T (N×K, orthonormal columns) is fixed at simulator init
("static projection"). The propagator works entirely with the projected
coefficient vector c(t) = T† Psi(t) and the projected Hamiltonian
H_proj(t) = T† H(t) T:

    H_proj = T† H_mid T            # K×K Hermitian
    D, V   = eigh(H_proj)          # K eigenvalues
    c      = V (exp(-i D dt) (V† c))

Final Psi_final = T c_final is reconstructed in the full N-dim basis.

Approximation error: any norm of Psi(t) outside the K-dim subspace at time t
is permanently lost. For the Ramsey problem this is small (~1% leakage out
of the J=1 dressed manifold across a 22 ms trajectory) when T is chosen as
the K eigenstates of H at the high-field plateau with largest overlap to the
J=1 manifold — see `select_subspace_*` helpers below.

The helper `select_subspace_by_overlap(H, K, target_vector)` is the most
general choice: pick K eigenvectors of H most strongly coupled to the supplied
target. For a single Psi0 this gives a clean physical truncation.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh as _scipy_eigh
from scipy.optimize import linear_sum_assignment

from .propagator import HMidFn, PropagationResult, _diagonalize


def select_subspace_by_overlap(
    H: npt.NDArray[np.complex128],
    K: int,
    target_vectors: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.intp]]:
    """Pick K eigenvectors of H with largest summed overlap to the target vectors.

    Args:
        H: (N, N) Hermitian Hamiltonian to diagonalize.
        K: subspace dimension.
        target_vectors: (N,) or (N, M) — the bare/dressed states whose
            "neighborhood" defines the relevant subspace. For a single
            initial state, M=1 is fine; the K eigenvectors with largest
            |⟨v|target⟩|² are returned.

    Returns:
        T: (N, K) orthonormal truncation matrix (the chosen eigenvectors as columns).
        idx: (K,) indices into the eigenvalue ordering.
    """
    if target_vectors.ndim == 1:
        target_vectors = target_vectors.reshape(-1, 1)
    D, V = _scipy_eigh(H, driver="evr")
    del D
    # Per-eigenvector summed overlap probability
    overlaps = np.sum(np.abs(V.conj().T @ target_vectors) ** 2, axis=1)  # (N,)
    if K > V.shape[1]:
        raise ValueError(f"K={K} exceeds basis size {V.shape[1]}")
    idx = np.argsort(overlaps)[::-1][:K]
    idx_sorted = np.sort(idx)
    T = V[:, idx_sorted].astype(np.complex128, copy=True)
    return T, idx_sorted


def select_subspace_J_manifold(
    H: npt.NDArray[np.complex128],
    QN: npt.NDArray,
    J_values: Sequence[int],
    *,
    K: Optional[int] = None,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.intp]]:
    """Pick eigenvectors of H whose dominant bare-basis character is in the
    requested J values. Optionally cap to the K eigenvectors with the largest
    summed J-projected probability.
    """
    j_mask = np.array([bs.J in set(J_values) for bs in QN])
    j_indices = np.flatnonzero(j_mask)
    if j_indices.size == 0:
        raise ValueError(f"no basis states with J in {J_values}")

    D, V = _scipy_eigh(H, driver="evr")
    del D
    # Probability of each eigenvector being in the requested J subspace
    j_probs = np.sum(np.abs(V[j_indices, :]) ** 2, axis=0)  # (N,)
    if K is None:
        # Take all eigenvectors that are *predominantly* in the J subspace
        K = int((j_probs > 0.5).sum())
        if K == 0:
            raise ValueError("no eigenvectors are predominantly in the requested J")
    idx = np.argsort(j_probs)[::-1][:K]
    idx_sorted = np.sort(idx)
    T = V[:, idx_sorted].astype(np.complex128, copy=True)
    return T, idx_sorted


def track_subspace_bases(
    points: Sequence[float],
    H_at_point: Callable[[float], npt.NDArray[np.complex128]],
    T_initial: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Track a K-dimensional eigenspace across a sequence of Hamiltonians.

    `T_initial` must be an orthonormal basis for the eigenspace at
    `points[0]`. For each later point, the full Hamiltonian is diagonalized and
    K eigenvectors are assigned to the previous basis vectors by maximum total
    overlap. Each selected vector is phase-aligned to the previous vector.

    Returns:
        bases: `(n_points, N, K)` complex array.
    """
    point_arr = np.asarray(points, dtype=np.float64)
    if point_arr.ndim != 1 or point_arr.size == 0:
        raise ValueError("points must be a non-empty 1-D sequence")
    if T_initial.ndim != 2:
        raise ValueError(f"T_initial must be 2-D; got {T_initial.shape}")

    N, K = T_initial.shape
    bases = np.empty((point_arr.size, N, K), dtype=np.complex128)
    bases[0] = np.asarray(T_initial, dtype=np.complex128)

    prev = bases[0]
    for i in range(1, point_arr.size):
        H = H_at_point(float(point_arr[i]))
        _D, V = _scipy_eigh(H, driver="evr")
        if V.shape[0] != N:
            raise ValueError(
                f"H_at_point returned basis size {V.shape[0]}, expected {N}"
            )
        overlaps = np.abs(prev.conj().T @ V) ** 2
        row_ind, col_ind = linear_sum_assignment(-overlaps)
        order = col_ind[np.argsort(row_ind)]
        tracked = V[:, order].astype(np.complex128, copy=True)

        for k in range(K):
            phase = np.vdot(prev[:, k], tracked[:, k])
            if abs(phase) > 0.0:
                tracked[:, k] *= np.conj(phase) / abs(phase)

        bases[i] = tracked
        prev = tracked

    return bases


def make_uniform_tracking_grid(
    t_start: float,
    t_end: float,
    basis_dt: float,
) -> npt.NDArray[np.float64]:
    """Uniform tracking times including both endpoints."""
    if t_end <= t_start:
        raise ValueError("t_end must be > t_start")
    if basis_dt <= 0.0:
        raise ValueError("basis_dt must be positive")
    n_intervals = max(1, int(np.ceil((t_end - t_start) / basis_dt)))
    return np.linspace(t_start, t_end, n_intervals + 1, dtype=np.float64)


def step_eigh_truncated(
    c: npt.NDArray[np.complex128],
    t_k: float,
    t_kp1: float,
    H_mid_fn: HMidFn,
    T: npt.NDArray[np.complex128],
    *,
    Tdag: Optional[npt.NDArray[np.complex128]] = None,
) -> npt.NDArray[np.complex128]:
    """Single midpoint step in the truncated subspace. `Tdag = T.conj().T`
    can be precomputed and passed in to avoid recomputation per step."""
    if Tdag is None:
        Tdag = T.conj().T
    t_mid = 0.5 * (t_k + t_kp1)
    dt = t_kp1 - t_k
    H_full = H_mid_fn(t_mid)
    H_proj = Tdag @ H_full @ T  # (K, K)
    D, V = _diagonalize(H_proj)
    tmp = V.conj().T @ c
    tmp *= np.exp(-1j * D * dt)[:, None]
    return V @ tmp


def propagate_midpoint_truncated_decomposed(
    Psi0: npt.NDArray[np.complex128],
    t_grid: npt.NDArray[np.float64],
    components: Sequence[npt.NDArray[np.complex128]],
    coeffs_at_t: Callable[[float], npt.NDArray[np.float64]],
    T: npt.NDArray[np.complex128],
    *,
    store_norm: bool = True,
    store_snapshots: bool = False,
) -> PropagationResult:
    """Truncated propagator with PRE-PROJECTED component matrices.

    `H(t) = sum_α coeffs_at_t(t)[α] * components[α]`. The truncation `T†·H·T`
    distributes over the sum, so we pre-project each component once
    (`H_α_proj = T† components[α] T`, K×K) and per-step we rebuild
    `H_proj(t) = sum_α coeffs_at_t(t)[α] * H_α_proj` — O(K² · n_components)
    per step instead of O(N² K) for the dense T·H·T product. For our problem
    (7 components, K=32, N=196) this is a ~50× per-step speedup over the
    generic truncated propagator.

    coeffs_at_t must return a 1-D array of length len(components).
    """
    if T.ndim != 2 or T.shape[0] != Psi0.shape[0]:
        raise ValueError(
            f"T must be (N, K) with N matching Psi0; got T.shape={T.shape}, "
            f"Psi0.shape={Psi0.shape}"
        )
    Tdag = T.conj().T

    # Pre-project components: K×K each
    proj_components = [Tdag @ comp @ T for comp in components]
    K_dim = T.shape[1]

    c = (Tdag @ np.asarray(Psi0, dtype=np.complex128)).copy()
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    T_grid_size = t_grid.shape[0]
    if T_grid_size < 2:
        raise ValueError("t_grid must have at least 2 points")
    K_in = c.shape[1]

    norm_trace: Optional[npt.NDArray[np.float64]] = None
    snapshots: Optional[npt.NDArray[np.complex128]] = None
    if store_norm:
        norm_trace = np.empty((T_grid_size, K_in), dtype=np.float64)
        norm_trace[0, :] = np.sum(np.abs(c) ** 2, axis=0)
    if store_snapshots:
        snapshots = np.empty((T_grid_size, T.shape[0], K_in), dtype=np.complex128)
        snapshots[0] = T @ c

    H_proj = np.empty((K_dim, K_dim), dtype=np.complex128)
    for k in range(T_grid_size - 1):
        t_mid = 0.5 * (float(t_grid[k]) + float(t_grid[k + 1]))
        dt = float(t_grid[k + 1] - t_grid[k])
        coeffs = coeffs_at_t(t_mid)
        # Rebuild H_proj as weighted sum of pre-projected components
        H_proj.fill(0.0)
        for alpha, w in enumerate(coeffs):
            if w != 0.0:
                H_proj += w * proj_components[alpha]
        D, V = _diagonalize(H_proj)
        tmp = V.conj().T @ c
        tmp *= np.exp(-1j * D * dt)[:, None]
        c = V @ tmp
        if norm_trace is not None:
            norm_trace[k + 1, :] = np.sum(np.abs(c) ** 2, axis=0)
        if snapshots is not None:
            snapshots[k + 1] = T @ c

    Psi_final = T @ c
    return PropagationResult(
        Psi_final=Psi_final,
        norm_trace=norm_trace,
        snapshots=snapshots,
        t_grid=np.asarray(t_grid, dtype=np.float64),
    )


def _nearest_basis_indices(
    t_mid: npt.NDArray[np.float64],
    track_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.intp]:
    right = np.searchsorted(track_times, t_mid, side="left")
    right = np.clip(right, 0, track_times.size - 1)
    left = np.clip(right - 1, 0, track_times.size - 1)
    use_right = np.abs(track_times[right] - t_mid) < np.abs(t_mid - track_times[left])
    return np.where(use_right, right, left).astype(np.intp)


def propagate_midpoint_tracked_decomposed(
    Psi0: npt.NDArray[np.complex128],
    t_grid: npt.NDArray[np.float64],
    components: Sequence[npt.NDArray[np.complex128]],
    coeffs_at_t: Callable[[float], npt.NDArray[np.float64]],
    track_times: npt.NDArray[np.float64],
    tracked_bases: npt.NDArray[np.complex128],
    *,
    store_norm: bool = True,
    store_snapshots: bool = False,
) -> PropagationResult:
    """Midpoint propagation in an adiabatically tracked truncated subspace.

    `tracked_bases[j]` is the full-basis `(N, K)` truncation matrix at
    `track_times[j]`. Each timestep uses the nearest tracked basis to the
    midpoint time. When the selected basis changes, the coefficient vector is
    projected from the old tracked basis to the new one before propagation.

    This is the V3b benchmark path. It keeps the Hamiltonian component
    decomposition used by V3a, but reprojects the components whenever the
    tracked basis changes.
    """
    if t_grid.shape[0] < 2:
        raise ValueError("t_grid must have at least 2 points")
    if track_times.ndim != 1 or track_times.size == 0:
        raise ValueError("track_times must be a non-empty 1-D array")
    if tracked_bases.ndim != 3:
        raise ValueError(
            f"tracked_bases must have shape (n_basis, N, K); got {tracked_bases.shape}"
        )
    if tracked_bases.shape[0] != track_times.size:
        raise ValueError("tracked_bases and track_times disagree on n_basis")
    if tracked_bases.shape[1] != Psi0.shape[0]:
        raise ValueError(
            f"tracked basis N={tracked_bases.shape[1]} does not match Psi0 N={Psi0.shape[0]}"
        )

    T_current = np.asarray(tracked_bases[0], dtype=np.complex128)
    c = (T_current.conj().T @ np.asarray(Psi0, dtype=np.complex128)).copy()
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    n_times = t_grid.shape[0]
    K_in = c.shape[1]
    K_dim = T_current.shape[1]

    norm_trace: Optional[npt.NDArray[np.float64]] = None
    snapshots: Optional[npt.NDArray[np.complex128]] = None
    if store_norm:
        norm_trace = np.empty((n_times, K_in), dtype=np.float64)
        norm_trace[0, :] = np.sum(np.abs(c) ** 2, axis=0)
    if store_snapshots:
        snapshots = np.empty((n_times, T_current.shape[0], K_in), dtype=np.complex128)
        snapshots[0] = T_current @ c

    def project_components(T: npt.NDArray[np.complex128]) -> list[np.ndarray]:
        Tdag = T.conj().T
        return [Tdag @ comp @ T for comp in components]

    proj_components = project_components(T_current)
    current_idx = 0

    mids = 0.5 * (t_grid[:-1] + t_grid[1:])
    basis_indices = _nearest_basis_indices(mids, track_times)
    H_proj = np.empty((K_dim, K_dim), dtype=np.complex128)

    for k in range(n_times - 1):
        basis_idx = int(basis_indices[k])
        if basis_idx != current_idx:
            T_next = np.asarray(tracked_bases[basis_idx], dtype=np.complex128)
            Tdag_next = T_next.conj().T
            c = (Tdag_next @ T_current) @ c
            T_current = T_next
            proj_components = project_components(T_current)
            current_idx = basis_idx

        t_mid = float(mids[k])
        dt = float(t_grid[k + 1] - t_grid[k])
        coeffs = coeffs_at_t(t_mid)
        H_proj.fill(0.0)
        for alpha, w in enumerate(coeffs):
            if w != 0.0:
                H_proj += w * proj_components[alpha]
        D, V = _diagonalize(H_proj)
        tmp = V.conj().T @ c
        tmp *= np.exp(-1j * D * dt)[:, None]
        c = V @ tmp

        if norm_trace is not None:
            norm_trace[k + 1, :] = np.sum(np.abs(c) ** 2, axis=0)
        if snapshots is not None:
            snapshots[k + 1] = T_current @ c

    Psi_final = T_current @ c
    return PropagationResult(
        Psi_final=Psi_final,
        norm_trace=norm_trace,
        snapshots=snapshots,
        t_grid=np.asarray(t_grid, dtype=np.float64),
    )


def propagate_midpoint_truncated(
    Psi0: npt.NDArray[np.complex128],
    t_grid: npt.NDArray[np.float64],
    H_at_t: HMidFn,
    T: npt.NDArray[np.complex128],
    *,
    store_norm: bool = True,
    store_snapshots: bool = False,
) -> PropagationResult:
    """Truncated-subspace midpoint propagator. Drop-in replacement for
    `propagate_midpoint(Psi0, t_grid, H_at_t)` plus a truncation matrix `T`.

    Returns a `PropagationResult` whose `Psi_final` is reconstructed in the
    full N-dim basis (Psi_final = T @ c_final). Norm trace measures the
    coefficient norm ‖c‖² (= subspace probability), so a value < 1 means
    population leaked outside the truncated subspace.
    """
    if T.ndim != 2 or T.shape[0] != Psi0.shape[0]:
        raise ValueError(
            f"T must be (N, K) with N matching Psi0; got T.shape={T.shape}, "
            f"Psi0.shape={Psi0.shape}"
        )
    Tdag = T.conj().T
    c = (Tdag @ np.asarray(Psi0, dtype=np.complex128)).copy()  # (K, K_in)
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    T_grid_size = t_grid.shape[0]
    if T_grid_size < 2:
        raise ValueError("t_grid must have at least 2 points")
    K_in = c.shape[1]

    norm_trace: Optional[npt.NDArray[np.float64]] = None
    snapshots: Optional[npt.NDArray[np.complex128]] = None
    if store_norm:
        norm_trace = np.empty((T_grid_size, K_in), dtype=np.float64)
        norm_trace[0, :] = np.sum(np.abs(c) ** 2, axis=0)
    if store_snapshots:
        snapshots = np.empty((T_grid_size, T.shape[0], K_in), dtype=np.complex128)
        snapshots[0] = T @ c

    for k in range(T_grid_size - 1):
        c = step_eigh_truncated(c, float(t_grid[k]), float(t_grid[k + 1]),
                                H_at_t, T, Tdag=Tdag)
        if norm_trace is not None:
            norm_trace[k + 1, :] = np.sum(np.abs(c) ** 2, axis=0)
        if snapshots is not None:
            snapshots[k + 1] = T @ c

    Psi_final = T @ c
    return PropagationResult(
        Psi_final=Psi_final,
        norm_trace=norm_trace,
        snapshots=snapshots,
        t_grid=np.asarray(t_grid, dtype=np.float64),
    )
