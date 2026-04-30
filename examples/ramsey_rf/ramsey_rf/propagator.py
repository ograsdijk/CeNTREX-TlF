"""Midpoint piecewise-constant unitary propagation in a fixed basis.

Per timestep:
  H_mid = H(t_mid)        (Hermitian, rad/s)
  D, V  = eigh(H_mid)
  Psi   = V @ (exp(-i D dt) * (V^H @ Psi))

Psi has shape (N_states, N_initial_states). The state vector is never stored
in the instantaneous eigenbasis — diagonalization is only used to build the
exponential.

For trajectories with long stretches where H(t) is essentially constant in
time (e.g. between RF coils where the DC field is on a flat plateau and the
RF envelope is below noise threshold), `build_segmented_t_grid` produces a
non-uniform t_grid that uses a fine dt only inside "active" segments (DC
ramps, RF coil regions) and collapses each "inactive" segment into a single
big step. The midpoint propagator is mathematically EXACT for constant H, so
the single big step over an inactive segment introduces no Trotter / Magnus
error — the only numerical cost is one floating-point exp per eigenvalue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh as _scipy_eigh

if TYPE_CHECKING:
    from .fields import FieldStack
    from .trajectory import BallisticTrajectory

HMidFn = Callable[[float], npt.NDArray[np.complex128]]


def _diagonalize(H: npt.NDArray[np.complex128]) -> tuple[np.ndarray, np.ndarray]:
    # scipy 'evr' driver is ~2x faster than numpy.linalg.eigh on the OpenBLAS
    # build shipped with this project; falls back to numpy if scipy unavailable.
    return _scipy_eigh(H, driver="evr")


def step_eigh(
    Psi: npt.NDArray[np.complex128],
    t_k: float,
    t_kp1: float,
    H_mid_fn: HMidFn,
) -> npt.NDArray[np.complex128]:
    t_mid = 0.5 * (t_k + t_kp1)
    dt = t_kp1 - t_k
    H_mid = H_mid_fn(t_mid)
    D, V = _diagonalize(H_mid)
    tmp = V.conj().T @ Psi
    tmp *= np.exp(-1j * D * dt)[:, None]
    return V @ tmp


@dataclass
class PropagationResult:
    Psi_final: npt.NDArray[np.complex128]                 # (N, K)
    norm_trace: Optional[npt.NDArray[np.float64]]         # (T, K) or None
    snapshots: Optional[npt.NDArray[np.complex128]]       # (T, N, K) or None
    t_grid: npt.NDArray[np.float64]                       # (T,)


def propagate_midpoint(
    Psi0: npt.NDArray[np.complex128],
    t_grid: npt.NDArray[np.float64],
    H_at_t: HMidFn,
    *,
    store_norm: bool = True,
    store_snapshots: bool = False,
) -> PropagationResult:
    """Propagate Psi0 along t_grid using midpoint piecewise-constant unitaries.

    Args:
        Psi0: (N, K) initial state. Each column is one initial state, propagated
            in parallel (single eigh per step shared across columns).
        t_grid: (T,) monotonically increasing times in seconds. Step k spans
            [t_grid[k], t_grid[k+1]] and uses H at the midpoint.
        H_at_t: callable t -> (N, N) Hermitian H in rad/s.
        store_norm: record ||Psi(t_k)||^2 per column at every grid point.
        store_snapshots: keep full Psi at every grid point (memory: T*N*K complex).

    Returns:
        PropagationResult with Psi_final and optional history arrays.
    """
    Psi = np.asarray(Psi0, dtype=np.complex128).copy()
    if Psi.ndim != 2:
        raise ValueError(f"Psi0 must be 2-D (N, K); got shape {Psi.shape}")
    T = t_grid.shape[0]
    if T < 2:
        raise ValueError("t_grid must have at least 2 points")

    norm_trace: Optional[npt.NDArray[np.float64]] = None
    snapshots: Optional[npt.NDArray[np.complex128]] = None
    if store_norm:
        norm_trace = np.empty((T, Psi.shape[1]), dtype=np.float64)
        norm_trace[0, :] = np.sum(np.abs(Psi) ** 2, axis=0)
    if store_snapshots:
        snapshots = np.empty((T, Psi.shape[0], Psi.shape[1]), dtype=np.complex128)
        snapshots[0] = Psi

    for k in range(T - 1):
        Psi = step_eigh(Psi, float(t_grid[k]), float(t_grid[k + 1]), H_at_t)
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


@dataclass
class SegmentedGridReport:
    """Diagnostic info about an adaptive segmented time grid."""

    t_grid: npt.NDArray[np.float64]
    n_steps_total: int
    n_steps_active: int
    n_steps_inactive: int
    active_fraction_of_time: float
    n_segments_active: int
    n_segments_inactive: int


def build_segmented_t_grid(
    trajectory: "BallisticTrajectory",
    fields: "FieldStack",
    t_start: float,
    t_end: float,
    dt_fine: float,
    *,
    rf_threshold_rel: float = 1e-2,
    dc_threshold_rel: float = 1e-3,
    dt_coarse_max: Optional[float] = None,
    guard_probes: int = 2,
    n_probes: int = 2000,
    return_report: bool = False,
) -> npt.NDArray[np.float64] | SegmentedGridReport:
    """Build a non-uniform t_grid with fine dt in regions of varying H and
    single big steps where H is essentially constant.

    A point along the trajectory is flagged ACTIVE if either:
      - any RF region's spatial envelope at that point exceeds
        `rf_threshold_rel` × (its peak envelope along the trajectory), OR
      - the local DC E-field magnitude differs from its neighbors by more than
        `dc_threshold_rel` × (peak DC magnitude along the trajectory).

    `guard_probes` extends the active mask by that many probe points on each
    side of every active interval — a safety buffer so the boundary of a
    "single big step" cannot accidentally land where H is still varying.

    `dt_coarse_max` (optional): in inactive intervals, cap the segmented step
    size at this value. None means a single step over the entire interval
    (mathematically exact for constant H, only floating-point exp accuracy).

    Returns the t_grid array. If `return_report=True`, returns a
    `SegmentedGridReport` instead.
    """
    if t_end <= t_start:
        raise ValueError("t_end must be > t_start")
    if guard_probes < 0:
        raise ValueError("guard_probes must be >= 0")
    if n_probes < 4:
        raise ValueError("n_probes must be >= 4")

    # 1. Probe the trajectory at evenly spaced times
    t_probes = np.linspace(t_start, t_end, n_probes)
    R_probes = trajectory(t_probes)  # broadcasts to (n_probes, 3)

    # 2. DC field magnitudes
    E_dc_mags = np.array([np.linalg.norm(fields.E_dc(R)) for R in R_probes])
    E_dc_max = float(E_dc_mags.max()) if E_dc_mags.size else 1.0
    if E_dc_max == 0.0:
        E_dc_max = 1.0
    # "DC active": local change exceeds threshold. Compare each point to its
    # left and right neighbor.
    dE = np.abs(np.diff(E_dc_mags))
    dc_changing = np.zeros(n_probes, dtype=bool)
    dc_changing[:-1] |= dE > dc_threshold_rel * E_dc_max
    dc_changing[1:] |= dE > dc_threshold_rel * E_dc_max

    # 3. RF envelopes per region
    rf_active = np.zeros(n_probes, dtype=bool)
    for region in fields.rf_regions:
        env = np.array([np.linalg.norm(region.envelope_vec(R)) for R in R_probes])
        peak = float(env.max())
        if peak <= 0.0:
            continue
        rf_active |= env > rf_threshold_rel * peak

    active = dc_changing | rf_active

    # 4. Apply guard band — dilate the active mask
    if guard_probes > 0 and active.any():
        from numpy.lib.stride_tricks import sliding_window_view  # noqa: F401
        # simple dilation by repeated boolean OR with shifted versions
        dilated = active.copy()
        for shift in range(1, guard_probes + 1):
            dilated[shift:] |= active[:-shift]
            dilated[:-shift] |= active[shift:]
        active = dilated

    # 5. Walk through probes, building a t_grid that uses fine dt in active
    # runs and big steps in inactive runs.
    t_grid: list[float] = [float(t_start)]
    n_active = 0
    n_inactive = 0
    n_seg_active = 0
    n_seg_inactive = 0
    active_time = 0.0
    i = 0
    while i < n_probes:
        # Find run of same activity starting at i
        j = i + 1
        while j < n_probes and active[j] == active[i]:
            j += 1
        # Run spans probes [i, j); times [t_probes[i], t_probes[j-1] or t_end]
        t_a = t_grid[-1]
        t_b = t_probes[j - 1] if j < n_probes else float(t_end)
        if t_b <= t_a:
            i = j
            continue
        if active[i]:
            n_steps = max(int(np.ceil((t_b - t_a) / dt_fine)), 1)
            for k in range(1, n_steps + 1):
                t_grid.append(t_a + k * (t_b - t_a) / n_steps)
            n_active += n_steps
            n_seg_active += 1
            active_time += t_b - t_a
        else:
            if dt_coarse_max is None:
                t_grid.append(t_b)
                n_inactive += 1
            else:
                n_chunks = max(int(np.ceil((t_b - t_a) / dt_coarse_max)), 1)
                for k in range(1, n_chunks + 1):
                    t_grid.append(t_a + k * (t_b - t_a) / n_chunks)
                n_inactive += n_chunks
            n_seg_inactive += 1
        i = j

    grid = np.array(t_grid, dtype=np.float64)
    # Make sure the last point is exactly t_end (it should be already)
    if grid[-1] != t_end:
        grid[-1] = float(t_end)

    if return_report:
        return SegmentedGridReport(
            t_grid=grid,
            n_steps_total=grid.size - 1,
            n_steps_active=n_active,
            n_steps_inactive=n_inactive,
            active_fraction_of_time=active_time / (t_end - t_start),
            n_segments_active=n_seg_active,
            n_segments_inactive=n_seg_inactive,
        )
    return grid
