"""High-level RF Ramsey simulator.

Composes basis building, H(t) construction, dressed initial states, propagation,
and detection. The result object carries enough diagnostics (per-J, norm trace,
optional snapshots) to drive the demo notebook and validation script directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .fields import FieldStack
from .hamiltonian import HFunc, build_basis, build_H_func
from .observables import per_j_populations, survival_probability
from .propagator import propagate_midpoint
from .states import (
    TargetSpec,
    dressed_initial_states,
    j_manifold_indices,
)
from .trajectory import BallisticTrajectory


@dataclass(kw_only=True)
class RamseyRFConfig:
    fields: FieldStack
    trajectory: BallisticTrajectory
    z_start: float
    z_final: float
    initial_targets: Optional[TargetSpec] = None
    initial_psi0: Optional[npt.NDArray[np.complex128]] = None
    """Explicit initial state vectors of shape (N, K). Bypasses initial_targets;
    columns are propagated independently. Use this for adiabatic-tracked initial
    states (see `adiabatic_dressed_initial_states`) or any custom Psi0."""
    Jmax: int = 6
    dt: Optional[float] = 0.5e-6
    n_steps: Optional[int] = None
    t_grid: Optional[npt.NDArray[np.float64]] = None
    """If provided, used directly (overrides dt/n_steps). Build with
    `build_segmented_t_grid` to use single-shot exact unitaries in regions
    where H is constant in time (e.g. between RF coils on a flat DC plateau).
    Must start at trajectory.t_at_z(z_start) and end at trajectory.t_at_z(z_final)."""
    initial_weights: Optional[npt.NDArray[np.float64]] = None
    detection_targets: Optional[TargetSpec] = None
    store_snapshots: bool = False
    store_norm: bool = True

    def __post_init__(self) -> None:
        if self.initial_targets is None and self.initial_psi0 is None:
            raise ValueError(
                "Provide either `initial_targets` (selector or list of states) "
                "or `initial_psi0` (explicit (N, K) array)."
            )


@dataclass
class RamseyRFResult:
    Psi_final: npt.NDArray[np.complex128]
    survival: npt.NDArray[np.float64]            # (K,)
    survival_weighted: float
    per_j: npt.NDArray[np.float64]               # (K, n_J)
    per_j_weighted: npt.NDArray[np.float64]      # (n_J,)
    J_values: list[int]
    norm_trace: Optional[npt.NDArray[np.float64]]
    snapshots: Optional[npt.NDArray[np.complex128]]
    t_grid: npt.NDArray[np.float64]
    QN: npt.NDArray
    eigenstate_indices_init: list[int]
    eigenstate_indices_det: list[int]
    initial_overlap_probs: list[float]
    weights: npt.NDArray[np.float64]


class RamseyRFSimulator:
    """Build and run one RF Ramsey trajectory.

    Optionally accepts a pre-built basis QN and H_func to avoid rebuilding them
    across many scan points — see `from_prebuilt`.
    """

    def __init__(
        self,
        cfg: RamseyRFConfig,
        *,
        QN: Optional[npt.NDArray] = None,
        H_func: Optional[HFunc] = None,
    ) -> None:
        self.cfg = cfg
        self.QN = build_basis(cfg.Jmax) if QN is None else QN
        self.H_func = build_H_func(self.QN) if H_func is None else H_func

        # Time grid from trajectory & z bounds (must have non-zero v_z)
        if cfg.trajectory.v[2] == 0.0:
            raise ValueError("trajectory.v[2] must be non-zero (axial flight required)")
        self.t_start = cfg.trajectory.t_at_z(cfg.z_start)
        self.t_end = cfg.trajectory.t_at_z(cfg.z_final)
        if self.t_end <= self.t_start:
            raise ValueError(
                f"t_end <= t_start (z_start={cfg.z_start}, z_final={cfg.z_final}, "
                f"v_z={cfg.trajectory.v[2]}). Make sure v_z and z bounds are consistent."
            )
        if cfg.t_grid is not None:
            grid = np.asarray(cfg.t_grid, dtype=np.float64)
            if grid.ndim != 1 or grid.size < 2:
                raise ValueError("cfg.t_grid must be 1-D with at least 2 points")
            if not np.all(np.diff(grid) > 0):
                raise ValueError("cfg.t_grid must be strictly increasing")
            if abs(grid[0] - self.t_start) > 1e-12 or abs(grid[-1] - self.t_end) > 1e-12:
                raise ValueError(
                    f"cfg.t_grid endpoints ({grid[0]}, {grid[-1]}) do not match "
                    f"trajectory bounds ({self.t_start}, {self.t_end})"
                )
            self.t_grid = grid
        elif cfg.dt is not None:
            n_steps = int(np.ceil((self.t_end - self.t_start) / cfg.dt))
            self.t_grid = np.linspace(self.t_start, self.t_end, n_steps + 1)
        elif cfg.n_steps is not None:
            self.t_grid = np.linspace(self.t_start, self.t_end, int(cfg.n_steps) + 1)
        else:
            raise ValueError("supply one of cfg.t_grid, cfg.dt, or cfg.n_steps")

        # Initial state at t_start.
        # Two paths:
        #   (a) cfg.initial_psi0 given → use directly (e.g. adiabatic-tracked)
        #   (b) cfg.initial_targets given → dress in H(E_dc(R_start)) via simple
        #       max-overlap (only correct in the weak-mixing limit)
        if cfg.initial_psi0 is not None:
            psi0 = np.asarray(cfg.initial_psi0, dtype=np.complex128)
            if psi0.ndim == 1:
                psi0 = psi0.reshape(-1, 1)
            if psi0.shape[0] != len(self.QN):
                raise ValueError(
                    f"initial_psi0 has {psi0.shape[0]} rows but basis has "
                    f"{len(self.QN)} states (Jmax={cfg.Jmax})."
                )
            # Renormalize each column defensively (cheap)
            norms = np.linalg.norm(psi0, axis=0)
            if np.any(norms == 0):
                raise ValueError("initial_psi0 has a zero-norm column")
            self.Psi0 = psi0 / norms
            self.eig_idx_init = []
            self.init_overlap_probs = []
        else:
            R_start = cfg.trajectory(self.t_start)
            H_start = self.H_func(cfg.fields.E_dc(R_start), [0.0, 0.0, 0.0])
            self.Psi0, self.eig_idx_init, self.init_overlap_probs = dressed_initial_states(
                H_start, self.QN, cfg.initial_targets
            )
        K = self.Psi0.shape[1]

        # Weights
        if cfg.initial_weights is None:
            self.weights = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            w = np.asarray(cfg.initial_weights, dtype=np.float64)
            if w.shape != (K,):
                raise ValueError(
                    f"initial_weights must have shape ({K},); got {w.shape}"
                )
            self.weights = w

        # H(t) closure used by the propagator
        self._H_at_t = self._make_H_at_t()

    def _make_H_at_t(self) -> Callable[[float], npt.NDArray[np.complex128]]:
        cfg = self.cfg
        H_func = self.H_func
        traj = cfg.trajectory
        fields = cfg.fields
        zero_B = np.zeros(3, dtype=np.float64)

        def H_at_t(t: float) -> npt.NDArray[np.complex128]:
            R = traj(t)
            E = fields.E_total(R, t)
            return H_func(E, zero_B)

        return H_at_t

    def run(self) -> RamseyRFResult:
        cfg = self.cfg
        prop = propagate_midpoint(
            self.Psi0,
            self.t_grid,
            self._H_at_t,
            store_norm=cfg.store_norm,
            store_snapshots=cfg.store_snapshots,
        )

        # Detection.
        #   - If detection_targets given: dress at H(R_end) and project onto closest.
        #   - Else if initial_targets given: same fallback (return-probability).
        #   - Else (initial_psi0 was used and no override): project onto Psi0
        #     itself — i.e. |<Psi0 | Psi_final>|^2 per column.
        if cfg.detection_targets is not None:
            R_end = cfg.trajectory(self.t_end)
            H_end = self.H_func(cfg.fields.E_dc(R_end), [0.0, 0.0, 0.0])
            survival, eig_idx_det = survival_probability(
                prop.Psi_final, H_end, self.QN, cfg.detection_targets
            )
        elif cfg.initial_targets is not None:
            R_end = cfg.trajectory(self.t_end)
            H_end = self.H_func(cfg.fields.E_dc(R_end), [0.0, 0.0, 0.0])
            survival, eig_idx_det = survival_probability(
                prop.Psi_final, H_end, self.QN, cfg.initial_targets
            )
        else:
            # initial_psi0 path with no detection override — project onto Psi0.
            overlaps = np.einsum("nk,nk->k", self.Psi0.conj(), prop.Psi_final)
            survival = np.abs(overlaps) ** 2
            eig_idx_det = []

        j_idx = j_manifold_indices(self.QN)
        per_j, J_values = per_j_populations(prop.Psi_final, j_idx)

        survival_weighted = float(np.sum(survival * self.weights))
        per_j_weighted = per_j.T @ self.weights  # (n_J,)

        return RamseyRFResult(
            Psi_final=prop.Psi_final,
            survival=survival,
            survival_weighted=survival_weighted,
            per_j=per_j,
            per_j_weighted=per_j_weighted,
            J_values=J_values,
            norm_trace=prop.norm_trace,
            snapshots=prop.snapshots,
            t_grid=prop.t_grid,
            QN=self.QN,
            eigenstate_indices_init=self.eig_idx_init,
            eigenstate_indices_det=eig_idx_det,
            initial_overlap_probs=self.init_overlap_probs,
            weights=self.weights,
        )
