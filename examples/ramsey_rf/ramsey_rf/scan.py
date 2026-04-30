"""Generic 1-D scan over an RF Ramsey configuration knob.

Each scan point deep-copies cfg, mutates one parameter, runs the simulator,
and stacks the results. The simulator's basis QN and H_func are built once and
reused across points (they don't depend on any scannable parameter).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from .hamiltonian import build_basis, build_H_func
from .simulator import RamseyRFConfig, RamseyRFResult, RamseyRFSimulator

ScanAxis = Literal["phi1", "phi2", "omega_rf", "velocity", "rf_amp_1", "rf_amp_2"]


@dataclass
class ScanSpec:
    """One-dimensional sweep over a single named knob.

    `rf_amp_i` is a multiplicative scaling factor on region i's envelope (so
    `values = np.linspace(0, 2, 21)` sweeps from off to 2x the nominal amplitude).
    `velocity` replaces v with `(v_hat) * value` (preserves trajectory direction).
    `omega_rf` and `phi1`/`phi2` are absolute (rad/s and rad respectively).
    """

    axis: ScanAxis
    values: npt.NDArray[np.float64]


@dataclass
class ScanResult:
    spec: ScanSpec
    survival_per_init: npt.NDArray[np.float64]      # (n_points, K)
    survival_weighted: npt.NDArray[np.float64]      # (n_points,)
    per_j: npt.NDArray[np.float64]                  # (n_points, K, n_J)
    per_j_weighted: npt.NDArray[np.float64]         # (n_points, n_J)
    J_values: list[int]
    weights: npt.NDArray[np.float64]
    individual_results: Optional[list[RamseyRFResult]] = None


def _scale_envelope(envelope_z, factor: float):
    """Wrap an existing envelope callable with a multiplicative factor."""
    def wrapped(z, _e=envelope_z, _f=factor):
        return _f * _e(z)
    return wrapped


def _apply_axis(cfg: RamseyRFConfig, axis: ScanAxis, value: float) -> None:
    """Mutate cfg in place to apply one scan value. cfg is assumed pre-deepcopied."""
    if axis == "phi1":
        cfg.fields.rf_regions[0].phi = float(value)
    elif axis == "phi2":
        if len(cfg.fields.rf_regions) < 2:
            raise ValueError("scan axis 'phi2' requires at least two RF regions")
        cfg.fields.rf_regions[1].phi = float(value)
    elif axis == "omega_rf":
        for region in cfg.fields.rf_regions:
            region.omega = float(value)
    elif axis == "velocity":
        v = np.asarray(cfg.trajectory.v, dtype=np.float64)
        speed = float(np.linalg.norm(v))
        if speed == 0.0:
            raise ValueError("cannot rescale a zero-magnitude velocity vector")
        v_hat = v / speed
        cfg.trajectory.v = v_hat * float(value)
    elif axis == "rf_amp_1":
        region = cfg.fields.rf_regions[0]
        region.envelope_z = _scale_envelope(region.envelope_z, float(value))
    elif axis == "rf_amp_2":
        if len(cfg.fields.rf_regions) < 2:
            raise ValueError("scan axis 'rf_amp_2' requires at least two RF regions")
        region = cfg.fields.rf_regions[1]
        region.envelope_z = _scale_envelope(region.envelope_z, float(value))
    else:
        raise ValueError(f"unknown scan axis: {axis!r}")


def run_scan(
    cfg: RamseyRFConfig,
    spec: ScanSpec,
    *,
    keep_individual_results: bool = False,
    progress: bool = False,
) -> ScanResult:
    """Run the simulator at each spec.values point, keeping per-point arrays.

    Builds the basis and H_func once and passes them to each simulator instance.
    Set `keep_individual_results=True` to retain the full RamseyRFResult per
    point (memory-heavy if snapshots are stored).
    """
    QN = build_basis(cfg.Jmax)
    H_func = build_H_func(QN)

    n_points = len(spec.values)
    survival_per_init: list[np.ndarray] = []
    survival_weighted = np.empty(n_points, dtype=np.float64)
    per_j_list: list[np.ndarray] = []
    per_j_weighted_list: list[np.ndarray] = []
    individual: Optional[list[RamseyRFResult]] = [] if keep_individual_results else None
    J_values: list[int] = []
    weights: Optional[np.ndarray] = None

    for i, v in enumerate(spec.values):
        cfg_i = copy.deepcopy(cfg)
        _apply_axis(cfg_i, spec.axis, float(v))
        sim = RamseyRFSimulator(cfg_i, QN=QN, H_func=H_func)
        res = sim.run()
        survival_per_init.append(res.survival)
        survival_weighted[i] = res.survival_weighted
        per_j_list.append(res.per_j)
        per_j_weighted_list.append(res.per_j_weighted)
        if individual is not None:
            individual.append(res)
        if not J_values:
            J_values = res.J_values
            weights = res.weights
        if progress:
            print(f"[scan {spec.axis}] point {i + 1}/{n_points}: value={v:g}, "
                  f"survival_weighted={res.survival_weighted:.6f}")

    return ScanResult(
        spec=spec,
        survival_per_init=np.stack(survival_per_init, axis=0),
        survival_weighted=survival_weighted,
        per_j=np.stack(per_j_list, axis=0),
        per_j_weighted=np.stack(per_j_weighted_list, axis=0),
        J_values=J_values,
        weights=weights if weights is not None else np.array([], dtype=np.float64),
        individual_results=individual,
    )
