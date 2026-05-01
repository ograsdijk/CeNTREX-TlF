"""Generic 1-D scan over an RF Ramsey configuration knob.

Each scan point deep-copies cfg, mutates one parameter, runs the simulator,
and stacks the results. The simulator's basis QN and H_func are built once and
reused across points (they don't depend on any scannable parameter).

Set `n_workers > 1` on `run_scan` to parallelize across scan points using
processes (uses cloudpickle to serialize the FieldStack closures across the
process boundary). Each worker rebuilds QN and H_func locally — cheap (~ms).
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
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


def _all_rf_regions(cfg: RamseyRFConfig) -> list:
    """Return all RF regions in cfg, electric first then magnetic, in a single
    flat list. The scan axes 'phi1'/'phi2'/'omega_rf'/'rf_amp_*' apply uniformly
    across this combined list — useful since a config typically uses one type."""
    return list(cfg.fields.rf_regions) + list(cfg.fields.rf_regions_B)


def _set_rf_phi(cfg: RamseyRFConfig, idx: int, value: float) -> None:
    """Set the phase of the idx-th RF region (electric first, then magnetic)."""
    n_e = len(cfg.fields.rf_regions)
    if idx < n_e:
        cfg.fields.rf_regions[idx].phi = float(value)
    else:
        cfg.fields.rf_regions_B[idx - n_e].phi = float(value)


def _scale_rf_envelope(cfg: RamseyRFConfig, idx: int, value: float) -> None:
    n_e = len(cfg.fields.rf_regions)
    if idx < n_e:
        region = cfg.fields.rf_regions[idx]
    else:
        region = cfg.fields.rf_regions_B[idx - n_e]
    region.envelope_z = _scale_envelope(region.envelope_z, float(value))


def _apply_axis(cfg: RamseyRFConfig, axis: ScanAxis, value: float) -> None:
    """Mutate cfg in place to apply one scan value. cfg is assumed pre-deepcopied."""
    n_total_rf = len(cfg.fields.rf_regions) + len(cfg.fields.rf_regions_B)

    if axis == "phi1":
        if n_total_rf < 1:
            raise ValueError("scan axis 'phi1' requires at least one RF region")
        _set_rf_phi(cfg, 0, value)
    elif axis == "phi2":
        if n_total_rf < 2:
            raise ValueError("scan axis 'phi2' requires at least two RF regions")
        _set_rf_phi(cfg, 1, value)
    elif axis == "omega_rf":
        for region in cfg.fields.rf_regions:
            region.omega = float(value)
        for region in cfg.fields.rf_regions_B:
            region.omega = float(value)
    elif axis == "velocity":
        v = np.asarray(cfg.trajectory.v, dtype=np.float64)
        speed = float(np.linalg.norm(v))
        if speed == 0.0:
            raise ValueError("cannot rescale a zero-magnitude velocity vector")
        v_hat = v / speed
        cfg.trajectory.v = v_hat * float(value)
    elif axis == "rf_amp_1":
        if n_total_rf < 1:
            raise ValueError("scan axis 'rf_amp_1' requires at least one RF region")
        _scale_rf_envelope(cfg, 0, value)
    elif axis == "rf_amp_2":
        if n_total_rf < 2:
            raise ValueError("scan axis 'rf_amp_2' requires at least two RF regions")
        _scale_rf_envelope(cfg, 1, value)
    else:
        raise ValueError(f"unknown scan axis: {axis!r}")


def _run_one_point(cfg: RamseyRFConfig, axis: ScanAxis, value: float,
                   QN=None, H_func=None) -> dict:
    """Mutate a config copy at one scan value, run the simulator, return a
    pickleable dict of result arrays (no full RamseyRFResult — too heavy)."""
    cfg_i = copy.deepcopy(cfg)
    _apply_axis(cfg_i, axis, float(value))
    sim = RamseyRFSimulator(cfg_i, QN=QN, H_func=H_func)
    res = sim.run()
    return {
        "survival": res.survival,
        "survival_weighted": res.survival_weighted,
        "per_j": res.per_j,
        "per_j_weighted": res.per_j_weighted,
        "J_values": list(res.J_values),
        "weights": res.weights,
    }


# -------- Worker for process-pool parallelism (cloudpickle round-trip) --------
# A single cfg blob is serialized once via cloudpickle, sent to the pool worker,
# deserialized, and reused across calls. Workers also limit BLAS to 1 thread so
# each scan point gets one core (otherwise OpenBLAS oversubscribes).
def _worker_init(cfg_blob: bytes) -> None:
    """Pool initializer: stash the deserialized cfg as a module global, and
    cap BLAS threads to 1 (we already have process-level parallelism)."""
    # These env vars only take effect when set BEFORE numpy/scipy import inside
    # the child interpreter. Spawn-mode children re-import everything, so this
    # works on Windows too.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    import cloudpickle  # local import — only needed in workers
    global _CFG, _QN, _H_FUNC
    _CFG = cloudpickle.loads(cfg_blob)
    _QN = build_basis(_CFG.Jmax)
    _H_FUNC = build_H_func(_QN)


def _worker_run(payload: tuple) -> dict:
    axis, value = payload
    return _run_one_point(_CFG, axis, value, QN=_QN, H_func=_H_FUNC)


def run_scan(
    cfg: RamseyRFConfig,
    spec: ScanSpec,
    *,
    n_workers: Optional[int] = None,
    keep_individual_results: bool = False,
    progress: bool = False,
) -> ScanResult:
    """Run the simulator at each spec.values point, keeping per-point arrays.

    Args:
        cfg: baseline configuration. Each scan point deep-copies cfg and
            mutates one parameter according to `spec`.
        spec: scan axis + values.
        n_workers: process-level parallelism. None or 1 → sequential (current
            process). 2+ → spawn that many worker processes via cloudpickle,
            with each worker capping its BLAS threads to 1. -1 → use
            `os.cpu_count() - 1`. Workers receive a single cloudpickle blob of
            cfg at startup and reuse it across calls.
        keep_individual_results: retain the full RamseyRFResult per point.
            Only supported with n_workers in {None, 1} (full results don't
            survive the worker boundary cleanly).
        progress: print one line per completed scan point.
    """
    if n_workers is not None and n_workers == -1:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    n_points = len(spec.values)
    sequential = (n_workers is None or n_workers <= 1)

    if not sequential and keep_individual_results:
        raise ValueError(
            "keep_individual_results=True is incompatible with n_workers>1; "
            "the full RamseyRFResult is not returned across the process boundary"
        )

    if sequential:
        QN = build_basis(cfg.Jmax)
        H_func = build_H_func(QN)
        results = []
        individual: Optional[list[RamseyRFResult]] = [] if keep_individual_results else None
        for i, v in enumerate(spec.values):
            if individual is not None:
                # need full RamseyRFResult — re-run inline, not via _run_one_point
                cfg_i = copy.deepcopy(cfg)
                _apply_axis(cfg_i, spec.axis, float(v))
                sim = RamseyRFSimulator(cfg_i, QN=QN, H_func=H_func)
                res = sim.run()
                individual.append(res)
                results.append({
                    "survival": res.survival,
                    "survival_weighted": res.survival_weighted,
                    "per_j": res.per_j,
                    "per_j_weighted": res.per_j_weighted,
                    "J_values": list(res.J_values),
                    "weights": res.weights,
                })
            else:
                results.append(_run_one_point(cfg, spec.axis, float(v),
                                              QN=QN, H_func=H_func))
            if progress:
                print(f"[scan {spec.axis}] point {i + 1}/{n_points}: "
                      f"value={spec.values[i]:g}, "
                      f"survival_weighted={results[-1]['survival_weighted']:.6f}")
    else:
        try:
            import cloudpickle
        except ImportError as e:
            raise ImportError(
                "n_workers > 1 requires cloudpickle for cross-process serialization "
                "(`pip install cloudpickle`). Use n_workers=1 for sequential mode."
            ) from e
        cfg_blob = cloudpickle.dumps(cfg)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers,
                      initializer=_worker_init, initargs=(cfg_blob,)) as pool:
            payload = [(spec.axis, float(v)) for v in spec.values]
            results = []
            individual = None
            iterator = pool.imap(_worker_run, payload)
            for i, r in enumerate(iterator):
                results.append(r)
                if progress:
                    print(f"[scan {spec.axis} | worker] point {i + 1}/{n_points}: "
                          f"value={spec.values[i]:g}, "
                          f"survival_weighted={r['survival_weighted']:.6f}")

    survival_per_init = [r["survival"] for r in results]
    survival_weighted = np.array([r["survival_weighted"] for r in results],
                                  dtype=np.float64)
    per_j_list = [r["per_j"] for r in results]
    per_j_weighted_list = [r["per_j_weighted"] for r in results]
    J_values = results[0]["J_values"]
    weights = results[0]["weights"]

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
