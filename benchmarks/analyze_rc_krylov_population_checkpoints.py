"""Compare saved full-density Krylov checkpoints to population product gates.

This is artifact-only analysis: it performs no model build or propagation.
The J=1 indices were resolved from the nominal 70-state QN ordering as
``[4, ..., 15]``; the named initial state has index 26.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rc_propagation_results"
KRYLOV_TAG = "idle_nominal_200us_period_krylov"
REFERENCE_TAG = "idle_nominal_200us_tight_dopri"
J1_INDICES = list(range(4, 16))
NAMED_INITIAL_INDEX = 26
ABS_TOL = 1e-7
REL_TOL = 1e-3
POPULATED_THRESHOLD = 1e-6
POPULATION_NONNEGATIVE_TOL = 1e-8
TRACE_TOL = 1e-9
REALITY_TOL = 1e-12
CHECKPOINTS = (4, 8, 12, 16, 24)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    indices: list[int],
    *,
    require_trace: bool,
) -> dict[str, Any]:
    values = np.diag(candidate)[indices]
    reference_values = np.diag(reference)[indices]
    finite = bool(np.all(np.isfinite(values)) and np.all(np.isfinite(reference_values)))
    reality_max = float(np.max(np.abs(values.imag)))
    error = np.abs(values.real - reference_values.real)
    absolute_error = float(np.max(error))
    populated = reference_values.real >= POPULATED_THRESHOLD
    relative_error = (
        float(np.max(error[populated] / reference_values.real[populated]))
        if np.any(populated)
        else 0.0
    )
    min_population = float(np.min(values.real))
    nonnegative = bool(min_population >= -POPULATION_NONNEGATIVE_TOL)
    trace_error = None
    if require_trace:
        trace_error = abs(float(np.sum(values.real)) - 1.0)
    passes = bool(
        finite
        and reality_max <= REALITY_TOL
        and nonnegative
        and absolute_error <= ABS_TOL
        and relative_error <= REL_TOL
        and (trace_error is None or trace_error <= TRACE_TOL)
    )
    return {
        "indices": indices,
        "state_count": len(indices),
        "max_abs_error": absolute_error,
        "max_relative_error_for_reference_populations_ge_1e-6": relative_error,
        "finite": finite,
        "diagonal_imaginary_max_abs": reality_max,
        "minimum_population": min_population,
        "nonnegative_with_1e-8_numerical_tolerance": nonnegative,
        "strictly_nonnegative": bool(min_population >= 0.0),
        "population_sum_trace_error": trace_error,
        "passes": passes,
    }


def analyze() -> dict[str, Any]:
    metadata_path = RESULTS / f"{KRYLOV_TAG}_progress.json"
    checkpoint_path = RESULTS / f"{KRYLOV_TAG}_checkpoint_densities.npz"
    reference_metadata_path = RESULTS / f"{REFERENCE_TAG}.json"
    reference_density_path = RESULTS / f"{REFERENCE_TAG}_densities.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Krylov artifact is not complete")
    with np.load(checkpoint_path) as archive:
        checkpoints = {q: np.asarray(archive[f"density_q{q}"], dtype=np.complex128) for q in CHECKPOINTS}
    with np.load(reference_density_path) as archive:
        reference = np.asarray(archive["densities"][-1], dtype=np.complex128)
    if reference.shape != (70, 70) or any(matrix.shape != (70, 70) for matrix in checkpoints.values()):
        raise ValueError("expected 70x70 full-density matrices")

    products: dict[str, dict[str, Any]] = {}
    all_indices = list(range(70))
    product_indices = {
        "all_populations": (all_indices, True),
        "J1_populations": (J1_INDICES, False),
        "named_initial_state_population": ([NAMED_INITIAL_INDEX], False),
    }
    for q, density in checkpoints.items():
        products[str(q)] = {
            name: product_metrics(density, reference, indices, require_trace=require_trace)
            for name, (indices, require_trace) in product_indices.items()
        }

    source_paths = {
        "krylov_progress_json": metadata_path,
        "krylov_checkpoint_npz": checkpoint_path,
        "reference_json": reference_metadata_path,
        "reference_density_npz": reference_density_path,
        "analysis_script": Path(__file__).resolve(),
    }
    return {
        "status": "complete",
        "analysis_scope": "saved artifact comparisons only; no solver runs or model builds",
        "claim_scope": "tests population observables extracted from density-reconstructed Arnoldi checkpoints; not native population-only projection timing",
        "photon_scope": "no photon Krylov result was produced or evaluated",
        "cost_interpretation": "per-q cumulative Krylov action time is an upper bound for population products because each period action and reconstruction retained the full density matrix; it is not a measured optimized population-only solve cost",
        "criteria": {
            "absolute_population_error_max": ABS_TOL,
            "relative_population_error_max_above_reference_population": REL_TOL,
            "relative_population_reference_threshold": POPULATED_THRESHOLD,
            "diagonal_imaginary_max_abs": REALITY_TOL,
            "minimum_population": f">= -{POPULATION_NONNEGATIVE_TOL} numerical tolerance; strict result also reported",
            "all_populations_trace_error_max": TRACE_TOL,
            "subset_global_trace_requirement": False,
        },
        "state_selection": {
            "n_states": 70,
            "J1_indices": J1_INDICES,
            "J1_count": len(J1_INDICES),
            "J1_index_basis": "nominal 70-state QN ordering resolved for X,J=1",
            "named_initial_label": "X_J2_F1_5o2_F2_mF0",
            "named_initial_index": NAMED_INITIAL_INDEX,
        },
        "cumulative_cost_upper_bound_seconds": {
            q: float(metadata["krylov"]["checkpoints"][str(q)]["cumulative_wall_including_linearity_probe_seconds"])
            for q in CHECKPOINTS
        },
        "products_by_checkpoint": products,
        "any_checkpoint_passes": {
            name: [q for q in CHECKPOINTS if products[str(q)][name]["passes"]]
            for name in product_indices
        },
        "source_artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "reference_status": metadata.get("reference_validation_status"),
        "krylov_method": metadata.get("method"),
    }


if __name__ == "__main__":
    report = analyze()
    output = RESULTS / f"{KRYLOV_TAG}_population_products.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
