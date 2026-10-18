"""Budgeted eight-column pilot for an exact RC one-period map.

This module measures a small batch of real Hermitian packed-basis actions. It
never constructs a period map. A full map is projected against explicit time
and storage budgets first; callers decide whether later construction is
worth its measured cost.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import bench_rc_propagation as driver
import numpy as np
import rc_propagation_methods as methods

PERIOD_S = 5e-6
PACKED_MAP_BYTES_PER_ENTRY = np.dtype(np.float64).itemsize
PILOT_COLUMN_COUNT = 8
PRIOR_COMMON_SUPPORT = {
    "packed_dimension": 4360,
    "common_sink_count": 4,
    "artifact": "sparse_weak_1us.json",
}


def _state_index(case: Any, state: Any) -> int:
    for index, candidate in enumerate(case.system.QN):
        if candidate is state or candidate == state:
            return index
    raise ValueError("selected state is absent from the OBE basis")


def _basis_columns(case: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select population and coherence coordinates spanning key RC sectors."""
    layout = case.prepared_dynamic.layout
    labels = case.state_labels
    x1 = int(labels["X_J1_F1_3o2_F1_mF0"])
    x2 = int(labels["X_J2_F1_5o2_F2_mF0"])
    x3 = int(labels["X_J3_F1_7o2_F3_mF0"])
    coupling = case.system.couplings[0]
    b_main = _state_index(case, coupling.excited_main)

    specs = [
        ("X_J1_population", "diagonal", x1, None),
        ("X_J2_population", "diagonal", x2, None),
        ("X_J3_population", "diagonal", x3, None),
        ("B_main_population", "diagonal", b_main, None),
        ("X_J1_X_J2_real_coherence", "real", min(x1, x2), max(x1, x2)),
        ("X_J1_X_J2_imag_coherence", "imag", min(x1, x2), max(x1, x2)),
        ("X_main_B_main_real_coherence", "real", min(x2, b_main), max(x2, b_main)),
        ("X_main_B_main_imag_coherence", "imag", min(x2, b_main), max(x2, b_main)),
    ]
    packed_columns = np.zeros((len(specs), layout.packed_len), dtype=np.float64)
    metadata: list[dict[str, Any]] = []
    seen: set[int] = set()
    for column, (name, kind, i, j) in enumerate(specs):
        if kind == "diagonal":
            packed_index = layout.diagonal_index(i)
        elif kind == "real":
            assert j is not None
            packed_index = layout.upper_real_index(i, j)
        else:
            assert kind == "imag" and j is not None
            packed_index = layout.upper_imag_index(i, j)
        if packed_index in seen:
            raise ValueError(f"pilot selection repeats packed coordinate {packed_index}")
        seen.add(packed_index)
        packed_columns[column, packed_index] = 1.0
        metadata.append(
            {
                "column": column,
                "name": name,
                "kind": kind,
                "state_indices": [int(i)] if j is None else [int(i), int(j)],
                "packed_coordinate": int(packed_index),
            }
        )
    if len(metadata) != PILOT_COLUMN_COUNT:
        raise AssertionError("periodic pilot must contain exactly eight columns")
    return packed_columns, metadata


def _propagate_chunk(
    case: Any,
    packed_columns: np.ndarray,
    *,
    reltol: float,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    layout = case.prepared_dynamic.layout
    rho0 = np.stack([layout.unpack(row) for row in packed_columns])
    started = time.perf_counter()
    times, densities, stats = driver.propagate_segmented_batch(
        case,
        rho0,
        (0.0, PERIOD_S),
        solver="tsit5",
        reltol=reltol,
        abstol=reltol / 100.0,
        execution_mode="structured",
        save_step_s=None,
        parallel=True,
        threads=threads,
    )
    elapsed = time.perf_counter() - started
    photon_yields = np.asarray(
        [
            sum(float(np.asarray(segment["photon_integral"]).reshape(-1)[i]) for segment in stats)
            for i in range(len(packed_columns))
        ],
        dtype=np.float64,
    )
    final = np.asarray(densities[:, -1], dtype=np.complex128)
    return np.asarray(times, dtype=np.float64), final, photon_yields, stats, elapsed


def run_periodic_pilot(
    case: Any,
    *,
    budget_seconds: float = 1800.0,
    threads: int = 8,
    output_path: str | Path | None = None,
    refinement_probe_limit_seconds: float = 120.0,
    max_map_bytes: int = 8 * 1024**3,
) -> dict[str, Any]:
    """Measure eight exact one-period actions and project full-map cost.

    The pilot uses the canonical interval ``[0, 5 us]`` with the RC driver's
    exact switch-boundary segmentation and absolute-time EOM phase. It returns
    full-density reconstructions and integrated photon values for its basis
    columns. Coherence basis matrices need not be positive or unit trace; no
    physical-state positivity or trace-one claim is made for these linearity
    probes.

    The first pass uses tsit5 at ``reltol=1e-9``. If that eight-column pass
    takes no more than ``refinement_probe_limit_seconds``, one tighter
    ``1e-10`` pass estimates sensitivity to tolerance. That small-sample
    refinement is not a converged physical reference and the returned
    ``accuracy_status`` remains ``"accuracy_pending"``. No periodic map is
    allocated or constructed.
    """
    if not np.isfinite(budget_seconds) or budget_seconds <= 0.0:
        raise ValueError("budget_seconds must be positive and finite")
    if not np.isfinite(refinement_probe_limit_seconds) or refinement_probe_limit_seconds < 0.0:
        raise ValueError("refinement_probe_limit_seconds must be finite and nonnegative")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if max_map_bytes <= 0:
        raise ValueError("max_map_bytes must be positive")
    if case.amplitude_envelope != "constant":
        raise ValueError("the exact periodic pilot requires constant amplitude envelopes")

    layout = case.prepared_dynamic.layout
    packed_columns, column_metadata = _basis_columns(case)
    # Avoid including lazy plan assignment in the measured propagation chunk.
    driver._prepare_segmented_plan(case)
    times, coarse_final, coarse_photons, stats, coarse_seconds = _propagate_chunk(
        case, packed_columns, reltol=1e-9, threads=threads
    )

    fine_final: np.ndarray | None = None
    fine_photons: np.ndarray | None = None
    refinement: dict[str, Any] = {"status": "not_run_chunk_exceeded_limit"}
    timing_seconds_for_projection = coarse_seconds
    if coarse_seconds <= refinement_probe_limit_seconds:
        fine_times, fine_final, fine_photons, fine_stats, fine_seconds = _propagate_chunk(
            case, packed_columns, reltol=1e-10, threads=threads
        )
        if not np.array_equal(times, fine_times):
            raise AssertionError("coarse and refined pilots used different switch sample times")
        density_differences = np.abs(fine_final - coarse_final)
        photon_differences = np.abs(fine_photons - coarse_photons)
        refinement = {
            "status": "sampled_tolerance_refinement_only",
            "coarse_reltol": 1e-9,
            "fine_reltol": 1e-10,
            "fine_chunk_seconds": float(fine_seconds),
            "max_final_matrix_element_difference": float(np.max(density_differences)),
            "max_photon_integral_difference": float(np.max(photon_differences)),
            "per_column_final_matrix_element_difference": np.max(
                density_differences, axis=(1, 2)
            ).tolist(),
            "per_column_photon_integral_difference": photon_differences.tolist(),
            "fine_segment_count": len(fine_stats),
        }
        timing_seconds_for_projection = max(coarse_seconds, fine_seconds)

    full_dimension = int(layout.packed_len)
    full_map_bytes = int(PACKED_MAP_BYTES_PER_ENTRY * full_dimension * full_dimension)
    action_seconds = timing_seconds_for_projection / PILOT_COLUMN_COUNT
    full_gate = methods.periodic_map_gate(
        layout,
        max_bytes=max_map_bytes,
        max_basis_actions=5000,
        estimated_action_seconds=2.0 * action_seconds,
        max_build_seconds=budget_seconds,
    )
    sequential_projection = action_seconds * full_dimension
    batched_chunks = math.ceil(full_dimension / PILOT_COLUMN_COUNT)
    batched_projection = batched_chunks * timing_seconds_for_projection

    active_dimension = int(PRIOR_COMMON_SUPPORT["packed_dimension"])
    support_matches_dimension = full_dimension == 4900
    active_map_bytes = int(PACKED_MAP_BYTES_PER_ENTRY * active_dimension * active_dimension)
    reduced_sequential_projection = action_seconds * active_dimension
    reduced_batched_projection = (
        math.ceil(active_dimension / PILOT_COLUMN_COUNT) * timing_seconds_for_projection
    )

    result: dict[str, Any] = {
        "status": "pilot_complete_map_not_constructed",
        "canonical_entry_time_s": 0.0,
        "period_s": PERIOD_S,
        "frequency_relation": "1.6 MHz EOM and 1 MHz switches share a 5 us period",
        "switch_segmentation": "exact boundaries from driver.switch_boundaries; full density carried across every segment",
        "eom_phase": "original-frame EOM retained at absolute global time",
        "solver": "tsit5",
        "execution_mode": "structured",
        "coarse_reltol": 1e-9,
        "coarse_abstol": 1e-11,
        "threads": int(threads),
        "pilot_column_count": PILOT_COLUMN_COUNT,
        "pilot_chunk_seconds": float(coarse_seconds),
        "projection_chunk_seconds": float(timing_seconds_for_projection),
        "projection_safety_factor": 2.0,
        "projection_safety_note": "The gate uses twice the sampled throughput cost; basis columns outside the probe may be harder.",
        "segment_count": len(stats),
        "segment_full_density_seconds": float(sum(s["full_density_wall_seconds"] for s in stats)),
        "segment_photon_integral_seconds": float(
            sum(s["photon_integral_wall_seconds"] for s in stats)
        ),
        "column_coordinates": column_metadata,
        "n_states": int(layout.n),
        "full_packed_dimension": full_dimension,
        "full_map_raw_bytes": full_map_bytes,
        "map_memory_budget_bytes": int(max_map_bytes),
        "map_memory_fits_budget": bool(full_map_bytes <= max_map_bytes),
        "full_layout_gate": {
            "feasible_within_projected_action_budget": bool(full_gate.feasible),
            "reason": full_gate.reason,
            "actions": int(full_gate.basis_actions),
            "estimated_action_seconds_from_8_column_chunk": float(action_seconds),
            "throughput_equivalent_seconds": float(sequential_projection),
            "projection_note": "Parallel chunk time divided by eight estimates throughput; it does not measure serial action latency.",
            "budget_seconds": float(budget_seconds),
            "projected_batched_chunks_of_8": int(batched_chunks),
            "projected_batched_seconds_if_throughput_scales": float(batched_projection),
            "safety_adjusted_batched_projection_seconds": float(2.0 * batched_projection),
            "batched_projection_within_budget": bool(2.0 * batched_projection <= budget_seconds),
        },
        "tentative_common_support": {
            **PRIOR_COMMON_SUPPORT,
            "full_packed_dimension": full_dimension,
            "applicable_dimension_match": bool(support_matches_dimension),
            "validated_for_this_canonical_pilot": False,
            "estimated_map_bytes_if_same_support": active_map_bytes,
            "estimated_basis_actions_if_same_support": active_dimension,
            "throughput_equivalent_seconds_if_same_support": float(reduced_sequential_projection),
            "projected_batched_seconds_if_same_support": float(reduced_batched_projection),
            "note": "prior support is tentative; do not use for map construction until revalidated against this exact model and all four switched generators/phase rotations",
        },
        "accuracy_status": "accuracy_pending",
        "accuracy_claim": False,
        "tolerance_refinement_probe": refinement,
        "final_density_trace": [
            [float(np.trace(item).real), float(np.trace(item).imag)] for item in coarse_final
        ],
        "final_density_hermiticity_max_abs": [
            float(np.max(np.abs(item - item.conj().T))) for item in coarse_final
        ],
        "coarse_photon_integrals": coarse_photons.tolist(),
        "fine_photon_integrals": None if fine_photons is None else fine_photons.tolist(),
        "no_map_constructed": True,
        "note": "basis coherence probes are linear Hermitian coordinates, not necessarily physical density matrices; trace/positivity metrics are diagnostic only",
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "times_s": times,
            "packed_basis_columns": packed_columns,
            "coarse_final_densities": coarse_final,
            "coarse_photon_integrals": coarse_photons,
        }
        if fine_final is not None and fine_photons is not None:
            arrays["fine_final_densities"] = fine_final
            arrays["fine_photon_integrals"] = fine_photons
        np.savez_compressed(path, **arrays)
        result["npz_path"] = str(path.resolve())
    result["coarse_final_densities"] = coarse_final
    result["fine_final_densities"] = fine_final
    return result


__all__ = ["run_periodic_pilot"]
