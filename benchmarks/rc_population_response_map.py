"""Reusable linear response map for population-mixture RC initial states.

For a fixed Lindblad generator schedule, each one-hot diagonal initial state
defines one response column.  The rows retain every packed Hermitian density
coordinate plus a solver-native photon integral, so mixtures can reconstruct
the full complex endpoint density without discarding coherences.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from centrex_tlf.lindblad.batch import solve_lindblad_batch
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout


@dataclass(frozen=True)
class PopulationMixtureResult:
    density: np.ndarray
    photon_yield: float
    populations: np.ndarray
    selected_populations: np.ndarray | None


@dataclass(frozen=True)
class PopulationResponseMap:
    """Column response map over an explicitly declared diagonal-input support."""

    n_states: int
    covered_indices: np.ndarray
    response_columns: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.n_states, bool) or int(self.n_states) != self.n_states or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        indices = np.asarray(self.covered_indices)
        if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("covered_indices must be a nonempty 1D integer array")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.n_states):
            raise ValueError("covered index is outside the density-state layout")
        if np.unique(indices).size != indices.size:
            raise ValueError("covered_indices must not contain duplicates")
        raw_responses = np.asarray(self.response_columns)
        if np.iscomplexobj(raw_responses):
            raise ValueError("response_columns must use real packed Hermitian coordinates")
        responses = np.asarray(raw_responses, dtype=np.float64)
        expected_rows = self.n_states * self.n_states + 1
        if responses.shape != (expected_rows, indices.size):
            raise ValueError(
                f"response_columns must have shape ({expected_rows}, {indices.size})"
            )
        if not np.all(np.isfinite(responses)):
            raise ValueError("response_columns must contain only finite values")
        object.__setattr__(self, "covered_indices", indices.copy())
        object.__setattr__(self, "response_columns", responses.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def layout(self) -> PackedHermitianLayout:
        return PackedHermitianLayout(self.n_states)

    @property
    def is_full_coverage(self) -> bool:
        return self.covered_indices.size == self.n_states and np.array_equal(
            np.sort(self.covered_indices), np.arange(self.n_states)
        )

    def _covered_weights(self, weights: np.ndarray) -> tuple[np.ndarray, bool]:
        raw_values = np.asarray(weights)
        if np.iscomplexobj(raw_values):
            raise ValueError("weights must be real-valued")
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim not in (1, 2) or not np.all(np.isfinite(values)):
            raise ValueError("weights must be a finite one- or two-dimensional real array")
        single = values.ndim == 1
        matrix = values.reshape(1, -1) if single else values
        if matrix.shape[0] == 0:
            raise ValueError("weights must contain at least one mixture")
        if np.any(matrix < 0.0):
            raise ValueError("population-mixture weights must be nonnegative")
        if matrix.shape[1] == self.n_states:
            covered_mask = np.zeros(self.n_states, dtype=bool)
            covered_mask[self.covered_indices] = True
            if np.any(matrix[:, ~covered_mask] != 0.0):
                raise ValueError("weights include a state outside this map's coverage")
            covered = matrix[:, self.covered_indices]
        elif matrix.shape[1] == self.covered_indices.size:
            covered = matrix
        else:
            raise ValueError(
                "weights must match either the full state layout or map coverage"
            )
        if not np.allclose(covered.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("every population-mixture row must sum to one")
        return covered, single

    def photons(self, weights: np.ndarray) -> float | np.ndarray:
        """Return photon yield without multiplying/reconstructing density rows."""
        covered, single = self._covered_weights(weights)
        result = self.response_columns[-1, :] @ covered.T
        return float(result[0]) if single else result

    def populations(
        self,
        weights: np.ndarray,
        *,
        selected_indices: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return populations without density unpacking.

        A single mixture returns shape ``(n_selected,)``. A matrix of mixtures
        returns ``(n_selected, n_mixtures)`` so the selected output rows can be
        multiplied directly by the mixture-weight matrix.
        """
        covered, single = self._covered_weights(weights)
        indices = (
            np.arange(self.n_states, dtype=np.int64)
            if selected_indices is None
            else np.asarray(selected_indices)
        )
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("selected_indices must be a 1D integer array")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.n_states):
            raise ValueError("selected population index is outside the state layout")
        values = self.response_columns[indices, :] @ covered.T
        return values[:, 0] if single else values

    def density(self, weights: np.ndarray) -> np.ndarray:
        """Return full complex endpoint rho, retaining every coherence."""
        covered, single = self._covered_weights(weights)
        packed = self.response_columns[:-1, :] @ covered.T
        if single:
            return self.layout.unpack(packed[:, 0])
        return np.stack([self.layout.unpack(row) for row in packed.T])

    def apply(
        self,
        weights: np.ndarray,
        *,
        selected_indices: list[int] | np.ndarray | None = None,
    ) -> PopulationMixtureResult:
        """Apply to a normalized nonnegative population mixture.

        ``weights`` may be full-layout length or, for a subset map, coverage
        length.  A full-layout vector with any nonzero weight outside coverage
        is rejected rather than silently projecting away part of the mixture.
        """
        covered, single = self._covered_weights(weights)
        if not single:
            raise ValueError("apply returns a single combined result; use warm methods for weight matrices")
        density = self.layout.unpack(self.response_columns[:-1, :] @ covered[0])
        populations = np.diag(density).real.copy()
        selected = None
        if selected_indices is not None:
            indices = np.asarray(selected_indices)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("selected_indices must be a 1D integer array")
            indices = indices.astype(np.int64, copy=False)
            if np.any(indices < 0) or np.any(indices >= self.n_states):
                raise ValueError("selected population index is outside the state layout")
            selected = populations[indices].copy()
        return PopulationMixtureResult(
            density=density,
            photon_yield=float(self.response_columns[-1, :] @ covered[0]),
            populations=populations,
            selected_populations=selected,
        )


def make_response_map(
    n_states: int,
    covered_indices: list[int] | np.ndarray,
    response_columns: np.ndarray,
    metadata: dict[str, Any],
) -> PopulationResponseMap:
    indices = np.asarray(covered_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("covered_indices must be a 1D integer array")
    return PopulationResponseMap(
        n_states=n_states,
        covered_indices=indices,
        response_columns=response_columns,
        metadata=metadata,
    )


@dataclass(frozen=True)
class ResponseBuildStats:
    total_wall_seconds: float
    batch_records: list[dict[str, Any]]
    per_column_allocated_wall_seconds: list[dict[str, Any]]
    n_solve_calls: int


def build_diagonal_response(
    case: Any,
    basis_indices: list[int] | np.ndarray,
    t_span: tuple[float, float],
    *,
    metadata: dict[str, Any],
    solver: str = "tsit5",
    reltol: float = 1e-9,
    execution_mode: str = "structured",
    batch_size: int = 8,
    parallel: bool = True,
    threads: int = 8,
    integral_weights: list[tuple[int, float]] | None = None,
) -> tuple[PopulationResponseMap, ResponseBuildStats]:
    """Build one-hot full-rho/photon responses by exact switch segmentation.

    Each batch is carried in the native packed-Hermitian layout between exact
    switch edges.  This is the same complete density state, including all
    coherences; only the final public response columns are unpacked on demand.
    Per-column wall time is reported as equal-share allocation of each parallel
    batch duration; native solver timing is available only at the batch level.
    """
    import bench_rc_propagation as driver

    indices = np.asarray(basis_indices)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("basis_indices must be a nonempty 1D integer array")
    indices = indices.astype(np.int64, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError("basis_indices must be unique")
    n_states = int(case.n_states)
    if np.any(indices < 0) or np.any(indices >= n_states):
        raise ValueError("basis index is outside the case state layout")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(threads, bool) or int(threads) != threads or threads <= 0:
        raise ValueError("threads must be a positive integer")
    t0, t1 = map(float, t_span)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("t_span must contain finite endpoints with t1 > t0")
    if case.prepared_segmented is None:
        raise ValueError("case must have its exact-segment prepared plan before map building")
    if integral_weights is None:
        integral_weights = [
            (int(i), float(driver.GAMMA))
            for i, state in enumerate(case.system.QN)
            if state.largest.electronic_state == driver.states.ElectronicState.B
        ]
    boundaries = driver.switch_boundaries(t0, t1)
    layout = case.prepared_segmented.layout
    packed_len = layout.packed_len
    response_columns = np.empty((packed_len + 1, indices.size), dtype=np.float64)
    batch_records: list[dict[str, Any]] = []
    allocated_costs: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    solve_calls = 0

    for start in range(0, indices.size, batch_size):
        batch_started = time.perf_counter()
        local_indices = indices[start : start + batch_size]
        # The native batch API accepts (n_trajectories, packed_len) directly.
        # Pack each one-hot diagonal input once, then carry packed output rows
        # without repeated 70x70 complex unpack/pack work at every switch edge.
        rho_batch = np.zeros((local_indices.size, packed_len), dtype=np.float64)
        for row_index, state_index in enumerate(local_indices):
            rho0 = np.zeros((n_states, n_states), dtype=np.complex128)
            rho0[state_index, state_index] = 1.0
            rho_batch[row_index] = layout.pack(rho0)
        photon_totals = np.zeros(local_indices.size, dtype=np.float64)
        segment_records: list[dict[str, Any]] = []
        for lo, hi in pairwise(boundaries):
            switch_state = driver.switch_values(float((lo + hi) / 2.0))
            phase = driver.SWITCH_PHASE_OFFSET_RAD
            parameter_row = np.asarray(
                [0.0, phase if switch_state[0] else -phase,
                 0.0, phase if switch_state[1] else -phase],
                dtype=np.complex128,
            )
            parameter_batch = np.repeat(parameter_row[np.newaxis, :], local_indices.size, axis=0)
            started = time.perf_counter()
            result = solve_lindblad_batch(
                case.prepared_segmented,
                rho_batch,
                (float(lo), float(hi)),
                solver=solver,
                execution_mode=execution_mode,
                reltol=reltol,
                abstol=reltol / 100.0,
                dt=1e-10,
                output="full_and_integral",
                output_when="final",
                save_start=False,
                collect_stats=True,
                parallel=parallel,
                threads=threads if parallel else None,
                parameter_batch=parameter_batch,
                parameter_slots=case.switch_override_slots,
                integral_weights=integral_weights,
                integral_method="solver",
            )
            wall = time.perf_counter() - started
            solve_calls += 1
            values = np.asarray(result.values, dtype=np.float64)
            if values.shape != (local_indices.size, packed_len + 1):
                raise RuntimeError(
                    f"mixed batch returned shape {values.shape}; expected "
                    f"({local_indices.size}, {packed_len + 1})"
                )
            photon_totals += values[:, -1]
            # full_and_integral is exactly packed density coordinates followed
            # by one accumulated integral. Never feed the photon coordinate
            # back as part of rho0.
            rho_batch = np.ascontiguousarray(values[:, :packed_len])
            segment_records.append(
                {
                    "t_span_s": [float(lo), float(hi)],
                    "switch_state": list(switch_state),
                    "wall_seconds": wall,
                    "solver_stats": dict(result.solver_stats or {}),
                }
            )
        for local_index, basis_index in enumerate(local_indices):
            response_columns[:-1, start + local_index] = rho_batch[local_index]
            response_columns[-1, start + local_index] = photon_totals[local_index]
            allocated_costs.append(
                {
                    "basis_index": int(basis_index),
                    "batch_index": len(batch_records),
                    "allocated_wall_seconds": float(
                        sum(row["wall_seconds"] for row in segment_records) / local_indices.size
                    ),
                    "allocation_method": "batch wall divided equally among parallel columns; estimate, not individual timing",
                }
            )
        batch_wall = time.perf_counter() - batch_started
        batch_records.append(
            {
                "basis_indices": local_indices.tolist(),
                "batch_wall_seconds": float(batch_wall),
                "segment_solver_wall_seconds_sum": float(
                    sum(row["wall_seconds"] for row in segment_records)
                ),
                "segment_call_count": len(segment_records),
                "segments": segment_records,
            }
        )

    elapsed = time.perf_counter() - total_started
    response_map = PopulationResponseMap(
        n_states=n_states,
        covered_indices=indices,
        response_columns=response_columns,
        metadata=metadata,
    )
    return response_map, ResponseBuildStats(
        total_wall_seconds=elapsed,
        batch_records=batch_records,
        per_column_allocated_wall_seconds=allocated_costs,
        n_solve_calls=solve_calls,
    )


@dataclass(frozen=True)
class MixtureValidation:
    density_max_abs_error: float
    photon_abs_error: float
    mixture_trace_error: float
    direct_trace_error: float
    mapped_hermiticity_error: float
    direct_hermiticity_error: float
    mapped_min_eigenvalue: float
    direct_min_eigenvalue: float
    population_max_abs_error: float
    population_max_relative_error_above_floor: float
    density_matches: bool
    photons_match: bool
    populations_match: bool
    physical_checks_pass: bool


def _physical_density_diagnostics(density: np.ndarray) -> tuple[float, float, float]:
    trace_error = float(abs(np.trace(density) - 1.0))
    hermiticity_error = float(np.max(np.abs(density - density.conj().T)))
    min_eigenvalue = float(np.linalg.eigvalsh((density + density.conj().T) / 2.0).min())
    return trace_error, hermiticity_error, min_eigenvalue


def validate_mixture_response(
    response_map: PopulationResponseMap,
    weights: np.ndarray,
    direct_result: dict[str, Any],
    *,
    density_atol: float = 1e-7,
    photon_atol: float = 1e-6,
    photon_rtol: float = 1e-5,
    population_atol: float = 1e-7,
    population_relative_tolerance: float = 1e-3,
    population_relative_floor: float = 1e-6,
    trace_tolerance: float = 1e-9,
    hermiticity_tolerance: float = 1e-9,
    minimum_eigenvalue_tolerance: float = 1e-8,
) -> MixtureValidation:
    """Compare map output with a fresh direct mixed-output solve result.

    ``direct_result`` follows ``run_rc_observables.propagate_endpoint``'s
    ``output_kind='photons'`` contract and therefore contains independent
    ``endpoint_density`` and ``photon_yield`` values from one trajectory.
    This function does no propagation itself, keeping physical runs in the
    dedicated benchmark runner.
    """
    mapped = response_map.apply(weights)
    direct_density = np.asarray(direct_result.get("endpoint_density"))
    if direct_density.shape != (response_map.n_states, response_map.n_states):
        raise ValueError("direct_result endpoint_density has incompatible shape")
    if not np.isfinite(direct_density).all():
        raise ValueError("direct_result endpoint_density must be finite")
    direct_photons = float(direct_result["photon_yield"])
    if not np.isfinite(direct_photons):
        raise ValueError("direct_result photon_yield must be finite")
    density_error = float(np.max(np.abs(mapped.density - direct_density)))
    photon_error = abs(mapped.photon_yield - direct_photons)
    trace_error, mapped_hermiticity, mapped_min_eigenvalue = _physical_density_diagnostics(
        mapped.density
    )
    direct_trace_error, direct_hermiticity, direct_min_eigenvalue = (
        _physical_density_diagnostics(direct_density)
    )
    mapped_populations = np.diag(mapped.density).real
    direct_populations = np.diag(direct_density).real
    population_errors = np.abs(mapped_populations - direct_populations)
    population_max_abs = float(population_errors.max(initial=0.0))
    above_floor = np.abs(direct_populations) > population_relative_floor
    population_relative_max = float(
        np.max(population_errors[above_floor] / np.abs(direct_populations[above_floor]))
        if np.any(above_floor)
        else 0.0
    )
    density_matches = density_error <= density_atol
    photons_match = photon_error <= photon_atol + photon_rtol * abs(direct_photons)
    populations_match = (
        population_max_abs <= population_atol
        and population_relative_max <= population_relative_tolerance
    )
    physical_checks_pass = all(
        np.isfinite(value)
        for value in (
            trace_error,
            direct_trace_error,
            mapped_hermiticity,
            direct_hermiticity,
            mapped_min_eigenvalue,
            direct_min_eigenvalue,
        )
    ) and all(
        (
            trace_error <= trace_tolerance,
            direct_trace_error <= trace_tolerance,
            mapped_hermiticity <= hermiticity_tolerance,
            direct_hermiticity <= hermiticity_tolerance,
            mapped_min_eigenvalue >= -minimum_eigenvalue_tolerance,
            direct_min_eigenvalue >= -minimum_eigenvalue_tolerance,
        )
    )
    return MixtureValidation(
        density_max_abs_error=density_error,
        photon_abs_error=float(photon_error),
        mixture_trace_error=trace_error,
        direct_trace_error=direct_trace_error,
        mapped_hermiticity_error=mapped_hermiticity,
        direct_hermiticity_error=direct_hermiticity,
        mapped_min_eigenvalue=mapped_min_eigenvalue,
        direct_min_eigenvalue=direct_min_eigenvalue,
        population_max_abs_error=population_max_abs,
        population_max_relative_error_above_floor=population_relative_max,
        density_matches=density_matches,
        photons_match=photons_match,
        populations_match=populations_match,
        physical_checks_pass=physical_checks_pass,
    )


def canonical_cache_key(metadata: dict[str, Any]) -> str:
    """Hash all supplied physical/runtime/provenance metadata deterministically."""
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_response_map(
    response_map: PopulationResponseMap,
    data_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Write a map and sidecar with checksums and its cache/invalidation key."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        cache_version=np.asarray(1, dtype=np.int64),
        covered_indices=response_map.covered_indices,
        response_columns=response_map.response_columns,
    )
    physical_metadata = dict(response_map.metadata)
    cache_key = canonical_cache_key(physical_metadata)
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    record = {
        "schema": "centrex_tlf.population_response_map.v1",
        "n_states": response_map.n_states,
        "n_covered": int(response_map.covered_indices.size),
        "coverage_indices": response_map.covered_indices.tolist(),
        "cache_key": cache_key,
        "invalidation_fields": sorted(physical_metadata),
        "physical_and_runtime_metadata": physical_metadata,
        "data_path": str(data_path),
        "data_sha256": data_hash,
    }
    metadata_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def load_response_map(
    data_path: Path,
    metadata_path: Path,
    *,
    expected_metadata: dict[str, Any],
) -> PopulationResponseMap:
    """Load only when both the cache key and response artifact hash match."""
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if record.get("schema") != "centrex_tlf.population_response_map.v1":
        raise ValueError("unsupported population-response map schema")
    expected_key = canonical_cache_key(expected_metadata)
    if record.get("cache_key") != expected_key:
        raise ValueError("population-response map cache key does not match current inputs")
    actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if record.get("data_sha256") != actual_hash:
        raise ValueError("population-response map data checksum does not match metadata")
    normalized_expected = json.loads(json.dumps(expected_metadata, sort_keys=True, default=str))
    if record.get("physical_and_runtime_metadata") != normalized_expected:
        raise ValueError("population-response map metadata differs from current inputs")
    with np.load(data_path) as archive:
        if int(np.asarray(archive["cache_version"]).item()) != 1:
            raise ValueError("unsupported population-response map cache version")
        indices = np.asarray(archive["covered_indices"], dtype=np.int64)
        responses = np.asarray(archive["response_columns"], dtype=np.float64)
    if indices.tolist() != record.get("coverage_indices"):
        raise ValueError("population-response coverage differs from metadata")
    if responses.shape[1] != record.get("n_covered"):
        raise ValueError("population-response column count differs from metadata")
    return PopulationResponseMap(
        n_states=int(record["n_states"]),
        covered_indices=indices,
        response_columns=responses,
        metadata=normalized_expected,
    )


__all__ = [
    "MixtureValidation",
    "PopulationMixtureResult",
    "PopulationResponseMap",
    "ResponseBuildStats",
    "build_diagonal_response",
    "canonical_cache_key",
    "load_response_map",
    "make_response_map",
    "save_response_map",
    "validate_mixture_response",
]
