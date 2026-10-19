"""Opt-in full-density maps for strictly periodic CUDA Lindblad evolution."""

from __future__ import annotations

import hashlib
import pickle
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from .cuda import CudaSolverOptions, _basis, _cupy, _packed_initial_states, _solve_cuda_batch
from .state_layout import PackedHermitianLayout

__all__ = ["PeriodicLindbladPropagator", "prepare_periodic_lindblad_propagator"]


def _digest(prepared: Any) -> bytes:
    return hashlib.sha256(pickle.dumps(prepared.to_payload(), protocol=5)).digest()


def _cycles(duration: float, period: float) -> tuple[int, float]:
    quotient = duration / period
    if not np.isfinite(quotient) or quotient > 2**53:
        raise ValueError("too many periods to resolve the fractional cycle in float64")
    nearest = round(quotient)
    if abs(quotient - nearest) <= 32 * np.finfo(float).eps * max(1.0, abs(quotient)):
        return int(nearest), 0.0
    count = int(np.floor(quotient))
    return count, float(duration - count * period)


class PeriodicLindbladPropagator:
    """Prepared cycle map for one fixed generator and declared period.

    Reuses initial J populations, coherences within the retained support, and
    different interaction durations. Detuning/power/envelope changes need a new
    propagator. Arbitrary entry phases and fractional cycles use direct CUDA
    prefix/remainder solves. Full matrices are reconstructed, including exact
    zeros outside a structurally invariant reduced support.

    Construct with :func:`prepare_periodic_lindblad_propagator`.
    """

    def __init__(
        self,
        prepared: Any,
        matrix: np.ndarray,
        keep: np.ndarray,
        period: float,
        phase_origin: float,
        parameter_values: dict,
        dt: float,
        solver: str,
        options: CudaSolverOptions,
        max_map_bytes: int,
        stats: dict,
    ):
        self.layout: PackedHermitianLayout = prepared.layout
        self._period, self._phase_origin, self.stats = period, phase_origin, stats
        self.keep = keep.copy()
        self.keep.flags.writeable = False
        self.excluded = np.setdiff1d(np.arange(self.layout.packed_len), keep)
        self._prepared, self._digest = prepared, _digest(prepared)
        self._matrix = np.ascontiguousarray(matrix)
        self._matrix.flags.writeable = False
        self._parameters = dict(parameter_values)
        self._dt, self._solver, self._options = dt, solver, options
        self._cache_limit = max(1, max_map_bytes // max(1, self._matrix.nbytes))
        self._powers: OrderedDict[int, Any] = OrderedDict()

    @property
    def period(self) -> float:
        return self._period

    @property
    def phase_origin(self) -> float:
        return self._phase_origin

    def clear_cuda_cache(self) -> None:
        """Release resident maps and powers; the CPU snapshot permits later reuse."""
        self._powers.clear()

    def _power(self, index: int) -> Any:
        cp = _cupy()
        if 0 not in self._powers:
            self._powers[0] = cp.asarray(self._matrix)
        if index not in self._powers:
            previous = max(k for k in self._powers if k < index)
            value = self._powers[previous]
            for _ in range(previous + 1, index + 1):
                value = value @ value
            if len(self._powers) >= self._cache_limit:
                candidates = [k for k in self._powers if k != 0]
                if candidates:
                    del self._powers[candidates[0]]
            if len(self._powers) < self._cache_limit:
                self._powers[index] = value
            return value
        self._powers.move_to_end(index)
        return self._powers[index]

    def _direct(self, packed: np.ndarray, start: float, end: float) -> np.ndarray:
        return _solve_cuda_batch(
            self._prepared,
            packed,
            (start, end),
            dt=self._dt,
            solver=self._solver,
            parameter_slots=list(self._parameters) if self._parameters else None,
            parameter_batch=np.repeat(
                np.array(list(self._parameters.values()), complex)[None], len(packed), axis=0
            )
            if self._parameters
            else None,
            output="full",
            cuda_options=self._options,
            maxiters=10_000_000,
        ).values

    def evaluate(
        self,
        rho0_batch: Any,
        times: Sequence[float],
        *,
        t0: float | None = None,
        output: str = "full",
        output_indices: Sequence[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Return `(initial_state, sample_time, output_width)` NumPy values.

        ``times`` are absolute seconds and may be unordered/repeated (e.g.
        ``length / velocities``). Initial states are specified at ``t0``, which
        defaults to ``phase_origin``. Supports full packed density, populations,
        and selected complex entries; photon integrals are not implemented.
        CUDA map powers are cached within the declared memory budget. Calls on
        one propagator should be sequential.
        """
        if _digest(self._prepared) != self._digest:
            raise ValueError("prepared model changed; prepare a new periodic propagator")
        t0 = self.phase_origin if t0 is None else float(t0)
        ts = np.asarray(times, dtype=float)
        if ts.ndim != 1 or not np.isfinite(ts).all() or not np.isfinite(t0) or np.any(ts < t0):
            raise ValueError("times must be finite, one-dimensional, and at or after t0")
        if output not in {"full", "populations", "selected"}:
            raise ValueError("periodic maps support full, populations, or selected output")
        if (output_indices is not None) != (output == "selected"):
            raise ValueError("output_indices is required only for selected output")
        indices = [] if output_indices is None else list(output_indices)
        if output == "selected" and not indices:
            raise ValueError("selected output needs at least one entry")
        for i, j in indices:
            self.layout.diagonal_index(i)
            self.layout.diagonal_index(j)
        packed = _packed_initial_states(self._prepared, rho0_batch)
        if np.any(packed[:, self.excluded] != 0):
            raise ValueError(
                "initial density lies outside the reduced periodic map; prepare with its support or reduce=False"
            )
        result = np.empty((len(packed), len(ts), self.layout.packed_len))
        cp = _cupy()
        with cp.cuda.Device(self._options.device_id):
            # Reduce the absolute entry time to its phase in the declared cycle.
            _, phase = _cycles((t0 - self.phase_origin) % self.period, self.period)
            prefix = self.period - phase if phase else 0.0
            boundary = None
            for target in np.unique(ts):
                duration = float(target - t0)
                if duration == 0:
                    value = packed.copy()
                elif duration < prefix:
                    value = self._direct(
                        packed, self.phase_origin + phase, self.phase_origin + phase + duration
                    )
                else:
                    if boundary is None:
                        start = (
                            self._direct(
                                packed, self.phase_origin + phase, self.phase_origin + self.period
                            )
                            if prefix
                            else packed
                        )
                        boundary = cp.asarray(np.ascontiguousarray(start[:, self.keep].T))
                    cycles, remainder = _cycles(duration - prefix, self.period)
                    y = boundary.copy()
                    bit = 0
                    while cycles:
                        if cycles & 1:
                            y = self._power(bit) @ y
                        cycles >>= 1
                        bit += 1
                    value = np.zeros_like(packed)
                    value[:, self.keep] = y.get().T
                    if remainder:
                        value = self._direct(
                            value, self.phase_origin, self.phase_origin + remainder
                        )
                if not np.isfinite(value).all():
                    raise RuntimeError(
                        "nonfinite periodic-map output; refine dt or shorten propagation"
                    )
                result[:, ts == target] = value[:, None]
        if output == "populations":
            return result[..., : self.layout.n]
        if output == "selected":
            matrices = self.layout.unpack_batch(result)
            return matrices[..., [i for i, j in indices], [j for i, j in indices]]
        return result

    def density_matrices(
        self, rho0_batch: Any, times: Sequence[float], *, t0: float | None = None
    ) -> np.ndarray:
        """Return full complex matrices of shape `(initial, time, n, n)`."""
        return self.layout.unpack_batch(self.evaluate(rho0_batch, times, t0=t0))


def prepare_periodic_lindblad_propagator(
    prepared: Any,
    period: float,
    rho0_batch: Any | None = None,
    *,
    parameter_values: Mapping[str, complex] | None = None,
    phase_origin: float = 0.0,
    dt: float,
    solver: str = "fixed_tsit5",
    reduce: bool = True,
    cuda_options: CudaSolverOptions | None = None,
    max_map_bytes: int = 1_000_000_000,
) -> PeriodicLindbladPropagator:
    """Build a reusable full-density CUDA map for one strictly periodic model.

    ODE defaults are unchanged. ``dt`` must be convergence-tested; ODE tolerances
    do not control this fixed-step map. Uses the usual CUDA decomposed/structured
    model restrictions and rejects nonperiodic/unproven time dependence (e.g.
    Gaussian time envelopes). Square-wave/EOM frequencies must be commensurate
    with ``period``. The graph proof is conservative and may reject a periodic
    expression it cannot establish structurally.

    A supplied initial batch enables exact reachability reduction. New initial
    support requires rebuilding; omit it or set ``reduce=False`` for arbitrary
    densities. Parameter values and entry-phase conventions are fixed in this
    propagator; use ``evaluate(t0=...)`` for arbitrary entry times. Fractional
    cycles are integrated directly. ``max_map_bytes`` bounds retained map/power
    storage, not temporary ODE or matrix-multiplication buffers. A CPU copy of
    the map is retained for cache release/re-upload. Preparation can be expensive.

    ``density_saveat`` and ``step_schedule`` are not supported here; evaluation
    takes its own times, with fixed dt clipped at the period's discontinuities.
    Explicit discontinuities use absolute times within the reference period.
    """
    from ._cuda_lowering import ScalarCompiler
    from ._density_support import public_coordinates, reachable_upper
    from ._periodicity import assert_periodic
    from .batch import _parameter_slot_indices

    period, phase_origin, dt = float(period), float(phase_origin), float(dt)
    if not np.isfinite([period, phase_origin, dt]).all() or period <= 0 or dt <= 0:
        raise ValueError("period and dt must be finite and positive, with a finite phase_origin")
    if not np.isfinite(phase_origin + period) or phase_origin + period <= phase_origin:
        raise ValueError("period cannot advance floating-point phase_origin")
    if solver not in {"fixed_tsit5", "fixed_ck54"}:
        raise ValueError("periodic CUDA maps require fixed_tsit5 or fixed_ck54")
    if not isinstance(reduce, bool):
        raise ValueError("reduce must be a bool")
    if (
        isinstance(max_map_bytes, bool)
        or not isinstance(max_map_bytes, Integral)
        or max_map_bytes < 1
    ):
        raise ValueError("max_map_bytes must be a positive integer")
    options = CudaSolverOptions() if cuda_options is None else cuda_options
    if not isinstance(options, CudaSolverOptions):
        raise TypeError("cuda_options must be CudaSolverOptions")
    if np.asarray(options.density_saveat).size or np.asarray(options.step_schedule).size:
        raise ValueError(
            "periodic maps do not accept density_saveat or step_schedule; supply evaluation times and fixed dt"
        )
    parameters = (
        {}
        if parameter_values is None
        else {str(k): complex(v) for k, v in parameter_values.items()}
    )
    slots = _parameter_slot_indices(prepared, list(parameters))
    graph = prepared.parameter_graph
    full = np.array(
        [complex(v["re"], v["im"]) if v["kind"] == "scalar" else 0 for v in graph["base_values"]],
        dtype=complex,
    )
    for slot, value in zip(slots, parameters.values(), strict=True):
        if graph["base_values"][slot]["kind"] != "scalar":
            raise ValueError("periodic maps require scalar base-parameter overrides")
        full[slot] = value
    if not np.isfinite(full).all():
        raise ValueError("periodic parameters must be finite")
    model = _basis(prepared)
    compiler = ScalarCompiler(
        graph, [c["expression"] for c in prepared.hamiltonian_plan["coefficients"]]
    )
    off, diagonal, _, _ = model
    assert_periodic(
        compiler, full, period, np.any(off != 0, axis=1) | np.any(diagonal != 0, axis=1)
    )
    packed = None if rho0_batch is None else _packed_initial_states(prepared, rho0_batch)
    keep = (
        public_coordinates(prepared.layout.n, reachable_upper(prepared, packed, model))
        if reduce and packed is not None
        else np.arange(prepared.layout.packed_len, dtype=np.int32)
    )
    size = len(keep) ** 2 * np.dtype(float).itemsize
    if size > max_map_bytes:
        raise ValueError(
            f"periodic map needs {size} bytes, exceeding max_map_bytes={max_map_bytes}"
        )
    cp = _cupy()
    if options.device_id >= cp.cuda.runtime.getDeviceCount():
        raise ValueError("CUDA device_id is unavailable")
    tick = time.perf_counter()
    matrix = np.empty((len(keep), len(keep)))
    with cp.cuda.Device(options.device_id):
        for lo in range(0, len(keep), options.batch_size):
            columns = keep[lo : lo + options.batch_size]
            basis = np.zeros((len(columns), prepared.layout.packed_len))
            basis[np.arange(len(columns)), columns] = 1.0
            answer = _solve_cuda_batch(
                prepared,
                basis,
                (phase_origin, phase_origin + period),
                dt=dt,
                solver=solver,
                parameter_slots=list(parameters) if parameters else None,
                parameter_batch=np.repeat(
                    np.array(list(parameters.values()), complex)[None], len(columns), axis=0
                )
                if parameters
                else None,
                output="full",
                cuda_options=options,
                maxiters=10_000_000,
            )
            matrix[:, lo : lo + len(columns)] = answer.values[:, keep].T
    stats = dict(
        construction_seconds=time.perf_counter() - tick,
        map_bytes=size,
        retained_coordinates=len(keep),
        full_coordinates=prepared.layout.packed_len,
        period=period,
        dt=dt,
        solver=solver,
        evolution_device="cuda",
        periodicity_validation="structural",
        error_control=False,
    )
    result = PeriodicLindbladPropagator(
        prepared,
        matrix,
        keep,
        period,
        phase_origin,
        parameters,
        dt,
        solver,
        options,
        max_map_bytes,
        stats,
    )
    with cp.cuda.Device(options.device_id):
        result._power(0)
    return result
