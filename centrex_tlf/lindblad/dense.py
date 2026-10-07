"""Static Lindblad evolution in a real eigenbasis, with optional CUDA projection."""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
from threadpoolctl import threadpool_limits

from .integral_output import INTEGRAL_OUTPUTS
from .plan_static import PreparedLindbladProblem
from .state_layout import PackedHermitianLayout

__all__ = ["DenseLindbladPropagator", "DenseLindbladSession", "prepare_dense_lindblad_propagator"]

_RATES = {"weighted_rate", "photon_rate", "excited_population_rate"}
_WORKER_EXTRACTOR: Any = None
_WORKER_BARRIER: Any = None
_WORKER_LIMIT: Any = None


def _physical_workers() -> int:
    import psutil

    return psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 1) // 2)


def _payload_digest(prepared: PreparedLindbladProblem) -> bytes:
    return hashlib.sha256(pickle.dumps(prepared.to_payload(), protocol=5)).digest()


class DenseLindbladSession:
    """Reusable process workers for one static prepared model.

    Use as a context manager and pass ``dense_session=session`` to dense solver
    calls. Initial states, runtime overrides, times and outputs may change between
    calls. The prepared model and execution mode must remain unchanged. Calls on
    one session are sequential; independent sessions can be used concurrently.
    Workers are created on entry (or lazily on the first solve), with one BLAS
    thread each, and are released by ``close()`` or context-manager exit.
    """

    def __init__(
        self,
        prepared: PreparedLindbladProblem,
        *,
        threads: int | None = None,
        execution_mode: str = "expanded_sparse",
    ):
        _assert_static(prepared)
        prepared.check_execution_mode(execution_mode)
        if prepared.rust_plan is None:
            raise ValueError("DenseLindbladSession requires a Rust-prepared problem")
        if threads is not None and (not isinstance(threads, int) or threads < 1):
            raise ValueError("threads must be a positive integer")
        self._prepared = prepared
        self._execution_mode = execution_mode
        self._workers = _physical_workers() if threads is None else threads
        self.startup_seconds = 0.0
        self._digest = _payload_digest(prepared)
        self._pool: ProcessPoolExecutor | None = None
        self._closed = False
        self._lock = threading.Lock()

    @property
    def prepared(self) -> PreparedLindbladProblem:
        return self._prepared

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    @property
    def workers(self) -> int:
        return self._workers

    def _validate(self, prepared: PreparedLindbladProblem, execution_mode: str) -> None:
        if self._closed:
            raise RuntimeError("DenseLindbladSession is closed")
        if prepared is not self.prepared or execution_mode != self.execution_mode:
            raise ValueError("dense_session requires the same prepared model and execution_mode")
        if _payload_digest(prepared) != self._digest:
            raise ValueError("prepared model changed after session creation; create a new session")

    def _start(self) -> float:
        if self._pool is not None:
            return 0.0
        start = time.perf_counter()
        context = get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(
                self.prepared.to_payload(),
                self.execution_mode,
                context.Barrier(self.workers),
            ),
        )
        try:
            ready = [pool.submit(_worker_ready) for _ in range(self.workers)]
            if len({f.result() for f in ready}) != self.workers:
                raise RuntimeError("dense session worker initialization failed")
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            self._closed = True
            raise
        self._pool = pool
        self.startup_seconds = time.perf_counter() - start
        return self.startup_seconds

    def __enter__(self) -> DenseLindbladSession:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("DenseLindbladSession is already in use")
        try:
            self._validate(self.prepared, self.execution_mode)
            self._start()
        finally:
            self._lock.release()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release workers; idempotent, and waits for an active call to finish."""
        with self._lock:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None

    @contextmanager
    def _lease(self, prepared: PreparedLindbladProblem, execution_mode: str):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("DenseLindbladSession is already in use")
        try:
            self._validate(prepared, execution_mode)
            reused = self._pool is not None
            startup = self._start()
            yield self._pool, startup, reused
        except BrokenProcessPool:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None
            raise
        finally:
            self._lock.release()


def _assert_static(prepared: PreparedLindbladProblem) -> None:
    from .solve import _plan_is_time_dependent

    if _plan_is_time_dependent(prepared):
        names = [
            str(prepared.parameter_graph["slot_names"][c["slot"]])
            for c in prepared.parameter_graph.get("compounds", [])
            if any(int(i["op"]) == 4 for i in c["expression"]["instructions"])
        ]
        detail = f" (time-dependent compounds: {', '.join(names)})" if names else ""
        raise ValueError(
            "dense_eig requires a time-independent Hamiltonian and parameter graph"
            f"{detail}; use solver='dopri5' for time-dependent drives"
        )
    if not np.all(np.isfinite(prepared.dense_c_array)):
        raise ValueError("collapse matrices must be finite and time independent")
    if prepared.expanded_rhs_plan is None:
        raise ValueError(
            "dense_eig requires hamiltonian_representation='decomposed' for analytic Liouvillian extraction"
        )


def _packed_batch(layout: PackedHermitianLayout, rho: Any) -> np.ndarray:
    array = np.asarray(rho)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim == 3:
        if array.shape[1:] != (layout.n, layout.n):
            raise ValueError("density matrix batch has the wrong state dimensions")
        if not np.allclose(array, array.conj().transpose(0, 2, 1), rtol=1e-12, atol=1e-14):
            raise ValueError("initial density matrices must be Hermitian")
        array = np.stack([layout.pack(np.asarray(r, dtype=complex)) for r in array])
    if array.ndim != 2 or array.shape[1] != layout.packed_len or not len(array):
        raise ValueError("expected a nonempty batch of packed states or density matrices")
    if np.iscomplexobj(array) and np.any(array.imag != 0):
        raise ValueError("packed Hermitian states must be real")
    array = np.asarray(array.real, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("initial states must be finite")
    return np.ascontiguousarray(array)


def _matrix_state_batch(layout: PackedHermitianLayout, rho: Any) -> np.ndarray:
    array = np.asarray(rho)
    if array.shape != (layout.n, layout.n):
        raise ValueError(f"rho0 must have shape {(layout.n, layout.n)}")
    return _packed_batch(layout, array[None])


class _StaticLiouvillian:
    """One evaluator/workspace per worker; each parameter point replaces overrides."""

    def __init__(self, payload: dict[str, Any], execution_mode: str, plan: Any = None):
        from ..centrex_tlf_rust import create_lindblad_rhs_evaluator_py, prepare_lindblad_problem_py

        if plan is None:
            plan = prepare_lindblad_problem_py(payload)
        self.evaluator = create_lindblad_rhs_evaluator_py(plan, execution_mode, True)
        if not hasattr(self.evaluator, "set_scalar_parameter_overrides_py"):
            raise RuntimeError(
                "dense_eig requires the updated Rust extension; rebuild/reinstall centrex-tlf"
            )
        self.layout = PackedHermitianLayout(int(payload["n_states"]))
        graph = payload["parameter_graph"]
        self.slots = {name: i for i, name in enumerate(graph["slot_names"])}
        self.base_count = len(graph["base_values"])

    def matrix(self, parameters: Mapping[str, complex]) -> sparse.csr_matrix:
        slots, values = [], []
        for name, value in parameters.items():
            if name not in self.slots:
                raise ValueError(f"unknown parameter slot {name!r}")
            slot = self.slots[name]
            if slot >= self.base_count:
                raise ValueError(f"cannot override compound parameter {name!r}")
            if not np.isfinite(value):
                raise ValueError("parameter values must be finite")
            slots.append(slot)
            values.append(complex(value))
        # Replaces the entire override set, including restoring unspecified
        # defaults; native invalidation prevents stale static Hamiltonian caches.
        self.evaluator.set_scalar_parameter_overrides_py(slots, np.asarray(values, complex))
        rows, cols, values = self.evaluator.jacobian_packed_sparse_py(0.0, 0.0, "analytic")
        n = self.layout.packed_len
        matrix = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Liouvillian contains nonfinite values")
        return matrix


def _phi(w: np.ndarray, times: np.ndarray, order: int) -> np.ndarray:
    wt = w[:, None] * times
    if order == 0:
        return np.exp(wt)
    safe = np.where(w != 0, w, 1)[:, None]
    if order == 1:
        return np.where(
            abs(wt) < 1e-7,
            times * (1 + wt / 2 + wt**2 / 6),
            np.expm1(wt) / safe,
        )
    return np.where(
        abs(wt) < 1e-4,
        times**2 * (0.5 + wt / 6 + wt**2 / 24 + wt**3 / 120),
        (np.expm1(wt) - wt) / safe**2,
    )


@dataclass
class _Projection:
    w: np.ndarray
    active: np.ndarray  # initial state, observable, mode
    feed: np.ndarray
    constant: np.ndarray
    integrated: bool

    def cpu(self, times: np.ndarray) -> np.ndarray:
        order = int(self.integrated)
        result = (self.active @ _phi(self.w, times, order)).real
        if np.any(self.feed != 0):
            result += (self.feed @ _phi(self.w, times, order + 1)).real
        result += self.constant[..., None] * (times if self.integrated else 1)
        return result.transpose(0, 2, 1)


def _torch_device(device: str) -> Any:
    if device == "cpu":
        return None
    if device != "cuda":
        raise ValueError("evolution_device must be 'cpu' or 'cuda'")
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "CUDA evolution requires optional PyTorch; install centrex-tlf[gpu] "
            "with a CUDA-enabled PyTorch build"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("evolution_device='cuda' requires an available CUDA device")
    return torch


def _cuda_project(packages: Sequence[_Projection], times: np.ndarray) -> list[np.ndarray]:
    torch = _torch_device("cuda")
    modes = max(len(p.w) for p in packages)
    if modes == 0:
        return [p.cpu(times) for p in packages]
    initials = max(p.active.shape[0] for p in packages)
    width = packages[0].active.shape[1]
    shape = (len(packages), initials, width, modes)
    w = np.zeros((len(packages), modes), dtype=complex)
    active, feed = np.zeros(shape, complex), np.zeros(shape, complex)
    constant = np.zeros(shape[:-1])
    for i, p in enumerate(packages):
        ni, _, nm = p.active.shape
        w[i, :nm] = p.w
        active[i, :ni, :, :nm], feed[i, :ni, :, :nm] = p.active, p.feed
        constant[i, :ni] = p.constant
    # Chunk sample times so modal functions/intermediates cannot fill VRAM.
    chunk = max(1, min(256, 64_000_000 // max(1, len(packages) * modes * 16)))
    outputs = [np.empty((p.active.shape[0], len(times), width)) for p in packages]
    has_feed = np.any(feed != 0)
    with torch.inference_mode():
        tw = torch.as_tensor(w, device="cuda")
        ta = torch.as_tensor(active, device="cuda").reshape(len(packages), -1, modes)
        tf = (
            torch.as_tensor(feed, device="cuda").reshape(len(packages), -1, modes)
            if has_feed
            else None
        )
        tc = torch.as_tensor(constant, device="cuda").reshape(len(packages), -1, 1)
        safe = torch.where(tw != 0, tw, torch.ones_like(tw))[..., None]
        for start in range(0, len(times), chunk):
            ts = torch.as_tensor(times[start : start + chunk], device="cuda")
            wt = tw[..., None] * ts
            expm1 = torch.expm1(wt)
            phi = torch.where(abs(wt) < 1e-7, ts * (1 + wt / 2 + wt**2 / 6), expm1 / safe)
            if packages[0].integrated:
                out = (ta @ phi).real + tc * ts
                if tf is not None:
                    psi = torch.where(
                        abs(wt) < 1e-4,
                        ts**2 * (0.5 + wt / 6 + wt**2 / 24 + wt**3 / 120),
                        (expm1 - wt) / safe**2,
                    )
                    out += (tf @ psi).real
            else:
                out = (ta @ torch.exp(wt)).real + tc
                if tf is not None:
                    out += (tf @ phi).real
            values = out.reshape(len(packages), initials, width, -1).cpu().numpy()
            for i, p in enumerate(packages):
                outputs[i][:, start : start + chunk] = values[i, : p.active.shape[0]].transpose(
                    0, 2, 1
                )
    return outputs


@dataclass
class DenseLindbladPropagator:
    """Snapshot of one static generator; reusable for initial states and sample times.

    Reduced propagators reject later initial support outside their retained
    subspace. Prepare with the union of desired initial states, or with
    ``rho0_batch=None`` to retain all nonsink packed variables. Full output
    reconstructs the original retained/compact model, not individual levels
    that the OBE builder compacted.
    """

    layout: PackedHermitianLayout
    keep: np.ndarray
    sinks: np.ndarray
    excluded: np.ndarray
    w: np.ndarray
    basis: np.ndarray
    lu: tuple[np.ndarray, np.ndarray] | None
    real: np.ndarray
    positive: np.ndarray
    feed_basis: np.ndarray
    stats: dict[str, Any]

    def _coefficients(self, rho0_batch: Any) -> tuple[np.ndarray, np.ndarray]:
        packed = _packed_batch(self.layout, rho0_batch)
        if np.any(packed[:, self.excluded] != 0):
            raise ValueError(
                "initial state lies outside this reduced propagator; prepare again with these initial states or reduce=False"
            )
        if self.lu is None:
            C = np.empty((0, len(packed)))
        else:
            C = la.lu_solve(self.lu, packed[:, self.keep].T, check_finite=False)
        return C, packed[:, self.sinks].T

    def _projection(self, rho0_batch: Any, rows: np.ndarray, integrated: bool) -> _Projection:
        C, initial_sinks = self._coefficients(rho0_batch)
        real, pos = self.real, self.positive

        def weights(projected: np.ndarray) -> np.ndarray:
            wr = np.einsum("om,mi->iom", projected[:, real], C[real]).astype(complex)
            wp = np.einsum(
                "om,mi->iom",
                projected[:, pos] + 1j * projected[:, pos + 1],
                C[pos] - 1j * C[pos + 1],
            )
            return np.concatenate((wr, wp), axis=-1)

        active = weights(rows[:, self.keep] @ self.basis)
        feed = weights(rows[:, self.sinks] @ self.feed_basis)
        return _Projection(
            np.concatenate((self.w[real], self.w[pos])),
            active,
            feed,
            (rows[:, self.sinks] @ initial_sinks).T,
            integrated,
        )

    def evaluate(
        self,
        rho0_batch: Any,
        times: Sequence[float],
        *,
        t0: float = 0.0,
        output: str = "full",
        output_indices: Sequence[tuple[int, int]] | None = None,
        integral_weights: Sequence[tuple[int, float]] | None = None,
        evolution_device: str = "cpu",
    ) -> np.ndarray:
        """Return `(initial_state, sample_time, output_width)` values.

        Times are absolute; initial states are specified at ``t0``. Full packed
        density reconstruction runs on the CPU even when CUDA projection is
        requested. Analytic integrals include initially populated dark sinks.
        """
        dt = np.asarray(times, dtype=float) - t0
        if dt.ndim != 1 or not np.isfinite(t0) or not np.all(np.isfinite(dt)) or np.any(dt < 0):
            raise ValueError("sample times must be finite and at or after t0")
        _torch_device(evolution_device)
        rows, integrated, complex_selected = _output_rows(
            self.layout, output, output_indices, integral_weights
        )
        if output == "full":
            C, s0 = self._coefficients(rho0_batch)
            values = np.zeros((C.shape[1], len(dt), self.layout.packed_len))
            for start in range(0, len(dt), 32):
                ts = dt[start : start + 32]
                modes = np.empty((len(self.w), C.shape[1], len(ts)))
                mul = _phi(self.w, ts, 0)
                modes[self.real] = C[self.real, :, None] * mul[self.real, None, :].real
                pair = (C[self.positive] - 1j * C[self.positive + 1])[:, :, None] * mul[
                    self.positive, None, :
                ]
                modes[self.positive], modes[self.positive + 1] = pair.real, -pair.imag
                block = (
                    (self.basis @ modes.reshape(len(self.w), -1)).reshape(
                        len(self.keep), C.shape[1], len(ts)
                    )
                    if len(self.w)
                    else np.empty((0, C.shape[1], len(ts)))
                )
                values[:, start : start + len(ts), self.keep] = block.transpose(1, 2, 0)
                sink_rows = np.zeros((len(self.sinks), self.layout.packed_len))
                sink_rows[np.arange(len(self.sinks)), self.sinks] = 1
                sinks = self._projection(rho0_batch, sink_rows, False).cpu(ts)
                values[:, start : start + len(ts), self.sinks] = sinks
            return values
        package = self._projection(rho0_batch, rows, integrated)
        values = package.cpu(dt) if evolution_device == "cpu" else _cuda_project([package], dt)[0]
        if complex_selected:
            values = values[..., 0::2] + 1j * values[..., 1::2]
        return values

    def density_matrices(
        self, rho0_batch: Any, times: Sequence[float], *, t0: float = 0.0
    ) -> np.ndarray:
        packed = self.evaluate(rho0_batch, times, t0=t0)
        result = np.empty((*packed.shape[:2], self.layout.n, self.layout.n), complex)
        for i in range(len(packed)):
            for j in range(packed.shape[1]):
                result[i, j] = self.layout.unpack(packed[i, j])
        return result


def _factor(
    matrix: sparse.csr_matrix, layout: PackedHermitianLayout, packed: np.ndarray | None
) -> DenseLindbladPropagator:
    start = time.perf_counter()
    csc = matrix.tocsc()
    sinks = np.array([i for i in range(layout.n) if csc.indptr[i] == csc.indptr[i + 1]], dtype=int)
    if packed is None:
        reachable = np.ones(layout.packed_len, dtype=bool)
    else:
        reachable = np.any(packed != 0, axis=0)
        frontier = list(np.flatnonzero(reachable))
        while frontier:
            col = frontier.pop()
            for row in csc.indices[csc.indptr[col] : csc.indptr[col + 1]]:
                if not reachable[row]:
                    reachable[row] = True
                    frontier.append(row)
    reachable[sinks] = False
    keep = np.flatnonzero(reachable)
    excluded = np.setdiff1d(np.arange(layout.packed_len), np.r_[keep, sinks])
    # Reduction is structural and exact: no magnitude threshold.
    assert matrix[excluded][:, np.r_[keep, sinks]].nnz == 0
    A = matrix[keep][:, keep].toarray()
    if len(keep):
        w, V = la.eig(A, check_finite=False)
        real, positive = np.flatnonzero(w.imag == 0), np.flatnonzero(w.imag > 0)
        if len(real) + 2 * len(positive) != len(w) or not np.array_equal(
            w[positive + 1], w[positive].conj()
        ):
            raise RuntimeError("unexpected conjugate-pair ordering in real eigendecomposition")
        R = np.empty(A.shape, order="F")
        R[:, real], R[:, positive], R[:, positive + 1] = (
            V[:, real].real,
            V[:, positive].real,
            V[:, positive].imag,
        )
        lu = la.lu_factor(R, check_finite=False)
        rcond, info = la.get_lapack_funcs("gecon", (lu[0],))(lu[0], la.norm(R, 1))
        if info or not np.isfinite(rcond) or rcond < 1e-12:
            raise ValueError(
                "dense_eig eigenbasis is singular or ill-conditioned; use solver='dopri5' (defective generators cannot use eigenbasis propagation)"
            )
        # Probe the entire eigenbasis with deterministic combinations. A full
        # A@V residual would add another cubic operation to every scan point.
        probes = np.random.default_rng(0).choice([-1.0, 1.0], size=(len(w), min(8, len(w))))
        projected = V @ probes
        residual = float(
            la.norm(A @ projected - (V * w) @ probes, np.inf)
            / max(1.0, la.norm(A, np.inf) * la.norm(projected, np.inf))
        )
        if residual > 1e-10:
            raise ValueError("dense_eig factorization residual is too large; use an ODE solver")
    else:
        w, R, real, positive, lu, rcond, residual = (
            np.array([], complex),
            np.empty((0, 0)),
            np.array([], int),
            np.array([], int),
            None,
            1.0,
            0.0,
        )
    feed_basis = matrix[sinks][:, keep] @ R
    stats = dict(
        active_dimension=len(keep),
        packed_dimension=layout.packed_len,
        sink_count=len(sinks),
        reciprocal_condition=float(rcond),
        eigenbasis_residual=residual,
        residual_probe_count=min(8, len(w)),
        factorization_seconds=time.perf_counter() - start,
    )
    return DenseLindbladPropagator(
        layout, keep, sinks, excluded, w, R, lu, real, positive, feed_basis, stats
    )


def prepare_dense_lindblad_propagator(
    prepared: PreparedLindbladProblem,
    rho0_batch: Any | None = None,
    *,
    parameter_values: Mapping[str, complex] | None = None,
    execution_mode: str = "expanded_sparse",
    reduce: bool = True,
) -> DenseLindbladPropagator:
    """Factor one static prepared model without modifying its runtime parameters.

    ``rho0_batch`` accepts `(initial, n, n)` matrices or `(initial, n*n)` packed
    states. Supplying it enables exact reachability reduction. With ``reduce=False``
    all nonsink variables are retained, allowing arbitrary later initial states.
    """
    _assert_static(prepared)
    prepared.check_execution_mode(execution_mode)
    if prepared.rust_plan is None:
        raise ValueError("dense_eig requires a Rust-prepared problem")
    packed = (
        None if rho0_batch is None or not reduce else _packed_batch(prepared.layout, rho0_batch)
    )
    with threadpool_limits(limits=1):
        extractor = _StaticLiouvillian(prepared.to_payload(), execution_mode, prepared.rust_plan)
        return _factor(extractor.matrix(parameter_values or {}), prepared.layout, packed)


def _output_rows(
    layout: PackedHermitianLayout, output: str, indices: Any, weights: Any
) -> tuple[np.ndarray, bool, bool]:
    weighted = output in INTEGRAL_OUTPUTS | _RATES
    if output not in {"full", "populations", "selected", *INTEGRAL_OUTPUTS, *_RATES}:
        raise ValueError(f"unsupported dense output {output!r}")
    if (indices is not None) != (output == "selected"):
        raise ValueError("output_indices is required only for output='selected'")
    if (weights is not None) != weighted:
        raise ValueError("integral_weights is required only for weighted outputs")
    if output == "full":
        return np.empty((0, layout.packed_len)), False, False
    if output == "populations":
        rows = np.zeros((layout.n, layout.packed_len))
        rows[:, : layout.n] = np.eye(layout.n)
    elif output == "selected":
        if not len(indices):
            raise ValueError("output_indices must be nonempty")
        rows = np.zeros((2 * len(indices), layout.packed_len))
        for k, (i, j) in enumerate(indices):
            layout.diagonal_index(i)
            layout.diagonal_index(j)
            if i == j:
                rows[2 * k, i] = 1
            else:
                lo, hi = sorted((i, j))
                rows[2 * k, layout.upper_real_index(lo, hi)] = 1
                rows[2 * k + 1, layout.upper_imag_index(lo, hi)] = 1 if i < j else -1
    else:
        rows = np.zeros((1, layout.packed_len))
        for i, value in weights:
            layout.diagonal_index(i)
            if not np.isfinite(value):
                raise ValueError("integral weights must be finite")
            rows[0, i] += float(value)
    return rows, output in INTEGRAL_OUTPUTS, output == "selected"


def _initialize_worker(payload: dict[str, Any], execution_mode: str, barrier: Any) -> None:
    global _WORKER_EXTRACTOR, _WORKER_LIMIT, _WORKER_BARRIER
    _WORKER_LIMIT = threadpool_limits(limits=1)
    _WORKER_EXTRACTOR = _StaticLiouvillian(payload, execution_mode)
    _WORKER_BARRIER = barrier


def _worker_ready() -> int:
    # One readiness task occupies each process until all plans are prepared.
    # Separates startup from steady-state work without approximating that split.
    _WORKER_BARRIER.wait(timeout=120)
    return os.getpid()


def _work(
    extractor: _StaticLiouvillian,
    parameters: dict[str, complex],
    packed: np.ndarray,
    options: dict[str, Any],
    times: np.ndarray,
    gpu: bool,
) -> Any:
    start = time.perf_counter()
    layout = extractor.layout
    matrix = extractor.matrix(parameters)
    extraction_seconds = time.perf_counter() - start
    propagator = _factor(matrix, layout, packed)
    projection_start = time.perf_counter()
    rows, integrated, selected = _output_rows(
        layout, options["output"], options["output_indices"], options["integral_weights"]
    )
    if gpu and options["output"] != "full":
        values = propagator._projection(packed, rows, integrated)
    else:
        values = propagator.evaluate(packed, times, **options)
    stats = dict(
        propagator.stats,
        parameter_bind_extract_seconds=extraction_seconds,
        projection_seconds=time.perf_counter() - projection_start,
        worker_total_seconds=time.perf_counter() - start,
    )
    return values, stats, selected


def _worker_task(*args: Any) -> Any:
    assert _WORKER_EXTRACTOR is not None
    return _work(_WORKER_EXTRACTOR, *args)


def _sample_times(
    span: Sequence[float], saveat: Any, save_start: bool, output_when: str
) -> np.ndarray:
    t0, t1 = span
    if not np.isfinite([t0, t1]).all() or t1 < t0:
        raise ValueError("t_span must be finite and increasing")
    if output_when == "final":
        return np.array([t1])
    if saveat is None:
        raise ValueError("dense output_when='saveat' requires explicit saveat")
    if np.isscalar(saveat):
        if not np.isfinite(saveat) or saveat <= 0:
            raise ValueError("saveat step must be positive and finite")
        times = np.arange(t0, t1, float(saveat))
        times = np.r_[times, t1]
    else:
        times = np.asarray(saveat, float)
    if (
        times.ndim != 1
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or np.any(times < t0)
        or np.any(times > t1)
    ):
        raise ValueError("saveat must be finite, strictly increasing and inside t_span")
    if not save_start and len(times) and times[0] == t0:
        times = times[1:]
    if not len(times):
        raise ValueError("saveat produced no output times")
    return times


def _solve_dense_batch(
    prepared: PreparedLindbladProblem,
    rho0_batch: Any,
    t_span: Sequence[float],
    *,
    parameter_batch: Any = None,
    parameter_slots: Any = None,
    execution_mode: str = "expanded_sparse",
    saveat: Any = None,
    save_start: bool = True,
    collect_stats: bool = False,
    output: str = "populations",
    output_indices: Any = None,
    output_when: str = "final",
    integral_weights: Any = None,
    integral_method: str = "solver",
    integral_saveat: Any = None,
    dense_output: bool = True,
    parallel: bool = True,
    threads: int | None = None,
    metadata: Any = None,
    stop_event: Any = None,
    evolution_device: str = "cpu",
    gpu_batch_size: int = 8,
    profile_startup: bool = False,
    dense_session: DenseLindbladSession | None = None,
) -> Any:
    from .batch import LindbladBatchResult, _parameter_slot_indices, _parameter_slot_names

    _assert_static(prepared)
    prepared.check_execution_mode(execution_mode)
    if prepared.rust_plan is None:
        raise ValueError("dense_eig requires a Rust-prepared problem")
    if stop_event is not None:
        raise ValueError("dense_eig does not support terminal events; use an ODE solver")
    if integral_method != "solver" or integral_saveat is not None:
        raise ValueError("dense_eig uses analytic integrals; sampled quadrature is not supported")
    if output_when not in {"saveat", "final"}:
        raise ValueError("output_when must be 'saveat' or 'final'")
    if not dense_output and output_when != "final":
        raise ValueError("dense_output=False requires output_when='final'")
    if output in _RATES and output_when != "saveat":
        raise ValueError("rate outputs require output_when='saveat'")
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ValueError("threads must be a positive integer")
    if not isinstance(gpu_batch_size, int) or gpu_batch_size < 1:
        raise ValueError("gpu_batch_size must be a positive integer")
    if profile_startup and not collect_stats:
        raise ValueError("profile_startup requires collect_stats=True")
    if dense_session is not None:
        if not isinstance(dense_session, DenseLindbladSession):
            raise TypeError("dense_session must be a DenseLindbladSession")
        if not parallel:
            raise ValueError("dense_session requires parallel=True")
        if threads is not None and threads != dense_session.workers:
            raise ValueError("threads must match dense_session.workers or be omitted")
        dense_session._validate(prepared, execution_mode)
    _torch_device(evolution_device)
    if len(t_span) != 2:
        raise ValueError("t_span must contain two values")
    span = tuple(map(float, t_span))
    times = _sample_times(span, saveat, save_start, output_when)
    _output_rows(prepared.layout, output, output_indices, integral_weights)
    packed = _packed_batch(prepared.layout, rho0_batch)
    _parameter_slot_indices(prepared, parameter_slots)
    names = _parameter_slot_names(parameter_slots)
    if names is not None and len(set(names)) != len(names):
        raise ValueError("duplicate parameter slots")
    if (parameter_batch is None) != (parameter_slots is None):
        raise ValueError("parameter_batch and parameter_slots must be provided together")
    parameter_values = None
    if parameter_batch is not None:
        parameter_values = np.asarray(parameter_batch, complex)
        if (
            parameter_values.shape != (len(packed), len(names or []))
            or not np.isfinite(parameter_values).all()
        ):
            raise ValueError(
                "parameter_batch must be finite with shape (trajectories, parameter_slots)"
            )
    groups: dict[tuple[complex, ...], list[int]] = {}
    for i in range(len(packed)):
        key = () if parameter_values is None else tuple(parameter_values[i])
        groups.setdefault(key, []).append(i)
    options = dict(
        output=output, output_indices=output_indices, integral_weights=integral_weights, t0=span[0]
    )
    payload = prepared.to_payload()
    work = [
        (
            indices,
            (
                dict(zip(names or [], key, strict=False)),
                packed[indices],
                options,
                times,
                evolution_device == "cuda",
            ),
        )
        for key, indices in groups.items()
    ]
    workers = _physical_workers() if threads is None else threads
    workers = min(workers, len(work)) if parallel else 1
    if dense_session is not None:
        workers = dense_session.workers
    pool_reused = False
    values: np.ndarray | None = None
    stats_list = []
    pending = []
    pending_bytes = 0
    start = time.perf_counter()

    def put(indices: list[int], array: np.ndarray) -> None:
        nonlocal values
        if values is None:
            values = np.empty((len(packed), *array.shape[1:]), array.dtype)
        values[indices] = array

    def flush() -> None:
        nonlocal pending_bytes
        if not pending:
            return
        projected = _cuda_project([p for _, p, _ in pending], times - span[0])
        for (indices, _, selected), array in zip(pending, projected, strict=False):
            if selected:
                array = array[..., 0::2] + 1j * array[..., 1::2]
            put(indices, array)
        pending.clear()
        pending_bytes = 0

    def accept(indices: list[int], result: Any) -> None:
        nonlocal pending_bytes
        array, stats, selected = result
        stats_list.append(stats)
        if isinstance(array, _Projection):
            size = array.active.nbytes + array.feed.nbytes
            if pending_bytes + size > 128_000_000:
                flush()
            pending.append((indices, array, selected))
            pending_bytes += size
            if len(pending) >= gpu_batch_size:
                flush()
        else:
            put(indices, array)

    with threadpool_limits(limits=1):
        startup_start = time.perf_counter()
        if workers == 1 and dense_session is None:
            extractor = _StaticLiouvillian(payload, execution_mode, prepared.rust_plan)
            startup_seconds = time.perf_counter() - startup_start
            compute_start = time.perf_counter()
            for indices, args in work:
                accept(indices, _work(extractor, *args))
            flush()
            steady_seconds = time.perf_counter() - compute_start
        else:
            context = get_context("spawn")
            pool_context = (
                dense_session._lease(prepared, execution_mode)
                if dense_session is not None
                else ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(
                        payload,
                        execution_mode,
                        context.Barrier(workers) if profile_startup else None,
                    ),
                )
            )
            with pool_context as handle:
                if dense_session is not None:
                    pool, startup_seconds, pool_reused = handle
                else:
                    pool = handle
                if profile_startup and dense_session is None:
                    ready = [pool.submit(_worker_ready) for _ in range(workers)]
                    assert len({f.result() for f in ready}) == workers
                    startup_seconds = time.perf_counter() - startup_start
                elif dense_session is None:
                    startup_seconds = None
                compute_start = time.perf_counter()
                # Bound queued tasks/results as well as GPU batches.
                iterator = iter(work)
                futures = {}
                for _ in range(min(len(work), workers * 2)):
                    indices, args = next(iterator)
                    futures[pool.submit(_worker_task, *args)] = indices
                try:
                    while futures:
                        future = next(as_completed(futures))
                        accept(futures.pop(future), future.result())
                        next_item = next(iterator, None)
                        if next_item is not None:
                            indices, args = next_item
                            futures[pool.submit(_worker_task, *args)] = indices
                finally:
                    for future in futures:
                        future.cancel()
                flush()
                steady_seconds = (
                    time.perf_counter() - compute_start
                    if profile_startup or dense_session is not None
                    else None
                )
    assert values is not None
    if output_when == "final":
        values = values[:, 0]
    stats = dict(
        solver="dense_eig",
        evolution_device=evolution_device,
        workers=workers,
        blas_threads_per_worker=1,
        factorization_count=len(groups),
        saved_points=len(times),
        total_seconds=time.perf_counter() - start,
        worker_startup_seconds=startup_seconds,
        steady_state_seconds=steady_seconds,
        evaluator_count=workers,
        startup_synchronized=(profile_startup or dense_session is not None) and workers > 1,
        pool_reused=pool_reused,
        session_startup_seconds=None if dense_session is None else dense_session.startup_seconds,
        factorizations=stats_list,
        full_reconstruction_device="cpu",
        function_evaluations=0,
    )
    return LindbladBatchResult(
        times,
        values,
        output,
        None if output_indices is None else list(output_indices),
        len(packed),
        names,
        parameter_values,
        stats if collect_stats else None,
        {} if metadata is None else dict(metadata),
    )


def _solve_dense_single(
    prepared: PreparedLindbladProblem, rho0: Any, t_span: Sequence[float], **kwargs: Any
) -> Any:
    from .solve import LindbladObservableResult, LindbladResult

    packed = _matrix_state_batch(prepared.layout, rho0)
    if (
        kwargs.get("saveat") is None
        and kwargs.get("output_when", "saveat") == "saveat"
        and kwargs.get("output", "full") not in INTEGRAL_OUTPUTS | _RATES
    ):
        kwargs["saveat"] = np.unique(np.asarray(t_span, float))
    result = _solve_dense_batch(
        prepared, packed, t_span, parallel=kwargs.get("dense_session") is not None, **kwargs
    )
    if result.output == "full":
        data = result.values[0]
        if data.ndim == 1:
            data = data[None]
        return LindbladResult(result.t, data, prepared.layout, result.solver_stats)
    data = result.values[0]
    if result.output in INTEGRAL_OUTPUTS | _RATES:
        data = data[..., 0]
    return LindbladObservableResult(
        result.t, data, result.output, result.output_indices, result.solver_stats
    )


def _dense_grid_scan(
    prepared: PreparedLindbladProblem,
    rho0: Any,
    t_span: Sequence[float],
    scan: Mapping[Any, Any],
    kwargs: dict[str, Any],
) -> Any:
    from .batch import _parameter_slot_names

    axes = [np.asarray(a, complex).reshape(-1) for a in scan.values()]
    if not axes or any(len(a) == 0 for a in axes):
        raise ValueError("scan axes must be nonempty")
    batch = kwargs.pop("rho0_batch", None)
    if batch is not None:
        if rho0 is not None:
            raise ValueError("pass rho0=None when supplying rho0_batch to a dense grid")
        packed = _packed_batch(prepared.layout, batch)
    elif np.asarray(rho0).ndim == 1:
        packed = _packed_batch(prepared.layout, rho0)
    else:
        packed = _matrix_state_batch(prepared.layout, rho0)
    mesh = np.meshgrid(*axes, indexing="ij")
    parameters = np.stack([a.ravel() for a in mesh], axis=1)
    result = _solve_dense_batch(
        prepared,
        np.tile(packed, (len(parameters), 1)),
        t_span,
        parameter_slots=list(scan),
        parameter_batch=np.repeat(parameters, len(packed), axis=0),
        **kwargs,
    )
    names = _parameter_slot_names(list(scan)) or []
    result.metadata.update(
        scan_kind="grid",
        grid_shape=tuple(len(a) for a in axes),
        grid_axes=dict(zip(names, axes, strict=False)),
        compact_grid=False,
        initial_condition_count=len(packed),
        trajectory_order="parameter_point_then_initial_state",
    )
    return result
