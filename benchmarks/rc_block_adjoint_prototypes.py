"""Benchmark-only shared-H block and scalar-adjoint RC prototypes.

These routines deliberately live outside ``centrex_tlf``.  They use the
existing prepared Hamiltonian/decay representation, integrate switch intervals
at their exact boundaries, and preserve absolute time for the smooth EOM phase.
The block solver shares one adaptive controller across its initial states; that
coupling is part of the measured method and must be reported.  The adjoint
returns scalar objectives only and cannot reconstruct a final density matrix.

No molecular case is built at import time.  The physical comparison runner is
owned by the benchmark checker.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
import time
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from centrex_tlf.lindblad.ir import evaluate_parameter_graph_py, fill_hamiltonian_py
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - environment dependency guard
    raise ImportError(
        "the benchmark-only block/adjoint prototypes require Numba for their "
        "packed-state and structured-decay loops"
    ) from exc


_NUMBA_KERNELS_WARMED = False


def warm_numba_kernels() -> float:
    """Compile the benchmark hotloops once and return their warm-up cost."""
    global _NUMBA_KERNELS_WARMED
    if _NUMBA_KERNELS_WARMED:
        return 0.0
    started = time.perf_counter()
    n = 2
    rho = np.zeros((1, n, n), dtype=np.complex128)
    packed = _pack_block_kernel(rho)
    _unpack_block_kernel(packed, n)
    rates = np.zeros(n, dtype=np.float64)
    indices = np.zeros(0, dtype=np.int64)
    jump_rates = np.zeros(0, dtype=np.float64)
    _structured_decay_batch_kernel(rho, rates, indices, indices, jump_rates)
    _adjoint_decay_kernel(rho[0], rates, indices, indices, jump_rates)
    _NUMBA_KERNELS_WARMED = True
    return time.perf_counter() - started


@njit(cache=False)
def _unpack_block_kernel(packed: np.ndarray, n: int) -> np.ndarray:
    k_count = packed.shape[0]
    result = np.zeros((k_count, n, n), dtype=np.complex128)
    for k in range(k_count):
        for i in range(n):
            result[k, i, i] = packed[k, i]
        cursor = n
        for i in range(n):
            for j in range(i + 1, n):
                value = packed[k, cursor] + 1j * packed[k, cursor + 1]
                result[k, i, j] = value
                result[k, j, i] = np.conjugate(value)
                cursor += 2
    return result


@njit(cache=False)
def _pack_block_kernel(matrices: np.ndarray) -> np.ndarray:
    k_count, n, _ = matrices.shape
    result = np.empty((k_count, n * n), dtype=np.float64)
    for k in range(k_count):
        for i in range(n):
            result[k, i] = matrices[k, i, i].real
        cursor = n
        for i in range(n):
            for j in range(i + 1, n):
                value = matrices[k, i, j]
                result[k, cursor] = value.real
                result[k, cursor + 1] = value.imag
                cursor += 2
    return result


@njit(cache=False)
def _structured_decay_batch_kernel(
    rho: np.ndarray,
    source_rates: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
    jump_rates: np.ndarray,
) -> np.ndarray:
    k_count, n, _ = rho.shape
    result = np.empty_like(rho)
    for k in range(k_count):
        for i in range(n):
            for j in range(n):
                result[k, i, j] = -0.5 * (source_rates[i] + source_rates[j]) * rho[k, i, j]
        for jump in range(len(jump_rates)):
            target = targets[jump]
            source = sources[jump]
            result[k, target, target] += jump_rates[jump] * rho[k, source, source]
    return result


@njit(cache=False)
def _adjoint_decay_kernel(
    lam: np.ndarray,
    source_rates: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
    jump_rates: np.ndarray,
) -> np.ndarray:
    n = lam.shape[0]
    result = np.empty_like(lam)
    for i in range(n):
        for j in range(n):
            result[i, j] = -0.5 * (source_rates[i] + source_rates[j]) * lam[i, j]
    for jump in range(len(jump_rates)):
        target = targets[jump]
        source = sources[jump]
        result[source, source] += jump_rates[jump] * lam[target, target]
    return result


def pack_density_block(matrices: np.ndarray, layout: PackedHermitianLayout) -> np.ndarray:
    """Pack K Hermitian matrices as a row-major ``(K, n**2)`` real block."""
    values = np.asarray(matrices, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1:] != (layout.n, layout.n):
        raise ValueError(f"matrices must have shape (K, {layout.n}, {layout.n})")
    if values.shape[0] == 0:
        raise ValueError("at least one density matrix is required")
    if not np.all(np.isfinite(values)) or not np.allclose(
        values, values.conjugate().swapaxes(1, 2), rtol=0.0, atol=1e-12
    ):
        raise ValueError("every input density matrix must be finite and Hermitian")
    return _pack_block_kernel(np.ascontiguousarray(values))


def unpack_density_block(packed: np.ndarray, layout: PackedHermitianLayout) -> np.ndarray:
    """Unpack a row-major ``(K, n**2)`` real block to Hermitian matrices."""
    values = np.asarray(packed, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != layout.packed_len:
        raise ValueError(f"packed must have shape (K, {layout.packed_len})")
    if values.shape[0] == 0:
        raise ValueError("at least one packed state is required")
    return _unpack_block_kernel(np.ascontiguousarray(values), layout.n)


def _jump_arrays(prepared: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jumps = prepared.structured_jumps
    return (
        np.asarray([jump["target"] for jump in jumps], dtype=np.int64),
        np.asarray([jump["source"] for jump in jumps], dtype=np.int64),
        np.asarray([jump["rate"] for jump in jumps], dtype=np.float64),
    )


def _hamiltonian_pattern(prepared: Any) -> sp.csr_matrix:
    """Return the fixed full-H CSR pattern implied by a prepared H plan."""
    plan = prepared.hamiltonian_plan
    n = int(prepared.layout.n)
    pattern = np.zeros((n, n), dtype=bool)

    def add_upper(i: int, j: int) -> None:
        pattern[i, j] = True
        pattern[j, i] = True

    kind = plan.get("kind", "entrywise")
    if kind == "decomposed":
        static = np.asarray(plan["static_matrix"], dtype=np.complex128)
        pattern |= static != 0.0
        pattern |= pattern.T
        for coefficient in plan.get("coefficients", []):
            if "basis_terms" in coefficient:
                for term in coefficient["basis_terms"]:
                    add_upper(int(term["i"]), int(term["j"]))
            elif "basis_row_segments" in coefficient:
                for segment in coefficient["basis_row_segments"]:
                    row = int(segment["row"])
                    start = int(segment["start_col"])
                    for offset, (real, imag) in enumerate(
                        zip(segment["values_re"], segment["values_im"], strict=True)
                    ):
                        if real != 0.0 or imag != 0.0:
                            add_upper(row, start + offset)
            else:
                basis = np.asarray(coefficient["basis_upper"], dtype=np.complex128)
                for i, j in zip(*np.nonzero(basis), strict=True):
                    add_upper(int(i), int(j))
    elif kind == "entrywise":
        for entry in plan["entries"]:
            add_upper(int(entry["i"]), int(entry["j"]))
    else:  # pragma: no cover - prepared plans validate their own kind
        raise ValueError(f"unsupported Hamiltonian plan kind {kind!r}")

    rows, cols = np.nonzero(pattern)
    matrix = sp.csr_matrix(
        (np.zeros(rows.size, dtype=np.complex128), (rows, cols)), shape=(n, n)
    )
    matrix.sort_indices()
    return matrix


@dataclass
class HamiltonianCSR:
    """Reusable sparse structure whose data are refreshed from H(t)."""

    prepared: Any
    matrix: sp.csr_matrix
    row_indices: np.ndarray
    col_indices: np.ndarray

    @classmethod
    def from_prepared(cls, prepared: Any) -> "HamiltonianCSR":
        matrix = _hamiltonian_pattern(prepared)
        rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr))
        cols = matrix.indices.astype(np.int64, copy=True)
        return cls(prepared, matrix, rows, cols)

    def at(self, time_s: float) -> sp.csr_matrix:
        slots = evaluate_parameter_graph_py(self.prepared.parameter_graph, float(time_s))
        dense = fill_hamiltonian_py(self.prepared.hamiltonian_plan, slots, float(time_s))
        self.matrix.data[:] = dense[self.row_indices, self.col_indices]
        return self.matrix


def block_rhs_from_hamiltonian(
    prepared: Any,
    packed_block: np.ndarray,
    hamiltonian: sp.spmatrix | np.ndarray,
) -> np.ndarray:
    """Evaluate one shared-H RHS for K packed Hermitian density matrices.

    The density block is laid out horizontally so one sparse-matrix/dense-block
    product evaluates Hρ for all K columns.  Since every packed density and H
    are Hermitian, ρH is recovered as (Hρ)†; no second Hamiltonian product is
    needed.  Structured single-jump decay is then applied to every matrix.
    """
    targets, sources, jump_rates = _jump_arrays(prepared)
    return _block_rhs_core(
        prepared,
        packed_block,
        hamiltonian,
        np.asarray(prepared.source_decay_rates, dtype=np.float64),
        targets,
        sources,
        jump_rates,
    )


def block_rhs(prepared: Any, hamiltonian: HamiltonianCSR, packed_block: np.ndarray, time_s: float) -> np.ndarray:
    """Time-dependent block RHS, refreshing shared coefficients once."""
    return block_rhs_from_hamiltonian(prepared, packed_block, hamiltonian.at(time_s))


@dataclass
class PreparedRHSContext:
    """Cache Hamiltonian CSR structure and decay arrays for one switch state."""

    prepared: Any
    hamiltonian: HamiltonianCSR
    source_rates: np.ndarray
    jump_targets: np.ndarray
    jump_sources: np.ndarray
    jump_rates: np.ndarray

    @classmethod
    def from_prepared(cls, prepared: Any) -> "PreparedRHSContext":
        targets, sources, rates = _jump_arrays(prepared)
        return cls(
            prepared,
            HamiltonianCSR.from_prepared(prepared),
            np.asarray(prepared.source_decay_rates, dtype=np.float64),
            targets,
            sources,
            rates,
        )

    def rhs_block(self, packed_block: np.ndarray, time_s: float) -> np.ndarray:
        return _block_rhs_core(
            self.prepared,
            packed_block,
            self.hamiltonian.at(time_s),
            self.source_rates,
            self.jump_targets,
            self.jump_sources,
            self.jump_rates,
        )

    def rhs_adjoint(
        self, packed_lambda: np.ndarray, time_s: float, running_weight: np.ndarray
    ) -> np.ndarray:
        return _adjoint_rhs_core(
            self.prepared,
            packed_lambda,
            self.hamiltonian.at(time_s),
            self.source_rates,
            self.jump_targets,
            self.jump_sources,
            self.jump_rates,
            running_weight,
        )


def _block_rhs_core(
    prepared: Any,
    packed_block: np.ndarray,
    hamiltonian: sp.spmatrix | np.ndarray,
    source_rates: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
    jump_rates: np.ndarray,
) -> np.ndarray:
    packed = np.asarray(packed_block, dtype=np.float64)
    if packed.ndim != 2 or packed.shape[1] != prepared.layout.packed_len:
        raise ValueError("packed_block must have shape (K, n_states**2)")
    rho = unpack_density_block(packed, prepared.layout)
    k_count, n, _ = rho.shape
    rho_columns = np.empty((n, n * k_count), dtype=np.complex128, order="F")
    for k in range(k_count):
        rho_columns[:, k * n : (k + 1) * n] = rho[k]

    left_product = np.asarray(hamiltonian @ rho_columns, dtype=np.complex128)
    left_batch = left_product.reshape(n, k_count, n).transpose(1, 0, 2)
    commutator = left_batch - left_batch.conjugate().swapaxes(1, 2)
    drho = -1j * commutator
    drho += _structured_decay_batch_kernel(rho, source_rates, targets, sources, jump_rates)
    return _pack_block_kernel(np.ascontiguousarray(drho))


def _prepare_contexts(case: Any) -> tuple[dict[tuple[int, int], PreparedRHSContext], float]:
    started = time.perf_counter()
    contexts = {
        key: PreparedRHSContext.from_prepared(prepared)
        for key, prepared in case.prepared_switch_states.items()
    }
    return contexts, time.perf_counter() - started


def integrate_segmented_block(
    case: Any,
    rho0_batch: np.ndarray,
    t_span: tuple[float, float],
    *,
    solver: str = "DOP853",
    reltol: float = 1e-9,
    abstol: float = 1e-11,
    dt: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Propagate K initial states with shared H products and one controller.

    Returned matrices are the full final density matrices.  Exact microwave
    edges are segment boundaries; each interval uses its own frozen switch
    plan and evaluates the smooth EOM modulation at absolute time.
    """
    from bench_rc_propagation import _prepare_switch_plans, switch_boundaries, switch_values

    t0, t1 = map(float, t_span)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("t_span must contain finite increasing endpoints")
    rho = np.asarray(rho0_batch, dtype=np.complex128)
    if rho.ndim == 2:
        rho = rho[None, ...]
    n = case.n_states
    if rho.ndim != 3 or rho.shape[1:] != (n, n) or rho.shape[0] == 0:
        raise ValueError(f"rho0_batch must have shape (K, {n}, {n}) with K>0")
    warmup_seconds = warm_numba_kernels()
    plan_started = time.perf_counter()
    _prepare_switch_plans(case, static=False)
    switch_plan_prep_seconds = time.perf_counter() - plan_started
    contexts, context_prep_seconds = _prepare_contexts(case)
    packed = pack_density_block(rho, case.prepared_dynamic.layout)
    edges = switch_boundaries(t0, t1)
    total_nfev = 0
    total_segments = 0
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for lo, hi in pairwise(edges):
        key = switch_values(0.5 * (float(lo) + float(hi)))
        context = contexts[key]
        prepared = context.prepared
        k_count = rho.shape[0]
        y0 = packed.reshape(-1)

        def rhs(time_s: float, y: np.ndarray) -> np.ndarray:
            block = y.reshape(k_count, prepared.layout.packed_len)
            return context.rhs_block(block, time_s).reshape(-1)

        solution = solve_ivp(
            rhs,
            (float(lo), float(hi)),
            y0,
            method=solver,
            rtol=reltol,
            atol=abstol,
            first_step=min(dt, float(hi - lo)),
            t_eval=[float(hi)],
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        packed = solution.y[:, -1].reshape(k_count, prepared.layout.packed_len)
        rho = unpack_density_block(packed, prepared.layout)
        total_nfev += int(solution.nfev)
        total_segments += 1
        rows.append({"lo_s": float(lo), "hi_s": float(hi), "nfev": int(solution.nfev)})
    return rho, {
        "wall_seconds": time.perf_counter() - started,
        "numba_warmup_seconds": warmup_seconds,
        "switch_plan_prep_seconds": switch_plan_prep_seconds,
        "rhs_context_prep_seconds": context_prep_seconds,
        "segments": total_segments,
        "nfev_shared_controller": total_nfev,
        "batch_size": int(rho.shape[0]),
        "solver": solver,
        "rtol": reltol,
        "atol": abstol,
        "initial_dt_s": dt,
        "segment_stats": rows,
        "adaptive_controller_scope": "one shared controller over the entire packed initial-state block",
        "output": "full final density for each initial state",
    }


@dataclass(frozen=True)
class ScalarAdjointObjective:
    """One scalar functional J=Tr(terminal*rho(T))+∫Tr(weight*rho)dt."""

    name: str
    running_weight: np.ndarray
    terminal: np.ndarray
    output_kind: str


def photon_objective(case: Any, decay_rate: float) -> ScalarAdjointObjective:
    """Construct the photon-integral objective Γ Tr(P_B ρ)."""
    weight = np.zeros((case.n_states, case.n_states), dtype=np.complex128)
    for idx, qn in enumerate(case.system.QN):
        if qn.largest.electronic_state.name == "B":
            weight[idx, idx] = float(decay_rate)
    terminal = np.zeros_like(weight)
    return ScalarAdjointObjective("photon_integral", weight, terminal, "scalar")


def selected_population_objective(case: Any, state_index: int) -> ScalarAdjointObjective:
    """Construct final population of one basis state as a terminal projector."""
    if not 0 <= int(state_index) < case.n_states:
        raise IndexError("state_index is outside the OBE basis")
    return population_projector_objective(case, [int(state_index)])


def population_projector_objective(
    case: Any, state_indices: Sequence[int], *, name: str | None = None
) -> ScalarAdjointObjective:
    """Construct a final-population objective for a group of basis states."""
    indices = np.asarray(tuple(state_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("state_indices must contain at least one basis index")
    if np.any(indices < 0) or np.any(indices >= case.n_states):
        raise IndexError("a state index is outside the OBE basis")
    if np.unique(indices).size != indices.size:
        raise ValueError("state_indices must not contain duplicates")
    terminal = np.zeros((case.n_states, case.n_states), dtype=np.complex128)
    terminal[indices, indices] = 1.0
    return ScalarAdjointObjective(
        name or f"final_population_group_{len(indices)}",
        np.zeros_like(terminal),
        terminal,
        "scalar",
    )


def j1_population_objective(case: Any) -> ScalarAdjointObjective:
    """Final population in the complete ground-state J=1 subspace."""
    from centrex_tlf.states import ElectronicState, QuantumSelector

    selector = QuantumSelector(electronic=ElectronicState.X, J=1)
    indices = np.asarray(selector.get_indices(case.system.QN), dtype=np.int64).ravel()
    if indices.size == 0:
        raise ValueError("the prepared OBE basis contains no ground-state J=1 levels")
    return population_projector_objective(case, indices, name="final_population_X_J1")


def adjoint_rhs_from_hamiltonian(
    prepared: Any,
    packed_lambda: np.ndarray,
    hamiltonian: sp.spmatrix | np.ndarray,
    running_weight: np.ndarray,
) -> np.ndarray:
    """RHS for physical-time backward solve: λ' = -L†(λ) - W."""
    targets, sources, jump_rates = _jump_arrays(prepared)
    return _adjoint_rhs_core(
        prepared,
        packed_lambda,
        hamiltonian,
        np.asarray(prepared.source_decay_rates, dtype=np.float64),
        targets,
        sources,
        jump_rates,
        running_weight,
    )


def _adjoint_rhs_core(
    prepared: Any,
    packed_lambda: np.ndarray,
    hamiltonian: sp.spmatrix | np.ndarray,
    source_rates: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
    jump_rates: np.ndarray,
    running_weight: np.ndarray,
) -> np.ndarray:
    packed = np.asarray(packed_lambda, dtype=np.float64).reshape(1, -1)
    lam = _unpack_block_kernel(np.ascontiguousarray(packed), prepared.layout.n)[0]
    left = np.asarray(hamiltonian @ lam, dtype=np.complex128)
    adjoint_commutator = 1j * (left - left.conjugate().T)
    decay = _adjoint_decay_kernel(lam, source_rates, targets, sources, jump_rates)
    derivative = -adjoint_commutator - decay - np.asarray(running_weight, dtype=np.complex128)
    return _pack_block_kernel(np.ascontiguousarray(derivative[None, :, :]))[0]


def adjoint_rhs(
    prepared: Any,
    hamiltonian: HamiltonianCSR,
    packed_lambda: np.ndarray,
    time_s: float,
    running_weight: np.ndarray,
) -> np.ndarray:
    return adjoint_rhs_from_hamiltonian(
        prepared, packed_lambda, hamiltonian.at(time_s), running_weight
    )


def integrate_segmented_adjoint(
    case: Any,
    objective: ScalarAdjointObjective,
    t_span: tuple[float, float],
    *,
    solver: str = "DOP853",
    reltol: float = 1e-9,
    abstol: float = 1e-11,
    dt: float = 1e-10,
    requested_output: str = "scalar",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Integrate one full-Hermitian scalar adjoint backward over exact edges.

    The diagonal of λ(0) gives the selected objective for every one-hot
    diagonal initial state; a diagonal mixture is its dot product with those
    values.  Multiple observables require multiple adjoints.  A scalar adjoint
    cannot return a full final density and rejects such a request explicitly.
    """
    if requested_output in {"full", "full_density", "density_matrix"}:
        raise ValueError(
            "a scalar adjoint returns objective values, not the full final density matrix"
        )
    if objective.output_kind != "scalar":
        raise ValueError("this prototype accepts scalar-output adjoint objectives only")
    from bench_rc_propagation import _prepare_switch_plans, switch_boundaries, switch_values

    t0, t1 = map(float, t_span)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("t_span must contain finite increasing endpoints")
    n = case.n_states
    expected = (n, n)
    if np.asarray(objective.running_weight).shape != expected or np.asarray(objective.terminal).shape != expected:
        raise ValueError(f"objective matrices must have shape {expected}")
    weight = np.asarray(objective.running_weight, dtype=np.complex128)
    terminal = np.asarray(objective.terminal, dtype=np.complex128)
    if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(terminal)):
        raise ValueError("objective matrices must be finite")
    if not np.allclose(weight, weight.conjugate().T, rtol=0.0, atol=1e-12):
        raise ValueError("running_weight must be Hermitian")
    if not np.allclose(terminal, terminal.conjugate().T, rtol=0.0, atol=1e-12):
        raise ValueError("terminal must be Hermitian")
    warmup_seconds = warm_numba_kernels()
    plan_started = time.perf_counter()
    _prepare_switch_plans(case, static=False)
    switch_plan_prep_seconds = time.perf_counter() - plan_started
    contexts, context_prep_seconds = _prepare_contexts(case)
    packed = _pack_block_kernel(np.ascontiguousarray(terminal[None, :, :]))[0]
    edges = switch_boundaries(t0, t1)
    total_nfev = 0
    total_segments = 0
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for lo, hi in reversed(list(pairwise(edges))):
        key = switch_values(0.5 * (float(lo) + float(hi)))
        context = contexts[key]

        def rhs(time_s: float, y: np.ndarray) -> np.ndarray:
            return context.rhs_adjoint(y, time_s, weight)

        solution = solve_ivp(
            rhs,
            (float(hi), float(lo)),
            packed,
            method=solver,
            rtol=reltol,
            atol=abstol,
            first_step=min(dt, float(hi - lo)),
            t_eval=[float(lo)],
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        packed = np.asarray(solution.y[:, -1], dtype=np.float64)
        total_nfev += int(solution.nfev)
        total_segments += 1
        rows.append({"lo_s": float(lo), "hi_s": float(hi), "nfev": int(solution.nfev)})
    lambda0 = _unpack_block_kernel(np.ascontiguousarray(packed[None, :]), n)[0]
    return lambda0, {
        "wall_seconds": time.perf_counter() - started,
        "numba_warmup_seconds": warmup_seconds,
        "switch_plan_prep_seconds": switch_plan_prep_seconds,
        "rhs_context_prep_seconds": context_prep_seconds,
        "segments": total_segments,
        "nfev_backward": total_nfev,
        "solver": solver,
        "rtol": reltol,
        "atol": abstol,
        "initial_dt_s": dt,
        "objective": objective.name,
        "output": "scalar objective response for all diagonal initial states",
        "segment_stats_reverse": rows,
    }


def diagonal_objective_responses(lambda0: np.ndarray) -> np.ndarray:
    """Return J(e_i) for every diagonal one-hot initial density."""
    matrix = np.asarray(lambda0, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("lambda0 must be square")
    return np.diag(matrix).real.copy()


def diagonal_mixture_response(lambda0: np.ndarray, populations: np.ndarray) -> float:
    """Evaluate an adjoint scalar objective for a normalized diagonal mixture."""
    weights = np.asarray(populations, dtype=np.float64)
    responses = diagonal_objective_responses(lambda0)
    if weights.shape != responses.shape or np.any(~np.isfinite(weights)):
        raise ValueError("populations must be a finite vector matching the basis")
    return float(np.dot(weights, responses))


def authorized_case_configs() -> dict[str, Any]:
    """Return the nominal, +20 MHz detuned, and optical 3Γ pilot configs."""
    from dataclasses import replace

    import bench_rc_propagation as driver

    nominal = replace(
        driver.CONFIGS["nominal"], name="nominal_137ns", entry_phase_s=137e-9
    )
    return {
        "nominal": nominal,
        "detuned_20MHz": replace(nominal, name="detuned_20MHz", detuning_mhz=20.0),
        "optical_3Gamma": replace(
            nominal, name="optical_3Gamma", optical_rabi_over_gamma=3.0
        ),
    }


def labeled_initial_block(case: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the requested J=1,2,3 one-hot densities and one mixture."""
    labels = (
        "X_J1_F1_3o2_F1_mF0",
        "X_J2_F1_5o2_F2_mF0",
        "X_J3_F1_7o2_F3_mF0",
    )
    onehots = np.stack([case.rho0(label) for label in labels])
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    mixture = np.einsum("k,kij->ij", weights, onehots)
    return onehots, weights, mixture


__all__ = [
    "HamiltonianCSR",
    "ScalarAdjointObjective",
    "adjoint_rhs",
    "adjoint_rhs_from_hamiltonian",
    "authorized_case_configs",
    "block_rhs",
    "block_rhs_from_hamiltonian",
    "diagonal_mixture_response",
    "diagonal_objective_responses",
    "integrate_segmented_adjoint",
    "integrate_segmented_block",
    "j1_population_objective",
    "labeled_initial_block",
    "pack_density_block",
    "photon_objective",
    "population_projector_objective",
    "selected_population_objective",
    "unpack_density_block",
]
