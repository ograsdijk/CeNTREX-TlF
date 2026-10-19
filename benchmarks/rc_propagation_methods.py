"""Reusable experimental propagators for the RC benchmark.

This module keeps numerical-method prototypes out of the library API.  It
contains three pieces: a phase-frame ODE builder, a Strang split using four
static polarization generators, and a memory/cost-gated explicit periodic map.
The benchmark driver is responsible for building the physical case and its
frozen-switch prepared problems.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import scipy.sparse as sp
import sympy as smp
from scipy.sparse.linalg import expm_multiply

from centrex_tlf import states
from centrex_tlf.lindblad.dense import DenseLindbladPropagator, prepare_dense_lindblad_propagator
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import PreparedLindbladProblem, prepare_lindblad_problem
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout


@dataclass(frozen=True)
class EOMPhase:
    """One common optical phase modulation and its rotating-frame convention."""

    beta: float
    omega_rad_s: float
    excited_indices: tuple[int, ...]
    coupling_symbols: tuple[str, ...]
    diagonal_sign: int

    def phi(self, time_s: float) -> float:
        return self.beta * float(np.sin(self.omega_rad_s * time_s))


def _find_phase_calls(expr: smp.Expr) -> list[tuple[smp.Expr, smp.Expr, smp.Expr]]:
    calls: list[tuple[smp.Expr, smp.Expr, smp.Expr]] = []
    for node in smp.preorder_traversal(expr):
        if isinstance(node, smp.Function) and node.func.__name__ == "phase_modulation":
            if len(node.args) != 3:
                raise ValueError("phase_modulation must have arguments (t, beta, omega)")
            calls.append((node.args[0], node.args[1], node.args[2]))
    return calls


def infer_eom_phase(system: Any, parameters: LindbladParameters) -> EOMPhase:
    """Read the EOM phase and its sign from the actual optical coupling setup."""
    phase_bindings: list[tuple[str, smp.Expr, smp.Expr]] = []
    for name, expr in parameters.parsed_compound_parameters.items():
        calls = _find_phase_calls(expr)
        for _t, beta_expr, omega_expr in calls:
            phase_bindings.append((name, beta_expr, omega_expr))
    if not phase_bindings:
        raise ValueError("runtime parameters contain no phase_modulation binding")

    signatures = {(str(beta), str(omega)) for _, beta, omega in phase_bindings}
    if len(signatures) != 1:
        raise ValueError("one common excited-manifold phase is required by this helper")
    beta_expr, omega_expr = phase_bindings[0][1:]
    beta = _real_parameter_value(parameters, beta_expr)
    omega = _real_parameter_value(parameters, omega_expr)
    if omega <= 0.0:
        raise ValueError("EOM angular frequency must be positive")

    coupling_names = tuple(sorted({name for name, _, _ in phase_bindings}))
    symbol_names = tuple(str(symbol) for symbol in system.coupling_symbols)
    if not set(coupling_names).issubset(symbol_names):
        raise ValueError("phase-modulated parameter does not bind an OBESystem coupling symbol")

    # The frame transform removes one common phase from every X-B drive.  It
    # cannot represent an additional, unmodulated optical coupling (or a
    # second independently modulated field), so reject such systems explicitly.
    phase_names = set(coupling_names)
    coupling_symbol_names = set(symbol_names)
    for row in range(system.H_symbolic.rows):
        row_electronic = system.QN[row].largest.electronic_state
        for col in range(row + 1, system.H_symbolic.cols):
            col_electronic = system.QN[col].largest.electronic_state
            if {row_electronic, col_electronic} != {
                states.ElectronicState.X,
                states.ElectronicState.B,
            }:
                continue
            entry_couplings = {
                str(symbol)
                for symbol in system.H_symbolic[row, col].free_symbols
                if str(symbol) in coupling_symbol_names
            }
            if entry_couplings - phase_names:
                raise ValueError(
                    "phase-frame helper requires every X-B optical coupling to share "
                    "the same phase modulation"
                )

    signs: set[int] = set()
    for coupling_name in coupling_names:
        coupling_index = symbol_names.index(coupling_name)
        coupling = system.couplings[coupling_index]
        ground_index = _state_index(system.QN, coupling.ground_main)
        excited_index = _state_index(system.QN, coupling.excited_main)
        # H[g,e] receives Ω exp(+iφ) when g<e, so D_e=exp(-iφ).
        # If the excited state is the upper-triangle row, the sign reverses.
        signs.add(-1 if ground_index < excited_index else 1)
    if len(signs) != 1:
        raise ValueError("phase-modulated optical couplings have inconsistent basis orientations")
    excited = tuple(
        idx for idx, qn in enumerate(system.QN)
        if qn.largest.electronic_state == states.ElectronicState.B
    )
    if not excited:
        raise ValueError("OBESystem has no excited B-state levels")
    phase = EOMPhase(beta, omega, excited, coupling_names, signs.pop())
    validate_jump_frame(system.C_array, phase)
    return phase


def _state_index(qn: Sequence[Any], target: Any) -> int:
    try:
        return next(i for i, state in enumerate(qn) if state is target or state == target)
    except StopIteration as exc:
        raise ValueError("a coupling main state is absent from the OBE basis") from exc


def _real_parameter_value(parameters: LindbladParameters, expr: smp.Expr) -> float:
    if expr.is_Number:
        return float(expr)
    name = str(expr)
    if name not in parameters.base_parameters:
        raise ValueError(f"phase parameter {name!r} must be a scalar base parameter")
    value = parameters.base_parameters[name]
    if isinstance(value, tuple) or abs(complex(value).imag) > 0.0:
        raise ValueError(f"phase parameter {name!r} must be real and scalar")
    return float(complex(value).real)


def validate_jump_frame(c_array: np.ndarray, phase: EOMPhase) -> None:
    """Require each jump to gain only one scalar phase under the frame change."""
    excited = set(phase.excited_indices)
    for jump_index, collapse in enumerate(np.asarray(c_array)):
        row, col = np.nonzero(np.abs(collapse) > 0.0)
        if not len(row):
            continue
        charges = {
            int(col_i in excited) - int(row_i in excited)
            for row_i, col_i in zip(row.tolist(), col.tolist(), strict=False)
        }
        if len(charges) > 1:
            raise ValueError(
                f"collapse operator {jump_index} does not acquire a common phase; "
                "the dissipator must be transformed explicitly"
            )


def common_reachable_coordinates(
    generators: Sequence[sp.spmatrix],
    initial_packed: np.ndarray,
    layout: PackedHermitianLayout,
    phase: EOMPhase,
) -> tuple[np.ndarray, np.ndarray]:
    """Return common structural reachability and coordinates safe as sinks.

    The first result is the exact union of coordinates reachable from any
    supplied initial state under *each* generator and under the phase-frame
    rotations.  The second result contains coordinates with no outgoing edge
    under any generator or phase rotation.  Callers may remove only these
    common sinks from their active block, and must retain their incoming feed
    if they need full-density reconstruction.

    Generator columns are treated as source coordinates and rows as targets,
    matching ``d rho / dt = L rho``.  Structural nonzeros are exact; no
    magnitude threshold is applied.  EOM phase rotations mix the real and
    imaginary coordinates of every coherence with exactly one B-state index,
    so either member of such a pair makes both reachable.
    """
    dim = layout.packed_len
    if not generators:
        raise ValueError("at least one generator is required")
    initial = np.asarray(initial_packed, dtype=float)
    if initial.ndim == 1:
        initial = initial[None, :]
    if initial.ndim != 2 or initial.shape[1] != dim:
        raise ValueError(f"initial_packed must have shape (batch, {dim})")
    if not len(initial) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_packed must be a nonempty finite batch")
    if not phase.excited_indices or any(
        index < 0 or index >= layout.n for index in phase.excited_indices
    ):
        raise ValueError("phase excited_indices must index states in the layout")

    adjacency: list[set[int]] = [set() for _ in range(dim)]
    generator_sink = np.ones(dim, dtype=bool)
    for index, raw in enumerate(generators):
        matrix = sp.csc_matrix(raw, dtype=float)
        if matrix.shape != (dim, dim):
            raise ValueError(f"generator {index} must have shape {(dim, dim)}")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"generator {index} must be finite")
        matrix.eliminate_zeros()
        for source in range(dim):
            targets = matrix.indices[matrix.indptr[source] : matrix.indptr[source + 1]]
            if targets.size:
                adjacency[source].update(int(target) for target in targets)
                generator_sink[source] = False

    excited = set(int(index) for index in phase.excited_indices)
    phase_pairs: list[tuple[int, int]] = []
    for i in range(layout.n):
        for j in range(i + 1, layout.n):
            if (i in excited) == (j in excited):
                continue
            real = layout.upper_real_index(i, j)
            imag = layout.upper_imag_index(i, j)
            adjacency[real].add(imag)
            adjacency[imag].add(real)
            phase_pairs.append((real, imag))

    reachable = np.any(initial != 0.0, axis=0)
    frontier = list(np.flatnonzero(reachable))
    while frontier:
        source = frontier.pop()
        for target in adjacency[source]:
            if not reachable[target]:
                reachable[target] = True
                frontier.append(target)

    # A coordinate coupled by the phase rotation is not a coordinate sink,
    # even when its generator column is empty.
    phase_coupled = np.zeros(dim, dtype=bool)
    for real, imag in phase_pairs:
        phase_coupled[real] = True
        phase_coupled[imag] = True
    common_sinks = np.flatnonzero(generator_sink & ~phase_coupled)
    return np.flatnonzero(reachable), common_sinks


def phase_frame_density(rho: np.ndarray, time_s: float, phase: EOMPhase) -> np.ndarray:
    """Convert an original-frame density matrix to the EOM phase frame."""
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density matrix must be square")
    diagonal = np.ones(matrix.shape[0], dtype=np.complex128)
    diagonal[list(phase.excited_indices)] = np.exp(1j * phase.diagonal_sign * phase.phi(time_s))
    # rho' = D† rho D, D=diag(exp(i*s*phi) on B).
    return diagonal.conj()[:, None] * matrix * diagonal[None, :]


def original_frame_density(rho_frame: np.ndarray, time_s: float, phase: EOMPhase) -> np.ndarray:
    """Convert a phase-frame density matrix back to the original frame."""
    matrix = np.asarray(rho_frame, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density matrix must be square")
    diagonal = np.ones(matrix.shape[0], dtype=np.complex128)
    diagonal[list(phase.excited_indices)] = np.exp(-1j * phase.diagonal_sign * phase.phi(time_s))
    return diagonal.conj()[:, None] * matrix * diagonal[None, :]


def phase_frame_problem(
    prepared: PreparedLindbladProblem,
) -> tuple[PreparedLindbladProblem, EOMPhase]:
    """Prepare the exactly equivalent phase-frame ODE problem.

    The runtime optical phase is removed from each affected Ω binding.  The
    Hamiltonian receives ``diagonal_sign * beta * omega*cos(omega*t)`` on the
    B manifold.  The sign is inferred from the main coupling's actual basis
    order, and collapse jumps are checked before the original C array is reused.
    """
    system = prepared.obe_system
    parameters = prepared.parameters
    phase = infer_eom_phase(system, parameters)
    transformed = copy.copy(system)
    hamiltonian = smp.Matrix(system.H_symbolic)
    t = smp.Symbol("t", real=True)
    shift = phase.diagonal_sign * phase.beta * phase.omega_rad_s * smp.cos(phase.omega_rad_s * t)
    for idx in phase.excited_indices:
        hamiltonian[idx, idx] += shift
    transformed.H_symbolic = hamiltonian
    transformed.system = None
    transformed.dissipator = None

    compound = dict(parameters.compound_parameters)
    for name in phase.coupling_symbols:
        expression = parameters.parsed_compound_parameters.get(name)
        if expression is None:
            continue
        stripped = expression.replace(
            lambda node: isinstance(node, smp.Function)
            and node.func.__name__ == "phase_modulation",
            lambda _node: smp.Integer(1),
        )
        compound[name] = str(smp.simplify(stripped))
    phase_parameters = LindbladParameters(dict(parameters.base_parameters), compound)
    return (
        prepare_lindblad_problem(
            transformed,
            phase_parameters,
            backend=prepared.backend,
            hamiltonian_representation="decomposed",
        ),
        phase,
    )


def _apply_phase_kick(rho: np.ndarray, phase: EOMPhase, delta_phi: float) -> np.ndarray:
    diagonal = np.ones(rho.shape[0], dtype=np.complex128)
    diagonal[list(phase.excited_indices)] = np.exp(
        -1j * phase.diagonal_sign * float(delta_phi)
    )
    return diagonal[:, None] * rho * diagonal.conj()[None, :]


@dataclass
class StaticSplit:
    """Four static Liouvillian factors with exact EOM phase kicks."""

    layout: PackedHermitianLayout
    factors: Mapping[tuple[int, int], Any]
    phase: EOMPhase
    factor_bytes_estimate: int
    factor_setup: Mapping[tuple[int, int], Mapping[str, float | int]]

    def propagate(
        self,
        rho0: np.ndarray,
        t_span: tuple[float, float],
        *,
        switch_values: Callable[[float], tuple[int, int]],
        switch_boundaries: Sequence[float],
        dt: float,
        return_photons: bool = False,
        integral_weights: Sequence[tuple[int, float]] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """Strang-split propagation, with every step ending at a switch boundary."""
        t0, t1 = map(float, t_span)
        if not np.isfinite(dt) or dt <= 0.0 or t1 < t0:
            raise ValueError("dt must be positive and t_span increasing")
        if return_photons and integral_weights is None:
            raise ValueError("integral_weights are required when return_photons=True")
        if integral_weights is not None and not return_photons:
            raise ValueError("set return_photons=True to request an integrated photon yield")
        rho = np.asarray(rho0, dtype=np.complex128).copy()
        if rho.shape != (self.layout.n, self.layout.n):
            raise ValueError("rho0 has the wrong density-matrix shape")
        rho = phase_frame_density(rho, t0, self.phase)
        integral = 0.0
        boundaries = np.asarray(switch_boundaries, dtype=float)
        if boundaries.ndim != 1 or np.any(~np.isfinite(boundaries)):
            raise ValueError("switch_boundaries must be a finite vector")
        if np.any(np.diff(boundaries) <= 0.0):
            raise ValueError("switch_boundaries must be strictly increasing")
        cuts = np.unique(np.r_[t0, boundaries[(boundaries > t0) & (boundaries < t1)], t1])
        for left, right in pairwise(cuts):
            count = max(1, int(np.ceil((right - left) / dt)))
            edges = np.linspace(left, right, count + 1)
            for start, stop in pairwise(edges):
                mid = 0.5 * (start + stop)
                key = tuple(int(v) for v in switch_values(float(mid)))
                if key not in self.factors:
                    raise KeyError(f"no static factor for switch state {key}")
                halfway = 0.5 * (start + stop)
                first = self.phase.phi(halfway) - self.phase.phi(float(start))
                second = self.phase.phi(float(stop)) - self.phase.phi(halfway)
                rho = _apply_phase_kick(rho, self.phase, first)
                if integral_weights is not None:
                    contribution = self.factors[key].evaluate(
                        rho[None, ...],
                        [float(stop - start)],
                        output="photon_integral",
                        integral_weights=integral_weights,
                    )
                    integral += float(np.asarray(contribution)[0, 0, 0].real)
                rho = self.factors[key].density_matrices(rho[None, ...], [float(stop - start)])[0, 0]
                rho = _apply_phase_kick(rho, self.phase, second)
        result = original_frame_density(rho, t1, self.phase)
        return (result, integral) if return_photons else result


@dataclass
class SparseExpmFactor:
    """Reusable sparse Liouvillian with per-interval exponential action."""

    layout: PackedHermitianLayout
    generator: sp.csr_matrix
    matrix_bytes: int
    keep_coordinates: np.ndarray | None = None

    def __post_init__(self) -> None:
        full_dim = self.layout.packed_len
        matrix = sp.csr_matrix(self.generator, dtype=float)
        if matrix.shape != (full_dim, full_dim):
            raise ValueError(f"generator must have full shape {(full_dim, full_dim)}")
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("generator must be finite")
        if self.keep_coordinates is None:
            self.keep_coordinates = np.arange(full_dim, dtype=int)
        else:
            keep = np.asarray(self.keep_coordinates, dtype=int)
            if keep.ndim != 1 or not len(keep):
                raise ValueError("keep_coordinates must be a nonempty index vector")
            if np.any(keep < 0) or np.any(keep >= full_dim) or len(np.unique(keep)) != len(keep):
                raise ValueError("keep_coordinates must contain unique packed indices in range")
            keep = np.sort(keep)
            excluded = np.setdiff1d(np.arange(full_dim), keep, assume_unique=True)
            if len(excluded) and matrix[excluded, :][:, keep].nnz:
                raise ValueError("keep_coordinates are not closed under generator action")
            self.keep_coordinates = keep
            matrix = matrix[keep, :][:, keep].tocsr()
        self.generator = matrix
        self.matrix_bytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)

    @property
    def excluded_coordinates(self) -> np.ndarray:
        return np.setdiff1d(
            np.arange(self.layout.packed_len), self.keep_coordinates, assume_unique=True
        )

    def _project_inputs(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 3 or batch.shape[1:] != (self.layout.n, self.layout.n):
            raise ValueError("rho_batch must have shape (batch,n,n)")
        packed = np.stack([self.layout.pack(rho) for rho in batch])
        excluded = self.excluded_coordinates
        if len(excluded) and np.any(packed[:, excluded] != 0.0):
            raise ValueError("initial density has nonzero support outside keep_coordinates")
        return packed[:, self.keep_coordinates]

    def _restore_outputs(self, packed_kept: np.ndarray) -> np.ndarray:
        packed_full = np.zeros((len(packed_kept), self.layout.packed_len), dtype=float)
        packed_full[:, self.keep_coordinates] = packed_kept
        return np.stack([self.layout.unpack(row) for row in packed_full])

    def density_matrices(self, rho_batch: np.ndarray, times: Sequence[float]) -> np.ndarray:
        if len(times) != 1 or times[0] < 0.0:
            raise ValueError("sparse split factor accepts one nonnegative interval endpoint")
        duration = float(times[0])
        packed = self._project_inputs(np.asarray(rho_batch, dtype=np.complex128))
        evolved = np.stack([expm_multiply(self.generator * duration, row) for row in packed])
        return self._restore_outputs(evolved)[:, None, :, :]

    def evaluate(
        self,
        rho_batch: np.ndarray,
        times: Sequence[float],
        *,
        output: str,
        integral_weights: Sequence[tuple[int, float]],
    ) -> np.ndarray:
        if output != "photon_integral" or len(times) != 1 or times[0] < 0.0:
            raise ValueError("sparse split factor supports one photon-integral endpoint")
        duration = float(times[0])
        weights_full = np.zeros(self.layout.packed_len, dtype=float)
        for index, value in integral_weights:
            weights_full[self.layout.diagonal_index(int(index))] += float(value)
        weights = weights_full[self.keep_coordinates]
        augmented = sp.bmat(
            [[self.generator, sp.csr_matrix((len(self.keep_coordinates), 1))],
             [sp.csr_matrix(weights[None, :]), sp.csr_matrix((1, 1))]],
            format="csr",
        )
        packed = self._project_inputs(np.asarray(rho_batch, dtype=np.complex128))
        values = []
        for row in packed:
            initial = np.r_[row, 0.0]
            values.append(float(expm_multiply(augmented * duration, initial)[-1]))
        return np.asarray(values, dtype=float)[:, None, None]


def prepare_static_factor(
    prepared: PreparedLindbladProblem,
    *,
    method: str = "dense",
    parameter_values: Mapping[str, complex] | None = None,
    execution_mode: str = "expanded_sparse",
    keep_coordinates: np.ndarray | None = None,
) -> tuple[DenseLindbladPropagator | SparseExpmFactor, dict[str, float | int]]:
    """Prepare one reusable static factor and return its setup timing/storage."""
    start = time.perf_counter()
    if method == "dense":
        if keep_coordinates is not None:
            raise ValueError("keep_coordinates reduction is currently supported only for sparse_expm")
        factor = prepare_dense_lindblad_propagator(
            prepared,
            parameter_values=parameter_values,
            execution_mode=execution_mode,
            reduce=False,
        )
        elapsed = time.perf_counter() - start
        estimated = estimate_dense_factor_bytes(prepared.layout.n)
        return factor, {"setup_seconds": elapsed, "estimated_bytes": estimated, "nnz": -1}
    if method == "sparse_expm":
        if prepared.rust_plan is None:
            raise ValueError("sparse exponential action requires a Rust-prepared problem")
        from centrex_tlf.lindblad.dense import _StaticLiouvillian

        extractor = _StaticLiouvillian(prepared.to_payload(), execution_mode, prepared.rust_plan)
        generator = extractor.matrix(parameter_values or {})
        factor = SparseExpmFactor(prepared.layout, generator, 0, keep_coordinates)
        elapsed = time.perf_counter() - start
        nbytes = factor.matrix_bytes
        return factor, {
            "setup_seconds": elapsed,
            "estimated_bytes": nbytes,
            "nnz": int(factor.generator.nnz),
        }
    raise ValueError("method must be 'dense' or 'sparse_expm'")


def estimate_dense_factor_bytes(n_states: int, *, bytes_per_matrix_entry: int = 64) -> int:
    """Conservative factorization workspace estimate for a packed OBE generator."""
    packed_dim = int(n_states) * int(n_states)
    return int(bytes_per_matrix_entry) * packed_dim * packed_dim


def prepare_static_split(
    prepared_by_switch_state: Mapping[tuple[int, int], PreparedLindbladProblem],
    phase: EOMPhase,
    *,
    max_total_bytes: int = 8 * 1024**3,
    bytes_per_matrix_entry: int = 64,
    method: str = "dense",
) -> StaticSplit:
    """Factor four static generators after a conservative all-factor memory gate.

    All factors use ``reduce=False`` so they accept coherences created by any
    preceding switch state.  The memory check runs before the first factorization.
    """
    if len(prepared_by_switch_state) != 4:
        raise ValueError("exactly four frozen switch-state generators are required")
    first = next(iter(prepared_by_switch_state.values()))
    n = first.layout.n
    reference_qn = tuple(first.obe_system.QN)
    reference_c = np.asarray(first.obe_system.C_array)
    for item in prepared_by_switch_state.values():
        if item.layout.n != n or len(item.obe_system.QN) != n:
            raise ValueError("all switch states must share one density-matrix basis")
        if any(left != right for left, right in zip(reference_qn, item.obe_system.QN, strict=False)):
            raise ValueError("switch-state generators use different QN ordering")
        if not np.array_equal(reference_c, np.asarray(item.obe_system.C_array)):
            raise ValueError("switch-state generators use different collapse operators")
    if method == "dense":
        per_factor = estimate_dense_factor_bytes(n, bytes_per_matrix_entry=bytes_per_matrix_entry)
        total = 4 * per_factor
        if total > int(max_total_bytes):
            raise MemoryError(
                f"dense split rejected before factorization: estimated workspace {total} bytes "
                f"for four generators exceeds max_total_bytes={int(max_total_bytes)}"
            )
    elif method == "sparse_expm":
        total = 0
    else:
        raise ValueError("method must be 'dense' or 'sparse_expm'")
    factors: dict[tuple[int, int], Any] = {}
    setup: dict[tuple[int, int], Mapping[str, float | int]] = {}
    measured_bytes = 0
    for key, prepared in prepared_by_switch_state.items():
        factor, metadata = prepare_static_factor(prepared, method=method)
        factors[key] = factor
        setup[key] = metadata
        measured_bytes += int(metadata["estimated_bytes"])
    return StaticSplit(first.layout, factors, phase, measured_bytes, setup)


@dataclass(frozen=True)
class PeriodMapGate:
    feasible: bool
    packed_dimension: int
    estimated_bytes: int
    basis_actions: int
    reason: str


@dataclass
class PeriodicMap:
    """Explicit map on packed densities for periods starting at ``entry_time_s``.

    The map is aligned to the modulation phase at this canonical origin and
    applies over ``[entry_time_s, entry_time_s + period_s]``.  A trajectory
    entering at another phase needs a prefix/suffix or a separately built map.
    """

    layout: PackedHermitianLayout
    period_s: float
    matrix: np.ndarray
    integral_row: np.ndarray | None = None
    entry_time_s: float = 0.0

    def apply(self, rho: np.ndarray) -> np.ndarray:
        array = np.asarray(rho, dtype=np.complex128)
        if array.ndim == 2:
            return self.layout.unpack(self.matrix @ self.layout.pack(array))
        if array.ndim != 3 or array.shape[1:] != (self.layout.n, self.layout.n):
            raise ValueError("rho must be one density matrix or a batch of density matrices")
        packed = np.stack([self.layout.pack(item) for item in array])
        evolved = packed @ self.matrix.T
        return np.stack([self.layout.unpack(item) for item in evolved])

    def repeat(
        self, rho: np.ndarray, cycles: int, *, return_integral: bool = False
    ) -> np.ndarray | tuple[np.ndarray, float | np.ndarray]:
        if cycles < 0:
            raise ValueError("cycles must be nonnegative")
        array = np.asarray(rho, dtype=np.complex128)
        batched = array.ndim == 3
        if array.ndim == 2:
            packed = self.layout.pack(array)[None, :]
        elif batched and array.shape[1:] == (self.layout.n, self.layout.n):
            packed = np.stack([self.layout.pack(item) for item in array])
        else:
            raise ValueError("rho must be one density matrix or a batch of density matrices")
        if return_integral and self.integral_row is None:
            raise ValueError("this periodic map was built without an integral observable")
        totals = np.zeros(len(packed), dtype=float)
        for _ in range(int(cycles)):
            if return_integral and self.integral_row is not None:
                totals += packed @ self.integral_row
            packed = packed @ self.matrix.T
        densities = np.stack([self.layout.unpack(item) for item in packed])
        result_density = densities if batched else densities[0]
        if not return_integral:
            return result_density
        return result_density, (totals if batched else float(totals[0]))


def periodic_map_gate(
    layout: PackedHermitianLayout,
    *,
    max_bytes: int = 8 * 1024**3,
    max_basis_actions: int = 5000,
    estimated_action_seconds: float | None = None,
    max_build_seconds: float | None = None,
) -> PeriodMapGate:
    """Check explicit map memory and O(n²) basis-action cost before allocation."""
    dim = layout.packed_len
    estimated = 8 * dim * dim
    actions = dim
    feasible = estimated <= int(max_bytes) and actions <= int(max_basis_actions)
    projected_seconds = None
    if estimated_action_seconds is not None:
        if not np.isfinite(estimated_action_seconds) or estimated_action_seconds < 0.0:
            raise ValueError("estimated_action_seconds must be finite and nonnegative")
        projected_seconds = estimated_action_seconds * actions
        if max_build_seconds is not None:
            if not np.isfinite(max_build_seconds) or max_build_seconds <= 0.0:
                raise ValueError("max_build_seconds must be positive and finite")
            feasible = feasible and projected_seconds <= max_build_seconds
    if feasible:
        reason = "within configured memory and basis-action gates"
    elif estimated > int(max_bytes):
        reason = f"map needs {estimated} bytes, above max_bytes={int(max_bytes)}"
    elif projected_seconds is not None and max_build_seconds is not None and projected_seconds > max_build_seconds:
        reason = (
            f"estimated construction time {projected_seconds:.3g}s exceeds "
            f"max_build_seconds={max_build_seconds:.3g}s"
        )
    else:
        reason = f"map needs {actions} full-period basis actions, above max_basis_actions={int(max_basis_actions)}"
    return PeriodMapGate(feasible, dim, estimated, actions, reason)


def build_periodic_map(
    layout: PackedHermitianLayout,
    one_period_action: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, float]],
    period_s: float,
    *,
    max_bytes: int = 8 * 1024**3,
    max_basis_actions: int = 5000,
    estimated_action_seconds: float | None = None,
    max_build_seconds: float | None = None,
    entry_time_s: float = 0.0,
) -> tuple[PeriodicMap | None, PeriodMapGate]:
    """Build a complete one-period map from a full-density action, if gated in.

    The callback must propagate the interval ``[entry_time_s, entry_time_s +
    period_s]`` and receives/returns an ``(n,n)`` Hermitian matrix.  This
    materializes every real packed-density basis column; callers must not pass
    an action that only supports population initial conditions.
    """
    if not np.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("period_s must be positive and finite")
    if not np.isfinite(entry_time_s):
        raise ValueError("entry_time_s must be finite")
    gate = periodic_map_gate(
        layout,
        max_bytes=max_bytes,
        max_basis_actions=max_basis_actions,
        estimated_action_seconds=estimated_action_seconds,
        max_build_seconds=max_build_seconds,
    )
    if not gate.feasible:
        return None, gate
    matrix = np.empty((gate.packed_dimension, gate.packed_dimension), dtype=np.float64)
    integral_row: np.ndarray | None = None
    action_has_integral: bool | None = None
    for column in range(gate.packed_dimension):
        basis = np.zeros(gate.packed_dimension, dtype=np.float64)
        basis[column] = 1.0
        action_result = one_period_action(layout.unpack(basis))
        if isinstance(action_result, tuple):
            if action_has_integral is False:
                raise ValueError("period action must return an observable for every basis state")
            action_has_integral = True
            if len(action_result) != 2:
                raise ValueError("period action tuple must be (rho_final, integrated_observable)")
            evolved_raw, observable = action_result
            if not np.isfinite(observable):
                raise ValueError("integrated observable must be finite")
            if integral_row is None:
                integral_row = np.empty(gate.packed_dimension, dtype=float)
            integral_row[column] = float(observable)
        else:
            if action_has_integral is True:
                raise ValueError("period action must return an observable for every basis state")
            action_has_integral = False
            evolved_raw = action_result
        evolved = np.asarray(evolved_raw, dtype=np.complex128)
        if evolved.shape != (layout.n, layout.n):
            raise ValueError("one_period_action returned a density matrix with the wrong shape")
        if not np.allclose(evolved, evolved.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError("one_period_action did not preserve Hermiticity")
        matrix[:, column] = layout.pack(evolved)
    if action_has_integral and integral_row is None:
        raise AssertionError("integral row construction did not complete")
    return PeriodicMap(layout, float(period_s), matrix, integral_row, float(entry_time_s)), gate


__all__ = [
    "EOMPhase",
    "PeriodMapGate",
    "PeriodicMap",
    "StaticSplit",
    "build_periodic_map",
    "common_reachable_coordinates",
    "estimate_dense_factor_bytes",
    "infer_eom_phase",
    "original_frame_density",
    "periodic_map_gate",
    "phase_frame_density",
    "phase_frame_problem",
    "prepare_static_split",
    "validate_jump_frame",
]
