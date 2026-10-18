"""Matrix-free Arnoldi action for repeated one-period RC propagation.

The module is intentionally independent of the RC model and solver.  A caller
supplies a real linear action for one complete period.  For density propagation
with a cumulative photon observable, augment the packed Hermitian density
vector with one scalar accumulator; the period callback maps ``(rho, y)`` to
``(M rho, y + q rho)``.  This turns both outputs into one linear action while
retaining every packed density coordinate, including coherences.

Finite-dimensional Arnoldi checkpoints are diagnostic approximations.  The
successive-checkpoint difference is not a rigorous error bound, especially for
nonnormal period maps or approximate/adaptive one-period callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

VectorAction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class KrylovCheckpoint:
    """Approximation at one requested Krylov dimension."""

    requested_dimension: int
    actual_dimension: int
    value: np.ndarray
    happy_breakdown: bool
    next_arnoldi_coefficient: float | None
    residual_indicator: float | None


@dataclass(frozen=True)
class ArnoldiPowerResult:
    """Results and diagnostics for a repeated matrix-free linear action."""

    cycles: int
    max_dimension: int
    action_count: int
    happy_breakdown: bool
    breakdown_dimension: int | None
    checkpoints: dict[int, KrylovCheckpoint]
    stopped_early: bool = False


def arnoldi_repeated_action(
    action: VectorAction,
    initial: np.ndarray,
    cycles: int,
    *,
    checkpoints: Iterable[int] = (40,),
    max_dimension: int = 40,
    breakdown_tolerance: float = 1e-13,
    stop_requested: Callable[[], bool] | None = None,
) -> ArnoldiPowerResult:
    """Approximate ``action**cycles(initial)`` with a real Arnoldi basis.

    ``action`` must be a deterministic, linear, real-vector action.  It may be
    implemented by integrating one complete physical period, but adaptively
    solved ODEs are only approximately linear; callers should run an explicit
    additivity/scaling probe and report its cost and discrepancy.

    The Krylov basis uses two-pass modified Gram-Schmidt for robustness on
    nonnormal actions.  At each requested dimension ``m``, the approximation
    is ``beta * V_m * H_m**cycles * e_1``.  All input/output coordinates are
    retained; for a packed Hermitian density, unpack the density component
    after this routine returns.

    The returned residual indicator is ``beta * |h[m,m-1]| *
    |(H_m**cycles e_1)[m-1]|``.  This is a useful diagnostic, not a certified
    global error bound for an integer power.
    """
    raw_vector = np.asarray(initial)
    if np.iscomplexobj(raw_vector):
        raise ValueError("initial must be real; pack complex Hermitian data into real coordinates")
    vector = np.asarray(raw_vector, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("initial must be a nonempty one-dimensional real vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("initial contains non-finite values")
    if isinstance(cycles, bool) or int(cycles) != cycles or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    cycles = int(cycles)
    if isinstance(max_dimension, bool) or int(max_dimension) != max_dimension or max_dimension <= 0:
        raise ValueError("max_dimension must be a positive integer")
    max_dimension = int(max_dimension)
    if not np.isfinite(breakdown_tolerance) or breakdown_tolerance < 0:
        raise ValueError("breakdown_tolerance must be finite and nonnegative")

    raw_checkpoints = tuple(checkpoints)
    if any(isinstance(value, bool | np.bool_) for value in raw_checkpoints):
        raise ValueError("checkpoint dimensions must be integers, not booleans")
    if any(int(value) != value for value in raw_checkpoints):
        raise ValueError("checkpoint dimensions must be integers")
    requested = sorted(set(int(value) for value in raw_checkpoints))
    if not requested or any(value <= 0 or value > max_dimension for value in requested):
        raise ValueError("checkpoints must be nonempty dimensions in [1, max_dimension]")
    beta = float(np.linalg.norm(vector))
    if beta == 0.0 or cycles == 0:
        outputs = {
            dim: KrylovCheckpoint(dim, 0, vector.copy(), True, 0.0, 0.0)
            for dim in requested
        }
        return ArnoldiPowerResult(cycles, max_dimension, 0, True, 0, outputs)

    basis = np.zeros((vector.size, max_dimension + 1), dtype=np.float64)
    hessenberg = np.zeros((max_dimension + 1, max_dimension), dtype=np.float64)
    basis[:, 0] = vector / beta
    action_count = 0
    actual_dimension = 0
    happy_breakdown = False
    stopped_early = False
    breakdown_dimension: int | None = None
    h_next: float | None = None

    for column in range(max_dimension):
        if stop_requested is not None and stop_requested():
            stopped_early = True
            break
        raw_candidate = np.asarray(action(basis[:, column]))
        action_count += 1
        if np.iscomplexobj(raw_candidate):
            raise ValueError("action must return a real vector")
        candidate = np.asarray(raw_candidate, dtype=np.float64)
        if candidate.shape != vector.shape:
            raise ValueError(
                f"action returned shape {candidate.shape}; expected {vector.shape}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError("action returned non-finite values")
        candidate_norm = float(np.linalg.norm(candidate))

        # Reorthogonalize: the second pass is important for near-invariant
        # subspaces and nonnormal maps where loss of orthogonality is common.
        for _ in range(2):
            for row in range(column + 1):
                coefficient = float(np.dot(basis[:, row], candidate))
                hessenberg[row, column] += coefficient
                candidate -= coefficient * basis[:, row]

        h_next = float(np.linalg.norm(candidate))
        hessenberg[column + 1, column] = h_next
        actual_dimension = column + 1
        if h_next <= breakdown_tolerance * max(candidate_norm, 1.0):
            happy_breakdown = True
            breakdown_dimension = actual_dimension
            break
        if column + 1 < max_dimension:
            basis[:, column + 1] = candidate / h_next

    completed_requested = requested
    if stopped_early and not happy_breakdown:
        completed_requested = tuple(q for q in requested if q <= actual_dimension)
    outputs: dict[int, KrylovCheckpoint] = {}
    for requested_dimension in completed_requested:
        dimension = min(requested_dimension, actual_dimension)
        projected = hessenberg[:dimension, :dimension]
        coordinates = np.zeros(dimension, dtype=np.float64)
        coordinates[0] = 1.0
        powered = np.linalg.matrix_power(projected, cycles) @ coordinates
        approximation = beta * (basis[:, :dimension] @ powered)
        coefficient = float(hessenberg[dimension, dimension - 1])
        checkpoint_breakdown = happy_breakdown and dimension == actual_dimension
        if checkpoint_breakdown:
            residual_indicator = 0.0
        else:
            residual_indicator = float(
                beta * abs(coefficient) * abs(powered[-1])
            )
        outputs[requested_dimension] = KrylovCheckpoint(
            requested_dimension=requested_dimension,
            actual_dimension=dimension,
            value=approximation,
            happy_breakdown=checkpoint_breakdown,
            next_arnoldi_coefficient=coefficient,
            residual_indicator=residual_indicator,
        )

    return ArnoldiPowerResult(
        cycles=cycles,
        max_dimension=max_dimension,
        action_count=action_count,
        happy_breakdown=happy_breakdown,
        breakdown_dimension=breakdown_dimension,
        checkpoints=outputs,
        stopped_early=stopped_early,
    )


__all__ = [
    "ArnoldiPowerResult",
    "KrylovCheckpoint",
    "arnoldi_repeated_action",
]
