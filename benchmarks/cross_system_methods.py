"""Benchmark-only adapters for eligible reusable dense and scalar adjoint paths.

These helpers do no work at import time and do not alter package defaults.
"""
from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from centrex_tlf import hamiltonian
from centrex_tlf.lindblad.dense import prepare_dense_lindblad_propagator


def prepare_static_dense(
    fixture: Any,
    *,
    execution_mode: str = "structured",
    reduce: bool = True,
    rho0_batch: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Prepare one reusable dense eigensystem; fixture construction is excluded."""
    if fixture.rc_case is not None:
        raise ValueError("dense_eig is limited to the static R2/R3 fixtures")
    initials = fixture.rho0[None, ...] if rho0_batch is None else np.asarray(rho0_batch)
    started = time.perf_counter()
    propagator = prepare_dense_lindblad_propagator(
        fixture.prepared, initials, execution_mode=execution_mode, reduce=reduce,
    )
    wall = time.perf_counter() - started
    stats = dict(propagator.stats)
    stats.update({
        "setup_wall_seconds": wall,
        "factorization_seconds": float(propagator.stats.get("factorization_seconds", 0.0)),
        "extraction_and_other_setup_seconds": max(
            0.0, wall - float(propagator.stats.get("factorization_seconds", 0.0))
        ),
        "execution_mode": execution_mode,
        "reduce": bool(reduce),
        "initial_count_prepared": len(initials),
        "backend": "rust_static_dense_eig_cpu_factorization",
    })
    return propagator, stats


def evaluate_static_dense(
    fixture: Any,
    propagator: Any,
    product: str,
    *,
    rho0_batch: np.ndarray | None = None,
) -> tuple[np.ndarray | dict[str, np.ndarray], float]:
    """Evaluate a final product; full density unpack/reconstruction is timed."""
    initials = fixture.rho0[None, ...] if rho0_batch is None else np.asarray(rho0_batch)
    t0, t1 = fixture.spec.t_span
    kwargs: dict[str, Any] = {"t0": t0}
    if product == "full_and_integral":
        # Dense eig has no native joint observable output. Reuse one factorized
        # propagator, time both projections, and label this as two evaluations.
        started = time.perf_counter()
        full = evaluate_static_dense(fixture, propagator, "full", rho0_batch=initials)[0]
        photons = evaluate_static_dense(
            fixture, propagator, "photon_integral", rho0_batch=initials,
        )[0]
        return {"full": full, "photon_integral": photons}, time.perf_counter() - started
    if product == "full":
        kwargs["output"] = "full"
    elif product == "photon_integral":
        kwargs.update(
            output="photon_integral",
            integral_weights=[(i, float(hamiltonian.Γ)) for i in fixture.excited_indices],
        )
    elif product == "populations":
        kwargs["output"] = "populations"
    elif product == "selected_1":
        i = fixture.selected_indices_1[0]
        kwargs.update(output="selected", output_indices=[(i, i)])
    elif product == "selected_4":
        kwargs.update(output="selected", output_indices=[(i, i) for i in fixture.selected_indices_4])
    elif product == "group_sum":
        kwargs.update(output="selected", output_indices=[(i, i) for i in fixture.group_indices])
    else:
        raise ValueError(f"unsupported dense product {product!r}")
    started = time.perf_counter()
    values = propagator.evaluate(initials, [t1], **kwargs)
    if product == "full":
        packed = np.asarray(values[:, -1, :], dtype=np.float64)
        output = np.stack([propagator.layout.unpack(row) for row in packed])
    else:
        output = np.asarray(values[:, -1, :])
        if product == "group_sum":
            output = np.sum(output, axis=-1, keepdims=True)
    return output, time.perf_counter() - started


def _objective_matrices(fixture: Any, product: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    n = len(fixture.system.QN)
    zero = np.zeros((n, n), dtype=np.complex128)
    objectives: list[tuple[str, np.ndarray, np.ndarray]] = []
    if product == "photon_integral":
        weight = np.zeros_like(zero)
        weight[np.asarray(fixture.excited_indices, dtype=int),
               np.asarray(fixture.excited_indices, dtype=int)] = float(hamiltonian.Γ)
        objectives.append(("photon_integral", weight, zero))
    elif product == "selected_1":
        for index in fixture.selected_indices_1:
            terminal = zero.copy()
            terminal[index, index] = 1.0
            objectives.append((f"population_{index}", zero, terminal))
    elif product == "selected_4":
        for index in fixture.selected_indices_4:
            terminal = zero.copy()
            terminal[index, index] = 1.0
            objectives.append((f"population_{index}", zero, terminal))
    elif product == "group_sum":
        terminal = zero.copy()
        index = np.asarray(fixture.group_indices, dtype=int)
        terminal[index, index] = 1.0
        objectives.append((fixture.group_label, zero, terminal))
    elif product in {"full", "populations"}:
        raise NotImplementedError(
            f"scalar adjoints do not return {product}; populations require one adjoint per state"
        )
    else:
        raise ValueError(f"unsupported adjoint product {product!r}")
    return objectives


def _adjoint_value(lambda0: np.ndarray, initial: np.ndarray) -> float:
    return float(np.real(np.trace(lambda0 @ initial)))


def evaluate_static_adjoint(
    fixture: Any,
    product: str,
    *,
    rho0_batch: np.ndarray | None = None,
    solver: str = "DOP853",
    reltol: float = 1e-8,
    atol: float = 1e-10,
    dt: float = 2e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate scalar adjoint objectives for static R2/R3 fixtures.

    The backward variable is tau=T-t, while Hamiltonian coefficients are
    evaluated at the absolute physical time T-t.  Each distinct requested
    observable is one adjoint solve; the density is never reconstructed.
    """
    if fixture.rc_case is not None:
        raise ValueError("use evaluate_rc_adjoint for the switched RC fixture")
    try:
        prototype = importlib.import_module("rc_block_adjoint_prototypes")
    except ModuleNotFoundError:
        prototype = importlib.import_module("benchmarks.rc_block_adjoint_prototypes")

    initials = fixture.rho0[None, ...] if rho0_batch is None else np.asarray(rho0_batch, dtype=np.complex128)
    if initials.ndim == 2:
        initials = initials[None, ...]
    n = len(fixture.system.QN)
    if initials.ndim != 3 or initials.shape[1:] != (n, n):
        raise ValueError(f"rho0_batch must have shape (K, {n}, {n})")
    t0, t1 = map(float, fixture.spec.t_span)
    if t1 <= t0:
        raise ValueError("fixture time span must increase")
    warm_started = time.perf_counter()
    prototype.warm_numba_kernels()
    warm_seconds = time.perf_counter() - warm_started
    context_started = time.perf_counter()
    context = prototype.PreparedRHSContext.from_prepared(fixture.prepared)
    context_seconds = time.perf_counter() - context_started
    objectives = _objective_matrices(fixture, product)
    started = time.perf_counter()
    answers = np.empty((len(initials), len(objectives)), dtype=np.float64)
    solve_stats = []
    duration = t1 - t0
    for objective_index, (name, running_weight, terminal) in enumerate(objectives):
        y0 = prototype.pack_density_block(terminal[None, ...], fixture.prepared.layout)[0]

        def rhs(tau: float, packed_lambda: np.ndarray, weight: np.ndarray = running_weight) -> np.ndarray:
            physical_time = t1 - float(tau)
            # Prototype RHS is lambda'_t=-L†lambda-W; changing to tau negates it.
            return -context.rhs_adjoint(packed_lambda, physical_time, weight)

        solution = solve_ivp(
            rhs, (0.0, duration), y0, method=solver, rtol=reltol, atol=atol,
            first_step=min(float(dt), duration), t_eval=[duration],
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        packed_lambda = np.asarray(solution.y[:, -1], dtype=np.float64)
        lambda0 = fixture.prepared.layout.unpack(packed_lambda)
        for initial_index, initial in enumerate(initials):
            answers[initial_index, objective_index] = _adjoint_value(lambda0, initial)
        solve_stats.append({"objective": name, "nfev": int(solution.nfev),
                            "status": int(solution.status), "message": solution.message})
    wall = time.perf_counter() - started
    stats = {
        "backend": "python_scipy_DOP853_scalar_adjoint_using_benchmark_rhs_context",
        "wall_seconds": wall, "warmup_seconds": warm_seconds,
        "rhs_context_setup_seconds": context_seconds,
        "solver": solver, "rtol": reltol, "atol": atol, "initial_dt_s": dt,
        "objective_count": len(objectives), "backward_solve_count": len(objectives),
        "initial_count": len(initials), "full_density_supported": False,
        "observable_solve_stats": solve_stats,
    }
    return answers[:, 0] if len(objectives) == 1 else answers, stats


def evaluate_rc_adjoint(
    fixture: Any,
    product: str,
    *,
    rho0_batch: np.ndarray | None = None,
    solver: str = "DOP853",
    reltol: float = 1e-8,
    atol: float = 1e-10,
    dt: float = 2e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the established exact-switch segmented scalar-adjoint prototype."""
    if fixture.rc_case is None:
        raise ValueError("evaluate_rc_adjoint requires the switched RC fixture")
    try:
        proto = importlib.import_module("rc_block_adjoint_prototypes")
    except ModuleNotFoundError:
        proto = importlib.import_module("benchmarks.rc_block_adjoint_prototypes")
    case = fixture.rc_case
    initial_batch = fixture.rho0[None, ...] if rho0_batch is None else np.asarray(rho0_batch)
    if initial_batch.ndim == 2:
        initial_batch = initial_batch[None, ...]
    if product == "photon_integral":
        objectives = [proto.photon_objective(case, float(hamiltonian.Γ))]
    elif product == "selected_1":
        objectives = [proto.selected_population_objective(case, i)
                      for i in fixture.selected_indices_1]
    elif product == "selected_4":
        objectives = [proto.selected_population_objective(case, i)
                      for i in fixture.selected_indices_4]
    elif product == "group_sum":
        objectives = [proto.population_projector_objective(
            case, fixture.group_indices, name=fixture.group_label,
        )]
    elif product in {"full", "populations"}:
        raise NotImplementedError(f"scalar RC adjoints do not return {product}")
    else:
        raise ValueError(f"unsupported RC adjoint product {product!r}")
    answers = np.empty((len(initial_batch), len(objectives)), dtype=np.float64)
    rows = []
    end_to_end_started = time.perf_counter()
    for j, objective in enumerate(objectives):
        lambda0, stats = proto.integrate_segmented_adjoint(
            case, objective, fixture.spec.t_span, solver=solver,
            reltol=reltol, abstol=atol, dt=dt,
        )
        for i, initial in enumerate(initial_batch):
            answers[i, j] = _adjoint_value(lambda0, initial)
        rows.append(stats)
    end_to_end_wall = time.perf_counter() - end_to_end_started
    return (answers[:, 0] if len(objectives) == 1 else answers), {
        "backend": "python_numba_scipy_segmented_scalar_adjoint",
        "wall_seconds": end_to_end_wall,
        "adjoint_integration_seconds": float(sum(row["wall_seconds"] for row in rows)),
        "setup_seconds": float(sum(row.get("numba_warmup_seconds", 0.0)
                                    + row.get("switch_plan_prep_seconds", 0.0)
                                    + row.get("rhs_context_prep_seconds", 0.0) for row in rows)),
        "objective_count": len(objectives), "backward_solve_count": len(objectives),
        "initial_count": len(initial_batch), "full_density_supported": False,
        "observable_solve_stats": rows,
    }
