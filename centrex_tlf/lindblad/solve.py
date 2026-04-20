from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse
from scipy.integrate import solve_ivp

from .plan_static import PreparedLindbladProblem, prepare_lindblad_problem
from .reference_dense import reference_rhs, structured_rhs

__all__ = [
    "LindbladResult",
    "LindbladMatrixResult",
    "prepare_lindblad_problem",
    "solve_lindblad",
]


@dataclass
class LindbladResult:
    t: npt.NDArray[np.float64]
    packed_y: npt.NDArray[np.float64]
    layout: Any

    def density_matrices(self) -> npt.NDArray[np.complex128]:
        matrices = np.empty((self.packed_y.shape[0], self.layout.n, self.layout.n), dtype=np.complex128)
        for idx, state in enumerate(self.packed_y):
            matrices[idx] = self.layout.unpack(state)
        return matrices

    def populations(self) -> npt.NDArray[np.float64]:
        matrices = self.density_matrices()
        return np.real(np.diagonal(matrices, axis1=1, axis2=2))


@dataclass
class LindbladMatrixResult:
    t: npt.NDArray[np.float64]
    matrix_y: npt.NDArray[np.complex128]
    layout: Any
    _packed_cache: npt.NDArray[np.float64] | None = None

    @property
    def packed_y(self) -> npt.NDArray[np.float64]:
        if self._packed_cache is None:
            self._packed_cache = _pack_matrix_snapshots(self.layout, self.matrix_y)
        return self._packed_cache

    def density_matrices(self) -> npt.NDArray[np.complex128]:
        return self.matrix_y

    def populations(self) -> npt.NDArray[np.float64]:
        return np.real(np.diagonal(self.matrix_y, axis1=1, axis2=2))


def _normalize_saveat(
    saveat: None | float | Sequence[float] | npt.NDArray[np.floating],
    t_span: tuple[float, float],
    save_start: bool,
) -> None | np.ndarray:
    if saveat is None:
        return None
    if isinstance(saveat, (float, int, np.floating, np.integer)):
        step = float(saveat)
        if step <= 0:
            raise ValueError("saveat step must be positive")
        values = np.arange(t_span[0], t_span[1] + 0.5 * step, step, dtype=np.float64)
    else:
        values = np.asarray(saveat, dtype=np.float64)
    if not save_start and values.size > 0 and np.isclose(values[0], t_span[0]):
        values = values[1:]
    return values


def _solve_python_reference(
    prepared: PreparedLindbladProblem,
    packed_rho0: np.ndarray,
    t_span: tuple[float, float],
    *,
    execution_mode: str,
    method: str,
    abstol: float,
    reltol: float,
    dt: float,
    saveat: None | np.ndarray,
    save_start: bool,
    maxiters: int,
) -> LindbladResult:
    rhs = reference_rhs if execution_mode == "reference" else structured_rhs
    solution = solve_ivp(
        lambda t, y: rhs(prepared, y, t),
        t_span=t_span,
        y0=packed_rho0,
        method=method,
        atol=abstol,
        rtol=reltol,
        first_step=dt,
        t_eval=saveat,
        max_step=np.inf,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    packed = np.asarray(solution.y.T, dtype=np.float64)
    times = np.asarray(solution.t, dtype=np.float64)
    if saveat is None and not save_start and times.size > 0 and np.isclose(times[0], t_span[0]):
        times = times[1:]
        packed = packed[1:]
    if packed.shape[0] > maxiters + 1:
        raise RuntimeError("python reference solver exceeded maxiters budget")
    return LindbladResult(t=times, packed_y=packed, layout=prepared.layout)


def _pack_matrix_snapshots(layout: Any, matrices: np.ndarray) -> np.ndarray:
    packed = np.empty((matrices.shape[0], layout.packed_len), dtype=np.float64)
    for idx, matrix in enumerate(matrices):
        packed[idx] = layout.pack(matrix)
    return packed


def _flatten_complex_matrix_state(rho: np.ndarray) -> np.ndarray:
    return np.asarray(rho, dtype=np.complex128).reshape(-1)


def _complex_to_split_real(flat_state: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat_state, dtype=np.complex128).reshape(-1)
    return np.concatenate((flat.real, flat.imag)).astype(np.float64, copy=False)


def _split_real_to_complex(split_state: np.ndarray) -> np.ndarray:
    split = np.asarray(split_state, dtype=np.float64).reshape(-1)
    if split.size % 2 != 0:
        raise ValueError("split-real state length must be even")
    n = split.size // 2
    return split[:n] + 1j * split[n:]


def _expression_uses_time(payload: dict[str, Any]) -> bool:
    return any(int(instr["op"]) == 4 for instr in payload["instructions"])


def _plan_is_time_dependent(prepared: PreparedLindbladProblem) -> bool:
    for compound in prepared.parameter_graph.get("compounds", []):
        if _expression_uses_time(compound["expression"]):
            return True
    hamiltonian_plan = prepared.hamiltonian_plan
    kind = hamiltonian_plan.get("kind", "entrywise")
    if kind == "decomposed":
        return any(
            _expression_uses_time(coefficient["expression"])
            for coefficient in hamiltonian_plan.get("coefficients", [])
        )
    for temp in hamiltonian_plan.get("temps", []):
        if _expression_uses_time(temp):
            return True
    return any(
        _expression_uses_time(entry["expression"])
        for entry in hamiltonian_plan.get("entries", [])
    )


def _solve_scipy_with_rust_matrix_rhs(
    prepared: PreparedLindbladProblem,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    *,
    execution_mode: str,
    abstol: float,
    reltol: float,
    dt: float,
    saveat: None | np.ndarray,
    save_start: bool,
    maxiters: int,
) -> LindbladMatrixResult:
    from ..centrex_tlf_rust import create_lindblad_rhs_evaluator_py

    if prepared.rust_plan is None:
        raise RuntimeError("rust plan is required for the Rust SciPy solver path")
    evaluator = create_lindblad_rhs_evaluator_py(
        prepared.rust_plan,
        execution_mode,
    )
    y0 = _flatten_complex_matrix_state(rho0)
    solution = solve_ivp(
        lambda t, y: np.asarray(evaluator.rhs_matrix_py(y, t), dtype=np.complex128),
        t_span=t_span,
        y0=y0,
        method="RK45",
        atol=abstol,
        rtol=reltol,
        first_step=dt,
        t_eval=saveat,
        max_step=np.inf,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    times = np.asarray(solution.t, dtype=np.float64)
    flat_states = np.asarray(solution.y.T, dtype=np.complex128)
    if saveat is None and not save_start and times.size > 0 and np.isclose(times[0], t_span[0]):
        times = times[1:]
        flat_states = flat_states[1:]
    if flat_states.shape[0] > maxiters + 1:
        raise RuntimeError("scipy rust solver exceeded maxiters budget")
    matrices = flat_states.reshape((-1, prepared.layout.n, prepared.layout.n))
    return LindbladMatrixResult(t=times, matrix_y=matrices, layout=prepared.layout)


def _solve_scipy_with_rust_packed_rhs(
    prepared: PreparedLindbladProblem,
    packed_rho0: np.ndarray,
    t_span: tuple[float, float],
    *,
    execution_mode: str,
    method: str,
    abstol: float,
    reltol: float,
    dt: float,
    saveat: None | np.ndarray,
    save_start: bool,
    maxiters: int,
    jacobian: str,
    jacobian_format: str,
) -> LindbladResult:
    from ..centrex_tlf_rust import create_lindblad_rhs_evaluator_py

    if prepared.rust_plan is None:
        raise RuntimeError("rust plan is required for the Rust SciPy solver path")
    evaluator = create_lindblad_rhs_evaluator_py(
        prepared.rust_plan,
        execution_mode,
    )
    y0 = np.asarray(packed_rho0, dtype=np.float64)
    is_time_dependent = _plan_is_time_dependent(prepared)
    jacobian_cache: scipy.sparse.csc_matrix | np.ndarray | None = None

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(
            evaluator.rhs_packed_py(np.ascontiguousarray(y, dtype=np.float64), t),
            dtype=np.float64,
        )

    def build_jacobian(t: float, _y: np.ndarray | None = None) -> scipy.sparse.csc_matrix | np.ndarray:
        nonlocal jacobian_cache
        if jacobian != "exact":
            raise RuntimeError("only jacobian='exact' is currently implemented")
        if jacobian_cache is not None and not is_time_dependent:
            return jacobian_cache
        rows, cols, values = evaluator.jacobian_packed_sparse_py(t)
        rows_arr = np.asarray(rows, dtype=np.int64)
        cols_arr = np.asarray(cols, dtype=np.int64)
        values_arr = np.asarray(values, dtype=np.float64)
        dim = y0.size
        matrix = scipy.sparse.csc_matrix((values_arr, (rows_arr, cols_arr)), shape=(dim, dim))
        if jacobian_format == "dense":
            dense = matrix.toarray()
            if not is_time_dependent:
                jacobian_cache = dense
            return dense
        if not is_time_dependent:
            jacobian_cache = matrix
        return matrix

    solution = solve_ivp(
        rhs,
        t_span=t_span,
        y0=y0,
        method=method,
        atol=abstol,
        rtol=reltol,
        first_step=dt,
        t_eval=saveat,
        max_step=np.inf,
        jac=build_jacobian if jacobian == "exact" else None,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    times = np.asarray(solution.t, dtype=np.float64)
    packed_states = np.asarray(solution.y.T, dtype=np.float64)
    if saveat is None and not save_start and times.size > 0 and np.isclose(times[0], t_span[0]):
        times = times[1:]
        packed_states = packed_states[1:]
    if packed_states.shape[0] > maxiters + 1:
        raise RuntimeError("scipy stiff rust solver exceeded maxiters budget")
    return LindbladResult(t=times, packed_y=packed_states, layout=prepared.layout)


def solve_lindblad(
    prepared_or_obe_system: PreparedLindbladProblem | Any,
    rho0: npt.NDArray[np.complex128],
    t_span: Sequence[float],
    *,
    parameters: Any | None = None,
    backend: str = "rust",
    solver: str = "explicit",
    abstol: float = 1e-7,
    reltol: float = 1e-4,
    dt: float = 1e-8,
    saveat: None | float | Sequence[float] | npt.NDArray[np.floating] = None,
    save_start: bool = True,
    maxiters: int = 100_000,
    execution_mode: str = "structured",
    jacobian: str = "exact",
    jacobian_format: str = "auto",
) -> LindbladResult | LindbladMatrixResult:
    if solver not in {"explicit", "dopri5", "bdf", "scipy", "scipy_bdf", "scipy_radau"}:
        raise NotImplementedError(
            "supported solvers are 'explicit'/'dopri5', 'bdf', 'scipy', 'scipy_bdf', and 'scipy_radau'"
        )
    if execution_mode not in {"reference", "structured", "structured_upper"}:
        raise NotImplementedError(
            "supported execution_mode values are 'reference', 'structured', and 'structured_upper'"
        )
    if len(t_span) != 2:
        raise ValueError("t_span must contain exactly two values")
    t_span_tuple = (float(t_span[0]), float(t_span[1]))
    if isinstance(prepared_or_obe_system, PreparedLindbladProblem):
        prepared = prepared_or_obe_system
    else:
        if parameters is None:
            raise TypeError("parameters are required when solving from an OBESystem")
        prepared = prepare_lindblad_problem(prepared_or_obe_system, parameters, backend=backend)
    rho0_array = np.asarray(rho0, dtype=np.complex128)
    packed_rho0 = prepared.layout.pack(rho0_array)
    saveat_values = _normalize_saveat(saveat, t_span_tuple, save_start)
    if backend == "rust" and solver == "scipy":
        return _solve_scipy_with_rust_matrix_rhs(
            prepared,
            rho0_array,
            t_span_tuple,
            execution_mode=execution_mode,
            abstol=abstol,
            reltol=reltol,
            dt=dt,
            saveat=saveat_values,
            save_start=save_start,
            maxiters=maxiters,
        )
    if backend == "rust" and solver in {"scipy_bdf", "scipy_radau"}:
        if jacobian not in {"exact", "none"}:
            raise NotImplementedError("jacobian must be 'exact' or 'none'")
        if jacobian_format not in {"auto", "sparse", "dense"}:
            raise NotImplementedError("jacobian_format must be 'auto', 'sparse', or 'dense'")
        chosen_format = "sparse" if jacobian_format == "auto" else jacobian_format
        return _solve_scipy_with_rust_packed_rhs(
            prepared,
            packed_rho0,
            t_span_tuple,
            execution_mode=execution_mode,
            method="BDF" if solver == "scipy_bdf" else "Radau",
            abstol=abstol,
            reltol=reltol,
            dt=dt,
            saveat=saveat_values,
            save_start=save_start,
            maxiters=maxiters,
            jacobian=jacobian,
            jacobian_format=chosen_format,
        )
    if backend == "rust" and solver == "bdf" and prepared.rust_plan is not None:
        from ..centrex_tlf_rust import solve_lindblad_bdf_py

        times, packed_states = solve_lindblad_bdf_py(
            prepared.rust_plan,
            packed_rho0,
            t_span_tuple[0],
            t_span_tuple[1],
            float(abstol),
            float(reltol),
            float(dt),
            None if saveat_values is None else np.asarray(saveat_values, dtype=np.float64),
            bool(save_start),
            int(maxiters),
            execution_mode,
        )
        return LindbladResult(
            t=np.asarray(times, dtype=np.float64),
            packed_y=np.asarray(packed_states, dtype=np.float64),
            layout=prepared.layout,
        )
    if backend == "rust" and prepared.rust_plan is not None:
        from ..centrex_tlf_rust import solve_lindblad_dopri5_py

        times, packed_states = solve_lindblad_dopri5_py(
            prepared.rust_plan,
            packed_rho0,
            t_span_tuple[0],
            t_span_tuple[1],
            float(abstol),
            float(reltol),
            float(dt),
            None if saveat_values is None else np.asarray(saveat_values, dtype=np.float64),
            bool(save_start),
            int(maxiters),
            execution_mode,
        )
        return LindbladResult(
            t=np.asarray(times, dtype=np.float64),
            packed_y=np.asarray(packed_states, dtype=np.float64),
            layout=prepared.layout,
        )
    if backend == "python" and solver != "explicit":
        raise NotImplementedError("the python backend only supports solver='explicit'")
    return _solve_python_reference(
        prepared,
        packed_rho0,
        t_span_tuple,
        execution_mode=execution_mode,
        method="RK45",
        abstol=abstol,
        reltol=reltol,
        dt=dt,
        saveat=saveat_values,
        save_start=save_start,
        maxiters=maxiters,
    )
