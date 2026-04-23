use crate::lindblad::plan::{parse_plan_payload, PreparedLindbladPlan};
use crate::lindblad::rhs::{
    build_packed_jacobian_sparse, build_split_jacobian_sparse, rhs_matrix_into,
    rhs_matrix_into_with_profile, rhs_packed, rhs_packed_into_with_profile,
    rhs_split_into_with_profile, ExecutionMode, RhsProfileStats, RhsWorkspace,
};
use crate::lindblad::solver_batch::{
    solve_lindblad_batch, solve_lindblad_grid_batch, BatchFastSolver, BatchOutputValues,
};
use crate::lindblad::solver_dopri5_fast::{
    solve_dopri5_fast, solve_dopri5_fast_output, solve_dopri5_fast_with_stats,
};
use crate::lindblad::solver_fast_common::{
    FastOutputKind, FastOutputOptions, FastOutputValues, FastOutputWhen,
};
use crate::lindblad::solver_ode::{solve_dopri5, solve_dopri5_with_stats, OdeSolverOptions};
use crate::lindblad::solver_stats::SolveStats;
use crate::lindblad::solver_tsit5_fast::{
    solve_tsit5_fast, solve_tsit5_fast_output, solve_tsit5_fast_with_stats,
};
use num_complex::Complex64;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods};
use std::cell::{Cell, RefCell};

fn solve_stats_to_dict<'py>(py: Python<'py>, stats: &SolveStats) -> PyResult<Py<PyDict>> {
    let summary = PyDict::new(py);
    summary.set_item("solver", stats.solver.as_str())?;
    summary.set_item("rhs_calls", stats.rhs_calls)?;
    summary.set_item("jacobian_calls", stats.jacobian_calls)?;
    summary.set_item("function_evaluations", stats.function_evaluations)?;
    summary.set_item("accepted_steps", stats.accepted_steps)?;
    summary.set_item("rejected_steps", stats.rejected_steps)?;
    summary.set_item("internal_steps", stats.internal_steps)?;
    summary.set_item("saved_points", stats.saved_points)?;
    summary.set_item("setup_seconds", stats.setup_seconds)?;
    summary.set_item("integration_seconds", stats.integration_seconds)?;
    summary.set_item("interpolation_seconds", stats.interpolation_seconds)?;
    summary.set_item("total_seconds", stats.total_seconds)?;
    summary.set_item("rhs_seconds", stats.rhs_seconds)?;
    summary.set_item("jacobian_seconds", stats.jacobian_seconds)?;
    summary.set_item("non_rhs_seconds", stats.non_rhs_seconds())?;
    summary.set_item("average_rhs_seconds", stats.average_rhs_seconds())?;
    summary.set_item("average_jacobian_seconds", stats.average_jacobian_seconds())?;
    Ok(summary.unbind())
}

fn parse_fast_output_options(
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    output_when: &str,
    dense_output: bool,
) -> PyResult<FastOutputOptions> {
    let kind =
        match output {
            "full" => {
                if output_indices.is_some() {
                    return Err(PyValueError::new_err(
                        "output_indices is only valid with output='selected'",
                    ));
                }
                FastOutputKind::Full
            }
            "populations" => {
                if output_indices.is_some() {
                    return Err(PyValueError::new_err(
                        "output_indices is only valid with output='selected'",
                    ));
                }
                FastOutputKind::Populations
            }
            "selected" => FastOutputKind::Selected(output_indices.ok_or_else(|| {
                PyValueError::new_err("output='selected' requires output_indices")
            })?),
            _ => {
                return Err(PyValueError::new_err(
                    "output must be 'full', 'populations', or 'selected'",
                ))
            }
        };
    let when = match output_when {
        "saveat" => FastOutputWhen::Saveat,
        "final" => FastOutputWhen::Final,
        _ => {
            return Err(PyValueError::new_err(
                "output_when must be 'saveat' or 'final'",
            ))
        }
    };
    Ok(FastOutputOptions {
        kind,
        when,
        dense_output,
    })
}

#[pyclass(module = "centrex_tlf.centrex_tlf_rust", unsendable)]
pub struct LindbladRhsEvaluator {
    plan: PreparedLindbladPlan,
    mode: ExecutionMode,
    workspace: RefCell<RhsWorkspace>,
    profiling_enabled: Cell<bool>,
    profile: RefCell<RhsProfileStats>,
}

#[pymethods]
impl LindbladRhsEvaluator {
    #[pyo3(signature = (enabled = true))]
    pub fn enable_profile_py(&self, enabled: bool) {
        self.profiling_enabled.set(enabled);
    }

    pub fn reset_profile_py(&self) {
        *self.profile.borrow_mut() = RhsProfileStats::default();
    }

    pub fn profile_summary_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let stats = self.profile.borrow().clone();
        let summary = PyDict::new(py);
        summary.set_item("enabled", self.profiling_enabled.get())?;
        summary.set_item("calls", stats.calls)?;
        summary.set_item("total_seconds", stats.total_seconds)?;
        summary.set_item("unpack_seconds", stats.unpack_seconds)?;
        summary.set_item("parameter_eval_seconds", stats.parameter_eval_seconds)?;
        summary.set_item("hamiltonian_fill_seconds", stats.hamiltonian_fill_seconds)?;
        summary.set_item("commutator_seconds", stats.commutator_seconds)?;
        summary.set_item("dissipator_seconds", stats.dissipator_seconds)?;
        summary.set_item("pack_seconds", stats.pack_seconds)?;
        if stats.calls > 0 {
            summary.set_item(
                "average_total_seconds",
                stats.total_seconds / stats.calls as f64,
            )?;
        } else {
            summary.set_item("average_total_seconds", 0.0)?;
        }
        Ok(summary.unbind())
    }

    pub fn rhs_packed_py<'py>(
        &self,
        py: Python<'py>,
        packed_state: PyReadonlyArray1<'py, f64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let state = packed_state.as_slice().map_err(PyValueError::new_err)?;
        let mut out = vec![0.0; self.plan.layout.packed_len()];
        let mut workspace = self.workspace.borrow_mut();
        if self.profiling_enabled.get() {
            let mut profile = self.profile.borrow_mut();
            rhs_packed_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                Some(&mut profile),
            )
            .map_err(PyValueError::new_err)?;
        } else {
            rhs_packed_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                None,
            )
            .map_err(PyValueError::new_err)?;
        }
        Ok(PyArray1::from_vec(py, out))
    }

    pub fn rhs_matrix_py<'py>(
        &self,
        py: Python<'py>,
        matrix_state: PyReadonlyArray1<'py, Complex64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let state = matrix_state.as_slice().map_err(PyValueError::new_err)?;
        let n = self.plan.n_states();
        if state.len() != n * n {
            return Err(PyValueError::new_err(format!(
                "expected full matrix state length {}, got {}",
                n * n,
                state.len()
            )));
        }
        let mut out = vec![Complex64::ZERO; n * n];
        let mut workspace = self.workspace.borrow_mut();
        if self.profiling_enabled.get() {
            let mut profile = self.profile.borrow_mut();
            rhs_matrix_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                Some(&mut profile),
            )
            .map_err(PyValueError::new_err)?;
        } else {
            rhs_matrix_into(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
            )
            .map_err(PyValueError::new_err)?;
        }
        Ok(PyArray1::from_vec(py, out))
    }

    pub fn rhs_split_py<'py>(
        &self,
        py: Python<'py>,
        split_state: PyReadonlyArray1<'py, f64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let state = split_state.as_slice().map_err(PyValueError::new_err)?;
        let n = self.plan.n_states() * self.plan.n_states();
        if state.len() != 2 * n {
            return Err(PyValueError::new_err(format!(
                "expected split-real state length {}, got {}",
                2 * n,
                state.len()
            )));
        }
        let mut out = vec![0.0; 2 * n];
        let mut workspace = self.workspace.borrow_mut();
        if self.profiling_enabled.get() {
            let mut profile = self.profile.borrow_mut();
            rhs_split_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                Some(&mut profile),
            )
            .map_err(PyValueError::new_err)?;
        } else {
            rhs_split_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                None,
            )
            .map_err(PyValueError::new_err)?;
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[pyo3(signature = (t, tol = 0.0))]
    pub fn jacobian_split_sparse_py<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        tol: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut workspace = self.workspace.borrow_mut();
        let (rows, cols, values) =
            build_split_jacobian_sparse(&self.plan, t, self.mode, &mut workspace, tol)
                .map_err(PyValueError::new_err)?;
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, cols),
            PyArray1::from_vec(py, values),
        ))
    }

    #[pyo3(signature = (t, tol = 0.0))]
    pub fn jacobian_packed_sparse_py<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        tol: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut workspace = self.workspace.borrow_mut();
        let (rows, cols, values) =
            build_packed_jacobian_sparse(&self.plan, t, self.mode, &mut workspace, tol)
                .map_err(PyValueError::new_err)?;
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, cols),
            PyArray1::from_vec(py, values),
        ))
    }
}

#[pyfunction(signature = (payload))]
pub fn prepare_lindblad_problem_py(payload: &Bound<'_, PyAny>) -> PyResult<PreparedLindbladPlan> {
    parse_plan_payload(payload)
}

#[pyfunction(signature = (plan, mode = "structured"))]
pub fn create_lindblad_rhs_evaluator_py(
    plan: PyRef<'_, PreparedLindbladPlan>,
    mode: &str,
) -> PyResult<LindbladRhsEvaluator> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    Ok(LindbladRhsEvaluator {
        plan: plan.clone(),
        mode: execution_mode,
        workspace: RefCell::new(RhsWorkspace::new(&plan)),
        profiling_enabled: Cell::new(false),
        profile: RefCell::new(RhsProfileStats::default()),
    })
}

#[pyfunction(signature = (plan, packed_state, t, mode = "structured"))]
pub fn lindblad_rhs_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_state: PyReadonlyArray1<'py, f64>,
    t: f64,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let rhs = rhs_packed(
        &plan,
        packed_state.as_slice().map_err(PyValueError::new_err)?,
        t,
        execution_mode,
    )
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, rhs))
}

#[pyfunction(signature = (plan, packed_vector, t, mode = "structured"))]
pub fn lindblad_jvp_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_vector: PyReadonlyArray1<'py, f64>,
    t: f64,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let jvp = rhs_packed(
        &plan,
        packed_vector.as_slice().map_err(PyValueError::new_err)?,
        t,
        execution_mode,
    )
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, jvp))
}

#[pyfunction(signature = (plan, t))]
pub fn evaluate_lindblad_hamiltonian_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    t: f64,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let matrix = plan
        .evaluate_hamiltonian(t)
        .map_err(PyValueError::new_err)?;
    let array = PyArray1::from_vec(py, matrix);
    array.reshape((plan.n_states(), plan.n_states()))
}

#[pyfunction(signature = (
    plan,
    packed_rho0_batch,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "expanded_sparse",
    solver = "dopri5_fast",
    output = "populations",
    output_indices = None,
    output_when = "final",
    dense_output = false,
    collect_stats = false,
    parameter_slot_indices = None,
    parameter_batch = None,
    parallel = true,
    threads = None
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_lindblad_batch_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0_batch: PyReadonlyArray2<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
    solver: &str,
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    output_when: &str,
    dense_output: bool,
    collect_stats: bool,
    parameter_slot_indices: Option<Vec<usize>>,
    parameter_batch: Option<PyReadonlyArray2<'py, Complex64>>,
    parallel: bool,
    threads: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Py<PyAny>,
    usize,
    usize,
    Py<PyDict>,
)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let solver = BatchFastSolver::from_str(solver).map_err(PyValueError::new_err)?;
    let output_options =
        parse_fast_output_options(output, output_indices, output_when, dense_output)?;
    if matches!(output_options.kind, FastOutputKind::Full) {
        return Err(PyValueError::new_err(
            "solve_lindblad_batch does not support output='full' yet",
        ));
    }
    let rho0_shape = packed_rho0_batch.shape();
    if rho0_shape.len() != 2 {
        return Err(PyValueError::new_err("packed_rho0_batch must be 2D"));
    }
    let trajectory_count = rho0_shape[0];
    if rho0_shape[1] != plan.layout.packed_len() {
        return Err(PyValueError::new_err(format!(
            "packed_rho0_batch second dimension must be {}, got {}",
            plan.layout.packed_len(),
            rho0_shape[1]
        )));
    }
    let y0_batch = packed_rho0_batch
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();
    let saveat_values = saveat
        .map(|values| values.as_slice().map(|slice| slice.to_vec()))
        .transpose()
        .map_err(PyValueError::new_err)?;
    let parameter_slot_indices = parameter_slot_indices.unwrap_or_default();
    let parameter_values = match parameter_batch {
        Some(values) => {
            let shape = values.shape();
            if shape.len() != 2 {
                return Err(PyValueError::new_err("parameter_batch must be 2D"));
            }
            if shape[0] != trajectory_count {
                return Err(PyValueError::new_err(format!(
                    "parameter_batch first dimension must be trajectory count {}, got {}",
                    trajectory_count, shape[0]
                )));
            }
            if shape[1] != parameter_slot_indices.len() {
                return Err(PyValueError::new_err(format!(
                    "parameter_batch second dimension must match parameter_slot_indices length {}, got {}",
                    parameter_slot_indices.len(),
                    shape[1]
                )));
            }
            Some(values.as_slice().map_err(PyValueError::new_err)?.to_vec())
        }
        None => None,
    };
    let plan_owned = plan.clone();
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat_values,
        save_start,
        maxiters,
        mode: execution_mode,
    };

    let result = py
        .detach(move || {
            solve_lindblad_batch(
                &plan_owned,
                solver,
                y0_batch.as_slice(),
                trajectory_count,
                t0,
                t1,
                &options,
                &output_options,
                collect_stats,
                parameter_slot_indices.as_slice(),
                parameter_values.as_deref(),
                parallel,
                threads,
            )
        })
        .map_err(PyValueError::new_err)?;

    let times = PyArray1::from_vec(py, result.times);
    let values: Py<PyAny> = match result.values {
        BatchOutputValues::Real(values) => PyArray1::from_vec(py, values).into_any().unbind(),
        BatchOutputValues::Complex(values) => PyArray1::from_vec(py, values).into_any().unbind(),
    };
    Ok((
        times,
        values,
        result.width,
        result.time_count,
        solve_stats_to_dict(py, &result.stats)?,
    ))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    parameter_slot_indices,
    parameter_axes,
    parameter_axis_lengths,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "expanded_sparse",
    solver = "dopri5_fast",
    output = "populations",
    output_indices = None,
    output_when = "final",
    dense_output = false,
    collect_stats = false,
    parallel = true,
    threads = None
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_lindblad_grid_batch_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    parameter_slot_indices: Vec<usize>,
    parameter_axes: PyReadonlyArray1<'py, Complex64>,
    parameter_axis_lengths: Vec<usize>,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
    solver: &str,
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    output_when: &str,
    dense_output: bool,
    collect_stats: bool,
    parallel: bool,
    threads: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Py<PyAny>,
    usize,
    usize,
    usize,
    Py<PyDict>,
)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let solver = BatchFastSolver::from_str(solver).map_err(PyValueError::new_err)?;
    let output_options =
        parse_fast_output_options(output, output_indices, output_when, dense_output)?;
    if matches!(output_options.kind, FastOutputKind::Full) {
        return Err(PyValueError::new_err(
            "solve_lindblad_batch does not support output='full' yet",
        ));
    }
    let y0 = packed_rho0
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();
    if y0.len() != plan.layout.packed_len() {
        return Err(PyValueError::new_err(format!(
            "packed_rho0 length must be {}, got {}",
            plan.layout.packed_len(),
            y0.len()
        )));
    }
    if parameter_slot_indices.len() != parameter_axis_lengths.len() {
        return Err(PyValueError::new_err(format!(
            "parameter_slot_indices length {} does not match parameter_axis_lengths length {}",
            parameter_slot_indices.len(),
            parameter_axis_lengths.len()
        )));
    }
    let trajectory_count = parameter_axis_lengths
        .iter()
        .try_fold(1usize, |acc, length| {
            if *length == 0 {
                None
            } else {
                acc.checked_mul(*length)
            }
        })
        .ok_or_else(|| PyValueError::new_err("parameter grid axes must be non-empty and finite"))?;
    let axes = parameter_axes
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();
    let saveat_values = saveat
        .map(|values| values.as_slice().map(|slice| slice.to_vec()))
        .transpose()
        .map_err(PyValueError::new_err)?;
    let plan_owned = plan.clone();
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat_values,
        save_start,
        maxiters,
        mode: execution_mode,
    };

    let result = py
        .detach(move || {
            solve_lindblad_grid_batch(
                &plan_owned,
                solver,
                y0.as_slice(),
                t0,
                t1,
                &options,
                &output_options,
                collect_stats,
                parameter_slot_indices.as_slice(),
                axes.as_slice(),
                parameter_axis_lengths.as_slice(),
                parallel,
                threads,
            )
        })
        .map_err(PyValueError::new_err)?;

    let times = PyArray1::from_vec(py, result.times);
    let values: Py<PyAny> = match result.values {
        BatchOutputValues::Real(values) => PyArray1::from_vec(py, values).into_any().unbind(),
        BatchOutputValues::Complex(values) => PyArray1::from_vec(py, values).into_any().unbind(),
    };
    Ok((
        times,
        values,
        result.width,
        result.time_count,
        trajectory_count,
        solve_stats_to_dict(py, &result.stats)?,
    ))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_dopri5_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states) = solve_dopri5(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_dopri5_profile_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Py<PyDict>,
)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states, stats) = solve_dopri5_with_stats(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array, solve_stats_to_dict(py, &stats)?))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_dopri5_fast_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states) = solve_dopri5_fast(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_dopri5_fast_profile_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Py<PyDict>,
)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states, stats) = solve_dopri5_fast_with_stats(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array, solve_stats_to_dict(py, &stats)?))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured",
    output = "full",
    output_indices = None,
    output_when = "saveat",
    dense_output = true,
    collect_stats = false
))]
pub fn solve_lindblad_dopri5_fast_output_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    output_when: &str,
    dense_output: bool,
    collect_stats: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Py<PyAny>, usize, Py<PyDict>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let output_options =
        parse_fast_output_options(output, output_indices, output_when, dense_output)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let result = solve_dopri5_fast_output(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
        &output_options,
        collect_stats,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, result.times);
    let values: Py<PyAny> = match result.values {
        FastOutputValues::Full(values) => PyArray1::from_vec(py, values).into_any().unbind(),
        FastOutputValues::Real(values) => PyArray1::from_vec(py, values).into_any().unbind(),
        FastOutputValues::Complex(values) => PyArray1::from_vec(py, values).into_any().unbind(),
    };
    Ok((
        times_array,
        values,
        result.width,
        solve_stats_to_dict(py, &result.stats)?,
    ))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_tsit5_fast_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states) = solve_tsit5_fast(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_tsit5_fast_profile_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Py<PyDict>,
)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states, stats) = solve_tsit5_fast_with_stats(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array, solve_stats_to_dict(py, &stats)?))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured",
    output = "full",
    output_indices = None,
    output_when = "saveat",
    dense_output = true,
    collect_stats = false
))]
pub fn solve_lindblad_tsit5_fast_output_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    output_when: &str,
    dense_output: bool,
    collect_stats: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Py<PyAny>, usize, Py<PyDict>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let output_options =
        parse_fast_output_options(output, output_indices, output_when, dense_output)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let result = solve_tsit5_fast_output(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
        &output_options,
        collect_stats,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, result.times);
    let values: Py<PyAny> = match result.values {
        FastOutputValues::Full(values) => PyArray1::from_vec(py, values).into_any().unbind(),
        FastOutputValues::Real(values) => PyArray1::from_vec(py, values).into_any().unbind(),
        FastOutputValues::Complex(values) => PyArray1::from_vec(py, values).into_any().unbind(),
    };
    Ok((
        times_array,
        values,
        result.width,
        solve_stats_to_dict(py, &result.stats)?,
    ))
}
