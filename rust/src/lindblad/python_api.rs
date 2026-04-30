use crate::lindblad::ode_impl::{LindbladRhs, LindbladStopEvent};
use crate::lindblad::plan::{parse_expression, parse_plan_payload, PreparedLindbladPlan};
use crate::lindblad::rhs::{
    build_packed_jacobian_sparse, build_split_jacobian_sparse, rhs_matrix_into,
    rhs_matrix_into_with_profile, rhs_packed, rhs_packed_into_with_profile,
    rhs_split_into_with_profile, ExecutionMode, RhsProfileStats, RhsWorkspace,
};
use crate::ode::batch::{solve_single, OdeSolver};
use crate::ode::output::{
    FullOutput, OdeOutputValues, PopulationsOutput, SelectedExtraction, SelectedOutput,
    WeightedIntegralOutput,
};
use num_complex::Complex64;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyDictMethods};
use std::cell::{Cell, RefCell};

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
fn packed_upper_idx(n: usize, i: usize, j: usize) -> usize {
    let mut off = 0usize;
    for row in 0..i {
        off += n - row - 1;
    }
    n + 2 * (off + (j - i - 1))
}

fn build_extractions(n: usize, indices: &[(usize, usize)]) -> Vec<SelectedExtraction> {
    indices
        .iter()
        .map(|&(i, j)| {
            if i == j {
                SelectedExtraction::Real(i)
            } else if i < j {
                let r = packed_upper_idx(n, i, j);
                SelectedExtraction::ComplexPair {
                    real_idx: r,
                    imag_idx: r + 1,
                }
            } else {
                let r = packed_upper_idx(n, j, i);
                SelectedExtraction::ComplexPairConj {
                    real_idx: r,
                    imag_idx: r + 1,
                }
            }
        })
        .collect()
}

fn parse_stop_event(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<LindbladStopEvent>> {
    let Some(obj) = obj else {
        return Ok(None);
    };
    let dict: &Bound<'_, PyDict> = obj.cast()?;
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("stop_event missing kind"))?
        .extract()?;
    match kind.as_str() {
        "population" => {
            let indices: Vec<usize> = dict
                .get_item("indices")?
                .ok_or_else(|| PyValueError::new_err("population stop_event missing indices"))?
                .extract()?;
            let threshold: f64 = dict
                .get_item("threshold")?
                .ok_or_else(|| PyValueError::new_err("population stop_event missing threshold"))?
                .extract()?;
            Ok(Some(LindbladStopEvent::PopulationThreshold { indices, threshold }))
        }
        "runtime_expression" => {
            let expression_obj = dict
                .get_item("expression")?
                .ok_or_else(|| PyValueError::new_err("runtime stop_event missing expression"))?;
            Ok(Some(LindbladStopEvent::RuntimeExpression {
                expression: parse_expression(&expression_obj)?,
            }))
        }
        other => Err(PyValueError::new_err(format!("unknown stop_event kind {other:?}"))),
    }
}

#[pyfunction(signature = (plan, packed_rho0, t0, t1, abstol, reltol, dt, saveat = None, save_start = true, maxiters = 100000, mode = "expanded_sparse", solver = "dopri5", output = "full", output_indices = None, output_when = "saveat", integral_weights = None, stop_event = None))]
pub fn solve_lindblad_ode_py<'py>(
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
    solver: &str,
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    output_when: &str,
    integral_weights: Option<Vec<(usize, f64)>>,
    stop_event: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Py<PyAny>, usize, Py<PyDict>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let ode_solver = OdeSolver::from_str(solver).map_err(PyValueError::new_err)?;
    let n = plan.layout.n;
    let dim = plan.layout.packed_len();
    let saveat_vec = saveat
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .transpose()
        .map_err(PyValueError::new_err)?;
    let capacity = saveat_vec.as_ref().map_or(maxiters + 1, |s| s.len() + 1);
    let y0 = packed_rho0.as_slice().map_err(PyValueError::new_err)?;
    let options = crate::ode::OdeOptions {
        abstol,
        reltol,
        dt,
        maxiters,
        save_start,
        saveat: saveat_vec,
    };
    let event = parse_stop_event(stop_event)?;
    let mut rhs = LindbladRhs::new(&plan, execution_mode).with_stop_event(event);
    let result = match output {
        "populations" => {
            let mut out = PopulationsOutput::new((0..n).collect(), capacity);
            let s = solve_single(&mut rhs, y0, t0, t1, &options, &mut out, ode_solver)
                .map_err(PyValueError::new_err)?;
            (out.finish(), s)
        }
        "selected" => {
            let idx = output_indices.as_deref().ok_or_else(|| {
                PyValueError::new_err("output='selected' requires output_indices")
            })?;
            let mut out = SelectedOutput::new(build_extractions(n, idx), capacity);
            let s = solve_single(&mut rhs, y0, t0, t1, &options, &mut out, ode_solver)
                .map_err(PyValueError::new_err)?;
            (out.finish(), s)
        }
        "weighted_integral" | "photon_integral" | "excited_population" => {
            let weights = integral_weights.ok_or_else(|| {
                PyValueError::new_err(format!("output='{output}' requires integral_weights"))
            })?;
            let mut out = WeightedIntegralOutput::new(weights);
            let s = solve_single(&mut rhs, y0, t0, t1, &options, &mut out, ode_solver)
                .map_err(PyValueError::new_err)?;
            (out.finish(), s)
        }
        _ => {
            let mut out = FullOutput::new(dim, capacity);
            let s = solve_single(&mut rhs, y0, t0, t1, &options, &mut out, ode_solver)
                .map_err(PyValueError::new_err)?;
            (out.finish(), s)
        }
    };
    let (r, stats) = result;
    let times_array = PyArray1::from_vec(py, r.times);
    let values: Py<PyAny> = match r.values {
        OdeOutputValues::Full(v) => PyArray1::from_vec(py, v).into_any().unbind(),
        OdeOutputValues::Real(v) => PyArray1::from_vec(py, v).into_any().unbind(),
        OdeOutputValues::Complex(v) => PyArray1::from_vec(py, v).into_any().unbind(),
    };
    let d = PyDict::new(py);
    d.set_item("solver", solver)?;
    d.set_item("accepted_steps", stats.accepted_steps)?;
    d.set_item("rejected_steps", stats.rejected_steps)?;
    d.set_item("rhs_calls", stats.rhs_calls)?;
    d.set_item("event_triggered", stats.event_triggered)?;
    if stats.event_triggered {
        d.set_item("event_time", stats.event_time)?;
        d.set_item("event_index", stats.event_index)?;
    }
    Ok((times_array, values, r.width, d.unbind()))
}

#[pyfunction(signature = (plan, packed_rho0_batch, t0, t1, abstol, reltol, dt, saveat = None, save_start = true, maxiters = 100000, mode = "expanded_sparse", solver = "dopri5", output = "populations", output_indices = None, output_when = "final", integral_weights = None, parameter_slot_indices = None, parameter_batch = None, parallel = true, threads = None, stop_event = None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_lindblad_batch_ode_py<'py>(
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
    integral_weights: Option<Vec<(usize, f64)>>,
    parameter_slot_indices: Option<Vec<usize>>,
    parameter_batch: Option<PyReadonlyArray2<'py, Complex64>>,
    parallel: bool,
    threads: Option<usize>,
    stop_event: Option<&Bound<'py, PyAny>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Py<PyAny>,
    usize,
    usize,
    Py<PyDict>,
)> {
    use crate::lindblad::ode_batch::solve_batch_ode;
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let ode_solver = OdeSolver::from_str(solver).map_err(PyValueError::new_err)?;
    let rho0_shape = packed_rho0_batch.shape();
    if rho0_shape.len() != 2 {
        return Err(PyValueError::new_err("packed_rho0_batch must be 2D"));
    }
    let trajectory_count = rho0_shape[0];
    if rho0_shape[1] != plan.layout.packed_len() {
        return Err(PyValueError::new_err(format!(
            "rho0 dim mismatch: {} vs {}",
            rho0_shape[1],
            plan.layout.packed_len()
        )));
    }
    let y0_batch = packed_rho0_batch
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();
    let mut saveat_vec = saveat
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .transpose()
        .map_err(PyValueError::new_err)?;
    if output_when == "final" {
        saveat_vec = Some(vec![t1]);
    }
    let slot_indices = parameter_slot_indices.unwrap_or_default();
    let param_values = match parameter_batch {
        Some(v) => {
            let s = v.shape();
            if s[0] != trajectory_count || s[1] != slot_indices.len() {
                return Err(PyValueError::new_err("parameter_batch shape mismatch"));
            }
            Some(v.as_slice().map_err(PyValueError::new_err)?.to_vec())
        }
        None => None,
    };
    let plan_owned = plan.clone();
    let event = parse_stop_event(stop_event)?;
    let options = crate::ode::OdeOptions {
        abstol,
        reltol,
        dt,
        maxiters,
        save_start: output_when != "final" && save_start,
        saveat: saveat_vec,
    };
    let result = py
        .detach(move || {
            solve_batch_ode(
                &plan_owned,
                ode_solver,
                &y0_batch,
                trajectory_count,
                t0,
                t1,
                &options,
                execution_mode,
                output,
                output_indices.as_deref(),
                integral_weights.as_deref(),
                &slot_indices,
                param_values.as_deref(),
                event,
                parallel,
                threads,
            )
        })
        .map_err(PyValueError::new_err)?;
    let times = PyArray1::from_vec(py, result.times);
    let values: Py<PyAny> = match result.values {
        crate::ode::output::OdeOutputValues::Real(v) => {
            PyArray1::from_vec(py, v).into_any().unbind()
        }
        crate::ode::output::OdeOutputValues::Complex(v) => {
            PyArray1::from_vec(py, v).into_any().unbind()
        }
        crate::ode::output::OdeOutputValues::Full(v) => {
            PyArray1::from_vec(py, v).into_any().unbind()
        }
    };
    let d = PyDict::new(py);
    d.set_item("solver", solver)?;
    d.set_item("accepted_steps", result.stats.accepted_steps)?;
    d.set_item("rejected_steps", result.stats.rejected_steps)?;
    d.set_item("rhs_calls", result.stats.rhs_calls)?;
    let event_count = result.event_triggered.iter().filter(|&&triggered| triggered).count();
    d.set_item("event_triggered", result.stats.event_triggered)?;
    d.set_item("event_triggered_by_trajectory", result.event_triggered)?;
    d.set_item("event_times", result.event_times)?;
    d.set_item("event_count", event_count)?;
    Ok((times, values, result.width, result.time_count, d.unbind()))
}

#[pyfunction(signature = (plan, packed_rho0, t0, t1, abstol, reltol, dt, parameter_slot_indices, parameter_axes, parameter_axis_lengths, saveat = None, save_start = true, maxiters = 100000, mode = "expanded_sparse", solver = "dopri5", output = "populations", output_indices = None, output_when = "final", integral_weights = None, parallel = true, threads = None, stop_event = None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_lindblad_grid_ode_py<'py>(
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
    integral_weights: Option<Vec<(usize, f64)>>,
    parallel: bool,
    threads: Option<usize>,
    stop_event: Option<&Bound<'py, PyAny>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Py<PyAny>,
    usize,
    usize,
    Py<PyDict>,
)> {
    use crate::lindblad::ode_batch::solve_grid_ode;
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let ode_solver = OdeSolver::from_str(solver).map_err(PyValueError::new_err)?;
    let y0 = packed_rho0
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();
    let axes = parameter_axes
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();
    let mut saveat_vec = saveat
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .transpose()
        .map_err(PyValueError::new_err)?;
    if output_when == "final" {
        saveat_vec = Some(vec![t1]);
    }
    let mut axis_offsets = Vec::with_capacity(parameter_axis_lengths.len());
    let mut off = 0usize;
    for &len in &parameter_axis_lengths {
        axis_offsets.push(off);
        off += len;
    }
    let plan_owned = plan.clone();
    let event = parse_stop_event(stop_event)?;
    let options = crate::ode::OdeOptions {
        abstol,
        reltol,
        dt,
        maxiters,
        save_start: output_when != "final" && save_start,
        saveat: saveat_vec,
    };
    let result = py
        .detach(move || {
            solve_grid_ode(
                &plan_owned,
                ode_solver,
                &y0,
                t0,
                t1,
                &options,
                execution_mode,
                output,
                output_indices.as_deref(),
                integral_weights.as_deref(),
                &parameter_slot_indices,
                &axes,
                &axis_offsets,
                &parameter_axis_lengths,
                event,
                parallel,
                threads,
            )
        })
        .map_err(PyValueError::new_err)?;
    let times = PyArray1::from_vec(py, result.times);
    let values: Py<PyAny> = match result.values {
        crate::ode::output::OdeOutputValues::Real(v) => {
            PyArray1::from_vec(py, v).into_any().unbind()
        }
        crate::ode::output::OdeOutputValues::Complex(v) => {
            PyArray1::from_vec(py, v).into_any().unbind()
        }
        crate::ode::output::OdeOutputValues::Full(v) => {
            PyArray1::from_vec(py, v).into_any().unbind()
        }
    };
    let d = PyDict::new(py);
    d.set_item("solver", solver)?;
    d.set_item("accepted_steps", result.stats.accepted_steps)?;
    d.set_item("rejected_steps", result.stats.rejected_steps)?;
    d.set_item("rhs_calls", result.stats.rhs_calls)?;
    let event_count = result.event_triggered.iter().filter(|&&triggered| triggered).count();
    d.set_item("event_triggered", result.stats.event_triggered)?;
    d.set_item("event_triggered_by_trajectory", result.event_triggered)?;
    d.set_item("event_times", result.event_times)?;
    d.set_item("event_count", event_count)?;
    Ok((times, values, result.width, result.time_count, d.unbind()))
}
