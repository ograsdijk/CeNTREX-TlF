use crate::effective_lindblad::plan::{parse_effective_lindblad_plan, EffectiveLindbladPlan};
use crate::effective_lindblad::rhs::{rhs_effective_lindblad, EffectiveLindbladWorkspace};
use crate::effective_lindblad::solver::{
    solve_effective_lindblad, solve_effective_lindblad_batch, EffectiveSolverOptions,
};
use crate::lindblad::eval::RuntimeValue;
use crate::lindblad::plan::parse_parameter_graph;
use crate::ode::output::OdeOutputValues;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn prepare_effective_lindblad_plan_py<'py>(
    py: Python<'py>,
    plan_dict: &Bound<'py, PyDict>,
) -> PyResult<EffectiveLindbladPlan> {
    let param_graph_any = plan_dict
        .get_item("parameter_graph")?
        .ok_or_else(|| PyValueError::new_err("missing 'parameter_graph'"))?;
    let parameter_graph = parse_parameter_graph(&param_graph_any).map_err(PyValueError::new_err)?;
    parse_effective_lindblad_plan(plan_dict, parameter_graph)
}

#[pyfunction(signature = (
    plan,
    y0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    solver = "dopri5",
    output = "full",
    output_indices = None,
    integral_weights = None,
))]
pub fn solve_effective_lindblad_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, EffectiveLindbladPlan>,
    y0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    solver: &str,
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    integral_weights: Option<Vec<(usize, f64)>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Py<PyAny>, usize)> {
    let options = EffectiveSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|arr| arr.as_slice().map(|s| s.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        solver: solver.to_string(),
    };
    let result = solve_effective_lindblad(
        &plan,
        y0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
        output,
        output_indices.as_deref(),
        integral_weights.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, result.times);
    let values: Py<PyAny> = match result.values {
        OdeOutputValues::Full(v) => PyArray1::from_vec(py, v).into_any().unbind(),
        OdeOutputValues::Real(v) => PyArray1::from_vec(py, v).into_any().unbind(),
        OdeOutputValues::Complex(v) => PyArray1::from_vec(py, v).into_any().unbind(),
    };
    Ok((times_array, values, result.width))
}

#[pyfunction]
pub fn debug_effective_lindblad_rhs_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, EffectiveLindbladPlan>,
    y0: PyReadonlyArray1<'py, f64>,
    t: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let y = y0.as_slice().map_err(PyValueError::new_err)?;
    let mut workspace = EffectiveLindbladWorkspace::new(&plan);
    let mut dy = vec![0.0f64; plan.real_dim];

    plan.parameter_graph
        .evaluate_into(
            t,
            &mut workspace.parameter_values,
            &mut workspace.eval_stack,
            &mut workspace.scalar_stack,
        )
        .map_err(PyValueError::new_err)?;

    let field_val = match &workspace.parameter_values[plan.field_coordinate_slot] {
        RuntimeValue::Scalar(c) => c.re,
        RuntimeValue::Tuple(_) => {
            return Err(PyValueError::new_err("field_coordinate is a tuple"));
        }
    };
    let rabi_val = match &workspace.parameter_values[plan.rabi_rate_slot] {
        RuntimeValue::Scalar(c) => c.re,
        RuntimeValue::Tuple(_) => {
            return Err(PyValueError::new_err("rabi_rate is a tuple"));
        }
    };
    let detuning_val = match &workspace.parameter_values[plan.detuning_slot] {
        RuntimeValue::Scalar(c) => c.re,
        RuntimeValue::Tuple(_) => {
            return Err(PyValueError::new_err("detuning is a tuple"));
        }
    };

    rhs_effective_lindblad(&plan, y, t, &mut workspace, &mut dy).map_err(PyValueError::new_err)?;

    let dy_norm: f64 = dy.iter().map(|x| x * x).sum::<f64>().sqrt();
    let dy_max: f64 = dy.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let has_nan = dy.iter().any(|x| x.is_nan());
    let has_inf = dy.iter().any(|x| x.is_infinite());

    let result = PyDict::new(py);
    result.set_item("field_val", field_val)?;
    result.set_item("rabi_val", rabi_val)?;
    result.set_item("detuning_val", detuning_val)?;
    result.set_item("dy_norm", dy_norm)?;
    result.set_item("dy_max", dy_max)?;
    result.set_item("has_nan", has_nan)?;
    result.set_item("has_inf", has_inf)?;
    result.set_item("last_interval", workspace.last_interval)?;
    result.set_item("dy", PyArray1::from_vec(py, dy))?;
    Ok(result)
}

#[pyfunction(signature = (plan, packed_rho0, t0, t1, abstol, reltol, dt, saveat = None, save_start = true, maxiters = 100000, solver = "dopri5", output = "populations", output_indices = None, integral_weights = None, parameter_slot_indices = None, parameter_batch = None, trajectory_count = 1, parallel = true, threads = None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_effective_lindblad_batch_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, EffectiveLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    solver: &str,
    output: &str,
    output_indices: Option<Vec<(usize, usize)>>,
    integral_weights: Option<Vec<(usize, f64)>>,
    parameter_slot_indices: Option<Vec<usize>>,
    parameter_batch: Option<PyReadonlyArray2<'py, f64>>,
    trajectory_count: usize,
    parallel: bool,
    threads: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
    Py<PyDict>,
)> {
    let y0 = packed_rho0.as_slice().map_err(PyValueError::new_err)?;
    let saveat_vec = saveat
        .map(|a| a.as_slice().map(|s| s.to_vec()))
        .transpose()
        .map_err(PyValueError::new_err)?;
    let slot_indices = parameter_slot_indices.unwrap_or_default();
    let param_batch: Vec<f64> = match parameter_batch {
        Some(arr) => {
            let shape = arr.shape();
            if shape[0] != trajectory_count || shape[1] != slot_indices.len() {
                return Err(PyValueError::new_err(format!(
                    "parameter_batch shape ({},{}) doesn't match ({},{})",
                    shape[0],
                    shape[1],
                    trajectory_count,
                    slot_indices.len()
                )));
            }
            arr.as_slice().map_err(PyValueError::new_err)?.to_vec()
        }
        None => vec![],
    };
    let options = EffectiveSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat_vec,
        save_start,
        maxiters,
        solver: solver.to_string(),
    };
    let plan_owned = plan.clone();
    let y0_owned = y0.to_vec();
    let output_owned = output.to_string();
    let output_indices_owned = output_indices;
    let integral_weights_owned = integral_weights;
    let result = py
        .detach(move || {
            solve_effective_lindblad_batch(
                &plan_owned,
                &y0_owned,
                t0,
                t1,
                &options,
                &output_owned,
                output_indices_owned.as_deref(),
                integral_weights_owned.as_deref(),
                &slot_indices,
                &param_batch,
                trajectory_count,
                parallel,
                threads,
            )
        })
        .map_err(PyValueError::new_err)?;
    let times = PyArray1::from_vec(py, result.times);
    let values = match result.values {
        OdeOutputValues::Real(v) => PyArray1::from_vec(py, v),
        OdeOutputValues::Full(v) => PyArray1::from_vec(py, v),
        _ => return Err(PyValueError::new_err("unexpected output type")),
    };
    let d = PyDict::new(py);
    d.set_item("solver", solver)?;
    d.set_item("accepted_steps", result.stats.accepted_steps)?;
    d.set_item("rejected_steps", result.stats.rejected_steps)?;
    d.set_item("rhs_calls", result.stats.rhs_calls)?;
    d.set_item("trajectory_count", trajectory_count)?;
    Ok((times, values, result.width, result.time_count, d.unbind()))
}
