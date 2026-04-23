use crate::effective_lindblad::plan::{parse_effective_lindblad_plan, EffectiveLindbladPlan};
use crate::effective_lindblad::rhs::{rhs_effective_lindblad, EffectiveLindbladWorkspace};
use crate::effective_lindblad::solver::{solve_effective_lindblad_dopri5, EffectiveSolverOptions};
use crate::lindblad::eval::RuntimeValue;
use crate::lindblad::plan::parse_parameter_graph;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
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
    maxiters = 100000
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
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
    };
    let (times, states) = solve_effective_lindblad_dopri5(
        &plan,
        y0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.real_dim == 0 {
        0
    } else {
        states.len() / plan.real_dim
    };
    let states_array = PyArray1::from_vec(py, states)
        .reshape((n_rows, plan.real_dim))
        .map_err(PyValueError::new_err)?;
    Ok((times_array, states_array))
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

    let l_norm: f64 = workspace
        .l_scratch
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();

    let result = PyDict::new(py);
    result.set_item("field_val", field_val)?;
    result.set_item("rabi_val", rabi_val)?;
    result.set_item("detuning_val", detuning_val)?;
    result.set_item("dy_norm", dy_norm)?;
    result.set_item("dy_max", dy_max)?;
    result.set_item("has_nan", has_nan)?;
    result.set_item("has_inf", has_inf)?;
    result.set_item("l_scratch_norm", l_norm)?;
    result.set_item("last_interval", workspace.last_interval)?;
    result.set_item("dy", PyArray1::from_vec(py, dy))?;
    Ok(result)
}
