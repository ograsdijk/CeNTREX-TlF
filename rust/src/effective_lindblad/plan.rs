use crate::lindblad::plan::ParameterGraph;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(module = "centrex_tlf.centrex_tlf_rust")]
#[derive(Clone)]
pub struct EffectiveLindbladPlan {
    #[pyo3(get)]
    pub n_states: usize,
    #[pyo3(get)]
    pub real_dim: usize,
    #[pyo3(get)]
    pub n_grid: usize,
    pub field_grid: Vec<f64>,
    pub l_combined: Vec<f64>,
    pub l_opt: Vec<f64>,
    pub l_det: Vec<f64>,
    pub dl_combined: Vec<f64>,
    pub dl_opt: Vec<f64>,
    pub dl_det: Vec<f64>,
    pub excited_indices: Vec<usize>,
    pub ground_indices: Vec<usize>,
    pub sink_indices: Vec<usize>,
    pub parameter_graph: ParameterGraph,
    pub field_coordinate_slot: usize,
    pub rabi_rate_slot: usize,
    pub detuning_slot: usize,
    pub is_time_dependent: bool,
}

fn required_f64_array(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let arr: PyReadonlyArray1<f64> = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract()?;
    Ok(arr.as_slice().map_err(PyValueError::new_err)?.to_vec())
}

fn required_usize_array(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<usize>> {
    let arr: PyReadonlyArray1<i64> = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract()?;
    Ok(arr
        .as_slice()
        .map_err(PyValueError::new_err)?
        .iter()
        .map(|&v| v as usize)
        .collect())
}

fn required_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    let val: i64 = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract()?;
    Ok(val as usize)
}

fn required_bool(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    let val: bool = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract()?;
    Ok(val)
}

pub fn parse_effective_lindblad_plan(
    dict: &Bound<'_, PyDict>,
    parameter_graph: ParameterGraph,
) -> PyResult<EffectiveLindbladPlan> {
    let n_states = required_usize(dict, "n_states")?;
    let real_dim = required_usize(dict, "real_dim")?;
    let n_grid = required_usize(dict, "n_grid")?;
    let field_grid = required_f64_array(dict, "field_grid")?;
    let l_combined = required_f64_array(dict, "l_combined")?;
    let l_opt = required_f64_array(dict, "l_opt")?;
    let l_det = required_f64_array(dict, "l_det")?;
    let dl_combined = required_f64_array(dict, "dl_combined")?;
    let dl_opt = required_f64_array(dict, "dl_opt")?;
    let dl_det = required_f64_array(dict, "dl_det")?;
    let excited_indices = required_usize_array(dict, "excited_indices")?;
    let ground_indices = required_usize_array(dict, "ground_indices")?;
    let sink_indices = required_usize_array(dict, "sink_indices")?;
    let field_coordinate_slot = required_usize(dict, "field_coordinate_slot")?;
    let rabi_rate_slot = required_usize(dict, "rabi_rate_slot")?;
    let detuning_slot = required_usize(dict, "detuning_slot")?;
    let is_time_dependent = required_bool(dict, "is_time_dependent")?;

    if field_grid.len() != n_grid {
        return Err(PyValueError::new_err("field_grid length mismatch"));
    }
    let mat_size = real_dim * real_dim;
    if l_combined.len() != n_grid * mat_size {
        return Err(PyValueError::new_err("l_combined size mismatch"));
    }
    if l_opt.len() != n_grid * mat_size {
        return Err(PyValueError::new_err("l_opt size mismatch"));
    }
    if l_det.len() != n_grid * mat_size {
        return Err(PyValueError::new_err("l_det size mismatch"));
    }
    let diff_size = (n_grid.saturating_sub(1)) * mat_size;
    if dl_combined.len() != diff_size {
        return Err(PyValueError::new_err("dl_combined size mismatch"));
    }

    Ok(EffectiveLindbladPlan {
        n_states,
        real_dim,
        n_grid,
        field_grid,
        l_combined,
        l_opt,
        l_det,
        dl_combined,
        dl_opt,
        dl_det,
        excited_indices,
        ground_indices,
        sink_indices,
        parameter_graph,
        field_coordinate_slot,
        rabi_rate_slot,
        detuning_slot,
        is_time_dependent,
    })
}
