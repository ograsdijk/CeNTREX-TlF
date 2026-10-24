use crate::lindblad::plan::ParameterGraph;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Debug)]
pub struct SparseOperator {
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub pchip_coeffs: Vec<f64>,
    pub grid: Vec<f64>,
    pub nnz: usize,
    pub dim: usize,
    pub n_grid: usize,
}

impl SparseOperator {
    pub fn sparse_matvec_interpolated(
        &self,
        idx: usize,
        dx: f64,
        scale: f64,
        y: &[f64],
        dy: &mut [f64],
    ) {
        let interval = idx.min(self.n_grid.saturating_sub(2));
        let coeff_base = interval * self.nnz * 4;
        for i in 0..self.dim {
            let mut sum = 0.0;
            for ptr in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let cb = coeff_base + ptr * 4;
                let c0 = self.pchip_coeffs[cb];
                let c1 = self.pchip_coeffs[cb + 1];
                let c2 = self.pchip_coeffs[cb + 2];
                let c3 = self.pchip_coeffs[cb + 3];
                let val = c0 + dx * (c1 + dx * (c2 + dx * c3));
                sum += val * y[self.col_indices[ptr]];
            }
            dy[i] += scale * sum;
        }
    }
}

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
    pub sparse_combined: SparseOperator,
    pub sparse_opt: SparseOperator,
    pub sparse_det: SparseOperator,
    pub excited_indices: Vec<usize>,
    pub ground_indices: Vec<usize>,
    pub sink_indices: Vec<usize>,
    pub parameter_graph: ParameterGraph,
    pub field_coordinate_slot: usize,
    pub rabi_rate_slot: usize,
    pub detuning_slot: usize,
    pub operator_interpolation: OperatorInterpolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorInterpolation {
    Linear,
    Pchip,
}

#[pymethods]
impl EffectiveLindbladPlan {
    #[getter]
    fn slot_names(&self) -> Vec<String> {
        self.parameter_graph.slot_names.clone()
    }
}

fn required_f64_array(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let arr: PyReadonlyArray1<f64> = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract()?;
    Ok(arr.as_slice().map_err(PyValueError::new_err)?.to_vec())
}

fn to_index(value: i64, key: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{key} contains negative value {value}")))
}

fn required_usize_array(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<usize>> {
    let arr: PyReadonlyArray1<i64> = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract()?;
    arr.as_slice()
        .map_err(PyValueError::new_err)?
        .iter()
        .map(|&v| to_index(v, key))
        .collect()
}

fn required_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    let val: i64 = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract()?;
    to_index(val, key)
}

fn check_indices(indices: &[usize], bound: usize, what: &str) -> PyResult<()> {
    match indices.iter().find(|&&index| index >= bound) {
        Some(index) => Err(PyValueError::new_err(format!(
            "{what} index {index} out of range for {bound}"
        ))),
        None => Ok(()),
    }
}

pub fn parse_effective_lindblad_plan(
    dict: &Bound<'_, PyDict>,
    parameter_graph: ParameterGraph,
) -> PyResult<EffectiveLindbladPlan> {
    let n_states = required_usize(dict, "n_states")?;
    let real_dim = required_usize(dict, "real_dim")?;
    let n_grid = required_usize(dict, "n_grid")?;
    let field_grid = required_f64_array(dict, "field_grid")?;
    let excited_indices = required_usize_array(dict, "excited_indices")?;
    let ground_indices = required_usize_array(dict, "ground_indices")?;
    let sink_indices = required_usize_array(dict, "sink_indices")?;
    let field_coordinate_slot = required_usize(dict, "field_coordinate_slot")?;
    let rabi_rate_slot = required_usize(dict, "rabi_rate_slot")?;
    let detuning_slot = required_usize(dict, "detuning_slot")?;

    if field_grid.len() != n_grid {
        return Err(PyValueError::new_err("field_grid length mismatch"));
    }
    if n_grid < 2 || field_grid.windows(2).any(|pair| !(pair[1] > pair[0])) {
        return Err(PyValueError::new_err(
            "field_grid must have at least two strictly increasing points",
        ));
    }
    check_indices(&excited_indices, n_states, "excited_indices")?;
    check_indices(&ground_indices, n_states, "ground_indices")?;
    check_indices(&sink_indices, n_states, "sink_indices")?;
    check_indices(
        &[field_coordinate_slot, rabi_rate_slot, detuning_slot],
        parameter_graph.slot_names.len(),
        "parameter slot",
    )?;

    fn parse_sparse_operator(
        dict: &Bound<'_, PyDict>,
        key: &str,
        dim: usize,
        n_grid: usize,
    ) -> PyResult<SparseOperator> {
        let obj = dict
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("missing sparse operator: {key}")))?;
        let sparse_dict: &Bound<'_, PyDict> = obj.cast()?;
        let row_ptrs = required_usize_array(sparse_dict, "row_ptrs")?;
        let col_indices = required_usize_array(sparse_dict, "col_indices")?;
        let pchip_coeffs = required_f64_array(sparse_dict, "pchip_coeffs")?;
        let grid = required_f64_array(sparse_dict, "grid")?;
        let nnz: usize = required_usize(sparse_dict, "nnz")?;
        // The matvec indexes all of these without checks.
        if row_ptrs.len() != dim + 1
            || row_ptrs.first() != Some(&0)
            || row_ptrs.last() != Some(&nnz)
            || row_ptrs.windows(2).any(|pair| pair[1] < pair[0])
        {
            return Err(PyValueError::new_err(format!(
                "{key}: row_ptrs must be {} non-decreasing offsets from 0 to nnz",
                dim + 1
            )));
        }
        if col_indices.len() != nnz {
            return Err(PyValueError::new_err(format!(
                "{key}: expected {nnz} col_indices, got {}",
                col_indices.len()
            )));
        }
        check_indices(&col_indices, dim, &format!("{key} column"))?;
        let expected_coeffs = (n_grid - 1) * nnz * 4;
        if pchip_coeffs.len() != expected_coeffs {
            return Err(PyValueError::new_err(format!(
                "{key}: expected {expected_coeffs} pchip_coeffs, got {}",
                pchip_coeffs.len()
            )));
        }
        Ok(SparseOperator {
            row_ptrs,
            col_indices,
            pchip_coeffs,
            grid,
            nnz,
            dim,
            n_grid,
        })
    }

    let sparse_combined = parse_sparse_operator(dict, "sparse_combined", real_dim, n_grid)?;
    let sparse_opt = parse_sparse_operator(dict, "sparse_opt", real_dim, n_grid)?;
    let sparse_det = parse_sparse_operator(dict, "sparse_det", real_dim, n_grid)?;

    let operator_interpolation = match dict
        .get_item("operator_interpolation")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .as_deref()
    {
        Some("pchip") => OperatorInterpolation::Pchip,
        _ => OperatorInterpolation::Linear,
    };

    Ok(EffectiveLindbladPlan {
        n_states,
        real_dim,
        n_grid,
        field_grid,
        sparse_combined,
        sparse_opt,
        sparse_det,
        excited_indices,
        ground_indices,
        sink_indices,
        parameter_graph,
        field_coordinate_slot,
        rabi_rate_slot,
        detuning_slot,
        operator_interpolation,
    })
}
