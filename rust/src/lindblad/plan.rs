use crate::lindblad::blas::BlasConfig;
use crate::lindblad::eval::{
    eval_expression_into, eval_scalar_expression_into, scalar_value, CompiledExpression,
    Instruction, InstructionOp, RuntimeValue,
};
use crate::lindblad::layout::{PackedHermitianLayout, UpperTriLayout};
use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList};

#[derive(Clone, Debug)]
pub struct CompoundExpression {
    pub slot: usize,
    pub expression: CompiledExpression,
}

#[derive(Clone, Debug)]
pub struct ParameterGraph {
    pub slot_names: Vec<String>,
    pub base_values: Vec<RuntimeValue>,
    pub compounds: Vec<CompoundExpression>,
}

impl ParameterGraph {
    pub fn evaluate_into(
        &self,
        t: f64,
        slots: &mut Vec<RuntimeValue>,
        eval_stack: &mut Vec<RuntimeValue>,
        scalar_stack: &mut Vec<Complex64>,
    ) -> Result<(), String> {
        slots.clear();
        slots.extend(self.base_values.iter().cloned());
        slots.resize(self.slot_names.len(), RuntimeValue::Scalar(Complex64::ZERO));
        for compound in &self.compounds {
            slots[compound.slot] = if compound.expression.scalar_only {
                RuntimeValue::Scalar(eval_scalar_expression_into(
                    &compound.expression,
                    slots.as_slice(),
                    t,
                    &[],
                    scalar_stack,
                )?)
            } else {
                eval_expression_into(&compound.expression, slots.as_slice(), t, &[], eval_stack)?
            };
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct HamiltonianEntry {
    pub i: usize,
    pub j: usize,
    pub expression: CompiledExpression,
}

#[derive(Clone, Debug)]
pub struct BasisTerm {
    pub i: usize,
    pub j: usize,
    pub value: Complex64,
}

#[derive(Clone, Debug)]
pub struct BasisRowSegment {
    pub row: usize,
    pub start_col: usize,
    pub values: Vec<Complex64>,
}

#[derive(Clone, Debug)]
pub struct DenseRowSegment {
    pub start_col: usize,
    pub coeff_indices: Vec<usize>,
    pub values: Vec<Vec<Complex64>>,
}

#[derive(Clone, Debug)]
pub struct DenseRowPlan {
    pub row: usize,
    pub segments: Vec<DenseRowSegment>,
}

#[derive(Clone, Debug)]
pub struct DecomposedHamiltonianCoefficient {
    pub expression: CompiledExpression,
    pub basis_terms: Vec<BasisTerm>,
    pub basis_row_segments: Vec<BasisRowSegment>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HamiltonianKind {
    Entrywise,
    Decomposed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseFillMode {
    Direct,
    UpperExpand,
}

#[derive(Clone, Debug)]
pub struct HamiltonianPlan {
    pub n: usize,
    pub temps: Vec<CompiledExpression>,
    pub entries: Vec<HamiltonianEntry>,
    pub kind: HamiltonianKind,
    pub dense_fill_mode: DenseFillMode,
    pub static_matrix: Vec<Complex64>,
    pub coefficients: Vec<DecomposedHamiltonianCoefficient>,
    pub row_plans: Vec<DenseRowPlan>,
}

impl HamiltonianPlan {
    pub fn fill_into(
        &self,
        parameter_values: &[RuntimeValue],
        t: f64,
        temps: &mut Vec<RuntimeValue>,
        eval_stack: &mut Vec<RuntimeValue>,
        scalar_stack: &mut Vec<Complex64>,
        matrix: &mut [Complex64],
    ) -> Result<(), String> {
        if matrix.len() != self.n * self.n {
            return Err(format!(
                "expected {} Hamiltonian elements, got {}",
                self.n * self.n,
                matrix.len()
            ));
        }
        if self.kind == HamiltonianKind::Decomposed {
            matrix.copy_from_slice(self.static_matrix.as_slice());
            let coeff_values = if !self.row_plans.is_empty() {
                let mut values = Vec::with_capacity(self.coefficients.len());
                for coefficient in &self.coefficients {
                    let value = if coefficient.expression.scalar_only {
                        eval_scalar_expression_into(
                            &coefficient.expression,
                            parameter_values,
                            t,
                            &[],
                            scalar_stack,
                        )?
                    } else {
                        scalar_value(eval_expression_into(
                            &coefficient.expression,
                            parameter_values,
                            t,
                            &[],
                            eval_stack,
                        )?)?
                    };
                    values.push(value);
                }
                Some(values)
            } else {
                None
            };
            if let Some(coeff_values) = coeff_values.as_ref() {
                for row_plan in &self.row_plans {
                    let row_offset = row_plan.row * self.n;
                    for segment in &row_plan.segments {
                        for local_offset in 0..segment.values[0].len() {
                            let col = segment.start_col + local_offset;
                            let mut contribution = Complex64::ZERO;
                            for (coeff_pos, coeff_index) in segment.coeff_indices.iter().enumerate()
                            {
                                contribution += coeff_values[*coeff_index]
                                    * segment.values[coeff_pos][local_offset];
                            }
                            matrix[row_offset + col] += contribution;
                            if row_plan.row != col {
                                matrix[col * self.n + row_plan.row] += contribution.conj();
                            }
                        }
                    }
                }
                return Ok(());
            }
            for coefficient in &self.coefficients {
                let value = if coefficient.expression.scalar_only {
                    eval_scalar_expression_into(
                        &coefficient.expression,
                        parameter_values,
                        t,
                        &[],
                        scalar_stack,
                    )?
                } else {
                    scalar_value(eval_expression_into(
                        &coefficient.expression,
                        parameter_values,
                        t,
                        &[],
                        eval_stack,
                    )?)?
                };
                if !coefficient.basis_row_segments.is_empty() {
                    for segment in &coefficient.basis_row_segments {
                        let row_offset = segment.row * self.n;
                        for (offset, entry) in segment.values.iter().enumerate() {
                            let col = segment.start_col + offset;
                            let contribution = value * *entry;
                            matrix[row_offset + col] += contribution;
                            if segment.row != col {
                                matrix[col * self.n + segment.row] += contribution.conj();
                            }
                        }
                    }
                } else {
                    for basis_term in &coefficient.basis_terms {
                        let contribution = value * basis_term.value;
                        matrix[basis_term.i * self.n + basis_term.j] += contribution;
                        if basis_term.i != basis_term.j {
                            matrix[basis_term.j * self.n + basis_term.i] += contribution.conj();
                        }
                    }
                }
            }
            return Ok(());
        }
        matrix.fill(Complex64::ZERO);
        temps.clear();
        for temp in &self.temps {
            temps.push(if temp.scalar_only {
                RuntimeValue::Scalar(eval_scalar_expression_into(
                    temp,
                    parameter_values,
                    t,
                    temps.as_slice(),
                    scalar_stack,
                )?)
            } else {
                eval_expression_into(temp, parameter_values, t, temps.as_slice(), eval_stack)?
            });
        }
        for entry in &self.entries {
            let value = if entry.expression.scalar_only {
                eval_scalar_expression_into(
                    &entry.expression,
                    parameter_values,
                    t,
                    temps.as_slice(),
                    scalar_stack,
                )?
            } else {
                scalar_value(eval_expression_into(
                    &entry.expression,
                    parameter_values,
                    t,
                    temps.as_slice(),
                    eval_stack,
                )?)?
            };
            matrix[entry.i * self.n + entry.j] = value;
            if entry.i != entry.j {
                matrix[entry.j * self.n + entry.i] = value.conj();
            }
        }
        Ok(())
    }

    pub fn fill_upper_into(
        &self,
        parameter_values: &[RuntimeValue],
        t: f64,
        temps: &mut Vec<RuntimeValue>,
        eval_stack: &mut Vec<RuntimeValue>,
        scalar_stack: &mut Vec<Complex64>,
        upper_layout: &UpperTriLayout,
        upper: &mut [Complex64],
    ) -> Result<(), String> {
        if upper_layout.n != self.n {
            return Err(format!(
                "upper-triangle layout dimension {} does not match Hamiltonian dimension {}",
                upper_layout.n, self.n
            ));
        }
        upper_layout.clear(upper)?;
        if self.kind == HamiltonianKind::Decomposed {
            for i in 0..self.n {
                for j in i..self.n {
                    let dense_index = i * self.n + j;
                    upper[upper_layout.index(i, j)?] = self.static_matrix[dense_index];
                }
            }
            for coefficient in &self.coefficients {
                let value = if coefficient.expression.scalar_only {
                    eval_scalar_expression_into(
                        &coefficient.expression,
                        parameter_values,
                        t,
                        &[],
                        scalar_stack,
                    )?
                } else {
                    scalar_value(eval_expression_into(
                        &coefficient.expression,
                        parameter_values,
                        t,
                        &[],
                        eval_stack,
                    )?)?
                };
                if !coefficient.basis_row_segments.is_empty() {
                    for segment in &coefficient.basis_row_segments {
                        for (offset, entry) in segment.values.iter().enumerate() {
                            let col = segment.start_col + offset;
                            let index = upper_layout.index(segment.row, col)?;
                            upper[index] += value * *entry;
                        }
                    }
                } else {
                    for basis_term in &coefficient.basis_terms {
                        let index = upper_layout.index(basis_term.i, basis_term.j)?;
                        upper[index] += value * basis_term.value;
                    }
                }
            }
            return Ok(());
        }

        temps.clear();
        for temp in &self.temps {
            temps.push(if temp.scalar_only {
                RuntimeValue::Scalar(eval_scalar_expression_into(
                    temp,
                    parameter_values,
                    t,
                    temps.as_slice(),
                    scalar_stack,
                )?)
            } else {
                eval_expression_into(temp, parameter_values, t, temps.as_slice(), eval_stack)?
            });
        }
        for entry in &self.entries {
            let value = if entry.expression.scalar_only {
                eval_scalar_expression_into(
                    &entry.expression,
                    parameter_values,
                    t,
                    temps.as_slice(),
                    scalar_stack,
                )?
            } else {
                scalar_value(eval_expression_into(
                    &entry.expression,
                    parameter_values,
                    t,
                    temps.as_slice(),
                    eval_stack,
                )?)?
            };
            upper[upper_layout.index(entry.i, entry.j)?] = value;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct HermitianSparsePattern {
    pub n: usize,
    pub nnz: usize,
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub col_ptrs: Vec<usize>,
    pub row_indices_csc: Vec<usize>,
    pub csc_to_csr: Vec<usize>,
}

impl HermitianSparsePattern {
    pub fn from_hamiltonian_plan(plan: &HamiltonianPlan) -> Self {
        let n = plan.n;
        let mut mask = vec![false; n * n];

        if plan.kind == HamiltonianKind::Decomposed {
            for i in 0..n {
                for j in i..n {
                    if plan.static_matrix[i * n + j] != Complex64::ZERO {
                        mask[i * n + j] = true;
                    }
                }
            }
            for coeff in &plan.coefficients {
                if !coeff.basis_row_segments.is_empty() {
                    for seg in &coeff.basis_row_segments {
                        for (offset, _) in seg.values.iter().enumerate() {
                            let col = seg.start_col + offset;
                            let (r, c) = if seg.row <= col {
                                (seg.row, col)
                            } else {
                                (col, seg.row)
                            };
                            mask[r * n + c] = true;
                        }
                    }
                } else {
                    for bt in &coeff.basis_terms {
                        let (r, c) = if bt.i <= bt.j {
                            (bt.i, bt.j)
                        } else {
                            (bt.j, bt.i)
                        };
                        mask[r * n + c] = true;
                    }
                }
            }
        } else {
            for entry in &plan.entries {
                let (r, c) = if entry.i <= entry.j {
                    (entry.i, entry.j)
                } else {
                    (entry.j, entry.i)
                };
                mask[r * n + c] = true;
            }
        }

        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::new();
        for i in 0..n {
            row_ptrs.push(col_indices.len());
            for j in i..n {
                if mask[i * n + j] {
                    col_indices.push(j);
                }
            }
        }
        row_ptrs.push(col_indices.len());
        let nnz = col_indices.len();

        let mut col_ptrs = vec![0usize; n + 1];
        let mut row_indices_csc = Vec::with_capacity(nnz);
        let mut csc_to_csr = Vec::with_capacity(nnz);

        let mut counts = vec![0usize; n];
        for i in 0..n {
            for ptr in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[ptr];
                if j > i {
                    counts[j] += 1;
                }
            }
        }
        let mut offset = 0usize;
        for j in 0..n {
            col_ptrs[j] = offset;
            offset += counts[j];
        }
        col_ptrs[n] = offset;

        row_indices_csc.resize(offset, 0);
        csc_to_csr.resize(offset, 0);
        let mut write_pos = col_ptrs.clone();
        for i in 0..n {
            for ptr in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[ptr];
                if j > i {
                    let pos = write_pos[j];
                    row_indices_csc[pos] = i;
                    csc_to_csr[pos] = ptr;
                    write_pos[j] += 1;
                }
            }
        }

        Self {
            n,
            nnz,
            row_ptrs,
            col_indices,
            col_ptrs,
            row_indices_csc,
            csc_to_csr,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructuredJump {
    pub target: usize,
    pub source: usize,
    pub rate: f64,
}

#[derive(Clone, Debug)]
pub struct IncomingTransfer {
    pub source: usize,
    pub rate: f64,
}

#[pyclass(module = "centrex_tlf.centrex_tlf_rust")]
#[derive(Clone)]
pub struct PreparedLindbladPlan {
    pub layout: PackedHermitianLayout,
    pub parameter_graph: ParameterGraph,
    pub hamiltonian_plan: HamiltonianPlan,
    pub dense_c_array: Vec<Complex64>,
    pub dense_cdagger_c: Vec<Complex64>,
    pub n_collapse: usize,
    pub structured_jumps: Vec<StructuredJump>,
    pub source_decay_rates: Vec<f64>,
    pub incoming_transfers_by_target: Vec<Vec<IncomingTransfer>>,
    pub blas_config: Option<BlasConfig>,
    pub hamiltonian_sparse_pattern: HermitianSparsePattern,
    pub is_time_dependent: bool,
}

impl PreparedLindbladPlan {
    pub fn n_states(&self) -> usize {
        self.layout.n
    }

    pub fn evaluate_hamiltonian(&self, t: f64) -> Result<Vec<Complex64>, String> {
        let mut parameter_values = Vec::new();
        let mut temps = Vec::new();
        let mut eval_stack = Vec::new();
        let mut scalar_stack = Vec::new();
        let mut matrix = vec![Complex64::ZERO; self.n_states() * self.n_states()];
        self.evaluate_hamiltonian_into(
            t,
            &mut parameter_values,
            &mut temps,
            &mut eval_stack,
            &mut scalar_stack,
            matrix.as_mut_slice(),
        )?;
        Ok(matrix)
    }

    pub fn evaluate_hamiltonian_into(
        &self,
        t: f64,
        parameter_values: &mut Vec<RuntimeValue>,
        temps: &mut Vec<RuntimeValue>,
        eval_stack: &mut Vec<RuntimeValue>,
        scalar_stack: &mut Vec<Complex64>,
        matrix: &mut [Complex64],
    ) -> Result<(), String> {
        self.parameter_graph
            .evaluate_into(t, parameter_values, eval_stack, scalar_stack)?;
        self.hamiltonian_plan.fill_into(
            parameter_values.as_slice(),
            t,
            temps,
            eval_stack,
            scalar_stack,
            matrix,
        )
    }
}

fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyErr::new::<PyKeyError, _>(format!("missing key {:?}", key)))
}

fn parse_runtime_value(obj: &Bound<'_, PyAny>) -> PyResult<RuntimeValue> {
    let dict: &Bound<'_, PyDict> = obj.cast()?;
    let kind: String = required_item(dict, "kind")?.extract()?;
    match kind.as_str() {
        "scalar" => Ok(RuntimeValue::Scalar(Complex64::new(
            required_item(dict, "re")?.extract()?,
            required_item(dict, "im")?.extract()?,
        ))),
        "tuple" => {
            let items_any = required_item(dict, "items")?;
            let items: &Bound<'_, PyList> = items_any.cast()?;
            let mut values = Vec::with_capacity(items.len());
            for item in items.iter() {
                match parse_runtime_value(&item)? {
                    RuntimeValue::Scalar(value) => values.push(value),
                    RuntimeValue::Tuple(_) => {
                        return Err(PyErr::new::<PyTypeError, _>(
                            "nested tuple runtime values are not supported",
                        ))
                    }
                }
            }
            Ok(RuntimeValue::Tuple(values))
        }
        _ => Err(PyErr::new::<PyValueError, _>(format!(
            "unknown runtime value kind {kind}"
        ))),
    }
}

fn parse_instruction(obj: &Bound<'_, PyAny>) -> PyResult<Instruction> {
    let dict: &Bound<'_, PyDict> = obj.cast()?;
    let op = InstructionOp::from_i64(required_item(dict, "op")?.extract()?)
        .map_err(PyValueError::new_err)?;
    let index = match dict.get_item("index")? {
        Some(value) => value.extract()?,
        None => 0_usize,
    };
    let argc = match dict.get_item("argc")? {
        Some(value) => value.extract()?,
        None => 0_usize,
    };
    let function = match dict.get_item("function")? {
        Some(value) => value.extract()?,
        None => 0_i64,
    };
    let re = match dict.get_item("re")? {
        Some(value) => value.extract()?,
        None => 0.0_f64,
    };
    let im = match dict.get_item("im")? {
        Some(value) => value.extract()?,
        None => 0.0_f64,
    };
    Ok(Instruction {
        op,
        index,
        argc,
        function,
        value: Complex64::new(re, im),
    })
}

fn parse_expression(obj: &Bound<'_, PyAny>) -> PyResult<CompiledExpression> {
    let dict: &Bound<'_, PyDict> = obj.cast()?;
    let instructions_any = required_item(dict, "instructions")?;
    let instructions_list: &Bound<'_, PyList> = instructions_any.cast()?;
    let mut instructions = Vec::with_capacity(instructions_list.len());
    for instruction in instructions_list.iter() {
        instructions.push(parse_instruction(&instruction)?);
    }
    Ok(CompiledExpression {
        instructions,
        scalar_only: dict
            .get_item("scalar_only")?
            .map(|value| value.extract())
            .transpose()?
            .unwrap_or(false),
        output_is_tuple: dict
            .get_item("output_is_tuple")?
            .map(|value| value.extract())
            .transpose()?
            .unwrap_or(false),
    })
}

fn parse_parameter_graph(obj: &Bound<'_, PyAny>) -> PyResult<ParameterGraph> {
    let dict: &Bound<'_, PyDict> = obj.cast()?;
    let slot_names: Vec<String> = required_item(dict, "slot_names")?.extract()?;

    let base_values_any = required_item(dict, "base_values")?;
    let base_values_list: &Bound<'_, PyList> = base_values_any.cast()?;
    let mut base_values = Vec::with_capacity(base_values_list.len());
    for value in base_values_list.iter() {
        base_values.push(parse_runtime_value(&value)?);
    }

    let compounds_any = required_item(dict, "compounds")?;
    let compounds_list: &Bound<'_, PyList> = compounds_any.cast()?;
    let mut compounds = Vec::with_capacity(compounds_list.len());
    for compound in compounds_list.iter() {
        let compound_dict: &Bound<'_, PyDict> = compound.cast()?;
        compounds.push(CompoundExpression {
            slot: required_item(compound_dict, "slot")?.extract()?,
            expression: parse_expression(&required_item(compound_dict, "expression")?)?,
        });
    }

    Ok(ParameterGraph {
        slot_names,
        base_values,
        compounds,
    })
}

fn parse_hamiltonian_plan(obj: &Bound<'_, PyAny>) -> PyResult<HamiltonianPlan> {
    let dict: &Bound<'_, PyDict> = obj.cast()?;
    let n: usize = required_item(dict, "n")?.extract()?;
    let kind_str: String = dict
        .get_item("kind")?
        .map(|value| value.extract())
        .transpose()?
        .unwrap_or_else(|| "entrywise".to_string());
    let kind = match kind_str.as_str() {
        "decomposed" => HamiltonianKind::Decomposed,
        "entrywise" => HamiltonianKind::Entrywise,
        other => {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "unknown hamiltonian kind: {other}"
            )))
        }
    };

    if kind == HamiltonianKind::Decomposed {
        let static_matrix_any = required_item(dict, "static_matrix")?;
        let static_matrix: PyReadonlyArrayDyn<'_, Complex64> = static_matrix_any.extract()?;
        let static_values = static_matrix
            .as_slice()
            .map_err(PyValueError::new_err)?
            .to_vec();
        let coefficients_any = required_item(dict, "coefficients")?;
        let coefficients_list: &Bound<'_, PyList> = coefficients_any.cast()?;
        let mut coefficients = Vec::with_capacity(coefficients_list.len());
        for coefficient in coefficients_list.iter() {
            let coefficient_dict: &Bound<'_, PyDict> = coefficient.cast()?;
            let basis_terms_any = required_item(coefficient_dict, "basis_terms")?;
            let basis_terms_list: &Bound<'_, PyList> = basis_terms_any.cast()?;
            let mut basis_terms = Vec::with_capacity(basis_terms_list.len());
            for term in basis_terms_list.iter() {
                let term_dict: &Bound<'_, PyDict> = term.cast()?;
                basis_terms.push(BasisTerm {
                    i: required_item(term_dict, "i")?.extract()?,
                    j: required_item(term_dict, "j")?.extract()?,
                    value: Complex64::new(
                        required_item(term_dict, "re")?.extract()?,
                        required_item(term_dict, "im")?.extract()?,
                    ),
                });
            }
            let basis_row_segments = match coefficient_dict.get_item("basis_row_segments")? {
                Some(value) => {
                    let segment_list: &Bound<'_, PyList> = value.cast()?;
                    let mut segments = Vec::with_capacity(segment_list.len());
                    for segment in segment_list.iter() {
                        let segment_dict: &Bound<'_, PyDict> = segment.cast()?;
                        let values_re: Vec<f64> =
                            required_item(segment_dict, "values_re")?.extract()?;
                        let values_im: Vec<f64> =
                            required_item(segment_dict, "values_im")?.extract()?;
                        if values_re.len() != values_im.len() {
                            return Err(PyErr::new::<PyValueError, _>(
                                "basis_row_segments values_re and values_im must match",
                            ));
                        }
                        let mut values = Vec::with_capacity(values_re.len());
                        for (re, im) in values_re.into_iter().zip(values_im.into_iter()) {
                            values.push(Complex64::new(re, im));
                        }
                        segments.push(BasisRowSegment {
                            row: required_item(segment_dict, "row")?.extract()?,
                            start_col: required_item(segment_dict, "start_col")?.extract()?,
                            values,
                        });
                    }
                    segments
                }
                None => Vec::new(),
            };
            coefficients.push(DecomposedHamiltonianCoefficient {
                expression: parse_expression(&required_item(coefficient_dict, "expression")?)?,
                basis_terms,
                basis_row_segments,
            });
        }
        return Ok(HamiltonianPlan {
            n,
            temps: Vec::new(),
            entries: Vec::new(),
            kind,
            dense_fill_mode: {
                let s: String = dict
                    .get_item("dense_fill_mode")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or_else(|| "direct".to_string());
                match s.as_str() {
                    "upper_expand" => DenseFillMode::UpperExpand,
                    _ => DenseFillMode::Direct,
                }
            },
            static_matrix: static_values,
            coefficients,
            row_plans: {
                let row_plans = match dict.get_item("row_plans")? {
                    Some(value) => {
                        let row_plan_list: &Bound<'_, PyList> = value.cast()?;
                        let mut row_plans = Vec::with_capacity(row_plan_list.len());
                        for row_plan in row_plan_list.iter() {
                            let row_plan_dict: &Bound<'_, PyDict> = row_plan.cast()?;
                            let segments_any = required_item(row_plan_dict, "segments")?;
                            let segments_list: &Bound<'_, PyList> = segments_any.cast()?;
                            let mut segments = Vec::with_capacity(segments_list.len());
                            for segment in segments_list.iter() {
                                let segment_dict: &Bound<'_, PyDict> = segment.cast()?;
                                let coeff_indices: Vec<usize> =
                                    required_item(segment_dict, "coeff_indices")?.extract()?;
                                let values_re: Vec<Vec<f64>> =
                                    required_item(segment_dict, "values_re")?.extract()?;
                                let values_im: Vec<Vec<f64>> =
                                    required_item(segment_dict, "values_im")?.extract()?;
                                if values_re.len() != values_im.len()
                                    || values_re.len() != coeff_indices.len()
                                {
                                    return Err(PyErr::new::<PyValueError, _>(
                                        "row_plans values must align with coeff_indices",
                                    ));
                                }
                                let mut values = Vec::with_capacity(values_re.len());
                                for (row_re, row_im) in
                                    values_re.into_iter().zip(values_im.into_iter())
                                {
                                    if row_re.len() != row_im.len() {
                                        return Err(PyErr::new::<PyValueError, _>(
                                            "row_plans value rows must have matching real/imag lengths",
                                        ));
                                    }
                                    let mut entry_values = Vec::with_capacity(row_re.len());
                                    for (re, im) in row_re.into_iter().zip(row_im.into_iter()) {
                                        entry_values.push(Complex64::new(re, im));
                                    }
                                    values.push(entry_values);
                                }
                                segments.push(DenseRowSegment {
                                    start_col: required_item(segment_dict, "start_col")?
                                        .extract()?,
                                    coeff_indices,
                                    values,
                                });
                            }
                            row_plans.push(DenseRowPlan {
                                row: required_item(row_plan_dict, "row")?.extract()?,
                                segments,
                            });
                        }
                        row_plans
                    }
                    None => Vec::new(),
                };
                row_plans
            },
        });
    }

    let temps_any = required_item(dict, "temps")?;
    let temps_list: &Bound<'_, PyList> = temps_any.cast()?;
    let mut temps = Vec::with_capacity(temps_list.len());
    for temp in temps_list.iter() {
        temps.push(parse_expression(&temp)?);
    }

    let entries_any = required_item(dict, "entries")?;
    let entries_list: &Bound<'_, PyList> = entries_any.cast()?;
    let mut entries = Vec::with_capacity(entries_list.len());
    for entry in entries_list.iter() {
        let entry_dict: &Bound<'_, PyDict> = entry.cast()?;
        entries.push(HamiltonianEntry {
            i: required_item(entry_dict, "i")?.extract()?,
            j: required_item(entry_dict, "j")?.extract()?,
            expression: parse_expression(&required_item(entry_dict, "expression")?)?,
        });
    }

    Ok(HamiltonianPlan {
        n,
        temps,
        entries,
        kind,
        dense_fill_mode: {
            let s: String = dict
                .get_item("dense_fill_mode")?
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or_else(|| "direct".to_string());
            match s.as_str() {
                "upper_expand" => DenseFillMode::UpperExpand,
                _ => DenseFillMode::Direct,
            }
        },
        static_matrix: vec![Complex64::ZERO; n * n],
        coefficients: Vec::new(),
        row_plans: Vec::new(),
    })
}

fn parse_blas_config(obj: &Bound<'_, PyAny>) -> PyResult<BlasConfig> {
    let dict: &Bound<'_, PyDict> = obj.cast()?;
    Ok(BlasConfig {
        library_path: required_item(dict, "library_path")?.extract()?,
        zher2k_symbol: required_item(dict, "zher2k_symbol")?.extract()?,
    })
}

pub fn parse_plan_payload(payload: &Bound<'_, PyAny>) -> PyResult<PreparedLindbladPlan> {
    let dict: &Bound<'_, PyDict> = payload.cast()?;
    let n_states: usize = required_item(dict, "n_states")?.extract()?;
    let layout = PackedHermitianLayout::new(n_states).map_err(PyValueError::new_err)?;

    let parameter_graph = parse_parameter_graph(&required_item(dict, "parameter_graph")?)?;
    let hamiltonian_plan = parse_hamiltonian_plan(&required_item(dict, "hamiltonian_plan")?)?;

    let source_decay_array: PyReadonlyArray1<'_, f64> =
        required_item(dict, "source_decay_rates")?.extract()?;
    let source_decay_rates = source_decay_array
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();

    let dense_c_array_any = required_item(dict, "dense_c_array")?;
    let dense_c_array: PyReadonlyArrayDyn<'_, Complex64> = dense_c_array_any.extract()?;
    let dense_shape = dense_c_array.shape().to_vec();
    if dense_shape.len() != 3 || dense_shape[1] != n_states || dense_shape[2] != n_states {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "dense_c_array must have shape (n_collapse, {n_states}, {n_states})"
        )));
    }
    let dense_values = dense_c_array
        .as_slice()
        .map_err(PyValueError::new_err)?
        .to_vec();

    let n_collapse_count = dense_shape[0];
    let collapse_size = n_states * n_states;
    let zero = Complex64::ZERO;
    let mut dense_cdagger_c = vec![zero; n_collapse_count * collapse_size];
    for collapse_idx in 0..n_collapse_count {
        let base = collapse_idx * collapse_size;
        let c = &dense_values[base..(base + collapse_size)];
        let cdc = &mut dense_cdagger_c[base..(base + collapse_size)];
        for i in 0..n_states {
            for j in 0..n_states {
                let mut value = zero;
                for alpha in 0..n_states {
                    value += c[alpha * n_states + i].conj() * c[alpha * n_states + j];
                }
                cdc[i * n_states + j] = value;
            }
        }
    }

    let structured_jumps_any = required_item(dict, "structured_jumps")?;
    let structured_jumps_list: &Bound<'_, PyList> = structured_jumps_any.cast()?;
    let mut structured_jumps = Vec::with_capacity(structured_jumps_list.len());
    for jump in structured_jumps_list.iter() {
        let jump_dict: &Bound<'_, PyDict> = jump.cast()?;
        structured_jumps.push(StructuredJump {
            target: required_item(jump_dict, "target")?.extract()?,
            source: required_item(jump_dict, "source")?.extract()?,
            rate: required_item(jump_dict, "rate")?.extract()?,
        });
    }

    let incoming_by_target_any = required_item(dict, "incoming_transfers_by_target")?;
    let incoming_by_target_list: &Bound<'_, PyList> = incoming_by_target_any.cast()?;
    let mut incoming_transfers_by_target = Vec::with_capacity(incoming_by_target_list.len());
    for target_entries in incoming_by_target_list.iter() {
        let target_list: &Bound<'_, PyList> = target_entries.cast()?;
        let mut transfers = Vec::with_capacity(target_list.len());
        for entry in target_list.iter() {
            let entry_dict: &Bound<'_, PyDict> = entry.cast()?;
            transfers.push(IncomingTransfer {
                source: required_item(entry_dict, "source")?.extract()?,
                rate: required_item(entry_dict, "rate")?.extract()?,
            });
        }
        incoming_transfers_by_target.push(transfers);
    }

    let blas_config = match dict.get_item("blas_config")? {
        Some(value) if !value.is_none() => Some(parse_blas_config(&value)?),
        _ => None,
    };

    let hamiltonian_sparse_pattern =
        HermitianSparsePattern::from_hamiltonian_plan(&hamiltonian_plan);

    let is_time_dependent = {
        let expr_uses_time = |expr: &CompiledExpression| -> bool {
            expr.instructions
                .iter()
                .any(|instr| instr.op == InstructionOp::Time)
        };
        let params_use_time = parameter_graph
            .compounds
            .iter()
            .any(|c| expr_uses_time(&c.expression));
        let ham_uses_time = if hamiltonian_plan.kind == HamiltonianKind::Decomposed {
            hamiltonian_plan
                .coefficients
                .iter()
                .any(|c| expr_uses_time(&c.expression))
        } else {
            hamiltonian_plan.temps.iter().any(|t| expr_uses_time(t))
                || hamiltonian_plan
                    .entries
                    .iter()
                    .any(|e| expr_uses_time(&e.expression))
        };
        params_use_time || ham_uses_time
    };

    Ok(PreparedLindbladPlan {
        layout,
        parameter_graph,
        hamiltonian_plan,
        dense_c_array: dense_values,
        dense_cdagger_c,
        n_collapse: n_collapse_count,
        structured_jumps,
        source_decay_rates,
        incoming_transfers_by_target,
        blas_config,
        hamiltonian_sparse_pattern,
        is_time_dependent,
    })
}
