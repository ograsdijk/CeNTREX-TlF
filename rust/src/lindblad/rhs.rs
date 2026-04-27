use crate::lindblad::blas::commutator_her2k;
use crate::lindblad::eval::RuntimeValue;
use crate::lindblad::layout::UpperTriLayout;
use crate::lindblad::plan::{ExpandedSparseRhsPlan, HermitianSparsePattern, PreparedLindbladPlan};
use num_complex::Complex64;
use std::time::Instant;

const NO_PACKED_IMAG_INDEX: usize = usize::MAX;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    ReferenceDense,
    StructuredBlas,
    StructuredUpper,
    ExpandedSparse,
}

impl ExecutionMode {
    pub fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "reference" => Ok(Self::ReferenceDense),
            "structured" => Ok(Self::StructuredBlas),
            "structured_upper" => Ok(Self::StructuredUpper),
            "expanded_sparse" => Ok(Self::ExpandedSparse),
            _ => Err(format!("unsupported execution mode {:?}", value)),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct PackedUpperIndex {
    re: usize,
    im: usize,
}

#[derive(Copy, Clone, Debug)]
struct PackedTermInput {
    re: usize,
    im: usize,
    imag_sign: f64,
}

fn build_upper_to_packed_indices(n: usize, upper_layout: &UpperTriLayout) -> Vec<PackedUpperIndex> {
    let mut indices = vec![
        PackedUpperIndex {
            re: 0,
            im: NO_PACKED_IMAG_INDEX,
        };
        upper_layout.len()
    ];
    for i in 0..n {
        indices[upper_layout.index_unchecked(i, i)] = PackedUpperIndex {
            re: i,
            im: NO_PACKED_IMAG_INDEX,
        };
    }
    let mut cursor = n;
    for i in 0..n {
        for j in (i + 1)..n {
            indices[upper_layout.index_unchecked(i, j)] = PackedUpperIndex {
                re: cursor,
                im: cursor + 1,
            };
            cursor += 2;
        }
    }
    indices
}

fn build_expanded_packed_inputs(
    plan: &PreparedLindbladPlan,
    upper_to_packed: &[PackedUpperIndex],
) -> Vec<PackedTermInput> {
    let Some(rhs_plan) = plan.expanded_rhs_plan.as_ref() else {
        return Vec::new();
    };
    rhs_plan
        .terms
        .iter()
        .map(|term| {
            let index = upper_to_packed[term.input];
            if index.im == NO_PACKED_IMAG_INDEX {
                PackedTermInput {
                    re: index.re,
                    im: index.re,
                    imag_sign: 0.0,
                }
            } else {
                PackedTermInput {
                    re: index.re,
                    im: index.im,
                    imag_sign: if term.input_conj { -1.0 } else { 1.0 },
                }
            }
        })
        .collect()
}

#[inline(always)]
fn write_packed_upper_parts(
    packed: &mut [f64],
    upper_to_packed: &[PackedUpperIndex],
    upper_index: usize,
    re: f64,
    im: f64,
) {
    let index = upper_to_packed[upper_index];
    packed[index.re] = re;
    if index.im != NO_PACKED_IMAG_INDEX {
        packed[index.im] = im;
    }
}

fn matmul_into(a: &[Complex64], b: &[Complex64], out: &mut [Complex64], n: usize) {
    out.fill(Complex64::ZERO);
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == Complex64::ZERO {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
}

fn add_commutator_dense(
    hamiltonian: &[Complex64],
    rho: &[Complex64],
    out: &mut [Complex64],
    scratch: &mut [Complex64],
    n: usize,
) {
    matmul_into(hamiltonian, rho, out, n);
    matmul_into(rho, hamiltonian, scratch, n);
    let neg_i = -Complex64::I;
    for idx in 0..out.len() {
        out[idx] = neg_i * (out[idx] - scratch[idx]);
    }
}

fn add_commutator(
    plan: &PreparedLindbladPlan,
    hamiltonian: &[Complex64],
    rho: &[Complex64],
    out: &mut [Complex64],
    scratch: &mut [Complex64],
    n: usize,
) -> Result<(), String> {
    if let Some(config) = &plan.blas_config {
        commutator_her2k(config, hamiltonian, rho, out, n)
    } else {
        add_commutator_dense(hamiltonian, rho, out, scratch, n);
        Ok(())
    }
}

fn fill_sparse_h_values(
    pattern: &HermitianSparsePattern,
    upper_layout: &UpperTriLayout,
    hamiltonian_upper: &[Complex64],
    h_sparse: &mut [Complex64],
) {
    let n = pattern.n;
    for i in 0..n {
        for ptr in pattern.row_ptrs[i]..pattern.row_ptrs[i + 1] {
            let j = pattern.col_indices[ptr];
            h_sparse[ptr] = hamiltonian_upper[upper_layout.index_unchecked(i, j)];
        }
    }
}

#[inline(always)]
fn get_rho_upper(
    upper_layout: &UpperTriLayout,
    rho_upper: &[Complex64],
    i: usize,
    j: usize,
) -> Complex64 {
    if i <= j {
        rho_upper[upper_layout.index_unchecked(i, j)]
    } else {
        rho_upper[upper_layout.index_unchecked(j, i)].conj()
    }
}

fn add_commutator_sparse_upper(
    pattern: &HermitianSparsePattern,
    h_sparse: &[Complex64],
    upper_layout: &UpperTriLayout,
    rho_upper: &[Complex64],
    drho_upper: &mut [Complex64],
) {
    let n = pattern.n;
    let neg_i = -Complex64::I;

    for i in 0..n {
        for j in i..n {
            let mut hrho = Complex64::ZERO;
            let mut rhoh = Complex64::ZERO;

            // H[i,k] nonzeros from CSR (upper triangle: k >= i)
            for ptr in pattern.row_ptrs[i]..pattern.row_ptrs[i + 1] {
                let k = pattern.col_indices[ptr];
                let h_ik = h_sparse[ptr];
                hrho += h_ik * get_rho_upper(upper_layout, rho_upper, k, j);
            }

            // H[i,k] nonzeros from lower triangle: H[i,k] = conj(H[k,i]) for k < i
            for csc_ptr in pattern.col_ptrs[i]..pattern.col_ptrs[i + 1] {
                let k = pattern.row_indices_csc[csc_ptr];
                let h_ik = h_sparse[pattern.csc_to_csr[csc_ptr]].conj();
                hrho += h_ik * get_rho_upper(upper_layout, rho_upper, k, j);
            }

            // H[k,j] nonzeros from CSR (upper triangle: k <= j, stored as row k)
            // We need column j of H. From CSC: rows k < j with H[k,j] nonzero
            for csc_ptr in pattern.col_ptrs[j]..pattern.col_ptrs[j + 1] {
                let k = pattern.row_indices_csc[csc_ptr];
                let h_kj = h_sparse[pattern.csc_to_csr[csc_ptr]];
                rhoh += get_rho_upper(upper_layout, rho_upper, i, k) * h_kj;
            }

            // H[k,j] nonzeros where k >= j (diagonal + lower): H[k,j] = conj(H[j,k]) for k > j
            // From CSR row j: columns k >= j where H[j,k] nonzero → H[k,j] = conj(H[j,k])
            for ptr in pattern.row_ptrs[j]..pattern.row_ptrs[j + 1] {
                let k = pattern.col_indices[ptr];
                let h_kj = if k == j {
                    h_sparse[ptr]
                } else {
                    h_sparse[ptr].conj()
                };
                rhoh += get_rho_upper(upper_layout, rho_upper, i, k) * h_kj;
            }

            drho_upper[upper_layout.index_unchecked(i, j)] = neg_i * (hrho - rhoh);
        }
    }
}

fn add_expanded_sparse_rhs(
    rhs_plan: &ExpandedSparseRhsPlan,
    term_values: &[Complex64],
    rho_upper: &[Complex64],
    out_upper: &mut [Complex64],
) -> Result<(), String> {
    if rhs_plan.output_ptrs.len() != out_upper.len() + 1 {
        return Err(format!(
            "expanded RHS output_ptrs length {} does not match output length {}",
            rhs_plan.output_ptrs.len(),
            out_upper.len()
        ));
    }
    if rho_upper.len() != out_upper.len() {
        return Err(format!(
            "expanded RHS expected matching input/output upper lengths, got {} and {}",
            rho_upper.len(),
            out_upper.len()
        ));
    }
    if term_values.len() != rhs_plan.terms.len() {
        return Err(format!(
            "expanded RHS expected {} cached term values, got {}",
            rhs_plan.terms.len(),
            term_values.len()
        ));
    }
    for (output_index, out) in out_upper.iter_mut().enumerate() {
        let mut acc = Complex64::ZERO;
        for term_index in rhs_plan.output_ptrs[output_index]..rhs_plan.output_ptrs[output_index + 1]
        {
            let term = &rhs_plan.terms[term_index];
            let mut value = rho_upper[term.input];
            if term.input_conj {
                value = value.conj();
            }
            acc += term_values[term_index] * value;
        }
        *out = acc;
    }
    Ok(())
}

fn add_expanded_sparse_rhs_packed(
    rhs_plan: &ExpandedSparseRhsPlan,
    term_values: &[Complex64],
    upper_to_packed: &[PackedUpperIndex],
    packed_inputs: &[PackedTermInput],
    packed_state: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    if rhs_plan.output_ptrs.len() != upper_to_packed.len() + 1 {
        return Err(format!(
            "expanded RHS output_ptrs length {} does not match packed upper length {}",
            rhs_plan.output_ptrs.len(),
            upper_to_packed.len()
        ));
    }
    if packed_state.len() != out.len() {
        return Err(format!(
            "expanded RHS expected matching packed input/output lengths, got {} and {}",
            packed_state.len(),
            out.len()
        ));
    }
    if term_values.len() != rhs_plan.terms.len() {
        return Err(format!(
            "expanded RHS expected {} cached term values, got {}",
            rhs_plan.terms.len(),
            term_values.len()
        ));
    }
    if packed_inputs.len() != rhs_plan.terms.len() {
        return Err(format!(
            "expanded RHS expected {} packed term inputs, got {}",
            rhs_plan.terms.len(),
            packed_inputs.len()
        ));
    }
    for (output_index, window) in rhs_plan.output_ptrs.windows(2).enumerate() {
        let mut acc_re = 0.0_f64;
        let mut acc_im = 0.0_f64;
        for term_index in window[0]..window[1] {
            let input = packed_inputs[term_index];
            let coefficient = term_values[term_index];
            let value_re = packed_state[input.re];
            let value_im = if input.imag_sign == 0.0 {
                0.0
            } else {
                input.imag_sign * packed_state[input.im]
            };
            acc_re += coefficient.re * value_re - coefficient.im * value_im;
            acc_im += coefficient.re * value_im + coefficient.im * value_re;
        }
        write_packed_upper_parts(out, upper_to_packed, output_index, acc_re, acc_im);
    }
    Ok(())
}

fn fill_expanded_sparse_term_values(
    rhs_plan: &ExpandedSparseRhsPlan,
    coeff_values: &[Complex64],
    out: &mut Vec<Complex64>,
) -> Result<(), String> {
    out.resize(rhs_plan.terms.len(), Complex64::ZERO);
    for (idx, term) in rhs_plan.terms.iter().enumerate() {
        let mut coefficient = Complex64::ONE;
        if let Some(coeff_index) = term.coeff_index {
            if coeff_index >= coeff_values.len() {
                return Err(format!(
                    "expanded RHS coefficient index {coeff_index} out of bounds for {} values",
                    coeff_values.len()
                ));
            }
            coefficient = coeff_values[coeff_index];
            if term.coeff_conj {
                coefficient = coefficient.conj();
            }
        }
        out[idx] = term.factor * coefficient;
    }
    Ok(())
}

fn add_dense_dissipator(plan: &PreparedLindbladPlan, rho: &[Complex64], out: &mut [Complex64]) {
    let n = plan.n_states();
    let collapse_size = n * n;
    let zero = Complex64::ZERO;
    for collapse_idx in 0..plan.n_collapse {
        let base = collapse_idx * collapse_size;
        let collapse = &plan.dense_c_array[base..(base + collapse_size)];
        let cdagger_c = &plan.dense_cdagger_c[base..(base + collapse_size)];

        for i in 0..n {
            for j in 0..n {
                let mut value = zero;
                for alpha in 0..n {
                    let c_ia = collapse[i * n + alpha];
                    if c_ia == zero {
                        continue;
                    }
                    for beta in 0..n {
                        value += c_ia * rho[alpha * n + beta] * collapse[j * n + beta].conj();
                    }
                }
                out[i * n + j] += value;
            }
        }

        for i in 0..n {
            for j in 0..n {
                let mut left = zero;
                let mut right = zero;
                for alpha in 0..n {
                    left += cdagger_c[i * n + alpha] * rho[alpha * n + j];
                    right += rho[i * n + alpha] * cdagger_c[alpha * n + j];
                }
                out[i * n + j] -= 0.5 * (left + right);
            }
        }
    }
}

fn add_structured_dissipator(
    plan: &PreparedLindbladPlan,
    rho: &[Complex64],
    out: &mut [Complex64],
) {
    let n = plan.n_states();
    for source in 0..n {
        let rate_source = plan.source_decay_rates[source];
        out[source * n + source] -= rate_source * rho[source * n + source];
        for col in (source + 1)..n {
            let rate = 0.5 * (rate_source + plan.source_decay_rates[col]);
            out[source * n + col] -= rate * rho[source * n + col];
        }
    }
    for (target, incoming) in plan.incoming_transfers_by_target.iter().enumerate() {
        let target_diag = target * n + target;
        for transfer in incoming {
            out[target_diag] += transfer.rate * rho[transfer.source * n + transfer.source];
        }
    }
}

fn add_structured_dissipator_upper(
    plan: &PreparedLindbladPlan,
    upper_layout: &UpperTriLayout,
    rho_upper: &[Complex64],
    out_upper: &mut [Complex64],
) {
    let n = upper_layout.n;
    for source in 0..n {
        let rate_source = plan.source_decay_rates[source];
        out_upper[upper_layout.index_unchecked(source, source)] -=
            rate_source * rho_upper[upper_layout.index_unchecked(source, source)];
        for col in (source + 1)..n {
            let rate = 0.5 * (rate_source + plan.source_decay_rates[col]);
            out_upper[upper_layout.index_unchecked(source, col)] -=
                rate * rho_upper[upper_layout.index_unchecked(source, col)];
        }
    }
    for (target, incoming) in plan.incoming_transfers_by_target.iter().enumerate() {
        let target_diag = upper_layout.index_unchecked(target, target);
        for transfer in incoming {
            out_upper[target_diag] += transfer.rate
                * rho_upper[upper_layout.index_unchecked(transfer.source, transfer.source)];
        }
    }
}

fn mirror_upper_to_lower(matrix: &mut [Complex64], n: usize) {
    for i in 0..n {
        matrix[i * n + i] = Complex64::new(matrix[i * n + i].re, 0.0);
        for j in (i + 1)..n {
            matrix[j * n + i] = matrix[i * n + j].conj();
        }
    }
}

pub struct RhsWorkspace {
    upper_layout: UpperTriLayout,
    rho: Vec<Complex64>,
    rho_upper: Vec<Complex64>,
    hamiltonian_upper: Vec<Complex64>,
    hamiltonian: Vec<Complex64>,
    drho: Vec<Complex64>,
    drho_upper: Vec<Complex64>,
    scratch: Vec<Complex64>,
    h_sparse_values: Vec<Complex64>,
    expanded_coeff_values: Vec<Complex64>,
    expanded_term_values: Vec<Complex64>,
    parameter_values: Vec<RuntimeValue>,
    hamiltonian_temps: Vec<RuntimeValue>,
    eval_stack: Vec<RuntimeValue>,
    scalar_stack: Vec<Complex64>,
    parameter_overrides: Vec<(usize, RuntimeValue)>,
    split_input: Vec<Complex64>,
    split_output: Vec<Complex64>,
    upper_to_packed: Vec<PackedUpperIndex>,
    expanded_packed_inputs: Vec<PackedTermInput>,
    pchip_hints: Vec<usize>,
    hamiltonian_valid: bool,
    h_sparse_valid: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RhsProfileStats {
    pub calls: u64,
    pub total_seconds: f64,
    pub unpack_seconds: f64,
    pub parameter_eval_seconds: f64,
    pub hamiltonian_fill_seconds: f64,
    pub commutator_seconds: f64,
    pub dissipator_seconds: f64,
    pub pack_seconds: f64,
}

impl RhsWorkspace {
    pub fn new(plan: &PreparedLindbladPlan) -> Self {
        let n = plan.n_states();
        let matrix_len = n * n;
        let upper_layout = UpperTriLayout::new(n).expect("valid upper-triangle layout");
        let upper_to_packed = build_upper_to_packed_indices(n, &upper_layout);
        let expanded_packed_inputs = build_expanded_packed_inputs(plan, &upper_to_packed);
        Self {
            upper_layout: upper_layout.clone(),
            rho: vec![Complex64::ZERO; matrix_len],
            rho_upper: vec![Complex64::ZERO; upper_layout.len()],
            hamiltonian_upper: vec![Complex64::ZERO; upper_layout.len()],
            hamiltonian: vec![Complex64::ZERO; matrix_len],
            drho: vec![Complex64::ZERO; matrix_len],
            drho_upper: vec![Complex64::ZERO; upper_layout.len()],
            scratch: vec![Complex64::ZERO; matrix_len],
            h_sparse_values: vec![Complex64::ZERO; plan.hamiltonian_sparse_pattern.nnz],
            expanded_coeff_values: Vec::with_capacity(plan.hamiltonian_plan.coefficients.len()),
            expanded_term_values: Vec::with_capacity(
                plan.expanded_rhs_plan
                    .as_ref()
                    .map(|rhs_plan| rhs_plan.terms.len())
                    .unwrap_or(0),
            ),
            parameter_values: Vec::with_capacity(plan.parameter_graph.slot_names.len()),
            hamiltonian_temps: Vec::with_capacity(plan.hamiltonian_plan.temps.len()),
            eval_stack: Vec::new(),
            scalar_stack: Vec::new(),
            parameter_overrides: Vec::new(),
            split_input: vec![Complex64::ZERO; matrix_len],
            split_output: vec![Complex64::ZERO; matrix_len],
            upper_to_packed,
            expanded_packed_inputs,
            pchip_hints: vec![0; plan.parameter_graph.pchip_tables.len()],
            hamiltonian_valid: false,
            h_sparse_valid: false,
        }
    }

    pub fn set_scalar_parameter_overrides(
        &mut self,
        slot_indices: &[usize],
        values: &[Complex64],
    ) -> Result<(), String> {
        if slot_indices.len() != values.len() {
            return Err(format!(
                "parameter override slot count {} does not match value count {}",
                slot_indices.len(),
                values.len()
            ));
        }
        self.parameter_overrides.clear();
        self.parameter_overrides.reserve(slot_indices.len());
        for (&slot, &value) in slot_indices.iter().zip(values.iter()) {
            self.parameter_overrides
                .push((slot, RuntimeValue::Scalar(value)));
        }
        self.hamiltonian_valid = false;
        self.h_sparse_valid = false;
        Ok(())
    }

    fn evaluate_parameter_graph(
        &mut self,
        plan: &PreparedLindbladPlan,
        t: f64,
    ) -> Result<(), String> {
        plan.parameter_graph.evaluate_with_overrides_into(
            t,
            self.parameter_overrides.as_slice(),
            &mut self.parameter_values,
            &mut self.eval_stack,
            &mut self.scalar_stack,
            &mut self.pchip_hints,
        )
    }
}

fn split_real_to_complex(input: &[f64], out: &mut [Complex64]) -> Result<(), String> {
    let n = out.len();
    if input.len() != 2 * n {
        return Err(format!(
            "expected split-real vector length {}, got {}",
            2 * n,
            input.len()
        ));
    }
    for idx in 0..n {
        out[idx] = Complex64::new(input[idx], input[n + idx]);
    }
    Ok(())
}

fn complex_to_split_real(input: &[Complex64], out: &mut [f64]) -> Result<(), String> {
    let n = input.len();
    if out.len() != 2 * n {
        return Err(format!(
            "expected split-real output length {}, got {}",
            2 * n,
            out.len()
        ));
    }
    for idx in 0..n {
        out[idx] = input[idx].re;
        out[n + idx] = input[idx].im;
    }
    Ok(())
}

fn rhs_from_workspace_rho(
    plan: &PreparedLindbladPlan,
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    mut profile: Option<&mut RhsProfileStats>,
) -> Result<(), String> {
    let n = plan.n_states();
    if mode == ExecutionMode::ExpandedSparse && plan.expanded_rhs_plan.is_none() {
        return Err(
            "execution_mode='expanded_sparse' requires a decomposed Hamiltonian plan".to_string(),
        );
    }
    let total_start = Instant::now();
    let can_skip = !plan.is_time_dependent && workspace.hamiltonian_valid;
    let parameter_start = Instant::now();
    if !can_skip {
        workspace.evaluate_parameter_graph(plan, t)?;
    }
    if let Some(stats) = profile.as_mut() {
        stats.parameter_eval_seconds += parameter_start.elapsed().as_secs_f64();
    }
    let hamiltonian_start = Instant::now();
    if !can_skip {
        match mode {
            ExecutionMode::ReferenceDense | ExecutionMode::StructuredBlas => {
                if plan.hamiltonian_plan.kind == crate::lindblad::plan::HamiltonianKind::Decomposed
                    && plan.hamiltonian_plan.dense_fill_mode
                        == crate::lindblad::plan::DenseFillMode::UpperExpand
                {
                    plan.hamiltonian_plan.fill_upper_into(
                        workspace.parameter_values.as_slice(),
                        t,
                        &mut workspace.hamiltonian_temps,
                        &mut workspace.eval_stack,
                        &mut workspace.scalar_stack,
                        &workspace.upper_layout,
                        workspace.hamiltonian_upper.as_mut_slice(),
                    )?;
                    workspace.upper_layout.expand_to_dense(
                        workspace.hamiltonian_upper.as_slice(),
                        workspace.hamiltonian.as_mut_slice(),
                    )?;
                } else {
                    plan.hamiltonian_plan.fill_into(
                        workspace.parameter_values.as_slice(),
                        t,
                        &mut workspace.hamiltonian_temps,
                        &mut workspace.eval_stack,
                        &mut workspace.scalar_stack,
                        workspace.hamiltonian.as_mut_slice(),
                    )?;
                }
            }
            ExecutionMode::StructuredUpper => {
                plan.hamiltonian_plan.fill_upper_into(
                    workspace.parameter_values.as_slice(),
                    t,
                    &mut workspace.hamiltonian_temps,
                    &mut workspace.eval_stack,
                    &mut workspace.scalar_stack,
                    &workspace.upper_layout,
                    workspace.hamiltonian_upper.as_mut_slice(),
                )?;
            }
            ExecutionMode::ExpandedSparse => {
                plan.hamiltonian_plan
                    .evaluate_decomposed_coefficients_into(
                        workspace.parameter_values.as_slice(),
                        t,
                        &mut workspace.eval_stack,
                        &mut workspace.scalar_stack,
                        &mut workspace.expanded_coeff_values,
                    )?;
                fill_expanded_sparse_term_values(
                    plan.expanded_rhs_plan
                        .as_ref()
                        .expect("expanded RHS plan checked above"),
                    workspace.expanded_coeff_values.as_slice(),
                    &mut workspace.expanded_term_values,
                )?;
            }
        }
        workspace.hamiltonian_valid = true;
        workspace.h_sparse_valid = false;
    }
    if let Some(stats) = profile.as_mut() {
        stats.hamiltonian_fill_seconds += hamiltonian_start.elapsed().as_secs_f64();
    }
    let commutator_start = Instant::now();
    match mode {
        ExecutionMode::ReferenceDense | ExecutionMode::StructuredBlas => {
            workspace.drho.fill(Complex64::ZERO);
            add_commutator(
                plan,
                workspace.hamiltonian.as_slice(),
                workspace.rho.as_slice(),
                workspace.drho.as_mut_slice(),
                workspace.scratch.as_mut_slice(),
                n,
            )?;
        }
        ExecutionMode::StructuredUpper => {
            workspace
                .upper_layout
                .clear(workspace.drho_upper.as_mut_slice())?;
            if !workspace.h_sparse_valid {
                fill_sparse_h_values(
                    &plan.hamiltonian_sparse_pattern,
                    &workspace.upper_layout,
                    workspace.hamiltonian_upper.as_slice(),
                    workspace.h_sparse_values.as_mut_slice(),
                );
                workspace.h_sparse_valid = true;
            }
            add_commutator_sparse_upper(
                &plan.hamiltonian_sparse_pattern,
                workspace.h_sparse_values.as_slice(),
                &workspace.upper_layout,
                workspace.rho_upper.as_slice(),
                workspace.drho_upper.as_mut_slice(),
            );
        }
        ExecutionMode::ExpandedSparse => {
            workspace
                .upper_layout
                .clear(workspace.drho_upper.as_mut_slice())?;
            add_expanded_sparse_rhs(
                plan.expanded_rhs_plan
                    .as_ref()
                    .expect("expanded RHS plan checked above"),
                workspace.expanded_term_values.as_slice(),
                workspace.rho_upper.as_slice(),
                workspace.drho_upper.as_mut_slice(),
            )?;
        }
    }
    if let Some(stats) = profile.as_mut() {
        stats.commutator_seconds += commutator_start.elapsed().as_secs_f64();
    }
    let dissipator_start = Instant::now();
    match mode {
        ExecutionMode::ReferenceDense => add_dense_dissipator(
            plan,
            workspace.rho.as_slice(),
            workspace.drho.as_mut_slice(),
        ),
        ExecutionMode::StructuredBlas => add_structured_dissipator(
            plan,
            workspace.rho.as_slice(),
            workspace.drho.as_mut_slice(),
        ),
        ExecutionMode::StructuredUpper => add_structured_dissipator_upper(
            plan,
            &workspace.upper_layout,
            workspace.rho_upper.as_slice(),
            workspace.drho_upper.as_mut_slice(),
        ),
        ExecutionMode::ExpandedSparse => {}
    }
    if let Some(stats) = profile.as_mut() {
        stats.calls += 1;
        stats.dissipator_seconds += dissipator_start.elapsed().as_secs_f64();
        stats.total_seconds += total_start.elapsed().as_secs_f64();
    }
    Ok(())
}

pub fn rhs_matrix_into_with_profile(
    plan: &PreparedLindbladPlan,
    rho: &[Complex64],
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    out: &mut [Complex64],
    profile: Option<&mut RhsProfileStats>,
) -> Result<(), String> {
    let n = plan.n_states();
    if rho.len() != n * n {
        return Err(format!(
            "expected full matrix state length {}, got {}",
            n * n,
            rho.len()
        ));
    }
    if out.len() != n * n {
        return Err(format!(
            "expected full matrix rhs length {}, got {}",
            n * n,
            out.len()
        ));
    }
    match mode {
        ExecutionMode::ReferenceDense | ExecutionMode::StructuredBlas => {
            workspace.rho.copy_from_slice(rho);
        }
        ExecutionMode::StructuredUpper | ExecutionMode::ExpandedSparse => {
            workspace
                .upper_layout
                .pack_from_dense(rho, workspace.rho_upper.as_mut_slice())?;
        }
    }
    rhs_from_workspace_rho(plan, t, mode, workspace, profile)?;
    match mode {
        ExecutionMode::ReferenceDense | ExecutionMode::StructuredBlas => {
            mirror_upper_to_lower(workspace.drho.as_mut_slice(), n);
            out.copy_from_slice(workspace.drho.as_slice());
        }
        ExecutionMode::StructuredUpper | ExecutionMode::ExpandedSparse => {
            workspace
                .upper_layout
                .expand_to_dense(workspace.drho_upper.as_slice(), out)?;
        }
    }
    Ok(())
}

pub fn rhs_matrix_into(
    plan: &PreparedLindbladPlan,
    rho: &[Complex64],
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    out: &mut [Complex64],
) -> Result<(), String> {
    rhs_matrix_into_with_profile(plan, rho, t, mode, workspace, out, None)
}

pub fn rhs_split_into_with_profile(
    plan: &PreparedLindbladPlan,
    split_state: &[f64],
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    out: &mut [f64],
    profile: Option<&mut RhsProfileStats>,
) -> Result<(), String> {
    let mut input = std::mem::take(&mut workspace.split_input);
    let mut output = std::mem::take(&mut workspace.split_output);
    input.fill(Complex64::ZERO);
    output.fill(Complex64::ZERO);
    split_real_to_complex(split_state, input.as_mut_slice())?;
    rhs_matrix_into_with_profile(
        plan,
        input.as_slice(),
        t,
        mode,
        workspace,
        output.as_mut_slice(),
        profile,
    )?;
    let result = complex_to_split_real(output.as_slice(), out);
    workspace.split_input = input;
    workspace.split_output = output;
    result
}

pub fn build_split_jacobian_sparse(
    plan: &PreparedLindbladPlan,
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    tol: f64,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>), String> {
    let dim_complex = plan.n_states() * plan.n_states();
    let dim_split = 2 * dim_complex;
    let tol_abs = tol.abs();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    let mut basis = vec![Complex64::ZERO; dim_complex];
    let mut output = vec![Complex64::ZERO; dim_complex];
    for col_complex in 0..dim_complex {
        basis[col_complex] = Complex64::ONE;
        rhs_matrix_into(
            plan,
            basis.as_slice(),
            t,
            mode,
            workspace,
            output.as_mut_slice(),
        )?;
        basis[col_complex] = Complex64::ZERO;

        let col_real = col_complex as i64;
        let col_imag = (dim_complex + col_complex) as i64;
        for row_complex in 0..dim_complex {
            let value = output[row_complex];
            let row_real = row_complex as i64;
            let row_imag = (dim_complex + row_complex) as i64;

            if value.re.abs() > tol_abs {
                rows.push(row_real);
                cols.push(col_real);
                values.push(value.re);
                rows.push(row_imag);
                cols.push(col_imag);
                values.push(value.re);
            }
            if value.im.abs() > tol_abs {
                rows.push(row_imag);
                cols.push(col_real);
                values.push(value.im);
                rows.push(row_real);
                cols.push(col_imag);
                values.push(-value.im);
            }
        }
    }

    if rows.iter().any(|row| *row < 0 || *row >= dim_split as i64) {
        return Err("split Jacobian row index out of bounds".to_string());
    }
    if cols.iter().any(|col| *col < 0 || *col >= dim_split as i64) {
        return Err("split Jacobian column index out of bounds".to_string());
    }
    Ok((rows, cols, values))
}

pub fn build_packed_jacobian_sparse(
    plan: &PreparedLindbladPlan,
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    tol: f64,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>), String> {
    let dim = plan.layout.packed_len();
    let tol_abs = tol.abs();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    let mut basis = vec![0.0_f64; dim];
    let mut output = vec![0.0_f64; dim];
    for col in 0..dim {
        basis[col] = 1.0;
        rhs_packed_into(
            plan,
            basis.as_slice(),
            t,
            mode,
            workspace,
            output.as_mut_slice(),
        )?;
        basis[col] = 0.0;
        for row in 0..dim {
            let value = output[row];
            if value.abs() > tol_abs {
                rows.push(row as i64);
                cols.push(col as i64);
                values.push(value);
            }
        }
    }
    Ok((rows, cols, values))
}

fn rhs_packed_expanded_sparse_into_with_profile(
    plan: &PreparedLindbladPlan,
    packed_state: &[f64],
    t: f64,
    workspace: &mut RhsWorkspace,
    out: &mut [f64],
    mut profile: Option<&mut RhsProfileStats>,
) -> Result<(), String> {
    let rhs_plan = plan.expanded_rhs_plan.as_ref().ok_or_else(|| {
        "execution_mode='expanded_sparse' requires a decomposed Hamiltonian plan".to_string()
    })?;
    if packed_state.len() != plan.layout.packed_len() {
        return Err(format!(
            "expected packed vector length {}, got {}",
            plan.layout.packed_len(),
            packed_state.len()
        ));
    }
    if out.len() != plan.layout.packed_len() {
        return Err(format!(
            "expected packed output length {}, got {}",
            plan.layout.packed_len(),
            out.len()
        ));
    }

    let total_start = Instant::now();
    let unpack_start = Instant::now();
    if let Some(stats) = profile.as_mut() {
        stats.unpack_seconds += unpack_start.elapsed().as_secs_f64();
    }

    let can_skip = !plan.is_time_dependent && workspace.hamiltonian_valid;
    let parameter_start = Instant::now();
    if !can_skip {
        workspace.evaluate_parameter_graph(plan, t)?;
    }
    if let Some(stats) = profile.as_mut() {
        stats.parameter_eval_seconds += parameter_start.elapsed().as_secs_f64();
    }

    let hamiltonian_start = Instant::now();
    if !can_skip {
        plan.hamiltonian_plan
            .evaluate_decomposed_coefficients_into(
                workspace.parameter_values.as_slice(),
                t,
                &mut workspace.eval_stack,
                &mut workspace.scalar_stack,
                &mut workspace.expanded_coeff_values,
            )?;
        fill_expanded_sparse_term_values(
            rhs_plan,
            workspace.expanded_coeff_values.as_slice(),
            &mut workspace.expanded_term_values,
        )?;
        workspace.hamiltonian_valid = true;
        workspace.h_sparse_valid = false;
    }
    if let Some(stats) = profile.as_mut() {
        stats.hamiltonian_fill_seconds += hamiltonian_start.elapsed().as_secs_f64();
    }

    let commutator_start = Instant::now();
    add_expanded_sparse_rhs_packed(
        rhs_plan,
        workspace.expanded_term_values.as_slice(),
        workspace.upper_to_packed.as_slice(),
        workspace.expanded_packed_inputs.as_slice(),
        packed_state,
        out,
    )?;
    if let Some(stats) = profile.as_mut() {
        stats.commutator_seconds += commutator_start.elapsed().as_secs_f64();
    }

    let dissipator_start = Instant::now();
    if let Some(stats) = profile.as_mut() {
        stats.calls += 1;
        stats.dissipator_seconds += dissipator_start.elapsed().as_secs_f64();
        stats.total_seconds += total_start.elapsed().as_secs_f64();
    }
    Ok(())
}

pub fn rhs_packed_into_with_profile(
    plan: &PreparedLindbladPlan,
    packed_state: &[f64],
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    out: &mut [f64],
    mut profile: Option<&mut RhsProfileStats>,
) -> Result<(), String> {
    if mode == ExecutionMode::ExpandedSparse {
        return rhs_packed_expanded_sparse_into_with_profile(
            plan,
            packed_state,
            t,
            workspace,
            out,
            profile,
        );
    }
    let unpack_start = Instant::now();
    match mode {
        ExecutionMode::ReferenceDense | ExecutionMode::StructuredBlas => {
            plan.layout
                .unpack_into(packed_state, workspace.rho.as_mut_slice())?;
        }
        ExecutionMode::StructuredUpper | ExecutionMode::ExpandedSparse => {
            workspace
                .upper_layout
                .unpack_packed_state(packed_state, workspace.rho_upper.as_mut_slice())?;
        }
    }
    if let Some(stats) = profile.as_mut() {
        stats.unpack_seconds += unpack_start.elapsed().as_secs_f64();
    }
    rhs_from_workspace_rho(plan, t, mode, workspace, profile.as_deref_mut())?;
    let pack_start = Instant::now();
    match mode {
        ExecutionMode::ReferenceDense | ExecutionMode::StructuredBlas => {
            plan.layout.pack_into(workspace.drho.as_slice(), out)?;
        }
        ExecutionMode::StructuredUpper | ExecutionMode::ExpandedSparse => {
            workspace
                .upper_layout
                .pack_packed_state(workspace.drho_upper.as_slice(), out)?;
        }
    }
    if let Some(stats) = profile.as_mut() {
        stats.pack_seconds += pack_start.elapsed().as_secs_f64();
    }
    Ok(())
}

pub fn rhs_packed_into(
    plan: &PreparedLindbladPlan,
    packed_state: &[f64],
    t: f64,
    mode: ExecutionMode,
    workspace: &mut RhsWorkspace,
    out: &mut [f64],
) -> Result<(), String> {
    rhs_packed_into_with_profile(plan, packed_state, t, mode, workspace, out, None)
}

pub fn rhs_packed(
    plan: &PreparedLindbladPlan,
    packed_state: &[f64],
    t: f64,
    mode: ExecutionMode,
) -> Result<Vec<f64>, String> {
    let mut workspace = RhsWorkspace::new(plan);
    let mut packed = vec![0.0; plan.layout.packed_len()];
    rhs_packed_into(
        plan,
        packed_state,
        t,
        mode,
        &mut workspace,
        packed.as_mut_slice(),
    )?;
    Ok(packed)
}
