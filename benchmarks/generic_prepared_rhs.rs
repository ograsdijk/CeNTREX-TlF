// Benchmark-only data preparation; no molecule-specific generated arithmetic.
use crate::lindblad::eval::InstructionOp;

pub struct GenericPrepared {
    static_upper: Vec<Complex64>,
    updates: Vec<Vec<(usize, Complex64)>>,
    damping: Vec<(usize, usize, f64)>,
    transfers: Vec<(usize, usize, f64)>,
    patterns: Vec<(Vec<bool>, HermitianSparsePattern)>,
    active: usize,
    sparse_values: Vec<Complex64>,
}

impl GenericPrepared {
    pub fn new(plan: &PreparedLindbladPlan, gates: &[usize]) -> Self {
        let n = plan.layout.n;
        let layout = UpperTriLayout::new(n).unwrap();
        let mut static_upper = Vec::new();
        let mut damping = Vec::new();
        for i in 0..n {
            for j in i..n {
                static_upper.push(plan.hamiltonian_plan.static_matrix[i * n + j]);
                let rate = 0.5 * (plan.source_decay_rates[i] + plan.source_decay_rates[j]);
                if rate != 0.0 {
                    damping.push((layout.index_unchecked(i, j), i * n + j, rate));
                }
            }
        }
        let updates =
            plan.hamiltonian_plan
                .coefficients
                .iter()
                .map(|c| {
                    if c.basis_row_segments.is_empty() {
                        c.basis_terms
                            .iter()
                            .map(|b| (layout.index_unchecked(b.i, b.j), b.value))
                            .collect()
                    } else {
                        c.basis_row_segments
                            .iter()
                            .flat_map(|s| {
                                s.values.iter().enumerate().map(|(k, v)| {
                                    (layout.index_unchecked(s.row, s.start_col + k), *v)
                                })
                            })
                            .collect()
                    }
                })
                .collect();
        let transfers = plan
            .incoming_transfers_by_target
            .iter()
            .enumerate()
            .flat_map(|(i, ts)| ts.iter().map(move |x| (i, x.source, x.rate)))
            .collect();
        // Build all gate combinations once. A coefficient is removed only when
        // simple zero propagation proves it identically zero for that gate state.
        let mut patterns = Vec::new();
        for bits in 0..(1usize << gates.len()) {
            let key: Vec<bool> = (0..gates.len()).map(|i| bits & (1 << i) != 0).collect();
            let mut hp = plan.hamiltonian_plan.clone();
            hp.coefficients.retain(|c| {
                let mut stack = Vec::new();
                for ins in &c.expression.instructions {
                    let z = match ins.op {
                        InstructionOp::Const => ins.value == Complex64::ZERO,
                        InstructionOp::Slot => gates
                            .iter()
                            .position(|&g| g == ins.index)
                            .map_or(false, |i| !key[i]),
                        InstructionOp::Time | InstructionOp::Temp => false,
                        InstructionOp::Neg | InstructionOp::Conj => stack.pop().unwrap(),
                        InstructionOp::Add | InstructionOp::Sub => {
                            let b = stack.pop().unwrap();
                            let a = stack.pop().unwrap();
                            a && b
                        }
                        InstructionOp::Mul => {
                            let b = stack.pop().unwrap();
                            let a = stack.pop().unwrap();
                            a || b
                        }
                        InstructionOp::BuiltinFunc | InstructionOp::HelperFunc => {
                            for _ in 0..ins.argc {
                                stack.pop();
                            }
                            false
                        }
                        _ => {
                            stack.pop();
                            stack.pop();
                            false
                        }
                    };
                    stack.push(z);
                }
                !stack[0]
            });
            patterns.push((key, HermitianSparsePattern::from_hamiltonian_plan(&hp)));
        }
        let mut result = Self {
            static_upper,
            updates,
            damping,
            transfers,
            patterns,
            active: 0,
            sparse_values: vec![Complex64::ZERO; plan.hamiltonian_sparse_pattern.nnz],
        };
        result.refresh(plan, gates);
        result
    }
    pub fn refresh(&mut self, plan: &PreparedLindbladPlan, gates: &[usize]) {
        self.active = gates.iter().enumerate().fold(0, |key, (i, &g)| {
            let value = match plan.parameter_graph.base_values[g] {
                RuntimeValue::Scalar(x) => x,
                _ => panic!("scalar gate"),
            };
            assert!(value == Complex64::ZERO || value == Complex64::new(1., 0.));
            key | (usize::from(value != Complex64::ZERO) << i)
        });
    }
}

pub fn benchmark_generic_rhs(
    plan: &PreparedLindbladPlan,
    y: &[f64],
    out: &mut [f64],
    t: f64,
    slots: &[RuntimeValue],
    coeffs: &[Complex64],
    w: &mut RhsWorkspace,
    g: &mut GenericPrepared,
    flags: usize,
    kernel: usize,
) -> Result<(), String> {
    plan.layout.unpack_into(y, &mut w.rho)?;
    if flags & 1 != 0 {
        w.hamiltonian_upper.copy_from_slice(&g.static_upper);
        for (c, updates) in coeffs.iter().zip(&g.updates) {
            for &(idx, value) in updates {
                w.hamiltonian_upper[idx] += *c * value;
            }
        }
    } else {
        w.upper_layout.clear(&mut w.hamiltonian_upper)?;
        let n = plan.layout.n;
        for i in 0..n {
            for j in i..n {
                w.hamiltonian_upper[w.upper_layout.index(i, j)?] =
                    plan.hamiltonian_plan.static_matrix[i * n + j];
            }
        }
        for (coefficient, value) in plan.hamiltonian_plan.coefficients.iter().zip(coeffs) {
            if !coefficient.basis_row_segments.is_empty() {
                for s in &coefficient.basis_row_segments {
                    for (k, v) in s.values.iter().enumerate() {
                        w.hamiltonian_upper[w.upper_layout.index(s.row, s.start_col + k)?] +=
                            *value * *v;
                    }
                }
            } else {
                for b in &coefficient.basis_terms {
                    w.hamiltonian_upper[w.upper_layout.index(b.i, b.j)?] += *value * b.value;
                }
            }
        }
    }
    let pattern = if flags & 8 != 0 {
        &g.patterns[g.active].1
    } else {
        &plan.hamiltonian_sparse_pattern
    };
    fill_sparse_h_values(
        pattern,
        &w.upper_layout,
        &w.hamiltonian_upper,
        &mut g.sparse_values,
    );
    match kernel {
        0 | 1 => add_commutator_sparse_dense_upper_soa(
            pattern,
            &g.sparse_values,
            &w.hamiltonian_upper,
            &w.rho,
            &mut w.scratch_re,
            &mut w.scratch_im,
            &w.upper_layout,
            &mut w.drho_upper,
            kernel == 0,
        ),
        2 => add_commutator_sparse_dense_upper_diag(
            pattern,
            &g.sparse_values,
            &w.hamiltonian_upper,
            &w.rho,
            &mut w.scratch,
            &w.upper_layout,
            &mut w.drho_upper,
        ),
        3 => add_commutator_sparse_dense_upper(
            pattern,
            &g.sparse_values,
            &w.rho,
            &mut w.scratch,
            &w.upper_layout,
            &mut w.drho_upper,
        ),
        _ => unreachable!(),
    }
    if flags & 2 != 0 {
        for &(idx, dense, rate) in &g.damping {
            w.drho_upper[idx] -= rate * w.rho[dense];
        }
        for &(dst, src, rate) in &g.transfers {
            w.drho_upper[w.upper_layout.index_unchecked(dst, dst)] +=
                rate * w.rho[src * plan.layout.n + src];
        }
    } else {
        add_structured_dissipator_upper_dense_rho(plan, &w.upper_layout, &w.rho, &mut w.drho_upper);
    }
    w.upper_layout.pack_packed_state(&w.drho_upper, out)
}
