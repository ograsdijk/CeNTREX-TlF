//! Native reverse-time adjoint solves for Hermitian Lindblad objectives.
//!
//! The supported dissipator is the package's structured single-transition jump
//! representation (each collapse operator has exactly one nonzero element).

use crate::lindblad::eval::RuntimeValue;
use crate::lindblad::layout::PackedHermitianLayout;
use crate::lindblad::plan::PreparedLindbladPlan;
use crate::ode::batch::{solve_single, OdeSolver};
use crate::ode::output::OdeOutput;
use crate::ode::{OdeOptions, OdeRhs, OdeStats};
use num_complex::Complex64;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct LastOutput {
    value: Vec<f64>,
    time: f64,
}

impl LastOutput {
    fn new(dim: usize) -> Self {
        Self {
            value: vec![0.0; dim],
            time: f64::NAN,
        }
    }
}

impl OdeOutput for LastOutput {
    fn initialize(&mut self, t: f64, y: &[f64]) {
        self.value.copy_from_slice(y);
        self.time = t;
    }
    fn push(&mut self, t: f64, y: &[f64]) {
        self.value.copy_from_slice(y);
        self.time = t;
    }
    fn times(&self) -> &[f64] {
        std::slice::from_ref(&self.time)
    }
}

struct AdjointRhs<'a> {
    plan: &'a PreparedLindbladPlan,
    terminal_time: f64,
    tau_offset: f64,
    segment_left: f64,
    segment_right: f64,
    layout: PackedHermitianLayout,
    rows: Vec<Vec<usize>>,
    overrides: Vec<(usize, RuntimeValue)>,
    slots: Vec<RuntimeValue>,
    temps: Vec<RuntimeValue>,
    eval_stack: Vec<RuntimeValue>,
    scalar_stack: Vec<Complex64>,
    h: Vec<Complex64>,
    lambda: Vec<Complex64>,
    h_lambda: Vec<Complex64>,
    d_lambda: Vec<Complex64>,
    weights: Vec<f64>,
}

impl<'a> AdjointRhs<'a> {
    fn new(
        plan: &'a PreparedLindbladPlan,
        terminal_time: f64,
        tau_offset: f64,
        segment_left: f64,
        segment_right: f64,
        weights: Vec<f64>,
        parameter_slot_indices: &[usize],
        parameter_values: &[Complex64],
    ) -> Result<Self, String> {
        let n = plan.n_states();
        validate_structured_jumps(plan)?;
        if weights.len() != n || weights.iter().any(|x| !x.is_finite()) {
            return Err("running weights must be a finite vector with one value per state".into());
        }
        if parameter_slot_indices.len() != parameter_values.len() {
            return Err("parameter slot/value lengths differ".into());
        }
        let mut overrides = Vec::with_capacity(parameter_values.len());
        for (&slot, &value) in parameter_slot_indices.iter().zip(parameter_values) {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err("parameter override values must be finite".into());
            }
            overrides.push((slot, RuntimeValue::Scalar(value)));
        }
        let mut rows = vec![Vec::new(); n];
        let pattern = &plan.hamiltonian_sparse_pattern;
        for i in 0..n {
            for ptr in pattern.row_ptrs[i]..pattern.row_ptrs[i + 1] {
                let j = pattern.col_indices[ptr];
                rows[i].push(j);
                if i != j {
                    rows[j].push(i);
                }
            }
        }
        Ok(Self {
            plan,
            terminal_time,
            tau_offset,
            segment_left,
            segment_right,
            layout: plan.layout.clone(),
            rows,
            overrides,
            slots: Vec::new(),
            temps: Vec::new(),
            eval_stack: Vec::new(),
            scalar_stack: Vec::new(),
            h: vec![Complex64::ZERO; n * n],
            lambda: vec![Complex64::ZERO; n * n],
            h_lambda: vec![Complex64::ZERO; n * n],
            d_lambda: vec![Complex64::ZERO; n * n],
            weights,
        })
    }
}

impl OdeRhs for AdjointRhs<'_> {
    fn eval(&mut self, tau: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        let n = self.plan.n_states();
        self.layout.unpack_into(y, &mut self.lambda)?;
        let mut t = self.terminal_time - (self.tau_offset + tau);
        // Evaluate split endpoints from the interior side of this physical segment.
        if t >= self.segment_right {
            t = next_down(t);
        }
        if t <= self.segment_left {
            t = next_up(t);
        }
        self.plan.parameter_graph.evaluate_with_overrides_into(
            t,
            &self.overrides,
            &mut self.slots,
            &mut self.eval_stack,
            &mut self.scalar_stack,
        )?;
        self.plan.hamiltonian_plan.fill_into(
            &self.slots,
            t,
            &mut self.temps,
            &mut self.eval_stack,
            &mut self.scalar_stack,
            &mut self.h,
        )?;

        // Sparse H*lambda; Hermiticity gives lambda*H = (H*lambda)^dagger.
        self.h_lambda.fill(Complex64::ZERO);
        for i in 0..n {
            for &k in &self.rows[i] {
                let hik = self.h[i * n + k];
                if hik == Complex64::ZERO {
                    continue;
                }
                for j in 0..n {
                    self.h_lambda[i * n + j] += hik * self.lambda[k * n + j];
                }
            }
        }
        let plus_i = Complex64::I;
        self.d_lambda.fill(Complex64::ZERO);
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                self.d_lambda[idx] =
                    plus_i * (self.h_lambda[idx] - self.h_lambda[j * n + i].conj());
            }
        }

        // Adjoint of the structured jump dissipator: diagonal transfer plus
        // source-rate coherence damping.
        for source in 0..n {
            let rate = self.plan.source_decay_rates[source];
            self.d_lambda[source * n + source] -= rate * self.lambda[source * n + source];
            for col in (source + 1)..n {
                let damp = 0.5 * (rate + self.plan.source_decay_rates[col]);
                self.d_lambda[source * n + col] -= damp * self.lambda[source * n + col];
                self.d_lambda[col * n + source] = self.d_lambda[source * n + col].conj();
            }
        }
        for (target, incoming) in self.plan.incoming_transfers_by_target.iter().enumerate() {
            for transfer in incoming {
                self.d_lambda[transfer.source * n + transfer.source] +=
                    transfer.rate * self.lambda[target * n + target];
            }
        }
        for i in 0..n {
            self.d_lambda[i * n + i].re += self.weights[i];
        }
        self.layout.pack_into(&self.d_lambda, dy)
    }
    fn dim(&self) -> usize {
        self.layout.packed_len()
    }
}

fn validate_structured_jumps(plan: &PreparedLindbladPlan) -> Result<(), String> {
    let n = plan.n_states();
    let size = n * n;
    if plan.dense_c_array.len() != plan.n_collapse * size {
        return Err("collapse operator storage is inconsistent".into());
    }
    let mut rates = vec![0.0; n];
    let mut transfers = vec![Vec::<(usize, f64)>::new(); n];
    for (idx, c) in plan.dense_c_array.chunks_exact(size).enumerate() {
        let mut nz = c.iter().enumerate().filter(|(_, z)| **z != Complex64::ZERO);
        let Some((flat, amp)) = nz.next() else {
            return Err(format!(
                "collapse operator {idx} is zero; adjoint requires single-transition jumps"
            ));
        };
        if nz.next().is_some() {
            return Err(format!("collapse operator {idx} is not a single-transition jump; general collapse adjoints are unsupported"));
        }
        let target = flat / n;
        let source = flat % n;
        let rate = amp.norm_sqr();
        rates[source] += rate;
        transfers[target].push((source, rate));
    }
    if rates.len() != plan.source_decay_rates.len()
        || rates
            .iter()
            .zip(&plan.source_decay_rates)
            .any(|(a, b)| (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0))
    {
        return Err("structured decay-rate cache disagrees with collapse operators".into());
    }
    let mut actual_sums = vec![vec![0.0; n]; n];
    for target in 0..n {
        let actual = &plan.incoming_transfers_by_target[target];
        for transfer in actual {
            if transfer.source >= n || !transfer.rate.is_finite() || transfer.rate < 0.0 {
                return Err("structured transfer cache contains an invalid entry".into());
            }
            actual_sums[target][transfer.source] += transfer.rate;
        }
        for source in 0..n {
            let expected: f64 = transfers[target]
                .iter()
                .filter(|x| x.0 == source)
                .map(|x| x.1)
                .sum();
            let actual = actual_sums[target][source];
            if (actual - expected).abs() > 1e-12 * actual.abs().max(expected.abs()).max(1.0) {
                return Err("structured transfer cache disagrees with collapse operators".into());
            }
        }
    }
    Ok(())
}

fn next_up(x: f64) -> f64 {
    next_toward(x, f64::INFINITY)
}
fn next_down(x: f64) -> f64 {
    next_toward(x, f64::NEG_INFINITY)
}
fn next_toward(x: f64, toward: f64) -> f64 {
    if x.is_nan() || x == toward {
        return toward;
    }
    if x == 0.0 {
        return if toward > 0.0 {
            f64::from_bits(1)
        } else {
            -f64::from_bits(1)
        };
    }
    let bits = x.to_bits();
    f64::from_bits(if (toward > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    })
}

fn integrate_adjoint(
    plan: &PreparedLindbladPlan,
    mut y: Vec<f64>,
    t0: f64,
    t1: f64,
    edges: Vec<f64>,
    weights: Vec<f64>,
    slots: Vec<usize>,
    values: Vec<Complex64>,
    abstol: f64,
    reltol: f64,
    dt: f64,
    maxiters: usize,
    ode_solver: OdeSolver,
) -> Result<(Vec<f64>, OdeStats), String> {
    let mut bounds = Vec::with_capacity(edges.len() + 2);
    bounds.push(t0);
    bounds.extend(edges);
    bounds.push(t1);
    let mut total_stats = OdeStats::default();
    let options = OdeOptions {
        abstol,
        reltol,
        dt,
        maxiters,
        save_start: true,
        saveat: None,
    };
    for seg in (0..bounds.len() - 1).rev() {
        let left = bounds[seg];
        let right = bounds[seg + 1];
        let tau_offset = t1 - right;
        let mut rhs = AdjointRhs::new(
            plan,
            t1,
            tau_offset,
            left,
            right,
            weights.clone(),
            &slots,
            &values,
        )?;
        let mut out = LastOutput::new(y.len());
        let stats = solve_single(
            &mut rhs,
            &y,
            0.0,
            right - left,
            &options,
            &mut out,
            ode_solver,
        )?;
        y = out.value;
        total_stats.accepted_steps += stats.accepted_steps;
        total_stats.rejected_steps += stats.rejected_steps;
        total_stats.rhs_calls += stats.rhs_calls;
    }
    Ok((y, total_stats))
}

#[pyfunction(signature = (plan, terminal_observable, integral_weights, t0, t1, discontinuities=None, abstol=1e-10, reltol=1e-8, dt=1e-9, maxiters=100000, solver="dopri5", parameter_slot_indices=None, parameter_values=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_lindblad_adjoint_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    terminal_observable: PyReadonlyArray2<'py, Complex64>,
    integral_weights: Vec<f64>,
    t0: f64,
    t1: f64,
    discontinuities: Option<Vec<f64>>,
    abstol: f64,
    reltol: f64,
    dt: f64,
    maxiters: usize,
    solver: &str,
    parameter_slot_indices: Option<Vec<usize>>,
    parameter_values: Option<Vec<Complex64>>,
) -> PyResult<(
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, pyo3::types::PyDict>,
)> {
    let n = plan.n_states();
    let c = terminal_observable
        .as_slice()
        .map_err(PyValueError::new_err)?;
    if terminal_observable.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "terminal_observable must have shape ({n}, {n})"
        )));
    }
    if c.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(PyValueError::new_err("terminal observable must be finite"));
    }
    for i in 0..n {
        for j in 0..n {
            if (c[i * n + j] - c[j * n + i].conj()).norm()
                > 1e-11 * c[i * n + j].norm().max(c[j * n + i].norm()).max(1.0)
            {
                return Err(PyValueError::new_err(
                    "terminal observable must be Hermitian",
                ));
            }
        }
    }
    if !t0.is_finite() || !t1.is_finite() || t1 <= t0 {
        return Err(PyValueError::new_err("t_span must be finite with t1 > t0"));
    }
    if !abstol.is_finite()
        || abstol <= 0.0
        || !reltol.is_finite()
        || reltol <= 0.0
        || !dt.is_finite()
        || dt <= 0.0
        || maxiters == 0
    {
        return Err(PyValueError::new_err(
            "solver tolerances, dt, and maxiters must be positive and finite",
        ));
    }
    let ode_solver = OdeSolver::from_str(solver).map_err(PyValueError::new_err)?;
    if !matches!(ode_solver, OdeSolver::Dopri5 | OdeSolver::Tsit5) {
        return Err(PyValueError::new_err(
            "adjoint currently supports adaptive dopri5 and tsit5 only",
        ));
    }
    let slots = parameter_slot_indices.unwrap_or_default();
    let values = parameter_values.unwrap_or_default();
    let mut y = vec![0.0; plan.layout.packed_len()];
    plan.layout
        .pack_into(c, &mut y)
        .map_err(PyValueError::new_err)?;
    let mut edges = discontinuities.unwrap_or_default();
    if edges.iter().any(|x| !x.is_finite() || *x <= t0 || *x >= t1) {
        return Err(PyValueError::new_err(
            "discontinuities must be finite and strictly inside t_span",
        ));
    }
    edges.sort_by(f64::total_cmp);
    if edges.windows(2).any(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err("discontinuities must be unique"));
    }
    validate_structured_jumps(&plan).map_err(PyValueError::new_err)?;
    let owned_plan = plan.clone();
    let (y, total_stats) = py
        .allow_threads(move || {
            integrate_adjoint(
                &owned_plan,
                y,
                t0,
                t1,
                edges,
                integral_weights,
                slots,
                values,
                abstol,
                reltol,
                dt,
                maxiters,
                ode_solver,
            )
        })
        .map_err(PyValueError::new_err)?;
    let mut matrix = vec![Complex64::ZERO; n * n];
    plan.layout
        .unpack_into(&y, &mut matrix)
        .map_err(PyValueError::new_err)?;
    let array = PyArray2::from_vec2(
        py,
        &matrix
            .chunks_exact(n)
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
    )
    .map_err(PyValueError::new_err)?;
    let stats = pyo3::types::PyDict::new(py);
    stats.set_item("accepted_steps", total_stats.accepted_steps)?;
    stats.set_item("rejected_steps", total_stats.rejected_steps)?;
    stats.set_item("rhs_calls", total_stats.rhs_calls)?;
    stats.set_item("solver", solver)?;
    Ok((array, stats))
}
