//! Benchmark-only shared DOPRI5 controller. This module is deliberately outside
//! the production batch/grid entry points.
use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::ExperimentalBatchRhs;
use crate::ode::dopri5::solve_dopri5;
use crate::ode::output::OdeOutput;
use crate::ode::{OdeOptions, OdeRhs, OdeStats};
use num_complex::Complex64;

pub struct SharedResult {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub width: usize,
    pub stats: OdeStats,
    pub controllers: Vec<u64>,
    pub rejected_controllers: Vec<u64>,
    pub attempted_steps: Vec<f64>,
}

struct SharedRhs<'a> {
    kernel: ExperimentalBatchRhs<'a>,
    dim: usize,
    trajectory_count: usize,
    weights: Vec<(usize, f64)>,
    controllers: Vec<u64>,
    rejected_controllers: Vec<u64>,
    attempted_steps: Vec<f64>,
    physical_state: Vec<f64>,
    physical_out: Vec<f64>,
    maximum_step: Option<f64>,
}

impl OdeRhs for SharedRhs<'_> {
    fn maximum_step(&self) -> Option<f64> { self.maximum_step }
    fn dim(&self) -> usize {
        self.trajectory_count * (self.dim + usize::from(!self.weights.is_empty()))
    }

    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        let stride = self.dim + usize::from(!self.weights.is_empty());
        for trajectory in 0..self.trajectory_count {
            let offset = trajectory * stride;
            let dst = trajectory * self.dim;
            self.physical_state[dst..dst + self.dim].copy_from_slice(&y[offset..offset + self.dim]);
        }
        self.kernel.eval(t, &self.physical_state, &mut self.physical_out)?;
        for trajectory in 0..self.trajectory_count {
            let offset = trajectory * stride;
            let src = trajectory * self.dim;
            dy[offset..offset + self.dim].copy_from_slice(&self.physical_out[src..src + self.dim]);
            if !self.weights.is_empty() {
                dy[offset + self.dim] = self.weights.iter().map(|&(i, w)| w * y[offset + i]).sum();
            }
        }
        Ok(())
    }

    fn adaptive_error_norm(
        &mut self, y: &[f64], yn: &[f64], h: f64, k: &[f64],
        total_dim: usize, atol: f64, rtol: f64,
    ) -> Option<f64> {
        const E1: f64 = 71.0 / 57600.0;
        const E3: f64 = -71.0 / 16695.0;
        const E4: f64 = 71.0 / 1920.0;
        const E5: f64 = -17253.0 / 339200.0;
        const E6: f64 = 22.0 / 525.0;
        const E7: f64 = -1.0 / 40.0;
        let stride = self.dim + usize::from(!self.weights.is_empty());
        let mut max_err = -1.0;
        let mut controller = 0;
        for trajectory in 0..self.trajectory_count {
            let mut sum = 0.0;
            for component in 0..self.dim {
                let i = trajectory * stride + component;
                let scale = atol + y[i].abs().max(yn[i].abs()) * rtol;
                let error = h * (E1 * k[i]
                    + E3 * k[2 * total_dim + i]
                    + E4 * k[3 * total_dim + i]
                    + E5 * k[4 * total_dim + i]
                    + E6 * k[5 * total_dim + i]
                    + E7 * k[6 * total_dim + i]);
                sum += (error / scale).powi(2);
            }
            let norm = (sum / self.dim as f64).sqrt();
            if norm > max_err {
                max_err = norm;
                controller = trajectory;
            }
        }
        self.controllers[controller] += 1;
        if max_err > 1.0 { self.rejected_controllers[controller] += 1; }
        self.attempted_steps.push(h);
        Some(max_err)
    }
}

struct SharedOutput {
    times: Vec<f64>,
    values: Vec<f64>,
    physical_dim: usize,
    trajectory_count: usize,
    width: usize,
    mode: String,
    has_integral: bool,
}

impl OdeOutput for SharedOutput {
    fn push(&mut self, t: f64, y: &[f64]) {
        self.times.push(t);
        let stride = self.physical_dim + usize::from(self.has_integral);
        for trajectory in 0..self.trajectory_count {
            let offset = trajectory * stride;
            match self.mode.as_str() {
                "full" => self.values.extend_from_slice(&y[offset..offset + self.physical_dim]),
                "populations" => self.values.extend_from_slice(&y[offset..offset + self.width]),
                "weighted_integral" => self.values.push(y[offset + self.physical_dim]),
                _ => unreachable!(),
            }
        }
    }
    fn times(&self) -> &[f64] { &self.times }
}

#[allow(clippy::too_many_arguments)]
pub fn solve_shared_experiment(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    parameter_count: usize,
    initial_count: usize,
    parameter_slots: &[usize],
    parameter_values: &[Complex64],
    t0: f64,
    t1: f64,
    options: &OdeOptions,
    mode: &str,
    weights: &[(usize, f64)],
    maximum_step: Option<f64>,
) -> Result<SharedResult, String> {
    let physical_dim = plan.layout.packed_len();
    let trajectory_count = parameter_count * initial_count;
    if y0.len() != trajectory_count * physical_dim {
        return Err("initial batch dimensions do not match plan".into());
    }
    if weights.iter().any(|&(index, _)| index >= plan.layout.n) {
        return Err("integral weight index out of bounds".into());
    }
    if !matches!(mode, "full" | "populations" | "weighted_integral") {
        return Err("unsupported experimental output mode".into());
    }
    if mode == "weighted_integral" && weights.is_empty() {
        return Err("weighted_integral requires weights".into());
    }
    let has_integral = mode == "weighted_integral";
    let kernel = ExperimentalBatchRhs::new(
        plan, parameter_count, initial_count, parameter_slots, parameter_values,
    )?;
    let mut rhs = SharedRhs {
        kernel, physical_state: vec![0.0; y0.len()], physical_out: vec![0.0; y0.len()],
        dim: physical_dim, trajectory_count,
        weights: if has_integral { weights.to_vec() } else { Vec::new() },
        controllers: vec![0; trajectory_count],
        rejected_controllers: vec![0; trajectory_count], attempted_steps: Vec::new(),
        maximum_step,
    };
    let mut initial = Vec::with_capacity(rhs.dim());
    for trajectory in 0..trajectory_count {
        initial.extend_from_slice(&y0[trajectory * physical_dim..(trajectory + 1) * physical_dim]);
        if has_integral { initial.push(0.0); }
    }
    let mut output = SharedOutput {
        times: Vec::new(), values: Vec::new(), physical_dim, trajectory_count,
        width: match mode { "full" => physical_dim, "populations" => plan.layout.n, _ => 1 },
        mode: mode.into(), has_integral,
    };
    let stats = solve_dopri5(&mut rhs, &initial, t0, t1, options, &mut output)?;
    Ok(SharedResult {
        times: output.times, values: output.values, width: output.width,
        stats, controllers: rhs.controllers, rejected_controllers: rhs.rejected_controllers,
        attempted_steps: rhs.attempted_steps,
    })
}
