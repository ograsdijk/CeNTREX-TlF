use crate::effective_lindblad::plan::EffectiveLindbladPlan;
use crate::effective_lindblad::rhs::{rhs_effective_lindblad, EffectiveLindbladWorkspace};
use crate::ode::batch::{solve_single, OdeSolver};
use crate::ode::output::{
    FullOutput, OdeOutputResult, OdeOutputValues, PopulationsOutput, SelectedExtraction,
    SelectedOutput, WeightedIntegralOutput,
};
use crate::ode::{OdeOptions, OdeRhs, OdeStats};
use rayon::prelude::*;

pub struct EffectiveLindbladRhs<'a> {
    plan: &'a EffectiveLindbladPlan,
    workspace: EffectiveLindbladWorkspace,
}

impl<'a> EffectiveLindbladRhs<'a> {
    pub fn new(plan: &'a EffectiveLindbladPlan) -> Self {
        Self {
            plan,
            workspace: EffectiveLindbladWorkspace::new(plan),
        }
    }

    pub fn new_with_overrides(
        plan: &'a EffectiveLindbladPlan,
        slot_indices: &[usize],
        values: &[f64],
    ) -> Result<Self, String> {
        let mut rhs = Self::new(plan);
        rhs.workspace
            .set_parameter_overrides(slot_indices, values)?;
        Ok(rhs)
    }
}

impl OdeRhs for EffectiveLindbladRhs<'_> {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        rhs_effective_lindblad(self.plan, y, t, &mut self.workspace, dy)
    }
    fn dim(&self) -> usize {
        self.plan.real_dim
    }
}

pub struct EffectiveSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub saveat: Option<Vec<f64>>,
    pub save_start: bool,
    pub maxiters: usize,
    pub solver: String,
}

fn split_real_diagonal_indices(n_states: usize) -> Vec<usize> {
    (0..n_states).map(|i| i * n_states + i).collect()
}

fn split_real_selected_extractions(
    n_states: usize,
    indices: &[(usize, usize)],
) -> Vec<SelectedExtraction> {
    let n2 = n_states * n_states;
    indices
        .iter()
        .map(|&(i, j)| {
            if i == j {
                SelectedExtraction::Real(i * n_states + i)
            } else {
                let real_idx = i * n_states + j;
                let imag_idx = n2 + i * n_states + j;
                SelectedExtraction::ComplexPair { real_idx, imag_idx }
            }
        })
        .collect()
}

pub fn solve_effective_lindblad(
    plan: &EffectiveLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &EffectiveSolverOptions,
    output_mode: &str,
    output_indices: Option<&[(usize, usize)]>,
    integral_weights: Option<&[(usize, f64)]>,
) -> Result<OdeOutputResult, String> {
    let mut rhs = EffectiveLindbladRhs::new(plan);
    let solver = OdeSolver::from_str(&options.solver)?;
    let ode_options = OdeOptions {
        abstol: options.abstol,
        reltol: options.reltol,
        dt: options.dt,
        maxiters: options.maxiters,
        save_start: options.save_start,
        saveat: options.saveat.clone(),
    };
    let capacity = options
        .saveat
        .as_ref()
        .map_or(options.maxiters + 1, |s| s.len() + 1);

    match output_mode {
        "populations" => {
            let indices = split_real_diagonal_indices(plan.n_states);
            let mut output = PopulationsOutput::new(indices, capacity);
            solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
            Ok(output.finish())
        }
        "selected" => {
            let idx = output_indices
                .ok_or_else(|| "output='selected' requires output_indices".to_string())?;
            let extractions = split_real_selected_extractions(plan.n_states, idx);
            let mut output = SelectedOutput::new(extractions, capacity);
            solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
            Ok(output.finish())
        }
        "weighted_integral" | "photon_integral" | "excited_population" => {
            let weights = integral_weights
                .ok_or_else(|| format!("output='{output_mode}' requires integral_weights"))?;
            let mut output = WeightedIntegralOutput::new(weights.to_vec());
            solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
            Ok(output.finish())
        }
        _ => {
            let mut output = FullOutput::new(plan.real_dim, capacity);
            solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
            Ok(output.finish())
        }
    }
}

pub struct EffectiveBatchResult {
    pub times: Vec<f64>,
    pub values: OdeOutputValues,
    pub width: usize,
    pub time_count: usize,
    pub stats: OdeStats,
}

#[allow(clippy::too_many_arguments)]
pub fn solve_effective_lindblad_batch(
    plan: &EffectiveLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &EffectiveSolverOptions,
    output_mode: &str,
    output_indices: Option<&[(usize, usize)]>,
    integral_weights: Option<&[(usize, f64)]>,
    parameter_slot_indices: &[usize],
    parameter_batch: &[f64],
    trajectory_count: usize,
    parallel: bool,
    threads: Option<usize>,
) -> Result<EffectiveBatchResult, String> {
    let dim = plan.real_dim;
    let n_states = plan.n_states;
    let parameter_width = parameter_slot_indices.len();
    if y0.len() != dim {
        return Err(format!("expected state length {dim}, got {}", y0.len()));
    }
    if parameter_batch.len() != trajectory_count * parameter_width {
        return Err(format!(
            "parameter_batch length {} != trajectory_count {} * parameter_width {}",
            parameter_batch.len(),
            trajectory_count,
            parameter_width,
        ));
    }

    let solver = OdeSolver::from_str(&options.solver)?;
    let ode_options = OdeOptions {
        abstol: options.abstol,
        reltol: options.reltol,
        dt: options.dt,
        maxiters: options.maxiters,
        save_start: options.save_start,
        saveat: options.saveat.clone(),
    };
    let capacity = options.saveat.as_ref().map_or(1, |s| s.len() + 1);

    let population_indices: Vec<usize> = split_real_diagonal_indices(n_states);
    let selected_extractions: Option<Vec<SelectedExtraction>> = if output_mode == "selected" {
        let idx = output_indices
            .ok_or_else(|| "output='selected' requires output_indices".to_string())?;
        Some(split_real_selected_extractions(n_states, idx))
    } else {
        None
    };
    let integral_weights_vec: Option<Vec<(usize, f64)>> = integral_weights.map(|w| w.to_vec());

    let solve_one = |trajectory: usize| -> Result<(OdeOutputResult, OdeStats), String> {
        let start = trajectory * parameter_width;
        let param_values = &parameter_batch[start..start + parameter_width];
        let mut rhs = if parameter_width > 0 {
            EffectiveLindbladRhs::new_with_overrides(plan, parameter_slot_indices, param_values)?
        } else {
            EffectiveLindbladRhs::new(plan)
        };
        match output_mode {
            "populations" => {
                let mut output = PopulationsOutput::new(population_indices.clone(), capacity);
                let stats = solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
                Ok((output.finish(), stats))
            }
            "selected" => {
                let mut output =
                    SelectedOutput::new(selected_extractions.clone().unwrap(), capacity);
                let stats = solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
                Ok((output.finish(), stats))
            }
            "weighted_integral" | "photon_integral" | "excited_population" => {
                let mut output = WeightedIntegralOutput::new(integral_weights_vec.clone().unwrap());
                let stats = solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
                Ok((output.finish(), stats))
            }
            _ => {
                let mut output = FullOutput::new(dim, capacity);
                let stats = solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
                Ok((output.finish(), stats))
            }
        }
    };

    let results: Vec<Result<(OdeOutputResult, OdeStats), String>> =
        if !parallel || threads == Some(1) {
            (0..trajectory_count).map(solve_one).collect()
        } else if let Some(n_threads) = threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| format!("failed to create thread pool: {e}"))?;
            pool.install(|| {
                (0..trajectory_count)
                    .into_par_iter()
                    .map(solve_one)
                    .collect()
            })
        } else {
            (0..trajectory_count)
                .into_par_iter()
                .map(solve_one)
                .collect()
        };

    let mut ref_times: Option<Vec<f64>> = None;
    let mut all_values = Vec::new();
    let mut width = 0usize;
    let mut time_count = 0usize;
    let mut total_stats = OdeStats::default();

    for (idx, result) in results.into_iter().enumerate() {
        let (r, stats) = result.map_err(|e| format!("trajectory {idx}: {e}"))?;
        total_stats.accepted_steps += stats.accepted_steps;
        total_stats.rejected_steps += stats.rejected_steps;
        total_stats.rhs_calls += stats.rhs_calls;
        width = r.width;
        time_count = r.times.len();
        if ref_times.is_none() {
            ref_times = Some(r.times);
        }
        match r.values {
            OdeOutputValues::Full(v) | OdeOutputValues::Real(v) => all_values.extend_from_slice(&v),
            OdeOutputValues::Complex(v) => {
                for c in v {
                    all_values.push(c.re);
                    all_values.push(c.im);
                }
            }
        }
    }

    Ok(EffectiveBatchResult {
        times: ref_times.unwrap_or_default(),
        values: OdeOutputValues::Real(all_values),
        width,
        time_count,
        stats: total_stats,
    })
}
