use crate::lindblad::ode_impl::{LindbladRhs, LindbladStopEvent};
use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::ExecutionMode;
use crate::ode::batch::{solve_single, OdeSolver};
use crate::ode::output::{
    FullOutput, OdeOutputResult, OdeOutputValues, PopulationsOutput, SelectedExtraction,
    SelectedOutput, WeightedIntegralOutput,
};
use crate::ode::{OdeOptions, OdeRhs, OdeStats};
use num_complex::Complex64;
use rayon::prelude::*;

fn packed_upper_real_index(n: usize, i: usize, j: usize) -> usize {
    let mut offset = 0usize;
    for row in 0..i {
        offset += n - row - 1;
    }
    n + 2 * (offset + (j - i - 1))
}

fn build_extractions(n: usize, indices: &[(usize, usize)]) -> Vec<SelectedExtraction> {
    indices
        .iter()
        .map(|&(i, j)| {
            if i == j {
                SelectedExtraction::Real(i)
            } else if i < j {
                let r = packed_upper_real_index(n, i, j);
                SelectedExtraction::ComplexPair {
                    real_idx: r,
                    imag_idx: r + 1,
                }
            } else {
                let r = packed_upper_real_index(n, j, i);
                SelectedExtraction::ComplexPairConj {
                    real_idx: r,
                    imag_idx: r + 1,
                }
            }
        })
        .collect()
}

enum ConcreteOutput {
    Populations(PopulationsOutput),
    Selected(SelectedOutput),
    Full(FullOutput),
    WeightedIntegral(WeightedIntegralOutput),
}

impl ConcreteOutput {
    fn solve<R: OdeRhs>(
        &mut self,
        rhs: &mut R,
        y0: &[f64],
        t0: f64,
        t1: f64,
        options: &OdeOptions,
        solver: OdeSolver,
    ) -> Result<OdeStats, String> {
        match self {
            Self::Populations(o) => solve_single(rhs, y0, t0, t1, options, o, solver),
            Self::Selected(o) => solve_single(rhs, y0, t0, t1, options, o, solver),
            Self::Full(o) => solve_single(rhs, y0, t0, t1, options, o, solver),
            Self::WeightedIntegral(o) => solve_single(rhs, y0, t0, t1, options, o, solver),
        }
    }

    fn finish(self) -> OdeOutputResult {
        match self {
            Self::Populations(o) => o.finish(),
            Self::Selected(o) => o.finish(),
            Self::Full(o) => o.finish(),
            Self::WeightedIntegral(o) => o.finish(),
        }
    }
}

#[derive(Clone)]
enum OutputSpec {
    Populations {
        indices: Vec<usize>,
    },
    Selected {
        extractions: Vec<SelectedExtraction>,
    },
    Full {
        dim: usize,
    },
    WeightedIntegral {
        weights: Vec<(usize, f64)>,
    },
}

impl OutputSpec {
    fn create(&self, capacity: usize) -> ConcreteOutput {
        match self {
            Self::Populations { indices } => {
                ConcreteOutput::Populations(PopulationsOutput::new(indices.clone(), capacity))
            }
            Self::Selected { extractions } => {
                ConcreteOutput::Selected(SelectedOutput::new(extractions.clone(), capacity))
            }
            Self::Full { dim } => ConcreteOutput::Full(FullOutput::new(*dim, capacity)),
            Self::WeightedIntegral { weights } => {
                ConcreteOutput::WeightedIntegral(WeightedIntegralOutput::new(weights.clone()))
            }
        }
    }
}

pub struct BatchOdeResult {
    pub times: Vec<f64>,
    pub values: OdeOutputValues,
    pub width: usize,
    pub time_count: usize,
    pub stats: OdeStats,
    pub event_triggered: Vec<bool>,
    pub event_times: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn solve_batch_ode(
    plan: &PreparedLindbladPlan,
    solver: OdeSolver,
    y0_batch: &[f64],
    trajectory_count: usize,
    t0: f64,
    t1: f64,
    options: &OdeOptions,
    execution_mode: ExecutionMode,
    output_mode: &str,
    output_indices: Option<&[(usize, usize)]>,
    integral_weights: Option<&[(usize, f64)]>,
    parameter_slot_indices: &[usize],
    parameter_batch: Option<&[Complex64]>,
    stop_event: Option<LindbladStopEvent>,
    parallel: bool,
    threads: Option<usize>,
) -> Result<BatchOdeResult, String> {
    let dim = plan.layout.packed_len();
    let n = plan.layout.n;
    let parameter_width = parameter_slot_indices.len();

    if y0_batch.len() != trajectory_count * dim {
        return Err(format!(
            "y0_batch length {} != trajectory_count {} * dim {}",
            y0_batch.len(),
            trajectory_count,
            dim,
        ));
    }

    let capacity = options.saveat.as_ref().map_or(1, |s| s.len() + 1);
    let output_spec = match output_mode {
        "populations" => OutputSpec::Populations {
            indices: (0..n).collect(),
        },
        "selected" => {
            let idx = output_indices
                .ok_or_else(|| "output='selected' requires output_indices".to_string())?;
            OutputSpec::Selected {
                extractions: build_extractions(n, idx),
            }
        }
        "full" => OutputSpec::Full { dim },
        "weighted_integral" | "photon_integral" | "excited_population" => {
            let weights = integral_weights
                .ok_or_else(|| format!("output='{output_mode}' requires integral_weights"))?;
            OutputSpec::WeightedIntegral {
                weights: weights.to_vec(),
            }
        }
        other => return Err(format!("unknown output mode: {other:?}")),
    };

    let solve_one = |trajectory: usize| -> Result<(OdeOutputResult, OdeStats), String> {
        let y0_start = trajectory * dim;
        let y0 = &y0_batch[y0_start..y0_start + dim];
        let mut rhs = if !parameter_slot_indices.is_empty() && parameter_batch.is_some() {
            let batch = parameter_batch.unwrap();
            let start = trajectory * parameter_width;
            let param_values = &batch[start..start + parameter_width];
            LindbladRhs::new_with_overrides_and_event(
                plan,
                execution_mode,
                parameter_slot_indices,
                param_values,
                stop_event.clone(),
            )?
        } else {
            LindbladRhs::new(plan, execution_mode).with_stop_event(stop_event.clone())
        };
        let mut output = output_spec.create(capacity);
        let stats = output.solve(&mut rhs, y0, t0, t1, options, solver)?;
        Ok((output.finish(), stats))
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

    let is_complex = output_mode == "selected";
    let mut ref_times: Option<Vec<f64>> = None;
    let mut all_f64 = Vec::new();
    let mut all_c64 = Vec::new();
    let mut width = 0usize;
    let mut time_count = 0usize;
    let mut total_stats = OdeStats::default();
    let mut event_triggered = Vec::with_capacity(trajectory_count);
    let mut event_times = Vec::with_capacity(trajectory_count);

    for (idx, result) in results.into_iter().enumerate() {
        let (r, stats) = result.map_err(|e| format!("trajectory {idx}: {e}"))?;
        total_stats.accepted_steps += stats.accepted_steps;
        total_stats.rejected_steps += stats.rejected_steps;
        total_stats.rhs_calls += stats.rhs_calls;
        if stats.event_triggered {
            total_stats.event_triggered = true;
        }
        event_triggered.push(stats.event_triggered);
        event_times.push(if stats.event_triggered {
            stats.event_time
        } else {
            *r.times.last().unwrap_or(&t1)
        });
        width = r.width;
        time_count = r.times.len();
        if ref_times.is_none() && stop_event.is_none() {
            ref_times = Some(r.times);
        }
        match r.values {
            OdeOutputValues::Full(v) => all_f64.extend_from_slice(&v),
            OdeOutputValues::Real(v) => all_f64.extend_from_slice(&v),
            OdeOutputValues::Complex(v) => all_c64.extend_from_slice(&v),
        }
    }

    Ok(BatchOdeResult {
        times: if stop_event.is_some() {
            event_times.clone()
        } else {
            ref_times.unwrap_or_default()
        },
        values: if is_complex {
            OdeOutputValues::Complex(all_c64)
        } else {
            OdeOutputValues::Real(all_f64)
        },
        width,
        time_count,
        stats: total_stats,
        event_triggered,
        event_times,
    })
}

pub fn grid_trajectory_count(axis_lengths: &[usize]) -> Result<usize, String> {
    if axis_lengths.is_empty() {
        return Err("parameter grid must contain at least one axis".to_string());
    }
    let mut count = 1usize;
    for &length in axis_lengths {
        if length == 0 {
            return Err("parameter grid axes must be non-empty".to_string());
        }
        count = count
            .checked_mul(length)
            .ok_or_else(|| "parameter grid trajectory count overflowed".to_string())?;
    }
    Ok(count)
}

pub fn grid_strides(axis_lengths: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; axis_lengths.len()];
    let mut running = 1usize;
    for idx in (0..axis_lengths.len()).rev() {
        strides[idx] = running;
        running *= axis_lengths[idx];
    }
    strides
}

pub fn fill_grid_parameter_values(
    trajectory: usize,
    axes: &[Complex64],
    axis_offsets: &[usize],
    axis_lengths: &[usize],
    strides: &[usize],
    out: &mut [Complex64],
) {
    for axis in 0..axis_lengths.len() {
        let axis_index = (trajectory / strides[axis]) % axis_lengths[axis];
        out[axis] = axes[axis_offsets[axis] + axis_index];
    }
}

#[allow(clippy::too_many_arguments)]
pub fn solve_grid_ode(
    plan: &PreparedLindbladPlan,
    solver: OdeSolver,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeOptions,
    execution_mode: ExecutionMode,
    output_mode: &str,
    output_indices: Option<&[(usize, usize)]>,
    integral_weights: Option<&[(usize, f64)]>,
    parameter_slot_indices: &[usize],
    axes: &[Complex64],
    axis_offsets: &[usize],
    axis_lengths: &[usize],
    stop_event: Option<LindbladStopEvent>,
    parallel: bool,
    threads: Option<usize>,
) -> Result<BatchOdeResult, String> {
    let trajectory_count = grid_trajectory_count(axis_lengths)?;
    let strides = grid_strides(axis_lengths);
    let dim = plan.layout.packed_len();
    let n = plan.layout.n;
    let capacity = options.saveat.as_ref().map_or(1, |s| s.len() + 1);

    let output_spec = match output_mode {
        "populations" => OutputSpec::Populations {
            indices: (0..n).collect(),
        },
        "selected" => {
            let idx = output_indices
                .ok_or_else(|| "output='selected' requires output_indices".to_string())?;
            OutputSpec::Selected {
                extractions: build_extractions(n, idx),
            }
        }
        "full" => OutputSpec::Full { dim },
        "weighted_integral" | "photon_integral" | "excited_population" => {
            let weights = integral_weights
                .ok_or_else(|| format!("output='{output_mode}' requires integral_weights"))?;
            OutputSpec::WeightedIntegral {
                weights: weights.to_vec(),
            }
        }
        other => return Err(format!("unknown output mode: {other:?}")),
    };

    let solve_one = |trajectory: usize| -> Result<(OdeOutputResult, OdeStats), String> {
        let mut param_values = vec![Complex64::ZERO; parameter_slot_indices.len()];
        fill_grid_parameter_values(
            trajectory,
            axes,
            axis_offsets,
            axis_lengths,
            &strides,
            &mut param_values,
        );
        let mut rhs = LindbladRhs::new_with_overrides(
            plan,
            execution_mode,
            parameter_slot_indices,
            &param_values,
        )?
        .with_stop_event(stop_event.clone());
        let mut output = output_spec.create(capacity);
        let stats = output.solve(&mut rhs, y0, t0, t1, options, solver)?;
        Ok((output.finish(), stats))
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

    let is_complex = output_mode == "selected";
    let mut ref_times: Option<Vec<f64>> = None;
    let mut all_f64 = Vec::new();
    let mut all_c64 = Vec::new();
    let mut width = 0usize;
    let mut time_count = 0usize;
    let mut total_stats = OdeStats::default();
    let mut event_triggered = Vec::with_capacity(trajectory_count);
    let mut event_times = Vec::with_capacity(trajectory_count);

    for (idx, result) in results.into_iter().enumerate() {
        let (r, stats) = result.map_err(|e| format!("trajectory {idx}: {e}"))?;
        total_stats.accepted_steps += stats.accepted_steps;
        total_stats.rejected_steps += stats.rejected_steps;
        total_stats.rhs_calls += stats.rhs_calls;
        if stats.event_triggered {
            total_stats.event_triggered = true;
        }
        event_triggered.push(stats.event_triggered);
        event_times.push(if stats.event_triggered {
            stats.event_time
        } else {
            *r.times.last().unwrap_or(&t1)
        });
        width = r.width;
        time_count = r.times.len();
        if ref_times.is_none() && stop_event.is_none() {
            ref_times = Some(r.times);
        }
        match r.values {
            OdeOutputValues::Full(v) => all_f64.extend_from_slice(&v),
            OdeOutputValues::Real(v) => all_f64.extend_from_slice(&v),
            OdeOutputValues::Complex(v) => all_c64.extend_from_slice(&v),
        }
    }

    Ok(BatchOdeResult {
        times: if stop_event.is_some() {
            event_times.clone()
        } else {
            ref_times.unwrap_or_default()
        },
        values: if is_complex {
            OdeOutputValues::Complex(all_c64)
        } else {
            OdeOutputValues::Real(all_f64)
        },
        width,
        time_count,
        stats: total_stats,
        event_triggered,
        event_times,
    })
}
