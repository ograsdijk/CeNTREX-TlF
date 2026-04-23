use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::solver_dopri5_fast::solve_dopri5_fast_output_with_parameter_overrides;
use crate::lindblad::solver_fast_common::{
    FastOutputKind, FastOutputOptions, FastOutputValues, FastOutputWhen,
};
use crate::lindblad::solver_ode::OdeSolverOptions;
use crate::lindblad::solver_stats::SolveStats;
use crate::lindblad::solver_tsit5_fast::solve_tsit5_fast_output_with_parameter_overrides;
use num_complex::Complex64;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchFastSolver {
    Dopri5Fast,
    Tsit5Fast,
}

impl BatchFastSolver {
    pub fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "dopri5_fast" => Ok(Self::Dopri5Fast),
            "tsit5_fast" => Ok(Self::Tsit5Fast),
            other => Err(format!(
                "batch solver must be 'dopri5_fast' or 'tsit5_fast', got {other:?}"
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub enum BatchOutputValues {
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

#[derive(Clone, Debug)]
pub struct BatchSolveOutput {
    pub times: Vec<f64>,
    pub values: BatchOutputValues,
    pub width: usize,
    pub time_count: usize,
    pub stats: SolveStats,
}

#[allow(clippy::too_many_arguments)]
fn solve_one(
    plan: &PreparedLindbladPlan,
    solver: BatchFastSolver,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    parameter_values: &[Complex64],
) -> Result<(Vec<f64>, FastOutputValues, usize, SolveStats), String> {
    let result = match solver {
        BatchFastSolver::Dopri5Fast => solve_dopri5_fast_output_with_parameter_overrides(
            plan,
            y0,
            t0,
            t1,
            options,
            output_options,
            collect_stats,
            parameter_slot_indices,
            parameter_values,
        )?,
        BatchFastSolver::Tsit5Fast => solve_tsit5_fast_output_with_parameter_overrides(
            plan,
            y0,
            t0,
            t1,
            options,
            output_options,
            collect_stats,
            parameter_slot_indices,
            parameter_values,
        )?,
    };
    Ok((result.times, result.values, result.width, result.stats))
}

fn merge_stats(solver: BatchFastSolver, stats: &[SolveStats], total_seconds: f64) -> SolveStats {
    let mut out = SolveStats {
        solver: match solver {
            BatchFastSolver::Dopri5Fast => "dopri5_fast_batch".to_string(),
            BatchFastSolver::Tsit5Fast => "tsit5_fast_batch".to_string(),
        },
        total_seconds,
        ..SolveStats::default()
    };
    for item in stats {
        out.rhs_calls += item.rhs_calls;
        out.jacobian_calls += item.jacobian_calls;
        out.function_evaluations += item.function_evaluations;
        out.accepted_steps += item.accepted_steps;
        out.rejected_steps += item.rejected_steps;
        out.internal_steps += item.internal_steps;
        out.saved_points += item.saved_points;
        out.setup_seconds += item.setup_seconds;
        out.integration_seconds += item.integration_seconds;
        out.interpolation_seconds += item.interpolation_seconds;
        out.rhs_seconds += item.rhs_seconds;
        out.jacobian_seconds += item.jacobian_seconds;
    }
    out
}

fn check_times(reference: &[f64], other: &[f64], trajectory: usize) -> Result<(), String> {
    if reference.len() != other.len() {
        return Err(format!(
            "trajectory {trajectory} produced {} times, expected {}",
            other.len(),
            reference.len()
        ));
    }
    for (idx, (&left, &right)) in reference.iter().zip(other.iter()).enumerate() {
        let tol = (left.abs().max(right.abs()) * 1e-12).max(1e-14);
        if (left - right).abs() > tol {
            return Err(format!(
                "trajectory {trajectory} time {idx} is {right}, expected {left}"
            ));
        }
    }
    Ok(())
}

fn solve_batch_serial(
    plan: &PreparedLindbladPlan,
    solver: BatchFastSolver,
    y0_batch: &[f64],
    trajectory_count: usize,
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    parameter_batch: Option<&[Complex64]>,
) -> Result<Vec<(Vec<f64>, FastOutputValues, usize, SolveStats)>, String> {
    let dim = plan.layout.packed_len();
    let parameter_width = parameter_slot_indices.len();
    let mut out = Vec::with_capacity(trajectory_count);
    for trajectory in 0..trajectory_count {
        let y0_start = trajectory * dim;
        let parameter_values = parameter_batch
            .map(|values| {
                let start = trajectory * parameter_width;
                &values[start..start + parameter_width]
            })
            .unwrap_or(&[]);
        out.push(solve_one(
            plan,
            solver,
            &y0_batch[y0_start..y0_start + dim],
            t0,
            t1,
            options,
            output_options,
            collect_stats,
            parameter_slot_indices,
            parameter_values,
        )?);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn solve_batch_parallel(
    plan: &PreparedLindbladPlan,
    solver: BatchFastSolver,
    y0_batch: &[f64],
    trajectory_count: usize,
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    parameter_batch: Option<&[Complex64]>,
) -> Result<Vec<(Vec<f64>, FastOutputValues, usize, SolveStats)>, String> {
    let dim = plan.layout.packed_len();
    let parameter_width = parameter_slot_indices.len();
    (0..trajectory_count)
        .into_par_iter()
        .map(|trajectory| {
            let y0_start = trajectory * dim;
            let parameter_values = parameter_batch
                .map(|values| {
                    let start = trajectory * parameter_width;
                    &values[start..start + parameter_width]
                })
                .unwrap_or(&[]);
            solve_one(
                plan,
                solver,
                &y0_batch[y0_start..y0_start + dim],
                t0,
                t1,
                options,
                output_options,
                collect_stats,
                parameter_slot_indices,
                parameter_values,
            )
        })
        .collect()
}

fn grid_trajectory_count(axis_lengths: &[usize]) -> Result<usize, String> {
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
            .ok_or_else(|| "parameter grid trajectory count overflowed usize".to_string())?;
    }
    Ok(count)
}

fn grid_strides(axis_lengths: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; axis_lengths.len()];
    let mut running = 1usize;
    for idx in (0..axis_lengths.len()).rev() {
        strides[idx] = running;
        running *= axis_lengths[idx];
    }
    strides
}

fn fill_grid_parameter_values(
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
fn solve_grid_serial(
    plan: &PreparedLindbladPlan,
    solver: BatchFastSolver,
    y0: &[f64],
    trajectory_count: usize,
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    axes: &[Complex64],
    axis_offsets: &[usize],
    axis_lengths: &[usize],
    strides: &[usize],
) -> Result<Vec<(Vec<f64>, FastOutputValues, usize, SolveStats)>, String> {
    let mut out = Vec::with_capacity(trajectory_count);
    let mut parameter_values = vec![Complex64::ZERO; parameter_slot_indices.len()];
    for trajectory in 0..trajectory_count {
        fill_grid_parameter_values(
            trajectory,
            axes,
            axis_offsets,
            axis_lengths,
            strides,
            parameter_values.as_mut_slice(),
        );
        out.push(solve_one(
            plan,
            solver,
            y0,
            t0,
            t1,
            options,
            output_options,
            collect_stats,
            parameter_slot_indices,
            parameter_values.as_slice(),
        )?);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn solve_grid_parallel(
    plan: &PreparedLindbladPlan,
    solver: BatchFastSolver,
    y0: &[f64],
    trajectory_count: usize,
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    axes: &[Complex64],
    axis_offsets: &[usize],
    axis_lengths: &[usize],
    strides: &[usize],
) -> Result<Vec<(Vec<f64>, FastOutputValues, usize, SolveStats)>, String> {
    (0..trajectory_count)
        .into_par_iter()
        .map(|trajectory| {
            let mut parameter_values = vec![Complex64::ZERO; parameter_slot_indices.len()];
            fill_grid_parameter_values(
                trajectory,
                axes,
                axis_offsets,
                axis_lengths,
                strides,
                parameter_values.as_mut_slice(),
            );
            solve_one(
                plan,
                solver,
                y0,
                t0,
                t1,
                options,
                output_options,
                collect_stats,
                parameter_slot_indices,
                parameter_values.as_slice(),
            )
        })
        .collect()
}

fn finalize_batch_results(
    solver: BatchFastSolver,
    trajectory_count: usize,
    total_start: Instant,
    results: Vec<(Vec<f64>, FastOutputValues, usize, SolveStats)>,
) -> Result<BatchSolveOutput, String> {
    let (times, first_values, width, first_stats) = results
        .first()
        .cloned()
        .ok_or_else(|| "empty batch result".to_string())?;
    let time_count = times.len();
    let mut stats = Vec::with_capacity(trajectory_count);
    stats.push(first_stats);

    match first_values {
        FastOutputValues::Real(first) => {
            let mut values = Vec::with_capacity(first.len() * trajectory_count);
            values.extend(first);
            for (trajectory, (other_times, other_values, other_width, other_stats)) in
                results.into_iter().enumerate().skip(1)
            {
                check_times(&times, &other_times, trajectory)?;
                if other_width != width {
                    return Err(format!(
                        "trajectory {trajectory} output width {other_width} does not match {width}"
                    ));
                }
                match other_values {
                    FastOutputValues::Real(row) => values.extend(row),
                    _ => return Err("batch output value type mismatch".to_string()),
                }
                stats.push(other_stats);
            }
            Ok(BatchSolveOutput {
                times,
                values: BatchOutputValues::Real(values),
                width,
                time_count,
                stats: merge_stats(solver, &stats, total_start.elapsed().as_secs_f64()),
            })
        }
        FastOutputValues::Complex(first) => {
            let mut values = Vec::with_capacity(first.len() * trajectory_count);
            values.extend(first);
            for (trajectory, (other_times, other_values, other_width, other_stats)) in
                results.into_iter().enumerate().skip(1)
            {
                check_times(&times, &other_times, trajectory)?;
                if other_width != width {
                    return Err(format!(
                        "trajectory {trajectory} output width {other_width} does not match {width}"
                    ));
                }
                match other_values {
                    FastOutputValues::Complex(row) => values.extend(row),
                    _ => return Err("batch output value type mismatch".to_string()),
                }
                stats.push(other_stats);
            }
            Ok(BatchSolveOutput {
                times,
                values: BatchOutputValues::Complex(values),
                width,
                time_count,
                stats: merge_stats(solver, &stats, total_start.elapsed().as_secs_f64()),
            })
        }
        FastOutputValues::Full(_) => Err("batch output='full' is not supported".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn solve_lindblad_batch(
    plan: &PreparedLindbladPlan,
    solver: BatchFastSolver,
    y0_batch: &[f64],
    trajectory_count: usize,
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    parameter_batch: Option<&[Complex64]>,
    parallel: bool,
    threads: Option<usize>,
) -> Result<BatchSolveOutput, String> {
    let total_start = Instant::now();
    let dim = plan.layout.packed_len();
    if trajectory_count == 0 {
        return Err("trajectory_count must be positive".to_string());
    }
    if y0_batch.len() != trajectory_count * dim {
        return Err(format!(
            "rho0 batch length {} does not match trajectory_count * packed_len = {}",
            y0_batch.len(),
            trajectory_count * dim
        ));
    }
    if matches!(output_options.kind, FastOutputKind::Full) {
        return Err("batch output='full' is intentionally not supported yet".to_string());
    }
    if output_options.when == FastOutputWhen::Saveat && options.saveat.is_none() {
        return Err("batch output_when='saveat' requires explicit saveat values".to_string());
    }
    let parameter_width = parameter_slot_indices.len();
    if let Some(values) = parameter_batch {
        if parameter_width == 0 {
            return Err("parameter_batch was provided but parameter slots are empty".to_string());
        }
        if values.len() != trajectory_count * parameter_width {
            return Err(format!(
                "parameter_batch length {} does not match trajectory_count * parameter width = {}",
                values.len(),
                trajectory_count * parameter_width
            ));
        }
    } else if parameter_width != 0 {
        return Err("parameter slots were provided without parameter_batch".to_string());
    }

    let results = if parallel {
        if let Some(thread_count) = threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .map_err(|err| err.to_string())?
                .install(|| {
                    solve_batch_parallel(
                        plan,
                        solver,
                        y0_batch,
                        trajectory_count,
                        t0,
                        t1,
                        options,
                        output_options,
                        collect_stats,
                        parameter_slot_indices,
                        parameter_batch,
                    )
                })?
        } else {
            solve_batch_parallel(
                plan,
                solver,
                y0_batch,
                trajectory_count,
                t0,
                t1,
                options,
                output_options,
                collect_stats,
                parameter_slot_indices,
                parameter_batch,
            )?
        }
    } else {
        solve_batch_serial(
            plan,
            solver,
            y0_batch,
            trajectory_count,
            t0,
            t1,
            options,
            output_options,
            collect_stats,
            parameter_slot_indices,
            parameter_batch,
        )?
    };

    finalize_batch_results(solver, trajectory_count, total_start, results)
}

#[allow(clippy::too_many_arguments)]
pub fn solve_lindblad_grid_batch(
    plan: &PreparedLindbladPlan,
    solver: BatchFastSolver,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    parameter_axes: &[Complex64],
    parameter_axis_lengths: &[usize],
    parallel: bool,
    threads: Option<usize>,
) -> Result<BatchSolveOutput, String> {
    let total_start = Instant::now();
    let dim = plan.layout.packed_len();
    if y0.len() != dim {
        return Err(format!(
            "rho0 length {} does not match packed_len {}",
            y0.len(),
            dim
        ));
    }
    if matches!(output_options.kind, FastOutputKind::Full) {
        return Err("batch output='full' is intentionally not supported yet".to_string());
    }
    if output_options.when == FastOutputWhen::Saveat && options.saveat.is_none() {
        return Err("batch output_when='saveat' requires explicit saveat values".to_string());
    }
    if parameter_slot_indices.len() != parameter_axis_lengths.len() {
        return Err(format!(
            "parameter slot count {} does not match axis count {}",
            parameter_slot_indices.len(),
            parameter_axis_lengths.len()
        ));
    }
    let expected_axis_values: usize = parameter_axis_lengths.iter().sum();
    if parameter_axes.len() != expected_axis_values {
        return Err(format!(
            "parameter_axes length {} does not match sum(axis_lengths) {}",
            parameter_axes.len(),
            expected_axis_values
        ));
    }
    let trajectory_count = grid_trajectory_count(parameter_axis_lengths)?;
    let mut axis_offsets = Vec::with_capacity(parameter_axis_lengths.len());
    let mut offset = 0usize;
    for &length in parameter_axis_lengths {
        axis_offsets.push(offset);
        offset += length;
    }
    let strides = grid_strides(parameter_axis_lengths);

    let results = if parallel {
        if let Some(thread_count) = threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .map_err(|err| err.to_string())?
                .install(|| {
                    solve_grid_parallel(
                        plan,
                        solver,
                        y0,
                        trajectory_count,
                        t0,
                        t1,
                        options,
                        output_options,
                        collect_stats,
                        parameter_slot_indices,
                        parameter_axes,
                        axis_offsets.as_slice(),
                        parameter_axis_lengths,
                        strides.as_slice(),
                    )
                })?
        } else {
            solve_grid_parallel(
                plan,
                solver,
                y0,
                trajectory_count,
                t0,
                t1,
                options,
                output_options,
                collect_stats,
                parameter_slot_indices,
                parameter_axes,
                axis_offsets.as_slice(),
                parameter_axis_lengths,
                strides.as_slice(),
            )?
        }
    } else {
        solve_grid_serial(
            plan,
            solver,
            y0,
            trajectory_count,
            t0,
            t1,
            options,
            output_options,
            collect_stats,
            parameter_slot_indices,
            parameter_axes,
            axis_offsets.as_slice(),
            parameter_axis_lengths,
            strides.as_slice(),
        )?
    };

    finalize_batch_results(solver, trajectory_count, total_start, results)
}
