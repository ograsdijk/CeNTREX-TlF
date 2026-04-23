use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::{
    rhs_packed_into, rhs_packed_into_with_profile, ExecutionMode, RhsProfileStats, RhsWorkspace,
};
use crate::lindblad::solver_stats::SolveStats;
use ode_solvers::continuous_output_model::ContinuousOutputModel;
use ode_solvers::dop_shared::{IntegrationError, OutputType, System};
use ode_solvers::dopri5::Dopri5;
use ode_solvers::DVector;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct OdeSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub saveat: Option<Vec<f64>>,
    pub save_start: bool,
    pub maxiters: usize,
    pub mode: ExecutionMode,
}

struct LindbladOdeSystem<'a> {
    plan: &'a PreparedLindbladPlan,
    mode: ExecutionMode,
    workspace: RefCell<RhsWorkspace>,
    error: Rc<RefCell<Option<String>>>,
    profile: Option<Rc<RefCell<RhsProfileStats>>>,
}

impl System<f64, DVector<f64>> for LindbladOdeSystem<'_> {
    fn system(&self, x: f64, y: &DVector<f64>, dy: &mut DVector<f64>) {
        let mut workspace = self.workspace.borrow_mut();
        let result = if let Some(profile) = &self.profile {
            let mut stats = profile.borrow_mut();
            rhs_packed_into_with_profile(
                &self.plan,
                y.as_slice(),
                x,
                self.mode,
                &mut workspace,
                dy.as_mut_slice(),
                Some(&mut stats),
            )
        } else {
            rhs_packed_into(
                &self.plan,
                y.as_slice(),
                x,
                self.mode,
                &mut workspace,
                dy.as_mut_slice(),
            )
        };
        if let Err(err) = result {
            *self.error.borrow_mut() = Some(err);
            dy.as_mut_slice().fill(0.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DenseOutputPlan {
    dx: f64,
    skip_start: bool,
}

fn close_to_start(time: f64, t0: f64) -> bool {
    (time - t0).abs() <= 1e-14
}

fn close_to_grid(time: f64, expected: f64, dx: f64) -> bool {
    (time - expected).abs() <= (dx.abs() * 1e-7).max(1e-14)
}

fn requested_times(saveat: &[f64], t0: f64, save_start: bool) -> Vec<f64> {
    let starts_at_t0 = saveat.first().is_some_and(|&time| close_to_start(time, t0));
    let mut times = Vec::with_capacity(saveat.len() + usize::from(save_start && !starts_at_t0));
    if save_start {
        times.push(t0);
    }
    for &time in saveat {
        if save_start && close_to_start(time, t0) {
            continue;
        }
        times.push(time);
    }
    times
}

fn dense_output_plan(
    saveat: &[f64],
    t0: f64,
    t1: f64,
    save_start: bool,
) -> Option<DenseOutputPlan> {
    let times = requested_times(saveat, t0, save_start);
    if times.is_empty() || (save_start && times.len() < 2) {
        return None;
    }
    let dx = if save_start {
        times[1] - times[0]
    } else {
        times[0] - t0
    };
    if !dx.is_finite() || dx <= 0.0 {
        return None;
    }
    if save_start && !close_to_start(times[0], t0) {
        return None;
    }
    if !close_to_grid(*times.last()?, t1, dx) {
        return None;
    }
    for (idx, &time) in times.iter().enumerate() {
        let expected_index = idx + usize::from(!save_start);
        let expected = t0 + expected_index as f64 * dx;
        if !close_to_grid(time, expected, dx) {
            return None;
        }
    }
    Some(DenseOutputPlan {
        dx,
        skip_start: !save_start,
    })
}

fn map_integration_error(err: IntegrationError) -> String {
    err.to_string()
}

pub fn solve_dopri5(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (times, states, _) = solve_dopri5_impl(plan, y0, t0, t1, options, false)?;
    Ok((times, states))
}

pub fn solve_dopri5_with_stats(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>, SolveStats), String> {
    solve_dopri5_impl(plan, y0, t0, t1, options, true)
}

fn solve_dopri5_impl(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    collect_stats: bool,
) -> Result<(Vec<f64>, Vec<f64>, SolveStats), String> {
    let total_start = Instant::now();
    if y0.len() != plan.layout.packed_len() {
        return Err(format!(
            "expected packed state length {}, got {}",
            plan.layout.packed_len(),
            y0.len()
        ));
    }
    if options.maxiters == 0 {
        return Err("maxiters must be positive".to_string());
    }
    if t1 < t0 {
        return Err("only forward integration is supported".to_string());
    }
    let maxiters_u32 = u32::try_from(options.maxiters)
        .map_err(|_| format!("maxiters {} exceeds u32::MAX", options.maxiters))?;
    let rhs_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let profile = collect_stats.then(|| Rc::new(RefCell::new(RhsProfileStats::default())));
    let setup_start = Instant::now();
    let dense_plan = options
        .saveat
        .as_deref()
        .and_then(|saveat| dense_output_plan(saveat, t0, t1, options.save_start));
    let output_type = if dense_plan.is_some() {
        OutputType::Dense
    } else if options.saveat.is_some() {
        OutputType::Continuous
    } else {
        OutputType::Sparse
    };
    let output_dx = dense_plan.map_or(t1 - t0, |plan| plan.dx);
    let initial_step = if options.dt.is_finite() && options.dt > 0.0 {
        options.dt.min(t1 - t0)
    } else {
        0.0
    };
    let system = LindbladOdeSystem {
        plan,
        mode: options.mode,
        workspace: RefCell::new(RhsWorkspace::new(plan)),
        error: rhs_error.clone(),
        profile: profile.clone(),
    };
    let y0 = DVector::from_vec(y0.to_vec());
    let mut stepper = Dopri5::from_param(
        system,
        t0,
        t1,
        output_dx,
        y0,
        options.reltol,
        options.abstol,
        0.9,
        0.04,
        0.2,
        10.0,
        t1 - t0,
        initial_step,
        maxiters_u32,
        1000,
        output_type,
    );
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let check_rhs_error = || -> Result<(), String> {
        if let Some(err) = rhs_error.borrow_mut().take() {
            return Err(format!("lindblad rhs failed inside ode_solvers: {err}"));
        }
        Ok(())
    };

    if let (Some(saveat), Some(dense_plan)) = (options.saveat.as_ref(), dense_plan) {
        let integration_start = Instant::now();
        let dopri_stats = stepper.integrate().map_err(map_integration_error)?;
        let integration_seconds = integration_start.elapsed().as_secs_f64();
        check_rhs_error()?;

        let times = requested_times(saveat, t0, options.save_start);
        let start = usize::from(dense_plan.skip_start);
        let end = start + times.len();
        if stepper.y_out().len() < end {
            return Err(format!(
                "dense output produced {} points, expected at least {}",
                stepper.y_out().len(),
                end
            ));
        }
        for (idx, &time) in times.iter().enumerate() {
            let actual = stepper.x_out()[start + idx];
            if !close_to_grid(actual, time, dense_plan.dx) {
                return Err(format!(
                    "dense output time mismatch at index {idx}: expected {time}, got {actual}"
                ));
            }
        }
        let mut states = Vec::with_capacity(times.len() * plan.layout.packed_len());
        for state in &stepper.y_out()[start..end] {
            states.extend_from_slice(state.as_slice());
        }
        let stats = finish_stats(
            profile,
            dopri_stats,
            times.len(),
            setup_seconds,
            integration_seconds,
            0.0,
            total_start.elapsed().as_secs_f64(),
        );
        return Ok((times, states, stats));
    }

    if let Some(saveat) = options.saveat.as_ref() {
        let mut output = ContinuousOutputModel::default();
        let integration_start = Instant::now();
        let dopri_stats = stepper
            .integrate_with_continuous_output_model(&mut output)
            .map_err(map_integration_error)?;
        let integration_seconds = integration_start.elapsed().as_secs_f64();
        check_rhs_error()?;

        let interpolation_start = Instant::now();
        let mut times = Vec::with_capacity(saveat.len() + usize::from(options.save_start));
        let mut states = Vec::with_capacity(
            (saveat.len() + usize::from(options.save_start)) * plan.layout.packed_len(),
        );
        let mut push_time = |time: f64| -> Result<(), String> {
            let value: DVector<f64> = output
                .evaluate(time)
                .ok_or_else(|| format!("continuous output unavailable at t={time}"))?;
            times.push(time);
            states.extend_from_slice(value.as_slice());
            Ok(())
        };
        if options.save_start {
            push_time(t0)?;
        }
        for &time in saveat {
            if (time - t0).abs() <= 1e-14 {
                continue;
            }
            push_time(time)?;
        }
        let interpolation_seconds = interpolation_start.elapsed().as_secs_f64();
        let stats = finish_stats(
            profile,
            dopri_stats,
            times.len(),
            setup_seconds,
            integration_seconds,
            interpolation_seconds,
            total_start.elapsed().as_secs_f64(),
        );
        return Ok((times, states, stats));
    }

    let integration_start = Instant::now();
    let dopri_stats = stepper.integrate().map_err(map_integration_error)?;
    let integration_seconds = integration_start.elapsed().as_secs_f64();
    check_rhs_error()?;
    let times = stepper.x_out().clone();
    let mut states = Vec::with_capacity(times.len() * plan.layout.packed_len());
    for state in stepper.y_out().iter() {
        states.extend_from_slice(state.as_slice());
    }
    let stats = finish_stats(
        profile,
        dopri_stats,
        times.len(),
        setup_seconds,
        integration_seconds,
        0.0,
        total_start.elapsed().as_secs_f64(),
    );
    Ok((times, states, stats))
}

fn finish_stats(
    profile: Option<Rc<RefCell<RhsProfileStats>>>,
    dopri_stats: ode_solvers::dop_shared::Stats,
    saved_points: usize,
    setup_seconds: f64,
    integration_seconds: f64,
    interpolation_seconds: f64,
    total_seconds: f64,
) -> SolveStats {
    let rhs_profile = profile
        .as_ref()
        .map(|stats| stats.borrow().clone())
        .unwrap_or_default();
    SolveStats {
        solver: "dopri5".to_string(),
        rhs_calls: rhs_profile.calls,
        jacobian_calls: 0,
        function_evaluations: u64::from(dopri_stats.num_eval),
        accepted_steps: u64::from(dopri_stats.accepted_steps),
        rejected_steps: u64::from(dopri_stats.rejected_steps),
        internal_steps: u64::from(dopri_stats.accepted_steps + dopri_stats.rejected_steps),
        saved_points: saved_points as u64,
        setup_seconds,
        integration_seconds,
        interpolation_seconds,
        total_seconds,
        rhs_seconds: rhs_profile.total_seconds,
        jacobian_seconds: 0.0,
    }
}
