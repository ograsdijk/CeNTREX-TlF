use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::{
    rhs_packed_into, rhs_packed_into_with_profile, ExecutionMode, RhsProfileStats, RhsWorkspace,
};
use crate::lindblad::solver_fast_common::{
    FastOutputKind, FastOutputOptions, FastOutputValues, FastOutputWhen, FastSolveOutput,
};
use crate::lindblad::solver_ode::OdeSolverOptions;
use crate::lindblad::solver_stats::SolveStats;
use num_complex::Complex64;
use std::time::Instant;

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
const C6: f64 = 1.0;
const C7: f64 = 1.0;

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const A71: f64 = 35.0 / 384.0;
const A72: f64 = 0.0;
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;

const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

const D1: f64 = -12715105075.0 / 11282082432.0;
const D3: f64 = 87487479700.0 / 32700410799.0;
const D4: f64 = -10690763975.0 / 1880347072.0;
const D5: f64 = 701980252875.0 / 199316789632.0;
const D6: f64 = -1453857185.0 / 822651844.0;
const D7: f64 = 69997945.0 / 29380423.0;

#[derive(Clone, Copy, Debug)]
struct Controller {
    alpha: f64,
    beta: f64,
    facc1: f64,
    facc2: f64,
    fac_old: f64,
    h_max: f64,
    reject: bool,
    safety_factor: f64,
    posneg: f64,
}

impl Controller {
    fn new(t0: f64, t1: f64) -> Self {
        let beta = 0.04;
        Self {
            alpha: 0.2 - beta * 0.75,
            beta,
            facc1: 1.0 / 0.2,
            facc2: 1.0 / 10.0,
            fac_old: 1.0e-4,
            h_max: (t1 - t0).abs(),
            reject: false,
            safety_factor: 0.9,
            posneg: sign(1.0, t1 - t0),
        }
    }

    fn accept(&mut self, err: f64, h: f64) -> (bool, f64) {
        let fac11 = err.powf(self.alpha);
        let mut fac = fac11 * self.fac_old.powf(-self.beta);
        fac = self.facc2.max(self.facc1.min(fac / self.safety_factor));
        let mut h_new = h / fac;

        if err <= 1.0 {
            self.fac_old = err.max(1.0e-4);
            if h_new.abs() > self.h_max {
                h_new = self.posneg * self.h_max;
            }
            if self.reject {
                h_new = self.posneg * h_new.abs().min(h.abs());
            }
            self.reject = false;
            (true, h_new)
        } else {
            h_new = h / self.facc1.min(fac11 / self.safety_factor);
            self.reject = true;
            (false, h_new)
        }
    }
}

#[derive(Clone, Debug)]
struct FastStats {
    rhs_calls: u64,
    function_evaluations: u64,
    accepted_steps: u64,
    rejected_steps: u64,
    rhs_seconds: f64,
}

impl FastStats {
    fn record_rhs(&mut self, seconds: f64) {
        self.rhs_calls += 1;
        self.function_evaluations += 1;
        self.rhs_seconds += seconds;
    }
}

impl Default for FastStats {
    fn default() -> Self {
        Self {
            rhs_calls: 0,
            function_evaluations: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            rhs_seconds: 0.0,
        }
    }
}

struct FastRhs<'a> {
    plan: &'a PreparedLindbladPlan,
    mode: ExecutionMode,
    workspace: RhsWorkspace,
    collect_stats: bool,
    stats: FastStats,
}

impl<'a> FastRhs<'a> {
    fn new(plan: &'a PreparedLindbladPlan, mode: ExecutionMode, collect_stats: bool) -> Self {
        Self {
            plan,
            mode,
            workspace: RhsWorkspace::new(plan),
            collect_stats,
            stats: FastStats::default(),
        }
    }

    fn new_with_parameter_overrides(
        plan: &'a PreparedLindbladPlan,
        mode: ExecutionMode,
        collect_stats: bool,
        parameter_slot_indices: &[usize],
        parameter_values: &[Complex64],
    ) -> Result<Self, String> {
        let mut rhs = Self::new(plan, mode, collect_stats);
        rhs.workspace
            .set_scalar_parameter_overrides(parameter_slot_indices, parameter_values)?;
        Ok(rhs)
    }

    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        if self.collect_stats {
            let start = Instant::now();
            let mut profile = RhsProfileStats::default();
            let result = rhs_packed_into_with_profile(
                self.plan,
                y,
                t,
                self.mode,
                &mut self.workspace,
                dy,
                Some(&mut profile),
            );
            self.stats.record_rhs(start.elapsed().as_secs_f64());
            result
        } else {
            self.stats.function_evaluations += 1;
            rhs_packed_into(self.plan, y, t, self.mode, &mut self.workspace, dy)
        }
    }
}

#[derive(Clone, Debug)]
struct SavePlan {
    times: Vec<f64>,
}

struct OutputBuffer {
    times: Vec<f64>,
    values: FastOutputValues,
    width: usize,
    kind: FastOutputKind,
    n: usize,
}

impl OutputBuffer {
    fn new(kind: FastOutputKind, n: usize, packed_len: usize, capacity: usize) -> Self {
        let width = match &kind {
            FastOutputKind::Full => packed_len,
            FastOutputKind::Populations => n,
            FastOutputKind::Selected(indices) => indices.len(),
        };
        let values = match &kind {
            FastOutputKind::Full => FastOutputValues::Full(Vec::with_capacity(capacity * width)),
            FastOutputKind::Populations => {
                FastOutputValues::Real(Vec::with_capacity(capacity * width))
            }
            FastOutputKind::Selected(_) => {
                FastOutputValues::Complex(Vec::with_capacity(capacity * width))
            }
        };
        Self {
            times: Vec::with_capacity(capacity),
            values,
            width,
            kind,
            n,
        }
    }

    fn push_state(&mut self, time: f64, state: &[f64]) {
        self.times.push(time);
        match (&self.kind, &mut self.values) {
            (FastOutputKind::Full, FastOutputValues::Full(values)) => {
                values.extend_from_slice(state);
            }
            (FastOutputKind::Populations, FastOutputValues::Real(values)) => {
                values.extend_from_slice(&state[..self.n]);
            }
            (FastOutputKind::Selected(indices), FastOutputValues::Complex(values)) => {
                for &(i, j) in indices {
                    values.push(packed_entry(self.n, state, i, j));
                }
            }
            _ => unreachable!("output kind/value storage mismatch"),
        }
    }
}

fn sign(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

fn close_to_start(time: f64, t0: f64) -> bool {
    (time - t0).abs() <= 1e-14
}

fn time_tol(time: f64) -> f64 {
    (time.abs() * 1e-12).max(1e-14)
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

fn save_plan(
    saveat: Option<&[f64]>,
    t0: f64,
    t1: f64,
    save_start: bool,
) -> Result<Option<SavePlan>, String> {
    let Some(saveat) = saveat else {
        return Ok(None);
    };
    let times = requested_times(saveat, t0, save_start);
    let mut previous = t0;
    for (idx, &time) in times.iter().enumerate() {
        if !time.is_finite() {
            return Err(format!("saveat contains non-finite value at index {idx}"));
        }
        if time < t0 - time_tol(t0) || time > t1 + time_tol(t1) {
            return Err(format!("saveat value {time} is outside [{t0}, {t1}]"));
        }
        if idx > 0 && time < previous - time_tol(previous) {
            return Err("dopri5_fast requires saveat values in ascending order".to_string());
        }
        previous = time;
    }
    Ok(Some(SavePlan { times }))
}

fn validate_output_options(n: usize, output_options: &FastOutputOptions) -> Result<(), String> {
    if let FastOutputKind::Selected(indices) = &output_options.kind {
        if indices.is_empty() {
            return Err("output='selected' requires at least one output index".to_string());
        }
        for &(i, j) in indices {
            if i >= n || j >= n {
                return Err(format!(
                    "selected output index ({i}, {j}) out of bounds for size {n}"
                ));
            }
        }
    }
    Ok(())
}

fn packed_upper_real_index(n: usize, i: usize, j: usize) -> usize {
    let mut offset = 0usize;
    for row in 0..i {
        offset += n - row - 1;
    }
    n + 2 * (offset + (j - i - 1))
}

fn packed_entry(n: usize, state: &[f64], i: usize, j: usize) -> Complex64 {
    if i == j {
        return Complex64::new(state[i], 0.0);
    }
    if i < j {
        let real_idx = packed_upper_real_index(n, i, j);
        Complex64::new(state[real_idx], state[real_idx + 1])
    } else {
        let real_idx = packed_upper_real_index(n, j, i);
        Complex64::new(state[real_idx], -state[real_idx + 1])
    }
}

pub fn solve_dopri5_fast(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let output = solve_dopri5_fast_output(
        plan,
        y0,
        t0,
        t1,
        options,
        &FastOutputOptions::default(),
        false,
    )?;
    match output.values {
        FastOutputValues::Full(states) => Ok((output.times, states)),
        _ => unreachable!("default dopri5_fast output must be full"),
    }
}

pub fn solve_dopri5_fast_with_stats(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>, SolveStats), String> {
    let output = solve_dopri5_fast_output(
        plan,
        y0,
        t0,
        t1,
        options,
        &FastOutputOptions::default(),
        true,
    )?;
    match output.values {
        FastOutputValues::Full(states) => Ok((output.times, states, output.stats)),
        _ => unreachable!("default dopri5_fast output must be full"),
    }
}

pub fn solve_dopri5_fast_output(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
) -> Result<FastSolveOutput, String> {
    solve_dopri5_fast_output_inner(
        plan,
        y0,
        t0,
        t1,
        options,
        output_options,
        collect_stats,
        &[],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn solve_dopri5_fast_output_with_parameter_overrides(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    parameter_values: &[Complex64],
) -> Result<FastSolveOutput, String> {
    solve_dopri5_fast_output_inner(
        plan,
        y0,
        t0,
        t1,
        options,
        output_options,
        collect_stats,
        parameter_slot_indices,
        parameter_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_dopri5_fast_output_inner(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
    output_options: &FastOutputOptions,
    collect_stats: bool,
    parameter_slot_indices: &[usize],
    parameter_values: &[Complex64],
) -> Result<FastSolveOutput, String> {
    let total_start = Instant::now();
    let dim = plan.layout.packed_len();
    validate_output_options(plan.layout.n, output_options)?;
    if parameter_slot_indices.len() != parameter_values.len() {
        return Err(format!(
            "parameter override slot count {} does not match value count {}",
            parameter_slot_indices.len(),
            parameter_values.len()
        ));
    }
    if y0.len() != dim {
        return Err(format!(
            "expected packed state length {}, got {}",
            dim,
            y0.len()
        ));
    }
    if options.maxiters == 0 {
        return Err("maxiters must be positive".to_string());
    }
    if t1 < t0 {
        return Err("only forward integration is supported".to_string());
    }
    if t1 == t0 {
        let capacity =
            usize::from(options.save_start || output_options.when == FastOutputWhen::Final);
        let mut output =
            OutputBuffer::new(output_options.kind.clone(), plan.layout.n, dim, capacity);
        if options.save_start || output_options.when == FastOutputWhen::Final {
            output.push_state(t0, y0);
        }
        let stats = SolveStats {
            solver: "dopri5_fast".to_string(),
            saved_points: output.times.len() as u64,
            total_seconds: total_start.elapsed().as_secs_f64(),
            ..SolveStats::default()
        };
        return Ok(FastSolveOutput {
            times: output.times,
            values: output.values,
            width: output.width,
            stats,
        });
    }

    let setup_start = Instant::now();
    let save_plan = if output_options.when == FastOutputWhen::Final {
        None
    } else {
        save_plan(options.saveat.as_deref(), t0, t1, options.save_start)?
    };
    let saved_capacity = if output_options.when == FastOutputWhen::Final {
        1
    } else {
        save_plan
            .as_ref()
            .map_or(options.maxiters + 1, |plan| plan.times.len())
    };
    let mut output = OutputBuffer::new(
        output_options.kind.clone(),
        plan.layout.n,
        dim,
        saved_capacity,
    );

    let mut rhs = FastRhs::new_with_parameter_overrides(
        plan,
        options.mode,
        collect_stats,
        parameter_slot_indices,
        parameter_values,
    )?;
    let mut y = y0.to_vec();
    let mut y_next = vec![0.0; dim];
    let mut y_tmp = vec![0.0; dim];
    let mut y_stiff = vec![0.0; dim];
    let mut rcont0 = vec![0.0; dim];
    let mut rcont1 = vec![0.0; dim];
    let mut rcont2 = vec![0.0; dim];
    let mut rcont3 = vec![0.0; dim];
    let mut rcont4 = vec![0.0; dim];
    let mut dense_scratch = vec![0.0; dim];
    let mut k = vec![0.0; 7 * dim];
    let mut f0 = vec![0.0; dim];
    let mut f1 = vec![0.0; dim];

    let mut x = t0;
    let mut h = if options.dt.is_finite() && options.dt > 0.0 {
        options.dt.min(t1 - t0)
    } else {
        hinit(
            &mut rhs,
            &y,
            x,
            t1,
            options.abstol,
            options.reltol,
            &mut f0,
            &mut f1,
            &mut y_tmp,
        )?
    };
    let mut controller = Controller::new(t0, t1);
    let posneg = sign(1.0, t1 - t0);
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let integration_start = Instant::now();
    rhs.eval(x, &y, &mut k[0..dim])?;

    let mut save_idx = 0usize;
    if let Some(plan) = &save_plan {
        while save_idx < plan.times.len() && close_to_start(plan.times[save_idx], t0) {
            output.push_state(plan.times[save_idx], &y);
            save_idx += 1;
        }
    } else if options.save_start && output_options.when == FastOutputWhen::Saveat {
        output.push_state(x, &y);
    }

    let mut n_step = 0usize;
    let mut last = false;
    let mut non_stiff = 0usize;
    let mut iasti = 0usize;

    while !last {
        if n_step > options.maxiters {
            return Err(format!(
                "Stopped at x = {x}. Need more than {n_step} steps."
            ));
        }
        if 0.1 * h.abs() <= f64::EPSILON * x.abs() {
            return Err(format!("Stopped at x = {x}. Step size underflow."));
        }
        if (x + 1.01 * h - t1) * posneg > 0.0 {
            h = t1 - x;
            last = true;
        }
        n_step += 1;

        fill_stage1(&y, h, &k, dim, &mut y_tmp);
        rhs.eval(x + h * C2, &y_tmp, &mut k[dim..2 * dim])?;
        fill_stage2(&y, h, &k, dim, &mut y_tmp);
        rhs.eval(x + h * C3, &y_tmp, &mut k[2 * dim..3 * dim])?;
        fill_stage3(&y, h, &k, dim, &mut y_tmp);
        rhs.eval(x + h * C4, &y_tmp, &mut k[3 * dim..4 * dim])?;
        fill_stage4(&y, h, &k, dim, &mut y_tmp);
        rhs.eval(x + h * C5, &y_tmp, &mut k[4 * dim..5 * dim])?;
        fill_stage5(&y, h, &k, dim, &mut y_tmp);
        y_stiff.copy_from_slice(&y_tmp);
        rhs.eval(x + h * C6, &y_tmp, &mut k[5 * dim..6 * dim])?;
        fill_solution_stage(&y, h, &k, dim, &mut y_next);
        rhs.eval(x + h * C7, &y_next, &mut k[6 * dim..7 * dim])?;

        let err = error_norm_from_stages(&y, &y_next, h, &k, dim, options.abstol, options.reltol);
        let (accepted, h_new) = controller.accept(err, h);

        if accepted {
            rhs.stats.accepted_steps += 1;

            if rhs.stats.accepted_steps % 1000 == 0 || iasti > 0 {
                let mut num = 0.0;
                let mut den = 0.0;
                for i in 0..dim {
                    let dk = k[6 * dim + i] - k[5 * dim + i];
                    let dy = y_next[i] - y_stiff[i];
                    num += dk * dk;
                    den += dy * dy;
                }
                let h_lamb = if den > 0.0 {
                    h * (num / den).sqrt()
                } else {
                    0.0
                };
                if h_lamb > 3.25 {
                    iasti += 1;
                    non_stiff = 0;
                    if iasti == 15 {
                        return Err(format!("The problem seems to become stiff at x = {x}."));
                    }
                } else {
                    non_stiff += 1;
                    if non_stiff == 6 {
                        iasti = 0;
                    }
                }
            }

            let x_old = x;
            x += h;

            if output_options.when == FastOutputWhen::Saveat {
                if let Some(plan) = &save_plan {
                    let mut dense_ready = false;
                    while save_idx < plan.times.len() && plan.times[save_idx] <= x + time_tol(x) {
                        let save_time = plan.times[save_idx];
                        if save_time >= x_old - time_tol(x_old) {
                            if (save_time - x_old).abs() <= time_tol(x_old) {
                                output.push_state(save_time, &y);
                            } else if (save_time - x).abs() <= time_tol(x) {
                                output.push_state(save_time, &y_next);
                            } else if output_options.dense_output {
                                if !dense_ready {
                                    fill_dense_rcont4(h, &k, &mut rcont4);
                                    fill_dense_coefficients(
                                        &y,
                                        &y_next,
                                        h,
                                        &k,
                                        dim,
                                        &mut rcont0,
                                        &mut rcont1,
                                        &mut rcont2,
                                        &mut rcont3,
                                    );
                                    dense_ready = true;
                                }
                                let theta = (save_time - x_old) / h;
                                fill_dense_output(
                                    theta,
                                    &rcont0,
                                    &rcont1,
                                    &rcont2,
                                    &rcont3,
                                    &rcont4,
                                    &mut dense_scratch,
                                );
                                output.push_state(save_time, &dense_scratch);
                            } else {
                                return Err(
                                    "dense_output=False cannot save interior adaptive-step times"
                                        .to_string(),
                                );
                            }
                        }
                        save_idx += 1;
                    }
                } else {
                    output.push_state(x, &y_next);
                }
            }

            std::mem::swap(&mut y, &mut y_next);
            let (first_stage, later_stages) = k.split_at_mut(dim);
            first_stage.copy_from_slice(&later_stages[5 * dim..6 * dim]);

            if last {
                break;
            }
        } else {
            last = false;
            if rhs.stats.accepted_steps >= 1 {
                rhs.stats.rejected_steps += 1;
            }
        }
        h = h_new;
    }

    if let Some(plan) = &save_plan {
        if save_idx != plan.times.len() {
            return Err(format!(
                "integration ended before all saveat values were written: {} of {}",
                save_idx,
                plan.times.len()
            ));
        }
    }

    let integration_seconds = integration_start.elapsed().as_secs_f64();
    if output_options.when == FastOutputWhen::Final {
        output.push_state(t1, &y);
    }

    let stats = finish_stats(
        rhs.stats,
        output.times.len(),
        setup_seconds,
        integration_seconds,
        total_start.elapsed().as_secs_f64(),
    );
    Ok(FastSolveOutput {
        times: output.times,
        values: output.values,
        width: output.width,
        stats,
    })
}

fn hinit(
    rhs: &mut FastRhs<'_>,
    y: &[f64],
    x: f64,
    x_end: f64,
    abstol: f64,
    reltol: f64,
    f0: &mut [f64],
    f1: &mut [f64],
    y_tmp: &mut [f64],
) -> Result<f64, String> {
    rhs.eval(x, y, f0)?;
    let posneg = sign(1.0, x_end - x);
    let h_max = (x_end - x).abs();
    let mut d0 = 0.0;
    let mut d1 = 0.0;
    for i in 0..y.len() {
        let sci = abstol + y[i].abs() * reltol;
        d0 += (y[i] / sci) * (y[i] / sci);
        d1 += (f0[i] / sci) * (f0[i] / sci);
    }
    let mut h0 = if d0 < 1.0e-10 || d1 < 1.0e-10 {
        1.0e-6
    } else {
        0.01 * (d0 / d1).sqrt()
    };
    h0 = h0.min(h_max);
    h0 = sign(h0, posneg);

    for i in 0..y.len() {
        y_tmp[i] = y[i] + f0[i] * h0;
    }
    rhs.eval(x + h0, y_tmp, f1)?;

    let mut d2 = 0.0;
    for i in 0..y.len() {
        let sci = abstol + y[i].abs() * reltol;
        d2 += ((f1[i] - f0[i]) / sci) * ((f1[i] - f0[i]) / sci);
    }
    d2 = d2.sqrt() / h0;
    let h1 = if d1.sqrt().max(d2.abs()) <= 1.0e-15 {
        1.0e-6_f64.max(h0.abs() * 1.0e-3)
    } else {
        (0.01 / d1.sqrt().max(d2)).powf(1.0 / 5.0)
    };
    Ok(sign((100.0 * h0.abs()).min(h1.min(h_max)), posneg))
}

fn fill_stage1(y: &[f64], h: f64, k: &[f64], _dim: usize, out: &mut [f64]) {
    for i in 0..y.len() {
        out[i] = y[i] + h * A21 * k[i];
    }
}

fn fill_stage2(y: &[f64], h: f64, k: &[f64], dim: usize, out: &mut [f64]) {
    for i in 0..y.len() {
        out[i] = y[i] + h * (A31 * k[i] + A32 * k[dim + i]);
    }
}

fn fill_stage3(y: &[f64], h: f64, k: &[f64], dim: usize, out: &mut [f64]) {
    for i in 0..y.len() {
        out[i] = y[i] + h * (A41 * k[i] + A42 * k[dim + i] + A43 * k[2 * dim + i]);
    }
}

fn fill_stage4(y: &[f64], h: f64, k: &[f64], dim: usize, out: &mut [f64]) {
    for i in 0..y.len() {
        out[i] = y[i]
            + h * (A51 * k[i] + A52 * k[dim + i] + A53 * k[2 * dim + i] + A54 * k[3 * dim + i]);
    }
}

fn fill_stage5(y: &[f64], h: f64, k: &[f64], dim: usize, out: &mut [f64]) {
    for i in 0..y.len() {
        out[i] = y[i]
            + h * (A61 * k[i]
                + A62 * k[dim + i]
                + A63 * k[2 * dim + i]
                + A64 * k[3 * dim + i]
                + A65 * k[4 * dim + i]);
    }
}

fn fill_solution_stage(y: &[f64], h: f64, k: &[f64], dim: usize, out: &mut [f64]) {
    for i in 0..y.len() {
        out[i] = y[i]
            + h * (A71 * k[i]
                + A72 * k[dim + i]
                + A73 * k[2 * dim + i]
                + A74 * k[3 * dim + i]
                + A75 * k[4 * dim + i]
                + A76 * k[5 * dim + i]);
    }
}

fn fill_dense_rcont4(h: f64, k: &[f64], out: &mut [f64]) {
    let dim = out.len();
    for i in 0..out.len() {
        out[i] = h
            * (D1 * k[i]
                + D3 * k[2 * dim + i]
                + D4 * k[3 * dim + i]
                + D5 * k[4 * dim + i]
                + D6 * k[5 * dim + i]
                + D7 * k[6 * dim + i]);
    }
}

fn error_norm_from_stages(
    y: &[f64],
    y_next: &[f64],
    h: f64,
    k: &[f64],
    dim: usize,
    abstol: f64,
    reltol: f64,
) -> f64 {
    let mut err = 0.0;
    for i in 0..y.len() {
        let sc = abstol + y[i].abs().max(y_next[i].abs()) * reltol;
        let err_i = h
            * (E1 * k[i]
                + E3 * k[2 * dim + i]
                + E4 * k[3 * dim + i]
                + E5 * k[4 * dim + i]
                + E6 * k[5 * dim + i]
                + E7 * k[6 * dim + i]);
        let scaled = err_i / sc;
        err += scaled * scaled;
    }
    (err / y.len() as f64).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn fill_dense_coefficients(
    y: &[f64],
    y_next: &[f64],
    h: f64,
    k: &[f64],
    dim: usize,
    rcont0: &mut [f64],
    rcont1: &mut [f64],
    rcont2: &mut [f64],
    rcont3: &mut [f64],
) {
    for i in 0..y.len() {
        let ydiff = y_next[i] - y[i];
        let bspl = k[i] * h - ydiff;
        rcont0[i] = y[i];
        rcont1[i] = ydiff;
        rcont2[i] = bspl;
        rcont3[i] = -k[6 * dim + i] * h + ydiff - bspl;
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_dense_output(
    theta: f64,
    rcont0: &[f64],
    rcont1: &[f64],
    rcont2: &[f64],
    rcont3: &[f64],
    rcont4: &[f64],
    out: &mut [f64],
) {
    let theta1 = 1.0 - theta;
    for i in 0..rcont0.len() {
        out[i] = rcont0[i]
            + (rcont1[i] + (rcont2[i] + (rcont3[i] + rcont4[i] * theta1) * theta) * theta1) * theta;
    }
}

fn finish_stats(
    fast_stats: FastStats,
    saved_points: usize,
    setup_seconds: f64,
    integration_seconds: f64,
    total_seconds: f64,
) -> SolveStats {
    SolveStats {
        solver: "dopri5_fast".to_string(),
        rhs_calls: fast_stats.rhs_calls,
        jacobian_calls: 0,
        function_evaluations: fast_stats.function_evaluations,
        accepted_steps: fast_stats.accepted_steps,
        rejected_steps: fast_stats.rejected_steps,
        internal_steps: fast_stats.accepted_steps + fast_stats.rejected_steps,
        saved_points: saved_points as u64,
        setup_seconds,
        integration_seconds,
        interpolation_seconds: 0.0,
        total_seconds,
        rhs_seconds: fast_stats.rhs_seconds,
        jacobian_seconds: 0.0,
    }
}
