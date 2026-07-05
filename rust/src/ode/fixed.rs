use crate::ode::common::{build_save_plan, close_to_start, time_tol};
use crate::ode::output::OdeOutput;
use crate::ode::{OdeOptions, OdeRhs, OdeStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedStepper {
    Dopri5,
    Rk2,
    Rk4,
}

fn crossed_zero(g0: f64, g1: f64) -> bool {
    g0 == 0.0 || g1 == 0.0 || (g0 < 0.0 && g1 > 0.0) || (g0 > 0.0 && g1 < 0.0)
}

fn next_step_limit(save_times: Option<&[f64]>, save_index: usize, x: f64, t1: f64, dt: f64) -> f64 {
    let mut target = (x + dt).min(t1);
    if let Some(times) = save_times {
        if let Some(&ts) = times.get(save_index) {
            if ts > x + time_tol(x) {
                target = target.min(ts);
            }
        }
    }
    target - x
}

fn rk2_step<R: OdeRhs>(
    rhs: &mut R,
    x: f64,
    h: f64,
    y: &[f64],
    yn: &mut [f64],
    k1: &mut [f64],
    k2: &mut [f64],
    yt: &mut [f64],
) -> Result<u64, String> {
    rhs.eval(x, y, k1)?;
    for i in 0..y.len() {
        yt[i] = y[i] + 0.5 * h * k1[i];
    }
    rhs.eval(x + 0.5 * h, yt, k2)?;
    for i in 0..y.len() {
        yn[i] = y[i] + h * k2[i];
    }
    Ok(2)
}

#[allow(clippy::too_many_arguments)]
fn dopri5_step<R: OdeRhs>(
    rhs: &mut R,
    x: f64,
    h: f64,
    y: &[f64],
    yn: &mut [f64],
    k1: &mut [f64],
    k2: &mut [f64],
    k3: &mut [f64],
    k4: &mut [f64],
    k5: &mut [f64],
    k6: &mut [f64],
    yt: &mut [f64],
) -> Result<u64, String> {
    rhs.eval(x, y, k1)?;
    for i in 0..y.len() {
        yt[i] = y[i] + h * (1.0 / 5.0) * k1[i];
    }
    rhs.eval(x + h * (1.0 / 5.0), yt, k2)?;
    for i in 0..y.len() {
        yt[i] = y[i] + h * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i]);
    }
    rhs.eval(x + h * (3.0 / 10.0), yt, k3)?;
    for i in 0..y.len() {
        yt[i] = y[i] + h * ((44.0 / 45.0) * k1[i] - (56.0 / 15.0) * k2[i] + (32.0 / 9.0) * k3[i]);
    }
    rhs.eval(x + h * (4.0 / 5.0), yt, k4)?;
    for i in 0..y.len() {
        yt[i] = y[i]
            + h * ((19372.0 / 6561.0) * k1[i] - (25360.0 / 2187.0) * k2[i]
                + (64448.0 / 6561.0) * k3[i]
                - (212.0 / 729.0) * k4[i]);
    }
    rhs.eval(x + h * (8.0 / 9.0), yt, k5)?;
    for i in 0..y.len() {
        yt[i] = y[i]
            + h * ((9017.0 / 3168.0) * k1[i] - (355.0 / 33.0) * k2[i]
                + (46732.0 / 5247.0) * k3[i]
                + (49.0 / 176.0) * k4[i]
                - (5103.0 / 18656.0) * k5[i]);
    }
    rhs.eval(x + h, yt, k6)?;
    for i in 0..y.len() {
        yn[i] = y[i]
            + h * ((35.0 / 384.0) * k1[i] + (500.0 / 1113.0) * k3[i] + (125.0 / 192.0) * k4[i]
                - (2187.0 / 6784.0) * k5[i]
                + (11.0 / 84.0) * k6[i]);
    }
    Ok(6)
}

#[allow(clippy::too_many_arguments)]
fn rk4_step<R: OdeRhs>(
    rhs: &mut R,
    x: f64,
    h: f64,
    y: &[f64],
    yn: &mut [f64],
    k1: &mut [f64],
    k2: &mut [f64],
    k3: &mut [f64],
    k4: &mut [f64],
    yt: &mut [f64],
) -> Result<u64, String> {
    rhs.eval(x, y, k1)?;
    for i in 0..y.len() {
        yt[i] = y[i] + 0.5 * h * k1[i];
    }
    rhs.eval(x + 0.5 * h, yt, k2)?;
    for i in 0..y.len() {
        yt[i] = y[i] + 0.5 * h * k2[i];
    }
    rhs.eval(x + 0.5 * h, yt, k3)?;
    for i in 0..y.len() {
        yt[i] = y[i] + h * k3[i];
    }
    rhs.eval(x + h, yt, k4)?;
    for i in 0..y.len() {
        yn[i] = y[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    Ok(4)
}

pub fn solve_fixed_step<R: OdeRhs, O: OdeOutput>(
    rhs: &mut R,
    y0: &[f64],
    t0: f64,
    t1: f64,
    opt: &OdeOptions,
    output: &mut O,
    stepper: FixedStepper,
) -> Result<OdeStats, String> {
    let dim = rhs.dim();
    if y0.len() != dim {
        return Err(format!("expected {dim}, got {}", y0.len()));
    }
    if opt.maxiters == 0 {
        return Err("maxiters must be positive".into());
    }
    if t1 < t0 {
        return Err("only forward integration".into());
    }
    if !opt.dt.is_finite() || opt.dt <= 0.0 {
        return Err("fixed-step solvers require positive finite dt".into());
    }
    if t1 == t0 {
        if opt.save_start {
            output.push(t0, y0);
        }
        return Ok(OdeStats::default());
    }

    let sp = build_save_plan(opt.saveat.as_deref(), t0, t1, opt.save_start)?;
    let save_times = sp.as_ref().map(|plan| plan.times.as_slice());
    let mut st = OdeStats::default();
    let mut y = y0.to_vec();
    let mut yn = vec![0.0; dim];
    let mut yt = vec![0.0; dim];
    let mut k1 = vec![0.0; dim];
    let mut k2 = vec![0.0; dim];
    let mut k3 = vec![0.0; dim];
    let mut k4 = vec![0.0; dim];
    let mut k5 = vec![0.0; dim];
    let mut k6 = vec![0.0; dim];
    let mut x = t0;
    let mut si = 0usize;

    if let Some(times) = save_times {
        while si < times.len() && close_to_start(times[si], t0) {
            output.push(times[si], &y);
            si += 1;
        }
    } else if opt.save_start {
        output.push(x, &y);
    }

    let mut event_old = rhs.event_value(x, &y)?;
    let mut steps = 0usize;
    while x < t1 - time_tol(t1) {
        if steps >= opt.maxiters {
            return Err(format!("Stopped at x={x}. Need more than {steps} steps."));
        }
        let h = next_step_limit(save_times, si, x, t1, opt.dt);
        if h <= 0.0 || 0.1 * h <= f64::EPSILON * x.abs().max(1.0) {
            return Err(format!("Stopped at x={x}. Step size underflow."));
        }

        let rhs_calls = match stepper {
            FixedStepper::Dopri5 => dopri5_step(
                rhs, x, h, &y, &mut yn, &mut k1, &mut k2, &mut k3, &mut k4, &mut k5, &mut k6,
                &mut yt,
            )?,
            FixedStepper::Rk2 => rk2_step(rhs, x, h, &y, &mut yn, &mut k1, &mut k2, &mut yt)?,
            FixedStepper::Rk4 => rk4_step(
                rhs, x, h, &y, &mut yn, &mut k1, &mut k2, &mut k3, &mut k4, &mut yt,
            )?,
        };
        st.rhs_calls += rhs_calls;
        st.accepted_steps += 1;
        steps += 1;

        let xo = x;
        x += h;
        let event_new = rhs.event_value(x, &yn)?;
        let mut event_hit = false;
        if let (Some(g0), Some(g1)) = (event_old, event_new) {
            if crossed_zero(g0, g1) {
                event_hit = true;
                st.event_triggered = true;
                st.event_time = if g0 == 0.0 { xo } else { x };
                st.event_index = 0;
            }
        }

        if let Some(times) = save_times {
            let stop_time = if event_hit { st.event_time } else { x };
            while si < times.len() && times[si] <= stop_time + time_tol(stop_time) {
                output.push(times[si], &yn);
                si += 1;
            }
        } else if !event_hit {
            output.push(x, &yn);
        }

        if event_hit {
            output.push(st.event_time, &yn);
            break;
        }

        std::mem::swap(&mut y, &mut yn);
        event_old = event_new;
    }

    if let Some(times) = save_times {
        if si != times.len() && !st.event_triggered {
            return Err(format!("ended before all saveat: {si} of {}", times.len()));
        }
    }
    Ok(st)
}
