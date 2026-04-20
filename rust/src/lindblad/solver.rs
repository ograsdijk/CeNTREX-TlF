use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::{rhs_packed, ExecutionMode};

#[derive(Clone, Debug)]
pub struct ExplicitSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub saveat: Option<Vec<f64>>,
    pub save_start: bool,
    pub save_everystep: bool,
    pub maxiters: usize,
    pub mode: ExecutionMode,
}

fn combine_state(base: &[f64], terms: &[(&[f64], f64)], h: f64, out: &mut [f64]) {
    for idx in 0..base.len() {
        let mut value = base[idx];
        for (k, coeff) in terms {
            value += h * coeff * k[idx];
        }
        out[idx] = value;
    }
}

fn error_norm(y: &[f64], y_new: &[f64], error: &[f64], abstol: f64, reltol: f64) -> f64 {
    let mut accum = 0.0;
    for idx in 0..y.len() {
        let scale = abstol + reltol * y[idx].abs().max(y_new[idx].abs());
        let ratio = error[idx] / scale;
        accum += ratio * ratio;
    }
    (accum / (y.len() as f64)).sqrt()
}

fn save_state(times: &mut Vec<f64>, states: &mut Vec<f64>, t: f64, y: &[f64]) {
    times.push(t);
    states.extend_from_slice(y);
}

pub fn solve_explicit(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &ExplicitSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if t1 < t0 {
        return Err("only forward integration is supported".to_string());
    }
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

    let mut saveat = options.saveat.clone();
    if let Some(values) = saveat.as_ref() {
        for window in values.windows(2) {
            if window[1] < window[0] {
                return Err("saveat values must be sorted in ascending order".to_string());
            }
        }
    }

    let n = y0.len();
    let mut times = Vec::new();
    let mut states = Vec::new();
    let mut y = y0.to_vec();
    let mut t = t0;
    let mut h = options.dt.abs().max(1e-16);
    let span = (t1 - t0).abs();
    if h > span && span > 0.0 {
        h = span;
    }

    if saveat.is_none() && options.save_start {
        save_state(&mut times, &mut states, t, y.as_slice());
    }

    if (t1 - t0).abs() <= f64::EPSILON {
        if saveat.is_none() && !options.save_start {
            save_state(&mut times, &mut states, t, y.as_slice());
        }
        return Ok((times, states));
    }

    let mut k1;
    let mut k2;
    let mut k3;
    let mut k4;
    let mut k5;
    let mut k6;
    let mut k7;
    let mut y_tmp = vec![0.0; n];
    let mut y5 = vec![0.0; n];
    let mut y4 = vec![0.0; n];
    let mut err = vec![0.0; n];
    let mut save_idx = 0_usize;
    let mut iterations = 0_usize;

    while t < t1 {
        if iterations >= options.maxiters {
            return Err("explicit solver exceeded maxiters budget".to_string());
        }
        iterations += 1;

        if let Some(values) = saveat.as_ref() {
            while save_idx < values.len() && values[save_idx] < t0 {
                save_idx += 1;
            }
        }

        let mut h_step = h.min(t1 - t);
        if let Some(values) = saveat.as_ref() {
            if save_idx < values.len() {
                let next_save = values[save_idx];
                if next_save < t - 1e-14 {
                    return Err("saveat contains a time behind the current integration point".to_string());
                }
                h_step = h_step.min((next_save - t).max(0.0));
            }
        }
        if h_step <= 1e-16 {
            if let Some(values) = saveat.as_ref() {
                if save_idx < values.len() && (values[save_idx] - t).abs() <= 1e-14 {
                    save_state(&mut times, &mut states, values[save_idx], y.as_slice());
                    save_idx += 1;
                    continue;
                }
            }
            h_step = (t1 - t).max(1e-16);
        }

        k1 = rhs_packed(plan, y.as_slice(), t, options.mode)?;

        combine_state(y.as_slice(), &[(k1.as_slice(), 1.0 / 5.0)], h_step, y_tmp.as_mut_slice());
        k2 = rhs_packed(plan, y_tmp.as_slice(), t + h_step * (1.0 / 5.0), options.mode)?;

        combine_state(
            y.as_slice(),
            &[
                (k1.as_slice(), 3.0 / 40.0),
                (k2.as_slice(), 9.0 / 40.0),
            ],
            h_step,
            y_tmp.as_mut_slice(),
        );
        k3 = rhs_packed(plan, y_tmp.as_slice(), t + h_step * (3.0 / 10.0), options.mode)?;

        combine_state(
            y.as_slice(),
            &[
                (k1.as_slice(), 44.0 / 45.0),
                (k2.as_slice(), -56.0 / 15.0),
                (k3.as_slice(), 32.0 / 9.0),
            ],
            h_step,
            y_tmp.as_mut_slice(),
        );
        k4 = rhs_packed(plan, y_tmp.as_slice(), t + h_step * (4.0 / 5.0), options.mode)?;

        combine_state(
            y.as_slice(),
            &[
                (k1.as_slice(), 19372.0 / 6561.0),
                (k2.as_slice(), -25360.0 / 2187.0),
                (k3.as_slice(), 64448.0 / 6561.0),
                (k4.as_slice(), -212.0 / 729.0),
            ],
            h_step,
            y_tmp.as_mut_slice(),
        );
        k5 = rhs_packed(plan, y_tmp.as_slice(), t + h_step * (8.0 / 9.0), options.mode)?;

        combine_state(
            y.as_slice(),
            &[
                (k1.as_slice(), 9017.0 / 3168.0),
                (k2.as_slice(), -355.0 / 33.0),
                (k3.as_slice(), 46732.0 / 5247.0),
                (k4.as_slice(), 49.0 / 176.0),
                (k5.as_slice(), -5103.0 / 18656.0),
            ],
            h_step,
            y_tmp.as_mut_slice(),
        );
        k6 = rhs_packed(plan, y_tmp.as_slice(), t + h_step, options.mode)?;

        combine_state(
            y.as_slice(),
            &[
                (k1.as_slice(), 35.0 / 384.0),
                (k3.as_slice(), 500.0 / 1113.0),
                (k4.as_slice(), 125.0 / 192.0),
                (k5.as_slice(), -2187.0 / 6784.0),
                (k6.as_slice(), 11.0 / 84.0),
            ],
            h_step,
            y5.as_mut_slice(),
        );
        k7 = rhs_packed(plan, y5.as_slice(), t + h_step, options.mode)?;

        combine_state(
            y.as_slice(),
            &[
                (k1.as_slice(), 5179.0 / 57600.0),
                (k3.as_slice(), 7571.0 / 16695.0),
                (k4.as_slice(), 393.0 / 640.0),
                (k5.as_slice(), -92097.0 / 339200.0),
                (k6.as_slice(), 187.0 / 2100.0),
                (k7.as_slice(), 1.0 / 40.0),
            ],
            h_step,
            y4.as_mut_slice(),
        );

        for idx in 0..n {
            err[idx] = y5[idx] - y4[idx];
        }
        let err_norm = error_norm(y.as_slice(), y5.as_slice(), err.as_slice(), options.abstol, options.reltol);
        if err_norm <= 1.0 {
            t += h_step;
            y.copy_from_slice(y5.as_slice());

            if let Some(values) = saveat.as_ref() {
                while save_idx < values.len() && t + 1e-14 >= values[save_idx] {
                    save_state(&mut times, &mut states, values[save_idx], y.as_slice());
                    save_idx += 1;
                }
            } else if options.save_everystep || t + 1e-14 >= t1 {
                save_state(&mut times, &mut states, t, y.as_slice());
            }

            let factor = if err_norm == 0.0 {
                5.0
            } else {
                (0.9 * err_norm.powf(-0.2)).clamp(0.2, 5.0)
            };
            h = (h_step * factor).min(t1 - t).max(1e-16);
        } else {
            let factor = (0.9 * err_norm.powf(-0.2)).clamp(0.1, 0.5);
            h = (h_step * factor).max(1e-16);
        }
    }

    if saveat.is_none() && !options.save_everystep && (times.last().copied().unwrap_or(f64::NAN) - t1).abs() > 1e-14
    {
        save_state(&mut times, &mut states, t1, y.as_slice());
    }

    if let Some(values) = saveat.take() {
        if save_idx != values.len() {
            return Err("integration ended before all saveat points were recorded".to_string());
        }
    }

    Ok((times, states))
}
