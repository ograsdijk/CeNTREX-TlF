use crate::effective_lindblad::plan::EffectiveLindbladPlan;
use crate::effective_lindblad::rhs::{rhs_effective_lindblad, EffectiveLindbladWorkspace};

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

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

pub struct EffectiveSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub saveat: Option<Vec<f64>>,
    pub save_start: bool,
    pub maxiters: usize,
}

pub fn solve_effective_lindblad_dopri5(
    plan: &EffectiveLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &EffectiveSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let dim = plan.real_dim;
    if y0.len() != dim {
        return Err(format!("expected state length {dim}, got {}", y0.len()));
    }
    if t1 <= t0 {
        if t1 == t0 {
            let mut times = Vec::new();
            let mut states = Vec::new();
            if options.save_start {
                times.push(t0);
                states.extend_from_slice(y0);
            }
            return Ok((times, states));
        }
        return Err("only forward integration supported".to_string());
    }

    let mut workspace = EffectiveLindbladWorkspace::new(plan);

    let mut y = y0.to_vec();
    let mut y_next = vec![0.0; dim];
    let mut k1 = vec![0.0; dim];
    let mut k2 = vec![0.0; dim];
    let mut k3 = vec![0.0; dim];
    let mut k4 = vec![0.0; dim];
    let mut k5 = vec![0.0; dim];
    let mut k6 = vec![0.0; dim];
    let mut k7 = vec![0.0; dim];
    let mut y_tmp = vec![0.0; dim];

    let save_plan: Option<Vec<f64>> = if let Some(saveat) = &options.saveat {
        let mut plan_times: Vec<f64> = Vec::new();
        if options.save_start {
            plan_times.push(t0);
        }
        for &ts in saveat {
            if ts > t0 && ts <= t1 {
                plan_times.push(ts);
            }
        }
        Some(plan_times)
    } else {
        None
    };

    let capacity = save_plan.as_ref().map_or(options.maxiters + 1, |p| p.len());
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity * dim);

    if options.save_start && save_plan.is_none() {
        times.push(t0);
        states.extend_from_slice(&y);
    }

    let mut t = t0;
    let mut h = options.dt.min(t1 - t0);
    let mut fac_old = 1.0e-4_f64;
    let safety = 0.9_f64;
    let beta = 0.04_f64;
    let alpha = 0.2 - beta * 0.75;
    let facc1: f64 = 1.0 / 0.2;
    let facc2: f64 = 1.0 / 10.0;
    let h_max = t1 - t0;

    rhs_effective_lindblad(plan, &y, t, &mut workspace, &mut k1)?;

    let mut save_idx = 0;
    let mut step_count = 0u64;

    while t < t1 {
        if t + 1.01 * h > t1 {
            h = t1 - t;
        }
        if h < 1e-100 {
            return Err("step size underflow".to_string());
        }

        for i in 0..dim {
            y_tmp[i] = y[i] + h * A21 * k1[i];
        }
        rhs_effective_lindblad(plan, &y_tmp, t + C2 * h, &mut workspace, &mut k2)?;

        for i in 0..dim {
            y_tmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        rhs_effective_lindblad(plan, &y_tmp, t + C3 * h, &mut workspace, &mut k3)?;

        for i in 0..dim {
            y_tmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        rhs_effective_lindblad(plan, &y_tmp, t + C4 * h, &mut workspace, &mut k4)?;

        for i in 0..dim {
            y_tmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        rhs_effective_lindblad(plan, &y_tmp, t + C5 * h, &mut workspace, &mut k5)?;

        for i in 0..dim {
            y_tmp[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        rhs_effective_lindblad(plan, &y_tmp, t + h, &mut workspace, &mut k6)?;

        for i in 0..dim {
            y_next[i] =
                y[i] + h * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
        }
        rhs_effective_lindblad(plan, &y_next, t + h, &mut workspace, &mut k7)?;

        let mut err = 0.0_f64;
        for i in 0..dim {
            let sk = options.abstol + options.reltol * y[i].abs().max(y_next[i].abs());
            let erri =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            err += (erri / sk) * (erri / sk);
        }
        err = (err / dim as f64).sqrt();

        let fac11 = err.powf(alpha);
        let fac = fac11 * fac_old.powf(-beta);
        let fac = facc2.max(facc1.min(fac / safety));
        let h_new = h / fac;

        if err <= 1.0 {
            fac_old = err.max(1.0e-4);
            t += h;

            if let Some(ref plan_times) = save_plan {
                while save_idx < plan_times.len() && plan_times[save_idx] <= t + 1e-14 {
                    let t_save = plan_times[save_idx];
                    if (t_save - t).abs() < 1e-14 {
                        times.push(t);
                        states.extend_from_slice(&y_next);
                    } else {
                        let theta = (t_save - (t - h)) / h;
                        for i in 0..dim {
                            y_tmp[i] = y[i]
                                + theta
                                    * h
                                    * (A71 * k1[i]
                                        + A73 * k3[i]
                                        + A74 * k4[i]
                                        + A75 * k5[i]
                                        + A76 * k6[i]);
                        }
                        times.push(t_save);
                        states.extend_from_slice(&y_tmp);
                    }
                    save_idx += 1;
                }
            } else {
                times.push(t);
                states.extend_from_slice(&y_next);
            }

            y.copy_from_slice(&y_next);
            k1.copy_from_slice(&k7);
            h = h_new.min(h_max);
        } else {
            h = h_new;
        }

        step_count += 1;
        if step_count > options.maxiters as u64 {
            return Err("exceeded maxiters".to_string());
        }
    }

    Ok((times, states))
}
