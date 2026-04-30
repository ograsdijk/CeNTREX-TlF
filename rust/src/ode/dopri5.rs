use crate::ode::common::{build_save_plan, close_to_start, hinit, sign, time_tol, Controller};
use crate::ode::output::OdeOutput;
use crate::ode::{OdeOptions, OdeRhs, OdeStats};

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
fn fill_solution(y: &[f64], h: f64, k: &[f64], dim: usize, out: &mut [f64]) {
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

fn error_norm(y: &[f64], yn: &[f64], h: f64, k: &[f64], dim: usize, atol: f64, rtol: f64) -> f64 {
    let mut err = 0.0;
    for i in 0..y.len() {
        let sc = atol + y[i].abs().max(yn[i].abs()) * rtol;
        let ei = h
            * (E1 * k[i]
                + E3 * k[2 * dim + i]
                + E4 * k[3 * dim + i]
                + E5 * k[4 * dim + i]
                + E6 * k[5 * dim + i]
                + E7 * k[6 * dim + i]);
        err += (ei / sc) * (ei / sc);
    }
    (err / y.len() as f64).sqrt()
}

fn fill_dense_rcont4(h: f64, k: &[f64], out: &mut [f64]) {
    let dim = out.len();
    for i in 0..dim {
        out[i] = h
            * (D1 * k[i]
                + D3 * k[2 * dim + i]
                + D4 * k[3 * dim + i]
                + D5 * k[4 * dim + i]
                + D6 * k[5 * dim + i]
                + D7 * k[6 * dim + i]);
    }
}

fn fill_dense_coeffs(
    y: &[f64],
    yn: &[f64],
    h: f64,
    k: &[f64],
    dim: usize,
    r0: &mut [f64],
    r1: &mut [f64],
    r2: &mut [f64],
    r3: &mut [f64],
) {
    for i in 0..y.len() {
        let d = yn[i] - y[i];
        let b = k[i] * h - d;
        r0[i] = y[i];
        r1[i] = d;
        r2[i] = b;
        r3[i] = -k[6 * dim + i] * h + d - b;
    }
}

fn dense_out(th: f64, r0: &[f64], r1: &[f64], r2: &[f64], r3: &[f64], r4: &[f64], out: &mut [f64]) {
    let th1 = 1.0 - th;
    for i in 0..r0.len() {
        out[i] = r0[i] + (r1[i] + (r2[i] + (r3[i] + r4[i] * th1) * th) * th1) * th;
    }
}

fn crossed_zero(g0: f64, g1: f64) -> bool {
    g0 == 0.0 || g1 == 0.0 || (g0 < 0.0 && g1 > 0.0) || (g0 > 0.0 && g1 < 0.0)
}

fn push_unique<O: OdeOutput>(output: &mut O, t: f64, y: &[f64]) {
    if output
        .times()
        .last()
        .is_some_and(|&last| (last - t).abs() <= time_tol(t))
    {
        return;
    }
    output.push(t, y);
}

pub fn solve_dopri5<R: OdeRhs, O: OdeOutput>(
    rhs: &mut R,
    y0: &[f64],
    t0: f64,
    t1: f64,
    opt: &OdeOptions,
    output: &mut O,
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
    if t1 == t0 {
        if opt.save_start {
            output.push(t0, y0);
        }
        return Ok(OdeStats::default());
    }
    let sp = build_save_plan(opt.saveat.as_deref(), t0, t1, opt.save_start)?;
    let mut st = OdeStats::default();
    let mut y = y0.to_vec();
    let mut yn = vec![0.0; dim];
    let mut yt = vec![0.0; dim];
    let mut ys = vec![0.0; dim];
    let mut r0 = vec![0.0; dim];
    let mut r1 = vec![0.0; dim];
    let mut r2 = vec![0.0; dim];
    let mut r3 = vec![0.0; dim];
    let mut r4 = vec![0.0; dim];
    let mut ds = vec![0.0; dim];
    let mut k = vec![0.0; 7 * dim];
    let mut f0 = vec![0.0; dim];
    let mut f1 = vec![0.0; dim];
    let mut x = t0;
    let mut h = if opt.dt.is_finite() && opt.dt > 0.0 {
        opt.dt.min(t1 - t0)
    } else {
        hinit(
            rhs, &y, x, t1, opt.abstol, opt.reltol, &mut f0, &mut f1, &mut yt,
        )?
    };
    let mut ctrl = Controller::new(t0, t1);
    let pn = sign(1.0, t1 - t0);
    rhs.eval(x, &y, &mut k[..dim])?;
    st.rhs_calls += 1;
    let mut event_old = rhs.event_value(x, &y)?;
    let mut si = 0usize;
    if let Some(p) = &sp {
        while si < p.times.len() && close_to_start(p.times[si], t0) {
            output.push(p.times[si], &y);
            si += 1;
        }
    } else if opt.save_start {
        output.push(x, &y);
    }
    let mut ns = 0usize;
    let mut last = false;
    let mut nst = 0usize;
    let mut ia = 0usize;
    while !last {
        if ns > opt.maxiters {
            return Err(format!("Stopped at x={x}. Need more than {ns} steps."));
        }
        if 0.1 * h.abs() <= f64::EPSILON * x.abs() {
            return Err(format!("Stopped at x={x}. Step size underflow."));
        }
        if (x + 1.01 * h - t1) * pn > 0.0 {
            h = t1 - x;
            last = true;
        }
        ns += 1;
        fill_stage1(&y, h, &k, dim, &mut yt);
        rhs.eval(x + h * C2, &yt, &mut k[dim..2 * dim])?;
        fill_stage2(&y, h, &k, dim, &mut yt);
        rhs.eval(x + h * C3, &yt, &mut k[2 * dim..3 * dim])?;
        fill_stage3(&y, h, &k, dim, &mut yt);
        rhs.eval(x + h * C4, &yt, &mut k[3 * dim..4 * dim])?;
        fill_stage4(&y, h, &k, dim, &mut yt);
        rhs.eval(x + h * C5, &yt, &mut k[4 * dim..5 * dim])?;
        fill_stage5(&y, h, &k, dim, &mut yt);
        ys.copy_from_slice(&yt);
        rhs.eval(x + h * C6, &yt, &mut k[5 * dim..6 * dim])?;
        fill_solution(&y, h, &k, dim, &mut yn);
        rhs.eval(x + h * C7, &yn, &mut k[6 * dim..7 * dim])?;
        st.rhs_calls += 6;
        let err = error_norm(&y, &yn, h, &k, dim, opt.abstol, opt.reltol);
        let (acc, hn) = ctrl.accept(err, h);
        if acc {
            st.accepted_steps += 1;
            if st.accepted_steps % 1000 == 0 || ia > 0 {
                let mut n2 = 0.0;
                let mut d2 = 0.0;
                for i in 0..dim {
                    let dk = k[6 * dim + i] - k[5 * dim + i];
                    let dy = yn[i] - ys[i];
                    n2 += dk * dk;
                    d2 += dy * dy;
                }
                let hl = if d2 > 0.0 { h * (n2 / d2).sqrt() } else { 0.0 };
                if hl > 3.25 {
                    ia += 1;
                    nst = 0;
                    if ia == 15 {
                        return Err(format!("Stiff at x={x}."));
                    }
                } else {
                    nst += 1;
                    if nst == 6 {
                        ia = 0;
                    }
                }
            }
            let xo = x;
            x += h;
            let event_new = rhs.event_value(x, &yn)?;
            let mut event_hit: Option<(f64, Vec<f64>)> = None;
            if let (Some(g0), Some(g1)) = (event_old, event_new) {
                if crossed_zero(g0, g1) {
                    if g0 == 0.0 {
                        event_hit = Some((xo, y.clone()));
                    } else if g1 == 0.0 {
                        event_hit = Some((x, yn.clone()));
                    } else {
                        fill_dense_rcont4(h, &k, &mut r4);
                        fill_dense_coeffs(
                            &y, &yn, h, &k, dim, &mut r0, &mut r1, &mut r2, &mut r3,
                        );
                        let mut lo = 0.0;
                        let mut hi = 1.0;
                        let mut glo = g0;
                        for _ in 0..60 {
                            let mid = 0.5 * (lo + hi);
                            dense_out(mid, &r0, &r1, &r2, &r3, &r4, &mut ds);
                            let gm = rhs.event_value(xo + mid * h, &ds)?.ok_or_else(|| {
                                "event disappeared during root finding".to_string()
                            })?;
                            if gm == 0.0 {
                                lo = mid;
                                hi = mid;
                                break;
                            }
                            if (glo < 0.0 && gm > 0.0) || (glo > 0.0 && gm < 0.0) {
                                hi = mid;
                            } else {
                                lo = mid;
                                glo = gm;
                            }
                        }
                        let theta = 0.5 * (lo + hi);
                        dense_out(theta, &r0, &r1, &r2, &r3, &r4, &mut ds);
                        event_hit = Some((xo + theta * h, ds.clone()));
                    }
                }
            }
            if let Some(p) = &sp {
                let mut dr = false;
                let event_time = event_hit.as_ref().map(|(t, _)| *t).unwrap_or(x);
                while si < p.times.len() && p.times[si] <= event_time + time_tol(event_time) {
                    let ts = p.times[si];
                    if ts >= xo - time_tol(xo) {
                        if (ts - xo).abs() <= time_tol(xo) {
                            output.push(ts, &y);
                        } else if (ts - x).abs() <= time_tol(x) {
                            output.push(ts, &yn);
                        } else {
                            if !dr {
                                fill_dense_rcont4(h, &k, &mut r4);
                                fill_dense_coeffs(
                                    &y, &yn, h, &k, dim, &mut r0, &mut r1, &mut r2, &mut r3,
                                );
                                dr = true;
                            }
                            dense_out((ts - xo) / h, &r0, &r1, &r2, &r3, &r4, &mut ds);
                            output.push(ts, &ds);
                        }
                    }
                    si += 1;
                }
            } else if event_hit.is_none() {
                output.push(x, &yn);
            }
            if let Some((te, ye)) = event_hit {
                push_unique(output, te, &ye);
                st.event_triggered = true;
                st.event_time = te;
                st.event_index = 0;
                break;
            }
            std::mem::swap(&mut y, &mut yn);
            let (f, l) = k.split_at_mut(dim);
            f.copy_from_slice(&l[5 * dim..6 * dim]);
            event_old = event_new;
            if last {
                break;
            }
        } else {
            last = false;
            if st.accepted_steps >= 1 {
                st.rejected_steps += 1;
            }
        }
        h = hn;
    }
    if let Some(p) = &sp {
        if si != p.times.len() {
            if st.event_triggered {
                return Ok(st);
            }
            return Err(format!(
                "ended before all saveat: {si} of {}",
                p.times.len()
            ));
        }
    }
    Ok(st)
}
