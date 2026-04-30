use crate::ode::common::{build_save_plan, close_to_start, hinit, sign, time_tol, Controller};
use crate::ode::output::OdeOutput;
use crate::ode::{OdeOptions, OdeRhs, OdeStats};

const C2: f64 = 0.161;
const C3: f64 = 0.327;
const C4: f64 = 0.9;
const C5: f64 = 0.9800255409045097;
const C6: f64 = 1.0;
const C7: f64 = 1.0;
const A21: f64 = 0.161;
const A31: f64 = -0.008480655492356989;
const A32: f64 = 0.335480655492357;
const A41: f64 = 2.8971530571054935;
const A42: f64 = -6.359448489975075;
const A43: f64 = 4.3622954328695815;
const A51: f64 = 5.325864828439257;
const A52: f64 = -11.748883564062828;
const A53: f64 = 7.4955393428898365;
const A54: f64 = -0.09249506636175525;
const A61: f64 = 5.86145544294642;
const A62: f64 = -12.92096931784711;
const A63: f64 = 8.159367898576159;
const A64: f64 = -0.071584973281401;
const A65: f64 = -0.028269050394068383;
const A71: f64 = 0.09646076681806523;
const A72: f64 = 0.01;
const A73: f64 = 0.4798896504144996;
const A74: f64 = 1.379008574103742;
const A75: f64 = -3.290069515436081;
const A76: f64 = 2.324710524099774;
const E1: f64 = -0.0017800110522257771;
const E2: f64 = -0.0008164344596567469;
const E3: f64 = 0.007880878010261995;
const E4: f64 = -0.1447110071732629;
const E5: f64 = 0.5823571654525552;
const E6: f64 = -0.458082105929187;
const E7: f64 = 1.0 / 66.0;
const R11: f64 = 1.0;
const R12: f64 = -2.763706197274826;
const R13: f64 = 2.9132554618219126;
const R14: f64 = -1.0530884977290216;
const R22: f64 = 0.1317;
const R23: f64 = -0.2234;
const R24: f64 = 0.1017;
const R32: f64 = 3.9302962368947516;
const R33: f64 = -5.941033872131505;
const R34: f64 = 2.490627285651253;
const R42: f64 = -12.411077166933676;
const R43: f64 = 30.33818863028232;
const R44: f64 = -16.548102889244902;
const R52: f64 = 37.50931341651104;
const R53: f64 = -88.1789048947664;
const R54: f64 = 47.37952196281928;
const R62: f64 = -27.896526289197286;
const R63: f64 = 65.09189467479366;
const R64: f64 = -34.87065786149661;
const R72: f64 = 1.5;
const R73: f64 = -4.0;
const R74: f64 = 2.5;

fn fill_stage1(y: &[f64], h: f64, k: &[f64], _d: usize, o: &mut [f64]) {
    for i in 0..y.len() {
        o[i] = y[i] + h * A21 * k[i];
    }
}
fn fill_stage2(y: &[f64], h: f64, k: &[f64], d: usize, o: &mut [f64]) {
    for i in 0..y.len() {
        o[i] = y[i] + h * (A31 * k[i] + A32 * k[d + i]);
    }
}
fn fill_stage3(y: &[f64], h: f64, k: &[f64], d: usize, o: &mut [f64]) {
    for i in 0..y.len() {
        o[i] = y[i] + h * (A41 * k[i] + A42 * k[d + i] + A43 * k[2 * d + i]);
    }
}
fn fill_stage4(y: &[f64], h: f64, k: &[f64], d: usize, o: &mut [f64]) {
    for i in 0..y.len() {
        o[i] = y[i] + h * (A51 * k[i] + A52 * k[d + i] + A53 * k[2 * d + i] + A54 * k[3 * d + i]);
    }
}
fn fill_stage5(y: &[f64], h: f64, k: &[f64], d: usize, o: &mut [f64]) {
    for i in 0..y.len() {
        o[i] = y[i]
            + h * (A61 * k[i]
                + A62 * k[d + i]
                + A63 * k[2 * d + i]
                + A64 * k[3 * d + i]
                + A65 * k[4 * d + i]);
    }
}
fn fill_solution(y: &[f64], h: f64, k: &[f64], d: usize, o: &mut [f64]) {
    for i in 0..y.len() {
        o[i] = y[i]
            + h * (A71 * k[i]
                + A72 * k[d + i]
                + A73 * k[2 * d + i]
                + A74 * k[3 * d + i]
                + A75 * k[4 * d + i]
                + A76 * k[5 * d + i]);
    }
}

fn error_norm(y: &[f64], yn: &[f64], h: f64, k: &[f64], d: usize, atol: f64, rtol: f64) -> f64 {
    let mut e = 0.0;
    for i in 0..y.len() {
        let s = atol + y[i].abs().max(yn[i].abs()) * rtol;
        let ei = h
            * (E1 * k[i]
                + E2 * k[d + i]
                + E3 * k[2 * d + i]
                + E4 * k[3 * d + i]
                + E5 * k[4 * d + i]
                + E6 * k[5 * d + i]
                + E7 * k[6 * d + i]);
        e += (ei / s) * (ei / s);
    }
    (e / y.len() as f64).sqrt()
}

fn dense_out_tsit5(th: f64, h: f64, y: &[f64], k: &[f64], d: usize, o: &mut [f64]) {
    let t2 = th * th;
    let t3 = t2 * th;
    let t4 = t3 * th;
    let b1 = R11 * th + R12 * t2 + R13 * t3 + R14 * t4;
    let b2 = R22 * t2 + R23 * t3 + R24 * t4;
    let b3 = R32 * t2 + R33 * t3 + R34 * t4;
    let b4 = R42 * t2 + R43 * t3 + R44 * t4;
    let b5 = R52 * t2 + R53 * t3 + R54 * t4;
    let b6 = R62 * t2 + R63 * t3 + R64 * t4;
    let b7 = R72 * t2 + R73 * t3 + R74 * t4;
    for i in 0..y.len() {
        o[i] = y[i]
            + h * (b1 * k[i]
                + b2 * k[d + i]
                + b3 * k[2 * d + i]
                + b4 * k[3 * d + i]
                + b5 * k[4 * d + i]
                + b6 * k[5 * d + i]
                + b7 * k[6 * d + i]);
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

pub fn solve_tsit5<R: OdeRhs, O: OdeOutput>(
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
                        let mut lo = 0.0;
                        let mut hi = 1.0;
                        let mut glo = g0;
                        for _ in 0..60 {
                            let mid = 0.5 * (lo + hi);
                            dense_out_tsit5(mid, h, &y, &k, dim, &mut ds);
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
                        dense_out_tsit5(theta, h, &y, &k, dim, &mut ds);
                        event_hit = Some((xo + theta * h, ds.clone()));
                    }
                }
            }
            if let Some(p) = &sp {
                let event_time = event_hit.as_ref().map(|(t, _)| *t).unwrap_or(x);
                while si < p.times.len() && p.times[si] <= event_time + time_tol(event_time) {
                    let ts = p.times[si];
                    if ts >= xo - time_tol(xo) {
                        if (ts - xo).abs() <= time_tol(xo) {
                            output.push(ts, &y);
                        } else if (ts - x).abs() <= time_tol(x) {
                            output.push(ts, &yn);
                        } else {
                            dense_out_tsit5((ts - xo) / h, h, &y, &k, dim, &mut ds);
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
