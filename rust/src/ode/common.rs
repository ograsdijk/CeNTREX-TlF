use crate::ode::OdeRhs;

pub fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

pub fn time_tol(t: f64) -> f64 {
    10.0 * f64::EPSILON * t.abs().max(1.0)
}

pub fn close_to_start(t: f64, t0: f64) -> bool {
    (t - t0).abs() <= time_tol(t0)
}

#[derive(Clone, Copy, Debug)]
pub struct Controller {
    pub alpha: f64,
    pub beta: f64,
    pub facc1: f64,
    pub facc2: f64,
    pub fac_old: f64,
    pub h_max: f64,
    pub reject: bool,
    pub safety_factor: f64,
    pub posneg: f64,
}

impl Controller {
    pub fn new(t0: f64, t1: f64) -> Self {
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

    pub fn accept(&mut self, err: f64, h: f64) -> (bool, f64) {
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

pub fn hinit<R: OdeRhs>(
    rhs: &mut R,
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

pub struct SavePlan {
    pub times: Vec<f64>,
}

pub fn build_save_plan(
    saveat: Option<&[f64]>,
    t0: f64,
    t1: f64,
    save_start: bool,
) -> Result<Option<SavePlan>, String> {
    let Some(saveat) = saveat else {
        return Ok(None);
    };
    let starts_at_t0 = saveat.first().is_some_and(|&time| close_to_start(time, t0));
    let mut times = Vec::with_capacity(saveat.len() + usize::from(save_start && !starts_at_t0));
    if save_start {
        times.push(t0);
    }
    for &time in saveat {
        if save_start && close_to_start(time, t0) {
            continue;
        }
        if !time.is_finite() {
            return Err("saveat contains non-finite value".to_string());
        }
        if time < t0 - time_tol(t0) || time > t1 + time_tol(t1) {
            return Err(format!("saveat value {time} is outside [{t0}, {t1}]"));
        }
        times.push(time);
    }
    Ok(Some(SavePlan { times }))
}
