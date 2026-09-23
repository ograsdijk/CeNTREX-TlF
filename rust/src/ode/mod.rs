pub mod batch;
pub mod common;
pub mod dopri5;
pub mod fixed;
pub mod output;
pub mod tsit5;

pub trait OdeRhs {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String>;
    fn dim(&self) -> usize;
    /// Number of leading state components used for adaptive error control.
    /// Auxiliary solver states can override this to leave physical step
    /// selection unchanged.
    fn error_control_dim(&self) -> usize {
        self.dim()
    }
    fn event_value(&mut self, _t: f64, _y: &[f64]) -> Result<Option<f64>, String> {
        Ok(None)
    }
}

struct IntegralAugmentedRhs<'a, R> {
    inner: &'a mut R,
    weights: Vec<(usize, f64)>,
    physical_dim: usize,
}

impl<R: OdeRhs> OdeRhs for IntegralAugmentedRhs<'_, R> {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        self.inner
            .eval(t, &y[..self.physical_dim], &mut dy[..self.physical_dim])?;
        dy[self.physical_dim] = self.weights.iter().map(|&(i, w)| w * y[i]).sum();
        Ok(())
    }

    fn dim(&self) -> usize {
        self.physical_dim + 1
    }

    fn error_control_dim(&self) -> usize {
        self.physical_dim
    }

    fn event_value(&mut self, t: f64, y: &[f64]) -> Result<Option<f64>, String> {
        self.inner.event_value(t, &y[..self.physical_dim])
    }
}

#[derive(Clone, Debug)]
pub struct OdeOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub maxiters: usize,
    pub save_start: bool,
    pub saveat: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct OdeStats {
    pub accepted_steps: u64,
    pub rejected_steps: u64,
    pub rhs_calls: u64,
    pub event_triggered: bool,
    pub event_time: f64,
    pub event_index: i64,
}

impl Default for OdeStats {
    fn default() -> Self {
        Self {
            accepted_steps: 0,
            rejected_steps: 0,
            rhs_calls: 0,
            event_triggered: false,
            event_time: f64::NAN,
            event_index: -1,
        }
    }
}
