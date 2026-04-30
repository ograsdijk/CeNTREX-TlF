pub mod batch;
pub mod common;
pub mod dopri5;
pub mod output;
pub mod tsit5;

pub trait OdeRhs {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String>;
    fn dim(&self) -> usize;
    fn event_value(&mut self, _t: f64, _y: &[f64]) -> Result<Option<f64>, String> {
        Ok(None)
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
