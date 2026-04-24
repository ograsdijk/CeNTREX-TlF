pub mod batch;
pub mod common;
pub mod dopri5;
pub mod output;
pub mod tsit5;

pub trait OdeRhs {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String>;
    fn dim(&self) -> usize;
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

#[derive(Clone, Debug, Default)]
pub struct OdeStats {
    pub accepted_steps: u64,
    pub rejected_steps: u64,
    pub rhs_calls: u64,
}
