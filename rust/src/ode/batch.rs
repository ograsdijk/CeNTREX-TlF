use crate::ode::dopri5::solve_dopri5;
use crate::ode::output::OdeOutput;
use crate::ode::tsit5::solve_tsit5;
use crate::ode::{OdeOptions, OdeRhs, OdeStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeSolver {
    Dopri5,
    Tsit5,
}

impl OdeSolver {
    pub fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "dopri5" => Ok(Self::Dopri5),
            "tsit5" => Ok(Self::Tsit5),
            other => Err(format!("solver must be 'dopri5' or 'tsit5', got {other:?}")),
        }
    }
}

pub fn solve_single<R: OdeRhs, O: OdeOutput>(
    rhs: &mut R,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeOptions,
    output: &mut O,
    solver: OdeSolver,
) -> Result<OdeStats, String> {
    match solver {
        OdeSolver::Dopri5 => solve_dopri5(rhs, y0, t0, t1, options, output),
        OdeSolver::Tsit5 => solve_tsit5(rhs, y0, t0, t1, options, output),
    }
}
