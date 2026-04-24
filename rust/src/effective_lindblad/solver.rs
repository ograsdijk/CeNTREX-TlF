use crate::effective_lindblad::plan::EffectiveLindbladPlan;
use crate::effective_lindblad::rhs::{rhs_effective_lindblad, EffectiveLindbladWorkspace};
use crate::ode::batch::{solve_single, OdeSolver};
use crate::ode::output::{FullOutput, OdeOutputResult};
use crate::ode::{OdeOptions, OdeRhs};

struct EffectiveLindbladRhs<'a> {
    plan: &'a EffectiveLindbladPlan,
    workspace: EffectiveLindbladWorkspace,
}

impl<'a> EffectiveLindbladRhs<'a> {
    fn new(plan: &'a EffectiveLindbladPlan) -> Self {
        Self {
            plan,
            workspace: EffectiveLindbladWorkspace::new(plan),
        }
    }
}

impl OdeRhs for EffectiveLindbladRhs<'_> {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        rhs_effective_lindblad(self.plan, y, t, &mut self.workspace, dy)
    }
    fn dim(&self) -> usize {
        self.plan.real_dim
    }
}

pub struct EffectiveSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub saveat: Option<Vec<f64>>,
    pub save_start: bool,
    pub maxiters: usize,
    pub solver: String,
}

pub fn solve_effective_lindblad(
    plan: &EffectiveLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &EffectiveSolverOptions,
) -> Result<OdeOutputResult, String> {
    let mut rhs = EffectiveLindbladRhs::new(plan);
    let solver = OdeSolver::from_str(&options.solver)?;
    let ode_options = OdeOptions {
        abstol: options.abstol,
        reltol: options.reltol,
        dt: options.dt,
        maxiters: options.maxiters,
        save_start: options.save_start,
        saveat: options.saveat.clone(),
    };
    let capacity = options
        .saveat
        .as_ref()
        .map_or(options.maxiters + 1, |s| s.len() + 1);
    let mut output = FullOutput::new(plan.real_dim, capacity);
    solve_single(&mut rhs, y0, t0, t1, &ode_options, &mut output, solver)?;
    Ok(output.finish())
}
