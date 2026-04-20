use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::{rhs_packed_into, ExecutionMode, RhsWorkspace};
use ode_solvers::continuous_output_model::ContinuousOutputModel;
use ode_solvers::dop_shared::{IntegrationError, OutputType, System};
use ode_solvers::dopri5::Dopri5;
use ode_solvers::DVector;
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct OdeSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub saveat: Option<Vec<f64>>,
    pub save_start: bool,
    pub maxiters: usize,
    pub mode: ExecutionMode,
}

struct LindbladOdeSystem {
    plan: PreparedLindbladPlan,
    mode: ExecutionMode,
    workspace: RefCell<RhsWorkspace>,
}

impl System<f64, DVector<f64>> for LindbladOdeSystem {
    fn system(&self, x: f64, y: &DVector<f64>, dy: &mut DVector<f64>) {
        let mut workspace = self.workspace.borrow_mut();
        match rhs_packed_into(
            &self.plan,
            y.as_slice(),
            x,
            self.mode,
            &mut workspace,
            dy.as_mut_slice(),
        ) {
            Ok(()) => {}
            Err(err) => panic!("lindblad rhs failed inside ode_solvers: {err}"),
        }
    }
}

fn map_integration_error(err: IntegrationError) -> String {
    err.to_string()
}

pub fn solve_dopri5(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &OdeSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>), String> {
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
    if t1 < t0 {
        return Err("only forward integration is supported".to_string());
    }
    let system = LindbladOdeSystem {
        plan: plan.clone(),
        mode: options.mode,
        workspace: RefCell::new(RhsWorkspace::new(plan)),
    };
    let y0 = DVector::from_vec(y0.to_vec());
    let mut stepper = Dopri5::from_param(
        system,
        t0,
        t1,
        options.dt,
        y0,
        options.reltol,
        options.abstol,
        0.9,
        0.04,
        0.2,
        10.0,
        t1 - t0,
        0.0,
        options.maxiters as u32,
        1000,
        if options.saveat.is_some() {
            OutputType::Continuous
        } else {
            OutputType::Sparse
        },
    );

    if let Some(saveat) = options.saveat.as_ref() {
        let mut output = ContinuousOutputModel::default();
        stepper
            .integrate_with_continuous_output_model(&mut output)
            .map_err(map_integration_error)?;

        let mut times = Vec::with_capacity(saveat.len() + usize::from(options.save_start));
        let mut states = Vec::with_capacity(
            (saveat.len() + usize::from(options.save_start)) * plan.layout.packed_len(),
        );
        let mut push_time = |time: f64| -> Result<(), String> {
            let value: DVector<f64> = output
                .evaluate(time)
                .ok_or_else(|| format!("continuous output unavailable at t={time}"))?;
            times.push(time);
            states.extend_from_slice(value.as_slice());
            Ok(())
        };
        if options.save_start {
            push_time(t0)?;
        }
        for &time in saveat {
            if (time - t0).abs() <= 1e-14 {
                continue;
            }
            push_time(time)?;
        }
        return Ok((times, states));
    }

    stepper.integrate().map_err(map_integration_error)?;
    let times = stepper.x_out().clone();
    let mut states = Vec::with_capacity(times.len() * plan.layout.packed_len());
    for state in stepper.y_out().iter() {
        states.extend_from_slice(state.as_slice());
    }
    Ok((times, states))
}
