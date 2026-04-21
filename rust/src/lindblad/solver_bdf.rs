use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::{rhs_packed_into, ExecutionMode, RhsWorkspace};
use diffsol::ode_solver::method::OdeSolverMethod;
use diffsol::ode_solver::method::OdeSolverStopReason;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{FaerSparseLU, FaerSparseMat, OdeBuilder, VectorHost};
use std::cell::RefCell;
use std::rc::Rc;

type M = FaerSparseMat<f64>;
type V = FaerVec<f64>;

#[derive(Clone, Debug)]
pub struct BdfSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub dt: f64,
    pub saveat: Option<Vec<f64>>,
    pub save_start: bool,
    pub maxiters: usize,
    pub mode: ExecutionMode,
}

pub fn solve_bdf(
    plan: &PreparedLindbladPlan,
    y0: &[f64],
    t0: f64,
    t1: f64,
    options: &BdfSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let dim = plan.layout.packed_len();
    if y0.len() != dim {
        return Err(format!(
            "expected packed state length {}, got {}",
            dim,
            y0.len()
        ));
    }
    if options.maxiters == 0 {
        return Err("maxiters must be positive".to_string());
    }
    if t1 < t0 {
        return Err("only forward integration is supported".to_string());
    }

    let plan_rc = Rc::new(plan.clone());
    let mode = options.mode;
    let workspace: Rc<RefCell<RhsWorkspace>> = Rc::new(RefCell::new(RhsWorkspace::new(plan)));
    let rhs_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));

    let rhs_workspace = workspace.clone();
    let rhs_plan = plan_rc.clone();
    let rhs_err = rhs_error.clone();
    let rhs_fn = move |x: &V, _p: &V, t: f64, y: &mut V| {
        let mut ws = rhs_workspace.borrow_mut();
        if let Err(err) =
            rhs_packed_into(&rhs_plan, x.as_slice(), t, mode, &mut ws, y.as_mut_slice())
        {
            *rhs_err.borrow_mut() = Some(err);
            y.as_mut_slice().fill(0.0);
        }
    };

    let jac_workspace = workspace.clone();
    let jac_plan = plan_rc.clone();
    let jac_err = rhs_error.clone();
    let jac_fn = move |_x: &V, _p: &V, t: f64, v: &V, y: &mut V| {
        let mut ws = jac_workspace.borrow_mut();
        if let Err(err) =
            rhs_packed_into(&jac_plan, v.as_slice(), t, mode, &mut ws, y.as_mut_slice())
        {
            *jac_err.borrow_mut() = Some(err);
            y.as_mut_slice().fill(0.0);
        }
    };

    let y0_vec = y0.to_vec();
    let problem = OdeBuilder::<M>::new()
        .t0(t0)
        .h0(options.dt)
        .rtol(options.reltol)
        .atol(vec![options.abstol; dim])
        .p(vec![0.0])
        .use_coloring(true)
        .rhs_implicit(rhs_fn, jac_fn)
        .init(
            move |_p: &V, _t: f64, y: &mut V| {
                y.as_mut_slice().copy_from_slice(&y0_vec);
            },
            dim,
        )
        .build()
        .map_err(|e| format!("diffsol build error: {e}"))?;

    let check_rhs_error = || -> Result<(), String> {
        if let Some(err) = rhs_error.borrow_mut().take() {
            return Err(format!("lindblad rhs failed inside diffsol bdf: {err}"));
        }
        Ok(())
    };

    let mut solver = problem
        .bdf::<FaerSparseLU<f64>>()
        .map_err(|e| format!("diffsol bdf init error: {e}"))?;

    if let Some(saveat) = &options.saveat {
        let mut times = Vec::with_capacity(saveat.len() + usize::from(options.save_start));
        let mut states = Vec::with_capacity((saveat.len() + usize::from(options.save_start)) * dim);

        if options.save_start {
            times.push(t0);
            states.extend_from_slice(y0);
        }

        let mut saveat_iter = saveat.iter().peekable();
        while let Some(&&t_save) = saveat_iter.peek() {
            if (t_save - t0).abs() <= 1e-14 {
                saveat_iter.next();
                continue;
            }
            solver
                .set_stop_time(t_save)
                .map_err(|e| format!("diffsol set_stop_time error: {e}"))?;
            let mut step_count = 0;
            loop {
                match solver.step() {
                    Ok(OdeSolverStopReason::InternalTimestep) => {}
                    Ok(OdeSolverStopReason::TstopReached) => break,
                    Ok(OdeSolverStopReason::RootFound(..)) => break,
                    Err(e) => return Err(format!("diffsol step error: {e}")),
                }
                check_rhs_error()?;
                step_count += 1;
                if step_count > options.maxiters {
                    return Err("diffsol bdf exceeded maxiters".to_string());
                }
            }
            check_rhs_error()?;
            let state = solver.state();
            times.push(state.t);
            states.extend_from_slice(state.y.as_slice());
            saveat_iter.next();
        }
        return Ok((times, states));
    }

    let mut times = Vec::new();
    let mut states = Vec::new();
    if options.save_start {
        times.push(t0);
        states.extend_from_slice(y0);
    }

    solver
        .set_stop_time(t1)
        .map_err(|e| format!("diffsol set_stop_time error: {e}"))?;
    let mut step_count = 0;
    loop {
        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => {
                let state = solver.state();
                times.push(state.t);
                states.extend_from_slice(state.y.as_slice());
            }
            Ok(OdeSolverStopReason::TstopReached) => {
                let state = solver.state();
                times.push(state.t);
                states.extend_from_slice(state.y.as_slice());
                break;
            }
            Ok(OdeSolverStopReason::RootFound(..)) => break,
            Err(e) => return Err(format!("diffsol step error: {e}")),
        }
        check_rhs_error()?;
        step_count += 1;
        if step_count > options.maxiters {
            return Err("diffsol bdf exceeded maxiters".to_string());
        }
    }
    Ok((times, states))
}
