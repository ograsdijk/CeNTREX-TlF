// Benchmark-only wrapper. The numerical solver and baseline RHS modules are
// included from the library by the Python source generator, not reimplemented.
use lindblad::eval::RuntimeValue;
use lindblad::plan::{parse_plan_payload, PreparedLindbladPlan};
use lindblad::rhs::{rhs_packed_into, ExecutionMode, RhsOptions, RhsWorkspace};
use num_complex::Complex64;
use numpy::{PyArray1, PyReadonlyArray1};
use ode::output::{FinalOnlyOutput, OdeOutputValues};
use ode::{OdeOptions, OdeRhs};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use std::time::Instant;

struct BenchRhs {
    plan: PreparedLindbladPlan,
    workspace: RhsWorkspace,
    mode: usize,
    slots: Vec<RuntimeValue>,
    eval_stack: Vec<RuntimeValue>,
    scalar_stack: Vec<Complex64>,
    coeffs: Vec<Complex64>,
    bases: Vec<Complex64>,
    rho: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl BenchRhs {
    fn new(plan: PreparedLindbladPlan, mode: usize) -> Self {
        let bases = plan.parameter_graph.base_values.iter().map(|v| match v {
            RuntimeValue::Scalar(x) => *x,
            _ => panic!("scalar benchmark only"),
        }).collect();
        Self {workspace: RhsWorkspace::new(&plan), plan, mode,
            slots: Vec::new(), eval_stack: Vec::new(), scalar_stack: Vec::new(),
            coeffs: Vec::new(), bases, rho: vec![Complex64::ZERO; N*N],
            scratch: vec![Complex64::ZERO; N*N]}
    }
}

impl OdeRhs for BenchRhs {
    fn dim(&self) -> usize { N*N }
    fn eval(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), String> {
        if self.mode < 2 {
            let mode = if self.mode == 0 {
                ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack
            } else { ExecutionMode::ExpandedSparse };
            return rhs_packed_into(&self.plan,y,t,mode,RhsOptions::default(),&mut self.workspace,out);
        }
        if self.mode == 4 {
            direct_coefficients(t,&self.bases,&mut self.coeffs);
        } else {
            self.plan.parameter_graph.evaluate_into(t,&mut self.slots,&mut self.eval_stack,&mut self.scalar_stack)?;
            self.plan.hamiltonian_plan.evaluate_decomposed_coefficients_into(
                &self.slots,t,&mut self.eval_stack,&mut self.scalar_stack,&mut self.coeffs)?;
        }
        if self.mode == 2 {
            self.plan.layout.unpack_into(y,&mut self.rho)?;
            generated_rows(&self.coeffs,y,out,&self.rho,&mut self.scratch);
        } else { generated_packed(&self.coeffs,y,out); }
        Ok(())
    }
}

#[pyfunction]
fn rhs_check<'py>(py: Python<'py>,payload: &Bound<'_, PyAny>,y: PyReadonlyArray1<'py,f64>,t:f64,mode:usize)
    -> PyResult<Bound<'py,PyArray1<f64>>> {
    let plan=parse_plan_payload(payload)?;
    let mut rhs=BenchRhs::new(plan,mode);
    let mut out=vec![0.;N*N];
    rhs.eval(t,y.as_slice()?,&mut out).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py,out))
}

#[pyfunction]
fn rhs_benchmark(payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'_,f64>,mode:usize,repeats:usize)
    ->PyResult<Vec<f64>> {
    let plan=parse_plan_payload(payload)?;
    let mut rhs=BenchRhs::new(plan,mode);
    let state=y.as_slice()?;
    let mut out=vec![0.;N*N];
    rhs.eval(125e-9,state,&mut out).map_err(PyValueError::new_err)?;
    let mut samples=Vec::new();
    for _ in 0..5 {
        let start=Instant::now();
        for _ in 0..repeats {
            for t in [125e-9,375e-9,625e-9,875e-9] {
                rhs.eval(t,std::hint::black_box(state),&mut out).map_err(PyValueError::new_err)?;
                std::hint::black_box(&out);
            }
        }
        samples.push(start.elapsed().as_secs_f64()/(4*repeats) as f64);
    }
    Ok(samples)
}

#[pyfunction]
fn trajectory<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'py,f64>,
    mode:usize,duration:f64,atol:f64,rtol:f64)
    ->PyResult<(Bound<'py,PyArray1<f64>>,f64,u64,u64,u64)> {
    let plan=parse_plan_payload(payload)?;
    let mut rhs=BenchRhs::new(plan,mode);
    let mut state=y.as_slice()?.to_vec();
    let start=Instant::now();
    let mut left=125e-9;
    let end=left+duration;
    let mut stats=(0,0,0);
    while left < end-1e-16 {
        let boundary=((left/250e-9+1e-8).floor()+1.)*250e-9;
        let right=boundary.min(end);
        // Evaluate square waves at this segment's midpoint. Each backend uses
        // identical fixed gates in RK endpoint stages, while EOM uses true time.
        let midpoint=(left+right)/2.;
        let gates=switch_gates(midpoint);
        for &idx in GATE_SLOTS {
            let value=match rhs.plan.parameter_graph.slot_names[idx].as_str() {
                "PA1"=>gates.0,"PY1"=>1.-gates.0,"PA2"=>gates.1,"PY2"=>1.-gates.1,_=>unreachable!()
            };
            rhs.plan.parameter_graph.base_values[idx]=RuntimeValue::Scalar(Complex64::new(value,0.));
            rhs.bases[idx]=Complex64::new(value,0.);
        }
        let vals:Vec<Complex64>=GATE_SLOTS.iter().map(|&i|rhs.bases[i]).collect();
        rhs.workspace.set_scalar_parameter_overrides(GATE_SLOTS,&vals).map_err(PyValueError::new_err)?;
        let options=OdeOptions {abstol:atol,reltol:rtol,dt:1e-10,maxiters:1000000,
            save_start:false,saveat:Some(vec![right])};
        let mut output=FinalOnlyOutput::new(N*N);
        let s=ode::tsit5::solve_tsit5(&mut rhs,&state,left,right,&options,&mut output)
            .map_err(PyValueError::new_err)?;
        stats.0+=s.rhs_calls;stats.1+=s.accepted_steps;stats.2+=s.rejected_steps;
        if let OdeOutputValues::Full(v)=output.finish().values {state=v;} else {unreachable!()}
        left=right;
    }
    Ok((PyArray1::from_vec(py,state),start.elapsed().as_secs_f64(),stats.0,stats.1,stats.2))
}

fn switch_gates(t:f64)->(f64,f64) {
    let a=if (std::f64::consts::TAU*1e6*t).sin()>=0. {1.} else {0.};
    let b=if (std::f64::consts::TAU*1e6*t+std::f64::consts::FRAC_PI_2).sin()>=0. {1.} else {0.};
    (a,b)
}

// MODULE_REGISTRATION is appended by the source generator.
