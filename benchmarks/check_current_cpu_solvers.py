"""Matched adaptive CPU solver comparison on the current Rust D+S+U path.

Builds isolated extensions from the live Rust modules. Benchmark methods are
selected inside one extension per model so compilation/layout overhead is
identical. A 10 us, explicitly segmented polarization-switch trajectory is
used to keep this a solver benchmark rather than an apparatus transit run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import bench_rust_specialized_rhs as base
import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf.constants import ED_XtB

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rust_specialized_rhs_results"
CACHE = base.CACHE / "current_cpu_solvers"
ROUNDS = 12
DURATION = 10e-6
ATOL = 1e-11
RTOL = 1e-9
METHODS = {"dopri5": 0, "tsit5": 1, "fixed_dopri5": 2, "fixed_rk2": 3, "fixed_rk4": 4}

WRAPPER = r"""
struct BenchRhs {
    plan:PreparedLindbladPlan, workspace:RhsWorkspace,
    gates:Vec<usize>, bases:Vec<Complex64>,
}
impl BenchRhs {
    fn new(plan:PreparedLindbladPlan)->Self {
        let gates=plan.parameter_graph.slot_names.iter().enumerate().filter_map(|(i,s)|
            if ["PA1","PY1","PA2","PY2"].contains(&s.as_str()) {Some(i)} else {None}).collect();
        let bases=plan.parameter_graph.base_values.iter().map(|v|match v {
            RuntimeValue::Scalar(x)=>*x,_=>panic!("scalar fixture")}).collect();
        Self{workspace:RhsWorkspace::new(&plan),plan,gates,bases}
    }
}
impl OdeRhs for BenchRhs {
    fn dim(&self)->usize {self.plan.layout.n*self.plan.layout.n}
    fn eval(&mut self,t:f64,y:&[f64],out:&mut[f64])->Result<(),String> {
        rhs_packed_into(&self.plan,y,t,ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack,
            RhsOptions::default(),&mut self.workspace,out)
    }
}

#[pyfunction]
fn trajectory<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'py,f64>,
    solver:usize,duration:f64,atol:f64,rtol:f64,dt:f64)
    ->PyResult<(Bound<'py,PyArray1<f64>>,f64,u64,u64,u64)> {
    let plan=parse_plan_payload(payload)?;
    let mut rhs=BenchRhs::new(plan);
    let mut state=y.as_slice()?.to_vec();
    let start=Instant::now();
    let mut left=125e-9;
    let end=left+duration;
    let mut stats=(0,0,0);
    while left < end-1e-16 {
        let boundary=((left/250e-9+1e-8).floor()+1.)*250e-9;
        let right=boundary.min(end);
        let midpoint=(left+right)/2.;
        let gates=switch_gates(midpoint);
        for &idx in &rhs.gates {
            let value=match rhs.plan.parameter_graph.slot_names[idx].as_str() {
                "PA1"=>gates.0,"PY1"=>1.-gates.0,"PA2"=>gates.1,"PY2"=>1.-gates.1,_=>unreachable!()
            };
            rhs.plan.parameter_graph.base_values[idx]=RuntimeValue::Scalar(Complex64::new(value,0.));
            rhs.bases[idx]=Complex64::new(value,0.);
        }
        let vals:Vec<Complex64>=rhs.gates.iter().map(|&i|rhs.bases[i]).collect();
        rhs.workspace.set_scalar_parameter_overrides(&rhs.gates,&vals).map_err(PyValueError::new_err)?;
        let options=OdeOptions {abstol:atol,reltol:rtol,dt,maxiters:1000000,
            save_start:false,saveat:Some(vec![right])};
        let mut output=FinalOnlyOutput::new(rhs.dim());
        let ode_solver=match solver {
            0=>OdeSolver::Dopri5,1=>OdeSolver::Tsit5,
            2=>OdeSolver::FixedDopri5,3=>OdeSolver::FixedRk2,4=>OdeSolver::FixedRk4,
            _=>return Err(PyValueError::new_err("unknown solver id")),
        };
        let s=solve_single(&mut rhs,&state,left,right,&options,&mut output,ode_solver)
            .map_err(PyValueError::new_err)?;
        stats.0+=s.rhs_calls;stats.1+=s.accepted_steps;stats.2+=s.rejected_steps;
        if let OdeOutputValues::Full(v)=output.finish().values {state=v;} else {unreachable!()}
        left=right;
    }
    Ok((PyArray1::from_vec(py,state),start.elapsed().as_secs_f64(),stats.0,stats.1,stats.2))
}

#[pyfunction]
fn trajectory_checkpoints<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'py,f64>,
    solver:usize,duration:f64,atol:f64,rtol:f64,dt:f64,checkpoints:Vec<f64>)
    ->PyResult<(Bound<'py,PyArray1<f64>>,f64,u64,u64,u64)> {
    let plan=parse_plan_payload(payload)?;
    let mut rhs=BenchRhs::new(plan);
    let mut state=y.as_slice()?.to_vec();
    let start=Instant::now();
    let t0=125e-9;
    let end=t0+duration;
    let mut checkpoints=checkpoints;
    checkpoints.sort_by(|a,b|a.total_cmp(b));
    for cp in &checkpoints {
        if !cp.is_finite() || *cp<=0. || *cp>duration {
            return Err(PyValueError::new_err("checkpoint durations must be finite and in (0,duration]"));
        }
    }
    if checkpoints.windows(2).any(|w|w[0]==w[1]) {
        return Err(PyValueError::new_err("checkpoint durations must be unique"));
    }
    let mut saved=vec![vec![0.;rhs.dim()];checkpoints.len()];
    let mut saved_mask=vec![false;checkpoints.len()];
    let mut left=t0;
    let mut stats=(0,0,0);
    while left < end-1e-16 {
        let boundary=((left/250e-9+1e-8).floor()+1.)*250e-9;
        let right=boundary.min(end);
        let gates=switch_gates((left+right)/2.);
        for &idx in &rhs.gates {
            let value=match rhs.plan.parameter_graph.slot_names[idx].as_str() {
                "PA1"=>gates.0,"PY1"=>1.-gates.0,"PA2"=>gates.1,"PY2"=>1.-gates.1,_=>unreachable!()
            };
            rhs.plan.parameter_graph.base_values[idx]=RuntimeValue::Scalar(Complex64::new(value,0.));
            rhs.bases[idx]=Complex64::new(value,0.);
        }
        let vals:Vec<Complex64>=rhs.gates.iter().map(|&i|rhs.bases[i]).collect();
        rhs.workspace.set_scalar_parameter_overrides(&rhs.gates,&vals).map_err(PyValueError::new_err)?;
        let mut targets:Vec<f64>=checkpoints.iter().map(|cp|t0+cp)
            .filter(|t|*t>left+1e-15 && *t<=right+1e-15).collect();
        if targets.last().map_or(true,|t|(*t-right).abs()>1e-15) {targets.push(right);}
        let options=OdeOptions {abstol:atol,reltol:rtol,dt,maxiters:1000000,
            save_start:false,saveat:Some(targets)};
        let mut output=FullOutput::new(rhs.dim(),options.saveat.as_ref().unwrap().len());
        let ode_solver=match solver {
            0=>OdeSolver::Dopri5,1=>OdeSolver::Tsit5,
            2=>OdeSolver::FixedDopri5,3=>OdeSolver::FixedRk2,4=>OdeSolver::FixedRk4,
            _=>return Err(PyValueError::new_err("unknown solver id")),
        };
        let s=solve_single(&mut rhs,&state,left,right,&options,&mut output,ode_solver)
            .map_err(PyValueError::new_err)?;
        stats.0+=s.rhs_calls;stats.1+=s.accepted_steps;stats.2+=s.rejected_steps;
        let output=output.finish();
        let OdeOutputValues::Full(values)=output.values else {unreachable!()};
        let mut end_state=None;
        for (row,&time) in output.times.iter().enumerate() {
            let value=&values[row*rhs.dim()..(row+1)*rhs.dim()];
            if (time-right).abs()<=1e-15 {end_state=Some(value.to_vec());}
            for (i,cp) in checkpoints.iter().enumerate() {
                if (time-(t0+cp)).abs()<=1e-15 {
                    saved[i].copy_from_slice(value);saved_mask[i]=true;
                }
            }
        }
        state=end_state.ok_or_else(||PyValueError::new_err("segment output missed its endpoint"))?;
        left=right;
    }
    if saved_mask.iter().any(|v|!*v) {
        return Err(PyValueError::new_err("solver did not return every requested checkpoint"));
    }
    let flat:Vec<f64>=saved.into_iter().flatten().collect();
    Ok((PyArray1::from_vec(py,flat),start.elapsed().as_secs_f64(),stats.0,stats.1,stats.2))
}

fn switch_gates(t:f64)->(f64,f64) {
    let a=if (std::f64::consts::TAU*1e6*t).sin()>=0. {1.} else {0.};
    let b=if (std::f64::consts::TAU*1e6*t+std::f64::consts::FRAC_PI_2).sin()>=0. {1.} else {0.};
    (a,b)
}
"""


def build_generic(tag=""):
    """One dynamic-dimension extension shared by all model fixtures."""
    suffix = f"_{tag}" if tag else ""
    module = "_current_cpu_solvers_generic" + suffix
    folder = CACHE / (tag if tag else "")
    folder.mkdir(parents=True, exist_ok=True)
    helper = (HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")
    imports = helper[: helper.index("struct BenchRhs")]
    imports += "use ode::batch::{solve_single,OdeSolver};\nuse ode::output::FullOutput;\n"
    source = imports + WRAPPER + f"""
#[pymodule]
fn {module}(m:&Bound<'_,PyModule>)->PyResult<()> {{
    m.add_function(wrap_pyfunction!(trajectory,m)?)?;
    m.add_function(wrap_pyfunction!(trajectory_checkpoints,m)?)?;
    Ok(())
}}
"""
    source = "\n".join([base_prefix(), source])
    (folder / "lib.rs").write_text(source, encoding="utf-8")
    (folder / "Cargo.toml").write_text(f'''[package]
name="{module}"
version="0.1.0"
edition="2021"
[lib]
name="{module}"
path="lib.rs"
crate-type=["cdylib"]
[dependencies]
pyo3={{version="0.27",features=["extension-module","num-complex"]}}
numpy="0.27"
num-complex="0.4"
[profile.release]
opt-level=3
lto="fat"
codegen-units=1
''', encoding="utf-8")
    env = os.environ.copy()
    env.update(PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(CACHE / "target"))
    with (folder / "build.log").open("w", encoding="utf-8") as log:
        done = subprocess.run(
            ["cargo", "build", "--release", "--offline", "--manifest-path", str(folder / "Cargo.toml")],
            env=env, stdout=log, stderr=log,
        )
    if done.returncode:
        raise RuntimeError((folder / "build.log").read_text(encoding="utf-8")[-7000:])
    target = folder / (module + ".pyd")
    target.write_bytes((CACHE / "target/release" / (module + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native, hashlib.sha256(source.encode()).hexdigest()


def base_prefix():
    repo = base.REPO
    return f'''#![allow(dead_code,unused_variables,unused_parens,non_snake_case)]
mod constants {{ pub const ED_XTB:f64={ED_XtB!r}; }}
#[path={json.dumps(str(repo / "rust/src/ode/mod.rs"))}] mod ode;
mod lindblad {{
#[path={json.dumps(str(repo / "rust/src/lindblad/cpu_kernel.rs"))}] pub mod cpu_kernel;
#[path={json.dumps(str(repo / "rust/src/lindblad/eval.rs"))}] pub mod eval;
#[path={json.dumps(str(repo / "rust/src/lindblad/layout.rs"))}] pub mod layout;
#[path={json.dumps(str(repo / "rust/src/lindblad/blas.rs"))}] pub mod blas;
#[path={json.dumps(str(repo / "rust/src/lindblad/plan.rs"))}] pub mod plan;
#[path={json.dumps(str(repo / "rust/src/lindblad/rhs.rs"))}] pub mod rhs;
}}
'''


def density_accuracy(prepared, actual, reference):
    rho = prepared.layout.unpack(np.asarray(actual))
    ref = prepared.layout.unpack(np.asarray(reference))
    if not np.isfinite(rho).all() or not np.isfinite(ref).all():
        return dict(finite=False,max_relative_above_1e6=None,
                    max_absolute_at_or_below_1e6=None,trace_difference=None,
                    trace_error_from_one=None,passed=False)
    delta = np.abs(rho - ref)
    magnitudes = np.abs(ref)
    high = magnitudes > 1e-6
    rel = np.zeros_like(magnitudes, dtype=float)
    rel[high] = delta[high] / magnitudes[high]
    absolute = float(delta[~high].max(initial=0.0))
    relative = float(rel[high].max(initial=0.0))
    trace_error = float(abs(np.trace(rho) - np.trace(ref)))
    trace_error_from_one = float(abs(np.trace(rho) - 1.0))
    return dict(
        max_relative_above_1e6=relative,
        max_absolute_at_or_below_1e6=absolute,
        trace_difference=trace_error,
        trace_error_from_one=trace_error_from_one,
        finite=True,
        passed=(relative <= 0.01 and absolute <= 1e-6 and trace_error <= 1e-9
                and trace_error_from_one <= 1e-9),
    )


def fixed_screen():
    """Bounded fixed-step feasibility check, then pair any accurate RC finalist."""
    RESULTS.mkdir(exist_ok=True)
    folder = CACHE
    module = "_current_cpu_solvers_generic"
    extension = folder / (module + ".pyd")
    spec = importlib.util.spec_from_file_location(module, extension)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    result = dict(
        note="Bounded fixed-step feasibility screen at dt=5 ns on RC; only an accurate finalist is repeated or checked on R2.",
        accuracy_criterion="Same full-density 1%/1e-6 and trace criterion as adaptive comparison.",
        systems=[],
    )
    outfile = RESULTS / "current_cpu_solvers_fixed_screen.json"
    for name in ["rc", "r2"]:
        prepared, rho0, _gates = base.model(name)
        payload = prepared.to_payload()
        y0 = prepared.layout.pack(rho0)
        ref = native.trajectory(payload,y0,METHODS["tsit5"],DURATION,1e-13,1e-11,1e-10)
        reference = np.asarray(ref[0])
        row = dict(system=name, n_states=prepared.layout.n, candidates=[])
        dt_values = [5e-9]
        for method in ["fixed_dopri5", "fixed_rk4", "fixed_rk2"]:
            for dt in dt_values:
                run = native.trajectory(payload,y0,METHODS[method],DURATION,ATOL,RTOL,dt)
                acc = density_accuracy(prepared,run[0],reference)
                row["candidates"].append(dict(method=method,dt_s=dt,seconds=run[1],
                    rhs_calls=run[2],steps=run[3],accuracy=acc))
        # RC can require sub-ns steps. Check one bounded RK4 point below that
        # scale; if it fails, further shrinking would be a different project.
        if name == "rc":
            dt=0.5e-9
            run=native.trajectory(payload,y0,METHODS["fixed_rk4"],DURATION,ATOL,RTOL,dt)
            acc=density_accuracy(prepared,run[0],reference)
            row["candidates"].append(dict(method="fixed_rk4",dt_s=dt,seconds=run[1],
                rhs_calls=run[2],steps=run[3],accuracy=acc))
        # If any fixed method meets the user-approved density criterion, paired
        # timings compare the fastest feasible RC fixed candidate against Tsit5.
        feasible = [c for c in row["candidates"] if c["accuracy"]["passed"]]
        if feasible:
            winner = min(feasible,key=lambda c:c["seconds"])
            pair = {winner["method"]:[],"tsit5":[]}
            ordering=[]
            for repeat in range(ROUNDS):
                order=[winner["method"],"tsit5"] if repeat%2==0 else ["tsit5",winner["method"]]
                ordering.append(order)
                for method in order:
                    mid=METHODS[method]
                    dt=winner["dt_s"] if method==winner["method"] else 1e-10
                    run=native.trajectory(payload,y0,mid,DURATION,ATOL,RTOL,dt)
                    acc=density_accuracy(prepared,run[0],reference)
                    if not acc["passed"]:
                        raise AssertionError((name,method,repeat,acc))
                    pair[method].append(dict(seconds=run[1],rhs_calls=run[2],steps=run[3],
                                             rejected=run[4],accuracy=acc))
            row["paired_finalist"] = dict(method=winner["method"],dt_s=winner["dt_s"],
                order_by_round=ordering,
                medians_s={m:float(np.median([x["seconds"] for x in pair[m]])) for m in pair},
                rounds=pair)
        result["systems"].append(row)
        outfile.write_text(json.dumps(result,indent=2),encoding="utf-8")
        if name == "rc" and not feasible:
            break
    outfile.write_text(json.dumps(result,indent=2),encoding="utf-8")


def main():
    RESULTS.mkdir(exist_ok=True)
    report = dict(
        description="Current production prepared H/DAG and D+S+U; segmented 10 us trajectory, not apparatus transit.",
        python=sys.version,
        platform=platform.platform(),
        duration_s=DURATION,
        rounds=ROUNDS,
        candidate_tolerances={"atol":ATOL,"rtol":RTOL},
        accuracy_criterion="1% entrywise relative above |rho_ref|=1e-6; 1e-6 absolute below; trace difference and |trace-1| <=1e-9.",
        systems=[],
    )
    output = RESULTS / "current_cpu_solvers_paired.json"
    progress = RESULTS / "current_cpu_solvers_progress.json"

    def save():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with threadpool_limits(limits=1):
        native_ext = None
        source_hash = None
        for name in ["rc", "r2", "r0", "q1"]:
            prepared, rho0, gates = base.model(name)
            payload = prepared.to_payload()
            y0 = prepared.layout.pack(rho0)
            if native_ext is None:
                native_ext, source_hash = build_generic()
            native = native_ext
            # High-accuracy adaptive reference uses the current same RHS and segmentation.
            ref_run = native.trajectory(payload, y0, 0, DURATION, 1e-13, 1e-11, 1e-10)
            rust_sources = ["rust/src/lindblad/eval.rs", "rust/src/lindblad/plan.rs",
                            "rust/src/lindblad/rhs.rs", "rust/src/ode/batch.rs",
                            "rust/src/ode/common.rs", "rust/src/ode/dopri5.rs",
                            "rust/src/ode/fixed.rs", "rust/src/ode/output.rs",
                            "rust/src/ode/tsit5.rs"]
            row = dict(system=name, n_states=prepared.layout.n, native_source_sha256=source_hash,
                       rust_source_sha256={p:hashlib.sha256((base.REPO/p).read_bytes()).hexdigest()
                                           for p in rust_sources},
                       reference={"method":"tsit5","atol":1e-13,"rtol":1e-11,
                                  "rhs_calls":ref_run[2],"accepted_steps":ref_run[3],
                                  "rejected_steps":ref_run[4]}, candidates=[])
            reference = np.asarray(ref_run[0])
            # Warm all requested adaptive methods before releasing the timed slot.
            for method in ["dopri5", "tsit5"]:
                candidate = native.trajectory(payload, y0, METHODS[method], DURATION, ATOL, RTOL, 1e-10)
                accuracy = density_accuracy(prepared, candidate[0], reference)
                if not accuracy["passed"]:
                    raise AssertionError((name, method, accuracy))
            timings = {method: [] for method in ["dopri5", "tsit5"]}
            details = {method: [] for method in timings}
            order_by_round = []
            for repeat in range(ROUNDS):
                # Exact counterbalancing: each solver runs first in six rounds.
                order = ["dopri5", "tsit5"] if repeat % 2 == 0 else ["tsit5", "dopri5"]
                order_by_round.append(list(order))
                for method in order:
                    run = native.trajectory(payload, y0, METHODS[method], DURATION, ATOL, RTOL, 1e-10)
                    accuracy = density_accuracy(prepared, run[0], reference)
                    assert accuracy["passed"], (name, method, repeat, accuracy)
                    timings[method].append(run[1])
                    details[method].append(dict(seconds=run[1], rhs_calls=run[2],
                                                accepted_steps=run[3],rejected_steps=run[4],
                                                round=repeat+1, position=order.index(method)+1,
                                                accuracy=accuracy))
                progress.write_text(json.dumps(dict(system=name,repeat=repeat+1,rounds=ROUNDS),indent=2))
            for method in timings:
                row["candidates"].append(dict(
                    method=method, median_s=float(np.median(timings[method])),
                    samples_s=timings[method], runs=details[method],
                ))
            row["paired_dopri5_over_tsit5_runtime_ratio"] = float(
                np.median(np.asarray(timings["dopri5"]) / np.asarray(timings["tsit5"]))
            )
            row["execution_order_by_round"] = order_by_round
            report["systems"].append(row)
            save()
    save()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixed-screen",action="store_true")
    args=parser.parse_args()
    fixed_screen() if args.fixed_screen else main()
