"""Isolated RC/R(2) trials for repeated-time H reuse and an AVX2 sparse commutator.

Builds benchmark-only Rust extensions from the current production plan/evaluator/RHS
and Tsit5. It never edits production sources or the installed extension. Timed runs
are intentionally behind --run because the coordinator assigns exclusive slots.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import bench_rust_specialized_rhs as base
import confirm_prepared_hamiltonian_integration as integrated
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "rust_stage_reuse_simd_results"
CACHE = REPO / ".cache/benchmarks/stage_reuse_simd_v1"
SYSTEMS = ["rc", "r2"]
ROUNDS = 12
MODES = ["control", "hooked_control", "time_cache", "avx2", "both"]
STRICT_ATOL, STRICT_RTOL = 1e-13, 1e-11
RUN_ATOL, RUN_RTOL = 1e-11, 1e-9
DURATION = 10e-6

NATIVE_WRAPPER = r"""
use lindblad::eval::RuntimeValue;
use lindblad::plan::parse_plan_payload;
use lindblad::rhs::{rhs_packed_into, rhs_packed_into_with_profile, ExecutionMode,
    RhsOptions, RhsProfileStats, RhsWorkspace};
use num_complex::Complex64;
use numpy::{PyArray1, PyReadonlyArray1};
use ode::output::{FinalOnlyOutput, OdeOutputValues};
use ode::{OdeOptions, OdeRhs};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use std::time::Instant;

struct BenchRhs {
    plan: lindblad::plan::PreparedLindbladPlan,
    workspace: RhsWorkspace,
    gates: Vec<usize>,
    bases: Vec<Complex64>,
    profile: RhsProfileStats,
    profile_enabled: bool,
    stage_times: Vec<u64>,
}
impl BenchRhs {
    fn new(plan: lindblad::plan::PreparedLindbladPlan, mode: usize, profile_enabled: bool) -> Self {
        let gates = plan.parameter_graph.slot_names.iter().enumerate().filter_map(|(i,s)|
            if ["PA1","PY1","PA2","PY2"].contains(&s.as_str()) {Some(i)} else {None}).collect();
        let bases = plan.parameter_graph.base_values.iter().map(|v| match v {
            RuntimeValue::Scalar(x) => *x, _ => panic!("scalar fixture") }).collect();
        let mut workspace = RhsWorkspace::new(&plan);
        workspace.bench_reuse = mode == 1 || mode == 3;
        workspace.bench_simd = mode == 2 || mode == 3;
        Self {plan,workspace,gates,bases,profile:RhsProfileStats::default(),profile_enabled,
            stage_times:Vec::new()}
    }
    fn dim(&self)->usize { self.plan.layout.n * self.plan.layout.n }
}
impl OdeRhs for BenchRhs {
    fn dim(&self)->usize {self.dim()}
    fn eval(&mut self,t:f64,y:&[f64],out:&mut[f64])->Result<(),String> {
        if self.profile_enabled {
            self.stage_times.push(t.to_bits());
            rhs_packed_into_with_profile(&self.plan,y,t,
                ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack,
                RhsOptions::default(),&mut self.workspace,out,Some(&mut self.profile))
        } else {
            rhs_packed_into(&self.plan,y,t,
                ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack,
                RhsOptions::default(),&mut self.workspace,out)
        }
    }
}

fn gates_at(t:f64)->(f64,f64) {
    let a=if (std::f64::consts::TAU*1e6*t).sin()>=0. {1.} else {0.};
    let b=if (std::f64::consts::TAU*1e6*t+std::f64::consts::FRAC_PI_2).sin()>=0. {1.} else {0.};
    (a,b)
}
fn profile_values(p:&RhsProfileStats)->Vec<f64> { vec![p.calls as f64,p.total_seconds,
    p.unpack_seconds,p.parameter_eval_seconds,p.hamiltonian_fill_seconds,
    p.commutator_seconds,p.dissipator_seconds,p.pack_seconds] }

#[pyfunction]
fn rhs_check<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'py,f64>,t:f64,mode:usize)
    ->PyResult<Bound<'py,PyArray1<f64>>> {
    let plan=parse_plan_payload(payload)?; let mut rhs=BenchRhs::new(plan,mode,false);
    let mut out=vec![0.;rhs.dim()]; rhs.eval(t,y.as_slice()?,&mut out).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py,out))
}

#[pyfunction]
fn same_time_override_check<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,
    y:PyReadonlyArray1<'py,f64>,t:f64,slots:Vec<usize>,values:Vec<(f64,f64)>)
    ->PyResult<Bound<'py,PyArray1<f64>>> {
    let plan=parse_plan_payload(payload)?; let mut rhs=BenchRhs::new(plan,1,false);
    let state=y.as_slice()?; let mut out=vec![0.;rhs.dim()];
    rhs.eval(t,state,&mut out).map_err(PyValueError::new_err)?;
    let vals:Vec<Complex64>=values.into_iter().map(|(re,im)|Complex64::new(re,im)).collect();
    rhs.workspace.set_scalar_parameter_overrides(&slots,&vals).map_err(PyValueError::new_err)?;
    rhs.eval(t,state,&mut out).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py,out))
}

#[pyfunction]
fn trajectory<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'py,f64>,
    mode:usize,duration:f64,atol:f64,rtol:f64,profile:bool)
    ->PyResult<(Bound<'py,PyArray1<f64>>,f64,u64,u64,u64,Vec<u64>,Vec<f64>)> {
    let plan=parse_plan_payload(payload)?; let mut rhs=BenchRhs::new(plan,mode,profile);
    let mut state=y.as_slice()?.to_vec(); let start=Instant::now();
    let mut left=125e-9; let end=left+duration; let mut stats=(0,0,0);
    while left<end-1e-16 {
        let boundary=((left/250e-9+1e-8).floor()+1.)*250e-9; let right=boundary.min(end);
        let gates=gates_at((left+right)/2.);
        for &idx in &rhs.gates {
            let value=match rhs.plan.parameter_graph.slot_names[idx].as_str() {
                "PA1"=>gates.0,"PY1"=>1.-gates.0,"PA2"=>gates.1,"PY2"=>1.-gates.1,_=>unreachable!()};
            rhs.plan.parameter_graph.base_values[idx]=RuntimeValue::Scalar(Complex64::new(value,0.));
            rhs.bases[idx]=Complex64::new(value,0.);
        }
        let vals:Vec<Complex64>=rhs.gates.iter().map(|&i|rhs.bases[i]).collect();
        rhs.workspace.set_scalar_parameter_overrides(&rhs.gates,&vals).map_err(PyValueError::new_err)?;
        let options=OdeOptions {abstol:atol,reltol:rtol,dt:1e-10,maxiters:1000000,
            save_start:false,saveat:Some(vec![right])};
        let mut output=FinalOnlyOutput::new(rhs.dim());
        let s=ode::tsit5::solve_tsit5(&mut rhs,&state,left,right,&options,&mut output)
            .map_err(PyValueError::new_err)?;
        stats.0+=s.rhs_calls; stats.1+=s.accepted_steps; stats.2+=s.rejected_steps;
        if let OdeOutputValues::Full(v)=output.finish().values {state=v;} else {unreachable!()}
        left=right;
    }
    let elapsed=start.elapsed().as_secs_f64(); let prof=profile_values(&rhs.profile);
    Ok((PyArray1::from_vec(py,state),elapsed,stats.0,stats.1,stats.2,rhs.stage_times,prof))
}

#[pyfunction]
fn avx2_available()->bool { #[cfg(target_arch="x86_64")] {std::is_x86_feature_detected!("avx2")} #[cfg(not(target_arch="x86_64"))] {false} }
"""
NATIVE_WRAPPER_CONTROL = NATIVE_WRAPPER.replace(
    "        workspace.bench_reuse = mode == 1 || mode == 3;\n        workspace.bench_simd = mode == 2 || mode == 3;\n",
    "        let _ = mode;\n",
)


def patch_rhs(source: str) -> str:
    source = source.replace(
        "pub struct RhsWorkspace {",
        "pub struct RhsWorkspace {\n    pub bench_reuse: bool,\n    pub bench_simd: bool,\n    pub bench_cached_time: Option<u64>,",
        1,
    )
    source = source.replace(
        "Self {\n            upper_layout: upper_layout.clone(),",
        "Self {\n            bench_reuse: false, bench_simd: false, bench_cached_time: None,\n            upper_layout: upper_layout.clone(),",
        1,
    )
    source = source.replace(
        "self.hamiltonian_valid_for = None;\n        self.h_sparse_valid = false;",
        "self.hamiltonian_valid_for = None;\n        self.h_sparse_valid = false;\n        self.bench_cached_time = None;",
        1,
    )
    old = "let can_skip = !plan.is_time_dependent && workspace.hamiltonian_valid_for == Some(cache_kind);"
    new = "let can_skip = (!plan.is_time_dependent && workspace.hamiltonian_valid_for == Some(cache_kind)) || (workspace.bench_reuse && workspace.hamiltonian_valid_for == Some(cache_kind) && workspace.bench_cached_time == Some(t.to_bits()));"
    assert source.count(old) == 1
    source = source.replace(old, new)
    marker = "workspace.hamiltonian_valid_for = Some(cache_kind);\n        workspace.h_sparse_valid = false;"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        "workspace.hamiltonian_valid_for = Some(cache_kind);\n        workspace.bench_cached_time = Some(t.to_bits());\n        workspace.h_sparse_valid = false;",
    )
    old = "if mode.uses_soa_scratch() {\n                add_commutator_sparse_dense_upper_soa("
    new = "if workspace.bench_simd {\n                crate::add_commutator_sparse_dense_upper_avx2(&plan.hamiltonian_sparse_pattern, workspace.h_sparse_values.as_slice(), workspace.rho.as_slice(), workspace.scratch.as_mut_slice(), &workspace.upper_layout, workspace.drho_upper.as_mut_slice());\n            } else if mode.uses_soa_scratch() {\n                add_commutator_sparse_dense_upper_soa("
    assert source.count(old) == 1
    source = source.replace(old, new)
    return source


def build(prepared, gates, hooked: bool):
    base.CACHE = CACHE
    base.RESULTS = RESULTS
    tag = "hooks" if hooked else "control"
    module = "_stage_reuse_simd_" + tag
    folder = CACHE / tag
    base.generate(prepared, gates, folder, module)
    generated = (folder / "lib.rs").read_text(encoding="utf-8")
    prefix = generated[: generated.index("// Benchmark-only wrapper")]
    rhs_path = REPO / "rust/src/lindblad/rhs.rs"
    current_rhs = rhs_path.read_text(encoding="utf-8")
    rhs_snapshot = patch_rhs(current_rhs) if hooked else current_rhs
    snapshot = folder / "rhs.rs"
    snapshot.write_text(rhs_snapshot, encoding="utf-8")
    prefix = prefix.replace(json.dumps(str(rhs_path)), json.dumps(str(snapshot)))
    avx = (HERE / "stage_reuse_simd_kernel.rs").read_text(encoding="utf-8") if hooked else ""
    wrapper = NATIVE_WRAPPER if hooked else NATIVE_WRAPPER_CONTROL
    source = prefix + wrapper + avx + f"\n#[pymodule]\nfn {module}(m:&Bound<'_,PyModule>)->PyResult<()> {{\n"
    source += "m.add_function(wrap_pyfunction!(rhs_check,m)?)?;m.add_function(wrap_pyfunction!(trajectory,m)?)?;"
    source += "m.add_function(wrap_pyfunction!(avx2_available,m)?)?;"
    if hooked:
        source += "m.add_function(wrap_pyfunction!(same_time_override_check,m)?)?;"
    source += "Ok(())}\n"
    source = "\n".join(line for line in source.splitlines()
        if not line.startswith(("const N:usize=", "const GATE_SLOTS:")))
    (folder / "lib.rs").write_text(source, encoding="utf-8")
    env = os.environ.copy()
    env.update(PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(CACHE / "target"))
    with (folder / "build.log").open("w", encoding="utf-8") as log:
        done = subprocess.run(["cargo", "build", "--release", "--offline", "--manifest-path", str(folder / "Cargo.toml")],
            env=env, stdout=log, stderr=log)
    if done.returncode:
        raise RuntimeError((folder / "build.log").read_text(encoding="utf-8")[-8000:])
    target = folder / (module + ".pyd")
    target.write_bytes((CACHE / "target/release" / (module + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    return native, {"wrapper": hashlib.sha256(source.encode()).hexdigest(),
        "rhs": hashlib.sha256(rhs_snapshot.encode()).hexdigest(),
        "eval": hashlib.sha256((REPO / "rust/src/lindblad/eval.rs").read_bytes()).hexdigest(),
        "plan": hashlib.sha256((REPO / "rust/src/lindblad/plan.rs").read_bytes()).hexdigest()}


def repeat_time_summary(bits):
    counts = Counter(bits)
    return {"rhs_evaluations": len(bits), "unique_times": len(counts),
        "times_repeated": sum(n > 1 for n in counts.values()),
        "extra_same_time_rhs_calls": sum(n - 1 for n in counts.values() if n > 1),
        "max_calls_at_one_time": max(counts.values(), default=0)}


def main():
    if "--run" not in sys.argv and "--build-only" not in sys.argv:
        raise SystemExit("Build/correctness mode only: rerun with --run during the coordinator's timing slot.")
    RESULTS.mkdir(exist_ok=True)
    report = {"python":sys.version,"platform":platform.platform(),"systems":[],
        "duration_s":DURATION,"rounds":ROUNDS,"modes":MODES,
        "tolerances":{"atol":RUN_ATOL,"rtol":RUN_RTOL,"strict_atol":STRICT_ATOL,"strict_rtol":STRICT_RTOL}}
    (RESULTS / "progress.json").write_text(json.dumps({"stage":"initializing"},indent=2),encoding="utf-8")

    def save(): (RESULTS / "stage_reuse_simd.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    def progress(stage, **details):
        record={"stage":stage,**details}
        (RESULTS/"progress.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
        print(stage,details,flush=True)

    with threadpool_limits(limits=1):
        first = base.model("r2")
        control, report["control_sources"] = build(first[0], first[2], False)
        hooks, report["hook_sources"] = build(first[0], first[2], True)
        report["avx2_available"] = bool(hooks.avx2_available())
        save()
        if "--build-only" in sys.argv:
            (RESULTS / "progress.json").write_text(json.dumps({"stage":"native builds complete"},indent=2),encoding="utf-8")
            return
        for name in SYSTEMS:
            progress("model setup",system=name)
            prepared,rho0,gates = first if name == "r2" else base.model(name)
            assert prepared.to_payload()["hamiltonian_plan"].get("prepared_hamiltonian_assembly",True)
            assert prepared.to_payload()["hamiltonian_plan"].get("shared_coefficient_dag",True)
            payload=prepared.to_payload(); y0=prepared.layout.pack(rho0); n=prepared.layout.n
            row={"system":name,"n_states":n,"gate_slots":gates,"times_by_round_s":[],
                "orders":[],"density_checks":[],"profile":{},"stage_times":{},"normalized_rhs_error":0.0}
            report["systems"].append(row); save()
            progress("random RHS and cache invalidation checks",system=name)
            rng=np.random.default_rng(1481); a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); a=a@a.conj().T
            random_y=prepared.layout.pack(a/np.trace(a))
            for t in [125e-9,375e-9,625e-9,875e-9]:
                expected=np.asarray(control.rhs_check(payload,random_y,t,0))
                for mode in [0,1,2,3]:
                    actual=np.asarray(hooks.rhs_check(payload,random_y,t,mode))
                    err=float(np.max(np.abs(actual-expected))/max(np.max(np.abs(expected)),1.0))
                    row["normalized_rhs_error"]=max(row["normalized_rhs_error"],err)
                    assert err<1e-10,(name,mode,t,err)
            # Same timestamp, changed gates: an old cached H must be invalidated.
            if gates:
                gate_ids=[payload["parameter_graph"]["slot_names"].index(g) for g in gates]
                vals=[]
                changed=copy.deepcopy(payload)
                for key,idx in zip(gates,gate_ids):
                    value=0.25 if key.startswith("PA") else 0.75
                    vals.append((value,0.0))
                    changed_value=copy.deepcopy(changed["parameter_graph"]["base_values"][idx])
                    changed_value["re"]=value; changed_value["im"]=0.0
                    changed["parameter_graph"]["base_values"][idx]=changed_value
                cached=np.asarray(hooks.same_time_override_check(payload,random_y,375e-9,gate_ids,vals))
                fresh=np.asarray(control.rhs_check(changed,random_y,375e-9,0))
                err=float(np.max(np.abs(cached-fresh))/max(np.max(np.abs(fresh)),1.0))
                row["same_time_gate_change_error"]=err; assert err<1e-10,(name,err)
            progress("strict reference and candidate correctness",system=name)
            strict=np.asarray(control.trajectory(payload,y0,0,DURATION,STRICT_ATOL,STRICT_RTOL,False)[0])
            methods={"control":(control,0),"hooked_control":(hooks,0),"time_cache":(hooks,1),"avx2":(hooks,2),"both":(hooks,3)}
            for tag,(native,mode) in methods.items():
                run=native.trajectory(payload,y0,mode,DURATION,RUN_ATOL,RUN_RTOL,False)
                result=prepared.layout.unpack(np.asarray(run[0])); acc=base.accuracy(result,prepared.layout.unpack(strict))
                assert acc["passed"],(name,tag,acc)
                assert abs(np.trace(result)-1)<1e-9,(name,tag,np.trace(result))
                row.setdefault("correctness",{})[tag]={"density":acc,"trace_error":float(abs(np.trace(result)-1)),
                    "rhs_calls":run[2],"accepted":run[3],"rejected":run[4]}

            progress("profile passes",system=name)
            for tag,mode in [("hooked_control",0),("time_cache",1)]:
                prof=hooks.trajectory(payload,y0,mode,DURATION,RUN_ATOL,RUN_RTOL,True)
                row["profile"][tag]={"fields":["calls","rhs_total","unpack","parameters","hamiltonian_fill","commutator","dissipator","pack"],
                    "values":prof[6],"repeat_time_summary":repeat_time_summary(prof[5]),"profiled_wall_s":prof[1]}
                row["stage_times"][tag]=repeat_time_summary(prof[5])
            rng_order=np.random.default_rng(9871)
            # Twelve randomized interleaved rounds; five modes per round.
            for repeat in range(ROUNDS):
                order=MODES.copy(); rng_order.shuffle(order)
                progress("timed round",system=name,round=repeat+1,total_rounds=ROUNDS,order=order)
                times=[None]*len(MODES); checks=[None]*len(MODES)
                for tag in order:
                    native,mode=methods[tag]; run=native.trajectory(payload,y0,mode,DURATION,RUN_ATOL,RUN_RTOL,False)
                    result=prepared.layout.unpack(np.asarray(run[0])); acc=base.accuracy(result,prepared.layout.unpack(strict))
                    assert acc["passed"],(name,tag,acc)
                    idx=MODES.index(tag); times[idx]=run[1]
                    checks[idx]={"density":acc,"trace_error":float(abs(np.trace(result)-1)),"rhs_calls":run[2],
                        "accepted":run[3],"rejected":run[4]}
                row["times_by_round_s"].append(times); row["orders"].append(order); row["density_checks"].append(checks); save()
            med=np.median(np.asarray(row["times_by_round_s"]),axis=0)
            row["median_s"]={tag:float(med[i]) for i,tag in enumerate(MODES)}
            row["reductions_vs_control_percent"]={tag:float(100*(1-med[i]/med[0])) for i,tag in enumerate(MODES[1:],1)}
            save()
    save()

if __name__=="__main__": main()
