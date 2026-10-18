"""One reusable Rust extension: factorial prepared-data and commutator experiments."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import bench_rust_specialized_rhs as base
import numpy as np
from threadpoolctl import threadpool_limits

MODULE = "_rhs_generic_prepared"
FOLDER = base.CACHE / "generic_prepared"
FAMILIES = ["D+S+U_SoA", "S+U_SoA", "D+S+U_complex", "S+U_complex"]
FLAGS = ["prepared_H", "nonzero_decay", "shared_coefficient_DAG", "active_pattern"]

WRAPPER = r"""
struct BenchRhs {
    plan:PreparedLindbladPlan, workspace:RhsWorkspace, mode:usize,
    slots:Vec<RuntimeValue>, eval_stack:Vec<RuntimeValue>, scalar_stack:Vec<Complex64>,
    coeffs:Vec<Complex64>, bases:Vec<Complex64>, gates:Vec<usize>,
    generic:lindblad::rhs::GenericPrepared, dag:lindblad::eval::CoefficientDag,
}
impl BenchRhs {
    fn new(plan:PreparedLindbladPlan,mode:usize)->Self {
        let gates=plan.parameter_graph.slot_names.iter().enumerate().filter_map(|(i,s)|
            if ["PA1","PY1","PA2","PY2"].contains(&s.as_str()) {Some(i)} else {None}).collect::<Vec<_>>();
        let generic=lindblad::rhs::GenericPrepared::new(&plan,&gates);
        let dag=lindblad::eval::CoefficientDag::new(plan.hamiltonian_plan.coefficients.iter().map(|c|c.expression.clone()));
        let bases=plan.parameter_graph.base_values.iter().map(|v|match v {RuntimeValue::Scalar(x)=>*x,_=>panic!("scalar fixture")}).collect();
        Self{workspace:RhsWorkspace::new(&plan),plan,mode,generic,dag,gates,bases,
            slots:Vec::new(),eval_stack:Vec::new(),scalar_stack:Vec::new(),coeffs:Vec::new()}
    }
}
impl OdeRhs for BenchRhs {
    fn dim(&self)->usize {self.plan.layout.n*self.plan.layout.n}
    fn eval(&mut self,t:f64,y:&[f64],out:&mut[f64])->Result<(),String> {
        if self.mode==0 {return rhs_packed_into(&self.plan,y,t,
            ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack,
            RhsOptions::default(),&mut self.workspace,out);}
        let flags=(self.mode-16)%16;let kernel=(self.mode-16)/16;
        self.plan.parameter_graph.evaluate_into(t,&mut self.slots,&mut self.eval_stack,&mut self.scalar_stack)?;
        if flags&4!=0 {self.dag.evaluate(&self.slots,t,&mut self.coeffs)?;} else {
            self.plan.hamiltonian_plan.evaluate_decomposed_coefficients_into(&self.slots,t,
                &mut self.eval_stack,&mut self.scalar_stack,&mut self.coeffs)?;
        }
        lindblad::rhs::benchmark_generic_rhs(&self.plan,y,out,t,&self.slots,&self.coeffs,
            &mut self.workspace,&mut self.generic,flags,kernel)
    }
}
#[pyfunction]
fn preparation_info(payload:&Bound<'_,PyAny>)->PyResult<(f64,usize,usize)> {
    let plan=parse_plan_payload(payload)?;let tick=Instant::now();let rhs=BenchRhs::new(plan,16);
    Ok((tick.elapsed().as_secs_f64(),rhs.dag.original_ops,rhs.dag.node_count()))
}
"""


def build(prepared, gates):
    base.generate(prepared, gates, FOLDER, MODULE)
    original = (FOLDER / "lib.rs").read_text(encoding="utf-8")
    prefix = original[: original.index("// Benchmark-only wrapper")]
    rhs_path = base.REPO / "rust/src/lindblad/rhs.rs"
    eval_path = base.REPO / "rust/src/lindblad/eval.rs"
    for path, helper in [
        (rhs_path, "generic_prepared_rhs.rs"),
        (eval_path, "generic_coefficient_dag.rs"),
    ]:
        snapshot = FOLDER / path.name
        snapshot.write_text(
            path.read_text(encoding="utf-8")
            + "\n"
            + (base.HERE / helper).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        prefix = prefix.replace(json.dumps(str(path)), json.dumps(str(snapshot)))
    wrapper = (base.HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")
    imports = wrapper[: wrapper.index("struct BenchRhs")]
    tail = wrapper[wrapper.index("#[pyfunction]") :]
    tail = tail.replace("N*N", "rhs.dim()").replace("GATE_SLOTS", "&rhs.gates")
    tail = tail.replace("=&rhs.gates.iter()", "=rhs.gates.iter()")
    tail = tail.replace(
        "let options=OdeOptions",
        "rhs.generic.refresh(&rhs.plan,&rhs.gates);\n        let options=OdeOptions",
    )
    registration = """
#[pymodule]
fn _rhs_generic_prepared(m:&Bound<'_,PyModule>)->PyResult<()> {
    m.add_function(wrap_pyfunction!(rhs_check,m)?)?;
    m.add_function(wrap_pyfunction!(rhs_benchmark,m)?)?;
    m.add_function(wrap_pyfunction!(trajectory,m)?)?;
    m.add_function(wrap_pyfunction!(preparation_info,m)?)?;
    Ok(())
}
"""
    source = prefix + imports + WRAPPER + tail + registration
    # Remove the model-size and gate constants: this binary serves every fixture.
    source = "\n".join(
        line
        for line in source.splitlines()
        if not line.startswith("const GATE_SLOTS:") and not line.startswith("const N:usize=")
    )
    (FOLDER / "lib.rs").write_text(source, encoding="utf-8")
    env = os.environ.copy()
    env.update(PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(base.CACHE / "target"))
    tick = time.perf_counter()
    with (FOLDER / "build.log").open("w", encoding="utf-8") as log:
        done = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--offline",
                "--manifest-path",
                str(FOLDER / "Cargo.toml"),
            ],
            env=env,
            stdout=log,
            stderr=log,
        )
    if done.returncode:
        raise RuntimeError((FOLDER / "build.log").read_text(encoding="utf-8")[-6500:])
    seconds = time.perf_counter() - tick
    target = FOLDER / (MODULE + ".pyd")
    target.write_bytes((base.CACHE / "target/release" / (MODULE + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(MODULE, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native, dict(
        compile_s=seconds, source_sha256=hashlib.sha256(source.encode()).hexdigest()
    )


def describe(mode):
    if mode == 0:
        return "library_D+S+U"
    kernel, flags = divmod(mode - 16, 16)
    return FAMILIES[kernel] + ":" + ",".join(f for i, f in enumerate(FLAGS) if flags & (1 << i))


def run(native, name, prepared, rho0, gates, build_info):
    report = dict(
        system=name,
        n_states=prepared.layout.n,
        build=build_info,
        flag_bits=FLAGS,
        families=FAMILIES,
        rhs=[],
        trajectories=[],
    )
    path = base.RESULTS / (name + "_generic_prepared.json")

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    payload = prepared.to_payload()
    report["preparation_s"], report["coefficient_instruction_count"], report["shared_DAG_nodes"] = (
        native.preparation_info(payload)
    )
    rng = np.random.default_rng(331)
    a = rng.normal(size=(prepared.layout.n, prepared.layout.n)) + 1j * rng.normal(
        size=(prepared.layout.n, prepared.layout.n)
    )
    a = a @ a.conj().T
    a /= np.trace(a)
    y = prepared.layout.pack(a)
    modes = [0, *range(16, 80)]
    for mode in modes:
        error = 0.0
        for t in np.array([125, 375, 625, 875]) * 1e-9:
            test = copy.deepcopy(payload)
            for g in gates:
                idx = test["parameter_graph"]["slot_names"].index(g)
                gate = base.driver.switch_values(float(t))[0 if g.endswith("1") else 1]
                test["parameter_graph"]["base_values"][idx]["re"] = (
                    gate if g.startswith("PA") else 1 - gate
                )
            actual = np.asarray(native.rhs_check(test, y, float(t), mode))
            ref = np.asarray(native.rhs_check(test, y, float(t), 0))
            error = max(error, float(abs(actual - ref).max() / max(abs(ref).max(), 1.0)))
        assert error < 1e-10, (name, mode, error)
        samples = native.rhs_benchmark(payload, y, mode, 1000)
        report["rhs"].append(
            dict(
                mode=mode,
                method=describe(mode),
                normalized_error=error,
                median_s=float(np.median(samples)),
                samples_s=samples,
            )
        )
        save()
        base.emit(f"{name}: generic factorial RHS {modes.index(mode) + 1}/{len(modes)}")
    # All singles, full combination, control, and three fastest per family.
    finalists = {0, 16, 17, 18, 20, 24, 31}
    for kernel in range(4):
        rows = [r for r in report["rhs"] if r["mode"] >= 16 and (r["mode"] - 16) // 16 == kernel]
        finalists.update(r["mode"] for r in sorted(rows, key=lambda r: r["median_s"])[:3])
    report["trajectory_finalists"] = sorted(finalists)
    y0 = prepared.layout.pack(rho0)
    duration = 10e-6
    reference = prepared.layout.unpack(
        np.asarray(native.trajectory(payload, y0, 0, duration, 1e-13, 1e-11)[0])
    )
    samples = {m: [] for m in finalists}
    last = {}
    for m in sorted(finalists):
        native.trajectory(payload, y0, m, duration, 1e-11, 1e-9)
    for repeat in range(5):
        order = sorted(finalists)
        rng.shuffle(order)
        for m in order:
            base.emit(f"{name}: generic trajectory repeat {repeat + 1}/5, {describe(m)}")
            run = native.trajectory(payload, y0, m, duration, 1e-11, 1e-9)
            samples[m].append(run[1])
            actual = prepared.layout.unpack(np.asarray(run[0]))
            accuracy = base.accuracy(actual, reference)
            assert accuracy["passed"], (name, m, accuracy)
            last[m] = dict(
                mode=m,
                method=describe(m),
                duration_s=duration,
                accuracy=accuracy,
                trace_error=float(abs(np.trace(actual) - 1)),
                rhs_calls=run[2],
                accepted=run[3],
                rejected=run[4],
            )
            report["trajectories"] = [
                dict(last[k], median_s=float(np.median(samples[k])), samples_s=samples[k])
                for k in sorted(last)
            ]
            save()
    base.emit(name + ": generic prepared factorial complete")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        first = base.model("r0")
        base.emit("Compiling one reusable generic benchmark extension")
        native, info = build(first[0], first[2])
        for name in ["r0", "rc", "r2", "q1"]:
            data = first if name == "r0" else base.model(name)
            run(native, name, *data, info)
