"""Matched before/after production D+S+U with prepared H assembly and scalar DAGs.

Both wrappers compile in the current environment. The baseline uses the frozen plan
and evaluator sources; the RHS/solver are shared and include pre-existing work.
"""

import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys

import bench_rust_specialized_rhs as base
import numpy as np
from threadpoolctl import threadpool_limits

BASELINE_REF = "4eea25f8a5ac6818501372f977eabe18fdd9495c"

WRAPPER = r"""
struct BenchRhs {
    plan:PreparedLindbladPlan, workspace:RhsWorkspace,
    gates:Vec<usize>, bases:Vec<Complex64>,
}
impl BenchRhs {
    fn new(plan:PreparedLindbladPlan,_mode:usize)->Self {
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
"""


def build(prepared, gates, before):
    tag = "before" if before else "after"
    module = "_rhs_integrated_" + tag
    folder = base.CACHE / ("integrated_" + tag)
    base.generate(prepared, gates, folder, module)
    original = (folder / "lib.rs").read_text(encoding="utf-8")
    prefix = original[: original.index("// Benchmark-only wrapper")]
    if before:
        for filename in ["eval.rs", "plan.rs"]:
            path = base.REPO / "rust/src/lindblad" / filename
            snapshot = folder / filename
            data = subprocess.check_output(
                ["git", "show", BASELINE_REF + ":rust/src/lindblad/" + filename], cwd=base.REPO
            )
            snapshot.write_bytes(data)
            prefix = prefix.replace(json.dumps(str(path)), json.dumps(str(snapshot)))
    helper = (base.HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")
    imports = helper[: helper.index("struct BenchRhs")]
    tail = helper[helper.index("#[pyfunction]") :]
    tail = tail.replace("N*N", "rhs.dim()").replace("GATE_SLOTS", "&rhs.gates")
    tail = tail.replace("=&rhs.gates.iter()", "=rhs.gates.iter()")
    source = (
        prefix
        + imports
        + WRAPPER
        + tail
        + f"""
#[pymodule]
fn {module}(m:&Bound<'_,PyModule>)->PyResult<()> {{
    m.add_function(wrap_pyfunction!(rhs_check,m)?)?;
    m.add_function(wrap_pyfunction!(rhs_benchmark,m)?)?;
    m.add_function(wrap_pyfunction!(trajectory,m)?)?;Ok(())
}}
"""
    )
    source = "\n".join(
        line
        for line in source.splitlines()
        if not line.startswith("const N:usize=") and not line.startswith("const GATE_SLOTS:")
    )
    (folder / "lib.rs").write_text(source, encoding="utf-8")
    env = os.environ.copy()
    env.update(PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(base.CACHE / "target"))
    with (folder / "build.log").open("w", encoding="utf-8") as log:
        done = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--offline",
                "--manifest-path",
                str(folder / "Cargo.toml"),
            ],
            env=env,
            stdout=log,
            stderr=log,
        )
    if done.returncode:
        raise RuntimeError((folder / "build.log").read_text(encoding="utf-8")[-5000:])
    target = folder / (module + ".pyd")
    target.write_bytes((base.CACHE / "target/release" / (module + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    hashes = {"wrapper": hashlib.sha256(source.encode()).hexdigest()}
    for filename in ["eval.rs", "plan.rs", "rhs.rs"]:
        path = (
            folder / filename
            if before and filename != "rhs.rs"
            else base.REPO / "rust/src/lindblad" / filename
        )
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return native, hashes


def main():
    report = dict(
        python=sys.version, platform=platform.platform(), baseline_ref=BASELINE_REF, systems=[]
    )
    path = base.RESULTS / "prepared_hamiltonian_integration.json"

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with threadpool_limits(limits=1):
        first = base.model("r0")
        base.emit("Compiling matched production before/after wrappers")
        old, report["before_sources"] = build(first[0], first[2], True)
        new, report["after_sources"] = build(first[0], first[2], False)
        for name in ["r0", "rc", "r2", "q1"]:
            prepared, rho0, gates = first if name == "r0" else base.model(name)
            payload = prepared.to_payload()
            y0 = prepared.layout.pack(rho0)
            rng = np.random.default_rng(771)
            n = prepared.layout.n
            a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            a = a @ a.conj().T
            random_y = prepared.layout.pack(a / np.trace(a))
            error = 0.0
            for scale, detuning in [(0.0, -3.7e6), (0.3, 0.0), (1.0, 0.0), (2.0, 5.1e6)]:
                for t in np.array([125, 375, 625, 875]) * 1e-9:
                    p = copy.deepcopy(payload)
                    graph = p["parameter_graph"]
                    for idx, slot in enumerate(graph["slot_names"][: len(graph["base_values"])]):
                        value = graph["base_values"][idx]
                        if slot.startswith("Ωt"):
                            value["re"] *= scale
                        if slot.startswith("δ"):
                            value["re"] = 2 * np.pi * detuning
                        if slot in gates:
                            gate = base.driver.switch_values(float(t))[
                                0 if slot.endswith("1") else 1
                            ]
                            value["re"] = gate if slot.startswith("PA") else 1 - gate
                    for y in [y0, random_y]:
                        expected = np.asarray(old.rhs_check(p, y, float(t), 0))
                        actual = np.asarray(new.rhs_check(p, y, float(t), 0))
                        error = max(
                            error,
                            float(abs(actual - expected).max() / max(abs(expected).max(), 1.0)),
                        )
            assert error < 1e-10, (name, error)
            row = dict(system=name, n_states=n, normalized_rhs_error=error, trajectories=[])
            report["systems"].append(row)
            duration = 10e-6
            reference = prepared.layout.unpack(
                np.asarray(old.trajectory(payload, y0, 0, duration, 1e-13, 1e-11)[0])
            )
            for native in [old, new]:
                native.trajectory(payload, y0, 0, duration, 1e-11, 1e-9)
            samples = {"before": [], "after": []}
            results = {}
            for repeat in range(5):
                base.emit(f"{name}: integrated before/after trajectory repeat {repeat + 1}/5")
                order = [("before", old), ("after", new)]
                if repeat % 2:
                    order.reverse()
                for tag, native in order:
                    run = native.trajectory(payload, y0, 0, duration, 1e-11, 1e-9)
                    accuracy = base.accuracy(prepared.layout.unpack(np.asarray(run[0])), reference)
                    assert accuracy["passed"], (name, tag, accuracy)
                    samples[tag].append(run[1])
                    results[tag] = dict(
                        method=tag,
                        duration_s=duration,
                        accuracy=accuracy,
                        rhs_calls=run[2],
                        accepted=run[3],
                        rejected=run[4],
                    )
                row["trajectories"] = [
                    dict(
                        results[tag],
                        median_s=float(np.median(samples[tag])),
                        samples_s=samples[tag],
                    )
                    for tag in ["before", "after"]
                ]
                row["reduction_percent"] = 100 * (
                    1 - np.median(samples["after"]) / np.median(samples["before"])
                )
                save()
            base.emit(name + ": integrated production comparison complete")
    save()


if __name__ == "__main__":
    main()
