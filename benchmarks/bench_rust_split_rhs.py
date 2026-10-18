"""Separate generated Hamiltonian and dissipator functions: Rust-only benchmark."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import bench_rust_cse_rhs as cse
import bench_rust_specialized_rhs as base
import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

HELPER = r"""
// Benchmark-only adapter appended to an exact snapshot of the library RHS.
// Reuses the unmodified D+S+U commutator and layout helpers above.
pub fn benchmark_split_rhs(
    plan:&PreparedLindbladPlan,y:&[f64],out:&mut[f64],c:&[Complex64],
    w:&mut RhsWorkspace,
    hfill:fn(&[Complex64],&mut[Complex64]),
    dadd:fn(&[f64],&mut[f64]),
    mut profile:Option<&mut[f64;6]>,
)->Result<(),String>{
    let mut tick=profile.as_ref().map(|_|Instant::now());
    plan.layout.unpack_into(y,&mut w.rho)?;
    if let Some(p)=profile.as_deref_mut(){p[0]+=tick.take().unwrap().elapsed().as_secs_f64();tick=Some(Instant::now());}
    hfill(c,&mut w.hamiltonian_upper);
    if let Some(p)=profile.as_deref_mut(){p[1]+=tick.take().unwrap().elapsed().as_secs_f64();tick=Some(Instant::now());}
    fill_sparse_h_values(&plan.hamiltonian_sparse_pattern,&w.upper_layout,&w.hamiltonian_upper,&mut w.h_sparse_values);
    if let Some(p)=profile.as_deref_mut(){p[2]+=tick.take().unwrap().elapsed().as_secs_f64();tick=Some(Instant::now());}
    add_commutator_sparse_dense_upper_soa(&plan.hamiltonian_sparse_pattern,&w.h_sparse_values,
        &w.hamiltonian_upper,&w.rho,&mut w.scratch_re,&mut w.scratch_im,
        &w.upper_layout,&mut w.drho_upper,true);
    if let Some(p)=profile.as_deref_mut(){p[3]+=tick.take().unwrap().elapsed().as_secs_f64();tick=Some(Instant::now());}
    w.upper_layout.pack_packed_state(&w.drho_upper,out)?;
    if let Some(p)=profile.as_deref_mut(){p[4]+=tick.take().unwrap().elapsed().as_secs_f64();tick=Some(Instant::now());}
    dadd(y,out);
    if let Some(p)=profile.as_deref_mut(){p[5]+=tick.take().unwrap().elapsed().as_secs_f64();}
    Ok(())
}
"""


def emitted_code(expr):
    text = cse.RealPrinter().doprint(expr)
    return re.sub(
        r"\b([yabz])(\d+)\b",
        lambda m: dict(y=f"y[{m[2]}]", a=f"c[{m[2]}].re", b=f"c[{m[2]}].im", z=f"z{m[2]}")[m[1]],
        text,
    )


def generators(prepared):
    n = prepared.layout.n
    hp = prepared.hamiltonian_plan
    upper = [(i, j) for i in range(n) for j in range(i, n)]
    indices = {pair: k for k, pair in enumerate(upper)}
    count = len(hp["coefficients"])
    cr = sp.symbols(f"a0:{count}", real=True)
    ci = sp.symbols(f"b0:{count}", real=True)
    dynamic = {}
    for k, coeff in enumerate(hp["coefficients"]):
        for tr in coeff["basis_terms"]:
            index = indices[tr["i"], tr["j"]]
            r, im = sp.Float(tr["re"]), sp.Float(tr["im"])
            dr, di = dynamic.setdefault(index, [sp.S.Zero, sp.S.Zero])
            dynamic[index] = [dr + r * cr[k] - im * ci[k], di + r * ci[k] + im * cr[k]]
    positions = sorted(dynamic)
    expressions = [e for k in positions for e in dynamic[k]]
    temps, reduced = sp.cse(expressions, symbols=sp.numbered_symbols("z"))
    static = np.asarray(hp["static_matrix"])
    h = [
        "const H_STATIC:[Complex64;"
        + str(len(upper))
        + "]=["
        + ",".join(base.c64(static[i, j]) for i, j in upper)
        + "];",
        "fn generated_hamiltonian(c:&[Complex64],upper:&mut[Complex64]){",
        "upper.copy_from_slice(&H_STATIC);",
    ]
    h += [f"let {sym}={emitted_code(expr)};" for sym, expr in temps]
    for k, index in enumerate(positions):
        h.append(
            f"upper[{index}]+=Complex64::new({emitted_code(reduced[2 * k])},{emitted_code(reduced[2 * k + 1])});"
        )
    h.append("}")
    # Independent dissipator CSE. It only reads the original density y, never
    # the output of the Hamiltonian part, and adds its derivative to out.
    y = sp.symbols(f"y0:{n * n}", real=True)
    equations = {}
    for i, j in upper:
        rate = 0.5 * (prepared.source_decay_rates[i] + prepared.source_decay_rates[j])
        if rate:
            if i == j:
                equations[i] = -sp.Float(rate) * y[i]
            else:
                k = prepared.layout.upper_real_index(i, j)
                equations[k] = -sp.Float(rate) * y[k]
                equations[k + 1] = -sp.Float(rate) * y[k + 1]
    for target, transfers in enumerate(prepared.incoming_transfers_by_target):
        for tr in transfers:
            equations[target] = (
                equations.get(target, sp.S.Zero) + sp.Float(tr["rate"]) * y[tr["source"]]
            )
    dpositions = sorted(equations)
    dtemps, dreduced = sp.cse([equations[k] for k in dpositions], symbols=sp.numbered_symbols("z"))
    d = ["fn generated_dissipator(y:&[f64],out:&mut[f64]){"]
    d += [f"let {sym}={emitted_code(expr)};" for sym, expr in dtemps]
    d += [f"out[{k}]+={emitted_code(expr)};" for k, expr in zip(dpositions, dreduced, strict=True)]
    d.append("}")
    return "\n".join(h + d), dict(
        hamiltonian_cse_temporaries=len(temps),
        dissipator_cse_temporaries=len(dtemps),
        hamiltonian_dynamic_entries=len(positions),
        dissipator_active_packed_entries=len(dpositions),
    )


def run(name):
    base.emit("Separate generated H / dissipator: " + name)
    prepared, rho0, gates = base.model(name)
    module = "_rhs_split_" + name
    folder = base.CACHE / ("split_" + name)
    base.generate(prepared, gates, folder, module)
    start = time.perf_counter()
    parts, info = generators(prepared)
    generated_s = time.perf_counter() - start
    source = (folder / "lib.rs").read_text(encoding="utf-8")
    original_rhs = base.REPO / "rust/src/lindblad/rhs.rs"
    snapshot = folder / "rhs_split.rs"
    snapshot.write_text(original_rhs.read_text(encoding="utf-8") + "\n" + HELPER, encoding="utf-8")
    source = source.replace(json.dumps(str(original_rhs)), json.dumps(str(snapshot)))
    a = source.index("fn generated_rows(")
    b = source.index("#[pymodule]", a)
    stubs = "fn generated_rows(_: &[Complex64],_: &[f64],_: &mut[f64],_: &[Complex64],_: &mut[Complex64]){unreachable!()}\nfn generated_packed(_: &[Complex64],_: &[f64],_: &mut[f64]){unreachable!()}\n"
    source = source[:a] + stubs + parts + "\n" + source[b:]
    source = source.replace("if self.mode == 4 {", "if self.mode >= 4 {")
    source = source.replace(
        "        if self.mode == 2 {",
        """        if self.mode == 6 {
            return lindblad::rhs::benchmark_split_rhs(&self.plan,y,out,&self.coeffs,&mut self.workspace,
                generated_hamiltonian,generated_dissipator,None);
        }
        if self.mode == 2 {""",
    )
    (folder / "lib.rs").write_text(source, encoding="utf-8")
    base.emit(name + ": compiling separate generated functions")
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["CARGO_TARGET_DIR"] = str(base.CACHE / "target")
    start = time.perf_counter()
    with (folder / "build.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
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
    if result.returncode:
        raise RuntimeError((folder / "build.log").read_text(encoding="utf-8")[-5000:])
    compile_s = time.perf_counter() - start
    target = folder / (module + ".pyd")
    target.write_bytes((base.CACHE / "target/release" / (module + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    payload = prepared.to_payload()
    n = prepared.layout.n
    rng = np.random.default_rng(723)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a = a @ a.conj().T
    a /= np.trace(a)
    y = prepared.layout.pack(a)
    error = 0.0
    for t in np.array([125, 375, 625, 875]) * 1e-9:
        test = copy.deepcopy(payload)
        for g in gates:
            idx = test["parameter_graph"]["slot_names"].index(g)
            gate = base.driver.switch_values(float(t))[0 if g.endswith("1") else 1]
            test["parameter_graph"]["base_values"][idx]["re"] = (
                gate if g.startswith("PA") else 1 - gate
            )
        for state in [y, prepared.layout.pack(rho0)]:
            ref = np.asarray(native.rhs_check(test, state, float(t), 0))
            actual = np.asarray(native.rhs_check(test, state, float(t), 6))
            error = max(error, float(abs(actual - ref).max() / max(abs(ref).max(), 1.0)))
    if error > 1e-10:
        raise RuntimeError(f"Separate generation failed correctness: {error}")
    report = dict(
        system=name,
        n_states=n,
        generation_s=generated_s,
        compile_s=compile_s,
        generators=info,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        baseline_rhs_source_sha256=hashlib.sha256(original_rhs.read_bytes()).hexdigest(),
        normalized_rhs_error=error,
        rhs=[],
        trajectories=[],
    )

    def save():
        (base.RESULTS / (name + "_split.json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    for mode, label in [(0, "D+S+U"), (6, "generated_H_plus_DSU_plus_generated_dissipator")]:
        samples = native.rhs_benchmark(payload, y, mode, 200)
        report["rhs"].append(
            dict(method=label, median_s=float(np.median(samples)), samples_s=samples)
        )
    save()
    for duration in [1e-6, 10e-6]:
        base.emit(f"{name}: separate-function trajectory test {duration * 1e6:g} us")
        reference = np.asarray(
            native.trajectory(payload, prepared.layout.pack(rho0), 0, duration, 1e-13, 1e-11)[0]
        )
        ref = prepared.layout.unpack(reference)
        for mode, label in [(0, "D+S+U"), (6, "generated_H_plus_DSU_plus_generated_dissipator")]:
            runs = [
                native.trajectory(payload, prepared.layout.pack(rho0), mode, duration, 1e-11, 1e-9)
                for _ in range(3)
            ]
            actual = prepared.layout.unpack(np.asarray(runs[-1][0]))
            report["trajectories"].append(
                dict(
                    method=label,
                    duration_s=duration,
                    median_s=float(np.median([x[1] for x in runs])),
                    samples_s=[x[1] for x in runs],
                    rhs_calls=runs[-1][2],
                    accepted=runs[-1][3],
                    rejected=runs[-1][4],
                    accuracy=base.accuracy(actual, ref),
                    trace_error=float(abs(np.trace(actual) - 1)),
                )
            )
            save()
    base.emit(name + ": separate-function test complete", rhs=report["rhs"])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for name in ["rc", "r2", "r0", "q1"]:
            run(name)
