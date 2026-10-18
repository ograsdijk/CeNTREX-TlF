"""Generate -i[H,rho] and D(rho) independently; no generic commutator in candidate.

Rust only. Separate complex upper-triangle CSE follows the expanded Julia-style
generation structure; solver and packed real error control are matched to Rust.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import bench_rust_specialized_rhs as base
import bench_rust_split_rhs as split
import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.ir import lower_expanded_sparse_rhs


def rust(expr):
    """Small exact emitter for the algebraic complex CSE expressions here."""
    if expr == sp.I:
        return "Complex64::new(0.,1.)"
    if isinstance(expr, sp.Symbol):
        prefix = str(expr)[0]
        index = int(str(expr)[1:])
        return {"c": "c", "r": "r", "z": "tmp"}[prefix] + f"[{index}]"
    if isinstance(expr, sp.Number):
        return repr(float(expr))
    if isinstance(expr, sp.Add):
        return "(" + "+".join(rust(arg) for arg in expr.args) + ")"
    if isinstance(expr, sp.Mul):
        return "(" + "*".join(rust(arg) for arg in expr.args) + ")"
    if isinstance(expr, sp.Pow) and expr.exp.is_Integer:
        return f"({rust(expr.base)}).powi({int(expr.exp)})"
    if expr.func == sp.conjugate:
        return f"({rust(expr.args[0])}).conj()"
    raise ValueError(f"Unsupported generated algebra: {expr!r}")


def hamiltonian_rhs(prepared):
    n = prepared.layout.n
    # Rebuild an H-only contribution plan. No dissipator terms enter this CSE.
    ep = lower_expanded_sparse_rhs(prepared.hamiltonian_plan, np.zeros(n), [[] for _ in range(n)])
    upper = [(i, j) for i in range(n) for j in range(i, n)]
    coefficients = sp.symbols(f"c0:{len(prepared.hamiltonian_plan['coefficients'])}")
    density = [
        sp.Symbol(f"r{k}", real=True) if i == j else sp.Symbol(f"r{k}")
        for k, (i, j) in enumerate(upper)
    ]
    equations = []
    for oi in range(len(upper)):
        terms = []
        for ti in range(ep["output_ptrs"][oi], ep["output_ptrs"][oi + 1]):
            ci = ep["coeff_indices"][ti]
            coefficient = sp.S.One if ci < 0 else coefficients[ci]
            if ep["coeff_conj"][ti]:
                coefficient = sp.conjugate(coefficient)
            value = density[ep["input_indices"][ti]]
            if ep["input_conj"][ti]:
                value = sp.conjugate(value)
            factor = sp.Float(ep["factors_re"][ti]) + sp.I * sp.Float(ep["factors_im"][ti])
            terms.append(factor * coefficient * value)
        equations.append(sp.Add(*terms))
    base.emit(f"{n}-state Hamiltonian-only CSE: {len(equations)} upper-triangle equations")
    # This is the same optimization setting used by the older Julia expanded
    # emitter, applied to H alone and only the independently evolved triangle.
    replacements, reduced = sp.cse(
        equations, symbols=sp.numbered_symbols("z"), optimizations="basic"
    )
    nt = len(replacements)
    source = [f"const NH_TEMP:usize={nt}; const NUP:usize={len(upper)};"]
    calls = []
    for begin in range(0, nt, 128):
        lines = [
            f"#[inline(never)] fn ham_temps_{begin}(c:&[Complex64],r:&[Complex64;NUP],tmp:&mut[Complex64;NH_TEMP]){{"
        ]
        lines += [
            f"tmp[{index}]=({rust(expr)}).into();"
            for index, (_symbol, expr) in enumerate(replacements[begin : begin + 128], begin)
        ]
        source.append("\n".join([*lines, "}"]))
        calls.append(f"ham_temps_{begin}(c,r,tmp);")
    for begin in range(0, len(upper), 32):
        lines = [
            f"#[inline(never)] fn ham_out_{begin}(c:&[Complex64],r:&[Complex64;NUP],tmp:&[Complex64;NH_TEMP],out:&mut[f64;N*N]){{"
        ]
        for oi in range(begin, min(begin + 32, len(upper))):
            i, j = upper[oi]
            lines.append(f"let value:Complex64=({rust(reduced[oi])}).into();")
            if i == j:
                lines.append(f"out[{i}]=value.re;")
            else:
                k = prepared.layout.upper_real_index(i, j)
                lines.append(f"out[{k}]=value.re;out[{k + 1}]=value.im;")
        source.append("\n".join([*lines, "}"]))
        calls.append(f"ham_out_{begin}(c,r,tmp,out);")
    # This packing conversion contains no Hamiltonian matrix multiplication or
    # sparse-plan traversal. All subsequent derivative arithmetic is generated.
    source.append(
        """
fn generated_hamiltonian_rhs(c:&[Complex64],y:&[f64],out:&mut[f64],rho:&mut[Complex64],tmp:&mut[Complex64]){
    let r:&mut[Complex64;NUP]=(&mut rho[..NUP]).try_into().unwrap();
    let tmp:&mut[Complex64;NH_TEMP]=tmp.try_into().unwrap();
    let out:&mut[f64;N*N]=out.try_into().unwrap();
    let mut u=0;let mut p=N;
    for i in 0..N {for j in i..N {
        r[u]=if i==j {Complex64::new(y[i],0.)} else {let v=Complex64::new(y[p],y[p+1]);p+=2;v};u+=1;
    }}
"""
        + "".join(calls)
        + "}"
    )
    return "\n".join(source), dict(
        hamiltonian_cse_temporaries=nt,
        upper_equations=len(upper),
        cse_optimizations="basic",
        operations_before=int(sum(sp.count_ops(e) for e in equations)),
        operations_after=int(
            sum(sp.count_ops(e) for _, e in replacements) + sum(sp.count_ops(e) for e in reduced)
        ),
    )


def run(name):
    base.emit("Full separately generated Hamiltonian RHS + dissipator: " + name)
    tick = time.perf_counter()
    prepared, rho0, gates = base.model(name)
    model_s = time.perf_counter() - tick
    module = "_rhs_separate_" + name
    folder = base.CACHE / ("separate_" + name)
    base.generate(prepared, gates, folder, module)
    tick = time.perf_counter()
    ham, info = hamiltonian_rhs(prepared)
    parts, diss_info = split.generators(prepared)
    # The separately CSE-generated dissipator is deliberately outside the
    # Hamiltonian CSE and has its own function and temporary namespace.
    dissipator = parts[parts.index("fn generated_dissipator(") :]
    info["dissipator_cse_temporaries"] = diss_info["dissipator_cse_temporaries"]
    info["dissipator_active_packed_entries"] = diss_info["dissipator_active_packed_entries"]
    source = (folder / "lib.rs").read_text(encoding="utf-8")
    a, b = source.index("fn generated_rows("), source.index("#[pymodule]")
    stubs = "fn generated_rows(_: &[Complex64],_: &[f64],_: &mut[f64],_: &[Complex64],_: &mut[Complex64]){unreachable!()}\nfn generated_packed(_: &[Complex64],_: &[f64],_: &mut[f64]){unreachable!()}\n"
    source = source[:a] + stubs + ham + "\n" + dissipator + "\n" + source[b:]
    source = source.replace("struct BenchRhs {", "struct BenchRhs {\n    ham_tmp:Vec<Complex64>,")
    source = source.replace(
        "Self {workspace:", "Self {ham_tmp:vec![Complex64::ZERO;NH_TEMP],workspace:"
    )
    source = source.replace("if self.mode == 4 {", "if self.mode >= 4 {")
    source = source.replace(
        "        if self.mode == 2 {",
        """        if self.mode == 8 {
            generated_hamiltonian_rhs(&self.coeffs,y,out,&mut self.rho,&mut self.ham_tmp);
            generated_dissipator(y,out);
            return Ok(());
        }
        if self.mode == 2 {""",
    )
    assert "benchmark_split_rhs" not in source
    (folder / "lib.rs").write_text(source, encoding="utf-8")
    generation_s = time.perf_counter() - tick
    base.emit(name + ": compiling full separately generated RHS", generation_s=generation_s, **info)
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["CARGO_TARGET_DIR"] = str(base.CACHE / "target")
    tick = time.perf_counter()
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
    compile_s = time.perf_counter() - tick
    target = folder / (module + ".pyd")
    target.write_bytes((base.CACHE / "target/release" / (module + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    payload = prepared.to_payload()
    n = prepared.layout.n
    rng = np.random.default_rng(472)
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
            actual = np.asarray(native.rhs_check(test, state, float(t), 8))
            error = max(error, float(abs(actual - ref).max() / max(abs(ref).max(), 1.0)))
    if error > 1e-10:
        raise RuntimeError(f"Separately generated H RHS correctness failed: {error}")
    report = dict(
        system=name,
        n_states=n,
        model_prep_s=model_s,
        generation_s=generation_s,
        compile_s=compile_s,
        generators=info,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_calls=[
            "direct_coefficients",
            "generated_hamiltonian_rhs",
            "generated_dissipator",
        ],
        candidate_uses_generic_commutator=False,
        normalized_rhs_error=error,
        rhs=[],
        trajectories=[],
    )

    def save():
        (base.RESULTS / (name + "_separate.json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    for mode, label in [(0, "D+S+U"), (8, "separately_generated_H_rhs_and_D_rhs")]:
        samples = native.rhs_benchmark(payload, y, mode, 1000)
        report["rhs"].append(
            dict(method=label, median_s=float(np.median(samples)), samples_s=samples)
        )
    save()
    for duration in [1e-6, 10e-6]:
        if duration > 1e-6 and report["rhs"][1]["median_s"] > 1.5 * report["rhs"][0]["median_s"]:
            report["long_trajectory_screened_out"] = True
            save()
            continue
        base.emit(f"{name}: full separate RHS trajectory test {duration * 1e6:g} us")
        y0 = prepared.layout.pack(rho0)
        reference = np.asarray(native.trajectory(payload, y0, 0, duration, 1e-13, 1e-11)[0])
        ref = prepared.layout.unpack(reference)
        samples = {0: [], 8: []}
        runs = {}
        for mode in [0, 8]:
            native.trajectory(payload, y0, mode, duration, 1e-11, 1e-9)
        for k in range(3):
            for mode in [0, 8] if k % 2 == 0 else [8, 0]:
                runs[mode] = native.trajectory(payload, y0, mode, duration, 1e-11, 1e-9)
                samples[mode].append(runs[mode][1])
        for mode, label in [(0, "D+S+U"), (8, "separately_generated_H_rhs_and_D_rhs")]:
            actual = prepared.layout.unpack(np.asarray(runs[mode][0]))
            report["trajectories"].append(
                dict(
                    method=label,
                    duration_s=duration,
                    median_s=float(np.median(samples[mode])),
                    samples_s=samples[mode],
                    rhs_calls=runs[mode][2],
                    accepted=runs[mode][3],
                    rejected=runs[mode][4],
                    accuracy=base.accuracy(actual, ref),
                    trace_error=float(abs(np.trace(actual) - 1)),
                )
            )
        save()
    base.emit(name + ": full separately generated RHS complete", rhs=report["rhs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=["rc", "r2", "r0", "q1"], default=["r0", "rc", "r2", "q1"]
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for name in args.cases:
            run(name)
