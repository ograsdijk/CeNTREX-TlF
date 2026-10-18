"""Rust-only global real packed RHS CSE versus the current D+S+U kernel."""

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

import bench_rust_specialized_rhs as base
import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter
from threadpoolctl import threadpool_limits


class RealPrinter(C99CodePrinter):
    def _print_Integer(self, expr):
        return str(expr) + ".0"

    def _print_Pow(self, expr):
        a, b = expr.args
        return f"({self._print(a)}).powf({self._print(b)})"


def cse_source(prepared):
    n = prepared.layout.n
    ep = prepared.expanded_rhs_plan
    upper = [(i, j) for i in range(n) for j in range(i, n)]
    ys = sp.symbols(f"y0:{n * n}", real=True)
    cr = sp.symbols(f"a0:{len(prepared.hamiltonian_plan['coefficients'])}", real=True)
    ci = sp.symbols(f"b0:{len(cr)}", real=True)
    equations = [sp.S.Zero] * (n * n)
    for oi, (i, j) in enumerate(upper):
        real_terms, imag_terms = [], []
        for ti in range(ep["output_ptrs"][oi], ep["output_ptrs"][oi + 1]):
            k = ep["coeff_indices"][ti]
            fr, fi = sp.Float(ep["factors_re"][ti]), sp.Float(ep["factors_im"][ti])
            if k < 0:
                ar, ai = fr, fi
            else:
                sr = cr[k]
                si = -ci[k] if ep["coeff_conj"][ti] else ci[k]
                ar, ai = fr * sr - fi * si, fr * si + fi * sr
            a, b = upper[ep["input_indices"][ti]]
            if a == b:
                x, z = ys[a], sp.S.Zero
            else:
                idx = prepared.layout.upper_real_index(a, b)
                x, z = ys[idx], ys[idx + 1]
                if ep["input_conj"][ti]:
                    z = -z
            real_terms.append(ar * x - ai * z)
            imag_terms.append(ar * z + ai * x)
        real, imag = sp.Add(*real_terms), sp.Add(*imag_terms)
        if i == j:
            equations[i] = real
        else:
            idx = prepared.layout.upper_real_index(i, j)
            equations[idx] = real
            equations[idx + 1] = imag
    replacements, reduced = sp.cse(equations, symbols=sp.numbered_symbols("z"))
    length = len(replacements)

    def code(expr):
        text = RealPrinter().doprint(expr)
        return re.sub(
            r"\b([yabz])(\d+)\b",
            lambda m: dict(y=f"y[{m[2]}]", a=f"c[{m[2]}].re", b=f"c[{m[2]}].im", z=f"tmp[{m[2]}]")[
                m[1]
            ],
            text,
        )

    chunks = []
    calls = []
    for begin in range(0, length, 128):
        lines = [
            f"#[inline(never)] fn cse_temp_{begin}(c:&[Complex64],y:&[f64;N*N],tmp:&mut[f64;NTEMP]){{"
        ]
        lines += [
            f"tmp[{index}]={code(expr)};"
            for index, (_sym, expr) in enumerate(replacements[begin : begin + 128], begin)
        ]
        chunks.append("\n".join([*lines, "}"]))
        calls.append(f"cse_temp_{begin}(c,y,tmp);")
    for begin in range(0, n * n, 64):
        lines = [
            f"#[inline(never)] fn cse_out_{begin}(c:&[Complex64],y:&[f64;N*N],tmp:&[f64;NTEMP],out:&mut[f64;N*N]){{"
        ]
        lines += [
            f"out[{index}]={code(expr)};"
            for index, expr in enumerate(reduced[begin : begin + 64], begin)
        ]
        chunks.append("\n".join([*lines, "}"]))
        calls.append(f"cse_out_{begin}(c,y,tmp,out);")
    chunks.append(
        "fn generated_cse(c:&[Complex64],y:&[f64;N*N],out:&mut[f64;N*N],tmp:&mut[f64;NTEMP]){"
        + "".join(calls)
        + "}"
    )
    return (
        "\n".join(chunks),
        length,
        sum(sp.count_ops(e) for e in equations),
        sum(sp.count_ops(e) for _, e in replacements) + sum(sp.count_ops(e) for e in reduced),
    )


def run(name):
    base.emit("Global packed RHS CSE: building " + name)
    prepared, rho0, gates = base.model(name)
    module = "_rhs_cse_" + name
    folder = base.CACHE / ("cse_" + name)
    base.generate(prepared, gates, folder, module)
    start = time.perf_counter()
    generated, temps, before, after = cse_source(prepared)
    source = (folder / "lib.rs").read_text(encoding="utf-8")
    a = source.index("fn generated_rows(")
    b = source.index("#[pymodule]", a)
    stubs = "fn generated_rows(_: &[Complex64],_: &[f64],_: &mut[f64],_: &[Complex64],_: &mut[Complex64]){unreachable!()}\nfn generated_packed(_: &[Complex64],_: &[f64],_: &mut[f64]){unreachable!()}\n"
    source = source[:a] + stubs + generated + "\n" + source[b:]
    source = source.replace(
        "struct BenchRhs {",
        f"const NTEMP:usize={temps};\nstruct BenchRhs {{\n    cse_tmp:Vec<f64>,",
    )
    source = source.replace("Self {workspace:", "Self {cse_tmp:vec![0.;NTEMP],workspace:")
    source = source.replace("if self.mode == 4 {", "if self.mode >= 4 {")
    source = source.replace(
        "        if self.mode == 2 {",
        """        if self.mode == 5 {
            generated_cse(&self.coeffs,y.try_into().unwrap(),out.try_into().unwrap(),self.cse_tmp.as_mut_slice().try_into().unwrap());
            return Ok(());
        }
        if self.mode == 2 {""",
    )
    (folder / "lib.rs").write_text(source, encoding="utf-8")
    generation = time.perf_counter() - start
    base.emit(f"{name}: compiling global CSE ({temps} temporaries; operations {before} -> {after})")
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
        raise RuntimeError((folder / "build.log").read_text(encoding="utf-8")[-4000:])
    compile_s = time.perf_counter() - start
    target = folder / (module + ".pyd")
    target.write_bytes((base.CACHE / "target/release" / (module + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    payload = prepared.to_payload()
    n = prepared.layout.n
    rng = np.random.default_rng(191)
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
            actual = np.asarray(native.rhs_check(test, state, float(t), 5))
            error = max(error, float(abs(actual - ref).max() / max(abs(ref).max(), 1.0)))
    if error > 1e-10:
        raise RuntimeError(f"CSE failed correctness: {error}")
    report = dict(
        system=name,
        n_states=n,
        cse_temporaries=temps,
        operations_before=int(before),
        operations_after=int(after),
        generation_s=generation,
        compile_s=compile_s,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        normalized_rhs_error=error,
        rhs=[],
        trajectories=[],
    )

    def save():
        (base.RESULTS / (name + "_cse.json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    for mode, label in [(0, "D+S+U"), (5, "generated_packed_global_CSE")]:
        samples = native.rhs_benchmark(payload, y, mode, 200)
        report["rhs"].append(
            dict(method=label, median_s=float(np.median(samples)), samples_s=samples)
        )
    save()
    base.emit(name + ": CSE trajectory validation")
    y0 = prepared.layout.pack(rho0)
    for duration in [1e-6, 10e-6]:
        if duration > 1e-6 and report["rhs"][1]["median_s"] > 1.5 * report["rhs"][0]["median_s"]:
            report["long_CSE_screened_out"] = True
            save()
            continue
        reference = np.asarray(native.trajectory(payload, y0, 0, duration, 1e-13, 1e-11)[0])
        ref = prepared.layout.unpack(reference)
        for mode, label in [(0, "D+S+U"), (5, "generated_packed_global_CSE")]:
            runs = [native.trajectory(payload, y0, mode, duration, 1e-11, 1e-9) for _ in range(3)]
            report["trajectories"].append(
                dict(
                    method=label,
                    duration_s=duration,
                    median_s=float(np.median([x[1] for x in runs])),
                    rhs_calls=runs[-1][2],
                    accepted=runs[-1][3],
                    rejected=runs[-1][4],
                    accuracy=base.accuracy(prepared.layout.unpack(np.asarray(runs[-1][0])), ref),
                )
            )
            save()
    base.emit(name + ": global CSE complete", rhs=report["rhs"])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for name in ["rc", "r2", "r0", "q1"]:
            run(name)
