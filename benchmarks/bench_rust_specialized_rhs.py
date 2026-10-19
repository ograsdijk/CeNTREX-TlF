"""Rust-only generated RHS experiment for RC and smaller optical systems.

Compiles isolated benchmark extensions; does not replace the installed library.
Reuses the actual library Rust RHS and Tsit5 source for native matched baselines.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import bench_rc_propagation as driver
import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf import couplings, lindblad, transitions
from centrex_tlf.constants import ED_XtB

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = REPO / ".cache/benchmarks/rust_specialized_rhs_v1"
RESULTS = HERE / "rust_specialized_rhs_results"
NAMES = [
    "D+S+U",
    "expanded_sparse",
    "generated_rows",
    "generated_packed",
    "generated_packed_compiled_coeffs",
]


def emit(stage, **values):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "progress.json").write_text(
        json.dumps(dict(stage=stage, **values), indent=2), encoding="utf-8"
    )
    print(stage, flush=True)


def f64(x):
    return repr(float(x))


def c64(x):
    v = complex(x)
    return f"Complex64::new({f64(v.real)},{f64(v.imag)})"


def model(name):
    if name == "rc":
        case = driver.build_case(driver.CONFIGS["nominal"])
        system, params = case.system, case.params_dynamic
    else:
        tr = dict(
            r2=transitions.R2_F1_7o2_F4, r0=transitions.R0_F1_3o2_F2, q1=transitions.Q1_F1_3o2_F2
        )[name]
        selectors = couplings.generate_transition_selectors(
            [tr], [[couplings.polarization_X]], phase_modulations=[True]
        )
        system = lindblad.generate_OBE_system_transitions(
            [tr],
            selectors,
            qn_compact=True,
            E=np.array([0.0, 0.0, 200.0]),
            B=np.array([0.0, 0.0, 0.01]),
            retain_opposite_parity_levels=True,
            normalize_pol=True,
            Jmax_X=5,
            Jmax_B=5,
        )
        params = lindblad.generate_lindblad_parameters(selectors)
    base, compound = dict(params.base_parameters), dict(params.compound_parameters)
    base.update(Ωt0=driver.GAMMA, β0=3.8, ωphase0=2 * np.pi * 1.6e6)
    gates = []
    for key in ["PA1", "PY1", "PA2", "PY2"]:
        if key in compound:
            compound.pop(key)
            base[key] = 1.0 if key.startswith("PA") else 0.0
            gates.append(key)
    prepared = lindblad.prepare_lindblad_problem(
        system,
        lindblad.LindbladParameters(base, compound),
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    n = prepared.layout.n
    if name == "rc":
        rho0 = case.rho0("X_J2_F1_5o2_F2_mF0")
    else:
        rho0 = np.zeros((n, n), complex)
        # Molecular label selection, independent of energy ordering.
        indices = [i for i, q in enumerate(system.QN) if q.largest.electronic_state.name == "X"]
        rho0[indices, indices] = 1 / len(indices)
    return prepared, rho0, gates


def expression_source(expression):
    stack = []
    for ins in expression["instructions"]:
        op = ins["op"]
        if op == 1:
            stack.append(c64(complex(ins.get("re", 0), ins.get("im", 0))))
        elif op == 2:
            stack.append(f"s{ins['index']}")
        elif op == 4:
            stack.append("Complex64::new(t,0.)")
        elif op in (5, 6, 7, 8, 9):
            b, a = stack.pop(), stack.pop()
            stack.append(
                f"({a}).powc({b})"
                if op == 9
                else f"({a}{ {5: '+', 6: '-', 7: '*', 8: '/'}[op] }{b})"
            )
        elif op == 10:
            stack.append(f"(-{stack.pop()})")
        elif op == 11:
            stack.append(f"({stack.pop()}).conj()")
        elif op == 12:
            a = stack.pop()
            fn = ins["function"]
            stack.append(
                f"({a})."
                + {
                    1: "sin()",
                    2: "cos()",
                    3: "tan()",
                    4: "exp()",
                    5: "norm().into()",
                    6: "re.into()",
                    7: "im.into()",
                }[fn]
            )
        elif op == 13 and ins["function"] == 3:
            w, b, t = stack.pop(), stack.pop(), stack.pop()
            stack.append(f"Complex64::from_polar(1.,({b}).re*((({w}).re)*({t}).re).sin())")
        else:
            raise ValueError(f"Unsupported benchmark coefficient instruction: {ins}")
    assert len(stack) == 1
    return stack[0]


def generate(prepared, gates, folder, module):
    n = prepared.layout.n
    hp = prepared.hamiltonian_plan
    ep = prepared.expanded_rhs_plan
    upper = [(i, j) for i in range(n) for j in range(i, n)]

    def packed_ref(i, j):
        if i == j:
            return f"Complex64::new(y[{i}],0.)"
        r = prepared.layout.upper_real_index(min(i, j), max(i, j))
        return f"Complex64::new(y[{r}],{'-' if i > j else ''}y[{r + 1}])"

    prefix = (
        f"#![allow(dead_code,unused_variables,unused_parens,non_snake_case)]\nconst N:usize={n};\n"
    )
    prefix += "mod constants { pub const ED_XTB:f64=" + f64(ED_XtB) + "; }\n"
    for mod in ["ode"]:
        prefix += f"#[path={json.dumps(str(REPO / 'rust/src' / mod / 'mod.rs'))}] mod {mod};\n"
    prefix += "mod lindblad {\n"
    for mod in ["eval", "layout", "blas", "cpu_kernel", "plan", "rhs"]:
        prefix += f"#[path={json.dumps(str(REPO / 'rust/src/lindblad' / (mod + '.rs')))}] pub mod {mod};\n"
    prefix += "}\n"
    graph = prepared.parameter_graph
    gate_ids = [graph["slot_names"].index(g) for g in gates]
    prefix += f"const GATE_SLOTS:&[usize]=&{gate_ids!r};\n"
    source = [prefix, (HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")]
    # Compile the runtime coefficient graph without specializing scan values.
    lines = [
        "fn direct_coefficients(t:f64,bases:&[Complex64],out:&mut Vec<Complex64>){",
        "out.clear();",
    ]
    for i in range(graph["n_base"]):
        lines.append(f"let s{i}=bases[{i}];")
    for comp in graph["compounds"]:
        lines.append(f"let s{comp['slot']}={expression_source(comp['expression'])};")
    for coeff in hp["coefficients"]:
        lines.append("out.push(" + expression_source(coeff["expression"]) + ");")
    lines.append("}")
    source.append("\n".join(lines))
    # System-specific sparse row products, retaining contiguous vector loops.
    entries = {}
    static = np.asarray(hp["static_matrix"])
    for i, j in upper:
        if static[i, j] != 0:
            entries.setdefault((i, j), []).append(c64(static[i, j]))
    for ci, coeff in enumerate(hp["coefficients"]):
        for term in coeff["basis_terms"]:
            entries.setdefault((term["i"], term["j"]), []).append(
                f"c[{ci}]*{c64(complex(term['re'], term['im']))}"
            )
    rows = [
        "fn generated_rows(c:&[Complex64],y:&[f64],out:&mut[f64],rho:&[Complex64],scratch:&mut[Complex64]){",
        "scratch.fill(Complex64::ZERO);",
    ]
    for ei, ((i, k), terms) in enumerate(entries.items()):
        if i == k:
            continue
        rows.append(f"let h{ei}=" + "+".join(terms) + ";")
        rows.append(
            f"for j in 0..N {{scratch[{i * n}+j]+=h{ei}*rho[{k * n}+j];scratch[{k * n}+j]+=h{ei}.conj()*rho[{i * n}+j];}}"
        )
    cursor = n
    for i, j in upper:
        if i == j:
            rows.append(f"out[{i}]=2.*scratch[{i * n + i}].im;")
        else:
            hd_i = "+".join(entries.get((i, i), ["Complex64::ZERO"]))
            hd_j = "+".join(entries.get((j, j), ["Complex64::ZERO"]))
            rows.append(
                f"let v=Complex64::new(0.,-1.)*(scratch[{i * n + j}]-scratch[{j * n + i}].conj()+(({hd_i})-({hd_j}))*rho[{i * n + j}]);out[{cursor}]=v.re;out[{cursor + 1}]=v.im;"
            )
            cursor += 2
    # Only nonzero damping / transfer operations are emitted.
    for i, j in upper:
        rate = 0.5 * (prepared.source_decay_rates[i] + prepared.source_decay_rates[j])
        if rate:
            if i == j:
                rows.append(f"out[{i}]-={f64(rate)}*y[{i}];")
            else:
                k = prepared.layout.upper_real_index(i, j)
                rows.append(f"out[{k}]-={f64(rate)}*y[{k}];out[{k + 1}]-={f64(rate)}*y[{k + 1}];")
    for target, transfers in enumerate(prepared.incoming_transfers_by_target):
        for tr in transfers:
            rows.append(f"out[{target}]+={f64(tr['rate'])}*y[{tr['source']}];")
    rows.append("}")
    source.append("\n".join(rows))
    # Tile generated packed equations to keep compiler/code-size costs bounded.
    calls = []
    for begin in range(0, len(upper), 32):
        lines = [f"#[inline(never)] fn packed_{begin}(c:&[Complex64],y:&[f64],out:&mut[f64]){{"]
        for oi in range(begin, min(begin + 32, len(upper))):
            terms = []
            for ti in range(ep["output_ptrs"][oi], ep["output_ptrs"][oi + 1]):
                ci = ep["coeff_indices"][ti]
                coeff = (
                    "Complex64::new(1.,0.)"
                    if ci < 0
                    else f"c[{ci}]" + (".conj()" if ep["coeff_conj"][ti] else "")
                )
                factor = c64(complex(ep["factors_re"][ti], ep["factors_im"][ti]))
                a, b = upper[ep["input_indices"][ti]]
                value = packed_ref(a, b) + (".conj()" if ep["input_conj"][ti] else "")
                terms.append(f"{coeff}*{factor}*{value}")
            lines.append("let v=" + ("+".join(terms) if terms else "Complex64::ZERO") + ";")
            i, j = upper[oi]
            if i == j:
                lines.append(f"out[{i}]=v.re;")
            else:
                k = prepared.layout.upper_real_index(i, j)
                lines.append(f"out[{k}]=v.re;out[{k + 1}]=v.im;")
        lines.append("}")
        source.append("\n".join(lines))
        calls.append(f"packed_{begin}(c,y,out);")
    source.append(
        "fn generated_packed(c:&[Complex64],y:&[f64],out:&mut[f64]){" + "".join(calls) + "}"
    )
    source.append(f"""#[pymodule] fn {module}(m:&Bound<'_,PyModule>)->PyResult<()>{{
        m.add_function(wrap_pyfunction!(rhs_check,m)?)?;
        m.add_function(wrap_pyfunction!(rhs_benchmark,m)?)?;
        m.add_function(wrap_pyfunction!(trajectory,m)?)?;Ok(())}}""")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "lib.rs").write_text("\n".join(source), encoding="utf-8")
    manifest = f'''[package]
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
'''
    (folder / "Cargo.toml").write_text(manifest, encoding="utf-8")
    return hashlib.sha256((folder / "lib.rs").read_bytes()).hexdigest()


def accuracy(actual, reference):
    diff = abs(actual - reference)
    mask = abs(reference) >= 1e-6
    rel = float((diff[mask] / abs(reference[mask])).max()) if mask.any() else 0.0
    small = float(diff[~mask].max()) if (~mask).any() else 0.0
    return dict(max_relative=rel, max_small_absolute=small, passed=rel <= 0.01 and small <= 1e-6)


def run_case(name):
    emit("Building " + name)
    prepared, rho0, gates = model(name)
    module = "_rhs_bench_" + name
    folder = CACHE / name
    start = time.perf_counter()
    source_hash = generate(prepared, gates, folder, module)
    generation_s = time.perf_counter() - start
    emit("Compiling isolated Rust prototypes: " + name)
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["CARGO_TARGET_DIR"] = str(CACHE / "target")
    start = time.perf_counter()
    with (folder / "build.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
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
    if completed.returncode:
        raise RuntimeError((folder / "build.log").read_text(encoding="utf-8")[-5000:])
    compile_s = time.perf_counter() - start
    binary = CACHE / "target/release" / (module + ".dll")
    # Use a .pyd suffix so importlib loads the isolated PyO3 extension.
    target = folder / (module + ".pyd")
    target.write_bytes(binary.read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    payload = prepared.to_payload()
    n = prepared.layout.n
    rng = np.random.default_rng(1234)
    random = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    random = random @ random.conj().T
    random /= np.trace(random)
    random = prepared.layout.pack(random)
    maxerrors = [0.0] * 5
    emit("RHS correctness and native-loop timings: " + name)
    for _ti, t in enumerate(np.array([125, 375, 625, 875]) * 1e-9):
        test = copy.deepcopy(payload)
        for g in gates:
            idx = test["parameter_graph"]["slot_names"].index(g)
            state = driver.switch_values(float(t))[0 if g.endswith("1") else 1]
            test["parameter_graph"]["base_values"][idx]["re"] = (
                state if g.startswith("PA") else 1 - state
            )
        for y in [random, prepared.layout.pack(rho0)]:
            ref = np.asarray(native.rhs_check(test, y, float(t), 0))
            for mode in range(1, 5):
                actual = np.asarray(native.rhs_check(test, y, float(t), mode))
                error = float(abs(actual - ref).max() / max(abs(ref).max(), 1.0))
                maxerrors[mode] = max(maxerrors[mode], error)
                if error > 1e-10:
                    raise RuntimeError(f"{name}/{NAMES[mode]} failed RHS check: {error}")
    rhs = []
    for mode, label in enumerate(NAMES):
        samples = native.rhs_benchmark(payload, random, mode, 200)
        rhs.append(
            dict(
                method=label,
                median_s=float(np.median(samples)),
                samples_s=samples,
                normalized_RHS_error=maxerrors[mode],
            )
        )
    report = dict(
        system=name,
        n_states=n,
        generation_s=generation_s,
        compile_s=compile_s,
        source_sha256=source_hash,
        rhs=rhs,
        trajectories=[],
        fields=dict(E_V_cm=[0, 0, 200], B_G=[0, 0, 0.01]),
        qn_compact=name != "rc",
        EOM_Hz=1.6e6,
        beta=3.8,
        compiler="cargo release opt3 fat LTO; no fastmath",
        hardware=dict(
            platform=platform.platform(),
            processor=platform.processor(),
            logical_cpus=os.cpu_count(),
        ),
        baseline_rhs_source_sha256=hashlib.sha256(
            (REPO / "rust/src/lindblad/rhs.rs").read_bytes()
        ).hexdigest(),
        tsit5_source_sha256=hashlib.sha256(
            (REPO / "rust/src/ode/tsit5.rs").read_bytes()
        ).hexdigest(),
    )

    def save():
        return (RESULTS / (name + ".json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    save()
    y0 = prepared.layout.pack(rho0)
    for duration in [1e-6, 10e-6]:
        emit(f"{name}: matched native Tsit5 trajectories, {duration * 1e6:g} us")
        reference = np.asarray(native.trajectory(payload, y0, 0, duration, 1e-13, 1e-11)[0])
        for mode, label in enumerate(NAMES):
            if duration > 1e-6 and rhs[mode]["median_s"] > 1.5 * rhs[0]["median_s"]:
                report.setdefault("long_trajectory_screened_out", []).append(label)
                save()
                continue
            results = [
                native.trajectory(payload, y0, mode, duration, 1e-11, 1e-9) for _ in range(3)
            ]
            matrix = prepared.layout.unpack(np.asarray(results[-1][0]))
            refmatrix = prepared.layout.unpack(reference)
            report["trajectories"].append(
                dict(
                    method=label,
                    duration_s=duration,
                    median_s=float(np.median([x[1] for x in results])),
                    samples_s=[x[1] for x in results],
                    rhs_calls=results[-1][2],
                    accepted=results[-1][3],
                    rejected=results[-1][4],
                    accuracy=accuracy(matrix, refmatrix),
                    trace_error=float(abs(np.trace(matrix) - 1)),
                    min_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),
                )
            )
            save()
    emit(name + " complete", rhs=rhs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", default=["rc", "r2", "r0", "q1"], choices=["rc", "r2", "r0", "q1"]
    )
    args = parser.parse_args()
    for name in args.cases:
        run_case(name)
    emit("All Rust-only benchmarks complete")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
