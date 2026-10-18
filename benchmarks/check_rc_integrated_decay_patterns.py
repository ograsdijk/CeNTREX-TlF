"""Balanced, paired tests of decay/pattern additions to integrated D+S+U.

Uses the real current parameter/H assembly/RHS/solver pipeline. Only sparse
pattern selection and the decay loop are patched in an isolated RHS snapshot.
An unpatched control detects snapshot/dispatch overhead. No installed code changes.
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
import confirm_prepared_hamiltonian_integration as integrated
import numpy as np
from threadpoolctl import threadpool_limits

LABELS = ["current", "nonzero_decay", "active_patterns", "both", "unpatched_control"]
ROUNDS = 12

HOOKS = r"""
struct BenchRcExtras { flags:usize, data:GenericPrepared, gates:Vec<usize> }
pub fn benchmark_configure_extras(plan:&PreparedLindbladPlan,w:&mut RhsWorkspace,
    flags:usize,gates:&[usize]) {
    w.bench_rc_extras=Some(BenchRcExtras {flags,data:GenericPrepared::new(plan,gates),gates:gates.to_vec()});
}
pub fn benchmark_refresh_extras(plan:&PreparedLindbladPlan,w:&mut RhsWorkspace) {
    let e=w.bench_rc_extras.as_mut().unwrap();
    if e.flags&2!=0 {
        e.data.refresh(plan,&e.gates);w.h_sparse_valid=false;
    }
}
pub fn benchmark_extra_counts(plan:&PreparedLindbladPlan,gates:&[usize])->Vec<usize> {
    let data=GenericPrepared::new(plan,gates);
    let mut result=vec![plan.hamiltonian_sparse_pattern.nnz,data.damping.len(),data.transfers.len()];
    result.extend(data.patterns.iter().map(|p|p.1.nnz));result
}
"""


def patched_rhs(original):
    source = original.replace(
        "pub struct RhsWorkspace {",
        "pub struct RhsWorkspace {\n    bench_rc_extras:Option<BenchRcExtras>,",
    )
    source = source.replace(
        "upper_layout: upper_layout.clone(),",
        "bench_rc_extras:None,\n            upper_layout: upper_layout.clone(),",
    )
    a = source.index("fn rhs_from_workspace_rho(")
    b = source.index("pub fn rhs_matrix_into_with_profile(", a)
    body = source[a:b]
    body = body.replace("&plan.hamiltonian_sparse_pattern", "sparse_pattern")
    marker = "let commutator_start = profile_timer(&profile);"
    assert body.count(marker) == 1
    body = body.replace(
        marker,
        marker
        + """
    let extras=workspace.bench_rc_extras.as_ref().unwrap();
    let sparse_pattern=if extras.flags&2!=0 {
        &extras.data.patterns[extras.data.active].1
    } else {&plan.hamiltonian_sparse_pattern};
""",
    )
    start = body.index(
        "| ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack => {",
        body.index("let dissipator_start"),
    )
    end = body.index("ExecutionMode::ExpandedSparse", start)
    decay = body[start:end]
    target = """add_structured_dissipator_upper_dense_rho(
                plan,
                &workspace.upper_layout,
                workspace.rho.as_slice(),
                workspace.drho_upper.as_mut_slice(),
            )"""
    assert decay.count(target) == 1
    replacement = (
        """if extras.flags&1!=0 {
                for &(idx,dense,rate) in &extras.data.damping {
                    workspace.drho_upper[idx]-=rate*workspace.rho[dense];
                }
                for &(dst,src,rate) in &extras.data.transfers {
                    workspace.drho_upper[workspace.upper_layout.index_unchecked(dst,dst)]
                        +=rate*workspace.rho[src*plan.layout.n+src];
                }
            } else {"""
        + target
        + "}"
    )
    body = body[:start] + decay.replace(target, replacement) + body[end:]
    return (
        source[:a]
        + body
        + source[b:]
        + "\n"
        + (base.HERE / "generic_prepared_rhs.rs").read_text(encoding="utf-8")
        + HOOKS
    )


def build(prepared, gates, patched):
    tag = "hooks" if patched else "control"
    module = "_rc_decay_patterns_" + tag
    folder = base.CACHE / ("integrated_decay_patterns_" + tag)
    base.generate(prepared, gates, folder, module)
    generated = (folder / "lib.rs").read_text(encoding="utf-8")
    prefix = generated[: generated.index("// Benchmark-only wrapper")]
    rhs_path = base.REPO / "rust/src/lindblad/rhs.rs"
    original = rhs_path.read_text(encoding="utf-8")
    if patched:
        snapshot_source = patched_rhs(original)
    else:
        snapshot_source = (
            original
            + """
pub fn benchmark_configure_extras(_: &PreparedLindbladPlan,_:&mut RhsWorkspace,_:usize,_:&[usize]){}
pub fn benchmark_refresh_extras(_: &PreparedLindbladPlan,_:&mut RhsWorkspace){}
pub fn benchmark_extra_counts(_: &PreparedLindbladPlan,_:&[usize])->Vec<usize>{Vec::new()}
"""
        )
    snapshot = folder / "rhs.rs"
    snapshot.write_text(snapshot_source, encoding="utf-8")
    prefix = prefix.replace(json.dumps(str(rhs_path)), json.dumps(str(snapshot)))
    native_source = (base.HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")
    imports = native_source[: native_source.index("struct BenchRhs")]
    wrapper = integrated.WRAPPER.replace(
        "Self{workspace:RhsWorkspace::new(&plan),plan,gates,bases}",
        "let mut workspace=RhsWorkspace::new(&plan);\n"
        "        lindblad::rhs::benchmark_configure_extras(&plan,&mut workspace,_mode,&gates);\n"
        "        Self{workspace,plan,gates,bases}",
    )
    assert "benchmark_configure_extras" in wrapper
    wrapper = wrapper.replace("let gates=", "let gates:Vec<usize>=")
    tail = native_source[native_source.index("#[pyfunction]") :]
    tail = tail.replace("N*N", "rhs.dim()").replace("GATE_SLOTS", "&rhs.gates")
    tail = tail.replace("=&rhs.gates.iter()", "=rhs.gates.iter()")
    tail = tail.replace(
        "let options=OdeOptions",
        "lindblad::rhs::benchmark_refresh_extras(&rhs.plan,&mut rhs.workspace);\n        let options=OdeOptions",
    )
    extra_function = """
#[pyfunction]
fn extra_counts(payload:&Bound<'_,PyAny>)->PyResult<Vec<usize>> {
    let rhs=BenchRhs::new(parse_plan_payload(payload)?,0);
    Ok(lindblad::rhs::benchmark_extra_counts(&rhs.plan,&rhs.gates))
}
"""
    source = (
        prefix
        + imports
        + wrapper
        + tail
        + extra_function
        + f"""
#[pymodule]
fn {module}(m:&Bound<'_,PyModule>)->PyResult<()> {{
    m.add_function(wrap_pyfunction!(rhs_check,m)?)?;
    m.add_function(wrap_pyfunction!(trajectory,m)?)?;
    m.add_function(wrap_pyfunction!(extra_counts,m)?)?;Ok(())
}}
"""
    )
    source = "\n".join(
        line
        for line in source.splitlines()
        if not line.startswith(("const N:usize=", "const GATE_SLOTS:"))
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
        raise RuntimeError((folder / "build.log").read_text(encoding="utf-8")[-6500:])
    target = folder / (module + ".pyd")
    target.write_bytes((base.CACHE / "target/release" / (module + ".dll")).read_bytes())
    spec = importlib.util.spec_from_file_location(module, target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    hashes = {
        "wrapper": hashlib.sha256(source.encode()).hexdigest(),
        "rhs_snapshot": hashlib.sha256(snapshot_source.encode()).hexdigest(),
    }
    for filename in ["plan.rs", "eval.rs", "rhs.rs"]:
        hashes[filename] = hashlib.sha256(
            (base.REPO / "rust/src/lindblad" / filename).read_bytes()
        ).hexdigest()
    return native, hashes


def statistics(row):
    times = np.asarray(row["times_by_round_s"])
    rng = np.random.default_rng(991)
    indices = rng.integers(0, len(times), size=(20000, len(times)))
    summary = []
    for mode in range(5):
        ratios = times[:, mode] / times[:, 0]
        med = float(np.median(ratios))
        boot = np.median(ratios[indices], axis=1)
        lo, hi = np.quantile(100 * (1 - boot), [0.025, 0.975])
        summary.append(
            dict(
                method=LABELS[mode],
                median_s=float(np.median(times[:, mode])),
                paired_median_reduction_percent=100 * (1 - med),
                paired_bootstrap_95_percent=[float(lo), float(hi)],
            )
        )
        control_ratios = times[:, mode] / times[:, 4]
        control_boot = np.median(control_ratios[indices], axis=1)
        summary[-1]["versus_unpatched_control"] = dict(
            paired_median_reduction_percent=100 * (1 - float(np.median(control_ratios))),
            paired_bootstrap_95_percent=np.quantile(
                100 * (1 - control_boot), [0.025, 0.975]
            ).tolist(),
        )
    # Negative log interaction means both together beat multiplicative independence.
    interaction = np.log(times[:, 3] * times[:, 0] / (times[:, 1] * times[:, 2]))
    boot = np.median(interaction[indices], axis=1)
    row["summary"] = summary
    row["log_interaction"] = dict(
        median=float(np.median(interaction)),
        bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )
    row["marginal_effects"] = []
    for label, numerator, denominator in [
        ("decay_with_patterns", 3, 2),
        ("patterns_with_decay", 3, 1),
    ]:
        ratios = times[:, numerator] / times[:, denominator]
        boot = np.median(ratios[indices], axis=1)
        row["marginal_effects"].append(
            dict(
                effect=label,
                paired_median_reduction_percent=100 * (1 - float(np.median(ratios))),
                paired_bootstrap_95_percent=np.quantile(100 * (1 - boot), [0.025, 0.975]).tolist(),
            )
        )


def main():
    path = base.RESULTS / "integrated_decay_patterns_paired.json"
    report = dict(
        python=sys.version,
        platform=platform.platform(),
        rounds=ROUNDS,
        systems=[],
        labels=LABELS,
        prepared_hamiltonian_assembly=True,
        shared_coefficient_dag=True,
    )

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with threadpool_limits(limits=1):
        first = base.model("r0")
        base.emit("Compiling isolated current-RHS decay/pattern hooks and unpatched control")
        hooks, report["hooked_sources"] = build(first[0], first[2], True)
        control, report["control_sources"] = build(first[0], first[2], False)
        for name in ["rc", "r2", "r0", "q1"]:
            prepared, rho0, gates = first if name == "r0" else base.model(name)
            payload = prepared.to_payload()
            assert payload["hamiltonian_plan"].get("prepared_hamiltonian_assembly", True)
            assert payload["hamiltonian_plan"].get("shared_coefficient_dag", True)
            y0 = prepared.layout.pack(rho0)
            n = prepared.layout.n
            rng = np.random.default_rng(773)
            a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            a = a @ a.conj().T
            random_y = prepared.layout.pack(a / np.trace(a))
            row = dict(
                system=name,
                n_states=n,
                gate_slots=gates,
                counts=hooks.extra_counts(payload),
                normalized_rhs_error=0.0,
                times_by_round_s=[],
                orders=[],
                density_checks=[],
            )
            report["systems"].append(row)
            # Includes all four physical gate configurations and changed powers/detunings.
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
                        reference = np.asarray(control.rhs_check(p, y, float(t), 0))
                        for mode in range(4):
                            actual = np.asarray(hooks.rhs_check(p, y, float(t), mode))
                            error = float(
                                abs(actual - reference).max() / max(abs(reference).max(), 1.0)
                            )
                            row["normalized_rhs_error"] = max(row["normalized_rhs_error"], error)
                            assert error < 1e-10, (name, mode, error)
            duration = 10e-6
            strict = prepared.layout.unpack(
                np.asarray(control.trajectory(payload, y0, 0, duration, 1e-13, 1e-11)[0])
            )

            def solve(mode, payload=payload, y0=y0, duration=duration):
                native = control if mode == 4 else hooks
                return native.trajectory(
                    payload, y0, 0 if mode == 4 else mode, duration, 1e-11, 1e-9
                )

            for mode in range(5):
                solve(mode)
            # Williams balanced crossover: each treatment occupies every position
            # and precedes every other treatment equally in each four-round block.
            design = np.array([[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]])
            for repeat in range(ROUNDS):
                if repeat % 4 == 0:
                    permutation = rng.permutation(4)
                    design_rows = rng.permutation(4)
                order = permutation[design[design_rows[repeat % 4]]].tolist()
                order.insert(repeat % 5, 4)
                times = [None] * 5
                checks = [None] * 5
                for mode in order:
                    base.emit(
                        f"{name}: paired decay/pattern round {repeat + 1}/{ROUNDS}: {LABELS[mode]}"
                    )
                    run = solve(mode)
                    actual = prepared.layout.unpack(np.asarray(run[0]))
                    accuracy = base.accuracy(actual, strict)
                    assert accuracy["passed"], (name, mode, accuracy)
                    times[mode] = run[1]
                    checks[mode] = dict(
                        accuracy=accuracy,
                        trace_error=float(abs(np.trace(actual) - 1)),
                        rhs_calls=run[2],
                        accepted=run[3],
                        rejected=run[4],
                    )
                row["times_by_round_s"].append(times)
                row["orders"].append(order)
                row["density_checks"].append(checks)
                save()
            statistics(row)
            save()
            base.emit(name + ": integrated decay/pattern paired comparison complete")
    save()


if __name__ == "__main__":
    main()
