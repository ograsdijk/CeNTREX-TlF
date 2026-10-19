"""Prototype exact density-support reduction on current RC and R(2) fixtures.

All Rust changes are copied into a task-local native snapshot. The library source
and installed extension are left untouched. Candidates evolve only the exact
reachable packed-Hermitian coordinates, reconstructing omitted entries as zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import bench_rc_propagation as rc_driver
import bench_rust_specialized_rhs as base
import check_shared_mixtures_sectors as sectors
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rust_specialized_rhs_results" / "density_reduction"
CACHE = base.REPO / ".cache" / "benchmarks" / "density_reduction_v1"
ROUNDS = 8
DURATION = 10e-6
ATOL, RTOL = 1e-11, 1e-9
STRICT_ATOL, STRICT_RTOL = 1e-13, 1e-11


def emit(stage: str, **values) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "progress.json").write_text(
        json.dumps(dict(stage=stage, **values), indent=2), encoding="utf-8"
    )
    print(stage, values, flush=True)


def source_hashes() -> dict[str, str]:
    paths = [
        base.REPO / "rust/src/lindblad/rhs.rs",
        base.REPO / "rust/src/ode/tsit5.rs",
        HERE / "specialized_rhs_native.rs",
        HERE / "check_density_reduction.py",
    ]
    return {str(path.relative_to(base.REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def coordinate_closure(prepared, seeds: list[int]) -> set[tuple[int, int]]:
    """Return the exact ordered-pair closure, mirroring the shared-sector audit."""
    payload = prepared.to_payload()
    ham = payload["hamiltonian_plan"]
    n = int(ham["n"])
    adjacency = [set() for _ in range(n)]

    def edge(i: int, j: int) -> None:
        if i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)

    for i, j in zip(*np.nonzero(np.asarray(ham["static_matrix"])), strict=True):
        edge(int(i), int(j))
    for coefficient in ham["coefficients"]:
        for term in coefficient["basis_terms"]:
            if float(term["re"]) != 0 or float(term["im"]) != 0:
                edge(int(term["i"]), int(term["j"]))
    outgoing = [[] for _ in range(n)]
    for target, transfers in enumerate(payload["incoming_transfers_by_target"]):
        for transfer in transfers:
            source = sectors._transfer_source(transfer)
            if source != target and sectors._transfer_rate(transfer) != 0:
                outgoing[source].append(target)
    reached = {(int(i), int(i)) for i in seeds}
    frontier = list(reached)
    while frontier:
        i, j = frontier.pop()
        successors = [(k, j) for k in adjacency[i]]
        successors.extend((i, k) for k in adjacency[j])
        if i == j:
            successors.extend((k, k) for k in outgoing[i])
        for pair in successors:
            if pair not in reached:
                reached.add(pair)
                frontier.append(pair)
    if any((j, i) not in reached for i, j in reached):
        raise AssertionError("density support closure is not Hermitian symmetric")
    return reached


def packed_support(n: int, coordinates: set[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Map upper support coordinates to compact packed slots and upper-row IDs."""
    upper_rows, full_slots = [], []
    upper_row = 0
    for i in range(n):
        for j in range(i, n):
            if (i, j) in coordinates:
                upper_rows.append(upper_row)
                if i == j:
                    full_slots.append(i)
                else:
                    packed_real = n + 2 * sum(n - k - 1 for k in range(i)) + 2 * (j - i - 1)
                    full_slots.extend((packed_real, packed_real + 1))
            upper_row += 1
    return upper_rows, full_slots


PROJECTED_RHS = r"""
pub fn rhs_packed_projected_expanded_into(
    plan:&PreparedLindbladPlan, packed_state:&[f64], t:f64,
    workspace:&mut RhsWorkspace, upper_rows:&[usize], projected_ptrs:&[usize],
    projected_terms:&[usize], out:&mut[f64]
)->Result<(),String> {
    let rhs_plan=plan.expanded_rhs_plan.as_ref().ok_or_else(||
        "projected expanded RHS requires a decomposed plan".to_string())?;
    if packed_state.len()!=plan.layout.packed_len() { return Err("projected RHS input shape mismatch".into()); }
    let expected:usize=upper_rows.iter().map(|&r| if workspace.upper_to_packed[r].im==NO_PACKED_IMAG_INDEX {1} else {2}).sum();
    if out.len()!=expected { return Err(format!("projected output needs {expected} values, got {}",out.len())); }
    if projected_ptrs.len()!=upper_rows.len()+1 || projected_ptrs.last().copied()!=Some(projected_terms.len()) {
        return Err("projected term map has inconsistent dimensions".into());
    }
    let can_skip=!plan.is_time_dependent && workspace.hamiltonian_valid_for==Some(HamiltonianCache::ExpandedSplit);
    if !can_skip {
        workspace.evaluate_parameter_graph(plan,t)?;
        plan.hamiltonian_plan.evaluate_decomposed_coefficients_into(
            workspace.parameter_values.as_slice(),t,&mut workspace.eval_stack,
            &mut workspace.scalar_stack,&mut workspace.expanded_coeff_values)?;
        fill_expanded_sparse_term_values_split(rhs_plan,
            workspace.expanded_coeff_values.as_slice(),
            &mut workspace.expanded_term_values_re,&mut workspace.expanded_term_values_im)?;
        workspace.hamiltonian_valid_for=Some(HamiltonianCache::ExpandedSplit);
        workspace.h_sparse_valid=false;
    }
    let inputs=workspace.expanded_packed_inputs.as_slice();
    let mut cursor=0;
    for (row_index,&output_index) in upper_rows.iter().enumerate() {
        let mut re=0.0; let mut im=0.0;
        for &term_index in &projected_terms[projected_ptrs[row_index]..projected_ptrs[row_index+1]] {
            let input=inputs[term_index];
            let cr=workspace.expanded_term_values_re[term_index];
            let ci=workspace.expanded_term_values_im[term_index];
            let vr=packed_state[input.re];
            let vi=if input.imag_sign==0.0 {0.0} else {input.imag_sign*packed_state[input.im]};
            re+=cr*vr-ci*vi;
            im+=cr*vi+ci*vr;
        }
        out[cursor]=re; cursor+=1;
        if workspace.upper_to_packed[output_index].im!=NO_PACKED_IMAG_INDEX {out[cursor]=im;cursor+=1;}
    }
    Ok(())
}
"""


def generic_benchmark_source(folder: Path, module: str, extra: str = "") -> str:
    """Replace generated system-specialized code with the generic current RHS wrapper."""
    generated = (folder / "lib.rs").read_text(encoding="utf-8")
    marker = "// Benchmark-only wrapper"
    assert generated.count(marker) == 1
    prefix = generated[: generated.index(marker)]
    helper = (HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")
    imports = helper[: helper.index("struct BenchRhs")]
    tail = helper[helper.index("#[pyfunction]") :]
    from confirm_prepared_hamiltonian_integration import WRAPPER as GENERIC_WRAPPER

    # The integration harness hard-codes D+S+U. This comparison also needs a
    # genuine full expanded-sparse control, selected by the existing mode arg.
    GENERIC_WRAPPER = GENERIC_WRAPPER.replace(
        "plan:PreparedLindbladPlan, workspace:RhsWorkspace,",
        "plan:PreparedLindbladPlan, workspace:RhsWorkspace, mode:usize,",
    ).replace("_mode:usize", "mode:usize").replace(
        "Self{workspace:RhsWorkspace::new(&plan),plan,gates,bases}",
        "Self{workspace:RhsWorkspace::new(&plan),plan,gates,bases,mode}",
    ).replace(
        "rhs_packed_into(&self.plan,y,t,ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack,",
        "rhs_packed_into(&self.plan,y,t,if self.mode == 1 {ExecutionMode::ExpandedSparse} else {ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack},",
    )

    registration_marker = "// MODULE_REGISTRATION is appended by the source generator."
    assert tail.count(registration_marker) == 1
    reduced_registration = ""
    if extra:
        reduced_registration = """
    m.add_function(wrap_pyfunction!(reduced_trajectory,m)?)?;
    m.add_function(wrap_pyfunction!(projected_rhs_check,m)?)?;
    m.add_function(wrap_pyfunction!(projected_term_counts,m)?)?;
"""
    registration = f"""
#[pymodule]
fn {module}(m:&Bound<'_,PyModule>)->PyResult<()> {{
    m.add_function(wrap_pyfunction!(rhs_check,m)?)?;
    m.add_function(wrap_pyfunction!(rhs_benchmark,m)?)?;
    m.add_function(wrap_pyfunction!(trajectory,m)?)?;
{reduced_registration}    Ok(())
}}
"""
    tail = tail.replace(registration_marker, extra + registration)
    return prefix + imports + GENERIC_WRAPPER + tail


REDUCED_RUST = r"""
use crate::lindblad::rhs::rhs_packed_projected_expanded_into;
fn build_projected_term_map(plan:&PreparedLindbladPlan, upper_rows:&[usize])->Result<(Vec<usize>,Vec<usize>),String>{
    let rhs=plan.expanded_rhs_plan.as_ref().ok_or_else(||"projected RHS requires expanded sparse plan".to_string())?;
    let mut retained=vec![false;rhs.output_ptrs.len()-1];
    for &row in upper_rows {if row>=retained.len(){return Err("projected output row out of bounds".into());}retained[row]=true;}
    // Closure invariance: no term from a retained input may feed an omitted row.
    for out_row in 0..retained.len(){
        for term_index in rhs.output_ptrs[out_row]..rhs.output_ptrs[out_row+1]{
            if retained[rhs.terms[term_index].input] && !retained[out_row]{
                return Err(format!("support closure is not invariant: input row {} feeds omitted output row {}",rhs.terms[term_index].input,out_row));
            }
        }
    }
    let mut ptrs=Vec::with_capacity(upper_rows.len()+1);let mut terms=Vec::new();ptrs.push(0);
    for &out_row in upper_rows{
        for term_index in rhs.output_ptrs[out_row]..rhs.output_ptrs[out_row+1]{
            if retained[rhs.terms[term_index].input]{terms.push(term_index);}
        }
        ptrs.push(terms.len());
    }
    Ok((ptrs,terms))
}

struct ReducedRhs { inner:BenchRhs, mode:usize, full_slots:Vec<usize>, upper_rows:Vec<usize>, projected_ptrs:Vec<usize>, projected_terms:Vec<usize>, full_y:Vec<f64>, full_dy:Vec<f64> }
impl OdeRhs for ReducedRhs {
    fn dim(&self)->usize { self.full_slots.len() }
    fn eval(&mut self,t:f64,y:&[f64],out:&mut[f64])->Result<(),String> {
        self.full_y.fill(0.0);
        for (k,&slot) in self.full_slots.iter().enumerate(){self.full_y[slot]=y[k];}
        if self.mode<2 {
            let mode=if self.mode==0 {ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack} else {ExecutionMode::ExpandedSparse};
            rhs_packed_into(&self.inner.plan,&self.full_y,t,mode,RhsOptions::default(),&mut self.inner.workspace,&mut self.full_dy)?;
            for (k,&slot) in self.full_slots.iter().enumerate(){out[k]=self.full_dy[slot];}
        } else {
            rhs_packed_projected_expanded_into(&self.inner.plan,&self.full_y,t,&mut self.inner.workspace,
                &self.upper_rows,&self.projected_ptrs,&self.projected_terms,out)?;
        }
        Ok(())
    }
}

#[pyfunction]
fn reduced_trajectory<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'py,f64>,
    full_slots:Vec<usize>,upper_rows:Vec<usize>,mode:usize,duration:f64,atol:f64,rtol:f64)
    ->PyResult<(Bound<'py,PyArray1<f64>>,f64,u64,u64,u64)> {
    let plan=parse_plan_payload(payload)?;
    let mut inner=BenchRhs::new(plan,0);
    let full_dim=N*N;
    let input=y.as_slice()?;
    if input.len()!=full_slots.len(){return Err(PyValueError::new_err("reduced state/slot shape mismatch"));}
    let (projected_ptrs,projected_terms)=build_projected_term_map(&inner.plan,&upper_rows).map_err(PyValueError::new_err)?;
    let mut state=input.to_vec(); let start=Instant::now();
    let mut rhs=ReducedRhs{inner,mode,full_slots,upper_rows,projected_ptrs,projected_terms,full_y:vec![0.;full_dim],full_dy:vec![0.;full_dim]};
    let mut left=125e-9; let end=left+duration; let mut stats=(0,0,0);
    while left<end-1e-16 {
        let boundary=((left/250e-9+1e-8).floor()+1.)*250e-9; let right=boundary.min(end);
        let gates=switch_gates((left+right)/2.);
        let vals:Vec<Complex64>=GATE_SLOTS.iter().map(|&idx|match rhs.inner.plan.parameter_graph.slot_names[idx].as_str(){
            "PA1"=>Complex64::new(gates.0,0.),"PY1"=>Complex64::new(1.-gates.0,0.),
            "PA2"=>Complex64::new(gates.1,0.),"PY2"=>Complex64::new(1.-gates.1,0.),_=>unreachable!()}).collect();
        rhs.inner.workspace.set_scalar_parameter_overrides(GATE_SLOTS,&vals).map_err(PyValueError::new_err)?;
        let options=OdeOptions{abstol:atol,reltol:rtol,dt:1e-10,maxiters:1000000,save_start:false,saveat:Some(vec![right])};
        let mut output=FinalOnlyOutput::new(rhs.dim());
        let s=ode::tsit5::solve_tsit5(&mut rhs,&state,left,right,&options,&mut output).map_err(PyValueError::new_err)?;
        stats.0+=s.rhs_calls;stats.1+=s.accepted_steps;stats.2+=s.rejected_steps;
        if let OdeOutputValues::Full(v)=output.finish().values{state=v;}else{unreachable!()}
        left=right;
    }
    Ok((PyArray1::from_vec(py,state),start.elapsed().as_secs_f64(),stats.0,stats.1,stats.2))
}

#[pyfunction]
fn projected_rhs_check<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,y:PyReadonlyArray1<'py,f64>,
    full_slots:Vec<usize>,upper_rows:Vec<usize>,t:f64)
    ->PyResult<Bound<'py,PyArray1<f64>>> {
    let plan=parse_plan_payload(payload)?; let mut inner=BenchRhs::new(plan,0);
    let input=y.as_slice()?;
    if input.len()!=N*N {return Err(PyValueError::new_err("full packed probe shape mismatch"));}
    let (projected_ptrs,projected_terms)=build_projected_term_map(&inner.plan,&upper_rows).map_err(PyValueError::new_err)?;
    let mut out=vec![0.;full_slots.len()];
    rhs_packed_projected_expanded_into(&inner.plan,input,t,&mut inner.workspace,&upper_rows,&projected_ptrs,&projected_terms,&mut out)
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py,out))
}

#[pyfunction]
fn projected_term_counts(payload:&Bound<'_,PyAny>,upper_rows:Vec<usize>)->PyResult<(usize,usize,usize)> {
    let plan=parse_plan_payload(payload)?;
    let rhs=plan.expanded_rhs_plan.as_ref().ok_or_else(||PyValueError::new_err("missing expanded plan"))?;
    let selected_total:usize=upper_rows.iter().map(|&row|rhs.output_ptrs[row+1]-rhs.output_ptrs[row]).sum();
    let (_,terms)=build_projected_term_map(&plan,&upper_rows).map_err(PyValueError::new_err)?;
    Ok((rhs.terms.len(),selected_total,terms.len()))
}
"""


def build_reduced(prepared, gates: list[str], tag: str):
    module = f"_density_reduction_{tag}"
    folder = CACHE / tag
    folder.mkdir(parents=True, exist_ok=True)
    base.generate(prepared, gates, folder, module)
    generated = generic_benchmark_source(folder, module, REDUCED_RUST)
    rhs_path = base.REPO / "rust/src/lindblad/rhs.rs"
    rhs_snapshot = folder / "rhs_projected_snapshot.rs"
    rhs_snapshot.write_text(rhs_path.read_text(encoding="utf-8") + "\n" + PROJECTED_RHS, encoding="utf-8")
    tsit_path = base.REPO / "rust/src/ode/tsit5.rs"
    tsit_snapshot = folder / "tsit5_density_snapshot.rs"
    tsit = tsit_path.read_text(encoding="utf-8")
    assert tsit.count("(e / control_dim as f64).sqrt()") == 1
    tsit = tsit.replace("(e / control_dim as f64).sqrt()", "(e / BENCH_FULL_DIM as f64).sqrt()")
    tsit = "const BENCH_FULL_DIM:usize=FULL_DIM_PLACEHOLDER;\n".replace("FULL_DIM_PLACEHOLDER", str(prepared.layout.n**2)) + tsit
    tsit_snapshot.write_text(tsit, encoding="utf-8")
    ode_mod_path = base.REPO / "rust/src/ode/mod.rs"
    ode_mod = ode_mod_path.read_text(encoding="utf-8")
    ode_dir = base.REPO / "rust/src/ode"
    for child in ("batch", "common", "dopri5", "fixed", "output"):
        declaration = f"pub mod {child};"
        ode_mod = ode_mod.replace(declaration, f"#[path={json.dumps(str(ode_dir / (child + '.rs')))}] pub mod {child};", 1)
    ode_mod = ode_mod.replace("pub mod tsit5;", f"#[path={json.dumps(str(tsit_snapshot))}] pub mod tsit5;", 1)
    ode_snapshot = folder / "ode_mod_density_snapshot.rs"
    ode_snapshot.write_text(ode_mod, encoding="utf-8")
    generated = generated.replace(json.dumps(str(rhs_path)), json.dumps(str(rhs_snapshot)))
    generated = generated.replace(json.dumps(str(ode_mod_path)), json.dumps(str(ode_snapshot)))
    (folder / "lib.rs").write_text(generated, encoding="utf-8")
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    target = CACHE / "target"
    env["CARGO_TARGET_DIR"] = str(target)
    log_path = folder / "build.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(["cargo", "build", "--release", "--offline", "--manifest-path", str(folder / "Cargo.toml")], env=env, stdout=log, stderr=log)
    if proc.returncode:
        raise RuntimeError(log_path.read_text(encoding="utf-8")[-8000:])
    suffix = __import__("sysconfig").get_config_var("EXT_SUFFIX")
    built = target / "release" / (module + ".dll")
    dest = folder / (module + suffix)
    shutil.copyfile(built, dest)
    import importlib.util
    spec = importlib.util.spec_from_file_location(module, dest)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension


def build_control(prepared, gates: list[str], tag: str):
    """Build an unpatched current-source control in this task's cache tree."""
    module = f"_density_control_{tag}"
    folder = CACHE / ("control_" + tag)
    folder.mkdir(parents=True, exist_ok=True)
    base.generate(prepared, gates, folder, module)
    generated = generic_benchmark_source(folder, module)
    (folder / "lib.rs").write_text(generated, encoding="utf-8")
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    target = CACHE / "target_control"
    env["CARGO_TARGET_DIR"] = str(target)
    log_path = folder / "build.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            ["cargo", "build", "--release", "--offline", "--manifest-path", str(folder / "Cargo.toml")],
            env=env,
            stdout=log,
            stderr=log,
        )
    if proc.returncode:
        raise RuntimeError(log_path.read_text(encoding="utf-8")[-8000:])
    suffix = __import__("sysconfig").get_config_var("EXT_SUFFIX")
    built = target / "release" / (module + ".dll")
    dest = folder / (module + suffix)
    shutil.copyfile(built, dest)
    import importlib.util
    spec = importlib.util.spec_from_file_location(module, dest)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension


def expand_reduced(n: int, reduced: np.ndarray, slots: list[int]) -> np.ndarray:
    full = np.zeros(n*n, dtype=np.float64)
    full[np.asarray(slots, dtype=int)] = reduced
    return full


def full_to_reduced(full: np.ndarray, slots: list[int]) -> np.ndarray:
    return np.asarray(full, dtype=np.float64)[np.asarray(slots, dtype=int)]


def random_positive_density(n: int, support: set[tuple[int, int]], rng) -> np.ndarray:
    """Build a PSD block density whose every nonzero entry lies in support."""
    active = sorted(i for i in range(n) if (i, i) in support)
    if not active:
        raise AssertionError("exact support contains no diagonal populations")
    rho = np.zeros((n, n), dtype=np.complex128)
    weights = {i: float(rng.uniform(0.2, 1.0)) for i in active}
    edges = [(i, j) for i in active for j in active if i < j and (i, j) in support]
    rng.shuffle(edges)
    paired = set()
    for i, j in edges:
        if i in paired or j in paired:
            continue
        rho[i, i] = weights[i]
        rho[j, j] = weights[j]
        phase = float(rng.uniform(-np.pi, np.pi))
        rho[i, j] = 0.2 * np.sqrt(weights[i] * weights[j]) * np.exp(1j * phase)
        rho[j, i] = rho[i, j].conjugate()
        paired.update((i, j))
    for i in active:
        if i not in paired:
            rho[i, i] = weights[i]
    rho /= np.trace(rho).real
    if np.linalg.eigvalsh(rho).min() < -1e-14:
        raise AssertionError("constructed probe density is not positive")
    return rho


def run() -> None:
    emit("initializing", rounds=ROUNDS, systems=["rc", "r2"])
    import bench_rust_specialized_rhs as builder
    systems = {}
    for name in ("rc", "r2"):
        if name == "rc":
            case = rc_driver.build_case(rc_driver.CONFIGS["nominal"])
            rc_mixtures, _ = sectors._initial_j_mixtures(case)
            prepared, gates = sectors._prepare_rc_static_gate_plan(case)
            # Equal population over the J=2 manifold, as in the RC mixture audit.
            rho0 = rc_mixtures[1]
        else:
            prepared, rho0, gates = builder.model(name)
        support = np.flatnonzero(np.diag(rho0).real > 0).tolist()
        reached = coordinate_closure(prepared, support)
        audit = sectors.exact_density_closure(prepared, [support], [name])["mixtures"][0]
        if len(reached) != audit["reached_coordinate_pairs"]:
            raise AssertionError(f"closure cross-check differs for {name}")
        upper_rows, full_slots = packed_support(prepared.layout.n, reached)
        if len(full_slots) != len(reached):
            raise AssertionError(f"packed support mapping mismatch for {name}: {len(full_slots)} vs {len(reached)}")
        systems[name] = (prepared, rho0, gates, upper_rows, full_slots, audit, reached)
    # Generate controls from the already-current native source, then build two
    # isolated reduced candidates: full RHS projection and sparse output-row RHS.
    controls = {}
    candidates = {}
    for name, (prepared, _rho0, gates, _upper_rows, slots, _, _) in systems.items():
        emit("building", system=name, n=prepared.layout.n, retained=len(slots), full=prepared.layout.n**2)
        controls[name] = build_control(prepared, gates, name)
        candidates[name] = build_reduced(prepared, gates, "density_" + name)

    all_results = {}
    def save_partial():
        RESULTS.mkdir(parents=True, exist_ok=True)
        partial = dict(rounds=ROUNDS, duration_seconds=DURATION, atol=ATOL, rtol=RTOL,
                       strict_atol=STRICT_ATOL, strict_rtol=STRICT_RTOL, systems=all_results)
        (RESULTS / "density_reduction_progress.json").write_text(
            json.dumps(partial, indent=2, default=lambda x: float(x)), encoding="utf-8"
        )

    for name, (prepared, rho0, _gates, upper_rows, slots, audit, reached) in systems.items():
        emit("reference", system=name)
        control = controls[name]
        reduced_ext = candidates[name]
        payload = prepared.to_payload()
        total_terms, selected_row_terms, used_terms = reduced_ext.projected_term_counts(payload, upper_rows)
        y0 = prepared.layout.pack(rho0)
        reference = np.asarray(control.trajectory(payload, y0, 0, DURATION, STRICT_ATOL, STRICT_RTOL)[0], dtype=float)
        # Random Hermitian-compliant states in the exact support verify projected
        # RHS rows at fixed times. We compare both expansion+full RHS and filtered RHS.
        rng = np.random.default_rng(20261017 + prepared.layout.n)
        omitted_slots = np.setdiff1d(np.arange(prepared.layout.packed_len), np.asarray(slots, dtype=int))
        rhs_errors, cross_errors, omitted_errors = [], [], []
        for probe_index in range(3):
            probe_rho = random_positive_density(prepared.layout.n, reached, rng)
            probe_full = prepared.layout.pack(probe_rho)
            full_rhs = np.asarray(control.rhs_check(payload, probe_full, 125e-9, 1), dtype=float)
            dsu_rhs = np.asarray(control.rhs_check(payload, probe_full, 125e-9, 0), dtype=float)
            projected = np.asarray(
                reduced_ext.projected_rhs_check(payload, probe_full, slots, upper_rows, 125e-9),
                dtype=float,
            )
            rhs_error = float(np.max(np.abs(projected - full_rhs[slots])))
            cross_error = float(np.max(np.abs(full_rhs[slots] - dsu_rhs[slots])))
            omitted_rhs = float(np.max(np.abs(full_rhs[omitted_slots]))) if omitted_slots.size else 0.0
            scale = max(1.0, float(np.max(np.abs(full_rhs[slots]))))
            if max(rhs_error, cross_error, omitted_rhs) > 1e-9 * scale:
                raise AssertionError(f"projected RHS mismatch for {name} probe {probe_index}: projected={rhs_error}, D+S+U-vs-expanded={cross_error}, omitted-row leakage={omitted_rhs}")
            rhs_errors.append(rhs_error)
            cross_errors.append(cross_error)
            omitted_errors.append(omitted_rhs)
        rhs_error = max(rhs_errors)
        cross_error = max(cross_errors)
        omitted_rhs = max(omitted_errors)
        # Trajectory-time comparisons below exercise the custom RHS; fixed-time
        # direct comparison against the full reference is included in raw checks.
        y_reduced = full_to_reduced(y0, slots)
        variants = ["full_dsu", "full_expanded", "reduced_dsu", "reduced_expanded_rows"]
        def solve(label, atol=ATOL, rtol=RTOL, *, control=control, payload=payload,
                  y0=y0, reduced_ext=reduced_ext, y_reduced=y_reduced, slots=slots,
                  upper_rows=upper_rows, prepared=prepared):
            if label.startswith("full_"):
                mode = 0 if label == "full_dsu" else 1
                result = control.trajectory(payload, y0, mode, DURATION, atol, rtol)
                return np.asarray(result[0], dtype=float), dict(elapsed=float(result[1]), rhs_calls=int(result[2]))
            mode = 0 if label == "reduced_dsu" else 2
            result = reduced_ext.reduced_trajectory(payload, y_reduced, slots, upper_rows, mode, DURATION, atol, rtol)
            return expand_reduced(prepared.layout.n, np.asarray(result[0]), slots), dict(elapsed=float(result[1]), rhs_calls=int(result[2]))
        # One correctness pass before timings, including strict-tolerance candidate.
        checks = {}
        for label in variants:
            got, stats = solve(label, STRICT_ATOL, STRICT_RTOL)
            err = base.accuracy(prepared.layout.unpack(got), prepared.layout.unpack(reference))
            trace = float(abs(np.trace(prepared.layout.unpack(got)) - 1.0))
            checks[label] = dict(error=err, trace_error=trace, strict_run_seconds=stats["elapsed"])
            if not err["passed"] or trace > 1e-9:
                raise AssertionError(f"strict candidate fails {name}/{label}: {checks[label]}")
        rounds = {label: [] for label in variants}
        system_base = dict(n_states=prepared.layout.n, full_coordinates=prepared.layout.n**2,
            retained_coordinates=len(slots), retained_fraction=len(slots)/(prepared.layout.n**2),
            closure=audit, correctness=checks, random_support_rhs_max_abs_error=rhs_error,
            random_support_omitted_rhs_max_abs=omitted_rhs,
            random_support_rhs_dsu_vs_expanded_max_abs_error=cross_error,
            expanded_rhs_terms=dict(full=total_terms, selected_rows=selected_row_terms,
                selected_rows_after_input_pruning=used_terms,
                removed_from_selected_rows=selected_row_terms-used_terms))
        all_results[name] = dict(system_base, rounds=rounds)
        save_partial()
        rng_order = random.Random(20261017 + prepared.layout.n)
        for r in range(ROUNDS):
            order = variants.copy()
            rng_order.shuffle(order)
            emit("timing_round", system=name, round=r+1, rounds=ROUNDS, order=order)
            for label in order:
                started = time.perf_counter()
                got, stats = solve(label)
                wall = time.perf_counter() - started
                err = base.accuracy(prepared.layout.unpack(got), prepared.layout.unpack(reference))
                trace = float(abs(np.trace(prepared.layout.unpack(got)) - 1.0))
                if not err["passed"] or trace > 1e-9:
                    raise AssertionError(f"timed result fails {name}/{label}: {err}, trace={trace}")
                rounds[label].append(dict(wall_seconds=wall, solver_seconds=stats["elapsed"], rhs_calls=stats["rhs_calls"], error=err, trace_error=trace))
            save_partial()
        emit("completed_system", system=name)
    report = dict(date=platform.platform(), python=sys.version, source_sha256=source_hashes(),
        rounds=ROUNDS, duration_seconds=DURATION,
        atol=ATOL, rtol=RTOL, strict_atol=STRICT_ATOL, strict_rtol=STRICT_RTOL,
        method="Tsit5, 250 ns switching segments, isolated exact-support coordinate reduction",
        systems=all_results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "density_reduction.json"
    path.write_text(json.dumps(report, indent=2, default=lambda x: float(x)), encoding="utf-8")
    emit("complete", output=str(path))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
