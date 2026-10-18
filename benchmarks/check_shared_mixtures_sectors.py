"""Isolated RC tests for shared-H mixture evolution and exact density closure.

The benchmark prototype is deliberately separate from the installed library. It
uses the current prepared-H + coefficient-DAG + D+S+U path and records results
under its own directory so it cannot overwrite another benchmark's progress.
"""

from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import bench_rust_specialized_rhs as base
import bench_rc_propagation as rc_driver
from centrex_tlf import lindblad, states

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rust_specialized_rhs_results" / "shared_mixtures_sectors"
CACHE = base.CACHE / "shared_mixtures_sectors"
MIXTURE_LABELS = (
    "X_J1_F1_3o2_F1_mF0",
    "X_J2_F1_5o2_F2_mF0",
    "X_J3_F1_7o2_F3_mF0",
)
ROUNDS = 8
ATOL = 1e-11
RTOL = 1e-9
STRICT_ATOL = 1e-13
STRICT_RTOL = 1e-11
DURATION = 10e-6


SHARED_RUST = r"""
struct SharedMixtureRhs { inner: BenchRhs }
impl OdeRhs for SharedMixtureRhs {
    fn dim(&self)->usize { 3*N*N }
    fn eval(&mut self,t:f64,y:&[f64],out:&mut[f64])->Result<(),String> {
        let width=N*N;
        for mixture in 0..3 {
            let a=mixture*width;
            rhs_packed_into(&self.inner.plan,&y[a..a+width],t,
                ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack,
                RhsOptions::default(),&mut self.inner.workspace,&mut out[a..a+width])?;
        }
        Ok(())
    }
}

#[pyfunction]
fn shared_trajectory<'py>(py:Python<'py>,payload:&Bound<'_,PyAny>,
    y:PyReadonlyArray2<'py,f64>,duration:f64,atol:f64,rtol:f64)
    ->PyResult<(Bound<'py,PyArray1<f64>>,f64,u64,u64,u64)> {
    let plan=parse_plan_payload(payload)?;
    if y.shape()[0]!=3 || y.shape()[1]!=N*N {
        return Err(PyValueError::new_err("expected three packed density states"));
    }
    let mut state=y.as_slice()?.to_vec();
    let mut inner=BenchRhs::new(plan,0);
    let start=Instant::now();
    let mut left=125e-9;
    let end=left+duration;
    let mut stats=(0,0,0);
    while left < end-1e-16 {
        let boundary=((left/250e-9+1e-8).floor()+1.)*250e-9;
        let right=boundary.min(end);
        let gates=switch_gates((left+right)/2.);
        let vals:Vec<Complex64>=inner.gates.iter().map(|&idx| {
            match inner.plan.parameter_graph.slot_names[idx].as_str() {
                "PA1"=>Complex64::new(gates.0,0.), "PY1"=>Complex64::new(1.-gates.0,0.),
                "PA2"=>Complex64::new(gates.1,0.), "PY2"=>Complex64::new(1.-gates.1,0.),
                _=>unreachable!()
            }
        }).collect();
        inner.workspace.set_scalar_parameter_overrides(&inner.gates,&vals)
            .map_err(PyValueError::new_err)?;
        let options=OdeOptions {abstol:atol,reltol:rtol,dt:1e-10,maxiters:1000000,
            save_start:false,saveat:Some(vec![right])};
        let mut rhs=SharedMixtureRhs {inner};
        let mut output=FinalOnlyOutput::new(3*N*N);
        let s=ode::tsit5::solve_tsit5(&mut rhs,&state,left,right,&options,&mut output)
            .map_err(PyValueError::new_err)?;
        stats.0+=s.rhs_calls;stats.1+=s.accepted_steps;stats.2+=s.rejected_steps;
        if let OdeOutputValues::Full(v)=output.finish().values {state=v;} else {unreachable!()}
        inner=rhs.inner;
        left=right;
    }
    Ok((PyArray1::from_vec(py,state),start.elapsed().as_secs_f64(),stats.0,stats.1,stats.2))
}
"""


def _transfer_source(transfer: object) -> int:
    if isinstance(transfer, dict):
        return int(transfer["source"])
    if hasattr(transfer, "source"):
        return int(transfer.source)
    return int(transfer[0])


def _transfer_rate(transfer: object) -> float:
    if isinstance(transfer, dict):
        return float(transfer["rate"])
    if hasattr(transfer, "rate"):
        return float(transfer.rate)
    return float(transfer[1])


def exact_density_closure(
    prepared, seed_groups: list[list[int]], mixture_names: tuple[str, ...] | list[str]
) -> dict:
    """Conservative structural closure over the union Hamiltonian and decay graph.

    Hamiltonian edges include every static entry and every lowered basis term,
    including terms whose runtime gate is zero in a particular configuration.
    Hamiltonian edges move either index of a density coordinate. Decay transfers
    move only diagonal population coordinates in this structured dissipator;
    damping alone creates no new support. No magnitude cutoff is used.
    """
    payload = prepared.to_payload()
    hamiltonian = payload["hamiltonian_plan"]
    n = int(hamiltonian["n"])
    adjacency = [set() for _ in range(n)]

    def add_edge(i: int, j: int) -> None:
        if i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)

    static = np.asarray(hamiltonian["static_matrix"], dtype=np.complex128)
    for i, j in zip(*np.nonzero(static), strict=True):
        add_edge(int(i), int(j))
    for coefficient in hamiltonian["coefficients"]:
        for term in coefficient["basis_terms"]:
            # A zero basis multiplier contributes no edge. Runtime coefficients
            # and gate values are otherwise treated as potentially nonzero.
            if float(term["re"]) != 0.0 or float(term["im"]) != 0.0:
                add_edge(int(term["i"]), int(term["j"]))

    decay_outgoing = [[] for _ in range(n)]
    decay_edges = 0
    for target, transfers in enumerate(payload["incoming_transfers_by_target"]):
        for transfer in transfers:
            source = _transfer_source(transfer)
            if source != target and _transfer_rate(transfer) != 0.0:
                decay_outgoing[source].append(target)
                decay_edges += 1

    rows = []
    full_coordinates = n * n
    for name, seeds in zip(mixture_names, seed_groups, strict=True):
        reached = {(int(seed), int(seed)) for seed in seeds}
        frontier = list(reached)
        while frontier:
            i, j = frontier.pop()
            successors = [(k, j) for k in adjacency[i]]
            successors.extend((i, k) for k in adjacency[j])
            if i == j:
                successors.extend((target, target) for target in decay_outgoing[i])
            for coordinate in successors:
                if coordinate not in reached:
                    reached.add(coordinate)
                    frontier.append(coordinate)
        asymmetric = sum((j, i) not in reached for i, j in reached)
        reachable_real_coordinates = len(reached)
        rows.append(
            dict(
                mixture=name,
                seeds=[int(seed) for seed in seeds],
                reached_coordinate_pairs=len(reached),
                hermitian_support_mismatches=asymmetric,
                reachable_real_density_coordinates=reachable_real_coordinates,
                full_real_density_coordinates=full_coordinates,
                removable_fraction=1.0 - reachable_real_coordinates / full_coordinates,
            )
        )
    return dict(
        n_states=n,
        full_real_density_coordinates=full_coordinates,
        exact_decay_transfer_edges=decay_edges,
        mixtures=rows,
        scope=(
            "10 us E=200 V/cm, B=0.01 G benchmark fixture; structural union over all "
            "lowered H terms and decay transfers; no threshold pruning"
        ),
    )


def _initial_j_mixtures(case) -> tuple[list[np.ndarray], list[list[int]]]:
    mixtures = []
    supports = []
    n = len(case.system.QN)
    for j in (1, 2, 3):
        indices = np.asarray(
            states.QuantumSelector(electronic=states.ElectronicState.X, J=j).get_indices(
                case.system.QN
            ),
            dtype=int,
        ).ravel()
        if indices.size == 0:
            raise RuntimeError(f"RC fixture contains no X,J={j} states")
        rho = np.zeros((n, n), dtype=np.complex128)
        rho[indices, indices] = 1.0 / indices.size
        mixtures.append(rho)
        supports.append(indices.tolist())
    return mixtures, supports


def _prepare_rc_static_gate_plan(case):
    """Mirror the existing native RHS benchmark's scalar PA/PY gate slots."""
    base_values = dict(case.params_dynamic.base_parameters)
    compounds = dict(case.params_dynamic.compound_parameters)
    base_values.update(
        Ωt0=rc_driver.GAMMA,
        β0=3.8,
        ωphase0=2 * np.pi * 1.6e6,
    )
    gates = []
    for key in ("PA1", "PY1", "PA2", "PY2"):
        if key in compounds:
            compounds.pop(key)
            base_values[key] = 1.0 if key.startswith("PA") else 0.0
            gates.append(key)
    prepared = lindblad.prepare_lindblad_problem(
        case.system,
        lindblad.LindbladParameters(base_values, compounds),
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    return prepared, gates


def _run_library_batch(prepared, gates, rho_batch, *, atol, rtol, parallel, threads=None):
    """Run independent mixtures, segmenting RC at the gate boundaries."""
    state = np.stack([prepared.layout.pack(rho) for rho in rho_batch])
    initial = 125e-9
    final = initial + DURATION
    stats = []
    segments = []
    if gates:
        while initial < final - 1e-16:
            boundary = ((initial / 250e-9 + 1e-8) // 1 + 1) * 250e-9
            right = min(boundary, final)
            pa1, pa2 = rc_driver.switch_values((initial + right) / 2.0)
            slot_values = dict(PA1=pa1, PY1=1 - pa1, PA2=pa2, PY2=1 - pa2)
            parameter_batch = np.tile(
                np.asarray([slot_values[name] for name in gates], dtype=np.complex128),
                (len(rho_batch), 1),
            )
            segments.append((initial, right, parameter_batch))
            initial = right
    else:
        segments.append((initial, final, None))
    for initial, right, parameter_batch in segments:
        result = lindblad.solve_lindblad_batch(
            prepared,
            state,
            (initial, right),
            parameter_slots=gates if gates else None,
            parameter_batch=parameter_batch,
            solver="tsit5",
            execution_mode="structured_sparse_dense_diag_soa_single_unpack",
            abstol=atol,
            reltol=rtol,
            dt=1e-10,
            output="full",
            output_when="final",
            parallel=parallel,
            threads=threads,
            collect_stats=True,
        )
        state = np.asarray(result.values, dtype=np.float64)
        stats.append(result.solver_stats)
    return state, stats


def _density_checks(prepared, packed_batch, strict_batch) -> list[dict]:
    rows = []
    for actual, reference in zip(packed_batch, strict_batch, strict=True):
        actual_rho = prepared.layout.unpack(actual)
        reference_rho = prepared.layout.unpack(reference)
        check = base.accuracy(actual_rho, reference_rho)
        check["trace_error"] = float(abs(np.trace(actual_rho) - 1.0))
        rows.append(check)
    return rows


def _shared_run(extension, prepared, rho_batch, *, atol, rtol):
    packed = np.stack([prepared.layout.pack(rho) for rho in rho_batch])
    values, elapsed, calls, accepted, rejected = extension.shared_trajectory(
        prepared.to_payload(), packed, DURATION, atol, rtol
    )
    return (
        np.asarray(values, dtype=np.float64).reshape((3, prepared.layout.packed_len)),
        dict(
            elapsed_seconds=float(elapsed),
            rhs_calls=int(calls),
            accepted_steps=int(accepted),
            rejected_steps=int(rejected),
        ),
    )


def _independent_cached_run(extension, prepared, rho_batch, *, atol, rtol):
    packed = []
    elapsed = 0.0
    calls = accepted = rejected = 0
    payload = prepared.to_payload()
    for rho in rho_batch:
        state = prepared.layout.pack(rho)
        values, duration, n_calls, n_accepted, n_rejected = extension.trajectory(
            payload, state, 0, DURATION, atol, rtol
        )
        packed.append(np.asarray(values, dtype=np.float64))
        elapsed += float(duration)
        calls += int(n_calls)
        accepted += int(n_accepted)
        rejected += int(n_rejected)
    return np.stack(packed), dict(
        summed_solver_seconds=elapsed,
        rhs_calls=calls,
        accepted_steps=accepted,
        rejected_steps=rejected,
    )


def _independent_native_run(extension, prepared, rho_batch, *, atol, rtol):
    packed = []
    elapsed = 0.0
    calls = accepted = rejected = 0
    payload = prepared.to_payload()
    for rho in rho_batch:
        values, duration, n_calls, n_accepted, n_rejected = extension.trajectory(
            payload, prepared.layout.pack(rho), 0, DURATION, atol, rtol
        )
        packed.append(np.asarray(values, dtype=np.float64))
        elapsed += float(duration)
        calls += int(n_calls)
        accepted += int(n_accepted)
        rejected += int(n_rejected)
    return np.stack(packed), dict(
        summed_solver_seconds=elapsed,
        rhs_calls=calls,
        accepted_steps=accepted,
        rejected_steps=rejected,
    )
def _run_mixture_benchmark(
    system_name, prepared, gates, rho_batch, mixture_names, extension, control_extension
) -> dict:
    strict, strict_stats = _independent_native_run(
        control_extension, prepared, rho_batch, atol=STRICT_ATOL, rtol=STRICT_RTOL
    )
    strict_checks = _density_checks(prepared, strict, strict)
    assert all(row["passed"] and row["trace_error"] < 1e-9 for row in strict_checks)
    candidates = (
        "independent_serial_production_native",
        "independent_parallel_production",
        "independent_serial_cached_native",
        "shared_lockstep_cached",
    )

    def solve(label: str):
        if label == "shared_lockstep_cached":
            return _shared_run(extension, prepared, rho_batch, atol=ATOL, rtol=RTOL)
        if label == "independent_serial_cached_native":
            return _independent_cached_run(extension, prepared, rho_batch, atol=ATOL, rtol=RTOL)
        if label == "independent_serial_production_native":
            return _independent_native_run(
                control_extension, prepared, rho_batch, atol=ATOL, rtol=RTOL
            )
        packed, stats = _run_library_batch(
            prepared,
            gates,
            rho_batch,
            atol=ATOL,
            rtol=RTOL,
            parallel=label == "independent_parallel_production",
            threads=3 if label == "independent_parallel_production" else 1,
        )
        return packed, dict(segment_stats=stats)

    for label in candidates:
        values, _ = solve(label)
        checks = _density_checks(prepared, values, strict)
        assert all(row["passed"] and row["trace_error"] < 1e-9 for row in checks), (label, checks)

    results = {label: [] for label in candidates}
    orders = []
    rng = np.random.default_rng(20261017)
    design = np.asarray(
        [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]],
        dtype=int,
    )
    for round_index in range(ROUNDS):
        if round_index % 4 == 0:
            label_order = rng.permutation(len(candidates))
            row_order = rng.permutation(len(design))
        order = [
            candidates[index]
            for index in label_order[design[row_order[round_index % 4]]]
        ]
        orders.append(order)
        for label in order:
            start = time.perf_counter()
            values, solver_stats = solve(label)
            wall = time.perf_counter() - start
            checks = _density_checks(prepared, values, strict)
            assert all(row["passed"] and row["trace_error"] < 1e-9 for row in checks), (
                label,
                round_index,
                checks,
            )
            results[label].append(
                dict(
                    wall_seconds=float(wall),
                    solver=solver_stats,
                    density_checks=checks,
                )
            )
            save_progress(
                dict(
                    status="running",
                    current=f"round {round_index + 1}/{ROUNDS}: {label}",
                    rounds=round_index + 1,
                    labels=candidates,
                    results=results,
                    orders=orders,
                )
            )

    medians = {
        label: float(np.median([row["wall_seconds"] for row in rows]))
        for label, rows in results.items()
    }
    reductions = {
        "independent_serial_cached_vs_production_native_percent": 100.0
        * (1.0 - medians["independent_serial_cached_native"] / medians["independent_serial_production_native"]),
        "shared_cached_vs_independent_cached_percent": 100.0
        * (1.0 - medians["shared_lockstep_cached"] / medians["independent_serial_cached_native"]),
        "shared_cached_vs_production_native_percent": 100.0
        * (1.0 - medians["shared_lockstep_cached"] / medians["independent_serial_production_native"]),
        "shared_cached_vs_production_parallel_wall_percent": 100.0
        * (1.0 - medians["shared_lockstep_cached"] / medians["independent_parallel_production"]),
    }
    paired = {}
    bootstrap_rng = np.random.default_rng(7421)
    for numerator, denominator in (
        ("independent_serial_cached_native", "independent_serial_production_native"),
        ("shared_lockstep_cached", "independent_serial_cached_native"),
        ("shared_lockstep_cached", "independent_serial_production_native"),
        ("shared_lockstep_cached", "independent_parallel_production"),
    ):
        ratios = np.asarray([
            1.0 - results[numerator][i]["wall_seconds"] / results[denominator][i]["wall_seconds"]
            for i in range(ROUNDS)
        ])
        bootstrap = np.median(
            bootstrap_rng.choice(ratios, size=(20000, len(ratios)), replace=True), axis=1
        )
        paired[f"{numerator}_vs_{denominator}"] = dict(
            median_reduction_percent=float(100.0 * np.median(ratios)),
            bootstrap_95_percentile_ci_percent=[
                float(100.0 * np.quantile(bootstrap, 0.025)),
                float(100.0 * np.quantile(bootstrap, 0.975)),
            ],
            paired_round_reductions_percent=[float(100.0 * x) for x in ratios],
        )
    # `rhs_calls` from shared OdeStats counts calls to the stacked OdeRhs::eval;
    # each such call evaluates one full density RHS for each of three mixtures.
    # Compare equivalent density RHS applications, not the outer stage-call count.
    serial_density_calls = float(
        np.median([
            row["solver"]["rhs_calls"]
            for row in results["independent_serial_production_native"]
        ])
    )
    shared_stage_calls = float(
        np.median([row["solver"]["rhs_calls"] for row in results["shared_lockstep_cached"]])
    )
    shared_density_calls = 3.0 * shared_stage_calls
    return dict(
        system=system_name,
        initial_mixtures=mixture_names,
        duration_s=DURATION,
        tolerance=dict(abstol=ATOL, reltol=RTOL),
        strict_reference=dict(
            abstol=STRICT_ATOL,
            reltol=STRICT_RTOL,
            checks=strict_checks,
            segment_solver_stats=strict_stats,
        ),
        shared_error_control="maximum per-mixture RMS normalized error",
        candidate_medians_seconds=medians,
        reductions_vs_independent_serial_percent=reductions,
        paired_reductions=paired,
        rhs_work_summary=dict(
            independent_serial_stage_calls_median=serial_density_calls,
            independent_serial_density_rhs_applications_median=serial_density_calls,
            shared_lockstep_stage_calls_median=shared_stage_calls,
            shared_lockstep_density_rhs_applications_median=shared_density_calls,
            shared_vs_serial_density_rhs_applications_percent=100.0
            * (shared_density_calls / serial_density_calls - 1.0),
            note=(
                "Shared solver rhs_calls count one outer evaluation per stage; the shared "+
                "RHS loop evaluates all three packed density matrices in each outer call."
            ),
        ),
        rounds=results,
        orders=orders,
        parallel_note=(
            "production parallel uses the library batch API with three independent workers. "
            "Unpatched production-native and time-cached independent serial candidates use "
            "the same native trajectory helper as the shared run. The cached serial ratio "
            "isolates cross-mixture reuse beyond same-time RK endpoint reuse."
        ),
    )


def save_progress(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    paths = (
        HERE / "check_shared_mixtures_sectors.py",
        HERE / "specialized_rhs_native.rs",
        base.REPO / "rust/src/lindblad/rhs.rs",
        base.REPO / "rust/src/ode/tsit5.rs",
        base.REPO / "rust/src/ode/mod.rs",
    )
    return {
        str(path.relative_to(base.REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _patched_rhs_source(source: str) -> str:
    """Add an exact same-time H cache only to the isolated benchmark snapshot."""
    source = source.replace(
        "pub struct RhsWorkspace {",
        "pub struct RhsWorkspace {\n    bench_shared_cached_time: Option<u64>,",
        1,
    )
    source = source.replace(
        "            upper_layout: upper_layout.clone(),",
        "            bench_shared_cached_time: None,\n            upper_layout: upper_layout.clone(),",
        1,
    )
    source = source.replace(
        "        self.parameter_overrides.clear();",
        "        self.bench_shared_cached_time = None;\n        self.parameter_overrides.clear();",
        1,
    )
    old = "let can_skip = !plan.is_time_dependent && workspace.hamiltonian_valid_for == Some(cache_kind);"
    new = """let cache_time_matches=workspace.bench_shared_cached_time==Some(t.to_bits());
    let can_skip=workspace.hamiltonian_valid_for==Some(cache_kind)
        && (!plan.is_time_dependent || cache_time_matches);"""
    assert source.count(old) == 1
    source = source.replace(old, new, 1)
    marker = "workspace.hamiltonian_valid_for = Some(cache_kind);"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        marker + "\n        workspace.bench_shared_cached_time=Some(t.to_bits());",
        1,
    )
    return source


def _max_mixture_error_norm_source(source: str, n_states: int) -> str:
    """Use the max local RMS across exactly three N×N density blocks."""
    start = source.index("fn error_norm(")
    end = source.index("\n}\n\nfn dense_out_tsit5", start) + 2
    replacement = r'''const BENCH_N_STATES:usize=BENCH_N_STATES_PLACEHOLDER;
fn error_norm(
    y: &[f64], yn: &[f64], h: f64, k: &[f64], d: usize,
    control_dim: usize, atol: f64, rtol: f64,
) -> f64 {
    let blocks=if control_dim==3*BENCH_N_STATES*BENCH_N_STATES {3} else {1};
    let block_dim=control_dim/blocks;
    let mut worst=0.0_f64;
    for block in 0..blocks {
        let mut sum=0.0;
        for local in 0..block_dim {
            let i=block*block_dim+local;
            let scale=atol+y[i].abs().max(yn[i].abs())*rtol;
            let error=h*(E1*k[i]+E2*k[d+i]+E3*k[2*d+i]+E4*k[3*d+i]
                +E5*k[4*d+i]+E6*k[5*d+i]+E7*k[6*d+i]);
            sum+=(error/scale)*(error/scale);
        }
        worst=worst.max((sum/block_dim as f64).sqrt());
    }
    worst
}'''
    replacement = replacement.replace("BENCH_N_STATES_PLACEHOLDER", str(n_states))
    return source[:start] + replacement + source[end:]


def build_shared_native(prepared, gates: list[str], tag: str):
    module = "_rhs_shared_mixtures_" + tag
    folder = CACHE / ("shared_native_" + tag)
    base.generate(prepared, gates, folder, module)
    generated = (folder / "lib.rs").read_text(encoding="utf-8")
    rhs_path = base.REPO / "rust/src/lindblad/rhs.rs"
    rhs_snapshot = folder / "rhs_shared_snapshot.rs"
    rhs_snapshot.write_text(_patched_rhs_source(rhs_path.read_text(encoding="utf-8")), encoding="utf-8")
    tsit_path = base.REPO / "rust/src/ode/tsit5.rs"
    tsit_snapshot = folder / "tsit5_max_mixture_snapshot.rs"
    tsit_snapshot.write_text(
        _max_mixture_error_norm_source(
            tsit_path.read_text(encoding="utf-8"), prepared.layout.n
        ), encoding="utf-8"
    )
    ode_mod_path = base.REPO / "rust/src/ode/mod.rs"
    ode_snapshot = folder / "ode_mod_shared_snapshot.rs"
    ode_module = ode_mod_path.read_text(encoding="utf-8")
    assert ode_module.count("pub mod tsit5;") == 1
    ode_dir = base.REPO / "rust/src/ode"
    for child in ("batch", "common", "dopri5", "fixed", "output"):
        declaration = f"pub mod {child};"
        assert ode_module.count(declaration) == 1
        ode_module = ode_module.replace(
            declaration,
            f"#[path={json.dumps(str(ode_dir / (child + '.rs')))}] pub mod {child};",
            1,
        )
    ode_module = ode_module.replace(
        "pub mod tsit5;",
        f"#[path={json.dumps(str(tsit_snapshot))}] pub mod tsit5;",
        1,
    )
    ode_snapshot.write_text(ode_module, encoding="utf-8")
    generated = generated.replace(
        json.dumps(str(rhs_path)), json.dumps(str(rhs_snapshot))
    ).replace(json.dumps(str(ode_mod_path)), json.dumps(str(ode_snapshot)))
    # Keep the production module imports and dimensions but discard the huge
    # generated per-coordinate benchmark RHS; this measures the current generic
    # production D+S+U kernel used by the public library path.
    prefix = generated[: generated.index("// Benchmark-only wrapper")]
    helper = (HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")
    imports = helper[: helper.index("struct BenchRhs")].replace(
        "use numpy::{PyArray1, PyReadonlyArray1};",
        "use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};",
    )
    from confirm_prepared_hamiltonian_integration import WRAPPER as GENERIC_WRAPPER

    tail = helper[helper.index("#[pyfunction]") :]
    tail = tail.replace("N*N", "rhs.dim()").replace("GATE_SLOTS", "&rhs.gates")
    tail = tail.replace("=&rhs.gates.iter()", "=rhs.gates.iter()")
    generated = (
        prefix
        + imports
        + GENERIC_WRAPPER
        + tail
        + SHARED_RUST
        + f"""
#[pymodule]
fn {module}(m:&Bound<'_,PyModule>)->PyResult<()> {{
    m.add_function(wrap_pyfunction!(rhs_check,m)?)?;
    m.add_function(wrap_pyfunction!(rhs_benchmark,m)?)?;
    m.add_function(wrap_pyfunction!(trajectory,m)?)?;
    m.add_function(wrap_pyfunction!(shared_trajectory,m)?)?;Ok(())
}}
"""
    )
    (folder / "lib.rs").write_text(generated, encoding="utf-8")
    cargo = (folder / "Cargo.toml").read_text(encoding="utf-8")
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["CARGO_TARGET_DIR"] = str(CACHE / "target_shared")
    log_path = folder / "build.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            ["cargo", "build", "--release", "--offline", "--manifest-path", str(folder / "Cargo.toml")],
            env=env, stdout=log, stderr=log,
        )
    if result.returncode:
        raise RuntimeError(log_path.read_text(encoding="utf-8")[-6000:])
    suffix = __import__("sysconfig").get_config_var("EXT_SUFFIX")
    library = folder / "target/release" / (module + ".dll")
    # The shared target is set outside the package folder; use cargo's actual path.
    library = CACHE / "target_shared/release" / (module + ".dll")
    import shutil

    destination = folder / (module + suffix)
    shutil.copyfile(library, destination)
    import importlib.util

    spec = importlib.util.spec_from_file_location(module, destination)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension


def build_independent_control_native(prepared, gates: list[str], tag: str):
    """Build an unpatched current-source control in the task-local cache."""
    module = "_rhs_independent_mixtures_" + tag
    folder = CACHE / ("independent_control_native_" + tag)
    base.generate(prepared, gates, folder, module)
    generated = (folder / "lib.rs").read_text(encoding="utf-8")
    prefix = generated[: generated.index("// Benchmark-only wrapper")]
    helper = (HERE / "specialized_rhs_native.rs").read_text(encoding="utf-8")
    imports = helper[: helper.index("struct BenchRhs")]
    from confirm_prepared_hamiltonian_integration import WRAPPER as GENERIC_WRAPPER

    tail = helper[helper.index("#[pyfunction]") :]
    tail = tail.replace("N*N", "rhs.dim()").replace("GATE_SLOTS", "&rhs.gates")
    tail = tail.replace("=&rhs.gates.iter()", "=rhs.gates.iter()")
    generated = (
        prefix
        + imports
        + GENERIC_WRAPPER
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
    (folder / "lib.rs").write_text(generated, encoding="utf-8")
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["CARGO_TARGET_DIR"] = str(CACHE / "target_independent")
    log_path = folder / "build.log"
    with log_path.open("w", encoding="utf-8") as log:
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
        raise RuntimeError(log_path.read_text(encoding="utf-8")[-6000:])
    import importlib.util
    import shutil
    import sysconfig

    destination = folder / (module + sysconfig.get_config_var("EXT_SUFFIX"))
    shutil.copyfile(CACHE / "target_independent/release" / (module + ".dll"), destination)
    spec = importlib.util.spec_from_file_location(module, destination)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="starting",
        rounds=ROUNDS,
        platform=platform.platform(),
        python=sys.version,
        source_sha256=source_hashes(),
        atol=ATOL,
        rtol=RTOL,
        strict_atol=STRICT_ATOL,
        strict_rtol=STRICT_RTOL,
        notes=[
            "10 us E=200 V/cm, B=0.01 G numerical performance fixtures; not apparatus transit predictions.",
            "RC uses 250 ns polarization segments, matching the benchmark helper; EOM remains at absolute time.",
            "All density coordinates retained in trajectory comparisons; closure is structural and threshold-free.",
        ],
    )
    save_progress(report)

    with threadpool_limits(limits=1):
        case = rc_driver.build_case(rc_driver.CONFIGS["nominal"])
        mixtures, rc_supports = _initial_j_mixtures(case)
        rc_prepared, rc_gates = _prepare_rc_static_gate_plan(case)
        if rc_prepared.layout.n != len(case.system.QN):
            raise RuntimeError("RC static-plan basis size does not match its source QN list")
        report["rc_density_closure"] = exact_density_closure(
            rc_prepared, rc_supports, ["uniform X,J=1", "uniform X,J=2", "uniform X,J=3"]
        )

        r2_prepared, r2_rho0, _ = base.model("r2")
        r2_support = np.flatnonzero(np.diag(r2_rho0).real > 0).tolist()
        report["r2_density_closure"] = exact_density_closure(
            r2_prepared, [r2_support], ["R2 fixture initial X population"]
        )
        r2_seed_indices = r2_support[:3]
        if len(r2_seed_indices) != 3:
            raise RuntimeError("R(2) fixture needs at least three supported X initial states")
        r2_mixtures = []
        for index in r2_seed_indices:
            rho = np.zeros_like(r2_rho0)
            rho[index, index] = 1.0
            r2_mixtures.append(rho)
        report["r2_seed_density_closures"] = exact_density_closure(
            r2_prepared,
            [[index] for index in r2_seed_indices],
            [f"R2 pure X basis seed {index}" for index in r2_seed_indices],
        )
        save_progress(dict(status="structural closure complete", report=report))

        for tag, prepared, batch, names in (
            (
                "rc",
                rc_prepared,
                mixtures,
                ["uniform X,J=1", "uniform X,J=2", "uniform X,J=3"],
            ),
            (
                "r2",
                r2_prepared,
                r2_mixtures,
                [f"pure X basis seed {index}" for index in r2_seed_indices],
            ),
        ):
            system_gates = rc_gates if tag == "rc" else []
            extension = build_shared_native(prepared, system_gates, tag)
            control_extension = build_independent_control_native(prepared, system_gates, tag)
            report[tag + "_shared_extension"] = str(CACHE / ("shared_native_" + tag))
            report[tag + "_unpatched_control_extension"] = str(
                CACHE / ("independent_control_native_" + tag)
            )
            report[tag + "_mixture_timing"] = _run_mixture_benchmark(
                tag.upper(),
                prepared,
                system_gates,
                batch,
                names,
                extension,
                control_extension,
            )
            save_progress(dict(status=f"{tag} trajectory benchmark complete", report=report))
    report["status"] = "complete"
    save_progress(report)
    (RESULTS / "shared_mixtures_sectors.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_report(report)
    print(json.dumps(report, indent=2), flush=True)


def write_report(report: dict) -> None:
    rc_closure = report["rc_density_closure"]
    r2_closure = report["r2_density_closure"]
    r2_seed_closure = report["r2_seed_density_closures"]
    timings = report["rc_mixture_timing"]
    lines = [
        "# Shared RC mixture and exact density closure checks",
        "",
        "The RC trajectory timing uses three independently initialized, uniform X-state J blocks under the same parameters. Each candidate evolves and returns all three full complex density matrices. The fixture spans 10 µs at E=200 V/cm and B=0.01 G; it is a numerical performance fixture, not a molecular transit prediction.",
        "",
        "## Structural density-coordinate closure",
        "",
        "The threshold-free structural closure is a safe upper bound on reachable support: it uses the union of all static and lowered Hamiltonian basis terms and all nonzero decay transfers. Hamiltonian couplings move either density index; population decay transfers move only diagonal coordinates. Any coordinate outside this closure remains exactly zero. The reported removable fraction is the portion the structure proves can be omitted; the graph may conservatively retain coordinates that cancel dynamically.",
        "",
        "| System / initial support | Coordinates in safe closure | Full coordinates | Proven removable |",
        "|---|---:|---:|---:|",
    ]
    for system_name, closure in (("RC", rc_closure), ("R(2), uniform fixture input", r2_closure),
                                 ("R(2), timing seeds", r2_seed_closure)):
        for mixture in closure["mixtures"]:
            lines.append(
                f"| {system_name}: {mixture['mixture']} | "
                f"{mixture['reachable_real_density_coordinates']:,} | "
                f"{mixture['full_real_density_coordinates']:,} | "
                f"{100*mixture['removable_fraction']:.2f}% |"
            )
    lines += [
        "",
        "## RC trajectory timing",
        "",
        "The unpatched and same-time-cached serial controls call the same native trajectory helper for each mixture. Shared lockstep calls one native solver with the three density blocks and reuses each stage H. A separate existing batch-API run with three workers reports practical parallel throughput. Shared step acceptance uses the maximum per-mixture RMS normalized error.",
        "",
        "| Candidate | Median wall time (s) |",
        "|---|---:|",
    ]
    for label, value in timings["candidate_medians_seconds"].items():
        lines.append(f"| {label} | {value:.6f} |")
    work = timings["rhs_work_summary"]
    lines += [
        "",
        "The shared candidate’s solver `rhs_calls` are outer stage calls, each of which evaluates three density RHSs. For this reason it performs "
        f"{work['shared_lockstep_stage_calls_median']:,.0f} outer stage calls but "
        f"{work['shared_lockstep_density_rhs_applications_median']:,.0f} density RHS applications, versus "
        f"{work['independent_serial_density_rhs_applications_median']:,.0f} density RHS applications in the three independent serial trajectories ({work['shared_vs_serial_density_rhs_applications_percent']:+.2f}% for shared). The stage-call count alone must not be read as total density-RHS work.",
    ]
    lines += [
        "",
        "Speed changes vs the unpatched native serial baseline: "
        + ", ".join(f"{key} {value:+.2f}%" for key, value in timings["reductions_vs_independent_serial_percent"].items())
        + ".",
        "",
        "Every timed density was compared entry-by-entry with the stricter tolerance result (1% relative error for entries at or above 1e-6 magnitude; 1e-6 absolute below that), and its trace was checked. The raw paired rounds and per-mixture error checks are in `shared_mixtures_sectors.json`.",
        "",
        "R(2) receives a corresponding three-seed full-density timing, using three distinct X basis states supported by the fixture. The R(2) closure is shown for both its original uniform X initial state and those benchmark seeds. See the raw JSON for timings and checks.",
        "",
        "## R(2) trajectory timing",
        "",
        "| Candidate | Median wall time (s) |",
        "|---|---:|",
    ]
    for label, value in report["r2_mixture_timing"]["candidate_medians_seconds"].items():
        lines.append(f"| {label} | {value:.6f} |")
    lines += [
        "",
        "## Paired timing reductions",
        "",
        "Percent change is computed within each round as 100×(1−candidate/reference); positive values mean the first candidate is faster. Intervals are nonparametric percentile bootstraps over the eight paired rounds and are descriptive given the small sample.",
        "",
        "| System | Comparison | Median paired reduction | 95% bootstrap interval |",
        "|---|---|---:|---:|",
    ]
    for system_name in ("rc", "r2"):
        paired = report[system_name + "_mixture_timing"]["paired_reductions"]
        for comparison, values in paired.items():
            label = comparison.replace("_vs_", " vs ").replace("_", " ")
            lo, hi = values["bootstrap_95_percentile_ci_percent"]
            lines.append(
                f"| {system_name.upper()} | {label} | {values['median_reduction_percent']:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] |"
            )
    lines += [
        "",
        "## Solver stage calls and density RHS work",
        "",
        "| System | Independent serial stage calls (= density RHSs) | Shared outer stage calls | Shared density RHS applications (3 per stage) | Change in density RHS applications |",
        "|---|---:|---:|---:|---:|",
    ]
    for system_name in ("rc", "r2"):
        work = report[system_name + "_mixture_timing"]["rhs_work_summary"]
        lines.append(
            f"| {system_name.upper()} | {work['independent_serial_density_rhs_applications_median']:,.0f} | "
            f"{work['shared_lockstep_stage_calls_median']:,.0f} | "
            f"{work['shared_lockstep_density_rhs_applications_median']:,.0f} | "
            f"{work['shared_vs_serial_density_rhs_applications_percent']:+.2f}% |"
        )
    lines += [
        "",
        "The same-time cache avoids rebuilding the Hamiltonian on the second and third mixture at a shared stage time, but it still evaluates the complete dissipative and Hamiltonian density RHS separately for each mixture. Shared lockstep therefore makes more total density RHS applications in these fixtures and has no measured wall-time advantage over independent serial integration.",
        "",
        "The maximum observed entry-relative error above the 1e-6 magnitude cutoff was "
        f"{max(x['max_relative'] for x in [c for rows in timings['rounds'].values() for r in rows for c in r['density_checks']]):.4%} for RC and "
        f"{max(x['max_relative'] for rows in report['r2_mixture_timing']['rounds'].values() for r in rows for x in r['density_checks']):.4%} for R(2). Maximum absolute error below the cutoff was "
        f"{max(x['max_small_absolute'] for rows in timings['rounds'].values() for r in rows for x in r['density_checks']):.3e} for RC and "
        f"{max(x['max_small_absolute'] for rows in report['r2_mixture_timing']['rounds'].values() for r in rows for x in r['density_checks']):.3e} for R(2); maximum trace error was "
        f"{max(x['trace_error'] for rows in timings['rounds'].values() for r in rows for x in r['density_checks']):.3e} for RC and "
        f"{max(x['trace_error'] for rows in report['r2_mixture_timing']['rounds'].values() for r in rows for x in r['density_checks']):.3e} for R(2).",
        "",
        "See the raw JSON for every paired round, per-mixture density checks, trace results and step statistics.",
        "",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
