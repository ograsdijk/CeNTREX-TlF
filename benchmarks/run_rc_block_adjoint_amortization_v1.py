"""Staged K=20/K=70 amortization study for benchmark-only RC block/adjoint methods.

The checker owns all physical execution. Importing this module performs no
simulation. Every stage is isolated, guarded, and writes new immutable artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from centrex_tlf.lindblad.batch import solve_lindblad_batch

import bench_rc_propagation as driver
import rc_block_adjoint_prototypes as proto
import run_rc_block_adjoint_pilot_v2 as pilot

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "rc_propagation_results" / "block_adjoint_amortization_v1"
CPU_LIMIT = 50.0
STAGE_TIMEOUT = 900.0
RTOL, ATOL, DT = 1e-9, 1e-11, 1e-10
STAGES = ("reference", "block_K20", "block_K70", "rust_full_K20", "rust_full_K70",
          "adjoint_photons", "adjoint_X_J1", "scalar_K20", "scalar_K70")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdefault(x: Any) -> Any:
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    return str(x)


def checkpoint(path: Path, value: dict[str, Any]) -> None:
    if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=jdefault), encoding="utf-8")
    tmp.replace(path)


def snapshot() -> dict[str, Any]:
    s = pilot.comparison.resource_snapshot()
    current = psutil.Process()
    ancestors = {current.pid, *(p.pid for p in current.parents())}
    runner_pids = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid in ancestors or "python" not in (proc.info["name"] or "").lower(): continue
            if any(Path(a).name.lower() == Path(__file__).name.lower()
                   for a in (proc.info.get("cmdline") or []) if isinstance(a, str)):
                runner_pids.append(proc.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess): pass
    s["known_amortization_runner_pids"] = runner_pids
    return s


def guard(s: dict[str, Any]) -> dict[str, Any]:
    samples = np.asarray(s.get("cpu_percent_samples", []), dtype=float)
    median = float(np.median(samples)) if samples.size else None
    okay = bool(samples.size and median <= CPU_LIMIT and not s.get("known_simulation_pids")
                and not s.get("known_amortization_runner_pids"))
    return {"okay": okay, "cpu_median_limit_percent": CPU_LIMIT,
            "cpu_median_percent": median, "reason": None if okay else "resource guard failed"}


def state_label(qn: Any, index: int) -> str:
    q = qn.largest
    return (f"{index}:X{getattr(q.electronic_state, 'name', q.electronic_state)}"
            f"_J{q.J:g}_F1{q.F1:g}_F{q.F:g}_mF{q.mF:g}_P{q.P}_Omega{q.Omega:g}")


def initial_sets(case: driver.RCCase) -> tuple[np.ndarray, list[int], list[str], np.ndarray, list[str]]:
    n = case.n_states
    qns = case.system.QN
    if n != 70: raise ValueError(f"expected the established 70-state RC model, got {n}")
    per_j: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for i, qn in enumerate(qns):
        q = qn.largest
        if getattr(q.electronic_state, "name", str(q.electronic_state)) == "X" and int(q.J) in per_j:
            per_j[int(q.J)].append(i)
    # Deterministic round-robin across sorted X J=1/2/3 labels, sampling each
    # manifold evenly while preserving a spread across all three rotational blocks.
    counts = {1: 7, 2: 7, 3: 6}
    selected: list[int] = []
    group_labels: dict[int, list[int]] = {}
    anchor_names = ("X_J1_F1_3o2_F1_mF0", "X_J2_F1_5o2_F2_mF0", "X_J3_F1_7o2_F3_mF0")
    anchors = {j: int(case.state_labels[name]) for j, name in zip((1, 2, 3), anchor_names, strict=True)}
    for j, count in counts.items():
        group = per_j[j]
        group.sort(key=lambda i: (float(qns[i].largest.F), float(qns[i].largest.F1),
                                  float(qns[i].largest.mF), int(qns[i].largest.P), i))
        if len(group) < count: raise ValueError(f"not enough X J={j} states for K20 selection")
        if anchors[j] not in group: raise ValueError(f"pilot anchor for J={j} absent from X manifold")
        remaining = [i for i in group if i != anchors[j]]
        extra_count = count - 1
        picks = np.linspace(0, len(remaining)-1, extra_count, dtype=int).tolist()
        group_labels[j] = [anchors[j], *[remaining[k] for k in picks]]
    for row in range(7):
        for j in (1, 2, 3):
            if row < len(group_labels[j]): selected.append(group_labels[j][row])
    if len(selected) != 20 or len(set(selected)) != 20: raise AssertionError("K20 selection invalid")
    matrices = np.zeros((n, n, n), dtype=np.complex128)
    idx = np.arange(n)
    matrices[idx, idx, idx] = 1.0
    labels = [state_label(qns[i], i) for i in selected]
    all_labels = [state_label(qn, i) for i, qn in enumerate(qns)]
    return matrices, selected, labels, np.arange(n, dtype=int).tolist(), all_labels


def span(case: driver.RCCase) -> tuple[float, float]:
    lo = float(case.config.entry_phase_s)
    return lo, lo + float(case.config.duration_s)


def parameters(case: driver.RCCase, lo: float, hi: float, count: int) -> np.ndarray:
    from bench_rc_propagation import SWITCH_PHASE_OFFSET_RAD
    rows=[]
    for a,b in pairwise(driver.switch_boundaries(lo,hi)):
        sw=driver.switch_values((float(a)+float(b))*0.5)
        row=np.asarray([0.0, SWITCH_PHASE_OFFSET_RAD if sw[0] else -SWITCH_PHASE_OFFSET_RAD,
                        0.0, SWITCH_PHASE_OFFSET_RAD if sw[1] else -SWITCH_PHASE_OFFSET_RAD], dtype=np.complex128)
        rows.append((float(a),float(b),np.tile(row,(count,1))))
    return rows


def rust_batch(case: driver.RCCase, rho: np.ndarray, interval_rows: list[Any], *,
               observable: str, solver: str, reltol: float, abstol: float, dt: float,
               parallel: bool, threads: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Segmented Rust batch; scalar products carry full density between switches."""
    rho=np.asarray(rho,dtype=np.complex128).copy()
    if rho.ndim==2: rho=rho[None]
    accum=np.zeros(len(rho),dtype=float)
    j1=pilot.observables._group_indices(case)["X_J1"]
    weights=[(int(i),float(driver.GAMMA)) for i,q in enumerate(case.system.QN)
             if getattr(q.largest.electronic_state,'name',str(q.largest.electronic_state))=='B']
    rows=[]
    for si,(lo,hi,pbatch) in enumerate(interval_rows):
        pbatch=np.tile(np.asarray(pbatch)[0],(len(rho),1))
        final=si==len(interval_rows)-1
        if observable=="full": output="full"
        elif observable=="photons": output="full_and_integral"
        elif observable=="X_J1": output="selected" if final else "full"
        else: raise ValueError(observable)
        kw={}
        if observable=="photons": kw.update(integral_weights=weights, integral_method="solver")
        if observable=="X_J1" and final: kw["output_indices"]=[(int(i),int(i)) for i in j1]
        t=time.perf_counter()
        out=solve_lindblad_batch(case.prepared_segmented,rho,(lo,hi),parameter_batch=pbatch,
            parameter_slots=case.switch_override_slots,solver=solver,execution_mode="structured",
            reltol=reltol,abstol=abstol,dt=dt,output=output,output_when="final",save_start=False,
            collect_stats=True,parallel=parallel,threads=threads,**kw)
        vals=np.asarray(out.values)
        if observable=="full": packed=vals
        elif observable=="photons":
            packed=vals[:,:-1]; accum+=vals[:,-1].real
        elif observable=="X_J1" and final:
            imag=float(np.max(np.abs(vals.imag)))
            if imag>1e-9 or not np.all(np.isfinite(vals)):
                raise RuntimeError(f"selected X J=1 output invalid: max imaginary residual {imag:g}")
            accum=vals.real.sum(axis=1)
            rows.append({"lo_s":lo,"hi_s":hi,"wall_seconds_including_python_ffi":time.perf_counter()-t,
                         "solver_stats":out.solver_stats,"batch_size":len(rho),"output":output,
                         "max_imaginary_residual":imag})
            break
        else: packed=vals
        rho=np.stack([case.prepared_dynamic.layout.unpack(np.asarray(v,dtype=float)) for v in packed])
        rows.append({"lo_s":lo,"hi_s":hi,"wall_seconds_including_python_ffi":time.perf_counter()-t,
                     "solver_stats":out.solver_stats,"batch_size":len(rho),"output":output})
    if observable=="full": return rho,{"segments":rows}
    return accum,{"segments":rows,"observable":observable}


def full_metrics(rho: np.ndarray, ref: np.ndarray) -> list[dict[str, float]]:
    ret=[]
    for a,b in zip(rho,ref,strict=True):
        pa=np.diag(a).real; pb=np.diag(b).real
        populated=pb>=1e-6
        rel=float(np.max(np.abs((pa[populated]-pb[populated])/pb[populated]))) if populated.any() else 0.0
        herm=float(np.max(np.abs(a-a.conj().T)))
        tr=complex(np.trace(a)); eig=float(np.linalg.eigvalsh((a+a.conj().T)/2).min())
        ret.append({"density_abs":float(np.max(np.abs(a-b))),"population_abs":float(np.max(np.abs(pa-pb))),
                    "population_rel_populated":rel,"trace_abs_error":abs(tr-1),"hermiticity_abs":herm,
                    "minimum_eigenvalue":eig})
    return ret


def observable_metrics(actual: np.ndarray, ref: np.ndarray, *, photons: bool) -> list[dict[str,float]]:
    rows=[]
    for a,b in zip(np.asarray(actual,float),np.asarray(ref,float),strict=True):
        err=abs(a-b); rel=err/abs(b) if abs(b)>=1e-6 else None
        rows.append({"value":float(a),"reference":float(b),"absolute_error":float(err),
                     "relative_error_if_reference_ge_1e-6":rel})
    return rows


def provenance(case: driver.RCCase) -> dict[str,Any]:
    import centrex_tlf.centrex_tlf_rust as rust
    import scipy, numba
    base=pilot.observables._current_provenance(case)
    return {**base,"runner_sha256":sha(Path(__file__)),"pilot_helper_sha256":sha(Path(pilot.__file__)),
            "prototype_sha256":sha(Path(proto.__file__)),"driver_sha256":sha(Path(driver.__file__)),
            "batch_api_sha256":sha(Path(solve_lindblad_batch.__code__.co_filename)),
            "native_sha256":sha(Path(rust.__file__)),"case":asdict(case.config),
            "scipy":scipy.__version__,"numba":numba.__version__,
            "python":sys.version,"platform":platform.platform(),"n_states":case.n_states,
            "solver_timed":{"rust":"tsit5 structured","prototype":"DOP853","rtol":RTOL,"atol":ATOL,"dt_s":DT},
            "cpu_gate_median_percent":CPU_LIMIT,"parallel_thread_counts":[3,8]}


def load_ref(path: Path, expected: dict[str,Any]) -> dict[str,np.ndarray]:
    meta=json.loads(path.parent.joinpath("reference.json").read_text(encoding="utf-8"))
    npz=path
    if meta.get("status")!="complete" or meta.get("accuracy_accepted") is not True:
        raise ValueError("reference is incomplete or failed strict acceptance")
    if sha(npz)!=meta.get("npz_sha256"): raise ValueError("reference NPZ checksum mismatch")
    normalize=lambda x: json.loads(json.dumps(x,sort_keys=True,default=jdefault))
    if normalize(meta.get("provenance")) != normalize(expected): raise ValueError("reference provenance mismatch")
    with np.load(npz) as data: return {k:np.asarray(data[k]) for k in data.files}


def run_stage(case_name: str, stage: str, outdir: Path) -> dict[str,Any]:
    if stage not in STAGES: raise ValueError(stage)
    dest=outdir/f"{stage}.json"
    if dest.exists(): raise FileExistsError(dest)
    before=snapshot(); g0=guard(before)
    if not g0["okay"]:
        payload={"status":"paused_resource_guard","stage":stage,"timings_accepted":False,
                 "resource_before":before,"guard_before":g0}
        checkpoint(dest,payload); return payload
    cfg=replace(proto.authorized_case_configs()[case_name],duration_s=1e-6,entry_phase_s=137e-9)
    t0=time.perf_counter(); case=driver.build_case(cfg,prepare_frozen=True); setup=time.perf_counter()-t0
    switch_t=time.perf_counter(); driver._prepare_switch_plans(case,static=False)
    switch_setup=time.perf_counter()-switch_t
    lo,hi=span(case); intervals=parameters(case,lo,hi,70)
    identity, k20, labels20, k70, labels70=initial_sets(case)
    # K70 identity inputs include every model basis state, including B and sink states.
    k70rho=identity
    k20rho=identity[k20]
    prov=provenance(case)
    base={"status":"complete","stage":stage,"case":asdict(cfg),"case_setup_seconds":setup,
          "span_s":[lo,hi],"switch_plan_setup_seconds":switch_setup,
          "K20_indices":k20,"K20_labels":labels20,"K70_indices":k70,
          "K70_labels":labels70,"resource_before":before,"guard_before":g0,"provenance":prov,
          "full_density_scope":"all 70 model states; one-hot inputs include X, B, and sinks"}
    refnpz=outdir/"strict_reference.npz"
    if stage=="reference":
        # Strict full-density reference on all 70 one-hots; independently cross-check solvers.
        if refnpz.exists(): raise FileExistsError(f"refusing to overwrite {refnpz}")
        full_d, full_stats=rust_batch(case,k70rho,intervals,observable="full",solver="dopri5",
            reltol=2e-11,abstol=2e-13,dt=5e-11,parallel=True,threads=3)
        cross,cross_stats=rust_batch(case,k70rho,intervals,observable="full",solver="tsit5",
            reltol=2e-11,abstol=2e-13,dt=5e-11,parallel=True,threads=3)
        checks=full_metrics(cross,full_d)
        fullpass=all(x["density_abs"]<=1e-8 and x["population_rel_populated"]<=1e-4 and
                     x["trace_abs_error"]<=1e-9 and x["hermiticity_abs"]<=1e-9 and
                     x["minimum_eigenvalue"]>=-1e-8 for x in checks)
        photons,phstats=rust_batch(case,k70rho,intervals,observable="photons",solver="dopri5",
            reltol=2e-11,abstol=2e-13,dt=5e-11,parallel=True,threads=3)
        photon_cross,pcstats=rust_batch(case,k70rho,intervals,observable="photons",solver="tsit5",
            reltol=2e-11,abstol=2e-13,dt=5e-11,parallel=True,threads=3)
        p_err=np.abs(photons-photon_cross)
        ppass=bool(np.all(p_err<=1e-6+1e-5*np.abs(photons)))
        acc=bool(fullpass and ppass)
        np.savez_compressed(refnpz,full_densities=full_d,photon_yields=photons,
                            K20_indices=np.asarray(k20),K20_labels=np.asarray(labels20),K70_labels=np.asarray(labels70))
        base.update({"full_density_crosscheck_per_state":checks,"photon_crosscheck_errors":p_err.tolist(),
                     "dopri5_full_stats":full_stats,"tsit5_full_stats":cross_stats,
                     "dopri5_photon_stats":phstats,"tsit5_photon_stats":pcstats,
                     "npz":refnpz.name,"npz_sha256":sha(refnpz),"accuracy_accepted":acc,
                     "timings_accepted":True,"interpretation":"strict reference only; no performance claim"})
    else:
        ref=load_ref(refnpz,prov)
        r70=ref["full_densities"]; rp70=ref["photon_yields"]
        if stage.startswith("block_K"):
            K=20 if stage.endswith("20") else 70; inp=k20rho if K==20 else k70rho
            jit_s=proto.warm_numba_kernels()
            t=time.perf_counter(); actual,stats=proto.integrate_segmented_block(case,inp,(lo,hi),solver="DOP853",reltol=RTOL,abstol=ATOL,dt=DT)
            wall=time.perf_counter()-t
            refd=r70[k20] if K==20 else r70
            checks=full_metrics(actual,refd)
            accepted=all(x["density_abs"]<=1e-7 and x["population_abs"]<=1e-7 and
                         x["population_rel_populated"]<=1e-3 and x["trace_abs_error"]<=1e-9 and
                         x["hermiticity_abs"]<=1e-9 and x["minimum_eigenvalue"]>=-1e-8 for x in checks)
            npz=outdir/f"{stage}.npz"; np.savez_compressed(npz,full_densities=actual)
            base.update({"K":K,"method":"shared adaptive controller, full final density","numba_warmup_seconds":jit_s,
                         "warm_amortized_wall_seconds":wall,
                         "first_use_method_total_seconds":jit_s+switch_setup+wall,
                         "first_use_total_including_case_build_seconds":setup+jit_s+switch_setup+wall,
                         "prototype_stats":stats,"per_state_accuracy":checks,
                         "accuracy_accepted":accepted,"timings_accepted":wall<=STAGE_TIMEOUT,
                         "npz":npz.name,"npz_sha256":sha(npz)})
        elif stage.startswith("rust_full_K"):
            K=20 if stage.endswith("20") else 70; inp=k20rho if K==20 else k70rho
            allrows=[]
            for threads in (1,3,8):
                par=threads>1; t=time.perf_counter()
                actual,stats=rust_batch(case,inp,parameters(case,lo,hi,K),observable="full",solver="tsit5",
                    reltol=RTOL,abstol=ATOL,dt=DT,parallel=par,threads=threads)
                wall=time.perf_counter()-t; reference=r70[k20] if K==20 else r70
                checks=full_metrics(actual,reference)
                accepted=all(x["density_abs"]<=1e-7 and x["population_abs"]<=1e-7 and
                             x["population_rel_populated"]<=1e-3 and x["trace_abs_error"]<=1e-9 and
                             x["hermiticity_abs"]<=1e-9 and x["minimum_eigenvalue"]>=-1e-8 for x in checks)
                allrows.append({"threads":threads,"parallel":par,"batch_size":K,"wall_seconds":wall,
                                "per_state_accuracy":checks,"accuracy_accepted":accepted,"stats":stats})
            base.update({"K":K,"trials":allrows,"accuracy_accepted":all(x["accuracy_accepted"] for x in allrows),
                         "timings_accepted":True})
        elif stage in ("adjoint_photons","adjoint_X_J1"):
            jit_s=proto.warm_numba_kernels()
            objective=proto.photon_objective(case,driver.GAMMA) if stage=="adjoint_photons" else proto.j1_population_objective(case)
            t=time.perf_counter(); lam,stats=proto.integrate_segmented_adjoint(case,objective,(lo,hi),solver="DOP853",reltol=RTOL,abstol=ATOL,dt=DT)
            wall=time.perf_counter()-t; vals=proto.diagonal_objective_responses(lam)
            diagonal_imag_max=float(np.max(np.abs(np.diag(lam).imag)))
            strict=rp70 if stage=="adjoint_photons" else np.asarray([np.diag(x).real[pilot.observables._group_indices(case)["X_J1"]].sum() for x in r70])
            err=np.abs(vals-strict); ifphot=stage=="adjoint_photons"
            if ifphot:
                passed=bool(np.all(err<=1e-6+1e-5*np.abs(strict)))
            else:
                populated=strict>=1e-6
                passed=bool(np.all(err<=1e-7) and np.all(err[populated]/strict[populated]<=1e-3))
            passed=bool(passed and diagonal_imag_max<=1e-9 and np.all(np.isfinite(vals)))
            npz=outdir/f"{stage}.npz"; np.savez_compressed(npz,all70_adjoint_responses=vals)
            base.update({"observable":objective.name,"numba_warmup_seconds":jit_s,
                         "warm_amortized_single_adjoint_wall_seconds":wall,
                         "first_use_method_total_seconds":jit_s+switch_setup+wall,
                         "first_use_total_including_case_build_seconds":setup+jit_s+switch_setup+wall,
                         "prototype_stats":stats,"all70_error_rows":observable_metrics(vals,strict,photons=ifphot),
                         "diagonal_imaginary_max_abs":diagonal_imag_max,
                         "accuracy_accepted":passed,"timings_accepted":wall<=STAGE_TIMEOUT,"npz":npz.name,"npz_sha256":sha(npz),
                         "reuse_for_K20":True,"adjoint_solves_charged_once":True})
        elif stage.startswith("scalar_K"):
            K=20 if stage.endswith("20") else 70; inp=k20rho if K==20 else k70rho
            results={}; allpass=True
            for obs in ("photons","X_J1"):
                repeats=[]
                for threads in (1,3,8):
                    par=threads>1; t=time.perf_counter()
                    actual,stats=rust_batch(case,inp,parameters(case,lo,hi,K),observable=obs,solver="tsit5",
                        reltol=RTOL,abstol=ATOL,dt=DT,parallel=par,threads=threads)
                    wall=time.perf_counter()-t
                    strict=(rp70[k20] if K==20 else rp70) if obs=="photons" else np.asarray([
                        np.diag(x).real[pilot.observables._group_indices(case)["X_J1"]].sum() for x in (r70[k20] if K==20 else r70)])
                    errs=observable_metrics(actual,strict,photons=obs=="photons")
                    if obs=="photons":
                        passed=all(x["absolute_error"]<=1e-6+1e-5*abs(x["reference"]) for x in errs)
                    else:
                        passed=all(x["absolute_error"]<=1e-7 and
                                   (x["reference"]<1e-6 or x["relative_error_if_reference_ge_1e-6"]<=1e-3)
                                   for x in errs)
                    allpass &= passed
                    repeats.append({"threads":threads,"parallel":par,"batch_size":K,"wall_seconds":wall,
                                    "values":np.asarray(actual).tolist(),"per_state_accuracy":errs,"accuracy_accepted":passed,"stats":stats})
                results[obs]=repeats
            base.update({"K":K,"timed_products":results,"accuracy_accepted":bool(allpass),"timings_accepted":True,
                         "timing_contract":"K independent initial states only; no mixture solve or validation inside timed wall"})
    after=snapshot(); g1=guard(after)
    base.update({"resource_after":after,"guard_after":g1,"timings_accepted":bool(base.get("timings_accepted",True) and g1["okay"]),
                 "whole_stage_elapsed_seconds":time.perf_counter()-t0})
    if stage!="reference": base["timings_accepted"]=bool(base["timings_accepted"] and base.get("accuracy_accepted",False))
    checkpoint(dest,base); return base


def worker(case: str, stage: str, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1): run_stage(case,stage,output)


def supervise(case: str, tag: str, timeout: float) -> None:
    target=(RESULTS/tag/case).resolve()
    if RESULTS.resolve() not in target.parents: raise ValueError("unsafe output path")
    target.mkdir(parents=True,exist_ok=False)
    statuses={}
    for stage in STAGES:
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",stage,"--case",case,"--output",str(target)]
        try:
            p=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True,timeout=timeout)
            if p.returncode: checkpoint(target/f"{stage}.failure.json",{"status":"stage_failed","rc":p.returncode,"stderr":p.stderr[-4000:]})
        except subprocess.TimeoutExpired:
            checkpoint(target/f"{stage}.timeout.json",{"status":"timed_out_unmeasured","timeout_seconds":timeout,
                "interpretation":"incomplete stage, not a correctness failure"})
        result=target/f"{stage}.json"
        statuses[stage]=json.loads(result.read_text()) if result.exists() else {"status":"timed_out_or_failed"}
    summary={"case":case,"stages":{k:{"status":v.get("status"),"timings_accepted":v.get("timings_accepted"),
               "accuracy_accepted":v.get("accuracy_accepted")} for k,v in statuses.items()},
             "status":"complete" if all(v.get("status")=="complete" and
                   v.get("timings_accepted") is True and v.get("accuracy_accepted") is True
                   for v in statuses.values()) else "partial",
             "timeouts_are_unmeasured_not_correctness_failures":True}
    checkpoint(target/"supervision.json",summary)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=("nominal","detuned_20MHz","optical_3Gamma"),default="nominal")
    p.add_argument("--tag"); p.add_argument("--timeout-seconds",type=float,default=STAGE_TIMEOUT)
    p.add_argument("--worker",choices=STAGES); p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.worker:
        if a.output is None: p.error("--worker requires --output")
        target=a.output.resolve()
        if RESULTS.resolve() not in target.parents: raise ValueError("worker path must be under results")
        worker(a.case,a.worker,target)
    else:
        if not a.tag: p.error("--tag required for supervised run")
        supervise(a.case,a.tag,a.timeout_seconds)


if __name__=="__main__": main()
