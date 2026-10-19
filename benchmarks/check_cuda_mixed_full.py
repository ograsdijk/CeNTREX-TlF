"""Full-transit density/coherence checks for the mixed-arithmetic prototypes."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import check_cuda_remaining as bench
import numpy as np
import sympy as smp

from centrex_tlf import hamiltonian, lindblad

cp=bench.cp


def solve(variant, prepared, initial, span, **kwargs):
    def compiler(*args,**kw):
        kw["code"]=bench.transform(kw["code"],prepared.layout.n,variant)
        return bench.RawModule(*args,**kw)

    cp.RawModule=compiler
    try:
        return lindblad.solve_lindblad_batch(prepared,initial,span,
            solver="fixed_tsit5",evolution_device="cuda",output="full",maxiters=1000000,
            cuda_options=lindblad.CudaSolverOptions(batch_size=len(initial),
                rhs="direct" if "direct" in variant else "tiled"),**kwargs)
    finally:
        cp.RawModule=bench.RawModule


def main():
    bench.OUT.mkdir(exist_ok=True)
    results=[]
    variants=["base_tiled","fp32_product_tiled","fp32_product_direct","fp32_rhs_direct"]
    for system in ["RC","R2_EOM"]:
        if system=="RC":
            from run_rc_gpu_integrator_suite import PREVIOUS

            context,_,_=bench.previous.fixtures.load()
            prepared=context["prepared"]
            det,ids=bench.previous.fixtures.points(9,"full")
            initial=context["rho0"][ids]
            end=context["physics"]["uv_length_m"]/150
            sigma=np.sqrt(2)*context["sigma_I"]/150
            names=["δ0","center","sigma_time","ω1","φ1","ω2","φ2"]
            parameters=np.column_stack((2*np.pi*1e6*det,*[np.full(9,x) for x in
                [end/2,sigma,2*np.pi*1e6,0.,2*np.pi*1e6,np.pi/2]]))
            oldrows=json.loads((PREVIOUS/"results.json").read_text())
            old=next(x for x in oldrows if x["job"].get("kind")=="full" and x["job"].get("velocity")==150.
                     and x["job"].get("method")=="staged_soa_minor" and x["job"].get("substeps")==512)
            refpath=Path(old["output"]).with_suffix(".npz")
            reference=np.load(refpath)["density_matrix"]
            darkprepared=prepared
            darknames=[*names,"Ωt0","mw_peak1","mw_peak2"]
            darkparameters=np.column_stack((parameters,np.zeros((9,3))))
            dt=250e-9/56
        else:
            prepared,rhos,_,ids,parameters,names,_,dt,_,_=bench.previous.r2_fixture()
            initial=rhos[ids]
            end=.02/150
            H=smp.diag(*prepared.obe_system.H_symbolic.diagonal())
            darkprepared=lindblad.prepare_lindblad_problem(SimpleNamespace(H_symbolic=H,
                C_array=prepared.dense_c_array),{"detuning":0.})
            darknames=names
            darkparameters=parameters
            refpath=bench.ROOT/"gpu_next_results/v1_20261012_174509/052_r2_full_reference_tsit5_b60_h0.5_th256_tile32_wave0.npz"
            reference=np.load(refpath)["rho"]
        for variant in variants:
            print(f"Full transit {system}: {variant}",flush=True)
            tick=time.perf_counter()
            light=solve(variant,prepared,initial,(0.,end),dt=dt,
                        parameter_slots=names,parameter_batch=parameters)
            dark=solve(variant,darkprepared,light.values,(end,end+25/float(hamiltonian.Γ)),dt=dt,
                       parameter_slots=darknames,parameter_batch=darkparameters)
            matrices=np.stack((prepared.layout.unpack_batch(light.values),prepared.layout.unpack_batch(dark.values)),axis=1)
            errors,passed=bench.clean_error(matrices.reshape(-1,prepared.layout.n,prepared.layout.n),
                                           reference.reshape(-1,prepared.layout.n,prepared.layout.n))
            row=dict(system=system,variant=variant,velocity=150.,end_seconds=end,
                     batch=len(initial),elapsed_seconds=time.perf_counter()-tick,
                     accuracy=errors,passed=passed,reference=str(refpath))
            results.append(row)
            (bench.OUT/"mixed_full.json").write_text(json.dumps(results,indent=2))
            print(f"  {row['elapsed_seconds']:.3f}s pass={passed}; relative={errors['max_relative_above_cutoff']:.5g}; trace={errors['trace_error']:.3g}",flush=True)


if __name__=="__main__":
    main()
