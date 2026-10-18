"""Check fixed-shape CUDA reuse on R(0), static and EOM, after packing updates."""
from __future__ import annotations

import gc
import json
import time

import bench_rust_specialized_rhs as models
import check_cuda_remaining as bench
import numpy as np
from check_cuda_session_reuse import reused_solve

from centrex_tlf import lindblad

cp=bench.cp


def main():
    bench.OUT.mkdir(exist_ok=True)
    print("Building R(0) E=200 V/cm, B=0.01 G performance fixture",flush=True)
    prepared,_,_=models.model("r0")
    n=prepared.layout.n
    slots0=prepared.parameter_graph["slot_names"]
    detuning=next(name for name in slots0[:len(prepared.parameter_graph["base_values"])] if name.startswith("δ"))
    names=[detuning,"β0"]
    slots=[slots0.index(name) for name in names]
    indices=[i for i,state in enumerate(prepared.obe_system.QN)
             if state.largest.electronic_state.name=="X" and state.largest.J==0]
    uniform=np.zeros((n,n),complex)
    uniform[indices,indices]=1/len(indices)
    psi=np.zeros(n,complex)
    psi[indices[:2]]=[1/np.sqrt(2),1j/np.sqrt(2)]
    rhos=np.stack((uniform,np.outer(psi,psi.conj())))
    diagonal=np.diag(np.asarray(prepared.hamiltonian_plan["static_matrix"],complex)).real
    # Compact spectators carry large absolute rotational energies but cannot
    # generate coherences with the driven block for these initial states.
    active=np.unique(bench.cuda._basis(prepared)[3])
    dt=min(1e-9,1/max(1.,float(np.ptp(diagonal[active]))+2*np.pi*20e6+4*float(models.driver.GAMMA)))
    rng=np.random.default_rng(928)
    results=[]
    batch=1024
    initial=rhos[np.arange(batch)%2]
    tick=time.perf_counter()
    packed=prepared.layout.pack_batch(initial)
    pack_seconds=time.perf_counter()-tick
    for beta in [0.,3.8]:
        for duration in [1e-6,10e-6]:
            span=(125e-9,125e-9+duration)
            pars=np.column_stack((2*np.pi*1e6*np.resize([-20.,0.,20.],batch),np.full(batch,beta)))
            print(f"R0 n={n}, beta={beta:g}, duration={duration*1e6:g} us, dt={dt*1e9:g} ns",flush=True)
            g,_,_=bench.setup(prepared,packed,pars,names,span,dt,"base_tiled")
            g.session_parameters=g.params.get()
            g.session_dummy=cp.empty(0,cp.int32)
            output=cp.empty((batch,n*n),cp.float64)
            timings={"fresh_packed":[],"reused":[]}
            maximum=0.
            for repeat,offset in enumerate([0.125,-0.25,0.5,0.125,-0.25,0.5]):
                current=pars.copy()
                current[:,0]+=2*np.pi*1e6*offset
                values={}
                for method in rng.permutation(["fresh_packed","reused"]):
                    if method=="fresh_packed":
                        tick=time.perf_counter()
                        answer=lindblad.solve_lindblad_batch(prepared,packed,span,
                            parameter_batch=current,parameter_slots=names,solver="fixed_tsit5",
                            evolution_device="cuda",dt=dt,output="full",collect_stats=True,
                            maxiters=1000000,cuda_options=lindblad.CudaSolverOptions(batch_size=batch))
                        timings[method].append(time.perf_counter()-tick)
                        values[method]=answer.values
                    else:
                        values[method],elapsed=reused_solve(g,current,slots,span,output)
                        timings[method].append(elapsed)
                np.testing.assert_allclose(values["fresh_packed"],values["reused"],atol=1e-12,rtol=1e-11)
                maximum=max(maximum,float(np.max(abs(values["fresh_packed"]-values["reused"]))))
                print(f"  round {repeat+1}/6",flush=True)
            row=dict(system="R0",n=n,batch=batch,beta=beta,omega_phase_rad_s=2*np.pi*1.6e6,
                     span=span,dt=dt,timings=timings,max_packed_difference=maximum,
                     pack_batch_seconds=pack_seconds,
                     trace_error=float(np.max(abs(values["reused"][:,:n].sum(axis=1)-1))))
            results.append(row)
            (bench.OUT/"r0_reuse.json").write_text(json.dumps(results,indent=2))
            print({k:float(np.median(v)) for k,v in timings.items()},flush=True)
            del g,output
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()


if __name__=="__main__":
    main()
