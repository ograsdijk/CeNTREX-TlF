"""Prototype fixed-shape CUDA graph/buffer reuse across detuning batches."""
from __future__ import annotations

import argparse
import gc
import json
import time

import check_cuda_remaining as bench
import numpy as np

from centrex_tlf import lindblad

cp = bench.cp


def reused_solve(g, parameters, slots, span, output):
    # Fixed shape, support/varying pattern, time grid and discontinuities.
    # Replace all scalar overrides, regenerate common waveform tables, reset
    # density/clock/error, and retain all graph/buffer addresses.
    cp.cuda.get_current_stream().synchronize()
    tick = time.perf_counter()
    full = g.session_parameters.copy()
    full[slots] = parameters.T
    g.params.set(np.ascontiguousarray(full))
    cp.copyto(g.y, g.initial)
    g.clock.set(np.array([span[0], 0., 0., 0., 0., 0.]))
    g.error.fill(0)
    if g.precompute:
        g.launch("precompute", (g.params,g.grid,g.table,np.int64(g.steps),
                               np.int32(g.K),np.int32(g.B),g.error),g.steps*7)
    g.advance(0,g.steps)
    dummy = g.session_dummy
    g.launch("sample", (g.y,output,g.public_pk,g.public_im,dummy,dummy,g.weights,
                        np.int32(g.B),np.int32(g.n),np.int32(g.n*g.n),np.int32(0),np.int32(0)),g.B*g.n*g.n)
    values = output.get()
    return values, time.perf_counter()-tick


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",type=int,default=512)
    parser.add_argument("--duration-us",type=float,default=1.)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--methods",nargs="+",choices=["fresh","fresh_packed","reused"],
                        default=["fresh","fresh_packed","reused"])
    args=parser.parse_args()
    if "reused" not in args.methods or not {"fresh","fresh_packed"}.intersection(args.methods):
        parser.error("methods must include reused and at least one fresh baseline")
    if not np.isfinite(args.duration_us) or args.duration_us<=0 or args.repeats<1:
        parser.error("duration must be finite and positive; repeats must be positive")
    bench.OUT.mkdir(exist_ok=True)
    results=[]
    rng=np.random.default_rng(162)
    for system,factory in [("RC",bench.previous.rc_fixture),("R2_EOM",bench.previous.r2_fixture)]:
        prepared,rhos,_,ids,pars0,names,span,dt,_,_=factory()
        span=(span[0],span[0]+args.duration_us*1e-6)
        select=np.arange(args.batch)%len(ids)
        initial,parameters=rhos[ids[select]],pars0[select]
        packed_initial=prepared.layout.pack_batch(initial)
        slots=[prepared.parameter_graph["slot_names"].index(name) for name in names]
        print(f"Preparing {system}, {args.duration_us:g} us, batch {args.batch}",flush=True)
        g,_,_=bench.setup(prepared,packed_initial,parameters,names,span,dt,"base_tiled")
        g.session_parameters=g.params.get()
        g.session_dummy=cp.empty(0,cp.int32)
        output=cp.empty((g.B,g.n*g.n),cp.float64)
        rows=[]
        for offset in [0.125,-0.25,0.5]:
            current=parameters.copy()
            current[:,0]+=2*np.pi*1e6*offset
            values={}
            timings={method:[] for method in args.methods}
            maximum=0.
            for repeat in range(args.repeats):
                for method in rng.permutation(args.methods):
                    if method!="reused":
                        tick=time.perf_counter()
                        answer=lindblad.solve_lindblad_batch(prepared,initial if method=="fresh" else packed_initial,span,
                            parameter_batch=current,parameter_slots=names,solver="fixed_tsit5",
                            evolution_device="cuda",dt=dt,output="full",collect_stats=True,
                            maxiters=1000000,cuda_options=lindblad.CudaSolverOptions(batch_size=g.B))
                        timings[method].append(time.perf_counter()-tick)
                        values[method]=answer.values
                    else:
                        values[method],elapsed=reused_solve(g,current,slots,span,output)
                        timings[method].append(elapsed)
                for method in args.methods:
                    np.testing.assert_allclose(values[method],values["reused"],atol=1e-12,rtol=1e-11)
                    maximum=max(maximum,float(np.max(abs(values[method]-values["reused"]))))
                print(f"{system} detuning offset {offset:+g}: round {repeat+1}/{args.repeats}",flush=True)
            rows.append(dict(offset_MHz=offset,timings=timings,
                             max_packed_difference=maximum))
        results.append(dict(system=system,batch=g.B,span=span,rows=rows,
                            scope="Fixed shapes/support/varying pattern/grid; changed detunings; all waveform tables regenerated"))
        suffix="" if args.duration_us==1. else f"_{args.duration_us:g}us"
        (bench.OUT/f"session_reuse_b{args.batch}{suffix}.json").write_text(json.dumps(results,indent=2))
        print({r["offset_MHz"]:{k:float(np.median(v)) for k,v in r["timings"].items()} for r in rows},flush=True)
        del g,output
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


if __name__=="__main__":
    main()
