"""Pinned output overlap and static photon-only projections; no ODE CPU reruns."""
import json,time
from pathlib import Path
import numpy as np
import cupy as cp
from threadpoolctl import threadpool_limits
from run_gpu_next_suite import fixtures,atomic
from rc_gpu_next_followup import NextGPU
import run_r2_gpu_reuse as dense

def pinned(shape,dtype):
    dtype=np.dtype(dtype);memory=cp.cuda.alloc_pinned_memory(int(np.prod(shape))*dtype.itemsize)
    return np.frombuffer(memory,dtype=dtype,count=int(np.prod(shape))).reshape(shape)

def ode_pipeline(job):
    context,source,excited,start,end,ref,_,refsource=fixtures(job)
    out=Path(job['output']);batch=job['batch'];n=context['prepared'].layout.n;chunks=4
    g=NextGPU(context,batch,start,end,dt_ns=job['dt_ns'],variant=job['variant'],
              threads=job.get('threads',256),tile=job.get('tile',0),waveform=job.get('waveform',False))
    g.capture();compute=cp.cuda.Stream(non_blocking=True);copy=cp.cuda.Stream(non_blocking=True)
    device=[cp.empty((batch,n,n),cp.complex128) for _ in range(2)]
    host=[pinned((batch,n,n),np.complex128) for _ in range(chunks)]
    ready=[cp.cuda.Event() for _ in range(chunks)];done=[cp.cuda.Event() for _ in range(2)]
    result={'job':job,'status':'complete','chunks':chunks,'rows':[]}
    baseline=None
    for overlap in [False,True]:
        samples=[]
        for repeat in range(3):
            cp.cuda.get_current_stream().synchronize();compute.synchronize();copy.synchronize();tick=time.perf_counter()
            for k in range(chunks):
                slot=k%2
                with compute:
                    g.reset();g.launch_to(start,end)
                    if overlap and k>=2:compute.wait_event(done[slot])
                    g.launch('unpack_density',(g.y,device[slot],np.int32(batch),np.int32(n)),batch*n*n)
                    ready[k].record(compute)
                if overlap:
                    copy.wait_event(ready[k])
                    device[slot].get(out=host[k],stream=copy,blocking=False);done[slot].record(copy)
                else:
                    compute.synchronize();device[slot].get(out=host[k],stream=compute,blocking=True)
            compute.synchronize();copy.synchronize();samples.append(time.perf_counter()-tick)
            if baseline is None:baseline=host[0].copy()
            err=max(float(abs(x-baseline).max()) for x in host)
            assert err<1e-12,err
        result['rows'].append({'overlap':overlap,'seconds':samples,'median_s':float(np.median(samples)),
                               'all_output_abs_vs_sequential':err})
        atomic(out,result);print('ODE pipeline',job['system'],overlap,np.median(samples),flush=True)
    result['speedup']=result['rows'][0]['median_s']/result['rows'][1]['median_s'];atomic(out,result)

def dense_block(f,a,times,initial,out):
    unique,inverse=np.unique(times,return_inverse=True);idx=cp.asarray(initial)
    m,integral=dense.modal(f,cp.asarray(unique),idx,cp,cp.asarray(inverse))
    block=cp.zeros((int(a['n'])**2,len(times)),cp.float64)
    block[cp.asarray(a['keep'])]=f['basis']@m
    block[cp.asarray(a['sinks'])]=f['feed']@integral
    n=int(a['n']);matrix=out.reshape(n,n,len(times));di=cp.arange(n);ui,uj=[cp.asarray(x) for x in np.triu_indices(n,1)]
    matrix[di,di]=block[:n];upper=block[n::2]+1j*block[n+1::2]
    matrix[ui,uj]=upper;matrix[uj,ui]=upper.conj()
    return out.reshape(n*n,len(times))

def dense_pipeline(job):
    out=Path(job['output']);source=Path(dense.INPUT)
    a,_=dense.old.reduce(dict(np.load(source)))
    f=dict(np.load(Path(__file__).resolve().parent/'r2_gpu_reuse_results/v2_20261012/factors.npz'))
    keys=['w','C','basis','feed','real','positive','negative'];f={k:f[k] for k in keys}
    gf={k:cp.asarray(v) for k,v in f.items()}
    nv=501;times=np.repeat(.02/np.linspace(50.,250.,nv),20);initial=np.tile(np.arange(20),nv);n=int(a['n'])
    result={'job':job,'status':'complete','kind':'R2 cached eigensystem, full complex matrix outputs','cases':len(times),'rows':[]}
    for chunk in [512,2048]:
        compute=cp.cuda.Stream(non_blocking=True);copy=cp.cuda.Stream(non_blocking=True)
        slices=[slice(i,min(i+chunk,len(times))) for i in range(0,len(times),chunk)]
        host=[pinned((n*n,sl.stop-sl.start),np.complex128) for sl in slices]
        device=[cp.empty(n*n*chunk,cp.complex128) for _ in range(2)]
        ready=[cp.cuda.Event() for _ in slices];done=[cp.cuda.Event() for _ in range(2)]
        # Compile all ufuncs and warm allocations before both timings.
        dense_block(gf,a,times[:20],initial[:20],device[0][:n*n*20]);cp.cuda.get_current_stream().synchronize()
        for overlap in [False,True]:
            samples=[]
            for repeat in range(3):
                compute.synchronize();copy.synchronize();tick=time.perf_counter()
                for k,sl in enumerate(slices):
                    slot=k%2;width=sl.stop-sl.start
                    with compute:
                        if overlap and k>=2:compute.wait_event(done[slot])
                        block=dense_block(gf,a,times[sl],initial[sl],device[slot][:n*n*width])
                        ready[k].record(compute)
                    if overlap:
                        copy.wait_event(ready[k]);block.get(out=host[k],stream=copy,blocking=False);done[slot].record(copy)
                    else:compute.synchronize();block.get(out=host[k],stream=compute,blocking=True)
                compute.synchronize();copy.synchronize();samples.append(time.perf_counter()-tick)
            values=np.concatenate(host,axis=1)
            with threadpool_limits(limits=1):reference=dense.project(f,a,times,initial,512,np,True)
            error=float(abs(values-reference).max());assert error<1e-10
            result['rows'].append({'chunk':chunk,'overlap':overlap,'seconds':samples,'median_s':float(np.median(samples)),
                                   'all_complex_output_abs_vs_cpu':error})
            atomic(out,result);print('Dense pipeline',chunk,overlap,np.median(samples),flush=True)
    atomic(out,result)

def dense_photons(job):
    out=Path(job['output']);folder=Path(__file__).resolve().parent/'r2_gpu_reuse_results/v2_20261012'
    f=dict(np.load(folder/'factors.npz'));a,_=dense.old.reduce(dict(np.load(dense.INPUT)))
    from centrex_tlf import hamiltonian
    obs=float(hamiltonian.Γ)*f['basis'][a['reduced_excited']].sum(axis=0)
    real,pos,neg=[f[k] for k in ['real','positive','negative']]
    w=np.r_[f['w'][real],f['w'][pos]]
    weights=np.concatenate([(obs[real,None]*f['C'][real]).astype(complex),
          (obs[pos]+1j*obs[neg])[:,None]*(f['C'][pos]-1j*f['C'][neg])],axis=0).T
    gw,gweights=cp.asarray(w),cp.asarray(weights)
    result={'job':job,'status':'complete','kind':'R2 cached static photon-only emission integrals','rows':[]}
    def evaluate(times,xp):
        ww,weights_= (gw,gweights) if xp is cp else (w,weights)
        t=xp.asarray(times);z=ww[:,None]*t[None,:]
        p=xp.where(abs(z)<1e-7,t[None,:]*(1+z/2+z*z/6),xp.expm1(z)/xp.where(abs(ww)>0,ww,1.)[:,None])
        values=(weights_@p).real
        return cp.asnumpy(values) if xp is cp else values
    evaluate(np.array([1e-6]),cp)
    for nv in [21,101,501,2001]:
        times=.02/np.linspace(50.,250.,nv);reference=None
        for backend,threads in [('cpu',1),('cpu',8),('cuda',1)]:
            samples=[]
            for _ in range(3):
                with threadpool_limits(limits=threads):
                    cp.cuda.get_current_stream().synchronize();tick=time.perf_counter();values=evaluate(times,cp if backend=='cuda' else np);samples.append(time.perf_counter()-tick)
            if reference is None:reference=values.copy()
            error=float(abs(values-reference).max());assert error<1e-7
            result['rows'].append({'backend':backend,'cpu_blas_threads':threads if backend=='cpu' else 0,
                 'velocities':nv,'initials':20,'cases':20*nv,'seconds':samples,'median_s':float(np.median(samples)),
                 'max_abs_vs_cpu':error})
            atomic(out,result);print('Dense photons',backend,nv,np.median(samples),flush=True)
    atomic(out,result)

def run(job):
    {'ode_pipeline':ode_pipeline,'dense_pipeline':dense_pipeline,'dense_photons':dense_photons}[job['kind']](job)
