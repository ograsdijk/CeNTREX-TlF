"""Isolated archived R2 full-density GPU reuse and ODE experiments."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'benchmarks/r2_f4_dense_threading_results/point_0_inputs.npz'
os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'benchmarks/.cupy_cache'))
import argparse,json,time,sys,hashlib,shutil,threading
from datetime import datetime,timezone
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
import cupy as cp
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout
import bench_r2_f4_dense_improvements as old

def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)

def load_factor(out):
    source=INPUT
    a,structure=old.reduce(dict(np.load(source)))
    _,stages,f=old.real_factor(a['A'],a['reduced_packed'],a['reduced_excited'],np.array([0.,1e-6]))
    f['feed']=a['sink_feed']@f['basis']
    np.savez(out/'factors.npz',**{k:v for k,v in f.items() if k!='real_basis'},keep=a['keep'],sinks=a['sinks'],n=a['n'])
    return a,f,{'source':str(source),'structure':structure,'factor_stages':stages}

def modal(f,times,initial,xp,inverse=None):
    w,C=f['w'],f['C']
    z=w[:,None]*times[None,:]
    e=xp.exp(z)
    safe=xp.where(abs(w)>0,w,1.)[:,None]
    p=xp.where(abs(z)<1e-7,times[None,:]*(1+z/2+z*z/6),xp.expm1(z)/safe)
    def one(mult):
        if inverse is not None:mult=mult[:,inverse]
        m=xp.empty((len(w),len(initial)),dtype=xp.float64)
        r,pos,neg=f['real'],f['positive'],f['negative']
        m[r]=mult[r].real*C[r[:,None],initial[None,:]]
        pair=mult[pos]*(C[pos[:,None],initial[None,:]]-1j*C[neg[:,None],initial[None,:]])
        m[pos]=pair.real;m[neg]=-pair.imag
        return m
    return one(e),one(p)

def project(f,a,times,initial,chunk,xp,complex_output=False):
    n=int(a['n']);full=np.empty((n*n,len(times)),dtype=complex if complex_output else float)
    if complex_output:ui,uj=[xp.asarray(x) for x in np.triu_indices(n,1)];di=xp.arange(n)
    for start in range(0,len(times),chunk):
        sl=slice(start,start+chunk)
        unique,inverse=np.unique(times[sl],return_inverse=True)
        ts=xp.asarray(unique);idx=xp.asarray(initial[sl])
        m,integral=modal(f,ts,idx,xp,xp.asarray(inverse))
        state=f['basis']@m;feed=f['feed']@integral
        block=xp.zeros((int(a['n'])**2,len(idx)),dtype=xp.float64)
        block[xp.asarray(a['keep'])]=state;block[xp.asarray(a['sinks'])]=feed
        if complex_output:
            matrix=xp.empty((n,n,len(idx)),dtype=xp.complex128)
            matrix[di,di]=block[:n]
            upper=block[n::2]+1j*block[n+1::2]
            matrix[ui,uj]=upper;matrix[uj,ui]=upper.conj()
            block=matrix.reshape(n*n,len(idx))
        full[:,sl]=cp.asnumpy(block) if xp is cp else block
    return full

def error(candidate,reference,layout):
    # Complex entry errors, rather than independent errors on Re and Im.
    def matrices(x):
        return x.T.reshape(-1,layout.n,layout.n) if np.iscomplexobj(x) else np.array([layout.unpack(v) for v in x.T])
    c=matrices(candidate);r=matrices(reference)
    d=abs(c-r);mask=abs(r)>=1e-6
    return {'max_complex_relative_above_1e_6':float((d[mask]/abs(r[mask])).max(initial=0)),
            'max_absolute_below_1e_6':float(d[~mask].max(initial=0)),
            'max_absolute':float(d.max()),'max_trace_error':float(abs(np.trace(c,axis1=1,axis2=2)-1).max()),
            'min_density_eigenvalue':float(np.linalg.eigvalsh(c).min())}

def dense(out,complex_output=False):
    with threadpool_limits(limits=1):a,f,info=load_factor(out)
    layout=PackedHermitianLayout(int(a['n']))
    tick=time.perf_counter();gf={k:cp.asarray(v) for k,v in f.items()};cp.cuda.get_current_stream().synchronize()
    info['upload_factors_s']=time.perf_counter()-tick
    result={'kind':'cached static R2 F4 full packed density, all coherences and sink populations',
            'velocity_definition':'Uniform static 20 mm interaction, duration=0.02/v; v=50..250 m/s. No generator changes.',
            'preparation':info,'output_format':'complex128 matrices' if complex_output else 'lossless real packed density','rows':[]}
    save(out/'dense_results.json',result)
    # Warm both paths; full output transfer and allocation included in every timing.
    t=.02/np.array([50.,150.,250.]);idx=np.array([0,10,19])
    with threadpool_limits(limits=1):ref=project(f,a,t,idx,128,np,complex_output)
    check=project(gf,a,t,idx,128,cp,complex_output)
    result['warmup_accuracy']=error(check,ref,layout)
    for nv in [1,21,101,501]:
        velocities=np.array([150.]) if nv==1 else np.linspace(50.,250.,nv)
        times=np.repeat(.02/velocities,20);initial=np.tile(np.arange(20),nv)
        reference=None
        configurations=[('cpu',threads,chunk) for threads in [1,8] for chunk in [128,512,2048,8192]]
        configurations += [('cuda',0,chunk) for chunk in [128,512,2048,8192]]
        for backend,threads,chunk in configurations:
            samples=[]
            for repeat in range(3):
                with threadpool_limits(limits=threads or 1):
                    cp.cuda.get_current_stream().synchronize();tick=time.perf_counter()
                    values=project(gf if backend=='cuda' else f,a,times,initial,chunk,cp if backend=='cuda' else np,complex_output)
                    wall=time.perf_counter()-tick
                samples.append(wall)
            if reference is None:reference=values.copy()
            probes=np.unique(np.linspace(0,len(times)-1,min(60,len(times)),dtype=int))
            diag=error(values[:,probes],reference[:,probes],layout)
            max_abs=float(abs(values-reference).max())
            assert max_abs<1e-7 and diag['max_trace_error']<1e-7
            row={'backend':backend,'cpu_blas_threads':threads,'chunk':chunk,'velocities':nv,'initials':20,
                 'combinations':len(times),'seconds':samples,'median_seconds':float(np.median(samples)),
                 'all_output_max_abs_vs_cpu':max_abs,'sampled_complex_diagnostics':diag}
            result['rows'].append(row);save(out/'dense_results.json',result)
            print(f'DENSE {backend} threads={threads} chunk={chunk} cases={len(times)} median={np.median(samples):.4f}s',flush=True)
    return result

def ode_sweep(out,long=False):
    with threadpool_limits(limits=1):a,f,info=load_factor(out)
    context,fixture,residual=ode_fixture(a)
    from rc_gpu_storage_followup import make_gpu
    layout=context['prepared'].layout
    result={'kind':'R2 fixed-step size and longer-time confirmation; archived static 41-level F4 fixture',
            'rhs_relative_residual':float(residual),'rows':[]}
    if long:
        cases=[(128,'tsit5','swap_const_exact',4.,20e-6),(128,'if_tsit5','exact',4.,1e-6)]
        cases += [(20,'tsit5','swap_const_exact',h,400e-6) for h in [1.,2.,4.]]
    else:
        cases=[(128,'tsit5','swap_const_exact',h,1e-6) for h in [.5,1.,2.,4.,8.]]
        cases += [(128,'if_tsit5','exact',h,1e-6) for h in [5.,6.,8.,12.]]
        cases += [(512,'tsit5',v,1.,1e-6) for v in ['base','swap_const_exact','swap_const_exact_packed','swap_const_exact_packed_direct']]
        cases += [(128,'tsit5','swap_const_exact',.5,20e-6),(128,'if_tsit5','exact',4.,20e-6)]
    for batch,integrator,variant,h,duration in cases:
        print(f'SWEEP B={batch} {integrator} {variant} {h}ns {duration*1e6:g}us',flush=True)
        g=make_gpu(context,batch,0.,duration,dt_ns=h,integrator=integrator,variant=variant,chunk=16)
        g.capture();values,timings,_=g.evolve(repeats=3 if duration==1e-6 else 1)
        packed=np.column_stack([layout.pack(x) for x in values])
        with threadpool_limits(limits=1):ref=project(f,a,np.full(batch,duration),np.arange(batch)%20,512,np)
        if np.isfinite(packed).all():diag=error(packed,ref,layout)
        else:diag={'max_complex_relative_above_1e_6':float('inf'),'max_absolute_below_1e_6':float('inf'),'max_trace_error':float('inf')}
        passed=diag['max_complex_relative_above_1e_6']<=.01 and diag['max_absolute_below_1e_6']<=1e-6 and diag['max_trace_error']<1e-6
        row={'batch':batch,'integrator':integrator,'variant':variant,'dt_ns':h,'duration_s':duration,
             'seconds':timings,'median_seconds':float(np.median(timings)),'accuracy':diag,'passed':passed}
        result['rows'].append(row);save(out/'ode_sweep_results.json',result)
        print(f'SWEEP {np.median(timings):.4f}s pass={passed} complex_error={diag["max_complex_relative_above_1e_6"]:.5g}',flush=True)
        del g;cp.get_default_memory_pool().free_all_blocks()
    return result

def eom(out):
    import rc_gpu_sparse_methods as m
    from rc_gpu_storage_followup import make_gpu
    with threadpool_limits(limits=1):a,f,info=load_factor(out)
    context,fixture,residual=ode_fixture(a);layout=context['prepared'].layout
    fixture['mask']=a['detuning_mask'].copy()
    fixture['base']['β0']=3.8;fixture['base']['ωphase0']=2*np.pi*1.6e6
    m.points=lambda count,pattern='grid':(np.resize(np.repeat([-20.,0.,20.],20),count),np.arange(count)%20)
    archive=ROOT/'benchmarks/rc_propagation_results/cross_system_eom_screen_v1/suite_v2/r2_f4'
    cpu=dict(np.load(archive/'reference.npz'))
    metadata=json.loads((INPUT.parent/'metadata.json').read_text())
    main=metadata['ground_indices'].index(metadata['initial_index']);cpu_index=20+main
    result={'scope':'R2 F4 EOM beta 3.8, 1.6 MHz, 125ns absolute entry, 1us duration; 60 detuning/initial cases repeated to fill batch',
            'cpu_reference':str(archive/'reference.npz'),'rows':[]}
    fine=None
    for variant,h in [('swap_const_exact',.5),('base',1.),('swap_const_exact',1.),
                      ('swap_const_exact_packed',1.),('swap_const_exact_packed_direct',1.),
                      ('swap_const_exact_packed',2.),('swap_const_exact_packed',4.)]:
        batch=512
        print(f'EOM B={batch} {variant} {h}ns',flush=True)
        g=make_gpu(context,batch,125e-9,1.125e-6,dt_ns=h,integrator='tsit5',variant=variant,chunk=16)
        g.capture();values,timings,_=g.evolve(repeats=3)
        packed=np.column_stack([layout.pack(x) for x in values])
        if fine is None:fine=packed.copy()
        diag=error(packed,fine,layout)
        cpu_diag=error(packed[:,cpu_index:cpu_index+1],layout.pack(cpu['dopri5_full'])[:,None],layout)
        passed=all(d['max_complex_relative_above_1e_6']<=.01 and d['max_absolute_below_1e_6']<=1e-6 and d['max_trace_error']<1e-6 for d in [diag,cpu_diag])
        result['rows'].append({'variant':variant,'dt_ns':h,'batch':batch,'seconds':timings,'median_seconds':float(np.median(timings)),
                               'fine_gpu_diagnostics':diag,'archived_cpu_main_diagnostics':cpu_diag,'passed':passed})
        save(out/'eom_results.json',result)
        np.savez_compressed(out/f'eom_{variant}_{h:g}ns.npz',rho=values[:60],fine_packed=fine[:,:60])
        print(f'EOM {np.median(timings):.4f}s pass={passed} error={diag["max_complex_relative_above_1e_6"]:.5g}, CPU={cpu_diag["max_complex_relative_above_1e_6"]:.5g}',flush=True)
        del g;cp.get_default_memory_pool().free_all_blocks()
    return result

def eom_long(out):
    import rc_gpu_sparse_methods as m
    from rc_gpu_storage_followup import make_gpu
    with threadpool_limits(limits=1):a,f,info=load_factor(out)
    context,fixture,residual=ode_fixture(a);layout=context['prepared'].layout
    fixture['mask']=a['detuning_mask'].copy()
    fixture['base']['β0']=3.8;fixture['base']['ωphase0']=2*np.pi*1.6e6
    m.points=lambda count,pattern='grid':(np.repeat([-20.,0.,20.],20),np.tile(np.arange(20),3))
    result={'scope':'R2 EOM beta3.8 1.6MHz, 60 distinct cases; endpoint duration 20mm/150m/s then 20mm/50m/s',
            'rows':[]}
    for duration in [.02/150.,.02/50.]:
        fine=None
        for h in [.5,1.,2.]:
            print(f'EOM LONG starting {duration*1e6:.3f}us h={h}ns B=60',flush=True)
            g=make_gpu(context,60,125e-9,125e-9+duration,dt_ns=h,integrator='tsit5',variant='swap_const_exact_packed',chunk=16)
            g.capture();values,timings,_=g.evolve(repeats=1)
            packed=np.column_stack([layout.pack(x) for x in values])
            if fine is None:fine=packed.copy()
            diag=error(packed,fine,layout)
            passed=diag['max_complex_relative_above_1e_6']<=.01 and diag['max_absolute_below_1e_6']<=1e-6 and diag['max_trace_error']<1e-6
            result['rows'].append({'duration_s':duration,'dt_ns':h,'seconds':timings,'accuracy':diag,'passed':passed})
            save(out/'eom_long_results.json',result)
            np.savez_compressed(out/f'eom_long_{duration:.8g}_{h:g}ns.npz',rho=values,reference_packed=fine)
            print(f'EOM LONG {timings[0]:.3f}s pass={passed} complex_error={diag["max_complex_relative_above_1e_6"]:.6g}',flush=True)
            del g;cp.get_default_memory_pool().free_all_blocks()
    return result

def matched_scan(out):
    import rc_gpu_sparse_methods as m
    from rc_gpu_storage_followup import make_gpu
    with threadpool_limits(limits=1):a,f,info=load_factor(out)
    context,fixture,residual=ode_fixture(a);n=int(a['n']);layout=context['prepared'].layout
    fixture['mask']=a['detuning_mask'].copy()
    rho=np.zeros((n,n,n),complex);rho[np.arange(n),np.arange(n),np.arange(n)]=1.
    context['rho0']=rho
    archive=ROOT/'benchmarks/rc_propagation_results/cross_system_parallel_dense_v1/suite_v1/r2_f4'
    previous=json.loads((archive/'result.json').read_text());reference=np.load(archive/'r_01.npz')['rho']
    points=previous['points'];groups={}
    for index,point in enumerate(points):groups.setdefault(point['optical_rabi_over_gamma'],[]).append(index)
    cpurows=[r for r in previous['rows'] if r.get('product')=='full' and r.get('workers')==8 and r.get('batch')=='K40_all_model_states']
    result={'scope':'Exact frozen CPU workload: 8 parameter points, all 40 one-hot populations, 1 us, full final density',
            'cpu_source':str(archive/'result.json'),'cpu_rows':cpurows,'rows':[]}
    for h in [.5]:
      for variant in ['swap_const_exact','swap_const_exact_packed','swap_const_exact_packed_direct']:
        instances=[]
        for scale,indices in groups.items():
            fixture['base']['Ωt0']=scale
            det=np.repeat([points[i]['detuning_mhz'] for i in indices],n);initial=np.tile(np.arange(n),len(indices))
            m.points=lambda count,pattern='grid',d=det,j=initial:(d.copy(),j.copy())
            g=make_gpu(context,len(det),0.,1e-6,dt_ns=h,integrator='tsit5',variant=variant,chunk=16)
            g.capture();instances.append((g,indices))
        seconds=[];diags=[]
        for repeat in range(3):
            cp.cuda.get_current_stream().synchronize();tick=time.perf_counter();outputs=[]
            for g,indices in instances:
                g.reset();g.launch_to(0.,1e-6);values=g.download()
                outputs.append((values,indices))
            seconds.append(time.perf_counter()-tick)
            for values,indices in outputs:
                ref=reference[indices].reshape(-1,n,n)
                diag=error(np.column_stack([layout.pack(x) for x in values]),np.column_stack([layout.pack(x) for x in ref]),layout)
                diags.append(diag)
            if repeat==2:
                np.savez_compressed(out/f'matched_{variant}.npz',
                    rho=np.concatenate([v for v,_ in outputs]),reference=np.concatenate([reference[idx].reshape(-1,n,n) for _,idx in outputs]),
                    point_indices=np.concatenate([np.repeat(idx,n) for _,idx in outputs]))
        passed=all(d['max_complex_relative_above_1e_6']<=.01 and d['max_absolute_below_1e_6']<=1e-6 and d['max_trace_error']<1e-6 for d in diags)
        result['rows'].append({'variant':variant,'dt_ns':h,'seconds':seconds,'median_seconds':float(np.median(seconds)),
                               'passed':passed,'diagnostics':diags})
        save(out/'matched_scan_results.json',result)
        print(f'MATCHED {variant}: {np.median(seconds):.4f}s pass={passed} max_error={max(d["max_complex_relative_above_1e_6"] for d in diags):.5g}',flush=True)
        instances.clear();del g;cp.get_default_memory_pool().free_all_blocks()
    return result

def ode_fixture(a):
    n=int(a['n']);layout=PackedHermitianLayout(n);L=a['L']
    H=np.zeros((n,n),complex)
    for j in range(n):H[:,j]=1j*layout.unpack(L[:,j])[:,j]
    for i in range(1,n):H[i,i]=L[layout.upper_imag_index(0,i),layout.upper_real_index(0,i)]
    H[0,0]=0.
    assert np.max(abs(H-H.conj().T))<1e-4
    rates=-np.diag(L)[:n].copy();feed=L[:n,:n].copy();np.fill_diagonal(feed,0.)
    assert feed.min()>=0.
    tp=[0];tc=[];tr=[]
    for row in feed:
        cols=np.flatnonzero(row);tc.extend(cols);tr.extend(row[cols]);tp.append(len(tc))
    basis=np.zeros((5,n,n),complex);basis[4]=H-np.diag(np.diag(H))
    support=abs(basis[4])>0;rows,cols=np.nonzero(support)
    f={'basis':basis,'diagonal':H.diagonal().real.copy(),'mask':np.zeros(n),'rows':rows.astype(np.int32),
       'columns':cols.astype(np.int32),'ptr':np.r_[0,np.cumsum(support.sum(axis=1))].astype(np.int32),
       'tp':np.array(tp,np.int32),'tc':np.array(tc,np.int32),'tr':np.array(tr), 'rates':rates,
       'base':{'mw_peak1':0.,'mw_peak2':0.,'PUVZ0':1.,'Ωt0':1.,'β0':0.,'ωphase0':0.}}
    rho=np.array([layout.unpack(x) for x in a['packed'].T])
    context={'prepared':SimpleNamespace(layout=layout),'rho0':rho,'physics':{'uv_length_m':.02,'post_decay_lifetimes':25},'sigma_I':1.}
    import rc_gpu_sparse_methods as m
    rng=np.random.default_rng(1729);z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));probe=z@z.conj().T;probe/=np.trace(probe)
    rhs=m.numpy_rhs(f,context,probe[None],np.zeros(1),0.)[0]
    residual=abs(layout.pack(rhs)-L@layout.pack(probe)).max()/abs(L@layout.pack(probe)).max()
    assert residual<1e-12,residual
    m.fixture=lambda _:f
    m.points=lambda count,pattern='grid':(np.zeros(count),np.arange(count)%len(rho))
    return context,f,residual

def ode(out):
    with threadpool_limits(limits=1):a,f,info=load_factor(out)
    context,fixture,residual=ode_fixture(a)
    from rc_gpu_storage_followup import make_gpu
    result={'kind':'archived static R2 F4, 41 levels, all 20 resolved initial populations',
            'structured_rhs_relative_residual':float(residual),'max_diagonal_rad_s':float(abs(fixture['diagonal']).max()),'rows':[]}
    layout=context['prepared'].layout
    # Explicit step stability is bounded by far-detuned retained parity levels.
    h_ns=min(.25,2./float(np.max(abs(f['w'])))*1e9)
    cases=[('tsit5','base',h_ns),('tsit5','swap_const_exact',h_ns),
           ('tsit5','swap_const_exact_packed',h_ns),('tsit5','swap_const_exact_packed_direct',h_ns),
           ('if_tsit5','exact',.25),('if_tsit5','exact',1.),('if_tsit5','exact',4.)]
    for batch in [128,512]:
        for integrator,variant,h in cases:
            # A short pilot first; long explicit runs only when the pilot is affordable.
            for duration in [20e-9,1e-6]:
                print(f'ODE starting B={batch} {integrator} {variant} h={h:.6g}ns span={duration*1e6:g}us',flush=True)
                tick=time.perf_counter()
                g=make_gpu(context,batch,0.,duration,dt_ns=h,integrator=integrator,variant=variant,chunk=16)
                g.capture();setup=time.perf_counter()-tick
                values,timings,_=g.evolve(repeats=1 if duration<1e-6 else 3)
                packed=np.column_stack([layout.pack(x) for x in values])
                with threadpool_limits(limits=1):reference=project(f,a,np.full(batch,duration),np.arange(batch)%20,512,np)
                diag=error(packed,reference,layout)
                passed=diag['max_complex_relative_above_1e_6']<=.01 and diag['max_absolute_below_1e_6']<=1e-6 and diag['max_trace_error']<1e-6
                row={'batch':batch,'integrator':integrator,'variant':variant,'dt_ns':h,'duration_s':duration,
                     'setup_s':setup,'seconds':timings,'median_seconds':float(np.median(timings)),'accuracy':diag,'passed':passed}
                result['rows'].append(row);save(out/'ode_results.json',result)
                print(f'ODE complete {np.median(timings):.3f}s pass={passed} error={diag["max_complex_relative_above_1e_6"]:.5g}',flush=True)
                np.savez_compressed(out/f'ode_{batch}_{integrator}_{variant}_{duration:.0e}.npz',rho=values[:20],reference=reference[:,:20])
                del g;cp.get_default_memory_pool().free_all_blocks()
                if duration<1e-6 and (not passed or timings[0]*50>150):break
    return result

if __name__=='__main__':
    deadline=datetime(2026,10,12,16,37,10,tzinfo=timezone.utc)
    remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    if remaining<=0:raise SystemExit('Inherited GPU benchmark deadline has passed')
    watchdog=threading.Timer(remaining,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['dense','dense_complex','ode','ode_sweep','ode_long','matched_scan','eom','eom_long'],required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--input',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.input is not None:INPUT=args.input
    sources=args.out/'source_snapshot';sources.mkdir(exist_ok=True)
    names=['run_r2_gpu_reuse.py','prepare_r2_gpu_cross_fixture.py','rc_gpu_sparse_methods.py','rc_gpu_integrators.py',
           'rc_gpu_fixed_followup.py','rc_gpu_storage_followup.py','rc_gpu_sparse_kernels.cu','rc_gpu_integrators.cu','rc_gpu_packed.cu','rc_gpu_integrating_factor.cu']
    hashes={}
    for name in names:
        path=ROOT/'benchmarks'/name;shutil.copyfile(path,sources/name);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(args.out/'provenance.json',{'mode':args.mode,'input':str(INPUT),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                                   'sources_sha256':hashes,'cupy_version':cp.__version__,'numpy_version':np.__version__,
                                   'device':str(cp.cuda.runtime.getDeviceProperties(0)['name'])})
    {'dense':dense,'dense_complex':lambda out:dense(out,True),'ode':ode,'ode_sweep':ode_sweep,'ode_long':lambda out:ode_sweep(out,True),
     'matched_scan':matched_scan,'eom':eom,'eom_long':eom_long}[args.mode](args.out)
