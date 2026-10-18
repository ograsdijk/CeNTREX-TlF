"""RC/R2 implementation experiments; isolated workers, no production writes."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['CUPY_CACHE_DIR']=str(HERE/'.cupy_cache')
import argparse,hashlib,json,subprocess,sys,time,shutil
from datetime import datetime,timezone
import numpy as np

def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for attempt in range(40):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)

def fixtures(job):
    import rc_gpu_sparse_methods as m
    from run_rc_gpu_integrator_suite import reference,PREVIOUS
    full=job.get('kind') in ['full','full_reference']
    if job['system']=='rc':
        context,source,_=m.load();excited=np.flatnonzero(np.asarray(context['prepared'].source_decay_rates)>0)
        if full:
            start=0.;end=context['physics']['uv_length_m']/job.get('velocity',150.)
            oldrows=json.loads((PREVIOUS/'results.json').read_text())
            old=next(r for r in oldrows if r['job'].get('kind')=='full' and r['job'].get('velocity')==job.get('velocity',150.) and
                         r['job'].get('method')=='staged_soa_minor' and r['job'].get('substeps')==512)
            refsource=Path(old['output']).with_suffix('.npz');data=np.load(refsource);ref=data['density_matrix'];popref=data['populations']
        else:
            start=146e-6;end=start+job.get('span_us',5)*1e-6
            ref,refsource=reference(dict(fixture='production',span_us=job.get('span_us',5)));popref=None
        return context,source,excited,start,end,ref,popref,refsource
    import run_r2_gpu_reuse as r2
    source=HERE/'r2_gpu_reuse_results/matched_cross_v1/inputs.npz';a=dict(np.load(source))
    context,f,_=r2.ode_fixture(a);f['mask']=a['detuning_mask'].copy()
    f['base']['β0']=3.8;f['base']['ωphase0']=2*np.pi*1.6e6
    m.points=lambda count,pattern='grid':(np.resize(np.repeat([-20.,0.,20.],20),count),np.arange(count)%20)
    start=0. if full else 125e-9
    end=start+(.02/job.get('velocity',150.) if full else job.get('span_us',1)*1e-6)
    refsource=HERE/'r2_gpu_reuse_results/eom_v1/eom_swap_const_exact_0.5ns.npz'
    ref=np.load(refsource)['rho'];popref=None
    if full:ref=popref=None
    return context,source,a['excited'],start,end,ref,popref,refsource

def passing(result):
    a=result.get('accuracy',{})
    return (result.get('status')=='complete' and a.get('passes_1percent_above_cutoff',False) and
            a.get('near_zero_max_abs',float('inf'))<=1e-6 and a.get('trace_error',float('inf'))<1e-6 and
            a.get('min_eigenvalue',-1)>=-1e-7)

def worker(job):
    if job.get('kind') in ['ode_pipeline','dense_pipeline','dense_photons']:
        from gpu_next_pipeline import run
        return run(job)
    import cupy as cp
    from rc_gpu_next_followup import NextGPU
    from rc_gpu_sparse_methods import diagnostics
    context,source,excited,start,end,ref,popref,refsource=fixtures(job)
    out=Path(job['output']);full=job.get('kind') in ['full','full_reference']
    args=dict(dt_ns=job['dt_ns'],integrator=job.get('integrator','tsit5'),variant=job['variant'],
              chunk=16,pattern='full' if full and job['system']=='rc' else 'grid',velocity=job.get('velocity',150.),
              threads=job.get('threads',256),tile=job.get('tile',0),waveform=job.get('waveform',False),
              photons=job.get('photons',False),excited=excited)
    setup=time.perf_counter();g=NextGPU(context,job['batch'],start,end,**args)
    # Independent dense NumPy derivative, with arbitrary coherent PSD probes.
    rng=np.random.default_rng(1729);z=rng.normal(size=(2,g.n,g.n))+1j*rng.normal(size=(2,g.n,g.n))
    probes=z@z.conj().swapaxes(-1,-2);probes/=np.trace(probes,axis1=-2,axis2=-1)[:,None,None]
    probe=probes[np.arange(g.batch)%2];g.y=g.encode(probe)
    g.clock.set(np.array([start,0.,end,g.dt,0.,0.,0.,0.]))
    g.rhs_new(g.y,g.ks[0],0.)
    got=g.download(g.ks[0])
    # numpy_rhs uses 150 m/s; full velocity checks need the exact envelope.
    import rc_gpu_sparse_methods as m
    H=m.hamiltonian(g.f,context,start,velocity=job.get('velocity',150.))
    Hz=H[None]+2*np.pi*1e6*g.det_host[:,None,None]*np.diag(g.f['mask'])[None]
    wanted=-1j*(Hz@probe-probe@Hz)-.5*(g.f['rates'][:,None]+g.f['rates'][None,:])*probe
    for i in range(g.n):
        for k in range(g.f['tp'][i],g.f['tp'][i+1]):wanted[:,i,i]+=g.f['tr'][k]*probe[:,g.f['tc'][k],g.f['tc'][k]]
    rhs_error=float(np.linalg.norm(got-wanted)/np.linalg.norm(wanted))
    if rhs_error>1e-12:raise ValueError(f'Coherent RHS mismatch {rhs_error}')
    # Restore the original density allocation before graph capture.
    g.y=g.encode(g.initial_host);g.initial=g.y.copy();g.reset()
    if full and g.waveform:
        from centrex_tlf import hamiltonian
        targets=([end] if job.get('output_mode')=='photon_only' else list(np.linspace(start,end,201)[1:]))
        targets.append(end+25/float(hamiltonian.Γ))
        g.prepare_waveform(targets);g.wave_full=True
    g.capture();setup_s=time.perf_counter()-setup
    photon_analytic_abs=None
    if job.get('photons'):
        from centrex_tlf import hamiltonian
        # Independent known solution: one excited level, all drives off.
        check=NextGPU(context,2,0.,1e-6,**dict(args,pattern='grid',waveform=False))
        initial=np.zeros((2,g.n,g.n),complex);initial[:,excited[0],excited[0]]=1.
        check.initial_host=initial;check.initial=check.encode(initial);check.y=check.initial.copy()
        p=check.p.get();p[2:5]=0.;check.p.set(p)
        check.capture();check.reset();check.launch_to(0.,1e-6)
        expected_photons=-np.expm1(-float(hamiltonian.Γ)*1e-6)
        photon_analytic_abs=float(abs(check.photon_integral.get()-expected_photons).max())
        if photon_analytic_abs>1e-7:raise ValueError(f'Analytic photon decay mismatch {photon_analytic_abs}')
        del check
    if job.get('reference'):
        refsource=Path(job['reference']);archive=np.load(refsource);ref=archive['rho'];popref=archive.get('populations')
    timings=[];populations=None;photons=None
    for repeat in range(job.get('repeats',3)):
        g.reset();cp.cuda.get_current_stream().synchronize();tick=time.perf_counter()
        if full:
            if job.get('output_mode')=='photon_only':
                g.launch_to(start,end);at_exit=g.photon_integral.copy()
                p=g.p.get();p[2:5]=0.;g.p.set(p);clock=g.clock.get();clock[4]=0.;g.clock.set(clock)
                from centrex_tlf import hamiltonian
                darkend=end+25/float(hamiltonian.Γ)
                g.launch_to(end,darkend);photons=cp.asnumpy(cp.stack((at_exit,g.photon_integral),axis=1))
            else:
                rho,populations,detail=g.full_transit()
                if g.photons:photons=cp.asnumpy(cp.stack((g.photon_exit,g.photon_integral),axis=1))
        else:
            g.launch_to(start,end)
            if job.get('output_mode')=='photon_only':photons=g.photon_integral.get()
            else:
                rho=g.download()
                if g.photons:photons=g.photon_integral.get()
        clock=g.clock.get();timings.append(time.perf_counter()-tick)
    # Accuracy work is outside output-specific timing, including diagnostic density downloads.
    if job.get('output_mode')=='photon_only':rho=g.download()
    expected=None
    if ref is not None:
        expected=ref[np.arange(len(rho))%len(ref)]
        if job.get('output_mode')=='photon_only' and expected.ndim==4:expected=expected[:,-1]
    else:expected=rho
    if not np.isfinite(rho).all() or np.max(abs(rho))>10.:
        atomic(out,dict(job=job,status='numerical_failure',median_s=float(np.median(timings)),seconds=timings));return
    accuracy=diagnostics(rho.reshape(-1,g.n,g.n),expected.reshape(-1,g.n,g.n))
    record=dict(job=job,status='complete',seconds=timings,median_s=float(np.median(timings)),setup_s=setup_s,
                rhs_relative_error=rhs_error,accuracy=accuracy,reference=str(refsource),
                input=str(source),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                live_device_bytes=cp.get_default_memory_pool().used_bytes(),kernel_threads=g.threads,tile=g.tile,
                accepted_steps=int(clock[6]),final_time=float(clock[0]),device=str(cp.cuda.runtime.getDeviceProperties(0)['name']))
    record['reference_is_self']=ref is None
    record['benchmark_source_sha256']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ['run_gpu_next_suite.py','rc_gpu_next_followup.py','rc_gpu_storage_followup.py','rc_gpu_fixed_followup.py','rc_gpu_integrators.py']}
    if populations is not None and popref is not None:
        record['population_history_max_abs']=float(abs(populations-popref[np.arange(len(populations))%len(popref)]).max())
    if g.photons:
        record['photon_analytic_decay_abs']=photon_analytic_abs
        if photons is None:photons=g.photon_integral.get()
        record['photons_min']=float(np.min(photons));record['photons_max']=float(np.max(photons))
        if job.get('photon_reference'):
            target=np.load(job['photon_reference'])['photons'];target=target[np.arange(len(photons))%len(target)]
            record['photon_max_abs']=float(abs(photons-target).max())
            record['photon_max_relative']=float((abs(photons-target)/np.maximum(abs(target),1e-6)).max())
            record['photon_passes_1percent']=bool(np.all(abs(photons-target)<=np.maximum(.01*abs(target),1e-6)))
        if job['system']=='r2' and not full:
            old=np.load(HERE/'rc_propagation_results/cross_system_eom_screen_v1/suite_v2/r2_f4/reference.npz')
            meta=json.loads((HERE/'r2_gpu_reuse_results/matched_cross_v1/metadata.json').read_text())
            k=20+meta['ground_indices'].index(meta['initial_index'])
            record['archived_cpu_photon_abs']=float(abs(photons[k]-old['dopri5_photon_integral'].ravel()[0]))
    arrays=dict(rho=rho[:min(483,len(rho))],reference=expected[:min(483,len(rho))])
    if populations is not None:arrays['populations']=populations
    if photons is not None:arrays['photons']=photons
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    out.with_suffix('.cu').write_text(g.source,encoding='utf-8');atomic(out,record)
    print(json.dumps(dict(seconds=record['median_s'],entry_error_percent=100*accuracy['max_relative_above_cutoff'],rhs=rhs_error)),flush=True)

class Runner:
    def __init__(self,out,resume=False):
        self.out=out;out.mkdir(parents=True,exist_ok=True)
        self.results=json.loads((out/'results.json').read_text()) if resume else []
        if resume:
            known={r['job']['output'] for r in self.results}
            for path in sorted(out.glob('[0-9][0-9][0-9]_*.json')):
                if path.name.endswith('.job.json'):continue
                data=json.loads(path.read_text())
                if 'job' in data and data['job']['output'] not in known:self.results.append(data);known.add(data['job']['output'])
            atomic(out/'results.json',self.results)
        elapsed=json.loads((out/'progress.json').read_text()).get('elapsed_s',0) if resume else 0
        self.started=time.time()-elapsed;self.stage='setup';self.current=None;self.resumed=resume
        self.snapshot();self.progress()
    def snapshot(self):
        suffix=datetime.now().strftime('%Y%m%d_%H%M%S')
        folder=self.out/('source_snapshot_followup_'+suffix if getattr(self,'resumed',False) else 'source_snapshot');folder.mkdir(exist_ok=True)
        names=['run_gpu_next_suite.py','rc_gpu_next_followup.py','rc_gpu_storage_followup.py','rc_gpu_fixed_followup.py',
               'rc_gpu_integrators.py','rc_gpu_integrators.cu','rc_gpu_sparse_methods.py','rc_gpu_sparse_kernels.cu','rc_gpu_packed.cu',
               'gpu_next_pipeline.py']
        hashes={}
        for name in names:
            p=HERE/name;shutil.copyfile(p,folder/name);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        atomic(self.out/('manifest_followup.json' if getattr(self,'resumed',False) else 'manifest.json'),dict(sources_sha256=hashes,time_cap=None,cpu_odes_rerun=False))
    def progress(self,status='running'):
        atomic(self.out/'progress.json',dict(status=status,stage=self.stage,current_job=self.current,completed_jobs=len(self.results),
               failed_jobs=sum(r.get('status') not in ['complete'] for r in self.results),elapsed_s=time.time()-self.started,runner_pid=os.getpid()))
    def run(self,**kw):
        j=dict(system='rc',kind='short',span_us=5,batch=512,dt_ns=250/56,integrator='tsit5',
               variant='swap_const_schedule_gentle_exact_devicepop_packed_direct',threads=256,tile=0,waveform=False,repeats=3)
        j.update(kw);i=len(self.results)+1
        stem=f"{i:03}_{j['system']}_{j['kind']}_{j['integrator']}_b{j['batch']}_h{j['dt_ns']:.5g}_th{j['threads']}_tile{j['tile']}_wave{int(j['waveform'])}"
        j['output']=str(self.out/(stem+'.json'));jobpath=self.out/(stem+'.job.json');atomic(jobpath,j)
        self.current=j;self.progress();print(stem,flush=True)
        with (self.out/(stem+'.stdout.log')).open('w') as stdout,(self.out/(stem+'.stderr.log')).open('w') as stderr:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(jobpath)],stdout=stdout,stderr=stderr)
            # Per-job hang protection, not a suite time cap; long confirmations may run longer.
            stop=time.time()+j.get('watchdog_seconds',1800)
            while child.poll() is None and time.time()<stop:time.sleep(2);self.progress()
            if child.poll() is None:child.kill();child.wait();r=dict(job=j,status='timeout')
            elif child.returncode:r=dict(job=j,status='failed',stderr=str(self.out/(stem+'.stderr.log')))
            else:r=json.loads(Path(j['output']).read_text())
        self.results.append(r);atomic(self.out/'results.json',self.results);self.current=None;self.progress();self.report()
        print('result',r.get('status'),r.get('median_s'),r.get('accuracy',{}).get('max_relative_above_cutoff'),flush=True)
        return r
    def report(self):
        lines=['# RC and R(2): further GPU techniques','',f'Stage: {self.stage}. No suite time cap. No production changes or CPU ODE reruns.','',
               'Output-specific timings include the requested GPU-to-host transfer; diagnostics and disk writes are separate. Full density/coherence target is 1% for reference magnitudes >=1e-6; near-zero absolute errors are separate.','',
               '| System | Kind | Method | Batch | Step ns | Threads | Tile | Wave table | Seconds | Entry error % | Pass |',
               '|---|---|---|---:|---:|---:|---:|---|---:|---:|---|']
        for r in self.results:
            j=r['job'];a=r.get('accuracy',{})
            if j['kind'] in ['ode_pipeline','dense_pipeline','dense_photons']:continue
            lines.append(f"| {j['system']} | {j['kind']} | {j['integrator']} | {j['batch']} | {j['dt_ns']:.5g} | {j['threads']} | {j['tile']} | {j['waveform']} | {r.get('median_s',r['status'])} | {100*a.get('max_relative_above_cutoff',float('inf')):.5g} | {passing(r)} |")
        lines+=['','## Output-specific tests','', 'Photon-only timings exclude density/population transfers. The full density is still evolved internally. Full-reference rows without an external reference are convergence baselines, not independent accuracy checks.','']
        for r in self.results:
            j=r['job']
            if j['kind'] in ['ode_pipeline','dense_pipeline','dense_photons']:
                lines.append(f"- {j['system']} {j['kind']}: {r.get('status')}; speedup {r.get('speedup','see rows')}. Details: [{Path(j['output']).name}]({Path(j['output']).name}).")
            if 'photon_max_relative' in r:
                lines.append(f"- {j['system']} {j['kind']} {j.get('output_mode','joint')}, v={j.get('velocity',150)} m/s: {r['median_s']:.6g} s, maximum photon relative error {100*r['photon_max_relative']:.6g}%.")
        (self.out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def execute(self):
        self.stage='1: block sizes and shared-column tiling';self.progress()
        winners={}
        for system in ['rc','r2']:
            variant='swap_const_schedule_gentle_exact_devicepop_packed_direct' if system=='rc' else 'swap_const_exact_packed'
            common=dict(system=system,variant=variant,span_us=5 if system=='rc' else 1,dt_ns=250/56 if system=='rc' else 2.)
            rows=[]
            for threads in [64,128,256,512]:rows.append(self.run(**common,threads=threads))
            staged=variant.replace('_direct','')
            for tile in [16,32]:
                for threads in [128,256,512]:rows.append(self.run(**dict(common,variant=staged),tile=tile,threads=threads))
            eligible=[r for r in rows if passing(r)]
            if not eligible:raise RuntimeError(f'No valid baseline for {system}')
            winners[system]=min(eligible,key=lambda r:r['median_s'])['job']
        self.stage='2: low-storage CK54, matched accuracy';self.progress()
        low={}
        for system in ['rc','r2']:
            w=winners[system];rows=[]
            for h in ([.75,1.,1.5,2.,3.] if system=='rc' else [.25,.5,1.,1.5,2.]):
                rows.append(self.run(**{k:w[k] for k in ['system','variant','span_us','threads','tile']},integrator='ck54',dt_ns=h,repeats=1))
            eligible=[r for r in rows if passing(r)]
            if eligible:low[system]=min(eligible,key=lambda r:r['median_s'])['job']
        self.stage='3: precomputed EOM/polarization stage waveforms';self.progress()
        for system in ['rc','r2']:
            w=winners[system]
            self.run(**{k:w[k] for k in ['system','variant','span_us','threads','tile','dt_ns']},waveform=True)
        atomic(self.out/'screen_winners.json',dict(tsit5=winners,low_storage=low))
        self.stage='screen complete; pipeline and long/output confirmations follow';self.progress();self.report()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.job:worker(json.loads(args.job.read_text()))
    else:
        out=args.out or HERE/('gpu_next_results/v1_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        Runner(out).execute()
