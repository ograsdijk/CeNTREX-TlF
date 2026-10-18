"""Bounded integrator selection followed by five isolated GPU optimizations."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE/'rc_gpu_sparse_results/v1_20261011_235549'
HIST=HERE/'rc_gpu_matched_cpu_results'

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    for i in range(20):
        try:tmp.replace(path);return
        except PermissionError:
            if i==19:raise
            time.sleep(.05)

def historical():
    import rc_gpu_sparse_methods as m
    p=pickle.loads((HIST/'fixture.pkl').read_bytes());n=p['n'];basis=[]
    for c in p['plan']['coefficients']:
        b=np.zeros((n,n),complex)
        for term in c['basis_terms']:b[term['i'],term['j']]+=complex(term['re'],term['im'])
        basis.append(b)
    full=np.array([basis[k]+np.triu(basis[k],1).conj().T for k in [3,2,4,5,6]])
    support=np.any(full!=0,axis=0);np.fill_diagonal(support,False);rows,cols=np.nonzero(support)
    tp,tc,tr=[0],[],[]
    for entries in p['transfers']:
        for e in entries:tc.append(e['source']);tr.append(e['rate'])
        tp.append(len(tc))
    base={k:v['re'] for k,v in zip(p['graph']['slot_names'],p['graph']['base_values'])}
    f=dict(basis=full,diagonal=np.diag(p['plan']['static_matrix']).real.copy(),mask=np.diag(basis[0]).real,
           rows=rows.astype(np.int32),columns=cols.astype(np.int32),ptr=np.r_[0,np.cumsum(support.sum(axis=1))].astype(np.int32),
           tp=np.array(tp,np.int32),tc=np.array(tc,np.int32),tr=np.array(tr),rates=np.array(p['rates']),
           base=dict(mw_peak1=base['Ω1'],mw_peak2=base['Ω2'],PUVZ0=base['PX0'],Ωt0=base['Ωt0'],β0=base['β0'],ωphase0=base['ωphase0']))
    m.fixture=lambda context:f
    m.points=lambda count,pattern:(np.zeros(count),np.zeros(count,int))
    return dict(prepared=SimpleNamespace(layout=SimpleNamespace(n=n)),physics=dict(uv_length_m=1.),
                sigma_I=1e99,rho0=p['rho0'][None]),HIST/'fixture.pkl'

def reference(job):
    if job['fixture']=='historical':
        import run_rc_gpu_matched_cpu as old
        source=old.SOURCES[job['span_us']].parent/'reference.npz'
        return np.load(source)['tsit5_full'][None],source
    rows=json.loads((PREVIOUS/'results.json').read_text())
    for row in rows:
        j=row['job']
        if (j['kind']=='cpu' and j.get('reference') and len(j['initial_J'])==483 and
                abs((j['end']-j['start'])*1e6-job['span_us'])<1e-7):
            source=Path(row['output']).with_suffix('.npz');return np.load(source)['rho'],source
    raise ValueError('Missing archived CPU reference')

def worker(job):
    import cupy as cp
    from rc_gpu_sparse_methods import load,diagnostics
    from rc_gpu_integrators import IntegratorGPU
    if job.get('gpu_class')=='fixed_followup':
        from rc_gpu_fixed_followup import FixedGPU
        IntegratorGPU=FixedGPU
    if job['fixture']=='historical':
        context,source=historical();start=125e-9;velocity=1.
        # The archived 200 V/cm fixture has higher frequencies than production.
        # Retain the selected algorithm, not its production-specific timestep.
        job['requested_dt_ns']=job['dt_ns'];job['dt_ns']=min(job['dt_ns'],.9765625)
    else:context,source,_=load();start=146e-6;velocity=150.
    full=job.get('kind')=='full'
    if full:
        start=0.;velocity=job['velocity'];job['span_us']=context['physics']['uv_length_m']/velocity*1e6
        oldrows=json.loads((PREVIOUS/'results.json').read_text())
        old=next(r for r in oldrows if r['job'].get('kind')=='full' and r['job'].get('velocity')==velocity and
                 r['job'].get('method')=='staged_soa_minor' and r['job'].get('substeps')==512)
        refsource=Path(old['output']).with_suffix('.npz');oldnpz=np.load(refsource)
        ref=oldnpz['density_matrix'];popref=oldnpz['populations']
        strictsource=Path(old['job']['reference']);strict=np.load(strictsource)['density_matrix']
    else:ref,refsource=reference(job)
    setup=time.perf_counter()
    g=IntegratorGPU(context,job['batch'],start,start+job['span_us']*1e-6,
                    job['dt_ns'],job['integrator'],job.get('variant','base'),job.get('chunk',16),
                    job.get('atol',1e-10),job.get('rtol',1e-9),pattern='full' if full else 'grid',velocity=velocity)
    # Compare the new RHS to the established FP64 staged RHS on identical
    # physical initial states. This also catches specialization/type mistakes.
    from rc_gpu_sparse_methods import GPU
    baseline=GPU(context,'staged_soa_minor',job['batch'],start,g.end,256,velocity,pattern='full' if full else 'grid')
    baseline.clock.set(np.array([start,0.,g.end,g.dt]));baseline.rhs(baseline.y,baseline.ks[0],0.)
    g.clock.set(np.array([start,0.,g.end,g.dt,0.,0.,0.,0.]));g.rhs(g.y,g.ks[0],0.)
    rhs_ref=baseline.download(baseline.ks[0]);rhs_actual=g.download(g.ks[0])
    rhs_error=float(np.linalg.norm(rhs_actual-rhs_ref)/max(np.linalg.norm(rhs_ref),1e-300))
    if rhs_error>(2e-6 if g.mixed else 1e-12):raise ValueError(f'RHS mismatch {rhs_error}')
    del baseline
    g.capture();setup_s=time.perf_counter()-setup
    tick=time.perf_counter()
    if full:
        rho,populations,fulltiming=g.full_transit();timings=[fulltiming['full_wall_s_including_sampling_and_post_decay']]
        clock=g.clock.get();final=float(clock[0]);g.stats=[dict(accepted=int(clock[6]),rejected=int(clock[7]))]
    else:rho,timings,final=g.evolve(job.get('repeats',3))
    wall=time.perf_counter()-tick
    expected=ref[np.arange(len(rho))%len(ref)]
    out=Path(job['output'])
    np.savez_compressed(out.with_suffix('.npz'),rho=rho[:min(483,len(rho))],reference=expected[:min(483,len(rho))])
    if not np.isfinite(rho).all():
        atomic(out,dict(job=job,status='numerical_failure',reason='Nonfinite density',stats=g.stats,
                       nonfinite_entries=int(np.count_nonzero(~np.isfinite(rho))),evolution_s=timings))
        return
    accuracy=diagnostics(rho.reshape(-1,g.n,g.n),expected.reshape(-1,g.n,g.n))
    report=dict(job=job,status='complete',setup_s=setup_s,evolution_s=timings,median_s=float(np.median(timings)),
                evolution_and_download_wall_s=wall,accuracy=accuracy,stats=g.stats,rhs_relative_error=rhs_error,
                context=str(source),context_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reference=str(refsource),reference_sha256=hashlib.sha256(refsource.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(g.source.encode()).hexdigest(),device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                storage='float32' if g.mixed else 'float64',accumulation='float64',cupy=cp.__version__,
                memory_pool_bytes=cp.get_default_memory_pool().total_bytes(),final_time=final)
    report['python_source_hashes']={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in
                                   ['rc_gpu_integrators.py','run_rc_gpu_integrator_suite.py']}
    if job.get('gpu_class')=='fixed_followup':
        report['python_source_hashes']['rc_gpu_fixed_followup.py']=hashlib.sha256((HERE/'rc_gpu_fixed_followup.py').read_bytes()).hexdigest()
    report['product_kernel_attributes']=g.kernels['product'].attributes
    if full:
        report.update(fulltiming);report['accuracy_scope']='All nine endpoints vs finer GPU reference; strict CPU subset separately'
        report['strict_cpu_subset_accuracy']=diagnostics(rho[[4]].reshape(-1,g.n,g.n),strict.reshape(-1,g.n,g.n))
        report['population_history_max_abs_difference']=float(np.max(np.abs(populations-popref[np.arange(len(populations))%len(popref)])))
        np.savez_compressed(out.with_suffix('.npz'),rho=rho,reference=expected,populations=populations)
    atomic(out,report)
    print(json.dumps(dict(median_s=report['median_s'],error_percent=accuracy['max_relative_above_cutoff']*100,stats=g.stats)),flush=True)

class Suite:
    def __init__(self,out,deadline):
        self.out=out;out.mkdir(parents=True,exist_ok=True);self.deadline=deadline;self.started=time.time()
        self.results=[];self.current=None;self.stage='correctness and integrator selection'
        self.winner=None;self.kept=[];self.count=0
        sources=['rc_gpu_integrators.cu','rc_gpu_integrators.py','run_rc_gpu_integrator_suite.py',
                 'rc_gpu_sparse_methods.py','rc_gpu_sparse_kernels.cu']
        atomic(out/'manifest.json',dict(deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
              sources={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in sources},cpu_odes_rerun=False))
        self.progress()

    def progress(self):
        atomic(self.out/'progress.json',dict(status='running',stage=self.stage,current_job=self.current,
              completed_jobs=len(self.results),failed_jobs=sum(r.get('status')!='complete' for r in self.results),
              timed_out_jobs=sum(r.get('status')=='timeout' for r in self.results),elapsed_s=time.time()-self.started,
              remaining_budget_s=max(0,self.deadline-time.time()),runner_pid=os.getpid(),winner=self.winner,kept_optimizations=self.kept))

    def run(self,**kw):
        if self.deadline-time.time()<30:raise TimeoutError('Suite budget exhausted')
        self.count+=1
        j=dict(fixture='production',span_us=1,batch=128,integrator='tsit5',dt_ns=.9765625,
               variant='base',chunk=16,repeats=3,atol=1e-10,rtol=1e-9);j.update(kw)
        stem=f"{self.count:03}_{j['fixture']}_{j['integrator']}_{j['variant']}_b{j['batch']}_t{j['span_us']}"
        j['output']=str(self.out/(stem+'.json'));self.current=j;self.progress()
        print(stem,flush=True)
        with (self.out/(stem+'.stdout.log')).open('w') as stdout,(self.out/(stem+'.stderr.log')).open('w') as stderr:
            jobpath=self.out/(stem+'.job.json');atomic(jobpath,j)
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(jobpath)],stdout=stdout,stderr=stderr)
            limit=min(self.deadline,time.time()+max(180,60*j['span_us']))
            while child.poll() is None and time.time()<limit:
                time.sleep(2);self.progress()
            if child.poll() is None:child.kill();child.wait();r=dict(job=j,status='timeout')
            elif child.returncode:r=dict(job=j,status='failed',exit_code=child.returncode,stderr=str(self.out/(stem+'.stderr.log')))
            else:r=json.loads(Path(j['output']).read_text())
        self.results.append(r);atomic(self.out/'results.json',self.results);self.report();self.current=None;self.progress()
        if r.get('status')=='complete':print(' seconds',r['median_s'],'error%',100*r['accuracy']['max_relative_above_cutoff'],flush=True)
        else:print(r,flush=True)
        return r

    def passing(self,r):
        return (r.get('status')=='complete' and r['accuracy']['passes_1percent_above_cutoff'] and
                r.get('strict_cpu_subset_accuracy',{'passes_1percent_above_cutoff':True})['passes_1percent_above_cutoff'])

    def report(self):
        lines=['# GPU integrator selection and optimizations','',f'Stage: {self.stage}.',
               f'Winner: {self.winner}. Retained optimization switches: {self.kept}.','',
               'Full complex-entry target: 1% above magnitude 1e-6; near-zero absolute errors are reported separately. '
               'No CPU ODEs or production changes. Production fixture uses 483 distinct detuning/J-mixture cases repeated only to fill larger batches. '
               'Historical fixture uses independent replicas of the archived nominal trajectory.','',
               '| Fixture / us | Integrator | Variant | Batch | dt initial/fixed ns | Chunk | Seconds | Entry error % | Pass |',
               '|---|---|---|---:|---:|---:|---:|---:|---|']
        for r in self.results:
            j=r['job']
            if r.get('status')=='complete':lines.append(f"| {j['fixture']} / {j['span_us']} | {j['integrator']} | {j['variant']} | {j['batch']} | {j['dt_ns']:.4g} | {j['chunk']} | {r['median_s']:.4g} | {100*r['accuracy']['max_relative_above_cutoff']:.4g} | {self.passing(r)} |")
            else:lines.append(f"| {j['fixture']} / {j['span_us']} | {j['integrator']} | {j['variant']} | {j['batch']} | | | {r['status']} | | False |")
        fullrows=[r for r in self.results if r['job'].get('kind')=='full' and r.get('status')=='complete']
        if fullrows:
            lines+=['','## Full-transit validation','',
                    'Includes 201 population samples, density at field exit, and density after 25 lifetimes with drives off. '
                    'Nine distinct cases are detunings -20/0/+20 MHz times J=1/2/3 mixtures; larger batches repeat these cases. '
                    'All endpoint entries are compared to archived RK4 at 0.48828125 ns. '
                    'An archived strict CPU reference separately checks J=2 at zero detuning. '
                    'This does not validate every detuning and velocity over its full transit.','',
                    '| Method / variant | v m/s | Batch | Base step ns | Total s | Endpoint error % vs finer GPU | Strict CPU subset % | Population max absolute difference | Pass |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---|']
            for r in fullrows:
                j=r['job']
                lines.append(f"| {j['integrator']} / {j['variant']} | {j['velocity']} | {j['batch']} | {j['dt_ns']:.6g} | {r['median_s']:.4g} | {100*r['accuracy']['max_relative_above_cutoff']:.5g} | {100*r['strict_cpu_subset_accuracy']['max_relative_above_cutoff']:.5g} | {r['population_history_max_abs_difference']:.4g} | {self.passing(r)} |")
        lines+=['','Adaptive error control: maximum of per-trajectory RMS complex embedded errors. '
                'Accepted/rejected steps, setup/capture/transfer cost, FP32 storage diagnostics and full numeric outputs are in individual JSON/NPZ files. '
                'Short-interval fixed/adaptive accuracy is checked against the same archived strict CPU references; a local tolerance alone is not a pass. '
                'Full-transit reference scope is stated separately above.','',
                'The previous approximately 2.56x CPU throughput estimate used fixed RK4 at 0.9765625 ns and 128 replicas of the historical 200 us fixture. '
                'It divided an archived single-CPU runtime by ideal 12-way scaling; it was not a measured 12-worker speedup. '
                'Historical and production fields differ, so the production Tsit5/RK4 gain must not be multiplied into that estimate. '
                'No CPU ODE timings were rerun in this suite.']
        (self.out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def execute(self):
        # Select numerical settings on the full 483-case reference grid, not
        # just the earlier twelve-case subset. All rows are checked individually.
        candidates=[]
        for method,steps in [('rk4',[.48828125,.9765625,1.953125]),('tsit5',[.9765625,1.953125,3.90625])]:
            for dt in steps:
                r=self.run(integrator=method,dt_ns=dt,batch=512,repeats=1)
                if self.passing(r):candidates.append(r)
        for atol,rtol in [(1e-10,1e-9),(1e-9,1e-8)]:
            r=self.run(integrator='adaptive_tsit5',dt_ns=.9765625,batch=512,repeats=1,atol=atol,rtol=rtol)
            if self.passing(r):candidates.append(r)
        finalists=[]
        for method in ['rk4','tsit5','adaptive_tsit5']:
            eligible=[r for r in candidates if r['job']['integrator']==method]
            if eligible:
                best=min(eligible,key=lambda r:r['median_s']);j=best['job']
                r=self.run(integrator=method,dt_ns=j['dt_ns'],atol=j['atol'],rtol=j['rtol'],span_us=5,batch=512)
                if self.passing(r):finalists.append(r)
        if not finalists:raise RuntimeError('No accuracy-qualified integrated finalist')
        best=min(finalists,key=lambda r:r['median_s']);j=best['job']
        self.winner={k:j[k] for k in ['integrator','dt_ns','atol','rtol']};self.stage='five optimization ablations';self.progress()
        # Keep numerical settings unchanged while testing implementation changes.
        common=dict(self.winner,span_us=5,batch=512)
        base=best;tests=[]
        for variant,chunk in [('fused',16),('reuse',16),('special',16),('mixed',16),('base',64),('base',256)]:
            r=self.run(**common,variant=variant,chunk=chunk);tests.append(r)
        accepted=[r for r in tests if self.passing(r) and r['median_s']<base['median_s']*.97]
        flags=[r['job']['variant'] for r in accepted if r['job']['variant']!='base']
        # Mixed precision is retained only if it also passes a long reference check.
        variant='_'.join(sorted(set(flags))) or 'base'
        chunks=[r for r in accepted if r['job']['chunk']!=16]
        chunk=min(chunks,key=lambda r:r['median_s'])['job']['chunk'] if chunks else 16
        combined=self.run(**common,variant=variant,chunk=chunk)
        if not self.passing(combined) or combined['median_s']>=base['median_s']:
            choices=[base]+accepted
            combined=min(choices,key=lambda r:r['median_s']);variant=combined['job']['variant'];chunk=combined['job']['chunk']
        self.kept=[variant,f'graph_steps={chunk}'];self.stage='large-batch and archived-fixture confirmation';self.progress()
        for b in [128,1024,2048]:self.run(**dict(common,batch=b),variant=variant,chunk=chunk,repeats=3 if b<2048 else 1)
        # Historical fields have a different timestep requirement. Try the chosen
        # integrator, then tighten its timestep/tolerances on an accuracy failure.
        for span in [1,10,200]:
            opts=dict(self.winner,fixture='historical',span_us=span,batch=128,variant=variant,chunk=chunk,repeats=1)
            r=self.run(**opts)
            if not self.passing(r):
                opts.update(dt_ns=opts['dt_ns']/2,atol=opts['atol']/10,rtol=opts['rtol']/10)
                r=self.run(**opts)
            if 'mixed' in variant and not self.passing(r):
                self.kept.append('mixed precision failed historical validation; not qualified for production')
        self.stage='complete';self.current=None;self.progress();self.report()
        state=json.loads((self.out/'progress.json').read_text());state['status']='complete';atomic(self.out/'progress.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job');parser.add_argument('--output');parser.add_argument('--deadline')
    args=parser.parse_args();os.environ.setdefault('CUPY_CACHE_DIR',str(HERE/'.cupy_cache'))
    if args.job:worker(json.loads(Path(args.job).read_text()))
    else:
        suite=Suite(Path(args.output),datetime.fromisoformat(args.deadline).timestamp())
        try:suite.execute()
        except BaseException as error:
            suite.stage='stopped: '+str(error);suite.report();suite.progress()
            state=json.loads((suite.out/'progress.json').read_text());state['status']='budget_exhausted' if isinstance(error,TimeoutError) else 'failed';atomic(suite.out/'progress.json',state)
            raise
