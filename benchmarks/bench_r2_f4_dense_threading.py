"""Limited dense threading / all-J2 population benchmark; no full scans."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

import bench_r2_f4_population_propagator as prior
from centrex_tlf import lindblad, states
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py

OUT=Path(__file__).resolve().parent/'r2_f4_dense_threading_results'
MATRICES={}
WORKER_LIMIT=None


def factor_and_project(L,packed,excited,times,n):
    stages={}
    tick=time.perf_counter()
    w,V=la.eig(L,check_finite=False)
    stages['eig_s']=time.perf_counter()-tick
    tick=time.perf_counter()
    lu,piv=la.lu_factor(V,check_finite=False)
    stages['lu_s']=time.perf_counter()-tick
    tick=time.perf_counter()
    C=la.lu_solve((lu,piv),packed.astype(complex),check_finite=False)
    stages['initial_columns_s']=time.perf_counter()-tick
    tick=time.perf_counter()
    wt=w[:,None]*times[None,:]
    small=abs(wt)<1e-7
    phi=np.expm1(wt)/np.where(abs(w)>0,w,1.)[:,None]
    series=times[None,:]*(1+wt/2+wt**2/6)
    phi[small]=series[small]
    photons=(prior.scan.GAMMA*V[excited].sum(axis=0)[:,None]*C).T@phi
    stages['photon_traces_s']=time.perf_counter()-tick
    stages['total_s']=sum(stages.values())
    return photons,stages,(w,V,C)


def initialize_worker():
    global WORKER_LIMIT
    WORKER_LIMIT=threadpool_limits(limits=1)
    for i in range(3):
        with np.load(OUT/f'point_{i}_inputs.npz') as a:
            MATRICES[i]={k:a[k].copy() for k in a.files}


def ready(_):
    time.sleep(.03)
    return True


def work(index):
    a=MATRICES[index]
    photons,stages,_=factor_and_project(a['L'],a['packed'],a['excited'],a['times'],int(a['n']))
    return stages,float(photons.real[:,-1].sum())


def main():
    OUT.mkdir(exist_ok=True)
    result={'generated_at':datetime.now().astimezone().isoformat(),
        'scope':'Pure X, 10 mW, existing three detunings, all 20 individually resolved X J=2 population projectors, 1401 cumulative photon samples through 350 us.',
        'physical_cores':8,'logical_threads':16,'thread_repeats':2,'worker_repeats':2,
        'thread_scaling':[],'worker_scaling':[],'population_scaling':[],'accuracy':[]}
    with threadpool_limits(limits=1):
        built=prior.scan.build(0.)
        system,prepared,rho,rabis,excited,metadata,_=built
        indices=[i for i,q in enumerate(system.QN)
                 if q.largest.electronic_state==states.ElectronicState.X and q.largest.J==2]
        assert len(indices)==20
        packed=np.zeros((prepared.layout.packed_len,len(indices)))
        for k,i in enumerate(indices):
            packed[prepared.layout.diagonal_index(i),k]=1.
        original_columns=[indices.index(i) for i in metadata['initial_indices']]
        result['resolved_ground_states']=[{'index':i,'label':str(system.QN[i].largest)} for i in indices]
        result['other_ground_states']='Three compact decay-only sinks; not individually resolved F1/F/mF states of other J manifolds.'
        for point,detuning in enumerate(prior.DETUNINGS):
            bound=prior.bind_point(prepared,rabis[prior.POWER_INDEX],detuning)
            ev=create_lindblad_rhs_evaluator_py(bound.rust_plan,'expanded_sparse',True)
            tick=time.perf_counter()
            row,col,val=ev.jacobian_packed_sparse_py(0.,0.,'analytic')
            dim=prepared.layout.packed_len
            L=sp.csr_matrix((val,(row,col)),shape=(dim,dim)).toarray()
            extract_seconds=time.perf_counter()-tick
            a={'L':L,'packed':packed,'excited':np.array(excited),'times':prior.scan.TIMES,'n':len(system.QN)}
            MATRICES[point]=a
            np.savez(OUT/f'point_{point}_inputs.npz',**a)
            print(f'Validate all 20 populations at {detuning:g} MHz',flush=True)
            photons,stages,(w,V,C)=factor_and_project(L,packed,excited,prior.scan.TIMES,len(system.QN))
            cached=prior.cached_curve(detuning)
            error=abs(photons.real[original_columns]-cached)
            trace=(V[:len(system.QN)].sum(axis=0)[:,None]*C).T@np.exp(w[:,None]*prior.scan.TIMES)
            record={'detuning_MHz':detuning,'extract_seconds':extract_seconds,
                'max_cached_four_state_error':float(error.max()),
                'max_scaled_error':float(np.max(error/(1e-7+1e-5*abs(cached)))),
                'all_20_trace_error':float(np.max(abs(trace-1))),
                'all_20_max_photon_imaginary':float(abs(photons.imag).max()),
                'all_20_min_photon_increment':float(np.diff(photons.real,axis=1).min()),
                'timings_20':stages}
            assert record['max_scaled_error']<1 and record['all_20_trace_error']<1e-7
            assert record['all_20_max_photon_imaginary']<1e-7 and record['all_20_min_photon_increment']>-1e-7
            result['accuracy'].append(record)
            np.savez_compressed(OUT/f'point_{point}_responses.npz',detuning_MHz=detuning,
                initial_indices=indices,times_s=prior.scan.TIMES,photons=photons.real)
            if point==0:
                # One independent ODE for a mixture of all 20 states checks the
                # additional columns without 20 new reference trajectories.
                weights=np.arange(1,21,dtype=float);weights/=weights.sum()
                mixture=np.zeros_like(rho[0])
                for i,weight in zip(indices,weights): mixture[i,i]=weight
                tick=time.perf_counter()
                ref=lindblad.solve_lindblad_batch(bound,mixture[None,:,:],(0.,float(prior.scan.TIMES[-1])),
                    solver='dopri5',execution_mode='expanded_sparse',parallel=False,
                    output='photon_integral',output_when='saveat',saveat=prior.scan.TIMES,
                    integral_weights=[(i,prior.scan.GAMMA) for i in excited],integral_method='solver',
                    dt=2e-9,reltol=1e-7,abstol=1e-9,maxiters=1000000)
                ref=np.asarray(ref.values).reshape(-1).real
                difference=abs(weights@photons.real-ref)
                record['all_state_mixture_reference_s']=time.perf_counter()-tick
                record['all_state_mixture_max_error']=float(difference.max())
                assert np.all(difference<=1e-7+1e-5*abs(ref))
                print(f'All-state mixture ODE check: max error {difference.max():.3g}',flush=True)
        # Population count comparison: same one-core eig cost, 4 versus 20 columns.
        a=MATRICES[0]
        for count,columns in [(4,original_columns),(20,list(range(20)))]:
            for repeat in range(2):
                _,stages,_=factor_and_project(a['L'],a['packed'][:,columns],a['excited'],a['times'],int(a['n']))
                result['population_scaling'].append({'population_count':count,'repeat':repeat,**stages})
    # One parameter point only; interleave forward/reverse thread orders.
    for repeat,order in enumerate([[1,2,4,8,16],[16,8,4,2,1]]):
        for threads in order:
            with threadpool_limits(limits=threads):
                _,stages,_=factor_and_project(a['L'],a['packed'],a['excited'],a['times'],int(a['n']))
            result['thread_scaling'].append({'blas_threads':threads,'repeat':repeat,**stages})
            print(f'BLAS {threads} threads: {stages["total_s"]:.3f} s for all 20 populations',flush=True)
    # Exactly three distinct points. Pool creation and initial matrix loading
    # excluded; pool dispatch and results collection included in wall time.
    for workers in [1,2,3]:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            list(pool.map(ready,range(workers*4)))
            for repeat in range(2):
                tick=time.perf_counter()
                outputs=list(pool.map(work,range(3)))
                wall=time.perf_counter()-tick
                result['worker_scaling'].append({'workers':workers,'blas_threads_per_worker':1,
                    'repeat':repeat,'three_point_wall_s':wall,'worker_stages':[o[0] for o in outputs]})
                print(f'{workers} independent workers: {wall:.3f} s for three points',flush=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    med=lambda items,key:float(np.median([p[key] for p in items]))
    report=['# Dense propagator: threading and all-ground-J2 populations','',result['scope'],'',
        'All 20 X J=2 F1/F/mF states are retained individually. Three other ground manifolds remain compact decay sinks; this does not resolve every state of every rotational J. The responses describe initially populated dressed eigenstates, not arbitrary coherent bare-state preparation.','',
        '## BLAS threads within one parameter point','',
        'Two repetitions, opposite thread-count orders, +6.8 MHz. All 20 populations. Dense eig, LU, initial-state projection, and all photon samples included; diagnostics and common Liouvillian extraction excluded.','',
        '| BLAS threads | Median seconds | Speedup versus one |','|---:|---:|---:|']
    base=med([p for p in result['thread_scaling'] if p['blas_threads']==1],'total_s')
    for threads in [1,2,4,8,16]:
        duration=med([p for p in result['thread_scaling'] if p['blas_threads']==threads],'total_s')
        report.append(f'| {threads} | {duration:.3f} | {base/duration:.2f}x |')
    report.extend(['','## Independent parameter workers','',
        'Same three distinct detunings, one BLAS thread per process, two repetitions. Startup and input loading excluded; dispatch/collection included. Only three points, so this does not test saturation of eight physical cores or sixteen logical threads.','',
        '| Workers | Median three-point seconds | Speedup versus one worker |','|---:|---:|---:|'])
    base=med([p for p in result['worker_scaling'] if p['workers']==1],'three_point_wall_s')
    for workers in [1,2,3]:
        duration=med([p for p in result['worker_scaling'] if p['workers']==workers],'three_point_wall_s')
        report.append(f'| {workers} | {duration:.3f} | {base/duration:.2f}x |')
    report.extend(['','## Population count','', '| Initial populations | Median one-thread seconds |','|---:|---:|'])
    for count in [4,20]:
        duration=med([p for p in result['population_scaling'] if p['population_count']==count],'total_s')
        report.append(f'| {count} | {duration:.3f} |')
    report.extend(['','## Accuracy','',
        f"All 20 photon curves passed monotonicity, reality, and trace checks at the three points. The four previously simulated states match the saved library to at most {max(p['max_cached_four_state_error'] for p in result['accuracy']):.3g} photons. An independently integrated weighted mixture of all 20 states differs by at most {result['accuracy'][0]['all_state_mixture_max_error']:.3g} photons over all 1401 timestamps.",
        '', 'State labels, exact settings, timing repetitions and diagnostics: [results.json](results.json). Each point has an NPZ with all 20 separate photon responses. Initial density matrices are projectors onto the field-dressed states; mixtures can be reconstructed with any nonnegative normalized population weights.',
        '', 'No full scan or production backend changes. Reproduce with `.venv/Scripts/python.exe benchmarks/bench_r2_f4_dense_threading.py`.'])
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Finished {OUT / "REPORT.md"}',flush=True)


if __name__=='__main__': main()
