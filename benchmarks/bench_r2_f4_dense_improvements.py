"""Benchmark-only exact block reduction, real eigenbasis, and worker scaling.

Uses three saved Liouvillians, no ODE reruns or full detuning scans. Validates
full packed density matrices including separately recovered sink populations.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

import bench_r2_f4_dense_threading as previous
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

OUT=Path(__file__).resolve().parent/'r2_f4_dense_improvements_results'
DATA={}
LIMIT=None


def phi(w,times):
    wt=w[:,None]*np.asarray(times)[None,:]
    result=np.expm1(wt)/np.where(abs(w)>0,w,1.)[:,None]
    small=abs(wt)<1e-7
    series=np.asarray(times)[None,:]*(1+wt/2+wt**2/6)
    result[small]=series[small]
    return result


def real_factor(A,packed,excited,times):
    stages={}
    tick=time.perf_counter()
    w,V=la.eig(A,check_finite=False)
    stages['eig_s']=time.perf_counter()-tick
    tick=time.perf_counter()
    real=np.flatnonzero(w.imag==0)
    positive=np.flatnonzero(w.imag>0)
    negative=positive+1
    np.testing.assert_allclose(w[negative],w[positive].conj(),atol=0,rtol=0)
    assert len(real)+2*len(positive)==len(w)
    R=np.empty(A.shape,dtype=float,order='F')
    R[:,real]=V[:,real].real
    R[:,positive]=V[:,positive].real
    R[:,negative]=V[:,positive].imag
    stages['real_basis_s']=time.perf_counter()-tick
    tick=time.perf_counter()
    lu,piv=la.lu_factor(R,check_finite=False)
    stages['lu_s']=time.perf_counter()-tick
    tick=time.perf_counter()
    C=la.lu_solve((lu,piv),packed,check_finite=False)
    stages['initial_columns_s']=time.perf_counter()-tick
    tick=time.perf_counter()
    observable=previous.prior.scan.GAMMA*R[excited].sum(axis=0)
    real_photons=(observable[real,None]*C[real]).T@phi(w[real].real,times)
    pair_weights=(observable[positive]+1j*observable[negative])[:,None]*(C[positive]-1j*C[negative])
    photons=real_photons+(pair_weights.T@phi(w[positive],times)).real
    stages['photon_traces_s']=time.perf_counter()-tick
    stages['total_s']=sum(stages.values())
    factors={'w':w,'basis':R,'C':C,'real':real,'positive':positive,'negative':negative,'real_basis':True}
    return photons,stages,factors


def complex_factor(A,packed,excited,times):
    photons,stages,(w,V,C)=previous.factor_and_project(A,packed,excited,times,41)
    return photons.real,stages,{'w':w,'basis':V,'C':C,'real_basis':False}


def modal_state(factors,t,integral=False):
    w,C=factors['w'],factors['C']
    multiplier=phi(w,[t])[:,0] if integral else np.exp(w*t)
    if not factors['real_basis']:
        return multiplier[:,None]*C
    real,positive,negative=[factors[k] for k in ['real','positive','negative']]
    modes=np.empty_like(C)
    modes[real]=multiplier[real,None].real*C[real]
    pair=multiplier[positive,None]*(C[positive]-1j*C[negative])
    modes[positive]=pair.real
    modes[negative]=-pair.imag
    return modes


def reconstruct(factors,t,a,reduced):
    state=factors['basis']@modal_state(factors,t)
    if not reduced: return state.real
    full=np.zeros((int(a['n'])**2,a['packed'].shape[1]))
    full[a['keep']]=state.real
    integrated=factors['basis']@modal_state(factors,t,integral=True)
    full[a['sinks']]=(a['sink_feed']@integrated).real
    return full


def reduce(a):
    L,n=a['L'],int(a['n'])
    layout=PackedHermitianLayout(n)
    ground=np.flatnonzero(a['packed'][:n].sum(axis=1))
    active=sorted(set(ground.tolist()+a['excited'].tolist()))
    sinks=np.array(sorted(set(range(n))-set(active)),dtype=int)
    keep=np.array(sorted(active+[k for i in active for j in active if i<j
                                for k in [layout.upper_real_index(i,j),layout.upper_imag_index(i,j)]]))
    excluded=np.array(sorted(set(range(n*n))-set(keep)-set(sinks)),dtype=int)
    # These must be identically zero, not merely below a pruning threshold.
    assert np.count_nonzero(L[:,sinks])==0
    assert np.count_nonzero(L[np.ix_(excluded,np.r_[keep,sinks])])==0
    assert np.count_nonzero(a['packed'][excluded])==0
    assert np.count_nonzero(a['packed'][sinks])==0
    A=L[np.ix_(keep,keep)]
    count,labels=connected_components(csr_matrix(A!=0),directed=True,connection='strong')
    result={**a,'A':A,'reduced_packed':a['packed'][keep],
        'reduced_excited':np.array([np.flatnonzero(keep==i)[0] for i in a['excited']]),
        'keep':keep,'sinks':sinks,'sink_feed':L[np.ix_(sinks,keep)]}
    return result,{'full_dimension':n*n,'active_dimension':len(keep),
        'removed_sink_coherences':len(excluded),'sink_count':len(sinks),
        'active_strong_block_sizes':sorted(np.bincount(labels).tolist(),reverse=True)}


def initializer():
    global LIMIT
    LIMIT=threadpool_limits(limits=1)
    for i in range(3):
        with np.load(OUT/f'point_{i}_worker_inputs.npz') as a:
            DATA[i]={k:a[k].copy() for k in a.files}


def ready(_):
    time.sleep(.04)
    return True


def work(i):
    a=DATA[i%3]
    photons,stages,_=real_factor(a['A'],a['packed'],a['excited'],a['times'])
    return stages['total_s'],float(photons[:,-1].sum())


def checkpoint(result):
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    result={'generated_at':datetime.now().astimezone().isoformat(),
        'scope':'Same three pure-X 10 mW F4 parameter points; all 20 initial X J2 populations; 1401 photon times.',
        'variants':[],'accuracy':[],'worker_scaling':[],
        'timing_note':'Two repetitions for method timings, one BLAS thread. Worker throughput uses 32 repeated tasks cycling over the same three matrices, not new detunings; two repetitions. Pool startup/input loading excluded.'}
    with threadpool_limits(limits=1):
        for point,detuning in enumerate(previous.prior.DETUNINGS):
            with np.load(previous.OUT/f'point_{point}_inputs.npz') as inputs:
                a={k:inputs[k].copy() for k in inputs.files}
            a,structure=reduce(a)
            np.savez(OUT/f'point_{point}_worker_inputs.npz',A=a['A'],packed=a['reduced_packed'],
                excited=a['reduced_excited'],times=a['times'])
            factors={};curves={}
            for repeat,order in enumerate([['full_complex','active_complex','active_real'],['active_real','active_complex','full_complex']]):
                for name in order:
                    reduced=name!='full_complex'
                    matrix=a['A'] if reduced else a['L']
                    packed=a['reduced_packed'] if reduced else a['packed']
                    exc=a['reduced_excited'] if reduced else a['excited']
                    function=real_factor if name=='active_real' else complex_factor
                    photons,stages,factor=function(matrix,packed,exc,a['times'])
                    result['variants'].append({'point':point,'detuning_MHz':detuning,'variant':name,'repeat':repeat,**stages})
                    factors[name]=factor;curves[name]=photons
                    print(f'{detuning:g} MHz {name} repeat {repeat}: {stages["total_s"]:.3f} s',flush=True)
            packed_error={name:0. for name in ['active_complex','active_real']}
            maximum_trace_error=0.;minimum_eigenvalue=0.;minimum_population=0.
            # Full density reconstruction includes optical/ground coherences
            # and each compact sink population, at prompt and late times.
            times=[0.,.25e-6,1e-6,10e-6,50e-6,.02/150,250e-6,350e-6]
            layout=PackedHermitianLayout(int(a['n']))
            for t in times:
                baseline=reconstruct(factors['full_complex'],t,a,False)
                for name in packed_error:
                    candidate=reconstruct(factors[name],t,a,True)
                    packed_error[name]=max(packed_error[name],float(abs(candidate-baseline).max()))
                candidate=reconstruct(factors['active_real'],t,a,True)
                maximum_trace_error=max(maximum_trace_error,float(abs(candidate[:int(a['n'])].sum(axis=0)-1).max()))
                minimum_population=min(minimum_population,float(candidate[:int(a['n'])].min()))
                for column in candidate.T:
                    minimum_eigenvalue=min(minimum_eigenvalue,float(la.eigvalsh(layout.unpack(column),check_finite=False).min()))
            photon_error={name:float(abs(curves[name]-curves['full_complex']).max()) for name in packed_error}
            with np.load(previous.OUT/f'point_{point}_responses.npz') as saved:
                cached_error=float(abs(curves['active_real']-saved['photons']).max())
            accuracy={'detuning_MHz':detuning,'structure':structure,
                'max_full_packed_error':packed_error,'max_photon_error':photon_error,
                'max_previous_all20_photon_error':cached_error,'max_trace_error':maximum_trace_error,
                'min_population':minimum_population,'min_density_eigenvalue':minimum_eigenvalue,
                'min_photon_increment':float(np.diff(curves['active_real'],axis=1).min())}
            assert max(packed_error.values())<1e-7 and max(photon_error.values())<1e-7
            assert cached_error<1e-7 and maximum_trace_error<1e-7 and minimum_eigenvalue>-1e-7
            assert accuracy['min_photon_increment']>-1e-7
            result['accuracy'].append(accuracy)
            np.savez_compressed(OUT/f'point_{point}_responses.npz',detuning_MHz=detuning,
                times_s=a['times'],active_real_photons=curves['active_real'],full_photons=curves['full_complex'])
            checkpoint(result)
            print(f'All density and photon checks passed at {detuning:g} MHz',flush=True)
    for workers in [1,4,8,16]:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initializer) as pool:
            list(pool.map(ready,range(workers*4)))
            for repeat in range(2):
                tick=time.perf_counter()
                values=list(pool.map(work,range(32)))
                wall=time.perf_counter()-tick
                result['worker_scaling'].append({'workers':workers,'blas_threads_per_worker':1,
                    'repeat':repeat,'tasks':32,'wall_s':wall,'points_per_second':32/wall,
                    'worker_compute_s':[v[0] for v in values]})
                checkpoint(result)
                print(f'{workers} workers repeat {repeat}: {wall:.3f} s / 32 points = {32/wall:.2f} points/s',flush=True)
    med=lambda rows,key:float(np.median([r[key] for r in rows]))
    report=['# Additional dense-propagator improvements','',result['scope'],'',
        'Exact reduction checks: sink columns have zero feedback; excluded coherences have exactly zero feed from retained variables and zero initial values. The 38-state driven block has 1444 packed variables; all three compact sink populations are recovered by analytic integration of their feed. No dipole pruning or secular approximation is introduced.',
        '', 'The active block is one strongly connected component, so graph-based block splitting offers no further reduction here.',
        '', 'The real-eigenbasis variant replaces complex eigenvector LU with real LU and evaluates only one member of each conjugate pair. Both variants retain full density reconstruction, including sink populations and zero-filled excluded coherences for the validated initial conditions.',
        '', '## One-worker timings','',result['timing_note'],'',
        '| Method | Median seconds per point | Speedup versus full complex |',
        '|---|---:|---:|']
    base=med([r for r in result['variants'] if r['variant']=='full_complex'],'total_s')
    for name in ['full_complex','active_complex','active_real']:
        duration=med([r for r in result['variants'] if r['variant']==name],'total_s')
        report.append(f'| {name} | {duration:.3f} | {base/duration:.2f}x |')
    report.extend(['','## Parallel throughput','',
        'Same three matrices repeated to supply enough independent tasks; no caching of decompositions across tasks. One BLAS thread per worker. Warm worker pools; process startup and input loading excluded.',
        '', '| Workers | Median 32-point seconds | Points per second |', '|---:|---:|---:|'])
    best=None
    for workers in [1,4,8,16]:
        duration=med([r for r in result['worker_scaling'] if r['workers']==workers],'wall_s')
        if best is None or duration<best[1]: best=(workers,duration)
        report.append(f'| {workers} | {duration:.3f} | {32/duration:.2f} |')
    report.extend(['',f'Best measured configuration: {best[0]} workers. Arithmetic projection for the 5406 parameter points in the original scan: {5406*best[1]/32/60:.1f} minutes of warm compute, excluding setup, extraction, output, and varying physics. This is not a measured full-scan runtime.',
        '', '## Accuracy and scope','',
        f"All 20 initial states passed. Maximum difference versus full packed density reconstruction at eight prompt/late times: {max(max(a['max_full_packed_error'].values()) for a in result['accuracy']):.3g}. Maximum photon difference over all 1401 times: {max(max(a['max_photon_error'].values()) for a in result['accuracy']):.3g}. Trace, density-matrix positivity, and cumulative-photon monotonicity checks passed.",
        '', 'Removing sink coherences is valid for initial states with zero such coherence. Arbitrary initial coherences involving sinks need the full path. Retained states are the existing compact 41-level model; this does not reconstruct individual members of compact spectator manifolds.',
        '', 'Only pure X and 10 mW tested. Two timing repetitions, three parameter matrices; wider polarization, power, transition, and conditioning coverage remains necessary before integration as a solver. No changes to centrex_tlf.',
        '', 'Detailed timing stages and numerical checks: [results.json](results.json). Reproduce with `.venv/Scripts/python.exe benchmarks/bench_r2_f4_dense_improvements.py`.'])
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Finished {OUT / "REPORT.md"}',flush=True)


if __name__=='__main__': main()
