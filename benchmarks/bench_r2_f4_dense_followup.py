"""Small benchmark of eigen routines/buffers and remaining power/polarizations."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

import bench_r2_f4_dense_improvements as improved
from centrex_tlf import states
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py

prior=improved.previous.prior
OUT=Path(__file__).resolve().parent/'r2_f4_dense_followup_results'


def factor(A,packed,excited,times,method='scipy',workspace=None):
    tick=time.perf_counter()
    native=method.startswith('lapack')
    if native:
        Af=workspace if workspace is not None else np.empty(A.shape,order='F')
        np.copyto(Af,A)
        geev,lwork=la.get_lapack_funcs(('geev','geev_lwork'),(Af,))
        optimum,info=lwork(len(Af),compute_vl=0,compute_vr=1)
        assert info==0
        wr,wi,_,R,info=geev(Af,compute_vl=0,compute_vr=1,lwork=int(optimum),overwrite_a=1)
        if info: raise ValueError(f'GEEV failed: {info}')
        w=wr+1j*wi
    elif method=='numpy':
        w,V=np.linalg.eig(A)
    elif method=='overwrite_fortran':
        w,V=la.eig(np.array(A,order='F'),check_finite=False,overwrite_a=True)
    elif method=='balanced':
        B,(scale,_)=la.matrix_balance(A,permute=False,separate=True)
        w,V=la.eig(B,check_finite=False)
        V=scale[:,None]*V
    else:
        w,V=la.eig(A,check_finite=False)
    real=np.flatnonzero(w.imag==0);positive=np.flatnonzero(w.imag>0);negative=positive+1
    np.testing.assert_array_equal(w[negative],w[positive].conj())
    assert len(real)+2*len(positive)==len(w)
    if not native:
        R=np.empty(A.shape,order='F')
        R[:,real]=V[:,real].real
        R[:,positive]=V[:,positive].real
        R[:,negative]=V[:,positive].imag
    eigen_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    lu,piv=la.lu_factor(R,check_finite=False)
    C=la.lu_solve((lu,piv),packed,check_finite=False)
    solve_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    obs=prior.scan.GAMMA*R[excited].sum(axis=0)
    photons=(obs[real,None]*C[real]).T@improved.phi(w[real].real,times)
    weight=(obs[positive]+1j*obs[negative])[:,None]*(C[positive]-1j*C[negative])
    photons+=(weight.T@improved.phi(w[positive],times)).real
    projection_seconds=time.perf_counter()-tick
    stages={'eigen_and_real_basis_s':eigen_seconds,'lu_and_initial_columns_s':solve_seconds,
            'photon_projection_s':projection_seconds,'total_s':eigen_seconds+solve_seconds+projection_seconds}
    factors={'w':w,'basis':R,'C':C,'real':real,'positive':positive,'negative':negative,'real_basis':True}
    return photons,stages,factors


def save(result):
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def cached(fraction,power,detuning):
    tag=f'z{round(fraction*100):02d}'
    for path in sorted(prior.scan.DATA.glob(f'{tag}_????_????.npz')):
        with np.load(path) as a:
            indices=np.flatnonzero(np.isclose(a['detuning_uv_MHz'],detuning,atol=1e-9,rtol=0))
            if len(indices): return a['photons'][:,power,indices[0]].copy()
    raise ValueError('Missing cached reference')


def main():
    OUT.mkdir(exist_ok=True)
    result={'generated_at':datetime.now().astimezone().isoformat(),
        'scope':'Eigen routines and matrix assembly at one existing point, then eight selected parameter points covering 0/3/5% Z and 1/10 mW. No full scan, no new ODE trajectories.',
        'blas_threads':1,'eigen_timings':[],'matrix_timings':[],'coverage':[]}
    with threadpool_limits(limits=1):
        with np.load(improved.previous.OUT/'point_0_inputs.npz') as inputs:
            a={k:inputs[k].copy() for k in inputs.files}
        a,_=improved.reduce(a)
        methods=['scipy','numpy','overwrite_fortran','balanced','lapack_native','lapack_reuse']
        workspace=np.empty(a['A'].shape,order='F')
        baseline,_,baseline_factors=factor(a['A'],a['reduced_packed'],a['reduced_excited'],a['times'])
        for repeat,order in enumerate([methods,methods[::-1]]):
            for method in order:
                photons,timings,factors=factor(a['A'],a['reduced_packed'],a['reduced_excited'],a['times'],
                                              method,workspace if method=='lapack_reuse' else None)
                error=float(abs(photons-baseline).max())
                assert error<1e-7
                state=improved.reconstruct(factors,350e-6,a,True)
                reference=improved.reconstruct(baseline_factors,350e-6,a,True)
                density_error=float(abs(state-reference).max())
                assert density_error<1e-7
                result['eigen_timings'].append({'method':method,'repeat':repeat,
                    'max_photon_error':error,'max_packed_density_error':density_error,**timings})
                print(f'{method} repeat {repeat}: {timings["total_s"]:.3f} s; photon error {error:.3g}',flush=True)
        # Time matrix allocation/scatter independently. Exact Jacobian extraction
        # is common to all choices and excluded from this allocation microbench.
        sparse=csr_matrix(a['L']);reduced=sparse[a['keep']][:,a['keep']].tocsr()
        out=np.empty(a['A'].shape,order='F')
        rows,cols=reduced.nonzero();values=reduced.data.copy()
        location=rows+len(a['A'])*cols
        for method in ['full_then_slice','direct_reduced','reuse_reduced','reuse_scatter']:
            for repeat in range(3):
                tick=time.perf_counter()
                for _ in range(100):
                    if method=='full_then_slice': output=sparse.toarray()[np.ix_(a['keep'],a['keep'])]
                    elif method=='direct_reduced': output=reduced.toarray(order='F')
                    elif method=='reuse_reduced': output=reduced.toarray(out=out)
                    else:
                        out.fill(0.);out.ravel(order='F')[location]=values;output=out
                seconds=(time.perf_counter()-tick)/100
                np.testing.assert_array_equal(output,a['A'])
                result['matrix_timings'].append({'method':method,'repeat':repeat,'seconds_per_assembly':seconds})
            print(f'{method}: {seconds*1000:.3f} ms per assembly',flush=True)
        med=lambda rows,key:float(np.median([r[key] for r in rows]))
        best=min(methods,key=lambda method:med([r for r in result['eigen_timings'] if r['method']==method],'total_s'))
        result['fastest_eigen_method']=best
        save(result)
        # Both powers at mixed polarization; 1 mW pure X supplements earlier
        # 10 mW pure X checks. Eight total selected points, cached references.
        cases=[(0.,0,[6.8,28.2]),(.03,1,[6.8,28.2]),(.05,1,[6.8,28.2]),(.03,0,[28.2]),(.05,0,[28.2])]
        current_fraction=None
        for fraction,power,detunings in cases:
            if current_fraction!=fraction:
                built=prior.scan.build(fraction);current_fraction=fraction
            system,prepared,rho,rabis,excited,metadata,_=built
            indices=[i for i,q in enumerate(system.QN) if q.largest.electronic_state==states.ElectronicState.X and q.largest.J==2]
            assert len(indices)==20
            packed=np.zeros((prepared.layout.packed_len,len(indices)))
            for column,i in enumerate(indices): packed[prepared.layout.diagonal_index(i),column]=1.
            reference_columns=[indices.index(i) for i in metadata['initial_indices']]
            for detuning in detunings:
                print(f'Coverage fz={fraction:g}, {prior.scan.POWERS[power]*1000:g} mW, {detuning:g} MHz',flush=True)
                bound=prior.bind_point(prepared,rabis[power],detuning)
                ev=create_lindblad_rhs_evaluator_py(bound.rust_plan,'expanded_sparse',True)
                tick=time.perf_counter()
                row,col,val=ev.jacobian_packed_sparse_py(0.,0.,'analytic')
                dim=prepared.layout.packed_len
                L=csr_matrix((val,(row,col)),shape=(dim,dim)).toarray()
                extraction_s=time.perf_counter()-tick
                data={'L':L,'packed':packed,'excited':np.array(excited),'times':prior.scan.TIMES,'n':len(system.QN)}
                data,structure=improved.reduce(data)
                photons,timing,factors=factor(data['A'],data['reduced_packed'],data['reduced_excited'],data['times'],best)
                full,_,full_factors=improved.complex_factor(L,packed,np.array(excited),data['times'])
                reference=cached(fraction,power,detuning)
                difference=abs(photons[reference_columns]-reference)
                maximum_density_error=0.;maximum_trace_error=0.;minimum_density_eigenvalue=0.
                layout=improved.PackedHermitianLayout(len(system.QN))
                for t in [0.,1e-6,.02/150,350e-6]:
                    state=improved.reconstruct(factors,t,data,True)
                    state_full=improved.reconstruct(full_factors,t,data,False)
                    maximum_density_error=max(maximum_density_error,float(abs(state-state_full).max()))
                    maximum_trace_error=max(maximum_trace_error,float(abs(state[:len(system.QN)].sum(axis=0)-1).max()))
                    for column in state.T:
                        minimum_density_eigenvalue=min(minimum_density_eigenvalue,float(la.eigvalsh(layout.unpack(column),check_finite=False).min()))
                record={'z_fraction':fraction,'power_mW':float(prior.scan.POWERS[power]*1000),
                    'detuning_MHz':detuning,'method':best,'timings':timing,'extraction_s':extraction_s,
                    'structure':structure,'max_cached_photon_error':float(difference.max()),
                    'max_scaled_cached_error':float(np.max(difference/(1e-7+1e-5*abs(reference)))),
                    'max_full_photon_error':float(abs(photons-full).max()),
                    'max_packed_density_error':maximum_density_error,'max_trace_error':maximum_trace_error,
                    'min_density_eigenvalue':minimum_density_eigenvalue,
                    'min_photon_increment':float(np.diff(photons,axis=1).min())}
                assert record['max_scaled_cached_error']<1
                assert record['max_full_photon_error']<1e-7 and maximum_density_error<1e-7
                assert maximum_trace_error<1e-7 and minimum_density_eigenvalue>-1e-7
                assert record['min_photon_increment']>-1e-7
                record['passed']=True
                result['coverage'].append(record);save(result)
                np.savez_compressed(OUT/f'z{fraction*100:02.0f}_p{record["power_mW"]:g}_d{detuning:g}.npz',
                    initial_indices=indices,times_s=data['times'],optimized_photons=photons,full_photons=full)
                print(f'PASS: cached photons {difference.max():.3g}; full density {maximum_density_error:.3g}',flush=True)
        report=['# Dense propagator: eigen routines, matrix buffers, and physics coverage','',result['scope'],'',
            '## Eigen routines','',
            'One +6.8 MHz pure-X 10 mW point; 20 initial states, 1401 samples. Two repetitions in opposite method orders, one BLAS thread. Includes required copies, eigensystem, real LU, and photon projection; excludes diagnostics.','',
            '| Method | Median seconds | Speedup versus SciPy |','|---|---:|---:|']
        base=med([r for r in result['eigen_timings'] if r['method']=='scipy'],'total_s')
        for method in methods:
            duration=med([r for r in result['eigen_timings'] if r['method']==method],'total_s')
            report.append(f'| {method} | {duration:.3f} | {base/duration:.3f}x |')
        report.extend(['',f'Fastest in this small sample: {best}. Close differences are not evidence of a robust scan-wide win.',
            '', 'Native LAPACK DGEEV returns a real eigenbasis directly, avoiding complex eigenvector assembly. Reuse variant preallocates its Fortran input buffer. Explicit balancing uses a diagonal similarity transform and restores the eigenvectors to the original coordinates.',
            '', '## Matrix construction','',
            '100 assemblies per repetition, three repetitions. Input Jacobian values/topology already available; this measures dense allocation and filling only. Sparse topology preparation/extraction is excluded.',
            '', '| Method | Median milliseconds |','|---|---:|'])
        for method in ['full_then_slice','direct_reduced','reuse_reduced','reuse_scatter']:
            duration=med([r for r in result['matrix_timings'] if r['method']==method],'seconds_per_assembly')
            report.append(f'| {method} | {duration*1000:.3f} |')
        report.extend(['','## Power and polarization coverage','',
            '| Z power | Power (mW) | Detuning (MHz) | Maximum cached photon error | Maximum full-density error |',
            '|---:|---:|---:|---:|---:|'])
        for r in result['coverage']:
            report.append(f"| {r['z_fraction']*100:g}% | {r['power_mW']:g} | {r['detuning_MHz']:g} | {r['max_cached_photon_error']:.3g} | {r['max_packed_density_error']:.3g} |")
        report.extend(['','All eight selected points passed the same photon accuracy gate (1e-7 + 1e-5 times reference magnitude), full-density comparison, trace, positivity and cumulative-count monotonicity checks. Cached native-ODE references cover four initial states; full dense references and physical checks cover all 20 initial J2 projectors.',
            '', 'Scope: supplements prior 10 mW pure-X checks; does not validate other transitions, time-dependent drives, arbitrary coherent sink preparation, or ill-conditioned eigensystems. No full scan and no centrex_tlf solver changes.',
            '', 'Timing and validation details: [results.json](results.json). Reproduce with `.venv/Scripts/python.exe benchmarks/bench_r2_f4_dense_followup.py`.'])
        (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Finished {OUT / "REPORT.md"}',flush=True)


if __name__=='__main__': main()
