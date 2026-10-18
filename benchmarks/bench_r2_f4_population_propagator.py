"""Three-point benchmark only: reuse one dense eigensystem for four populations.

Keeps the completed scan's physics and all 1401 cumulative photon timestamps.
No production solver changes and no detuning scan. Run with the project venv.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from threadpoolctl import threadpool_info, threadpool_limits

REPO=Path(__file__).resolve().parent.parent
OBE=REPO/'reports/r2_f4_earth_field/obe'
OUT=Path(__file__).resolve().parent/'r2_f4_population_propagator_results'
sys.path.insert(0,str(OBE))
import scan
from centrex_tlf import lindblad
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py, prepare_lindblad_problem_py
from centrex_tlf.lindblad.ir import encode_runtime_value

DETUNINGS=[6.8,28.2,80.0]
POWER_INDEX=1


def cached_curve(detuning):
    for path in sorted(scan.DATA.glob('z00_????_????.npz')):
        with np.load(path) as chunk:
            indices=np.flatnonzero(np.isclose(chunk['detuning_uv_MHz'],detuning,atol=1e-9,rtol=0))
            if indices.size:
                return chunk['photons'][:,POWER_INDEX,indices[0],:].copy()
    raise ValueError(f'No saved reference for {detuning}')


def bind_point(prepared,rabi,detuning):
    graph=copy.deepcopy(prepared.parameter_graph)
    for name,value in [('rabi',rabi),('detuning',2*np.pi*1e6*detuning)]:
        index=graph['slot_names'].index(name)
        assert index<graph['n_base']
        graph['base_values'][index]=encode_runtime_value(value)
    payload=prepared.to_payload()
    payload['parameter_graph']=graph
    return replace(prepared,parameter_graph=graph,rust_plan=prepare_lindblad_problem_py(payload))


def dense_curves(prepared,rho,excited,times):
    timings={}
    tick=time.perf_counter()
    evaluator=create_lindblad_rhs_evaluator_py(prepared.rust_plan,'expanded_sparse',True)
    rows,cols,values=evaluator.jacobian_packed_sparse_py(0.,0.,'analytic')
    dim=prepared.layout.packed_len
    L=sp.csr_matrix((values,(rows,cols)),shape=(dim,dim)).toarray()
    packed=np.column_stack([prepared.layout.pack(r) for r in rho])
    timings['extract_seconds']=time.perf_counter()-tick
    # Validate at a second time and on all four initial states before timing eig.
    later=evaluator.jacobian_packed_sparse_py(float(times[-1])*.37,0.,'analytic')
    for first,second in zip((rows,cols,values),later):
        np.testing.assert_array_equal(first,second)
    for column in packed.T:
        rhs=np.asarray(evaluator.rhs_packed_py(np.ascontiguousarray(column),0.))
        np.testing.assert_allclose(L@column,rhs,atol=1e-10,rtol=1e-12)
    tick=time.perf_counter()
    eigenvalues,V=la.eig(L,check_finite=False)
    timings['eig_seconds']=time.perf_counter()-tick
    tick=time.perf_counter()
    lu,piv=la.lu_factor(V,check_finite=False)
    timings['lu_seconds']=time.perf_counter()-tick
    tick=time.perf_counter()
    coefficients=la.lu_solve((lu,piv),packed.astype(complex),check_finite=False)
    timings['four_initial_states_seconds']=time.perf_counter()-tick
    tick=time.perf_counter()
    wt=eigenvalues[:,None]*times[None,:]
    small=np.abs(wt)<1e-7
    denominator=np.where(np.abs(eigenvalues)>0,eigenvalues,1.)[:,None]
    phi=np.expm1(wt)/denominator
    series=np.broadcast_to(times,wt.shape)*(1+wt/2+wt**2/6)
    phi[small]=series[small]
    # Project only the photon observable, avoiding full density-matrix traces.
    observable=scan.GAMMA*V[excited].sum(axis=0)
    photons=(observable[:,None]*coefficients).T@phi
    timings['all_four_photon_traces_seconds']=time.perf_counter()-tick
    # Diagnostics excluded from production-cost timings.
    gecon=la.get_lapack_funcs('gecon',(lu,))
    rcond,info=gecon(lu,np.linalg.norm(V,1))
    assert info==0
    sample=np.linspace(0,dim-1,16,dtype=int)
    residual=np.linalg.norm(L@V[:,sample]-V[:,sample]*eigenvalues[sample])/(
        np.linalg.norm(L)*np.linalg.norm(V[:,sample]))
    n=prepared.layout.n
    traces=(V[:n].sum(axis=0)[:,None]*coefficients).T@np.exp(wt)
    diagnostics={'states':n,'packed_dimension':dim,'liouvillian_nnz':int(np.count_nonzero(L)),
        'estimated_eigenvector_condition_1norm':float(1/rcond),
        'sampled_eigen_residual':float(residual),
        'max_trace_error':float(np.max(abs(traces-1))),
        'max_photon_imaginary':float(np.max(abs(photons.imag))),
        'minimum_photons':float(photons.real.min()),
        'minimum_photon_increment':float(np.diff(photons.real,axis=1).min())}
    timings['total_seconds']=sum(timings.values())
    return photons.real,timings,diagnostics


def main():
    OUT.mkdir(exist_ok=True)
    results={'generated_at':datetime.now().astimezone().isoformat(),
        'scope':'Three detunings, pure X, 10 mW, four independent starting states, 1401 times through 350 us.',
        'detunings_MHz':DETUNINGS,'blas_threads':1,'reference_rayon_threads':4,
        'timing_note':'One timing per point. Dense uses one BLAS thread; independent reference uses four Rayon threads. Diagnostics excluded from dense total; common system build excluded from both.',
        'threadpools':threadpool_info(),'points':[]}
    with threadpool_limits(limits=1):
        built=scan.build(0.)
        system,prepared,rho,rabis,excited,metadata,_=built
        manifest=json.loads((scan.DATA/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['source_sha256']==scan.fingerprint(),'Source differs from completed scan'
        results['source_sha256']=manifest['source_sha256']
        results['system_build_prepare_seconds']=metadata['build_prepare_seconds']
        for detuning in DETUNINGS:
            bound=bind_point(prepared,rabis[POWER_INDEX],detuning)
            print(f'Detuning {detuning:g} MHz: dense factorization',flush=True)
            dense,timings,diagnostics=dense_curves(bound,rho,excited,scan.TIMES)
            cached=cached_curve(detuning)
            print(f'Dense total {timings["total_seconds"]:.3f} s; max cached error {np.max(abs(dense-cached)):.3g}. Timing four independent ODEs.',flush=True)
            tick=time.perf_counter()
            reference=lindblad.solve_lindblad_batch(bound,rho,(0.,float(scan.TIMES[-1])),
                solver='dopri5',execution_mode='expanded_sparse',parallel=True,threads=4,
                output='photon_integral',output_when='saveat',saveat=scan.TIMES,
                integral_weights=[(i,scan.GAMMA) for i in excited],integral_method='solver',
                dt=2e-9,reltol=1e-7,abstol=1e-9,maxiters=1000000,collect_stats=True)
            ode_seconds=time.perf_counter()-tick
            ref=np.asarray(reference.values).reshape(4,len(scan.TIMES)).real
            error=abs(dense-ref)
            passed=bool(np.all(error<=1e-7+1e-5*abs(ref)) and diagnostics['max_trace_error']<1e-7
                        and diagnostics['max_photon_imaginary']<1e-7
                        and diagnostics['minimum_photon_increment']>-1e-7)
            point={'detuning_MHz':detuning,'dense_timings':timings,'diagnostics':diagnostics,
                'independent_four_thread_ode_seconds':ode_seconds,
                'wall_speedup_vs_four_thread_ode':ode_seconds/timings['total_seconds'],
                'max_abs_photon_error':float(error.max()),
                'max_scaled_photon_error':float(np.max(error/(1e-7+1e-5*abs(ref)))),
                'max_ode_vs_cached_error':float(np.max(abs(ref-cached))),
                'max_dense_vs_cached_error':float(np.max(abs(dense-cached))),
                'passed_accuracy_gate':passed,'reference_solver_stats':reference.solver_stats}
            results['points'].append(point)
            np.savez_compressed(OUT/f'detuning_{detuning:g}.npz',times_s=scan.TIMES,
                components=np.array(scan.COMPONENTS),dense_photons=dense,ode_photons=ref,cached_photons=cached)
            (OUT/'results.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
            print(json.dumps(point,default=str),flush=True)
    dense_total=sum(p['dense_timings']['total_seconds'] for p in results['points'])
    ode_total=sum(p['independent_four_thread_ode_seconds'] for p in results['points'])
    report=['# Four-population dense-propagator benchmark','',results['scope'],'',
        'Reuses the existing exact-propagator approach. At each fixed parameter point, extract the real packed Liouvillian, eigendecompose once, solve for all four initial projectors together, and analytically integrate the photon observable at every saved time. No shared adaptive timestep controller and no physics approximation.','',
        results['timing_note'],'',
        '| UV detuning (MHz) | Dense total (s) | Four-thread independent ODE (s) | Wall speedup | Max photon error | Accuracy gate |',
        '|---:|---:|---:|---:|---:|---|']
    for p in results['points']:
        report.append(f"| {p['detuning_MHz']:g} | {p['dense_timings']['total_seconds']:.3f} | {p['independent_four_thread_ode_seconds']:.3f} | {p['wall_speedup_vs_four_thread_ode']:.2f}x | {p['max_abs_photon_error']:.3g} | {'PASS' if p['passed_accuracy_gate'] else 'FAIL'} |")
    report.extend(['',f'Three-point total: dense {dense_total:.3f} s; independent ODE {ode_total:.3f} s; ratio {ode_total/dense_total:.2f}x.',
        '', 'Accuracy gate: every cumulative photon value within 1e-7 + 1e-5 times the reference magnitude; trace error and imaginary photon residual below 1e-7; photon increments at least -1e-7. References are the current native solver-integral path, not post-solve trapezoids.',
        '', 'Limits: one timing per point, one polarization and power. This tests feasibility, not full-scan throughput. Production scans can run eight or sixteen independent detunings concurrently; dense parallelism and memory use need separate consideration. Dense factorization scales steeply with retained state count. No production backend changes were made.',
        '', 'Details and stages: [results.json](results.json). All four photon traces and the completed-scan references are saved in the point NPZs.',
        '', 'Reproduce: `.venv/Scripts/python.exe benchmarks/bench_r2_f4_population_propagator.py`.'])
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Finished: {OUT / "REPORT.md"}',flush=True)


if __name__=='__main__':
    main()
