"""Matched full-solve factorial tests of diagonal/decay/packing fusions."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import bench_rust_specialized_rhs as base
import check_block_simd_dsu as harness
import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

HERE=Path(__file__).resolve().parent
CACHE=base.REPO/'.cache/benchmarks/fused_dsu_v1'
RESULTS=HERE/'rust_specialized_rhs_results/fused_dsu'
ROUNDS=8
MODES={'production':0,'hooked_control':0,'simd':8}
for label,flags in [('decay',2),('pack',4),('decay_pack',6),('diagonal_decay',3),('diagonal_decay_pack',7)]:
    MODES[label]=flags
    MODES['simd_'+label]=flags|8


def patch_rhs(source):
    source=source.replace('pub struct RhsWorkspace {','pub struct RhsWorkspace {\n    bench_fusion_flags:usize,',1)
    source=source.replace('upper_layout: upper_layout.clone(),','bench_fusion_flags:0,\n            upper_layout: upper_layout.clone(),',1)
    start=source.index('fn rhs_from_workspace_rho(')
    end=source.index('pub fn rhs_matrix_into_with_profile(',start)
    body=source[start:end]
    body=body.replace('mut profile: Option<&mut RhsProfileStats>,','mut profile: Option<&mut RhsProfileStats>,\n    mut packed_output:Option<&mut[f64]>,',1)
    marker='if mode.uses_soa_scratch() {\n                add_commutator_sparse_dense_upper_soa('
    assert body.count(marker)==1
    body=body.replace(marker,'''if workspace.bench_fusion_flags != 0 {
                bench_fused_dsu(plan,workspace,packed_output.as_deref_mut());
            } else '''+marker)
    marker='let dissipator_start = profile_timer(&profile);\n    match mode {'
    assert body.count(marker)==1
    body=body.replace(marker,'''let dissipator_start = profile_timer(&profile);
    let fused=mode==ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack
        && workspace.bench_fusion_flags&6!=0;
    if !fused {match mode {''')
    marker='    if let Some(stats) = profile.as_mut() {\n        stats.calls += 1;'
    assert body.count(marker)==1
    body=body.replace(marker,'    }\n'+marker)
    source=source[:start]+body+source[end:]
    source=source.replace('rhs_from_workspace_rho(plan, t, mode, workspace, profile)?;',
                          'rhs_from_workspace_rho(plan, t, mode, workspace, profile, None)?;')
    marker='rhs_from_workspace_rho(plan, t, mode, workspace, profile.as_deref_mut())?;'
    assert source.count(marker)==1
    source=source.replace(marker,'''rhs_from_workspace_rho(plan,t,mode,workspace,profile.as_deref_mut(),Some(out))?;
    if mode==ExecutionMode::ExperimentalStructuredSparseDenseDiagSoaSingleUnpack
        && workspace.bench_fusion_flags&4!=0 {return Ok(());}''')
    helpers=(HERE/'block_simd_dsu.rs').read_text()
    helpers=helpers[helpers.index('#[inline]\nfn bench_scalar_row'):helpers.index('fn bench_simd_dsu(')]
    return source+'\n'+helpers+'\n'+(HERE/'fused_dsu.rs').read_text()


def run():
    harness.CACHE=CACHE
    harness.patch_rhs=patch_rhs
    CACHE.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    print('Building current-source controls and fused kernels',flush=True)
    control,ch=harness.build(False)
    hooks,hh=harness.build(True)
    if '--build-only' in sys.argv:
        return
    files=[Path(__file__),HERE/'fused_dsu.rs',HERE/'block_simd_dsu.rs',HERE/'check_block_simd_dsu.py']
    files += [base.REPO/p for p in ['rust/src/lindblad/plan.rs','rust/src/lindblad/eval.rs','rust/src/ode/tsit5.rs']]
    report={'rounds':ROUNDS,'modes':MODES,'sources':{'production':ch,'hooks':hh},
            'source_files':{str(p.relative_to(base.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'systems':{}}
    def save():
        (RESULTS/'results.json').write_text(json.dumps(report,indent=2))
    for name in ['rc','r2','r0','q1']:
        print(name+': preparing fixture',flush=True)
        payload,rho0,_=harness.fixture(name)
        n=len(rho0)
        layout=PackedHermitianLayout(n)
        payload['benchmark_support']=[]
        y0=layout.pack(rho0)
        rng=np.random.default_rng(820+n)
        rhs_error=0.0
        # Full arbitrary coherences, changed powers/detunings and all gate states.
        import copy
        for scale,detuning in [(0.0,-2.3e6),(0.3,0.0),(1.0,0.0),(2.0,4.1e6)]:
            p=copy.deepcopy(payload)
            graph=p['parameter_graph']
            for idx,slot in enumerate(graph['slot_names'][:len(graph['base_values'])]):
                if slot.startswith('Ωt'):
                    graph['base_values'][idx]['re']*=scale
                if slot.startswith('δ'):
                    graph['base_values'][idx]['re']=2*np.pi*detuning
            a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
            a=a@a.conj().T
            y=layout.pack(a/np.trace(a))
            for t in np.array([125,375,625,875])*1e-9:
                pa,pb=base.driver.switch_values(float(t))
                for idx,slot in enumerate(graph['slot_names'][:len(graph['base_values'])]):
                    if slot in ['PA1','PY1','PA2','PY2']:
                        gate=pa if slot.endswith('1') else pb
                        graph['base_values'][idx]['re']=gate if slot.startswith('PA') else 1-gate
                expected=np.asarray(control.rhs_check(p,y,float(t),0))
                for flag in set(MODES.values()):
                    actual=np.asarray(hooks.rhs_check(p,y,float(t),flag))
                    err=float(np.max(abs(actual-expected))/max(1,float(np.max(abs(expected)))))
                    rhs_error=max(rhs_error,err)
                    assert err<1e-10,(name,flag,err)
        reference=layout.unpack(np.asarray(control.trajectory(payload,y0,0,10e-6,1e-13,1e-11)[0]))
        row={'n':n,'max_normalized_rhs_error':rhs_error,'samples':{k:[] for k in MODES},'orders':[]}
        report['systems'][name]=row
        def solve(key,payload=payload,y0=y0,layout=layout,reference=reference,name=name):
            native=control if key=='production' else hooks
            out=native.trajectory(payload,y0,MODES[key],10e-6,1e-11,1e-9)
            rho=layout.unpack(np.asarray(out[0]))
            error=base.accuracy(rho,reference)
            trace=float(abs(np.trace(rho)-1))
            assert error['passed'] and trace<1e-9,(name,key,error,trace)
            return {'seconds':out[1],'rhs_calls':out[2],'accepted':out[3],'rejected':out[4],'accuracy':error,'trace_error':trace}
        for key in MODES:
            solve(key)
        for repeat in range(ROUNDS):
            order=list(MODES)
            rng.shuffle(order)
            row['orders'].append(order)
            print(f'{name}: round {repeat+1}/{ROUNDS}',flush=True)
            for key in order:
                row['samples'][key].append(solve(key))
            save()
    print('Complete',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
