"""FMA, wider vectors and vectorized finalization versus the winning fused kernel."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import check_block_simd_dsu as harness
import check_fused_dsu as fusion
import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

HERE=Path(__file__).resolve().parent
CACHE=harness.base.REPO/'.cache/benchmarks/extended_simd_dsu_v1'
RESULTS=HERE/'rust_specialized_rhs_results/extended_simd_dsu'
ROUNDS=8
ISA=r'''
#[pyfunction]
fn isa()->Vec<bool> {
    #[cfg(target_arch="x86_64")] {
        vec![std::is_x86_feature_detected!("avx2"),std::is_x86_feature_detected!("fma"),std::is_x86_feature_detected!("avx512f")]
    }
    #[cfg(not(target_arch="x86_64"))] {vec![false,false,false]}
}
'''


def patch_rhs(source):
    source=fusion.patch_rhs(source)
    source=source.replace('bench_fusion_flags:usize,','bench_fusion_flags:usize,\n    bench_hdiag:Vec<f64>,',1)
    source=source.replace('bench_fusion_flags:0,','bench_fusion_flags:0, bench_hdiag:Vec::new(),',1)
    source=source.replace('w.bench_fusion_flags=flags;',r'''
    #[cfg(target_arch="x86_64")] {
        assert!(flags&16==0 || (std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2")));
        assert!(flags&32==0 || (std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2")));
        assert!(flags&64==0 || std::is_x86_feature_detected!("avx2"));
    }
    w.bench_hdiag=vec![0.0;w.upper_layout.n];
    w.bench_fusion_flags=flags;''',1)
    call='bench_row(h,&w.rho[k*n..(k+1)*n],&mut w.scratch_re[i*n..(i+1)*n],&mut w.scratch_im[i*n..(i+1)*n],simd);'
    assert source.count(call)==2
    source=source.replace(call,call.replace('bench_row(','ext_dispatch(').replace(',simd);',',simd,flags);'))
    marker='    let mut cursor=n;\n    for i in 0..n {'
    assert source.count(marker)==1
    source=source.replace(marker,'''    if flags&64!=0 {
        assert!(pack && fused_decay && !combine);
        #[cfg(target_arch="x86_64")]
        unsafe {ext_finalize(plan,w,packed.as_deref_mut().unwrap());return;}
    }
'''+marker)
    return source+'\n'+(HERE/'extended_simd_dsu.rs').read_text()


def run():
    harness.CACHE=CACHE
    CACHE.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    print('Building matched winning baseline and extended SIMD kernels',flush=True)
    harness.patch_rhs=fusion.patch_rhs
    baseline,bh=harness.build(True,module_prefix='_extended_simd_baseline_')
    harness.patch_rhs=patch_rhs
    experiment,eh=harness.build(True,module_prefix='_extended_simd_options_',extra_rust=ISA,exported_functions=('isa',))
    avx2,fma,wide=experiment.isa()
    print(f'CPU features: AVX2={avx2}, FMA={fma}, AVX512F={wide}',flush=True)
    if '--build-only' in sys.argv:
        return
    modes={'previous_best':14,'hooked_control':14}
    skipped={}
    for label,flags in [('fma',16),('avx512',32),('fma_avx512',48),('finalize',64),
                        ('fma_finalize',80),('avx512_finalize',96),('all_three',112)]:
        if (flags&16 and not(fma and avx2)) or (flags&32 and not(wide and fma and avx2)) or (flags&64 and not avx2):
            skipped[label]='required ISA unavailable'
        else:
            modes[label]=14|flags
    files=[Path(__file__),HERE/'extended_simd_dsu.rs',HERE/'fused_dsu.rs',HERE/'block_simd_dsu.rs',HERE/'check_fused_dsu.py',HERE/'check_block_simd_dsu.py']
    files += [harness.base.REPO/p for p in ['rust/src/lindblad/plan.rs','rust/src/lindblad/eval.rs','rust/src/ode/tsit5.rs']]
    report={'rounds':ROUNDS,'features':{'avx2':avx2,'fma':fma,'avx512f':wide},'modes':modes,'skipped':skipped,
            'sources':{'previous_best':bh,'experiment':eh},
            'source_files':{str(p.relative_to(harness.base.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'systems':{}}
    def save():
        (RESULTS/'results.json').write_text(json.dumps(report,indent=2))
    for name in ['rc','r2','r0','q1']:
        print(name+': validating densities',flush=True)
        data=(fusion.CACHE/(name+'_fixture.pkl')).read_bytes()
        payload,rho0,_=pickle.loads(data)['data']
        payload['benchmark_support']=[]
        n=len(rho0)
        layout=PackedHermitianLayout(n)
        y0=layout.pack(rho0)
        rng=np.random.default_rng(712+n)
        rhs_error=0.0
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
                pa,pb=harness.base.driver.switch_values(float(t))
                for idx,slot in enumerate(graph['slot_names'][:len(graph['base_values'])]):
                    if slot in ['PA1','PY1','PA2','PY2']:
                        gate=pa if slot.endswith('1') else pb
                        graph['base_values'][idx]['re']=gate if slot.startswith('PA') else 1-gate
                expected=np.asarray(baseline.rhs_check(p,y,float(t),14))
                for flag in set(modes.values()):
                    actual=np.asarray(experiment.rhs_check(p,y,float(t),flag))
                    error=float(np.max(abs(actual-expected))/max(1,float(np.max(abs(expected)))))
                    rhs_error=max(rhs_error,error)
                    assert error<1e-10,(name,flag,error)
        reference=layout.unpack(np.asarray(baseline.trajectory(payload,y0,14,10e-6,1e-13,1e-11)[0]))
        row={'n':n,'fixture_sha256':hashlib.sha256(data).hexdigest(),'max_normalized_rhs_error':rhs_error,
             'samples':{k:[] for k in modes},'orders':[]}
        report['systems'][name]=row
        def solve(key,payload=payload,y0=y0,layout=layout,reference=reference,name=name):
            native=baseline if key=='previous_best' else experiment
            out=native.trajectory(payload,y0,modes[key],10e-6,1e-11,1e-9)
            rho=layout.unpack(np.asarray(out[0]))
            accuracy=harness.base.accuracy(rho,reference)
            trace=float(abs(np.trace(rho)-1))
            assert accuracy['passed'] and trace<1e-9,(name,key,accuracy,trace)
            return {'seconds':out[1],'rhs_calls':out[2],'accepted':out[3],'rejected':out[4],
                    'accuracy':accuracy,'trace_error':trace}
        for key in modes:
            solve(key)
        for repeat in range(ROUNDS):
            order=list(modes)
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
