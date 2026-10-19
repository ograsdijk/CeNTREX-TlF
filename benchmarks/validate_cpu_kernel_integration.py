"""Matched full-density validation of the integrated CPU dispatch and kernels."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import check_block_simd_dsu as harness
import check_fused_dsu as fixtures
import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf import centrex_tlf_rust as rust
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

ROOT=Path(__file__).resolve().parent
CACHE=harness.base.REPO/'.cache/benchmarks/cpu_kernel_integration_v1'
RESULTS=ROOT/'rust_specialized_rhs_results/cpu_kernel_integration'


def run():
    harness.CACHE=CACHE
    RESULTS.mkdir(parents=True,exist_ok=True)
    # One current-source generic binary for every kernel and model. The stub
    # benchmark configure call is inert; all dispatch comes from production code.
    native,hashes=harness.build(False,module_prefix='_integrated_cpu_kernel_')
    kernels=["original","auto","avx2","avx2_fma","avx512","avx512_fma"]
    report={'rounds':8,'sources':hashes,'systems':{},'source_files':{}}
    for name in ['cpu_kernel','rhs','plan','eval']:
        path=harness.base.REPO/f'rust/src/lindblad/{name}.rs'
        report['source_files'][str(path.relative_to(harness.base.REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        (RESULTS/'results.json').write_text(json.dumps(report,indent=2))
    for name in ['rc','r2','r0','q1']:
        fixture_path=fixtures.CACHE/(name+'_fixture.pkl')
        if fixture_path.exists():
            payload,rho0,_=pickle.loads(fixture_path.read_bytes())['data']
        else:
            # Rebuild the documented recipe on a fresh checkout; cached inputs
            # are a convenience, not a prerequisite for running validation.
            payload,rho0,_=harness.fixture(name)
        payload['benchmark_support']=[]
        payload['cpu_kernel']='original'
        layout=PackedHermitianLayout(len(rho0))
        y=layout.pack(rho0)
        rng=np.random.default_rng(730+len(rho0))
        a=rng.normal(size=rho0.shape)+1j*rng.normal(size=rho0.shape)
        a=a@a.conj().T
        probe=layout.pack(a/np.trace(a))
        reference=layout.unpack(np.asarray(native.trajectory(payload,y,0,10e-6,1e-13,1e-11)[0]))
        row={'n':len(rho0),'resolved':{},'samples':{},'orders':[],'max_normalized_rhs_error':0.0}
        report['systems'][name]=row
        choices={}
        for kernel in kernels:
            p=dict(payload,cpu_kernel=kernel)
            try:
                plan=rust.prepare_lindblad_problem_py(p)
            except ValueError as error:
                if 'not supported by this CPU/OS' in str(error):
                    continue
                raise
            row['resolved'][kernel]=plan.resolved_cpu_kernel
            for t in np.array([125,375,625,875])*1e-9:
                expected=np.asarray(native.rhs_check(payload,probe,float(t),0))
                actual=np.asarray(native.rhs_check(p,probe,float(t),0))
                error=float(np.max(abs(actual-expected))/max(1,float(np.max(abs(expected)))))
                row['max_normalized_rhs_error']=max(row['max_normalized_rhs_error'],error)
                assert error<1e-10,(name,kernel,error)
            choices[kernel]=p
            row['samples'][kernel]=[]
        def solve(kernel,y=y,reference=reference,layout=layout,name=name,choices=choices):
            out=native.trajectory(choices[kernel],y,0,10e-6,1e-11,1e-9)
            rho=layout.unpack(np.asarray(out[0]))
            accuracy=harness.base.accuracy(rho,reference)
            trace=float(abs(np.trace(rho)-1))
            assert accuracy['passed'] and trace<1e-9,(name,kernel,accuracy,trace)
            return {'seconds':out[1],'rhs_calls':out[2],'accepted':out[3],'rejected':out[4],
                    'accuracy':accuracy,'trace_error':trace}
        for kernel in choices:
            solve(kernel)
        for repeat in range(8):
            order=list(choices)
            rng.shuffle(order)
            row['orders'].append(order)
            print(f'{name}: integrated kernel round {repeat+1}/8',flush=True)
            for kernel in order:
                row['samples'][kernel].append(solve(kernel))
            save()
    print('Integrated CPU validation complete',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
