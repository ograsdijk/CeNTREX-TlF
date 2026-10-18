"""Confirm fusion finalists against the original winning SIMD implementation."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import check_block_simd_dsu as harness
import check_fused_dsu as fusion
import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

HERE=Path(__file__).resolve().parent
RESULTS=fusion.RESULTS


def run():
    original_patch=harness.patch_rhs
    harness.CACHE=fusion.CACHE
    control,ch=harness.build(False,module_prefix='_fusion_confirm_')
    old,oh=harness.build(True,module_prefix='_fusion_original_simd_')
    harness.patch_rhs=fusion.patch_rhs
    fused,fh=harness.build(True,module_prefix='_fusion_finalists_')
    harness.patch_rhs=original_patch
    modes={'production':(control,0),'original_simd':(old,2),'decay_pack_simd':(fused,14),
           'all_three_simd':(fused,15),'all_three_scalar':(fused,7)}
    report={'rounds':8,'sources':{'production':ch,'original_simd':oh,'fusion':fh},'systems':{}}
    def save():
        (RESULTS/'confirmation.json').write_text(json.dumps(report,indent=2))
    for name in ['rc','r2','r0','q1']:
        # Identical serialized physical input from the completed factorial run;
        # only kernel selection differs. No symbolic rebuild or changed recipe.
        data=(fusion.CACHE/(name+'_fixture.pkl')).read_bytes()
        cached=pickle.loads(data)
        payload,rho0,_=cached['data']
        n=len(rho0)
        layout=PackedHermitianLayout(n)
        payload['benchmark_support']=list(range(n*n))
        y0=layout.pack(rho0)
        reference=layout.unpack(np.asarray(control.trajectory(payload,y0,0,10e-6,1e-13,1e-11)[0]))
        row={'n':n,'fixture_sha256':hashlib.sha256(data).hexdigest(),
             'samples':{key:[] for key in modes},'orders':[]}
        report['systems'][name]=row
        def solve(key,payload=payload,y0=y0,layout=layout,reference=reference,name=name):
            native,flag=modes[key]
            out=native.trajectory(payload,y0,flag,10e-6,1e-11,1e-9)
            rho=layout.unpack(np.asarray(out[0]))
            accuracy=harness.base.accuracy(rho,reference)
            trace=float(abs(np.trace(rho)-1))
            assert accuracy['passed'] and trace<1e-9,(name,key,accuracy,trace)
            return {'seconds':out[1],'rhs_calls':out[2],'accepted':out[3],'rejected':out[4],
                    'accuracy':accuracy,'trace_error':trace}
        for key in modes:
            solve(key)
        rng=np.random.default_rng(2029+n)
        for repeat in range(8):
            order=list(modes)
            rng.shuffle(order)
            row['orders'].append(order)
            print(f'{name}: confirmation {repeat+1}/8',flush=True)
            for key in order:
                row['samples'][key].append(solve(key))
            save()
    print('Confirmation complete',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
