"""Sequential output, packed-storage and integrating-factor GPU experiments."""
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from run_rc_gpu_integrator_suite import Suite,atomic

here=Path(__file__).resolve().parent
out=here/'rc_gpu_integrator_results/v3_20261012_135724'
deadline=datetime.fromisoformat('2026-10-12T16:37:10+00:00').timestamp()
s=Suite(out,deadline);s.started=datetime.fromisoformat('2026-10-12T13:57:24+00:00').timestamp()
base='swap_const_schedule_gentle';dt=250/56
def run(variant,**kw):
    opts=dict(gpu_class='storage_followup',integrator='tsit5',dt_ns=dt,variant=variant,
              batch=512,span_us=5,repeats=3)
    opts.update(kw);return s.run(**opts)
def full(variant,velocity=150,batch=128,step=dt,integrator='tsit5'):
    return run(variant,kind='full',velocity=velocity,batch=batch,dt_ns=step,
               integrator=integrator,span_us=44000/velocity,repeats=1)
try:
    s.stage='1 and 2: graph tails and GPU population sampling'
    outputs=[]
    for variant in [base,base+'_exact',base+'_devicepop',base+'_exact_devicepop']:
        r=full(variant)
        if s.passing(r):outputs.append(r)
    if not outputs:raise RuntimeError('All output variants failed')
    best=min(outputs,key=lambda r:r['median_s']);best_variant=best['job']['variant']
    s.winner=dict(integrator='tsit5',dt_ns=dt,variant=best_variant)
    s.kept=[best_variant];s.stage='3: packed Hermitian density storage'
    packed=[]
    for variant in [best_variant,best_variant+'_packed',best_variant+'_packed_direct']:
        r=run(variant)
        if s.passing(r):packed.append(r)
    bestshort=min(packed,key=lambda r:r['median_s'])
    if 'packed' in bestshort['job']['variant']:
        r=full(bestshort['job']['variant'])
        if s.passing(r):outputs.append(r)
    # Do not spend hours on invalid methods; every step size is independently
    # checked against the 483-case strict CPU near-peak reference.
    s.stage='4: diagonal integrating-factor Tsit5'
    ifpassed=[]
    for step in [dt,8.,12.,16.,24.]:
        r=run(best_variant,integrator='if_tsit5',dt_ns=step,repeats=1)
        if s.passing(r):ifpassed.append(r)
    if ifpassed:
        bestif=min(ifpassed,key=lambda r:r['median_s'])
        r=full(best_variant,step=bestif['job']['dt_ns'],integrator='if_tsit5')
        if s.passing(r):outputs.append(r)
    best=min(outputs,key=lambda r:r['median_s']);j=best['job']
    s.winner={k:j[k] for k in ['integrator','dt_ns','variant']};s.kept=[j['variant']]
    s.stage='selected method: velocity, batch and repeated timing confirmation'
    confirmations=[]
    for v in [50,250]:confirmations.append(full(j['variant'],velocity=v,batch=9,step=j['dt_ns'],integrator=j['integrator']))
    for batch in [128,1024]:
        confirmations.append(run(j['variant'],integrator=j['integrator'],dt_ns=j['dt_ns'],batch=batch))
    for _ in range(2):confirmations.append(full(j['variant'],step=j['dt_ns'],integrator=j['integrator']))
    s.stage='complete' if all(s.passing(r) for r in confirmations) else 'complete; selected setting failed confirmation'
    s.report();s.progress()
    state=json.loads((out/'progress.json').read_text());state['status']='complete';atomic(out/'progress.json',state)
except BaseException as error:
    s.stage=f'stopped: {error}';s.report();s.progress()
    state=json.loads((out/'progress.json').read_text());state['status']='budget_exhausted' if isinstance(error,TimeoutError) else 'failed'
    atomic(out/'progress.json',state)
    raise
