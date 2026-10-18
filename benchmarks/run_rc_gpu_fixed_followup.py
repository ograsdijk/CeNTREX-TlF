"""Bounded sequential suite; only archived CPU/finer GPU references are read."""
import json
import hashlib
import time
from datetime import datetime,timezone
from pathlib import Path
from run_rc_gpu_integrator_suite import Suite,atomic

here=Path(__file__).resolve().parent
out=here/'rc_gpu_integrator_results/v2_20261012_133710'
# Three hours including inspection and implementation, not three hours of jobs.
deadline=datetime.fromisoformat('2026-10-12T16:37:10+00:00').timestamp()
s=Suite(out,deadline)
s.started=datetime.fromisoformat('2026-10-12T13:37:10+00:00').timestamp()
s.stage='finer fixed-step grid'
manifest=json.loads((out/'manifest.json').read_text())
manifest['sources'].update({p:hashlib.sha256((here/p).read_bytes()).hexdigest() for p in
                          ['rc_gpu_fixed_followup.py','run_rc_gpu_fixed_followup.py']})
atomic(out/'manifest.json',manifest)
common=dict(gpu_class='fixed_followup',integrator='tsit5',batch=512,span_us=5,repeats=3)
def full(dt,variant,velocity=150,batch=128):
    return s.run(gpu_class='fixed_followup',kind='full',velocity=velocity,batch=batch,
                 integrator='tsit5',dt_ns=dt,variant=variant,span_us=44000/velocity,repeats=1)
try:
    candidates=[]
    for dt in [3.90625,250/60,250/56,5.0]:
        r=s.run(**common,dt_ns=dt)
        if s.passing(r):candidates.append(r)
    if not candidates:raise RuntimeError('No fixed timestep passed')
    best=min(candidates,key=lambda r:r['median_s']);dt=best['job']['dt_ns']
    s.winner=dict(integrator='tsit5',dt_ns=dt)
    s.stage='fixed-step bookkeeping ablations'
    variants=[best]
    for variant in ['swap','const','swap_const']:
        r=s.run(**common,dt_ns=dt,variant=variant)
        if s.passing(r):variants.append(r)
    winner=min(variants,key=lambda r:r['median_s'])
    variant=winner['job']['variant']
    s.kept=[variant]
    s.stage='matched full-transit baselines and schedule tests'
    baseline=full(3.90625,'base')
    tuned=full(dt,'base')
    improved=full(dt,variant) if variant!='base' else tuned
    finalists=[r for r in [baseline,tuned,improved] if s.passing(r)]
    for schedule in ['schedule_gentle','schedule_strong']:
        combined=variant+'_'+schedule if variant!='base' else schedule
        r=full(dt,combined)
        if s.passing(r):finalists.append(r)
    if not finalists:raise RuntimeError('No full-transit finalist')
    fullwinner=min(finalists,key=lambda r:r['median_s'])
    chosen_dt=fullwinner['job']['dt_ns'];chosen_variant=fullwinner['job']['variant']
    s.winner=dict(integrator='tsit5',dt_ns=chosen_dt,variant=chosen_variant)
    s.stage='velocity and batch confirmation'
    confirmations=[]
    for velocity in [50,250]:confirmations.append(full(chosen_dt,chosen_variant,velocity,batch=9))
    for batch in [128,1024]:
        s.run(**dict(common,batch=batch),dt_ns=chosen_dt,variant=chosen_variant)
    # Repeat the representative transit independently to assess timing stability.
    repeats=[fullwinner]
    for _ in range(2):repeats.append(full(chosen_dt,chosen_variant))
    s.kept=[chosen_variant]
    s.stage='complete' if all(s.passing(r) for r in confirmations+repeats) else 'complete; selected setting failed a confirmation'
    s.report();s.progress()
    state=json.loads((out/'progress.json').read_text());state['status']='complete';atomic(out/'progress.json',state)
except BaseException as error:
    s.stage=f'stopped: {error}';s.report();s.progress()
    state=json.loads((out/'progress.json').read_text());state['status']='budget_exhausted' if isinstance(error,TimeoutError) else 'failed'
    atomic(out/'progress.json',state)
    raise
