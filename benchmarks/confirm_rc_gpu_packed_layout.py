"""Compare packed staged/direct layouts on the representative full transit."""
import json
import time
from pathlib import Path
from datetime import datetime
from run_rc_gpu_integrator_suite import Suite,atomic

out=Path(__file__).resolve().parent/'rc_gpu_integrator_results/v3_20261012_135724'
deadline=datetime.fromisoformat('2026-10-12T16:37:10+00:00').timestamp()
while time.time()<deadline:
    state=json.loads((out/'progress.json').read_text())
    if state['status']!='running':break
    time.sleep(3)
else:raise TimeoutError('Existing benchmark deadline reached')
if state['stage']!='complete':raise RuntimeError(f"Parent suite stopped: {state}")
s=Suite.__new__(Suite);s.out=out;s.deadline=deadline;s.started=time.time()-state['elapsed_s']
s.results=json.loads((out/'results.json').read_text());s.count=len(s.results);s.current=None
s.winner=state['winner'];s.kept=state['kept_optimizations'];s.stage='packed staged/direct full-transit cross-check';s.progress()
try:
    variant='swap_const_schedule_gentle_exact_devicepop_packed'
    opts=dict(gpu_class='storage_followup',integrator='tsit5',variant=variant,dt_ns=250/56,
              kind='full',velocity=150,batch=128,span_us=44000/150,repeats=1)
    r=s.run(**opts)
    existing=[x for x in s.results[:-1] if x['job'].get('kind')=='full' and x['job']['velocity']==150 and
              x['job']['variant']==s.winner['variant'] and s.passing(x)]
    previous=min(x['median_s'] for x in existing)
    if s.passing(r) and r['median_s']<.97*previous:
        confirmation=[]
        for _ in range(2):confirmation.append(s.run(**opts))
        for velocity in [50,250]:confirmation.append(s.run(**dict(opts,velocity=velocity,batch=9,span_us=44000/velocity)))
        if all(s.passing(x) for x in confirmation):
            s.winner=dict(integrator='tsit5',dt_ns=250/56,variant=variant);s.kept=[variant]
    s.stage='complete';s.report();s.progress()
    state=json.loads((out/'progress.json').read_text());state['status']='complete';atomic(out/'progress.json',state)
except BaseException as error:
    s.stage=f'stopped: {error}';s.report();s.progress()
    state=json.loads((out/'progress.json').read_text());state['status']='failed';atomic(out/'progress.json',state)
    raise
