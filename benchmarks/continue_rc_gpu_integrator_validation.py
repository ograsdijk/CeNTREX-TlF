"""Finish endpoint validation after the sequential selection suite exits."""
import json
import time
from datetime import datetime
from pathlib import Path
from run_rc_gpu_integrator_suite import Suite, atomic

out = Path(__file__).resolve().parent / 'rc_gpu_integrator_results/v1_20261012_052500'
deadline = datetime.fromisoformat(json.loads((out / 'manifest.json').read_text())['deadline_utc']).timestamp()
while time.time() < deadline:
    state = json.loads((out / 'progress.json').read_text())
    if state['status'] != 'running':
        break
    time.sleep(3)
else:
    raise TimeoutError('Original benchmark deadline reached')
if state['status'] != 'complete':
    raise RuntimeError(f"Selection suite stopped: {state['status']}")
s = Suite.__new__(Suite)
s.out, s.deadline = out, deadline
s.started = time.time() - state['elapsed_s']
s.results = json.loads((out / 'results.json').read_text())
s.count, s.current = len(s.results), None
s.winner, s.kept = state['winner'], state['kept_optimizations']
s.stage = 'full-transit endpoint and population-history confirmation'
s.progress()
try:
    for method, dt, velocity, batch in [('tsit5', 3.90625, 150, 128),
                                        ('rk4', 1.953125, 150, 128),
                                        ('tsit5', 3.90625, 50, 9),
                                        ('tsit5', 3.90625, 250, 9)]:
        r = s.run(kind='full', velocity=velocity, batch=batch,
                  integrator=method, dt_ns=dt, span_us=44000 / velocity, repeats=1)
        if not s.passing(r):
            s.run(kind='full', velocity=velocity, batch=batch,
                  integrator=method, dt_ns=dt / 2, span_us=44000 / velocity, repeats=1)
    s.stage = 'complete'
    s.report()
    s.progress()
    state = json.loads((out / 'progress.json').read_text())
    state['status'] = 'complete'
    atomic(out / 'progress.json', state)
except BaseException as error:
    s.stage = f'Endpoint validation stopped: {error}'
    s.report()
    s.progress()
    state = json.loads((out / 'progress.json').read_text())
    state['status'] = 'failed'
    atomic(out / 'progress.json', state)
    raise
