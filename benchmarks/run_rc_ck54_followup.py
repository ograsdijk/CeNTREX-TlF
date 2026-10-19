"""RC-specific CK54 accuracy refinement and matched full-transit comparisons."""
from pathlib import Path
from datetime import datetime
import json,subprocess
from run_gpu_next_suite import Runner,passing,atomic

def main():
    old=Path('benchmarks/gpu_next_results/v1_20261012_174509')
    out=Path('benchmarks/gpu_next_results')/('rc_ck54_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    runner=Runner(out)
    subprocess.Popen(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(Path('benchmarks/gpu_next_progress.ps1').resolve()),
                      '-ProgressPath',str((out/'progress.json').resolve())],creationflags=subprocess.CREATE_NEW_CONSOLE)
    common=dict(system='rc',integrator='ck54',variant='swap_const_schedule_gentle_exact_devicepop_packed',threads=256,tile=32)
    runner.stage='RC CK54: refine largest accurate step';runner.progress()
    rows=[runner.run(**common,dt_ns=h,repeats=1,waveform=True) for h in [1.2,1.3,1.35,1.4]]
    valid=[r for r in rows if passing(r)]
    if not valid:raise RuntimeError('No accurate CK54 screen setting')
    best=min(valid,key=lambda r:r['median_s']);h=best['job']['dt_ns']
    runner.stage='RC CK54: tune staged/direct RHS and waveform precomputation';runner.progress()
    rows=[best]
    rows.append(runner.run(**common,dt_ns=h,repeats=3,waveform=False))
    rows.append(runner.run(**common,dt_ns=h,repeats=3,waveform=True))
    for th in [128,256]:
        rows.append(runner.run(**dict(common,threads=th,tile=0,variant=common['variant']+'_direct'),dt_ns=h,repeats=3,waveform=True))
    best=min([r for r in rows if passing(r)],key=lambda r:r['median_s'])
    cfg={k:best['job'][k] for k in ['system','integrator','variant','threads','tile','dt_ns','waveform']}
    atomic(out/'selected.json',dict(settings=cfg,short_accuracy=best['accuracy']))
    # Fresh Tsit5 comparison includes identical output transfer and batch size.
    tsit=dict(system='rc',integrator='tsit5',variant=common['variant'],threads=256,tile=32,dt_ns=250/56,waveform=True)
    runner.stage='RC CK54: same-session short baseline';runner.progress()
    runner.run(**tsit,repeats=3)
    runner.stage='RC CK54: full density plus photons, 50/150/250 m/s';runner.progress()
    prior=json.loads((old/'results.json').read_text())
    for v in [150.,50.,250.]:
        ref=next(r for r in prior if r.get('status')=='complete' and r['job']['system']=='rc'
                 and r['job']['kind']=='full_reference' and r['job'].get('velocity')==v)
        reference=str(Path(ref['job']['output']).with_suffix('.npz'))
        args=dict(kind='full',velocity=v,batch=9,repeats=1,photons=True,reference=reference,photon_reference=reference)
        runner.run(**cfg,**args)
        runner.run(**cfg,**args,output_mode='photon_only')
        runner.run(**tsit,**args)
    runner.stage='RC CK54: matched 128-trajectory full-density throughput';runner.progress()
    runner.run(**cfg,kind='full',velocity=150.,batch=128,repeats=1)
    runner.run(**tsit,kind='full',velocity=150.,batch=128,repeats=1)
    runner.stage='complete';runner.report();runner.progress('complete')

if __name__=='__main__':main()
