"""Serial follow-up after the main suite; never contend for the GPU."""
from pathlib import Path
import json, time, subprocess
from run_gpu_next_suite import Runner, passing, atomic
from continue_gpu_next_suite import settings, archive

def main():
    out=Path('benchmarks/gpu_next_results/v1_20261012_174509')
    while True:
        try:state=json.loads((out/'progress.json').read_text())
        except (PermissionError,json.JSONDecodeError):time.sleep(.2);continue
        if state['status']=='complete':break
        if time.time()-(out/'progress.json').stat().st_mtime>120:
            raise RuntimeError('Main runner stopped updating; inspect it before continuing.')
        time.sleep(3)
    runner=Runner(out,resume=True)
    subprocess.Popen(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(Path('benchmarks/gpu_next_progress.ps1').resolve()),
                      '-ProgressPath',str((out/'progress.json').resolve())],creationflags=subprocess.CREATE_NEW_CONSOLE)
    runner.stage='11: corrected waveform full-transit confirmation';runner.progress()
    winner_jobs=json.loads((out/'screen_winners.json').read_text())
    for v,batch in [(150.,128),(50.,9),(250.,9)]:
        cfg={k:winner_jobs['rc'][k] for k in ['system','variant','span_us','threads','tile','dt_ns','waveform']}
        runner.run(**cfg,kind='full',velocity=v,batch=batch,repeats=1,validation_revision='physical_time_lookup')
    runner.stage='12: R(2) low-storage step refinement';runner.progress()
    w=json.loads((out/'screen_winners.json').read_text())['r2'];cfg={k:w[k] for k in ['system','variant','span_us','threads','tile']}
    candidates=[]
    for h in [1.75,1.875,1.9,1.95]:
        r=runner.run(**cfg,integrator='ck54',dt_ns=h,repeats=3)
        if passing(r):candidates.append(r)
    if candidates:
        best=min(candidates,key=lambda r:r['median_s'])
        w=runner.run(**dict(settings(best),waveform=True),integrator='ck54',repeats=3) if not best['job']['waveform'] else best
        if passing(w) and w['median_s']<best['median_s']:best=w
        runner.stage='13: low-storage full EOM and photon checks';runner.progress()
        cfg=settings(best)
        # One independent coherent/PSD RHS check and one analytic photon decay
        # check are included by every worker; then compare long paths to fine GPU.
        for v in [150.,50.,250.]:
            refs=[r for r in runner.results if r.get('status')=='complete' and r['job']['system']=='r2'
                  and r['job']['kind']=='full_reference' and r['job'].get('velocity')==v]
            if not refs:continue
            ref=refs[0]
            result=runner.run(**cfg,integrator='ck54',kind='full',velocity=v,batch=60,photons=True,repeats=1,
                              reference=archive(ref),photon_reference=archive(ref))
            if passing(result):
                runner.run(**cfg,integrator='ck54',kind='full',velocity=v,batch=60,photons=True,output_mode='photon_only',
                           repeats=1,reference=archive(ref),photon_reference=archive(ref))
    runner.stage='14: cached static photon integrals versus saved complete curves';runner.progress()
    runner.run(system='r2',kind='dense_photons')
    runner.stage='complete';runner.report();runner.progress('complete')

if __name__=='__main__':main()
