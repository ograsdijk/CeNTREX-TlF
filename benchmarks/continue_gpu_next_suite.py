"""Resume isolated GPU experiments, including output-specific validation."""
from pathlib import Path
import json
from run_gpu_next_suite import Runner, atomic, passing

KEYS=['system','variant','span_us','threads','tile','dt_ns','waveform']
def settings(r):return {k:r['job'][k] for k in KEYS}
def archive(r):return str(Path(r['job']['output']).with_suffix('.npz'))

def main():
    out=Path('benchmarks/gpu_next_results/v1_20261012_174509')
    runner=Runner(out,resume=True)
    def stage(name):runner.stage=name;runner.progress()
    # Completed output files are reusable after an interrupted parent/monitor.
    def run(**kw):
        for r in runner.results:
            if r.get('status')=='complete' and all(r['job'].get(k)==v for k,v in kw.items()):return r
        return runner.run(**kw)
    best={s:min([r for r in runner.results[:20] if r['job']['system']==s and passing(r)],key=lambda r:r['median_s']) for s in ['rc','r2']}
    stage('2: finish low-storage matched-accuracy screen')
    for h in [.25,.5,1.,1.5,2.]:
        run(**dict(settings(best['r2']),dt_ns=h),integrator='ck54',repeats=1)
    stage('3: precomputed stage waveforms')
    for s in best:
        w=run(**dict(settings(best[s]),waveform=True))
        if passing(w) and w['median_s']<best[s]['median_s']:best[s]=w
    atomic(out/'screen_winners.json',{s:r['job'] for s,r in best.items()})
    stage('4: pinned output transfer overlap')
    for s in best:run(**settings(best[s]),kind='ode_pipeline')
    run(system='r2',kind='dense_pipeline')
    stage('5: cached static photon-only projection')
    run(system='r2',kind='dense_photons')
    stage('6: photon quadrature and output-only short tests')
    for s in best:
        cfg=settings(best[s])
        fine=run(**dict(cfg,dt_ns=.9765625 if s=='rc' else .5,waveform=False),photons=True,repeats=1)
        for mode in ['joint','photon_only']:
            run(**cfg,photons=True,output_mode=mode,photon_reference=archive(fine))
    stage('7: larger batches with tuned kernels')
    for s in best:
        cfg=settings(best[s])
        for batch in [1024,2048,4096]:run(**cfg,batch=batch,repeats=1)
    stage('8: full RC trajectories, existing independent reference')
    for v,batch in [(150.,128),(50.,9),(250.,9)]:
        run(**settings(best['rc']),kind='full',velocity=v,batch=batch,repeats=1)
    stage('9: full R(2) EOM trajectories and photon-only output')
    for v in [150.,50.,250.]:
        cfg=settings(best['r2'])
        fine=run(**dict(cfg,dt_ns=.5,waveform=False),kind='full_reference',velocity=v,batch=60,photons=True,repeats=1)
        if fine.get('status')!='complete':continue
        selected=None
        for h in [2.,1.]:
            r=run(**dict(cfg,dt_ns=h),kind='full',velocity=v,batch=60,photons=True,repeats=1,
                  reference=archive(fine),photon_reference=archive(fine))
            if passing(r):selected=r;break
        if selected:
            run(**settings(selected),kind='full',velocity=v,batch=60,photons=True,output_mode='photon_only',repeats=1,
                reference=archive(fine),photon_reference=archive(fine))
    stage('10: full RC photon quadrature and photon-only output')
    for v in [150.,50.,250.]:
        cfg=settings(best['rc'])
        fine=run(**dict(cfg,dt_ns=.9765625,waveform=False),kind='full_reference',velocity=v,batch=9,photons=True,repeats=1)
        if fine.get('status')!='complete':continue
        for mode in ['joint','photon_only']:
            run(**cfg,kind='full',velocity=v,batch=9,photons=True,output_mode=mode,repeats=1,
                reference=archive(fine),photon_reference=archive(fine))
    stage('complete');runner.report();runner.progress('complete')

if __name__=='__main__':main()
