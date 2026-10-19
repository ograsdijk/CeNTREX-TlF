"""Final serial checks of waveform correction for each requested output mode."""
from pathlib import Path
import time,json
from run_gpu_next_suite import Runner
from continue_gpu_next_suite import archive

def main():
    out=Path('benchmarks/gpu_next_results/v1_20261012_174509')
    while True:
        try:state=json.loads((out/'progress.json').read_text())
        except (PermissionError,json.JSONDecodeError):time.sleep(.2);continue
        if state['status']=='complete':break
        if time.time()-(out/'progress.json').stat().st_mtime>120:raise RuntimeError('Follow-up runner stopped updating')
        time.sleep(3)
    runner=Runner(out,resume=True);runner.stage='15: final corrected waveform output-specific checks';runner.progress()
    winners=json.loads((out/'screen_winners.json').read_text())
    for s in ['rc','r2']:
        cfg={k:winners[s][k] for k in ['system','variant','span_us','threads','tile','dt_ns','waveform']}
        runner.run(**cfg,repeats=3,validation_revision='physical_time_lookup')
        for v in [150.,50.,250.]:
            ref=next(r for r in runner.results if r.get('status')=='complete' and r['job']['system']==s and
                     r['job']['kind']=='full_reference' and r['job'].get('velocity')==v)
            for mode in ['joint','photon_only']:
                runner.run(**cfg,kind='full',velocity=v,batch=9 if s=='rc' else 60,photons=True,
                           output_mode=mode,repeats=1,reference=archive(ref),photon_reference=archive(ref),
                           validation_revision='physical_time_lookup')
    runner.stage='complete';runner.report();runner.progress('complete')

if __name__=='__main__':main()
