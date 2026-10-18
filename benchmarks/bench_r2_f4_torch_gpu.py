"""Forward-only double-precision PyTorch GPU dense benchmark, three matrices."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bench_r2_f4_dense_improvements as improved

OUT=Path(__file__).resolve().parent/'r2_f4_torch_gpu_results'
INPUT=improved.OUT
GAMMA=improved.previous.prior.scan.GAMMA


def synchronize(): torch.cuda.synchronize()


def gpu_phi(w,t):
    wt=w.unsqueeze(-1)*t
    denominator=torch.where(w.abs()>0,w,torch.ones_like(w)).unsqueeze(-1)
    return torch.where(wt.abs()<1e-7,t*(1+wt/2+wt**2/6),torch.expm1(wt)/denominator)


def run_gpu(arrays):
    synchronize();total_tick=time.perf_counter();tick=total_tick
    A=torch.as_tensor(np.stack([a['A'] for a in arrays]),device='cuda',dtype=torch.float64)
    packed=torch.as_tensor(np.stack([a['packed'] for a in arrays]),device='cuda',dtype=torch.float64)
    times=torch.as_tensor(arrays[0]['times'],device='cuda',dtype=torch.float64)
    exc=torch.as_tensor(arrays[0]['excited'],device='cuda',dtype=torch.int64)
    synchronize();transfer_s=time.perf_counter()-tick
    tick=time.perf_counter()
    w,V=torch.linalg.eig(A)
    synchronize();eig_s=time.perf_counter()-tick
    tick=time.perf_counter()
    C=torch.linalg.solve(V,packed.to(torch.complex128))
    synchronize();solve_s=time.perf_counter()-tick
    tick=time.perf_counter()
    observable=GAMMA*V[:,exc,:].sum(dim=1)
    weighted=observable.unsqueeze(-1)*C
    photons=weighted.transpose(-1,-2)@gpu_phi(w,times)
    synchronize();project_s=time.perf_counter()-tick
    tick=time.perf_counter()
    output=photons.cpu().numpy()
    synchronize();download_s=time.perf_counter()-tick
    timing={'batch_size':len(arrays),'host_to_device_s':transfer_s,'eig_s':eig_s,
        'initial_states_s':solve_s,'photon_projection_s':project_s,'device_to_host_s':download_s,
        'total_s':time.perf_counter()-total_tick}
    return output,timing,(w,V,C)


def diagnostics(output,factors,arrays):
    w,V,C=factors
    trace_error=0.
    for i,a in enumerate(arrays):
        ts=torch.tensor([0.,1e-6,.02/150,350e-6],device='cuda',dtype=torch.float64)
        # Packed active populations are the first 38 variables after reduction.
        trace_active=V[i,:38].sum(dim=0)
        feed=torch.as_tensor(a['sink_feed'].sum(axis=0),device='cuda',dtype=torch.float64).to(torch.complex128)
        trace_sink=feed@V[i]
        trace=(trace_active[:,None]*C[i]).T@torch.exp(w[i,:,None]*ts)
        trace+=(trace_sink[:,None]*C[i]).T@gpu_phi(w[i],ts)
        trace_error=max(trace_error,float((trace-1).abs().max().item()))
    return {'max_trace_error':trace_error,'max_photon_imaginary':float(abs(output.imag).max()),
            'min_photon_increment':float(np.diff(output.real,axis=-1).min()),
            'min_photons':float(output.real.min())}


def save(result):
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(False);torch.set_num_threads(1)
    properties=torch.cuda.get_device_properties(0)
    result={'generated_at':datetime.now().astimezone().isoformat(),
        'torch_version':torch.__version__,'cuda_build':torch.version.cuda,'gpu':properties.name,
        'gpu_memory_bytes':properties.total_memory,'cuda_capability':list(torch.cuda.get_device_capability()),
        'linalg_backend':str(torch.backends.cuda.preferred_linalg_library()),
        'dtype':'float64 / complex128','autograd':False,
        'scope':'Three saved reduced 1444-dimensional Liouvillians, pure X 10 mW; all 20 populations; 1401 photon samples. Batch-eight cycles over the same three inputs; no new scan points.',
        'cpu':[],'gpu':[],'validation':[]}
    arrays=[];references=[]
    with threadpool_limits(limits=1):
        for point in range(3):
            with np.load(improved.previous.OUT/f'point_{point}_inputs.npz') as inputs:
                full={k:inputs[k].copy() for k in inputs.files}
            reduced,_=improved.reduce(full)
            arrays.append({'A':reduced['A'],'packed':reduced['reduced_packed'],
                'excited':reduced['reduced_excited'],'times':reduced['times'],'sink_feed':reduced['sink_feed']})
            with np.load(INPUT/f'point_{point}_responses.npz') as reference:
                references.append(reference['active_real_photons'].copy())
        # Full-size warm-up records first-call cost separately, excludes it
        # from steady-state repetitions (CUDA context/library loading).
        print('GPU warm-up: one full parameter point',flush=True)
        warm_output,warm_timing,_=run_gpu([arrays[0]])
        result['gpu_warmup']=warm_timing;save(result)
        print(f'Warm-up total {warm_timing["total_s"]:.3f} s',flush=True)
        for point,a in enumerate(arrays):
            for repeat in range(2):
                _,stage,_=improved.real_factor(a['A'],a['packed'],a['excited'],a['times'])
                result['cpu'].append({'point':point,'repeat':repeat,**stage})
                output,timing,factors=run_gpu([a])
                difference=abs(output[0].real-references[point])
                check=diagnostics(output,factors,[a])
                scaled=float(np.max(difference/(1e-7+1e-5*abs(references[point]))))
                passed=bool(scaled<1 and check['max_trace_error']<1e-7
                            and check['max_photon_imaginary']<1e-7 and check['min_photon_increment']>-1e-7)
                result['gpu'].append({'point':point,'repeat':repeat,**timing})
                result['validation'].append({'point':point,'repeat':repeat,
                    'max_photon_difference':float(difference.max()),'max_scaled_error':scaled,'passed':passed,**check})
                save(result)
                print(f'Point {point}: CPU {stage["total_s"]:.3f} s; GPU {timing["total_s"]:.3f} s; max error {difference.max():.3g}; pass={passed}',flush=True)
                np.savez_compressed(OUT/f'point_{point}_responses.npz',gpu_photons=output[0].real,
                                    cpu_photons=references[point],times_s=a['times'])
        for batch_size in [3,8]:
            items=[arrays[i%3] for i in range(batch_size)]
            for repeat in range(2):
                output,timing,_=run_gpu(items)
                error=max(float(abs(output[i].real-references[i%3]).max()) for i in range(batch_size))
                scaled=max(float(np.max(abs(output[i].real-references[i%3])/(1e-7+1e-5*abs(references[i%3])))) for i in range(batch_size))
                result['gpu'].append({'repeat':repeat,'max_photon_difference':error,'max_scaled_error':scaled,**timing})
                save(result)
                print(f'GPU batch {batch_size}: {timing["total_s"]:.3f} s; {batch_size/timing["total_s"]:.2f} points/s; max error {error:.3g}',flush=True)
    result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
    save(result)
    med=lambda rows,key:float(np.median([r[key] for r in rows]))
    cpu=med(result['cpu'],'total_s')
    report=['# PyTorch GPU dense forward benchmark','',result['scope'],'',
        f"GPU: {properties.name}; PyTorch {torch.__version__}, CUDA build {torch.version.cuda}. Double precision; autograd disabled.",
        '', 'GPU path: general non-Hermitian eigendecomposition, complex solve for all initial-state columns, analytic cumulative photon projection. CPU comparison uses the optimized reduced real eigenbasis. CUDA is synchronized around each stage. Input stacking/transfers and result download are included; common Liouvillian construction, validation, and file saving excluded.',
        '', f'Full-size GPU first-call warm-up: {warm_timing["total_s"]:.3f} s (separate from steady-state timings).',
        '', '| Configuration | Median seconds | Parameter points/s |','|---|---:|---:|',
        f'| Optimized CPU, one thread, one point | {cpu:.3f} | {1/cpu:.2f} |']
    for size in [1,3,8]:
        duration=med([r for r in result['gpu'] if r['batch_size']==size],'total_s')
        report.append(f'| PyTorch GPU, batch {size} | {duration:.3f} | {size/duration:.2f} |')
    previous=json.loads((INPUT/'results.json').read_text(encoding='utf-8'))
    eight=med([r for r in previous['worker_scaling'] if r['workers']==8],'wall_s')
    report.extend([f'| Previously measured CPU, eight workers, 32 points | {eight:.3f} | {32/eight:.2f} |',
        '', 'The eight-worker CPU row is from the previous benchmark, not a simultaneous rerun. It is the relevant scan-throughput reference; beating one CPU thread alone would not establish a scan speedup.',
        '', '## Accuracy','',
        f"Maximum single-point photon difference: {max(r['max_photon_difference'] for r in result['validation']):.3g}. All single-point gates passed: {all(r['passed'] for r in result['validation'])}. Batched maximum scaled photon error: {max(r['max_scaled_error'] for r in result['gpu'] if r['batch_size']>1):.3g} (pass below 1).",
        '', f"Peak PyTorch CUDA allocation: {result['peak_gpu_allocated_bytes']/1e9:.2f} GB; reserved: {result['peak_gpu_reserved_bytes']/1e9:.2f} GB. Library workspace may have additional allocation.",
        '', 'No differentiability implementation, no production changes, and no full scan. Detailed stage timings and settings: [results.json](results.json).',
        '', 'Reproduce using the isolated benchmark environment: `benchmarks/.venv-gpu-bench/Scripts/python.exe benchmarks/bench_r2_f4_torch_gpu.py`.'])
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Finished {OUT / "REPORT.md"}',flush=True)


if __name__=='__main__': main()
