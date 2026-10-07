"""Validate public dense APIs on F4 references; time a small unique-point grid."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf import lindblad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'reports/r2_f4_earth_field/obe'))
import scan

OUT = Path(__file__).resolve().parent/'dense_library_integration_results'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--throughput-only',action='store_true')
    parser.add_argument('--normal-startup-only',action='store_true')
    args=parser.parse_args()
    normal_startup=args.normal_startup_only
    throughput_only=args.throughput_only or normal_startup
    OUT.mkdir(exist_ok=True)
    result = dict(generated_at=datetime.now().astimezone().isoformat(), validation=[], throughput=[])
    if throughput_only:
        previous=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
        result['validation']=previous['validation']
        result['before_plan_reuse_throughput']=previous.get('before_plan_reuse_throughput',previous['throughput'])
        if normal_startup:
            result['throughput']=previous['throughput']
            result['normal_startup_throughput']=[]
    inputs = Path(__file__).resolve().parent/'r2_f4_dense_threading_results'
    with np.load(inputs/'point_0_inputs.npz') as saved:
        packed = saved['packed'].T.copy()
    saved_refs = Path(__file__).resolve().parent/'r2_f4_dense_improvements_results'
    followup = Path(__file__).resolve().parent/'r2_f4_dense_followup_results'
    with threadpool_limits(limits=1):
        for fraction in ([0.] if throughput_only else [0., .03, .05]):
            _, prepared, _, rabis, excited, _, _ = scan.build(fraction)
            options = dict(solver='dense_eig', rho0_batch=packed,
                           output='photon_integral', output_when='saveat',
                           saveat=scan.TIMES, integral_weights=[(i,scan.GAMMA) for i in excited],
                           parallel=False, collect_stats=True)
            for power in [0,1]:
                if throughput_only: break
                for detuning in [6.8,28.2]:
                    result_cpu = lindblad.grid_scan(prepared, None, (0.,350e-6),
                        scan={'detuning':[2*np.pi*1e6*detuning], 'rabi':[rabis[power]]}, **options)
                    cpu = result_cpu.values[...,0]
                    path = followup/f'z{fraction*100:02.0f}_p{scan.POWERS[power]*1000:g}_d{detuning:g}.npz'
                    if fraction == 0 and power == 1:
                        point = 0 if detuning == 6.8 else 1
                        path = saved_refs/f'point_{point}_responses.npz'
                        key = 'active_real_photons'
                    elif path.exists():
                        key = 'optimized_photons'
                    else:
                        # Prior coverage lacks 3/5% 1 mW at 6.8 MHz; skip those.
                        continue
                    with np.load(path) as saved:
                        reference = saved[key]
                    error = float(abs(cpu-reference).max())
                    assert error < 1e-7
                    result_gpu = lindblad.grid_scan(prepared, None, (0.,350e-6),
                        scan={'detuning':[2*np.pi*1e6*detuning], 'rabi':[rabis[power]]},
                        evolution_device='cuda', **options)
                    gpu_error = float(abs(cpu-result_gpu.values[...,0]).max())
                    assert gpu_error < 1e-7
                    factor = lindblad.prepare_dense_lindblad_propagator(prepared, packed,
                        parameter_values={'detuning':2*np.pi*1e6*detuning, 'rabi':rabis[power]})
                    density = factor.density_matrices(packed, [0.,1e-6,.02/150,350e-6])
                    trace_error = float(abs(np.trace(density,axis1=-2,axis2=-1)-1).max())
                    minimum_eigenvalue = float(np.linalg.eigvalsh(density).min())
                    assert trace_error < 1e-7 and minimum_eigenvalue > -1e-7
                    row = dict(z_fraction=fraction,power_mW=float(scan.POWERS[power]*1000),detuning_MHz=detuning,
                               photon_reference_error=error,cpu_gpu_error=gpu_error,
                               trace_error=trace_error,min_density_eigenvalue=minimum_eigenvalue,
                               factorization_stats=factor.stats)
                    result['validation'].append(row)
                    print(f'PASS fz={fraction}, p={row["power_mW"]:g}, d={detuning:g}; reference {error:.3g}, GPU {gpu_error:.3g}; factor {factor.stats["factorization_seconds"]:.3f}s',flush=True)
            if fraction == 0:
                benchmark_prepared, benchmark_rabis, benchmark_excited = prepared, rabis, excited
    # Unique nearby parameter values avoid grouping duplicate points in the library.
    detunings = np.array([ [6.8,28.2,80.][i%3]+i*.0001 for i in range(32)])
    options = dict(solver='dense_eig', rho0_batch=packed, output='photon_integral',
                   output_when='saveat',saveat=scan.TIMES,
                   integral_weights=[(i,scan.GAMMA) for i in benchmark_excited],
                   parallel=True,threads=8,collect_stats=True,profile_startup=not normal_startup)
    for repeat in range(2):
        order = ['cpu','cuda'] if repeat == 0 else ['cuda','cpu']
        results = {}
        for device in order:
            tick=time.perf_counter()
            answer=lindblad.grid_scan(benchmark_prepared,None,(0.,350e-6),
                scan={'detuning':2*np.pi*1e6*detunings,'rabi':[benchmark_rabis[1]]},
                evolution_device=device,**options)
            wall=time.perf_counter()-tick
            assert answer.solver_stats['factorization_count']==32
            rows=result['normal_startup_throughput'] if normal_startup else result['throughput']
            rows.append(dict(device=device,repeat=repeat,tasks=32,wall_s=wall,
                                             points_per_second=32/wall,stats=answer.solver_stats))
            results[device]=answer.values
            with np.load(saved_refs/'point_0_responses.npz') as reference:
                error=float(abs(answer.values[:len(packed),:,0]-reference['active_real_photons']).max())
            assert error < 1e-7
            rows[-1]['first_point_reference_error']=error
            steady=answer.solver_stats['steady_state_seconds']
            if steady is not None:
                print(f'Public API {device}: total {wall:.3f}s; startup {answer.solver_stats["worker_startup_seconds"]:.3f}s; steady {steady:.3f}s = {32/steady:.2f} points/s; reference error {error:.3g}',flush=True)
            else:
                print(f'Normal startup API {device}: total {wall:.3f}s, {32/wall:.2f} points/s; reference error {error:.3g}',flush=True)
        assert float(abs(results['cpu']-results['cuda']).max()) < 1e-7
        (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    config = json.loads((Path(__file__).resolve().parent/'r2_f4_gpu_concurrency_results/system_config.json').read_text(encoding='utf-8'))
    (OUT/'system_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    report=['# Dense solver library integration validation','',
            'Public solver APIs checked against saved F4 responses at ten selected power/polarization/detuning combinations. Twenty independent initial states, 1401 photon samples; full density checks at four times.', '',
            f'Maximum saved-reference photon error: {max(r["photon_reference_error"] for r in result["validation"]):.3g}. Maximum CPU/CUDA difference: {max(r["cpu_gpu_error"] for r in result["validation"]):.3g}. Trace and density positivity checks passed.', '',
            'Small 32-point unique-parameter grid, repeated twice in opposite order. Timings include parameter binding, analytic generator extraction, reduction, conditioning/residual checks, fresh eight-worker process startup, communication, transfers, projection and result collation. Common OBE construction and report writing are excluded. No ODE scan rerun.', '',
            'Each CPU worker now reuses one native plan/evaluator/workspace, replacing parameter overrides with cache invalidation. Worker readiness is synchronized to measure startup separately. Steady-state timing includes extraction, reduction, decomposition, communication, projection, transfers and collation; it excludes worker startup and teardown.', '',
            '| Evolution | Total seconds | Total points/s | Worker startup seconds | Steady seconds | Steady points/s |','|---|---:|---:|---:|---:|---:|']
    for device in ['cpu','cuda']:
        seconds=float(np.median([r['wall_s'] for r in result['throughput'] if r['device']==device]))
        startup=float(np.median([r['stats']['worker_startup_seconds'] for r in result['throughput'] if r['device']==device]))
        steady=float(np.median([r['stats']['steady_state_seconds'] for r in result['throughput'] if r['device']==device]))
        report.append(f'| {device} | {seconds:.3f} | {32/seconds:.2f} | {startup:.3f} | {steady:.3f} | {32/steady:.2f} |')
    if 'before_plan_reuse_throughput' in result:
        report += ['', 'Before this fix, the same public 32-point API measured approximately 10.00 s CPU-only and 10.25 s CPU/GPU including startup. Prior warm preassembled-matrix benchmarks measured 8.53 and 9.17 points/s respectively, with parameter binding/generator extraction excluded; those remain a narrower timing boundary.']
    if 'normal_startup_throughput' in result:
        report += ['', '## Default startup scheduling', '', 'Without startup profiling, ready workers begin solving while other workers initialize. The following cold-call timings use this default scheduling (two repetitions), including startup and teardown.', '', '| Evolution | Total seconds | Points/s |', '|---|---:|---:|']
        for device in ['cpu','cuda']:
            wall=float(np.median([r['wall_s'] for r in result['normal_startup_throughput'] if r['device']==device]))
            report.append(f'| {device} | {wall:.3f} | {32/wall:.2f} |')
    report += ['', 'System: AMD Ryzen 7 9800X3D, eight physical cores / sixteen logical processors, 64 GB RAM class; RTX 5070 Ti. One BLAS thread per CPU worker. Software/driver/BLAS details: [system_config.json](system_config.json). Detailed numerical and timing data: [results.json](results.json).']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


if __name__=='__main__': main()
