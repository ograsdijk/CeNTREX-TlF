"""Report completed R2 GPU experiments without rerunning benchmarks."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'r2_gpu_reuse_results'
def read(path):return json.loads((OUT/path).read_text())

def main():
    dense=read('v2_20261012/dense_results.json')
    complex_dense=read('dense_complex_v1/dense_results.json')
    matched=read('matched_cross_final/matched_scan_results.json')
    eom=read('eom_v1/eom_results.json')
    long=read('long_confirmation_v1/ode_sweep_results.json')
    cpu_native={r['repeat']:r['wall_seconds'] for r in matched['cpu_rows'] if r['method']=='native_forward'}
    cpu=float(np.median(list(cpu_native.values())))
    winner=min(matched['rows'],key=lambda r:r['median_seconds'])
    lines=['# R(2) GPU ODE and static eigensystem reuse: completed','',
           'Hardware: RTX 5070 Ti; Ryzen 9800X3D (8 physical cores). FP64 throughout. No CPU ODE benchmarks were rerun. All files are benchmark artifacts; no production solver or scan configuration was changed.','',
           '## One static eigensystem, many populations and velocities','',
           'The earlier GPU dense tests reached batches of 8 different matrices and 20 initial populations per matrix; they refactored every task. The new test factors one archived R(2) F4 generator on the CPU once, uploads its real eigenbasis and initial modal coefficients, then evaluates all 20 resolved X J=2 initial projectors at many propagation times. Velocities span 50–250 m/s and set t = 0.020 m / v in a uniform static field. All other generator parameters are fixed. Arbitrary diagonal mixtures of these 20 populations follow by linear combination.','',
           'The 41-level model has 1444 active real packed coordinates; its three sink populations are recovered through the integrated feed exactly. Output is the complete lossless packed density (1681 real values per case), including all coherences and sinks, returned to host memory. Zero sink coherences follow exact reachability checks. CUDA reconstructs the full packed density in this prototype. The existing public CUDA projection path does not yet provide this implementation.','',
           'Timings include modal function evaluation, full reconstruction, allocation and GPU-to-CPU transfer. They exclude the shared eigendecomposition, initial solves, one-time factor upload/context startup, disk writing and diagnostics. Each timing is the median of three repetitions. Both CPU and GPU reuse exponential/integral functions across initial populations. CPU batch sizes 128/512/2048/8192 and BLAS thread counts 1/8 were checked; GPU uses the same batch-size sweep. The table compares the best measured CPU and GPU configurations.','',
           '| Initial populations × velocities | Cases | CPU seconds | GPU seconds | Speedup |',
           '|---|---:|---:|---:|---:|']
    summary=[]
    for nv in [1,21,101,501]:
        rows=[r for r in dense['rows'] if r['velocities']==nv]
        c=min((r for r in rows if r['backend']=='cpu'),key=lambda r:r['median_seconds'])
        g=min((r for r in rows if r['backend']=='cuda'),key=lambda r:r['median_seconds'])
        ratio=c['median_seconds']/g['median_seconds']
        lines.append(f'| 20 × {nv} | {20*nv:,} | {c["median_seconds"]:.5f} | {g["median_seconds"]:.5f} | {ratio:.2f}× |')
        summary.append({'velocities':nv,'cpu':c,'gpu':g,'speedup':ratio})
    prep=dense['preparation']
    max_abs=max(r['all_output_max_abs_vs_cpu'] for r in dense['rows'])
    lines += ['',f'One-time CPU factor/initial preparation was {prep["factor_stages"]["total_s"]:.3f} s. Initial CUDA factor upload, including context startup in this run, was {prep["upload_factors_s"]:.3f} s. These costs matter for a cold, small job; the speedups above concern reused evaluation. At 20 outputs, the CPU wins.',
              '',f'Every output coordinate was compared with the CPU result: maximum absolute difference {max_abs:.3g}. Complex-entry, trace and positivity diagnostics were also evaluated on up to 60 distributed outputs per row. This is far below the 1% criterion. The initial v1 timings were superseded by v2 after improving CPU reuse of modal functions; quote v2 only.','',
              '## R(2) ODE against the existing CPU benchmark','',
              'This uses the frozen 40-level cross-system recipe: 8 detuning/power points, all 40 one-hot model populations, 1 µs evolution, and full final density. GPU batches are grouped by optical Rabi rate (200, 80 and 40 trajectories). The CPU number is the existing eight-worker D+S+U result, not a newly run CPU benchmark.','',
              '| Method | Seconds | Result |','|---|---:|---|',f'| Archived CPU D+S+U, 8 workers | {cpu:.4f} | Strict archived tolerances |']
    for r in matched['rows']:lines.append(f'| GPU {r["variant"]}, 0.5 ns | {r["median_seconds"]:.4f} | {"Pass" if r["passed"] else "Fail"} |')
    worst=max(d['max_complex_relative_above_1e_6'] for r in matched['rows'] for d in r['diagnostics'])
    near=max(d['max_absolute_below_1e_6'] for r in matched['rows'] for d in r['diagnostics'])
    lines += ['',f'Best passing GPU time: {winner["median_seconds"]:.4f} s, {cpu/winner["median_seconds"]:.2f}× faster than the archived CPU run. Worst complex-entry relative error for |ρij| ≥ 1e-6 was {100*worst:.4f}%; maximum error below the cutoff was {near:.3g}. Trace and positivity diagnostics pass. A 1 ns scan failed at 10.75% worst relative error and is retained in `matched_cross_v1/`; its faster timing is excluded.',
              '', 'The physical configuration matches the archived CPU fixture. The currently installed native binary has a different hash from the historical CPU binary; this is a comparison to archived CPU performance, not a fresh same-build timing. All 320 output densities were validated against the saved strict CPU reference. Setup/compilation/capture, diagnostics and disk compression are excluded from GPU evolution/output timings, as are historical CPU fixture startup costs.','',
              '## Smaller-system EOM and storage checks','',
              'The same R(2) model was tested with β=3.8, 1.6 MHz optical phase modulation, entry at absolute t=125 ns, and 1 µs evolution. Sixty distinct cases (detuning −20/0/+20 MHz × 20 initial projectors) are repeated to fill batch 512. This is an ODE test. The saved strict CPU main-state result is checked separately; the full batch is compared with a 0.5 ns GPU reference.','',
              '| Variant | Step (ns) | Seconds | Accuracy |','|---|---:|---:|---|']
    for r in eom['rows']:lines.append(f'| {r["variant"]} | {r["dt_ns"]:g} | {r["median_seconds"]:.4f} | {"Pass" if r["passed"] else "Fail"} |')
    base=next(r for r in eom['rows'] if r['variant']=='base')
    packed=next(r for r in eom['rows'] if r['variant']=='swap_const_exact_packed' and r['dt_ns']==1.)
    best=min((r for r in eom['rows'] if r['passed']),key=lambda r:r['median_seconds'])
    lines += ['',f'At the same 1 ns step, packed/staged storage plus the stage/tail changes cuts runtime by {100*(1-packed["median_seconds"]/base["median_seconds"]):.1f}%. Packed/direct is effectively tied here; staged is slightly faster. The 2 ns step passes with worst complex-entry error {100*best["fine_gpu_diagnostics"]["max_complex_relative_above_1e_6"]:.4f}% across the batch; 4 ns fails at 13.81%. This EOM confirmation covers 1 µs, not a full experimental transit.',
              '', 'Static nominal R(2) checks at 400 µs also pass for fixed Tsit5 at 1/2/4 ns over all 20 ground projectors. The integrating-factor prototype passes at 4 ns over 1 µs but reaches 1.0313% error at 20 µs, so it is not selected. These static nominal checks do not license a 4 ns step for the wider detuning or EOM scans.','',
              '## Artifacts and scope','',
              '- `v2_20261012/dense_results.json`: final dense batch sweep; factors stored alongside it.','- `matched_cross_final/`: passing GPU scan, candidate/reference densities, source snapshots and hashes.','- `matched_cross_v1/`: rejected 1 ns wider scan.','- `matched_cross_v1/ode_sweep_results.json`: static step/storage/IF screening.','- `long_confirmation_v1/`: longer static checks through 400 µs.','- `eom_v1/`: EOM candidates, fine references, CPU reference checks and source snapshots.','- `v1_20261012/ode_results.json`: original 41-level smaller-system storage/IF pilot.','',
              'The strongest dense reuse gain comes from keeping factors resident and reconstructing many outputs together. It does not mean GPU eigendecomposition itself is faster. Library integration and full-transit EOM validation remain separate work. Core results were complete before the inherited 16:37:10 UTC cutoff. The complex-output follow-up completed on the later keep-going turn. The deadline guard then refused to launch the planned longer EOM jobs; those were not run.']
    complex_lines=['## Ordinary complex density-matrix outputs','',
                   'A follow-up returns actual complex128 41×41 matrices on the host, including GPU unpacking and the larger full-complex transfer. The CPU uses the same vectorized reconstruction. This preserves the reuse gain with ordinary matrix outputs, not just lossless packed storage. Factorization and one-time upload remain excluded.','',
                   '| Initial populations × velocities | CPU seconds | GPU seconds | Speedup |',
                   '|---|---:|---:|---:|']
    complex_summary=[]
    for nv in [1,21,101,501]:
        rows=[r for r in complex_dense['rows'] if r['velocities']==nv]
        c=min((r for r in rows if r['backend']=='cpu'),key=lambda r:r['median_seconds'])
        g=min((r for r in rows if r['backend']=='cuda'),key=lambda r:r['median_seconds'])
        ratio=c['median_seconds']/g['median_seconds']
        complex_lines.append(f'| 20 × {nv} | {c["median_seconds"]:.5f} | {g["median_seconds"]:.5f} | {ratio:.2f}× |')
        complex_summary.append({'velocities':nv,'cpu':c,'gpu':g,'speedup':ratio})
    complex_lines += ['',f'Maximum difference over every returned complex entry: {max(r["all_output_max_abs_vs_cpu"] for r in complex_dense["rows"]):.3g}. Raw results and source provenance are in `dense_complex_v1/`.','']
    position=lines.index('## Artifacts and scope');lines[position:position]=complex_lines
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sources=OUT/'source_snapshot';sources.mkdir(exist_ok=True)
    for name in ['run_r2_gpu_reuse.py','prepare_r2_gpu_cross_fixture.py','report_r2_gpu_reuse.py']:shutil.copyfile(HERE/name,sources/name)
    artifacts={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps({'artifacts_sha256':artifacts,'dense_summary':summary,'complex_dense_summary':complex_summary,'cpu_archived_full_scan_seconds':cpu,
                                              'best_gpu_ode_seconds':winner['median_seconds']},indent=2),encoding='utf-8')
    print(OUT/'REPORT.md')

if __name__=='__main__':main()
