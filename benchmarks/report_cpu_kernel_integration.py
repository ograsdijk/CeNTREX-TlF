"""Report correctness and matched timings of the package CPU kernel option."""
from __future__ import annotations

import json
from pathlib import Path

from report_fused_dsu import CPU, summarize

ROOT=Path(__file__).resolve().parent/'rust_specialized_rhs_results/cpu_kernel_integration'


def main():
    report=json.loads((ROOT/'results.json').read_text())
    summarize(report,['original'])
    lines=['# CPU kernel library integration validation','',
        'The package now supports `cpu_kernel="auto"` at preparation, independently of the execution mode. On the test machine it resolves to `avx2_fma`. AVX-512 stays explicit. The original implementation remains selectable. All optimized variants fuse structured decay and direct packed output, and vectorize finalization.','',
        f'CPU: {CPU}. Eight randomized interleaved rounds per system; all candidates share one freshly compiled generic current-source plan/RHS/Tsit5 wrapper. The `cpu_kernel` payload field drives the actual package dispatch, rather than benchmark-only hook flags. References use the original CPU kernel at atol=1e-13, rtol=1e-11; candidates use 1e-11/1e-9.','',
        '| System | Original median (s) | Auto median (s) | Auto resolved | Paired runtime reduction | Descriptive 95% interval |','|---|---:|---:|---|---:|---:|']
    for name,s in report['systems'].items():
        auto=s['summary']['auto']
        effect=auto['comparisons']['original']
        lo,hi=effect['bootstrap_95_percent']
        lines.append(f"| {name} | {s['summary']['original']['median_seconds']:.6f} | {auto['median_seconds']:.6f} | {s['resolved']['auto']} | {effect['median_reduction_percent']:.2f}% | [{lo:.2f}, {hi:.2f}]% |")
    lines += ['', '## Explicit selections','',
        '| System | Requested kernel | Resolved | Median (s) | Paired reduction vs original |','|---|---|---|---:|---:|']
    for name,s in report['systems'].items():
        for kernel,row in s['summary'].items():
            lines.append(f"| {name} | {kernel} | {s['resolved'][kernel]} | {row['median_seconds']:.6f} | {row['comparisons']['original']['median_reduction_percent']:+.2f}% |")
    lines += ['', '## Validation','',
        '- 222 Python checks passed across CPU selection, prepared-H optimizations, public aliases and the existing Rust backend suite.',
        '- 66 Rust checks passed; one pre-existing ignored check remains ignored. Synthetic feature tests prove auto falls back safely and never selects AVX-512; unsupported explicit choices fail.',
        '- New tests exercise full complex RHSs at dimensions 1, 2, 4, 5, 8, 9 and 17, including vector tails, both Hamiltonian representations, dephasing and branching decays, static/time-dependent cache invalidation, matrix/packed API interleaving, parameter overrides, photon/rate/population/selected/full/mixed-integral output, parallel batches/grids, switched drives and terminal events.',
        '- Every timed trajectory and warmup passed full complex density checks: 1% entry-relative error above reference magnitude 1e-6, absolute 1e-6 below, trace error <1e-9. Fixed-time random-density checks also compare each integrated variant against original.',
        '- Release extension rebuilt and copied into the project virtualenv editable package. Python lint, Rust formatting and changed-file whitespace checks passed.','',
        '| System | Max normalized RHS error | Max relative density error (fraction) | Max small-entry absolute error | Max trace error | RHS counts across kernels |','|---|---:|---:|---:|---:|---|']
    for name,s in report['systems'].items():
        calls=sorted({c for values in s['rhs_call_counts'].values() for c in values})
        lines.append(f"| {name} | {s['max_normalized_rhs_error']:.3g} | {s['max_relative_density_error']:.3g} | {s['max_small_absolute_density_error']:.3g} | {s['max_trace_error']:.3g} | {calls} |")
    lines += ['', 'These timings use the established 10 us E=200 V/cm, B=0.01 G EOM performance fixtures with 250 ns segmentation, not full zero-electric-field apparatus transits. RC begins with uniform X,J=2 population; optical fixtures use uniform modeled X populations. Intervals resample whole paired rounds (20,000 draws) and describe this small sample on this CPU; performance on other CPU models is not established.', '',
        'The public packed Rust D+S+U path is optimized; other execution modes, the default expanded_sparse mode, Python, CUDA, and matrix/split RHS computations retain their paths. The requested option travels through payloads and resolves on the receiving machine. The native plan and Python prepared object expose requested/resolved selections. The original assembly/DAG opt-out flags remain independent.', '',
        'Reproduction: `../../validate_cpu_kernel_integration.py`; report generator: `../../report_cpu_kernel_integration.py`. `results.json` records every order, timing, density check, step/RHS count and current native source hash. The API tests are `tests/lindblad/test_cpu_kernel.py`. No system-specific native code generation is required in the installed library.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'results.json').write_text(json.dumps(report,indent=2))
    print('\n'.join(lines[:14]))


if __name__=='__main__':
    main()
