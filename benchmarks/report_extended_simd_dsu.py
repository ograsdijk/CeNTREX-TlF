"""Report matched FMA/AVX-512/finalization trials."""
from __future__ import annotations

import json
from pathlib import Path

from report_fused_dsu import CPU, summarize

ROOT=Path(__file__).resolve().parent/'rust_specialized_rhs_results/extended_simd_dsu'


def main():
    report=json.loads((ROOT/'results.json').read_text())
    summarize(report,['previous_best','hooked_control'])
    lines=['# FMA, AVX-512 and vector finalization on fused D+S+U','',
        'Vectorizing finalization gives the most consistent improvement. The best observed combination differs by system: all three for RC, FMA plus AVX2 finalization for R(2)/Q(1), and finalization alone for R(0) (FMA is nearly tied). AVX-512 is not a universally beneficial setting. No production kernels/defaults changed.','',
        f'CPU: **{CPU}**. Runtime checks confirmed AVX2, FMA and AVX512F support. All improvements below are incremental to the previous winning **AVX2 + fused decay + direct packed output** implementation, freshly compiled separately as the true control. They must not be read as comparisons with the older generic solver.','',
        '## Best observed combinations','',
        '| System | Combination | Previous best (s) | Candidate (s) | Paired reduction | Descriptive 95% interval |','|---|---|---:|---:|---:|---:|']
    for name,s in report['systems'].items():
        candidates={k:v for k,v in s['summary'].items() if k not in ['previous_best','hooked_control']}
        best=min(candidates,key=lambda k:candidates[k]['median_seconds'])
        s['best_observed']=best
        row=candidates[best]
        effect=row['comparisons']['previous_best']
        lo,hi=effect['bootstrap_95_percent']
        lines.append(f"| {name} | {best} | {s['summary']['previous_best']['median_seconds']:.6f} | {row['median_seconds']:.6f} | {effect['median_reduction_percent']:.2f}% | [{lo:.2f}, {hi:.2f}]% |")
    lines += ['', 'For a common combination without AVX-512, FMA plus AVX2 finalization reduced runtime by '+', '.join(f"{name}: {s['summary']['fma_finalize']['comparisons']['previous_best']['median_reduction_percent']:.2f}%" for name,s in report['systems'].items())+'. The small difference between finalization alone and FMA plus finalization for R(0) should not be interpreted as a firm winner.','',
        '## All combinations','',
        '| System | Option | Median solve (s) | Reduction vs previous winner | Descriptive 95% interval |','|---|---|---:|---:|---:|']
    for name,s in report['systems'].items():
        for label,row in s['summary'].items():
            effect=row['comparisons']['previous_best']
            lo,hi=effect['bootstrap_95_percent']
            lines.append(f"| {name} | {label} | {row['median_seconds']:.6f} | {effect['median_reduction_percent']:+.2f}% | [{lo:+.2f}, {hi:+.2f}]% |")
    lines += ['', '## Implementation','',
        '- FMA fuses the multiply-add/subtract inside complex Hamiltonian products. Accumulation across Hamiltonian couplings remains in its original order, but product rounding changes. Scalar tails remain available.',
        '- AVX-512 processes eight complex density entries per sparse-coupling row update, with 512-bit real and imaginary vectors. AVX2/FMA handles residual entries. Required ISA support is checked before use.',
        '- Finalization uses AVX2 for four off-diagonal entries at once: transposed scratch reads use gathers, diagonal Hamiltonian values are prepared in a compact array, and phase evolution, decay damping and interleaved packed stores are vectorized. Diagonal population-transfer sums and short tails remain scalar. The finalization width remains AVX2 even when the commutator uses AVX-512.',
        '- All options retain the sparse CSR/CSC traversal, Hamiltonian diagonal split, full density coordinates, and exact decay channels. No approximation, coefficient cutoff, state reduction or timestep change is introduced.', '',
        'The hooked control is the larger experimental kernel with all three additional options disabled. Comparing candidates with the separate previous-winner binary includes dispatch/layout overhead rather than hiding it. Eight randomized interleaved full-solve rounds were run per system: 288 timed trajectories total, plus warmups and strict references. Intervals resample whole paired rounds (20,000 draws) and are descriptive for this small sample. The best row is selected from the tested candidates, so its interval is not selection-adjusted or a cross-CPU guarantee.','',
        '## Accuracy and scope','',
        'All 288 timed trajectories pass full complex-density checks, including coherences, against strict previous-winner Tsit5 references. Candidate atol/rtol=1e-11/1e-9, reference=1e-13/1e-11. Criterion: 1% entry-relative error above reference magnitude 1e-6, absolute error <=1e-6 below, trace error <1e-9. Arbitrary positive full density matrices are also checked across four EOM times/all microwave configurations, drive multipliers 0/0.3/1/2, and changed detunings.','',
        '| System | Max normalized RHS error | Max relative density error (fraction) | Max small-entry absolute error | Max trace error | RHS counts across options |','|---|---:|---:|---:|---:|---|']
    for name,s in report['systems'].items():
        calls=sorted({c for values in s['rhs_call_counts'].values() for c in values})
        lines.append(f"| {name} | {s['max_normalized_rhs_error']:.3g} | {s['max_relative_density_error']:.3g} | {s['max_small_absolute_density_error']:.3g} | {s['max_trace_error']:.3g} | {calls} |")
    lines += ['', 'Physical payloads are identical to the preceding fusion benchmark, reused from saved fixtures and individually hashed in the raw data. These are 10 us E=200 V/cm, B=0.01 G performance fixtures with optical EOM beta=3.8 at 1.6 MHz. RC uses uniform X,J=2 initial population and exact 250 ns segmentation for the four microwave configurations; R(2), R(0), Q(1) use uniform modeled X populations. They are not full zero-E apparatus transits. Preparation/build costs are excluded from timing.','',
        'Reproduction: `../../check_extended_simd_dsu.py`; kernels: `../../extended_simd_dsu.rs` layered on the prior fused D+S+U implementation. `../../report_extended_simd_dsu.py` creates this report. `results.json` contains raw round orders, timings, density checks, step/RHS counts, fixture/native hashes, CPU features and summaries. All work was implemented/run directly, without agents.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'results.json').write_text(json.dumps(report,indent=2))
    print('\n'.join(lines[:16]))


if __name__=='__main__':
    main()
