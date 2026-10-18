"""Summarize matched fusion screening and independent finalist confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent/'rust_specialized_rhs_results/fused_dsu'
CPU='AMD Ryzen 7 9800X3D 8-Core Processor'


def summarize(report,references):
    rng=np.random.default_rng(4078)
    for system in report['systems'].values():
        summary={}
        for name,rows in system['samples'].items():
            a=np.array([r['seconds'] for r in rows])
            assert len(a)==8
            row={'median_seconds':float(np.median(a)),'comparisons':{}}
            for reference in references:
                b=np.array([r['seconds'] for r in system['samples'][reference]])
                reduction=100*(1-a/b)
                boot=np.median(reduction[rng.integers(0,8,(20000,8))],axis=1)
                row['comparisons'][reference]={'median_reduction_percent':float(np.median(reduction)),
                    'bootstrap_95_percent':np.percentile(boot,[2.5,97.5]).tolist()}
            summary[name]=row
        system['summary']=summary
        system['rhs_call_counts']={name:sorted({r['rhs_calls'] for r in rows}) for name,rows in system['samples'].items()}
        system['max_relative_density_error']=max(r['accuracy']['max_relative'] for rows in system['samples'].values() for r in rows)
        system['max_small_absolute_density_error']=max(r['accuracy']['max_small_absolute'] for rows in system['samples'].values() for r in rows)
        system['max_trace_error']=max(r['trace_error'] for rows in system['samples'].values() for r in rows)
        assert all(r['accuracy']['passed'] and r['trace_error']<1e-9 for rows in system['samples'].values() for r in rows)
    report['cpu']=CPU


def main():
    screen=json.loads((ROOT/'results.json').read_text())
    confirmation=json.loads((ROOT/'confirmation.json').read_text())
    summarize(screen,['production','simd'])
    summarize(confirmation,['production','original_simd','decay_pack_simd'])
    lines=['# D+S+U diagonal, decay and packing fusion trials','',
        'Fusing decay and packed output adds a measured gain beyond the earlier SIMD winner for RC, R(2) and R(0). Q(1) has a positive median estimate, but its paired interval spans zero, so that additional gain is uncertain. Combining diagonal phase and damping arithmetic does not add a consistent further gain. No production library changes were made.','',
        f'Host CPU: **{CPU}**. Eight randomized interleaved full-solve rounds per system in each stage. Percentages are medians of within-round runtime reductions; positive means faster. Brackets are descriptive paired 95% bootstrap intervals (20,000 whole-round resamples), not cross-machine guarantees.','',
        '## Direct confirmation against the earlier SIMD winner','',
        '| System | Original D+S+U (s) | Earlier SIMD (s) | SIMD + fused decay/packing (s) | Reduction vs D+S+U | Additional reduction vs earlier SIMD |','|---|---:|---:|---:|---:|---:|']
    for name,s in confirmation['systems'].items():
        summary=s['summary']
        row=summary['decay_pack_simd']
        total=row['comparisons']['production']
        extra=row['comparisons']['original_simd']
        lines.append(f"| {name} | {summary['production']['median_seconds']:.6f} | {summary['original_simd']['median_seconds']:.6f} | {row['median_seconds']:.6f} | {total['median_reduction_percent']:.2f}% | {extra['median_reduction_percent']:.2f}% [{extra['bootstrap_95_percent'][0]:.2f}, {extra['bootstrap_95_percent'][1]:.2f}] |")
    lines += ['', 'The confirmation compiles the earlier SIMD kernel separately, preserving its original code rather than using SIMD-only within the larger experimental fusion function. The latter had additional layout/dispatch overhead, especially on the smaller systems. Thus the initial screen alone would overstate some marginal gains. Physical payloads are reused exactly from the screen, with their hashes recorded.', '',
        '## Tested operations and combinations','',
        '- Diagonal phase evolution is already included in the current commutator finalization. The additional diagonal experiment combines its complex coefficient with decay damping; it is only meaningful with fused decay.',
        '- Decay fusion subtracts damping and applies incoming population transfers while writing the commutator result, removing the separate dissipator traversal.',
        '- Output fusion writes directly into the ODE solver packed output slice, removing the intermediate upper-triangle result and later packing traversal. Packing-only retains a separate decay traversal on packed data.',
        '- Each meaningful combination was tested both with and without SIMD. The sparse Hamiltonian traversal, diagonal split, exact decay channels and full density coordinates are retained. No new cutoff or physical approximation is introduced.','',
        '## Full factorial screen','',
        '| System | Variant | Median solve (s) | Reduction vs production |','|---|---|---:|---:|']
    for name,s in screen['systems'].items():
        for label,row in s['summary'].items():
            lines.append(f"| {name} | {label} | {row['median_seconds']:.6f} | {row['comparisons']['production']['median_reduction_percent']:+.2f}% |")
    lines += ['', '## Confirmation of the diagonal arithmetic combination','',
        '| System | All three with SIMD vs decay/packing with SIMD | Descriptive 95% interval |','|---|---:|---:|']
    for name,s in confirmation['systems'].items():
        row=s['summary']['all_three_simd']['comparisons']['decay_pack_simd']
        lo,hi=row['bootstrap_95_percent']
        lines.append(f"| {name} | {row['median_reduction_percent']:+.2f}% | [{lo:+.2f}, {hi:+.2f}]% |")
    lines += ['', 'Without SIMD, the all-three variant gave paired runtime reductions vs production of '+', '.join(f"{name}: {s['summary']['all_three_scalar']['comparisons']['production']['median_reduction_percent']:+.2f}%" for name,s in confirmation['systems'].items())+'. These are not universally positive; kernel selection should remain available.', '',
        '## Accuracy and scope','',
        'All 576 timed trajectories (416 screening, 160 confirmation), plus warmups, pass full complex-density checks against strict current-source Tsit5 references. Candidate atol/rtol are 1e-11/1e-9; reference 1e-13/1e-11. Criterion: 1% relative error for reference entries of magnitude >=1e-6, absolute error <=1e-6 below; trace error <1e-9. The screen also checks arbitrary positive full densities with coherences at four EOM times/all RC switching configurations, Rabi multipliers 0, 0.3, 1 and 2, and changed detunings. Fusion algebra may change floating-point rounding but not the model.', '',
        '| System | Maximum relative density error (fraction) | Max small-entry absolute error | Max trace error | RHS counts across variants |','|---|---:|---:|---:|---|']
    for name,s in confirmation['systems'].items():
        calls=sorted({c for values in s['rhs_call_counts'].values() for c in values})
        lines.append(f"| {name} | {s['max_relative_density_error']:.3g} | {s['max_small_absolute_density_error']:.3g} | {s['max_trace_error']:.3g} | {calls} |")
    lines += ['', 'These are the existing short 10 us E=200 V/cm, B=0.01 G performance fixtures, optical EOM beta=3.8 at 1.6 MHz. RC uses uniform X,J=2 population and exact 250 ns segments for the four microwave polarization configurations; optical fixtures use uniform modeled X populations. They are not full zero-electric-field apparatus transits. Fixed timestep and solver settings were not tuned in this experiment.', '',
        'Reproduction: `../../check_fused_dsu.py` runs the screen; `../../confirm_fused_dsu.py` confirms against the original SIMD implementation using saved identical physical payloads. Kernel: `../../fused_dsu.rs`, reusing the vector row helper from `../../block_simd_dsu.rs`. `../../report_fused_dsu.py` generates this report. Raw round orders, timings, density errors, accepted/rejected/RHS counts, summaries and native-source hashes are in `results.json` and `confirmation.json`. All work was performed directly by the main agent, without delegated agents.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'results.json').write_text(json.dumps(screen,indent=2))
    (ROOT/'confirmation.json').write_text(json.dumps(confirmation,indent=2))
    print('\n'.join(lines[:15]))


if __name__=='__main__':
    main()
