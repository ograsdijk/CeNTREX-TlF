# D+S+U diagonal, decay and packing fusion trials

Fusing decay and packed output adds a measured gain beyond the earlier SIMD winner for RC, R(2) and R(0). Q(1) has a positive median estimate, but its paired interval spans zero, so that additional gain is uncertain. Combining diagonal phase and damping arithmetic does not add a consistent further gain. No production library changes were made.

Host CPU: **AMD Ryzen 7 9800X3D 8-Core Processor**. Eight randomized interleaved full-solve rounds per system in each stage. Percentages are medians of within-round runtime reductions; positive means faster. Brackets are descriptive paired 95% bootstrap intervals (20,000 whole-round resamples), not cross-machine guarantees.

## Direct confirmation against the earlier SIMD winner

| System | Original D+S+U (s) | Earlier SIMD (s) | SIMD + fused decay/packing (s) | Reduction vs D+S+U | Additional reduction vs earlier SIMD |
|---|---:|---:|---:|---:|---:|
| rc | 2.697033 | 2.129138 | 2.051135 | 24.46% | 4.23% [3.28, 6.58] |
| r2 | 0.261938 | 0.231035 | 0.221409 | 14.69% | 4.11% [3.42, 4.73] |
| r0 | 0.080156 | 0.078505 | 0.069811 | 12.58% | 11.01% [8.20, 11.24] |
| q1 | 0.158739 | 0.144141 | 0.137163 | 14.19% | 6.46% [-8.73, 13.92] |

The confirmation compiles the earlier SIMD kernel separately, preserving its original code rather than using SIMD-only within the larger experimental fusion function. The latter had additional layout/dispatch overhead, especially on the smaller systems. Thus the initial screen alone would overstate some marginal gains. Physical payloads are reused exactly from the screen, with their hashes recorded.

## Tested operations and combinations

- Diagonal phase evolution is already included in the current commutator finalization. The additional diagonal experiment combines its complex coefficient with decay damping; it is only meaningful with fused decay.
- Decay fusion subtracts damping and applies incoming population transfers while writing the commutator result, removing the separate dissipator traversal.
- Output fusion writes directly into the ODE solver packed output slice, removing the intermediate upper-triangle result and later packing traversal. Packing-only retains a separate decay traversal on packed data.
- Each meaningful combination was tested both with and without SIMD. The sparse Hamiltonian traversal, diagonal split, exact decay channels and full density coordinates are retained. No new cutoff or physical approximation is introduced.

## Full factorial screen

| System | Variant | Median solve (s) | Reduction vs production |
|---|---|---:|---:|
| rc | production | 2.765805 | +0.00% |
| rc | hooked_control | 2.755873 | -0.83% |
| rc | simd | 2.254475 | +17.67% |
| rc | decay | 2.744827 | +1.89% |
| rc | simd_decay | 2.126646 | +22.64% |
| rc | pack | 2.778082 | +0.89% |
| rc | simd_pack | 2.177364 | +22.05% |
| rc | decay_pack | 2.694990 | +2.45% |
| rc | simd_decay_pack | 2.069886 | +25.23% |
| rc | diagonal_decay | 2.774197 | -1.09% |
| rc | simd_diagonal_decay | 2.154388 | +21.78% |
| rc | diagonal_decay_pack | 2.648044 | +4.31% |
| rc | simd_diagonal_decay_pack | 2.071895 | +25.31% |
| r2 | production | 0.280126 | +0.00% |
| r2 | hooked_control | 0.283963 | -3.21% |
| r2 | simd | 0.273736 | +0.97% |
| r2 | decay | 0.278460 | -4.34% |
| r2 | simd_decay | 0.251996 | +6.81% |
| r2 | pack | 0.304110 | -6.21% |
| r2 | simd_pack | 0.242319 | +9.78% |
| r2 | decay_pack | 0.280787 | +4.94% |
| r2 | simd_decay_pack | 0.232241 | +16.79% |
| r2 | diagonal_decay | 0.279088 | -3.59% |
| r2 | simd_diagonal_decay | 0.249229 | +8.90% |
| r2 | diagonal_decay_pack | 0.271209 | +0.13% |
| r2 | simd_diagonal_decay_pack | 0.226342 | +15.75% |
| r0 | production | 0.080771 | +0.00% |
| r0 | hooked_control | 0.083586 | -2.47% |
| r0 | simd | 0.087263 | -8.86% |
| r0 | decay | 0.081165 | +0.20% |
| r0 | simd_decay | 0.079493 | +0.44% |
| r0 | pack | 0.079584 | +1.15% |
| r0 | simd_pack | 0.078374 | +1.33% |
| r0 | decay_pack | 0.072782 | +9.39% |
| r0 | simd_decay_pack | 0.072458 | +10.04% |
| r0 | diagonal_decay | 0.078913 | +3.15% |
| r0 | simd_diagonal_decay | 0.079156 | +1.74% |
| r0 | diagonal_decay_pack | 0.071740 | +10.76% |
| r0 | simd_diagonal_decay_pack | 0.072161 | +10.63% |
| q1 | production | 0.150352 | +0.00% |
| q1 | hooked_control | 0.159247 | -4.62% |
| q1 | simd | 0.157640 | -3.17% |
| q1 | decay | 0.154513 | -2.37% |
| q1 | simd_decay | 0.144652 | +4.97% |
| q1 | pack | 0.152865 | -1.11% |
| q1 | simd_pack | 0.145301 | +3.59% |
| q1 | decay_pack | 0.144504 | +4.05% |
| q1 | simd_decay_pack | 0.135507 | +11.75% |
| q1 | diagonal_decay | 0.155805 | -3.76% |
| q1 | simd_diagonal_decay | 0.147633 | +1.75% |
| q1 | diagonal_decay_pack | 0.143841 | +5.61% |
| q1 | simd_diagonal_decay_pack | 0.135433 | +11.51% |

## Confirmation of the diagonal arithmetic combination

| System | All three with SIMD vs decay/packing with SIMD | Descriptive 95% interval |
|---|---:|---:|
| rc | +0.25% | [-2.67, +0.84]% |
| r2 | +0.91% | [-0.71, +2.55]% |
| r0 | +0.64% | [-1.21, +3.04]% |
| q1 | -0.05% | [-2.86, +5.50]% |

Without SIMD, the all-three variant gave paired runtime reductions vs production of rc: +1.64%, r2: -0.28%, r0: +13.41%, q1: +5.90%. These are not universally positive; kernel selection should remain available.

## Accuracy and scope

All 576 timed trajectories (416 screening, 160 confirmation), plus warmups, pass full complex-density checks against strict current-source Tsit5 references. Candidate atol/rtol are 1e-11/1e-9; reference 1e-13/1e-11. Criterion: 1% relative error for reference entries of magnitude >=1e-6, absolute error <=1e-6 below; trace error <1e-9. The screen also checks arbitrary positive full densities with coherences at four EOM times/all RC switching configurations, Rabi multipliers 0, 0.3, 1 and 2, and changed detunings. Fusion algebra may change floating-point rounding but not the model.

| System | Maximum relative density error (fraction) | Max small-entry absolute error | Max trace error | RHS counts across variants |
|---|---:|---:|---:|---|
| rc | 0.000845 | 7.74e-11 | 1.11e-15 | [83495] |
| r2 | 3.71e-07 | 6.8e-14 | 1.33e-15 | [38531] |
| r0 | 1.3e-08 | 6.47e-15 | 2.66e-15 | [68813] |
| q1 | 1.98e-07 | 1.93e-13 | 8.88e-16 | [54353] |

These are the existing short 10 us E=200 V/cm, B=0.01 G performance fixtures, optical EOM beta=3.8 at 1.6 MHz. RC uses uniform X,J=2 population and exact 250 ns segments for the four microwave polarization configurations; optical fixtures use uniform modeled X populations. They are not full zero-electric-field apparatus transits. Fixed timestep and solver settings were not tuned in this experiment.

Reproduction: `../../check_fused_dsu.py` runs the screen; `../../confirm_fused_dsu.py` confirms against the original SIMD implementation using saved identical physical payloads. Kernel: `../../fused_dsu.rs`, reusing the vector row helper from `../../block_simd_dsu.rs`. `../../report_fused_dsu.py` generates this report. Raw round orders, timings, density errors, accepted/rejected/RHS counts, summaries and native-source hashes are in `results.json` and `confirmation.json`. All work was performed directly by the main agent, without delegated agents.
