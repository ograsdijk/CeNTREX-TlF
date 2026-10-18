# Exact density-support reduction benchmark

Eight randomized interleaved rounds per system. Current prepared H and coefficient DAG are enabled. Positive reductions mean faster than full D+S+U. Intervals are descriptive paired bootstrap intervals, not general performance guarantees.

| System | Method | Median solver time (s) | Paired runtime reduction | 95% interval |
|---|---|---:|---:|---:|
| rc | full_dsu | 2.722359 | +0.00% | [+0.00, +0.00]% |
| rc | full_expanded | 14.465852 | -430.42% | [-440.34, -412.79]% |
| rc | reduced_dsu | 2.980921 | -8.54% | [-10.08, -3.52]% |
| rc | reduced_expanded_rows | 16.687459 | -513.26% | [-518.86, -495.50]% |
| r2 | full_dsu | 0.271431 | +0.00% | [+0.00, +0.00]% |
| r2 | full_expanded | 0.615293 | -127.89% | [-133.95, -123.09]% |
| r2 | reduced_dsu | 0.271775 | -0.41% | [-7.25, +2.45]% |
| r2 | reduced_expanded_rows | 0.507301 | -87.73% | [-94.06, -80.77]% |

The reduced D+S+U candidate keeps only reachable ODE coordinates, expands them into a full packed density for each existing RHS call, and gathers the derivatives. The reduced expanded-sparse candidate additionally removes output rows and terms with unreachable inputs, but still assembles the complete coefficient/term-value arrays. Neither is a blockwise D+S+U kernel. A future blockwise kernel could behave differently.

RC uses a uniform J=2 initial mixture (4360 of 4900 real density coordinates retained). R(2) uses its uniform X fixture input (727 of 1681 retained). Closure uses every Hamiltonian basis term and directed diagonal decay transfers without a new magnitude threshold. The expanded Liouvillian is checked for closure invariance. Three compatible random positive density probes agree with full RHS calculations, and omitted derivative rows are checked for zero leakage.

All candidate trajectories pass complete complex-density checks against a strict Tsit5 reference: 1% relative error above reference magnitude 1e-6, absolute error <=1e-6 below, trace error <=1e-9. Reduced error control retains the full-dimensional RMS denominator, matching the zeros omitted from the reference state. Output is reconstructed to the full density matrix.

These are 10 microsecond E=200 V/cm, B=0.01 G EOM performance fixtures, with existing 250 ns segmentation; not full zero-E apparatus trajectories. Candidate atol/rtol=1e-11/1e-9; reference=1e-13/1e-11. Timings exclude preparation/compilation. The full expanded-sparse control genuinely selects ExpandedSparse; it does not alias D+S+U.

Reproducer: ../../check_density_reduction.py. Raw timings, density errors, term counts and source hashes: density_reduction.json. No production code changes.
