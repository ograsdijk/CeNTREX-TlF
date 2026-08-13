# Scan Speedup Investigation — Summary (2026-07-13)

Three candidates for speeding up OBE parameter scans (r2-in-static-E-field
peak-ratio workload: ~10 polarization fractions x ~120 detunings), measured on
the notebook-identical system. Details in `single_system_polarization_scan.md`
and `exact_propagator_and_threads.md` in this folder. Context:
`../step_size_diagnostics_results/step_size_diagnostics_report.md` (per-
trajectory stepping is oscillation-limited at ~1.1 s; tolerance and detuning
do not change cost; the per-level rotating frame was measured 2.4-6x SLOWER
and is not an option).

## 1. Single system for the whole polarization x detuning grid — ADOPT

One OBE system with X and Z coupling fields and runtime amplitude symbols
(`px=sqrt(1-fz)`, `pz=sqrt(fz)`), one prepared plan, whole 2D grid as one
`parameter_scan` over a (detuning, px, pz) table. Normalization verified
exactly: the Z component does not couple the mF=0 -> mF'=1 main states, so
`main_coupling(fz) = sqrt(1-fz) * main_X` to machine precision and a single
constant `rabi` binding is correct. All 10 fz curves match per-fz-rebuilt
references (peak positions exact; the only visible difference, 1.5e-5 rel at
fz=3e-4, traces to the reference's own coupling-threshold zeroing — the
single-system curve is the more faithful one).

Measured (10 fz x 36 detunings): **2.22x end-to-end** (165.3 -> 74.5 s);
build+prepare overhead **11.6x** lower (87.6 -> 7.5 s). No library changes
needed. Script: `bench_single_system_polarization_scan.py`.

## 2. Exact propagator (eig of the time-independent Liouvillian) — NICHE

Numerically clean, contrary to the non-normality worry: cond(V) ~ 15, eig
residual ~ 6e-15; populations match a reltol-1e-9 reference to ~7e-11.
Cost per grid point (jacobian probe + eig + LU: 1.07-1.28 s) vs one dopri5
trajectory (1.13 s):

- Current scan shape (one initial state, final-only photon integral):
  **break-even — no gain.**
- Each ADDITIONAL initial state: ~1.5-3 ms (**~400-750x** cheaper) — wins for
  velocity/population averaging over many rho0 at fixed parameters.
- Full 801-point saveat trajectory: ~60-75 ms (**~15-20x**) — wins for
  trajectory-resolved outputs (e.g. Ramsey-style/time-trace studies).

Keep as a tool for those shapes; not a replacement for the default scan
path. Script: `bench_exact_propagator.py`.

## 3. Thread scaling + batch worker reuse — FREE THROUGHPUT

- `grid_scan` over 120 detunings, 16 logical cores: 156.7 s (1 thread) ->
  **17.1 s (threads=None)**, 9.19x at 57% efficiency (94% at 2 threads;
  SMT/bandwidth-limited beyond 8). Results bit-identical across thread
  counts. Use `threads=None` (the notebook already does).
- `solve_batch_ode` now reuses per-thread workers like the grid path
  (`rust/src/lindblad/ode_batch.rs`, rebuilt, 295 tests pass):
  `parameter_scan` on cheap trajectories 1.37x serial / 1.18x at 16 threads,
  now matching `grid_scan` throughput, outputs bit-identical.

## Net effect on the real workload

10 fz x 120 detunings, 16 cores: previously ~10 builds + 1200 stepped
trajectories; with the single-system restructure + full threading the same
scan is ~7.5 s setup + ~150 s solve. Combined with the earlier setup-path
work (builds were ~60 s each at the session start), the end-to-end scan went
from ~10+ minutes to ~2.5 minutes. Remaining per-trajectory cost is
oscillation-limited physics (see step-size diagnostics); the only unexplored
lever for that is an exponential/interaction-picture integrator, whose
10-25x estimate is UNCONFIRMED and weakened by the rotating-frame negative
result.
