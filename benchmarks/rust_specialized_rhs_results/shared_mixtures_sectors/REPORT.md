# Shared RC mixture and exact density closure checks

The RC trajectory timing uses three independently initialized, uniform X-state J blocks under the same parameters. Each candidate evolves and returns all three full complex density matrices. The fixture spans 10 µs at E=200 V/cm and B=0.01 G; it is a numerical performance fixture, not a molecular transit prediction.

## Structural density-coordinate closure

The threshold-free structural closure is a safe upper bound on reachable support: it uses the union of all static and lowered Hamiltonian basis terms and all nonzero decay transfers. Hamiltonian couplings move either density index; population decay transfers move only diagonal coordinates. Any coordinate outside this closure remains exactly zero. The reported removable fraction is the portion the structure proves can be omitted; the graph may conservatively retain coordinates that cancel dynamically.

| System / initial support | Coordinates in safe closure | Full coordinates | Proven removable |
|---|---:|---:|---:|
| RC: uniform X,J=1 | 4,360 | 4,900 | 11.02% |
| RC: uniform X,J=2 | 4,360 | 4,900 | 11.02% |
| RC: uniform X,J=3 | 4,360 | 4,900 | 11.02% |
| R(2), uniform fixture input: R2 fixture initial X population | 727 | 1,681 | 56.75% |
| R(2), timing seeds: R2 pure X basis seed 0 | 727 | 1,681 | 56.75% |
| R(2), timing seeds: R2 pure X basis seed 1 | 727 | 1,681 | 56.75% |
| R(2), timing seeds: R2 pure X basis seed 2 | 727 | 1,681 | 56.75% |

## RC trajectory timing

The unpatched and same-time-cached serial controls call the same native trajectory helper for each mixture. Shared lockstep calls one native solver with the three density blocks and reuses each stage H. A separate existing batch-API run with three workers reports practical parallel throughput. Shared step acceptance uses the maximum per-mixture RMS normalized error.

| Candidate | Median wall time (s) |
|---|---:|
| independent_serial_production_native | 7.781259 |
| independent_parallel_production | 5.199394 |
| independent_serial_cached_native | 7.663732 |
| shared_lockstep_cached | 8.082277 |

The shared candidate’s solver `rhs_calls` are outer stage calls, each of which evaluates three density RHSs. For this reason it performs 86,219 outer stage calls but 258,657 density RHS applications, versus 244,479 density RHS applications in the three independent serial trajectories (+5.80% for shared). The stage-call count alone must not be read as total density-RHS work.

Speed changes vs the unpatched native serial baseline: independent_serial_cached_vs_production_native_percent +1.51%, shared_cached_vs_independent_cached_percent -5.46%, shared_cached_vs_production_native_percent -3.87%, shared_cached_vs_production_parallel_wall_percent -55.45%.

Every timed density was compared entry-by-entry with the stricter tolerance result (1% relative error for entries at or above 1e-6 magnitude; 1e-6 absolute below that), and its trace was checked. The raw paired rounds and per-mixture error checks are in `shared_mixtures_sectors.json`.

R(2) receives a corresponding three-seed full-density timing, using three distinct X basis states supported by the fixture. The R(2) closure is shown for both its original uniform X initial state and those benchmark seeds. See the raw JSON for timings and checks.

## R(2) trajectory timing

| Candidate | Median wall time (s) |
|---|---:|
| independent_serial_production_native | 0.715342 |
| independent_parallel_production | 0.414167 |
| independent_serial_cached_native | 0.709179 |
| shared_lockstep_cached | 0.710405 |

## Paired timing reductions

Percent change is computed within each round as 100×(1−candidate/reference); positive values mean the first candidate is faster. Intervals are nonparametric percentile bootstraps over the eight paired rounds and are descriptive given the small sample.

| System | Comparison | Median paired reduction | 95% bootstrap interval |
|---|---|---:|---:|
| RC | independent serial cached native vs independent serial production native | +1.24% | [+0.18%, +2.11%] |
| RC | shared lockstep cached vs independent serial cached native | -4.72% | [-5.97%, -3.78%] |
| RC | shared lockstep cached vs independent serial production native | -3.64% | [-4.12%, -3.34%] |
| RC | shared lockstep cached vs independent parallel production | -54.12% | [-94.40%, -50.45%] |
| R2 | independent serial cached native vs independent serial production native | +0.87% | [-1.45%, +2.06%] |
| R2 | shared lockstep cached vs independent serial cached native | +0.28% | [-1.96%, +1.81%] |
| R2 | shared lockstep cached vs independent serial production native | +1.09% | [-1.28%, +2.26%] |
| R2 | shared lockstep cached vs independent parallel production | -72.12% | [-79.65%, -70.15%] |

## Solver stage calls and density RHS work

| System | Independent serial stage calls (= density RHSs) | Shared outer stage calls | Shared density RHS applications (3 per stage) | Change in density RHS applications |
|---|---:|---:|---:|---:|
| RC | 244,479 | 86,219 | 258,657 | +5.80% |
| R2 | 103,623 | 36,611 | 109,833 | +5.99% |

The same-time cache avoids rebuilding the Hamiltonian on the second and third mixture at a shared stage time, but it still evaluates the complete dissipative and Hamiltonian density RHS separately for each mixture. Shared lockstep therefore makes more total density RHS applications in these fixtures and has no measured wall-time advantage over independent serial integration.

The maximum observed entry-relative error above the 1e-6 magnitude cutoff was 0.0845% for RC and 0.0001% for R(2). Maximum absolute error below the cutoff was 7.736e-11 for RC and 1.245e-12 for R(2); maximum trace error was 1.998e-15 for RC and 5.551e-15 for R(2).

See the raw JSON for every paired round, per-mixture density checks, trace results and step statistics.
