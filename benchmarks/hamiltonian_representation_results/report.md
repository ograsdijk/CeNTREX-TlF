# `representation="auto"` cost model — measurement

Run: `uv run python benchmarks/bench_hamiltonian_representation.py`
(Windows, release build, medians of 5 for lowering/prepare, 3 for solve.)

Addresses the audit item "the Hamiltonian lowering `auto` heuristic exists, but its cost
model is still not documented or benchmark-justified"
(`centrex_tlf/lindblad/ir.py:434-441`).

## Cost model scores

```
entrywise_cost  = len(temps) + len(entries)
decomposed_cost = len(coefficients) + 0.15 * basis_term_count + (1 if static)
```

| system | n_states | entrywise cost | decomposed cost | ratio | auto picks |
| --- | ---: | ---: | ---: | ---: | --- |
| A — R(0) F1'=3/2 F'=2 | 65 | 2146.0 | 4.0 | 537x | decomposed |
| B — R(2) F1'=7/2 F'=3, opposite parity retained | 154 | 11936.0 | 10.1 | 1182x | decomposed |

## Measured time

| system | representation | lower [ms] | prepare [ms] | solve structured [ms] | solve expanded_sparse [ms] |
| --- | --- | ---: | ---: | ---: | ---: |
| A | entrywise | 30.9 | 45.5 | 3.09 | n/a |
| A | decomposed | 25.3 | 42.3 | 3.31 | **1.76** |
| B | entrywise | 217.8 | 836.6 | 13.74 | n/a |
| B | decomposed | 171.8 | 811.2 | 16.24 | **11.88** |

`expanded_sparse` is unavailable for entrywise by construction — `lower_expanded_sparse_rhs`
returns `None` for a non-decomposed plan (`ir.py:453`) and `rhs.rs:1256` then raises.

## Conclusion

**The `0.15` and `+1` constants do not matter and cannot be usefully calibrated.** The two
branches are separated by a factor of 537–1182 on real systems, so no plausible value of
either constant changes the decision. They would have to move by three orders of magnitude
to flip a single case.

**The heuristic's choice is also the empirically right one.** Decomposed is faster to lower
(−18% / −21%) and to prepare, and it is the only representation that unlocks
`expanded_sparse`, which is the fastest solve path on both systems (1.76 ms vs 3.09 ms;
11.88 ms vs 13.74 ms). Entrywise's single measured win — `structured` solve at 154 states,
13.74 ms vs 16.24 ms — is irrelevant because decomposed's `expanded_sparse` beats both.

**Context that limits how much this matters:** `"auto"` is not the default
(`plan_static.py:120` uses `"decomposed"`), nothing in the package passes it, and choosing
entrywise while requesting `expanded_sparse` produces a clear error rather than a silent
slowdown.

Recommendation: document the constants as an inconsequential tie-breaker rather than
recalibrating them. This follows the stopping rule already used for
`PARTITIONED_PACKED_MAX_STATES` in `IMPLEMENTATION_AUDIT.md` — when measurement shows a
statistic does not separate the cases, record why and keep the existing rule.

## Caveat

`rhs_calls` came back `None` from `solver_stats` in this harness, so the per-RHS-call
column could not be computed; the solve times above are whole-trajectory wall clock over an
identical 10 us span and are directly comparable, but they are not normalized per RHS call.
This is the same stats-key drift noted elsewhere in the audit.
