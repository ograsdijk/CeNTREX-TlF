"""Benchmarks for the RF Ramsey simulator.

`reference.py` builds a one-shot truth reference at Jmax=6 (cached to disk).
`perf_bench.py` runs each variant against it and writes PERF_REPORT.md.

Variants live next to the baseline propagator under ../ramsey_rf/, importable
as `from ramsey_rf.propagator_truncated import propagate_midpoint_truncated` etc.
"""
