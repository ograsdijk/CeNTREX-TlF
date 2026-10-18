# Full separately generated Hamiltonian RHS and dissipator

Completed 2026-10-16. This is the previously missing Rust-only experiment:
generate the complete `-i[H,rho]` contribution and the dissipator independently,
then compile both as system-specific functions. The candidate does **not** call
the generic D+S+U commutator. No Julia benchmark or production library change was made.

## Results

One CPU thread on AMD Ryzen 7 9800X3D; one BLAS thread. Release Rust 1.98.0,
opt-level 3, fat LTO, codegen-units 1, no fastmath. Median of three trajectory
observations per method, alternating order after warmup:

| System | Levels | Simulated duration | D+S+U (s) | Full generated H RHS + D RHS (s) | Change |
|---|---:|---:|---:|---:|---:|
| RC | 70 | 1 us | 0.599963 | 3.282792 | 5.47 times slower |
| R(2) | 41 | 10 us | 0.278848 | 0.300788 | 7.87% slower |
| R(0) | 18 | 10 us | 0.085708 | 0.075116 | 12.36% less time |
| Q(1) | 26 | 10 us | 0.163023 | 0.156775 | 3.83% less time |

RC's 10 us test was screened out because its generated RHS exceeded 1.5 times
the baseline cost. These are short benchmark trajectories, not full apparatus
transits or parallel scan throughput. Small gains, especially Q(1), warrant more
paired observations before promotion to the library.

Native RHS timings (five samples, each 1000 repetitions at four EOM times):

| System | D+S+U (us/call) | Full generated (us/call) |
|---|---:|---:|
| RC | 32.176 | 171.749 |
| R(2) | 6.163 | 6.801 |
| R(0) | 1.018 | 0.896 |
| Q(1) | 2.595 | 2.509 |

The RC slowdown is in RHS evaluation, not extra adaptive steps: both candidates
take 18,683 calls, 3,113 accepted steps and zero rejected steps in the 1 us test.
The earlier [hybrid benchmark](SPLIT_FUNCTIONS_REPORT.md), which generates H
assembly and D but retains the generic commutator, remains a different candidate;
its 8.7% RC improvement must not be attributed to this full generated RHS.

## Generated implementation

`bench_rust_separate_generated_rhs.py` builds Hamiltonian-only equations from the
Hermitian upper triangle, with real diagonals and conjugated lower entries.
It applies `sympy.cse(..., optimizations="basic")` independently to H RHS and D
RHS. This uses the old Julia emitter's CSE setting without running Julia.
The candidate calls `direct_coefficients`, `generated_hamiltonian_rhs`, then
`generated_dissipator`, which adds D using the original density input.

Hamiltonian arithmetic is emitted as fixed complex scalar expressions; density
packing is converted to complex upper entries. Temporary blocks of 128 and output
blocks of 32 bound compilation size. The dissipator uses a separate function and
temporary namespace. Its CSE pass finds zero reusable temporaries for these
fixtures. RC H CSE produces 9,740 temporaries and reduces symbolic operation count
from 389,107 to 223,558 (complex symbolic operations, not hardware FLOPs).

This demonstrates this generator's performance, not a limit on every possible
specialized layout, SIMD strategy or symbolic simplification. Large generated code
and temporary storage are plausible causes of RC's cost; no hardware-counter
profiling was performed to establish the cause.

## Preparation cost

| System | Fresh model preparation (s) | RHS generation (s) | Native compilation (s) |
|---|---:|---:|---:|
| RC | 12.818 | 124.885 | 89.034 |
| R(2) | 3.735 | 9.395 | 11.535 |
| R(0) | 1.618 | 0.970 | 6.044 |
| Q(1) | 2.062 | 3.141 | 7.012 |

Compilation uses already cached Cargo dependencies. RC totals about 227 s including
model preparation; the earlier roughly 9 s generation/compile figure belonged to
the hybrid variant. Generated binaries live in `.cache`, but source generators,
reports, timings and checks live in `benchmarks` for inclusion in Git.

## Matched physics and validation

These reuse the earlier performance fixtures, not the zero-electric-field
production RC model: E=[0,0,200] V/cm, B=[0,0,0.01] G, optical EOM 1.6 MHz with
beta=3.8. RC uses P(2) plus J=1↔2 and J=2↔3 microwaves, 1 MHz polarization
switching with a quarter-period offset. Gates are frozen at each exact 250 ns
segment midpoint while EOM coefficients retain absolute time. Optical fixtures
are R(2) F'=4, R(0) F'=2 and Q(1) F'=2 with opposite excited parity retained.
Optical Rabi is Gamma; other fixture details are in the shared `model` builder.

All candidates use the same packed state, Tsit5 source, initial state and boundaries.
Candidate atol/rtol are 1e-11/1e-9, compared against D+S+U at 1e-13/1e-11.
Every tested trajectory passes the complex density-entry criterion: 1% relative
error for reference magnitudes at least 1e-6, absolute error at most 1e-6 below
that cutoff. This includes coherences, not only populations.

Full RHS checks use random positive Hermitian densities and the trajectory initial
state at four EOM times/all four RC switch configurations. Additional independent
dense-reference checks isolate contributions: ground-supported random coherences
have zero D, while identity and H-squared densities have zero commutator (up to
roundoff). Both contributions pass normalized error/residual <1e-10 on all four
systems. These structured isolated checks complement the arbitrary-density sum
checks; they do not independently isolate D on every possible coherence.

## Reproduce and raw evidence

From the repo with its virtualenv, set PYTHONIOENCODING=utf-8 and run:

```
.venv/Scripts/python.exe benchmarks/bench_rust_separate_generated_rhs.py
.venv/Scripts/python.exe benchmarks/check_rust_separate_rhs_parts.py
```

The second command reuses the binaries generated by the first. Per-system
`*_separate.json` files record samples, density errors, source hashes and preparation
costs; `*_separate_parts_check.json` records isolated contribution checks.
