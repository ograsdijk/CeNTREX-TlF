# `h_mat_elems_generic` lookup maps vs linear scan — measurement

Run:

```
cargo test --release bench_h_mat_elems -- --ignored --nocapture
```

(Windows, release build; source `rust/src/generate_hamiltonian.rs`, test
`generate_hamiltonian::tests::bench_h_mat_elems_lookup_vs_linear_scan`.)

Addresses the audit item "the new `h_mat_elems_generic()` lookup maps should be
benchmarked against the old linear scan for very small bases, although the project
normally operates at 64 or more states where the map-based path should be favored".

## What is being compared

`h_mat_elems_from_applied` builds one `HashMap<BasisState, Complex64>` per applied
state, then does `n^2/2` hashed lookups. The linear-scan reference
(`h_mat_elems_from_applied_linear_scan`, `cfg(test)`) instead rescans each applied
state's term list per matrix element: `n^2/2 * k` struct comparisons, where `k` is
terms per applied state.

`test_linear_scan_reference_matches_lookup_maps` pins that the two agree to 1e-12
across four X operators and four basis sizes, so the timings compare like with like.
`std::hint::black_box` wraps both inputs and each result matrix — without it LLVM can
strip the plain loop down to the one element the harness reads while the opaque
`HashMap` survives, which would fake a win for the scan. Timings were identical with
and without it, so no such stripping was happening.

## Repeats and variability

Each cell is 7 interleaved trials, each averaging over `reps` calls (`reps` scales as
`1/n^2`, clamped to `[5, 20000]`). Map and scan alternate *within* a trial rather than
running as two blocks, so slow drift — thermal throttling, a background task — hits both
roughly equally instead of loading onto whichever ran second.

**Individual timings are noisy.** Per-cell relative spread `(max-min)/median` across
trials reaches 91–99%, worst on the small-`n` cells where the timer resolution and the
`reps` clamp bite hardest, but also on some large ones (X `n=196` `h_sx` hit 85% in one
run). Anyone quoting a single microsecond figure from the table below should treat it as
good to roughly a factor of 2, not to three digits.

**The ratio is not noisy.** Because the two implementations are timed against each other
inside the same trial, the common-mode noise cancels. Across three independent process
runs:

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| worst single-trial `scan/map` | 0.384 | 0.411 | 0.389 |
| worst per-cell spread | 91.3% | 95.8% | 92.2% |

and the 48 per-cell median ratios agree run-to-run within ±0.02 almost everywhere
(largest disagreement 0.13 → 0.15 → 0.14 at X `n=64` `h_ff`). Cross-run medians of the
absolute times agree to ~5%.

The bench asserts `worst_ratio < 1.0` — i.e. it fails if any *single trial* ever had the
map at least as fast, rather than checking that the medians happen to be separated. The
observed worst case is 0.41, a 2.4x win for the scan in the least favourable trial of
~340 measured. That is the bound the conclusion rests on.

## Measured (median us per call over 7 trials)

X uncoupled basis, J = 0..jmax (`n = 4*(jmax+1)^2`):

| n | op | terms/state | map [us] | scan [us] | scan/map |
|---:|---|---:|---:|---:|---:|
| 4 | h_ff | 5.50 | 0.80 | 0.047 | 0.06 |
| 16 | h_ff | 8.75 | 6.45 | 0.51 | 0.08 |
| 36 | h_ff | 8.44 | 20.83 | 2.09 | 0.10 |
| 64 | h_ff | 10.22 | 65.28 | 9.47 | 0.15 |
| 100 | h_ff | 10.10 | 128.23 | 20.69 | 0.16 |
| 144 | h_ff | 11.17 | 251.76 | 53.70 | 0.21 |
| 196 | h_ff | 11.00 | 424.81 | 96.85 | 0.23 |
| 256 | h_ff | 11.73 | 811.48 | 291.01 | 0.36 |
| 256 | h_zz | 1.00 | 702.99 | 247.81 | 0.35 |

B coupled basis (Omega basis), J = 1..jmax:

| n | op | terms/state | map [us] | scan [us] | scan/map |
|---:|---|---:|---:|---:|---:|
| 12 | h_mhf_f | 2.83 | 2.87 | 0.15 | 0.05 |
| 32 | h_zz | 6.06 | 17.69 | 1.36 | 0.08 |
| 60 | h_mhf_f | 3.37 | 55.63 | 3.69 | 0.07 |
| 96 | h_zz | 6.98 | 127.39 | 13.78 | 0.11 |
| 192 | h_zz | 7.39 | 457.46 | 65.28 | 0.14 |
| 320 | h_zz | 7.62 | 1361.32 | 367.11 | 0.27 |

## Conclusion

**The premise of the audit item is wrong in both directions.** The lookup maps are not
merely unhelpful at small bases — the linear scan is faster at *every* size measured,
X and B, from 4 to 320 states, by 3x to 20x, and no single trial in any run went the
other way. There is no crossover in the range the project operates in.

The reason is that `k` is small and roughly constant: 1 to 12 terms per applied state
in X, 1 to 7.6 in B, set by the operator's selection rules rather than by basis size.
So the map pays `n` allocations plus `n^2/2` SipHash lookups to avoid `n^2/2 * k`
integer-struct comparisons, where `k` is single digits and the comparisons are cheap
and branch-predictable. The `scan/map` ratio does climb with `n` (0.06 to 0.36 in X),
so a crossover exists in principle, but linear extrapolation puts it near `n ~ 1000` —
far past the 64-320 states these builders see.

**No code change made.** The absolute stakes do not justify touching a core numeric
path: at the OBE default `Jmax_X = 4` (n = 100) the seven X operators cost ~0.8 ms
total with the maps, and they run under `rayon` in parallel, against an end-to-end
build of ~2.8 s. Reverting would save well under a millisecond. This follows the same
stopping rule as `PARTITIONED_PACKED_MAX_STATES` and the `representation="auto"` cost
model: record the measurement and the reasoning, keep the existing code.

If the assembly path ever becomes hot, the better fix is not reverting to the scan but
dropping the `n^2` structure entirely: build **one** `basis -> index` map over `qn`
(built once, not `n` times) and scatter each applied state's `k` terms into the result,
giving `O(n*k)` instead of `O(n^2)`. That is a real algorithmic change and needs its
own correctness work — both current implementations assume Hermiticity and mirror the
lower triangle, which a scatter would have to preserve.

## Caveats

- Laptop, Windows, not a quiet benchmarking machine; see the variability section for
  what that costs. The conclusion is stated as a ratio bound precisely because the
  absolute times are not trustworthy to better than a factor of ~2.
- `reps` scales as `1/n^2`, so the largest bases average over only ~30 calls per trial.
  Seven trials of ~30 calls is thin; it is enough here only because the effect being
  measured is 3-20x.
- X-uncoupled and B-coupled operators only, in the Omega basis. Operators with much
  larger `k` than these would shift the balance, but `k` is set by selection rules and
  none of the shipped operators come close.
