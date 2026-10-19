# Package integration plan: CUDA coefficient and Hamiltonian sharing

Status: implemented for the 0.3.0 release. Results and validation are recorded in
[the integration report](spb_cu1_gpu_results/integrated_sharing/REPORT.md).

- Coefficient sharing and single-/multiple-component H sharing are integrated,
  including per-trajectory storage for a nonseparable remainder.
- Both controls default to `auto`; `off` disables either independently.
  `hamiltonian_sharing="on"` bypasses the arithmetic performance gate for
  explicit benchmarking, while retaining structural/storage eligibility checks.
- Multiple H components were about 10% slower in the tested RC workload.
  Auto uses a conservative arithmetic-cost gate to avoid that regression.
- The final CUDA/periodic suite passed 62 tests. CU1, RC and R(2) evidence and
  raw timings are retained with source hashes. CPU/dense-eig code is unchanged.
- Included in the 0.3.0 release preparation after package acceptance checks.

The design sections below retain the original implementation plan; these notes
record the resulting API and the measured RC exception.

## Evidence and objective

The isolated CU1 prototype reduced a complete 1024-trajectory varying-height scan
from 130.918 to 77.105 s (1.698x). Complex density differences were near machine
precision. Shared-waveform caching supplied most of the gain; sharing the
off-diagonal Hamiltonian added 7–13% in separate short screens. The already-shared
envelope control was essentially unchanged.

Evidence: [factorization report](spb_cu1_gpu_results/factorization/REPORT.md).
These results establish CU1 performance, not RC performance. The prototype doubled
the waveform-table allocation by leaving unused columns in place; package
integration must improve that layout rather than copying it verbatim.

## Scope and proposed public controls

Integrate both optimizations into the existing CUDA fixed-step backend. Keep CPU,
static dense eigendecomposition, physical recipes, timestep selection, RHS
selection and graph_steps defaults unchanged. Do not install the benchmark's
copied solver or its process-local class substitutions as a second backend.

Proposed additions to `CudaSolverOptions`:

```python
CudaSolverOptions(
    coefficient_sharing="auto",  # "auto" or "off"
    hamiltonian_sharing="auto",  # "auto" or "off"
)
```

Target default is `auto` only after the acceptance checks below pass. Auto uses
sharing where supported and worthwhile, with the existing computation as the
fallback. Both `off` restores the existing coefficient/Hamiltonian paths; the
existing `precompute` option keeps its current meaning. The controls are separate
so users and benchmarks can isolate each optimization. During development exercise
the new paths explicitly before enabling their target defaults.

`precompute=False` must not start allocating time tables. Static-factor caching
can still be used if beneficial. Existing `precompute_bytes` bounds the complete
time table, including new shared subexpressions; do not silently raise the default
budget. Unsupported expressions fall back normally; invalid physics/input retains
the original validation/error behavior rather than being hidden by a fallback.

## 1. Build an explicit lowering plan

Primary file: `centrex_tlf/lindblad/_cuda_lowering.py`.

- Classify scalar DAG nodes by both time dependence and variation across the
  actual parameter batch. Distinguish constants, static trajectory factors,
  shared waveforms and expressions that require evaluation per trajectory/stage.
- Expose the exact transverse factor of common-height unrotated Gaussian
  multipass helpers. Preserve all longitudinal profiles, pass order/losses,
  normalization and switching. Use helper identities and algebraic structure,
  never approximate equality at a sample point or CU1-specific parameter names.
- Keep rewritten children, symbolic representations and dependency metadata
  consistent. Deduplicate only equivalent nodes evaluated with the same time
  convention. Switch helpers must retain the clipped step's interior-time
  convention even when their subexpressions are cached.
- Produce explicit coefficient/cache mappings and optional separable-H factors.
  Use conservative rewrites with a valid domain: no unsafe complex square-root,
  conjugation, branch or division identities. Validate widths and preserve real
  helper argument checks. Varying pass heights, widths or temporal profiles must
  not be falsely identified as one common transverse factor.
- Analyze independently driven transitions separately. Caching does not require
  all drives to have the same amplitude or waveform. Initially support the
  prototype's verified multipass identities and simple generic scalar DAG cases;
  leave more elaborate symbolic factorization as a later extension.

## 2. Integrate bounded caches

Primary file: `centrex_tlf/lindblad/cuda.py`.

- Evaluate static trajectory factors once per chunk. Evaluate shared time nodes
  once per stage, using the existing fixed grid when precomputation is enabled.
- Pack only live common coefficients and shared subexpressions into the time
  table. Do not reserve unused original variable-coefficient columns alongside
  appended copies. Account for static buffers separately and use a conservative
  policy for cheaply recomputed expressions or large static caches.
- Respect batch chunking, per-chunk support reduction and parameter dependence.
  Start with conservative solve-wide dependence where required for compatibility;
  chunk-local detection is optional and must not add repeated expensive analysis.
- Budget exhaustion must disable only unsupported/unaffordable caching, with a
  correct direct-evaluation fallback. Cover both precomputed and live evaluation.
- Guard inactive/zero-width captured graph steps before reading stage caches.
  Preserve FSAL, output-boundary clipping, restart behavior and all step schedules.
- Use ordinary generated kernel interfaces; no benchmark-style global monkeypatch,
  copied backend, hard-coded slot names or kernel source string replacement.

Expose lightweight chunk statistics: active coefficient-sharing mode, static and
shared expression counts, waveform-table bytes, H storage bytes, and fallback
reason where useful. Preserve existing statistics keys and make diagnostic modes
describe the computation actually used. Do not expose internal cache details in
normal simulation progress displays.

## 3. Integrate shared off-diagonal Hamiltonians

Primary files: `cuda.py`, `centrex_tlf/lindblad/_cuda_kernels.py`.

- First integrate the verified case: all nonzero off-diagonal components share
  one finite real trajectory factor. Keep existing shared storage when the whole
  H is already trajectory-independent.
- Generate the unscaled shared H from the factorized coefficient DAG; do not
  normalize by a selected trajectory's amplitude. This avoids sample dependence,
  underflow-sensitive division and zero-amplitude denominators.
- Apply the static factor only to the off-diagonal Hamiltonian commutator.
  Preserve diagonal terms, Doppler/detuning, decay, photon accumulation and output
  reconstruction. Support zero drive and complex polarizations without violating
  conjugation/Hermiticity; fall back when the factor itself cannot safely be real.
- Support the direct RHS first, which is the measured winning configuration.
  Tiled RHS must remain correct and can use existing H storage unless separately
  validated/benchmarked. Do not silently change the user's RHS choice.
- Extend this to multiple independently scaled components, including separate UV,
  MW(1–2) and MW(2–3) contributions. Express the off-diagonal Hamiltonian as
  `H_off,b(t) = sum_g A_g(b) * H_g(t)`: construct each shared component once per
  stage and apply its static trajectory factor during commutator evaluation.
  Keep any nonseparable remainder on the existing per-trajectory path. A shared
  component need not correspond to a named transition; derive groups from exact
  coefficient structure and compatible static factors, including conjugation.
- Automatic selection is dependence analysis, not model-name recognition or
  approximate sampled matching. Identify static factors and shared time nodes,
  group compatible contributions, and select the existing shared-H path, one
  scaled component, multiple scaled components, or per-trajectory storage.
  Account for component count, support, batch size and added RHS work; auto must
  retain the existing path when extra components are unhelpful. Record the actual
  component count and selected/fallback path in chunk statistics.
- Implement the verified single-factor case first, then prototype and benchmark
  the multiple-component extension before enabling it in auto. This extension is
  now in scope, but its RC speed benefit remains to be measured. Do not assume
  that saving H assembly offsets the cost of evaluating several commutators.

## 4. Correctness and focused performance acceptance

Add relevant tests to `tests/lindblad/test_cuda.py` and focused lowering tests that
can run without a GPU. Port useful prototype cases into normal tests, not the
copied benchmark modules. Exercise both `fixed_tsit5` and `fixed_ck54`.

Required correctness coverage:

- Common height, varied height/power, zero drive, common width changes and complex
  polarization; two independent drives, different pass centers, varied temporal
  frequencies, and unsupported/nonseparable cases.
- Multiple shared H components with independent UV/microwave-like factors,
  different time profiles and overlapping matrix support. Include zero factors,
  complex polarizations, a nonseparable remainder and chunk-dependent sharing.
  Compare one-component, multiple-component and existing H paths on the same
  small model, ensuring no drive scales another drive or the diagonal/decay terms.
- All four on/off combinations of the controls on a compact model; auto fallback
  with precompute disabled, insufficient budgets and unsupported H sharing.
- Multiple chunks, changed initial coherences, density reduction on/off, both RHS
  choices, selected/full/population/photon outputs and density_saveat metadata.
- Odd step counts and graph tails, boundaries coinciding with outputs, affine
  switching/Doppler reversals and step schedules. Include empty/zero-length edge
  cases already supported by the solver.
- Same-step complex densities should agree near roundoff; investigate maximum
  absolute differences above 1e-10. Preserve existing independent-reference tests
  and CU1 strict-reference checks, including coherences, positivity and trace.

Performance scope is deliberately small:

1. Reuse the CU1 grids, dt and 12 us interval. Compare both-off, waveform-only
   and combined modes on varying heights; include the already-shared control
   and height-plus-power case. Three interleaved rounds after warmup, batch 1024.
2. Check RC on its existing fixed-power/shared-envelope workload and a small
   varying-height/power workload with independently varied UV and both microwave
   powers, keeping forward velocity shared within a batch. Compare both-off,
   coefficient-only, single-factor/fallback and multiple-component paths where
   applicable. This measures the additional value of grouping independent drives.
   Use large simultaneous GPU batches and short physical intervals for timing;
   use a small representative set for full-transit density validation. Do not
   change the RC physics to force an optimization to apply.
3. Use one compact R(2) correctness/regression case for a different level layout;
   no new broad solver/timestep sweep. Benchmark R(2) further only if a regression
   or different sharing structure requires it.
4. Confirm the winning integrated CU1 path against both-off on one paired
   full-transit scan. Repeat only if timing variation obscures the result.

Measure complete public-call wall time, setup, integration event spans and actual
buffer sizes. Record CPU load and obtain peak allocated memory when practical;
buffer size is not peak device memory. Compare matching outputs and numerical
settings. Accept a reproducible gain beyond variation, correct bounded-memory
fallbacks and no meaningful regression for the already-shared control. Auto must
decline unhelpful transformations if their setup/storage overhead causes a
material regression. Make no RC speedup claim until measured.

## 5. Documentation and delivery

- Update `README_OBE_SOLVER.md` with option semantics, the disable example,
  supported factorization, memory budget behavior and unchanged dt validation
  requirements. Add a brief orientation note to `AGENTS.md` if appropriate.
- Keep the prototype evidence as historical measurements; store integrated
  benchmarks separately with source hashes, GPU/software details and raw rounds.
- Review that no CPU/dense-eig semantics or unrelated recipe defaults changed.
  Run focused CUDA/lowering tests and existing relevant CUDA regression tests;
  broaden checks only for new failures or shared infrastructure changes.
- Organize reviewable changes as lowering/cache integration, shared-H kernels,
  and tests/docs/evidence. Commit/push only when separately authorized.

Completion means controlled coefficient and single-/multiple-component H paths,
observable safe fallbacks,
passing complex-density checks, documented memory behavior and measured CU1/RC
results. The later user instruction to proceed authorized implementation and
validation. Publishing remains separately authorized.
