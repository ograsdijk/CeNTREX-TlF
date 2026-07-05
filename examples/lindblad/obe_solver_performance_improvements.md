# OBE Solver Performance Improvement Investigation

This note summarizes the current performance bottlenecks and likely improvement paths for the Lindblad OBE single-trajectory and grid-scan solvers. It focuses on solve time only, not OBE-system or reduced-Hamiltonian construction.

## Current State

The main production solve path is:

- Python wrappers in `centrex_tlf/lindblad/solve.py` and `centrex_tlf/lindblad/batch.py`
- Rust native single-solve API in `rust/src/lindblad/python_api.rs`
- Rust batch/grid glue in `rust/src/lindblad/ode_batch.rs`
- Rust adaptive ODE solvers in `rust/src/ode/dopri5.rs` and `rust/src/ode/tsit5.rs`
- Rust Lindblad RHS in `rust/src/lindblad/rhs.rs`

For realistic R(2) retained-opposite-parity calculations, the dominant cost is RHS evaluation inside Rust. Existing benchmark data for the R(2) system gives roughly:

| Case | Runtime | RHS calls | Notes |
| --- | ---: | ---: | --- |
| Single R(2) trajectory | `1.8-2.0 s` | `~171k` | On/near resonance |
| 25-point R(2) scan, 1 thread | `60.7 s` | `5.28M` | `0.41 trajectories/s` |
| 25-point R(2) scan, 8 threads | `11.7 s` | `5.28M` | `2.14 trajectories/s` |
| Single decay-sink variant | `1.84 s` | `~171k` | About `1.09x` faster than per-J sinks |

The scan effort is detuning-dependent. The existing detuning stats show about `171k` RHS calls near several resonant regions and up to about `329k` RHS calls near scan edges. That means parallel scan performance is affected by load imbalance as well as raw RHS speed.

## Already Improved

Final integral outputs can now run with no dense `saveat` grid:

```python
output = "photon_integral"
output_when = "final"
saveat = None
```

This lets Rust integrate photon/weighted observables over accepted solver steps and return one scalar per trajectory. It avoids saving dense population traces and avoids post-solve Python trapezoid integration.

The small integral-output benchmark found:

| Method | Mean time |
| --- | ---: |
| In-solver final integral, `saveat=None` | `0.002137 s` |
| Save populations and integrate in Python | `0.002850 s` |
| In-solver cumulative trace | `0.002665 s` |

This was a `1.33x` improvement over post-solve integration in the small benchmark. On the full R(2) system, the gain is expected to be smaller because RHS calls dominate, but this should still be the default for photon-count frequency scans and fitting.

## Highest-Value Improvement Candidates

### 1. Use Final In-Solver Integrals By Default For Scans

This is mostly a usage/API-default improvement, not a new solver algorithm.

Recommended scan settings:

```python
result = lindblad.grid_scan(
    prepared,
    rho0,
    t_span,
    scan={detuning_name: detunings_rad_s},
    output="photon_integral",
    output_when="final",
    saveat=None,
    integral_weights=photon_integral_weights,
    execution_mode="expanded_sparse",
    solver="dopri5",
)
```

Expected impact:

- Small to moderate speedup for R(2)-scale systems
- Larger memory reduction for large scans
- Cleaner scan outputs: one scalar per trajectory

Risk: low. This path is already implemented and tested.

### 2. Preallocate Direct Final Outputs In Rust Grid Scans

Current grid solves create one output object per trajectory, finish it, then concatenate all values afterward in `rust/src/lindblad/ode_batch.rs`.

For final scalar outputs, especially `photon_integral/final`, the grid solver could preallocate the final result vector and write each trajectory directly to its output slot. That would avoid per-trajectory result vectors and the final `extend_from_slice` collation pass.

Expected impact:

- Low to moderate for long R(2) trajectories
- More noticeable for small systems, short trajectories, and large grid scans
- Lower memory churn for large scans

Risk: low to medium. The shape contract must remain unchanged:

- Final batch/grid output: `(trajectory_count, width)`
- Saveat batch/grid output: `(trajectory_count, n_times, width)`

### 3. Reuse Per-Thread RHS And Output Workspaces In Grid Scans

Current grid solve construction inside each trajectory creates:

- A fresh `LindbladRhs`
- A fresh `RhsWorkspace`
- A fresh parameter-value vector
- A fresh output object

For large trajectories, this is secondary to RHS evaluation. For fitting/coarse scans with many shorter trajectories, this allocation/setup work can matter.

Recommended implementation:

- Use Rayon thread-local state or `map_init`
- Allocate one `RhsWorkspace` per worker thread
- Reuse scratch vectors and output buffers where possible
- Reset parameter overrides and output state per trajectory

Expected impact:

- Low for long single trajectories
- Moderate for large scans with short or reduced systems
- Also helps reduce allocator pressure at high thread counts

Risk: medium. Care is needed to preserve thread safety and avoid leaking state between trajectories.

### 4. Optimize The `expanded_sparse` Packed RHS Kernel

The hot path for the current default mode is `rhs_packed_expanded_sparse_into_with_profile` and especially `add_expanded_sparse_rhs_packed` in `rust/src/lindblad/rhs.rs`.

Current API note: `execution_mode="expanded_sparse"` uses split-input grouping by default. Pass `use_split_input_rhs=False` to `solve_lindblad`, `solve_lindblad_batch`, `parameter_scan`, `initial_condition_scan`, or `grid_scan` to disable the grouped input terms while retaining the split real/imag coefficient representation. This flag is intended for benchmarking and for checking full-system performance when the grouped path is not beneficial.

Potential kernel improvements:

- Store term coefficients as separate real and imaginary arrays
- Store packed input indices as separate arrays
- Split diagonal-real inputs from complex off-diagonal inputs to remove the `imag_sign == 0.0` branch in the inner loop
- Group terms by output width/length to improve cache behavior
- Consider specialized kernels for purely real coefficient blocks

Expected impact:

- Potentially high, because every accepted solver step calls the RHS six to seven times
- More valuable than output-only changes for R(2)-scale solves

Risk: medium to high. Requires careful numerical regression tests against the current RHS and line-shape outputs.

### 5. Investigate A Fixed-Step Fast Path For Coarse Scans

The tolerance sweep showed that accepted steps and wall time barely changed across reasonable tolerances. Very loose tolerances distorted photon counts before providing useful speedup.

This suggests the step size may be constrained by dynamics/stability rather than by the requested error tolerance. A fixed-step RK path could avoid adaptive-controller and error-estimator overhead.

Validation requirements:

- Compare photon counts
- Compare normalized line shapes
- Check peak positions to below `0.1-0.2 MHz`
- Check near opposite-parity and normal-parity features separately

Expected impact:

- Unknown until benchmarked
- Could help coarse scans/fitting initialization
- Should not replace adaptive solving for high-quality final traces without validation

Risk: medium.

## Lower-Value Or Less Promising Candidates

### Loosening Tolerances

The paired tolerance sweep did not show useful speed gains across a broad tolerance range. Very loose tolerances produced incorrect photon counts before substantially reducing work.

Conclusion: not a useful primary speed lever.

### Collapsing Decay-Only States

The existing benchmark found only about a `1.09x` speedup from collapsing decay-only ground states for the tested R(2) model.

Conclusion: useful but not enough by itself.

### Sparse `expm_multiply`

The tested sparse augmented-Liouvillian `expm_multiply` path was much slower than adaptive Rust ODE for the R(2)-scale constant-coefficient case.

Conclusion: not promising in its current SciPy sparse form. A future native/operator-specific exponential method would need to be a separate investigation.

## Recommended Next Work

1. Update scan/fitting notebooks and examples to use:

   ```python
   output="photon_integral"
   output_when="final"
   saveat=None
   ```

2. Implement direct preallocated final-output storage for batch/grid solves.

3. Add a microbenchmark for grid-scan overhead using a small fixed system and many trajectories.

4. Add an RHS-kernel benchmark that calls the packed `expanded_sparse` RHS directly and reports calls/sec.

5. Optimize `add_expanded_sparse_rhs_packed` using split real/imag arrays and branch removal.

6. Only after the RHS kernel is benchmarked, test a fixed-step scan path for coarse/fitting use cases.

## Practical Default For Current R(2) Frequency Scans

Use:

```python
result = lindblad.grid_scan(
    prepared,
    rho0,
    (0.0, t_end),
    scan={detuning_name: detunings_rad_s},
    solver="dopri5",
    execution_mode="expanded_sparse",
    output="photon_integral",
    output_when="final",
    saveat=None,
    integral_weights=photon_integral_weights,
    parallel=True,
    threads=threads,
    collect_stats=True,
)
```

For diagnostic time traces, use `photon_rate` or cumulative `photon_integral` with `output_when="saveat"` and an explicit `saveat`. For photon-count scans and fitting, avoid dense `saveat` unless a trace is actually needed.
