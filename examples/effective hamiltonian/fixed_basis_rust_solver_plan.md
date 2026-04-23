# Fixed-Basis Grid Rust Solver Plan

## Goal

Implement the fixed-basis compact interpolated method in Rust.

The physics/setup stays in Python:

- build the fixed-basis compact model,
- evaluate endpoint `OperatorBundle`s on a 1D coordinate grid,
- precompute the Liouvillian component operators,
- pass those numeric arrays to Rust.

Rust only does propagation:

- evaluate runtime scalar functions,
- interpolate the precomputed operator grid,
- apply the RHS,
- run fast ODE solvers.

This matches the fixed-basis method already validated against the instantaneous-basis reference.

## Scope

First version supports one scalar interpolation coordinate:

```text
s
```

Python decides what `s` means. Examples:

- `Ez`
- `Ex`
- `Ey`
- `Bz`
- `Bx`
- `By`
- field magnitude along a fixed direction
- any one-parameter path through electric or magnetic field space

Multi-dimensional electric/magnetic interpolation is out of scope for the first version.

## Python Precompute Layer

Python prepares the fixed grid. For each grid value `s_i`, Python evaluates the fixed-basis model at the corresponding electric or magnetic field.

Example for electric `z`:

```python
E = E_origin + s_i * E_direction
B = fixed_B
```

Example for magnetic `z`:

```python
E = fixed_E
B = B_origin + s_i * B_direction
```

For each endpoint bundle, precompute:

```text
L_internal(s_i)    — superoperator for field-dependent internal Hamiltonian
L_opt(s_i)         — superoperator for optical coupling (multiplied by Ω/2)
L_det(s_i)         — superoperator for detuning shift
L_diss(s_i)        — superoperator for Lindblad dissipator
jump_rate_operator(s_i) — Σ C†C, used for photon scattering rate observable
```

The `jump_rate_operator` is not part of the RHS — it is an observable operator
used to compute the scattering rate `Tr[ρ · J]` during post-processing or as a
native reduction during integration (photon integral = ∫ Tr[ρ · J] dt).

Then convert each complex superoperator into a real operator acting on the
existing packed Hermitian density-matrix layout:

```text
packed_dy = A_packed @ packed_y
```

This keeps Rust simple and preserves the exact Python fixed-basis interpolation
semantics.

## Runtime Scalar Expressions

Do not pass Python callbacks into Rust. Python builds typed scalar expression objects and lowers them to a compact Rust payload.

The expression system should be class-based, not string-based.

Example:

```python
t = Time()
z = Variable("z")

variables = RuntimeVariables(
    z=Linear(t, offset=z0, slope=velocity),
)

field_coordinate = Tabulated(z, z_grid, ez_profile_vcm)
rabi_rate = Gaussian(z, amplitude=omega0, center=z_laser, sigma=sigma_z)
detuning = Constant(0.0)
```

This means:

```text
z(t) = z0 + velocity*t
Ez(t) = interp(z_grid, ez_profile_vcm, z(t))
Omega(t) = omega0 * exp(-(z(t)-z_laser)^2 / (2*sigma_z^2))
delta(t) = 0
```

The same system also supports direct time dependence:

```python
field_coordinate = Sinusoid(
    t,
    offset=25.0,
    amplitude=20.0,
    angular_frequency=2*np.pi/T,
)
rabi_rate = Constant(2*np.pi*1e6)
```

This means:

```text
Ez(t) = 25 + 20*sin(2*pi*t/T)
Omega(t) = constant
```

## First Expression Types

Implement these scalar expression classes first:

```python
Time()
Variable(name)
Constant(value)
Linear(input, offset, slope)
Sinusoid(input, offset, amplitude, angular_frequency, phase=0.0)
SquareWave(input, low, high, period, phase=0.0, duty=0.5)
Tabulated(input, grid, values)
Gaussian(input, amplitude, center, sigma, baseline=0.0)
```

All scalar quantities can use this system:

- `field_coordinate`
- `rabi_rate`
- `detuning`
- later: laser phase, polarization weights, or other scalar controls

## What `Tabulated` Means

`Tabulated(input, grid, values)` represents a sampled function evaluated by
monotone piecewise cubic Hermite interpolation (PCHIP).

PCHIP is preferred over linear interpolation because it produces continuous
first derivatives. With linear interpolation, `ds/dt` is discontinuous at grid
boundaries, causing the ODE solver to reject steps and reduce `dt` at every
grid point. PCHIP avoids this while preserving monotonicity (no overshoots).

Example:

```python
field_coordinate = Tabulated(z, z_grid, ez_profile_vcm)
```

At runtime Rust evaluates:

```text
z = z(t)
Ez = pchip_interp(z_grid, ez_profile_vcm, z)
```

Then `Ez` is used as the coordinate for the precomputed OBE operator grid.

There are two separate interpolations:

```text
1. Spatial/profile interpolation (PCHIP):
   Ez = pchip_interp(z_grid, ez_profile_vcm, z(t))

2. OBE operator-grid interpolation (linear):
   L = linear_interp(s_grid, L_grid, Ez)
```

This distinction is important. `Tabulated` describes a runtime scalar profile
(smooth, PCHIP). The fixed-grid plan describes how Liouvillian operators vary
with the scalar coordinate (linear interpolation between precomputed grid
points is acceptable here because operator variation with field is typically
smooth and the grid can be refined).

## Rust Plan

Add a new Rust plan type, separate from `PreparedLindbladPlan`:

```rust
FixedBasisGridPlan {
    n_states: usize,
    packed_len: usize,
    grid: Vec<f64>,
    operators: Vec<GridOperator>,
    metadata: ...
}

enum GridOperator {
    Dense {
        name: String,
        matrices: Vec<Vec<f64>>,  // n_grid x (packed_len * packed_len)
    },
    Sparse {
        name: String,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<Vec<f64>>,    // n_grid x nnz
    },
}
```

During `prepare_fixed_basis_grid`, Python computes the density of each operator
across all grid points. If density < 30%, store as CSR sparse; otherwise dense.
Typical densities for TlF systems:

- `L_internal`: very sparse (block-diagonal in J), ~5-10% density
- `L_opt`: sparse (only coupling entries), ~2-5% density
- `L_det`: diagonal or near-diagonal, ~1-2% density
- `L_diss`: sparse (structured decay), ~5-10% density

For n=65 (packed_len=4225), sparse storage reduces each matvec from
O(packed_len²) = O(17.8M) to O(nnz × packed_len) ~ O(200K), a ~90x reduction.

Python reports the chosen format and density per operator at preparation time.

## Rust RHS

### Workspace

Pre-allocate all buffers at solver initialization:

```rust
FixedBasisGridWorkspace {
    dy_lo: Vec<f64>,       // packed_len, accumulator for grid point i
    dy_hi: Vec<f64>,       // packed_len, accumulator for grid point i+1
    last_interval: usize,  // cached grid interval index (hint)
}
```

### Interval Lookup with Caching

The ODE stepper calls the RHS ~7 times per step at nearby `t` values (for
stages k1..k7). The field coordinate `s(t)` changes negligibly between stages.
Cache the last interval index and check it first:

```rust
fn find_interval(&mut self, s: f64, grid: &[f64]) -> (usize, f64) {
    let hint = self.last_interval;
    if hint < grid.len() - 1 && grid[hint] <= s && s <= grid[hint + 1] {
        let w = (s - grid[hint]) / (grid[hint + 1] - grid[hint]);
        return (hint, w);
    }
    // fallback: binary search
    let idx = grid.partition_point(|&g| g <= s).saturating_sub(1);
    let idx = idx.min(grid.len() - 2);
    let w = (s - grid[idx]) / (grid[idx + 1] - grid[idx]);
    self.last_interval = idx;
    (idx, w)
}
```

For monotonic trajectories (molecule flying through a field), the hint hits on
6/7 RHS calls per step — O(1) instead of O(log n_grid).

### RHS Evaluation

At each RHS call:

```text
s = field_coordinate(t)
Omega = rabi_rate(t)
delta = detuning(t)
```

Find the interval in `grid` (cached):

```text
(i, w) = find_interval(s, grid)
```

Apply the RHS using fused endpoint form without constructing interpolated
matrices. For sparse operators, use sparse matvec:

```text
dy_lo =
  L_internal_i @ y + 0.5*Omega * L_opt_i @ y
  + delta * L_det_i @ y + L_diss_i @ y

dy_hi =
  L_internal_{i+1} @ y + 0.5*Omega * L_opt_{i+1} @ y
  + delta * L_det_{i+1} @ y + L_diss_{i+1} @ y

dy = (1-w) * dy_lo + w * dy_hi
```

No allocation inside RHS.

No Python callbacks inside RHS.

No matrix construction or interpolation inside RHS — only matvec + axpy.

## Python API Sketch

Preparation:

```python
prepared = prepare_fixed_basis_grid(
    model,
    coordinate_grid=np.linspace(0.0, 50.0, 41),
    varying_field="electric",
    field_origin=(0.0, 0.0, 0.0),
    field_direction=(0.0, 0.0, 1.0),
    fixed_magnetic_field=(0.0, 0.0, 1e-5),
)
```

Time-dependent field:

```python
t = Time()

result = solve_fixed_basis_grid(
    prepared,
    rho0,
    t_span=(0.0, 2e-6),
    field_coordinate=Sinusoid(
        t,
        offset=25.0,
        amplitude=20.0,
        angular_frequency=2*np.pi/2e-6,
    ),
    rabi_rate=Constant(2*np.pi*1e6),
    detuning=Constant(0.0),
    solver="dopri5_fast",
)
```

Position-dependent field and laser:

```python
t = Time()
z = Variable("z")

variables = RuntimeVariables(
    z=Linear(t, offset=z0, slope=velocity),
)

result = solve_fixed_basis_grid(
    prepared,
    rho0,
    t_span=(0.0, 20e-6),
    variables=variables,
    field_coordinate=Tabulated(z, z_grid, ez_profile_vcm),
    rabi_rate=Gaussian(z, amplitude=omega0, center=z_laser, sigma=sigma_z),
    detuning=Constant(0.0),
    solver="dopri5_fast",
)
```

Numeric shorthand should adapt to `Constant`:

```python
rabi_rate=2*np.pi*1e6
detuning=0.0
```

## Solver Integration

First implementation:

- add `dopri5_fast` for `FixedBasisGridPlan`,
- reuse the existing packed output conventions where practical,
- support `output="full"`, `output="populations"`, and `output="selected"` if the shared fast-output code can be reused cleanly.

Second implementation:

- add `tsit5_fast`,
- refactor the fast solvers around a shared RHS trait,
- add final-only and reduced-output paths if not included in the first pass.

## Validation

Validate Rust fixed-grid against the existing Python fixed-grid dense method:

- same coordinate grid,
- same `rho0`,
- same `T_EVAL`,
- same field trajectory,
- same `rabi_rate`,
- same `detuning`.

Test cases:

- constant coordinate,
- linear ramp,
- sinusoid,
- tabulated spatial profile with `z(t)=z0+v*t`.

Metrics:

- max packed-state difference,
- photons relative error,
- excited integral relative error,
- final excited population error,
- sink population error,
- trace error.

Expected agreement should be solver-tolerance limited.

## Benchmark

Update `benchmark_q1_effective_hamiltonian.py` to compare:

- Python fixed-basis dense BDF,
- Rust fixed-basis grid `dopri5_fast`,
- Rust fixed-basis grid `tsit5_fast` once implemented,
- Python instantaneous-basis dense BDF,
- static compact Rust baseline.

## Key Difference from Current Python Implementation

The current Python runtime (`effective_hamiltonian_runtime.py`) performs the
following at each `effective_bundle()` call during the ODE integration:

1. Interpolate the full recycling decay kernel (n² × n² PSD matrix)
2. PSD-project (eigendecompose, clamp negatives, reconstruct)
3. Eigendecompose again to extract collapse operators C_k = √λ_k · reshape(v_k)
4. Build the dissipator superoperator from the collapse operators

This is O(n⁸) per RHS call (eigendecomposition of n² × n² matrix) and dominates
the runtime for large systems.

The Rust plan avoids this entirely by precomputing the full dissipator
superoperator L_diss at each grid point in Python, then interpolating the
superoperator directly (piecewise linear) in Rust. The interpolated
superoperator is not guaranteed PSD, but for small interpolation intervals and
smooth collapse operator variation, the error is solver-tolerance limited.

If PSD preservation is needed, an alternative: precompute and store the collapse
operators at each grid point, interpolate them directly (element-wise linear),
and apply the Lindblad formula in Rust. This is cheaper than eigendecomposition
but requires storing n_collapse × n × n arrays per grid point.

## Later Optimizations

After correctness:

- add native photon-integral and excited-population reductions,
- add batch/grid scan support (reuse `solver_batch.rs` pattern with
  per-trajectory expression parameter overrides),
- add adaptive coordinate-grid diagnostics (warn if operator variation between
  adjacent grid points exceeds a threshold),
- add multidimensional interpolation if needed (bilinear for 2D field grids),
- consider PCHIP for the operator-grid interpolation as well if linear
  interpolation causes step rejection at operator grid boundaries.
