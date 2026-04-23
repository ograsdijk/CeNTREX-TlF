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
L_internal(s_i)
L_opt(s_i)
L_det(s_i)
L_diss(s_i)
jump_rate_operator(s_i)
```

Then convert each complex operator into a dense real operator acting on the existing packed Hermitian density-matrix layout:

```text
packed_dy = A_packed @ packed_y
```

This keeps Rust simple and preserves the exact Python fixed-basis interpolation semantics.

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

`Tabulated(input, grid, values)` represents a sampled function evaluated by linear interpolation.

Example:

```python
field_coordinate = Tabulated(z, z_grid, ez_profile_vcm)
```

At runtime Rust evaluates:

```text
z = z(t)
Ez = linear_interp(z_grid, ez_profile_vcm, z)
```

Then `Ez` is used as the coordinate for the precomputed OBE operator grid.

There are two separate interpolations:

```text
1. Spatial/profile interpolation:
   Ez = interp(z_grid, ez_profile_vcm, z(t))

2. OBE operator-grid interpolation:
   L = interp(s_grid, L_grid, Ez)
```

This distinction is important. `Tabulated` describes a runtime scalar profile. The fixed-grid plan describes how Liouvillian operators vary with the scalar coordinate.

## Rust Plan

Add a new Rust plan type, separate from `PreparedLindbladPlan`:

```rust
FixedBasisGridPlan {
    n_states: usize,
    packed_len: usize,
    grid: Vec<f64>,
    l_internal: Vec<Vec<f64>>,
    l_opt: Vec<Vec<f64>>,
    l_det: Vec<Vec<f64>>,
    l_diss: Vec<Vec<f64>>,
    jump_rates: Vec<Vec<f64>>,
    metadata: ...
}
```

Each operator matrix has shape:

```text
packed_len x packed_len
```

stored as contiguous `f64`.

## Rust RHS

At each RHS call:

```text
s = field_coordinate(t)
Omega = rabi_rate(t)
delta = detuning(t)
```

Find the interval in `grid`:

```text
s_i <= s <= s_{i+1}
w = (s - s_i) / (s_{i+1} - s_i)
```

Apply the RHS without constructing an interpolated matrix:

```text
dy =
  interp(L_internal, s) @ y
  + 0.5*Omega * interp(L_opt, s) @ y
  + delta * interp(L_det, s) @ y
  + interp(L_diss, s) @ y
```

Fused endpoint form:

```text
dy =
  (1-w)*(L_internal_i@y + 0.5*Omega*L_opt_i@y + delta*L_det_i@y + L_diss_i@y)
  + w*(L_internal_{i+1}@y + 0.5*Omega*L_opt_{i+1}@y + delta*L_det_{i+1}@y + L_diss_{i+1}@y)
```

No allocation inside RHS.

No Python callbacks inside RHS.

No dense complex matrix reconstruction inside RHS.

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

## Later Optimizations

After correctness:

- exploit sparsity/structure in packed operators,
- avoid storing full dense packed matrices if memory becomes important,
- add native photon-integral and excited-population reductions,
- add batch/grid scan support,
- add adaptive coordinate-grid diagnostics,
- add multidimensional interpolation if needed.
