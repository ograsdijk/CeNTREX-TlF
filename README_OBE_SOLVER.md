# Rust OBE Solver

This document describes the newer Lindblad/OBE solver path used by
`centrex_tlf.lindblad.solve_lindblad`.

## Recommended Path

For non-stiff optical Bloch equation workloads, use the Rust backend with the
expanded sparse RHS:

```python
from centrex_tlf.lindblad import prepare_lindblad_problem, solve_lindblad

prepared = prepare_lindblad_problem(
    obe_system,
    parameters,
    backend="rust",
    hamiltonian_representation="decomposed",
)

result = solve_lindblad(
    prepared,
    rho0,
    (0.0, 10e-6),
    solver="dopri5_fast",
    execution_mode="expanded_sparse",
    saveat=saveat,
    abstol=1e-9,
    reltol=1e-7,
    dt=1e-10,
)
```

Current defaults are:

- `prepare_lindblad_problem(..., hamiltonian_representation="decomposed")`
- `solve_lindblad(..., backend="rust", solver="explicit", execution_mode="expanded_sparse")`
- On the Rust backend, `solver="explicit"` is routed to `solver="dopri5_fast"`.
- On the Python backend, `solver="explicit"` remains the Python/SciPy reference path.

## Available Solvers

The main solver choices are:

- `dopri5_fast`: custom Rust Dormand-Prince 5(4) integrator. This is usually the fastest full-output path.
- `tsit5_fast`: custom Rust Tsitouras 5(4) integrator. This is useful to compare against Julia-style Tsit5 behavior and can be competitive for final-only output.
- `dopri5`: older Rust wrapper around `ode_solvers`. Kept for comparison and compatibility.
- `scipy`: SciPy `RK45` using the Rust matrix RHS callback.
- `scipy_bdf`: SciPy `BDF` using the Rust packed RHS callback and optional exact sparse Jacobian.
- `scipy_radau`: SciPy `Radau` using the Rust packed RHS callback and optional exact sparse Jacobian.
- `python` backend with `solver="explicit"`: pure Python reference implementation, mainly for correctness checks.

The native Rust `bdf`/DiffSol path was removed. Stiff methods are still
available through SciPy as `scipy_bdf` and `scipy_radau`.

## Execution Modes

The RHS execution modes are:

- `expanded_sparse`: default and recommended. Python lowers the decomposed Hamiltonian and dissipator into an entrywise sparse update graph over the packed Hermitian state. Rust evaluates that graph directly.
- `structured_upper`: builds the RHS using upper-triangular Hermitian structure and matrix-style kernels.
- `structured`: older structured dense/matrix path.
- `reference`: dense reference path, useful for debugging only.

For the fast solvers, output can be reduced:

- `output="full"` returns the packed density matrix trajectory.
- `output="populations"` returns only diagonal populations.
- `output="selected"` returns selected `(i, j)` density-matrix entries.
- `output_when="saveat"` records at requested save points.
- `output_when="final"` records only the final output.
- `dense_output=False` avoids dense interpolation coefficient work when it is not needed.

Selected entries are specified with zero-based density-matrix indices in
`output_indices`. The indices refer to the full `rho[i, j]` matrix, not to the
packed internal storage. Populations can be requested either with
`output="populations"` or as selected diagonal entries such as `(0, 0)`.
Coherences are selected with off-diagonal entries such as `(2, 7)`.

```python
# Save selected populations and coherences at each requested save point.
selected = solve_lindblad(
    prepared,
    rho0,
    (0.0, 10e-6),
    solver="dopri5_fast",
    execution_mode="expanded_sparse",
    saveat=saveat,
    output="selected",
    output_indices=[
        (0, 0),  # population rho[0, 0]
        (5, 5),  # population rho[5, 5]
        (2, 7),  # coherence rho[2, 7]
    ],
)

# selected.values has shape (len(selected.t), len(output_indices)).
rho_00 = selected.values[:, 0]
rho_55 = selected.values[:, 1]
rho_27 = selected.values[:, 2]
```

```python
# Save only the final value of selected entries.
final_selected = solve_lindblad(
    prepared,
    rho0,
    (0.0, 10e-6),
    solver="tsit5_fast",
    execution_mode="expanded_sparse",
    output="selected",
    output_indices=[(0, 0), (2, 7)],
    output_when="final",
    dense_output=False,
)

# final_selected.values has shape (len(output_indices),).
final_rho_00 = final_selected.values[0]
final_rho_27 = final_selected.values[1]
```

For Hermitian conjugate coherences, request the entry you want to inspect. For
example, `(2, 7)` returns `rho[2, 7]`, while `(7, 2)` returns `rho[7, 2]`.

## How It Works

`prepare_lindblad_problem` lowers the symbolic OBE system once:

- Parameters are adapted into a compact runtime graph.
- The Hamiltonian is lowered into a decomposed representation,
  `H(t) = sum_k c_k(t) H_k`, so Rust only reevaluates scalar coefficients at
  runtime.
- Collapse operators are converted into structured decay data.
- The `expanded_sparse` plan combines Hamiltonian and dissipator contributions
  into per-output update terms for the packed upper-triangular Hermitian density
  matrix.
- The Rust `PreparedLindbladPlan` stores only the data needed by Rust. Python-only
  reference structures are no longer serialized into the Rust plan.

During integration, Rust keeps the OBE state in packed real form. Diagonal
entries are stored as real populations. Off-diagonal upper-triangular entries are
stored as real/imaginary pairs. Lower-triangular entries are recovered from
Hermiticity when needed, avoiding redundant state variables and redundant RHS
work.

## Batch Solving and Grid Scans

For many independent trajectories, use `solve_lindblad_batch` instead of a
Python loop. The batch path keeps the trajectory loop in Rust, optionally runs
trajectories in parallel with Rayon, and returns one aggregated NumPy array.

```python
from centrex_tlf.lindblad import solve_lindblad_batch

batch = solve_lindblad_batch(
    prepared,
    rho0_batch,  # shape: (n_trajectories, n, n) or (n_trajectories, packed_len)
    (0.0, 200e-6),
    solver="dopri5_fast",
    execution_mode="expanded_sparse",
    output="selected",
    output_indices=[(0, 0), (2, 7)],
    output_when="final",
    dense_output=False,
    parallel=True,
)

# Final selected output shape: (n_trajectories, len(output_indices)).
final_rho_00 = batch.values[:, 0]
final_rho_27 = batch.values[:, 1]
```

Supported batch outputs in the first implementation are:

- `output="populations"` with shape `(n_trajectories, n_states)` for final output.
- `output="selected"` with shape `(n_trajectories, n_selected)` for final output.
- `output="populations"` with shape `(n_trajectories, n_times, n_states)` for save-at output.
- `output="selected"` with shape `(n_trajectories, n_times, n_selected)` for save-at output.

Full density-matrix batch output is intentionally not supported yet because it
can allocate very large arrays.

Parameter scans can avoid rebuilding the Rust plan for every trajectory. The
batch solver accepts per-trajectory base-parameter overrides and reevaluates
compound parameters in Rust:

```python
from centrex_tlf.lindblad import parameter_scan

parameter_values = np.array(
    [
        [0.5e6, -1.0e6],
        [1.0e6, 0.0],
        [2.0e6, 1.0e6],
    ],
    dtype=np.complex128,
)

scan_result = parameter_scan(
    prepared,
    rho0,
    (0.0, 200e-6),
    parameter_slots=["Ω0", "δ0"],
    parameter_batch=parameter_values,
    output="populations",
    output_when="final",
)
```

For structured parameter grids, use `grid_scan`. It creates the Cartesian
product of the provided axes and stores grid metadata on the result:

```python
from centrex_tlf.lindblad import grid_scan

grid = grid_scan(
    prepared,
    rho0,
    (0.0, 200e-6),
    scan={
        "δ0": np.linspace(-5e6, 5e6, 101),
        "Ω0": 2 * np.pi * np.array([0.5e6, 1.0e6, 2.0e6]),
    },
    output="selected",
    output_indices=[(0, 0), (5, 5)],
    output_when="final",
)

grid_shape = grid.metadata["grid_shape"]
values_on_grid = grid.values.reshape(*grid_shape, grid.values.shape[-1])
```

Only base parameters can be scanned directly. Compound parameters are updated by
changing their base-parameter dependencies.

## Benchmark

Benchmark problem:

- Transition: `transitions.R0_F1_3o2_F2`
- Coupling: one `polarization_Z` transition group
- State count: 65
- Time span: `0` to `10 us`
- Save points: 201 for full-output rows
- Initial state: equal population over ground states
- Tolerances: `abstol=1e-9`, `reltol=1e-7`, `dt=1e-10`
- Build: Rust release build, staged extension loaded directly

Machine:

- CPU: AMD Ryzen 7 9800X3D, user-provided; environment reports 16 logical processors
- OS: Windows `10.0.26200`
- Architecture: AMD64
- Python: `3.11.13`
- Rust: `rustc 1.94.0`

Timings from this checkout:

| Solver | Execution mode | Output | Median | Min | RHS calls | Accepted/rejected |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `dopri5_fast` | `expanded_sparse` | full `saveat` | 5.477 ms | 5.017 ms | 361 | 59 / 1 |
| `tsit5_fast` | `expanded_sparse` | full `saveat` | 7.090 ms | 6.535 ms | 337 | 55 / 1 |
| `dopri5_fast` | `structured_upper` | full `saveat` | 7.175 ms | 6.948 ms | 361 | 59 / 1 |
| `tsit5_fast` | `structured_upper` | full `saveat` | 7.440 ms | 6.802 ms | 337 | 55 / 1 |
| `scipy` | `expanded_sparse` | full `saveat` | 73.361 ms | 70.646 ms | n/a | n/a |
| `scipy_bdf` | `expanded_sparse` | full `saveat` | 134.711 ms | 120.834 ms | n/a | n/a |
| `dopri5_fast` | `expanded_sparse` | final populations | 5.028 ms | 4.024 ms | 361 | 59 / 1 |
| `tsit5_fast` | `expanded_sparse` | final populations | 5.635 ms | 4.325 ms | 337 | 55 / 1 |

These numbers are for a small benchmark system. For larger rotational-cooling
systems, solver step count and stiffness dominate more strongly, and SciPy stiff
methods can be much slower even with an exact Jacobian.
