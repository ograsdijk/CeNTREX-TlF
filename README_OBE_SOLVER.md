# OBE and Effective Lindblad Solver Map

This document is the practical map for the current density-matrix solver stack.
It covers the full OBE/Lindblad path in `centrex_tlf.lindblad`, the Rust-side
batch and scan APIs, and the lower-dimensional effective-Hamiltonian Lindblad
path in `centrex_tlf.effective_hamiltonian`.

## Recommended Paths

Use the full OBE Rust path when you need the complete Hilbert-space model:

```python
import numpy as np

from centrex_tlf.lindblad import prepare_lindblad_problem, solve_lindblad

saveat = np.linspace(0.0, 10e-6, 201)
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
    solver="dopri5",
    execution_mode="expanded_sparse",
    saveat=saveat,
    output="populations",
    output_when="saveat",
    abstol=1e-9,
    reltol=1e-7,
    dt=1e-10,
)
```

Use Rust-side scans when many independent trajectories share one prepared OBE
system. For final-value scans, keep output compact:

```python
import numpy as np

from centrex_tlf.lindblad import grid_scan

# `delta0` and `omega0` are Parameter objects registered on the same
# LindbladParameters used to prepare `prepared`.
# `target_indices` are the density-matrix diagonal indices to collect.
target_entries = [(idx, idx) for idx in target_indices]

scan_result = grid_scan(
    prepared,
    rho0,
    (0.0, 200e-6),
    scan={delta0: detuning_axis, omega0: rabi_axis},
    solver="dopri5",
    execution_mode="expanded_sparse",
    output="selected",
    output_indices=target_entries,
    output_when="final",
    dense_output=False,
    parallel=True,
)

grid_shape = scan_result.metadata["grid_shape"]
target_population = scan_result.values.reshape(*grid_shape, len(target_indices)).real.sum(axis=-1)
```

Use the effective-Hamiltonian Rust path only after constructing an effective
model. It is a reduced model path, not a drop-in replacement for a full OBE
solve.

## Solver Selection

| Path | Use for | Main entrypoints | Notes |
| --- | --- | --- | --- |
| Full OBE Rust solvers | Non-stiff full OBE trajectories and scans | `prepare_lindblad_problem`, `solve_lindblad`, `grid_scan` | Recommended default. Use `hamiltonian_representation="decomposed"` and `execution_mode="expanded_sparse"`. |
| Full OBE SciPy stiff fallback | Stiff or difficult problems where native Rust solvers struggle | `solve_lindblad(..., solver="scipy_bdf")`, `solve_lindblad(..., solver="scipy_radau")` | Uses Rust RHS/Jacobian probing where available, but usually has more Python/SciPy overhead. |
| Python/reference OBE | Correctness checks and debugging | `solve_lindblad(..., backend="python", solver="python_rk45")` | Not intended for production scan throughput. |
| Effective-Hamiltonian Lindblad | Lower-dimensional effective models | `prepare_effective_lindblad_rust_plan`, `solve_effective_lindblad`, effective scans | Requires an effective model prepared separately. Output/API details differ from full OBE. |

Current full OBE solver choices:

| Solver | Status | Typical role |
| --- | --- | --- |
| `dopri5` | Recommended Rust solver | Custom Rust Dormand-Prince 5(4). Usually the first solver to try. |
| `tsit5` | Recommended Rust alternative | Custom Rust Tsitouras 5(4). Useful for comparison and sometimes competitive for final-only outputs. |
| `scipy_rk45` | SciPy RK45 path | SciPy RK45 using Rust matrix RHS callback. |
| `scipy_bdf` | Stiff fallback | SciPy BDF using Rust packed RHS and optional exact sparse Jacobian path. |
| `scipy_radau` | Stiff fallback | SciPy Radau using Rust packed RHS and optional exact sparse Jacobian path. |
| `python_rk45` | Python/reference path | Python/reference RK45 implementation for correctness checks and debugging. |

Native Rust stiff/BDF solving is not implemented. Use `scipy_bdf` or
`scipy_radau` when a stiff fallback is needed.

In short: for full OBE examples, prefer `dopri5` or `tsit5`. For
effective-Hamiltonian examples, use `dopri5` or `tsit5`.

## Full OBE Outputs

`solve_lindblad` accepts a prepared problem or an OBE system plus parameters. A
prepared problem is preferred for repeated solves and scans because symbolic
lowering is done once.

Full OBE output modes:

| Output | Meaning | Single-solve shape |
| --- | --- | --- |
| `full` | Packed density matrix trajectory for Rust fast path, or matrix trajectory for matrix paths | `(n_times, packed_len)` for `LindbladResult`; matrix paths expose density matrices |
| `populations` | Diagonal populations only | `(n_times, n_states)` for `output_when="saveat"`; `(n_states,)` for `output_when="final"` |
| `selected` | Selected density-matrix entries | `(n_times, n_selected)` for `saveat`; `(n_selected,)` for `final` |
| `weighted_integral` | Time integral of weighted populations | Observable result array |
| `photon_integral` | Weighted integral used for photon/scattering style outputs | Observable result array |
| `excited_population` | Weighted excited-population style integral/output | Observable result array |

Selected entries use full density-matrix indices, not packed-storage indices:

```python
import numpy as np

saveat = np.linspace(0.0, 10e-6, 201)
selected = solve_lindblad(
    prepared,
    rho0,
    (0.0, 10e-6),
    solver="dopri5",
    execution_mode="expanded_sparse",
    saveat=saveat,
    output="selected",
    output_indices=[
        (0, 0),  # population rho[0, 0]
        (5, 5),  # population rho[5, 5]
        (2, 7),  # coherence rho[2, 7]
    ],
)

rho_00 = selected.values[:, 0]
rho_27 = selected.values[:, 2]
```

Integral outputs require `integral_weights`:

```python
import numpy as np

gamma_rate = 1.0  # replace with the decay/scattering rate for these states
photon_weights = [(int(idx), float(gamma_rate)) for idx in excited_indices]
t_eval = np.linspace(0.0, 100e-6, 1001)

signal = solve_lindblad(
    prepared,
    rho0,
    (0.0, 100e-6),
    solver="dopri5",
    execution_mode="expanded_sparse",
    saveat=t_eval,
    output="photon_integral",
    integral_weights=photon_weights,
    output_when="saveat",
)
```

`output_when="saveat"` records requested save points. `output_when="final"`
records only the final value. Use `dense_output=False` only when no interior
save points are needed; it is intended for final-only work and rejects interior
`saveat` points.

## Terminal Events

Full OBE solves support one terminal `stop_event`. Without `stop_event`, solver
behavior is unchanged. If the event triggers, `result.t[-1]` is the event time,
the event value is appended as the final output point even when it is not in
`saveat`, and later `saveat` points are skipped. With `output_when="final"`, the
result contains only the terminal value.

Runtime-expression events stop when the expression crosses zero:

```python
from centrex_tlf.lindblad import Time

stop_event = Time() - 250e-6

result = solve_lindblad(
    prepared,
    rho0,
    (0.0, 1e-3),
    stop_event=stop_event,
    output="populations",
    output_when="final",
)
```

Population threshold events stop when the summed population in the selected
state indices reaches the threshold:

```python
from centrex_tlf.lindblad.events import population

stop_event = population(target_indices, threshold=0.95)

scan_result = grid_scan(
    prepared,
    rho0,
    (0.0, 1e-3),
    scan={delta0: detuning_axis, omega0: rabi_axis},
    stop_event=stop_event,
    output="selected",
    output_indices=[(idx, idx) for idx in target_indices],
    output_when="final",
    dense_output=False,
)

grid_shape = scan_result.metadata["grid_shape"]
time_to_threshold_us = scan_result.t.reshape(grid_shape) * 1e6
reached_threshold = scan_result.metadata["event_triggered"].reshape(grid_shape)
```

When `collect_stats=True`, single solves include `event_triggered`,
`event_time`, `event_index`, and `event_name` in `solver_stats`. Batch and grid
solves support `stop_event` only with `output_when="final"`; using an event with
`output_when="saveat"` raises `ValueError`. For event batch/grid solves,
`result.t` contains per-trajectory terminal or final times, and metadata includes
per-trajectory `event_triggered` and `event_times` arrays.

## Batch Solving and Full OBE Scans

For independent trajectories, use `solve_lindblad_batch`, `parameter_scan`, or
`grid_scan` instead of a Python loop. These APIs keep the trajectory loop in
Rust and return one aggregated NumPy array.

```python
from centrex_tlf.lindblad import solve_lindblad_batch

batch = solve_lindblad_batch(
    prepared,
    rho0_batch,  # shape: (n_trajectories, n, n) or (n_trajectories, packed_len)
    (0.0, 200e-6),
    solver="dopri5",
    execution_mode="expanded_sparse",
    output="populations",
    output_when="final",
    dense_output=False,
    parallel=True,
)

final_populations = batch.values  # shape: (n_trajectories, n_states)
```

`parameter_scan` varies base parameter slots using an explicit trajectory table:

```python
import numpy as np

from centrex_tlf.lindblad import parameter_scan

# `omega0` and `delta0` are registered base Parameter objects.
parameter_values = np.array(
    [
        [0.5e6, -1.0e6],
        [1.0e6, 0.0],
        [2.0e6, 1.0e6],
    ],
    dtype=np.complex128,
)

scan = parameter_scan(
    prepared,
    rho0,
    (0.0, 200e-6),
    parameter_slots=[omega0, delta0],
    parameter_batch=parameter_values,
    output="populations",
    output_when="final",
    dense_output=False,
)
```

`grid_scan` varies one-dimensional axes and creates the Cartesian product in the
Rust grid path:

```python
import numpy as np

grid = grid_scan(
    prepared,
    rho0,
    (0.0, 200e-6),
    scan={
        delta0: np.linspace(-5e6, 5e6, 101),
        omega0: 2 * np.pi * np.array([0.5e6, 1.0e6, 2.0e6]),
    },
    output="selected",
    output_indices=[(0, 0), (5, 5)],
    output_when="final",
    dense_output=False,
)

values_on_grid = grid.values.reshape(*grid.metadata["grid_shape"], grid.values.shape[-1])
```

Full OBE batch/scan outputs support `populations`, `selected`,
`weighted_integral`, `photon_integral`, and `excited_population`. Final output
has shape `(n_trajectories, width)`. Save-at output has shape
`(n_trajectories, n_times, width)`.

`grid_scan` metadata includes:

- `metadata["scan_kind"] == "grid"`
- `metadata["grid_shape"]`
- `metadata["grid_axes"]`
- `metadata["compact_grid"] == True` for the compact Rust OBE grid path

Parallelism uses Rayon. With `parallel=True` and `threads=None`, Rayon uses its
global thread pool. Passing `threads=N` builds a local pool for that batch/grid
call. Use explicit `threads` only when you need to cap worker count; otherwise
prefer `threads=None` or set `RAYON_NUM_THREADS` before Python initializes Rayon.

## Runtime Parameters and Helper Expressions

Use `LindbladParameters` for named runtime parameters and symbolic bindings:

```python
import numpy as np

from centrex_tlf.lindblad import LindbladParameters, Time, gaussian, sine

# Example assumes a transition selector produced by
# couplings.generate_transition_selectors(...).
selector = selectors[0]

params = LindbladParameters()
t = Time()

omega0 = params.real("omega0", 2 * np.pi * 1e6)
delta0 = params.real("delta0", 0.0)
z0 = params.real("z0", -0.01)
vz = params.real("vz", 180.0)
sigma_z = params.real("sigma_z", 0.003)

z = z0 + vz * t
params.bind(selector.Ω, omega0 * gaussian(z, center=0.0, sigma=sigma_z), finalize=False)
params.bind(selector.δ, sine(t, offset=delta0, amplitude=0.0), finalize=False)
params._finalize()
```

Common methods:

- `params.real(name, default)` registers a real base parameter.
- `params.complex(name, default)` registers a complex base parameter.
- `params.bind(symbol, expression, finalize=False)` binds a Hamiltonian or
  polarization symbol to a scalar or runtime expression.
- `params.drive(selector, rabi=..., detuning=..., finalize=False)` binds the
  Rabi and detuning symbols for a transition selector.

Scan keys can be `Parameter` objects or legacy string names. New code should
prefer `Parameter` objects:

```python
parameter_scan(..., parameter_slots=[omega0, delta0], parameter_batch=parameter_values)
grid_scan(..., scan={delta0: detuning_axis, omega0: rabi_axis})
```

Only base parameters can be scanned directly. Compound parameters update when
their base-parameter dependencies are overridden in Rust.

The polymorphic helper functions work numerically and as `RuntimeExpression`
builders when any argument is expression-like. Current helper coverage includes:

- Gaussian/profile helpers: `gaussian_1d`, `gaussian_2d`,
  `gaussian_2d_rotated`, `gaussian`.
- Modulation/waveform helpers: `phase_modulation`, `square_wave`,
  `resonant_polarization_modulation`, `sawtooth_wave`, `variable_on_off`,
  `variable_on_off_duty`, `variable_on_off_duty_invT`, `square_wave_profile`,
  `alternating_sign`, `linear`, `sine`.
- Intensity/Rabi helpers: `multipass_2d_intensity`, `rabi_from_intensity`,
  `multipass_2d_rabi`, `gaussian_beam_rabi`.
- Interpolation helpers: `linear_interp`, `pchip_interp`, `tabulated`,
  `pchip_tabulated`.

Tuple-valued registered parameters are supported for helpers such as multipass
profiles and tabulated interpolation.

## Effective-Hamiltonian Lindblad Solver

The effective-Hamiltonian path works on prepared effective models from
`centrex_tlf.effective_hamiltonian`. It propagates a lower-dimensional density
matrix using the generic Rust ODE machinery and effective operators.

Typical preparation flow:

```python
import numpy as np

from centrex_tlf.effective_hamiltonian import (
    default_effective_density_matrix,
    prepare_effective_lindblad_rust_plan,
    prepare_lindblad_safe_compact_interpolated_model,
    solve_effective_lindblad,
)
from centrex_tlf.lindblad import LindbladParameters

model = prepare_lindblad_safe_compact_interpolated_model(...)

params = LindbladParameters()
# Bind model-specific runtime parameters here.

plan = prepare_effective_lindblad_rust_plan(
    model,
    params,
    operator_interpolation="linear",  # or "pchip"
)
rho0 = default_effective_density_matrix(model)
t_eval = np.linspace(0.0, 100e-6, 1001)

result = solve_effective_lindblad(
    plan,
    rho0,
    (0.0, 100e-6),
    saveat=t_eval,
    solver="dopri5",
    output="full",
    output_when="saveat",
)
```

Effective single-solve outputs:

| Output | Meaning |
| --- | --- |
| `full` | Effective density matrices in `result.rho`; `result.density_matrices()` returns the same data. |
| `populations` | Population array in `result.rho`. |
| `selected` | Selected effective density-matrix entries. |
| `weighted_integral`, `photon_integral`, `excited_population` | Integral-style observable arrays. |

Effective batch scans live in `centrex_tlf.effective_hamiltonian.rust_plan`.
Because the names overlap with full OBE scans, import aliases are usually
clearer:

```python
import numpy as np

from centrex_tlf.effective_hamiltonian.rust_plan import (
    grid_scan as effective_grid_scan,
    parameter_scan as effective_parameter_scan,
)

# `velocity` and `rabi_rate` are base Parameter objects in the effective plan's
# LindbladParameters.
effective_scan = effective_parameter_scan(
    plan,
    rho0,
    (0.0, 100e-6),
    parameter_slots=[velocity],
    parameter_batch=velocity_values.reshape(-1, 1),
    output="populations",
    output_when="final",
    parallel=True,
)

effective_grid = effective_grid_scan(
    plan,
    rho0,
    (0.0, 100e-6),
    scan={velocity: velocity_axis, rabi_rate: rabi_axis},
    output="populations",
    output_when="final",
)
```

Effective batch `parameter_scan` and `grid_scan` currently support
`output="populations"` and `output="full"`. Final output has shape
`(n_trajectories, width)`, and save-at output has shape
`(n_trajectories, n_times, width)`. Effective grid results include
`metadata["grid_shape"]` and `metadata["grid_axes"]`.

`operator_interpolation="linear"` and `"pchip"` are supported when preparing
the effective Rust plan. Use the interpolation mode that matches how the
effective operator grid was validated for the model.

## Performance Notes

Exact timings depend on system size, stiffness, Hamiltonian time dependence,
output mode, save-point count, scan dimensions, and hardware.

High-level guidance:

- Prepare once and reuse `PreparedLindbladProblem` or effective Rust plans.
- Prefer `expanded_sparse` for full OBE Rust solves unless debugging another
  execution mode.
- For scans that only need final objectives, use `output_when="final"` and
  `dense_output=False`.
- Prefer `output="selected"` or `output="populations"` over full trajectories
  when the full density matrix is not needed.
- Use Rust `grid_scan` for structured Cartesian scans. It avoids materializing a
  repeated initial-state batch and full Cartesian parameter table in Python.
- Keep `threads=None` unless you need to cap parallelism. If you need a process
  wide Rayon worker count, set `RAYON_NUM_THREADS` before Python starts.
- Use SciPy stiff fallbacks only when native Rust solvers are not suitable; they
  can be much slower for large scan workloads.

## Examples and Timing Scripts

Curated examples:

- `examples/lindblad/r0_f2_batch_grid_scan.ipynb`: compact Rust OBE grid scan.
- `examples/lindblad/rotational_cooling_parameter_scans.ipynb`: large
  rotational-cooling OBE scan pattern.
- `examples/lindblad/rotational_cooling_terminal_event_scans.ipynb`:
  rotational-cooling time-to-threshold scan using terminal events.
- `examples/lindblad/q1_circular_polarization_switching_scan.ipynb`:
  photon-integral scan output.
- `examples/lindblad/q1_effective_fixed_basis_vs_static_regular_rust.ipynb`:
  effective-Hamiltonian versus full static OBE comparison.

Useful timing and validation scripts:

- `benchmarks/benchmark_obe.py`
- `benchmarks/benchmark_julia_comparison.py`
- `benchmarks/bench_square_wave.py`
- `benchmarks/bench_effective_batch_vs_python.py`
- `benchmarks/bench_q1_timedep_methods.py`
- `benchmarks/validate_effective_lindblad_timedep.py`

Run benchmarks in the target environment when exact numbers matter. Do not
treat notebook output timings or historical local tables as portable
performance claims.
