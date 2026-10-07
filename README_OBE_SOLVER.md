# OBE and Effective Lindblad Solver Map

This document is the practical map for the current density-matrix solver stack.
It covers the full OBE/Lindblad path in `centrex_tlf.lindblad`, the Rust-side
batch and scan APIs, and the lower-dimensional effective-Hamiltonian Lindblad
path in `centrex_tlf.effective_hamiltonian`.

## Hamiltonian Construction Bounds

Default X-state diagonalization includes two additional rotational levels on
each side of the selected ground-state J range: `max(0, min(J) - 2)` through
`max(J) + 2`. These extra construction states contribute mixing without adding
levels to the retained OBE system. Explicit `Jmin_X` and `Jmax_X` independently
override the corresponding bounds. B-state construction remains `J=1` through
`max(J') + 2` by default.

Rebuild cached Hamiltonians, OBE systems and prepared problems made with the
previous unpadded X defaults. Increasing only `Jmax_X` does not restore missing
lower-J mixing. Check convergence again at large electric fields.

Explicit construction bounds must contain every requested parent J and respect
the physical lower limit (X: zero; B: one). Invalid bounds raise `ValueError`.
Excited-state matching uses one-to-one assignment, including the Omega-to-parity
path. This prevents duplicate eigenstates but does not replace adiabatic tracking
when bare labels become unreliable at strong or reoriented fields.

Microwave-only systems are supported: both driven X manifolds are retained, the
B block is empty, and `C_array` has shape `(0, n, n)` when no decay is present.
Additional optical decay channels keep their state indices aligned through
compaction and level insertion. `normalize_pol` applies to both explicit and
automatic main-pair selection.

Both OBE builders and their setup wrappers apply `H_func_X`, `H_func_B`, and
`transform`. Custom B physics is used during ground-level discovery as well as
final construction. Callbacks receive `(E, B)` and return full construction
matrices in rad/s: X in the uncoupled basis, B in the Omega basis for the OBE
builders. The lower-level `generate_reduced_hamiltonian_transitions` also supports
custom parity-basis B matrices with `use_omega_basis=False`.

Precomputed matrices and transforms should use matching explicit J bounds and
the package's generated basis ordering. Matrix dimensions, finite entries,
Hermiticity (allowing floating-point roundoff), and the square/unitary X transform
are validated; basis ordering cannot be inferred from a bare callback.
Rebuild systems that previously supplied customization to the transition builder:
those arguments were formerly ignored.

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
| `dense_eig` | Static-generator spectral propagation | CPU decomposition shared by all initial states and sample times; optional CUDA observable evaluation. |

Native Rust stiff/BDF solving is not implemented. Use `scipy_bdf` or
`scipy_radau` when a stiff fallback is needed.

## Dense Solver for Constant Fields and Drives

`solver="dense_eig"` uses the Rust prepared model's analytic packed Liouvillian,
then factors it on the CPU with SciPy. It performs **one decomposition per unique
parameter row**, solves all that row's initial states together, and evaluates
exponential factors at every requested time. The default remains `dopri5`.

The solver structurally rejects explicit Hamiltonian time dependence and time
dependence through compound parameters, including indirect dependencies. It
also rejects terminal events and sampled-quadrature integrals. Use an ODE solver
for envelopes, modulation, switching, or events. Collapse matrices in the
prepared model are constant. A singular or poorly conditioned eigenbasis raises
an error recommending an ODE solver; defective generators cannot be propagated
by this eigenbasis method.

Prepare with `backend="rust"` and `hamiltonian_representation="decomposed"`. All
existing static output modes are supported, including full packed density,
complex selected entries, weighted rates, and analytic cumulative integrals.
`abstol`, `reltol`, `dt`, and `maxiters` are ODE settings and do not control dense
solver accuracy. Factorization residual and conditioning diagnostics are
available with `collect_stats=True`.

For a grid with several independently prepared populations:

```python
from centrex_tlf.lindblad import grid_scan

# initial_states has shape (n_initial, n_states, n_states).
result = grid_scan(
    prepared,
    None,
    (0.0, 350e-6),
    rho0_batch=initial_states,
    scan={"detuning": detunings_rad_s, "rabi": rabi_values},
    solver="dense_eig",
    output="photon_integral",
    integral_weights=[(i, decay_rate) for i in excited_indices],
    output_when="saveat",
    saveat=saveat,
    parallel=True,
    threads=8,
    collect_stats=True,
)
photons = result.values[..., 0].reshape(
    len(detunings_rad_s), len(rabi_values), len(initial_states), len(saveat)
)
```

Grid ordering is parameter-point first (the existing Cartesian axis order), then
initial state. `metadata["grid_shape"]` describes the parameter axes;
`metadata["initial_condition_count"]` supplies the extra initial-state dimension.
The explicit `rho0_batch` grid option is available with `dense_eig`; pass
`rho0=None` with it. Single-state grids preserve existing result shapes.
`solve_lindblad_batch` and `initial_condition_scan` accept the existing matrix or
packed batches. Batch trajectories with identical parameter rows share a
factorization; result rows retain their input ordering.

Parallel dense scans use processes across distinct parameter rows and one BLAS
thread per process. Each worker prepares one native plan, evaluator and workspace;
parameter points replace scalar overrides and invalidate coefficient caches
without rebuilding the plan. `threads` sets the process count, capped by the number of
unique rows; otherwise the solver uses the physical-core count. All initial
states at one parameter point share one process. On Windows and other platforms
using process spawning, run parallel scripts under an
`if __name__ == "__main__":` guard. Small scans can use `parallel=False` to avoid
process startup. Eigenbasis storage scales as the square of the packed generator
dimension, and full trajectories may require much more memory than photons.
With `collect_stats=True, profile_startup=True`, `worker_startup_seconds` and `steady_state_seconds`
separate worker setup from solving; `factorizations` records parameter binding
and generator extraction, decomposition/reduction, and projection times per point.
Reported steady-state time still includes communication and output collation.
Startup profiling synchronizes worker readiness; it is opt-in because normal
scans can overlap early work with later worker startup. Without this profiling
option, parallel startup/steady-state fields are `None`; total time and per-point
stage timings are still available with `collect_stats=True`.

### Reusing Process Workers Across Calls

`DenseLindbladSession` keeps process workers and their native evaluators alive for
repeated dense scans of one prepared model. This mainly helps short scans and
fitting calls; a single large scan already amortizes process startup.

```python
from centrex_tlf.lindblad import DenseLindbladSession, grid_scan

def run_repeated_scans(prepared, rho0, detunings, rabi_values, t_end):
    results = []
    with DenseLindbladSession(prepared, threads=8) as session:
        for rabi in rabi_values:
            results.append(grid_scan(
                prepared, rho0, (0.0, t_end),
                scan={"detuning": detunings, "rabi": [rabi]},
                solver="dense_eig", dense_session=session,
                output="populations", output_when="final",
            ))
    return results

# Invoke from an if __name__ == "__main__": guard on Windows.
```

The context manager initializes all workers once and joins them on exit, including
when a scan raises. Explicit `close()` is also supported and is idempotent. A
session used without a context starts lazily on its first solve and must be closed.
The worker count defaults to physical cores and stays fixed even for small calls.
`parallel=True` is required for batch/grid APIs; omit their `threads` option or
match the session's worker count. Single `solve_lindblad` calls also accept a
session and submit their one parameter point to its pool.

Runtime overrides, initial states, output modes, sampling times and CPU/CUDA
projection may vary between calls. The prepared object and `execution_mode` must
match the session. Mutating its model payload/default parameters is rejected;
create a new session after rebuilding the model (including polarization changes).
Each call computes new decompositions; the session does not cache responses or
eigenbases. One session accepts one call at a time; overlapping calls are rejected.
Worker crashes invalidate the session. Ordinary worker exceptions propagate,
cancel queued work, and leave the session usable; already-running tasks may finish.

With `collect_stats=True`, `pool_reused` identifies an already initialized pool,
`worker_startup_seconds` measures setup inside that call (zero after context
entry), and `steady_state_seconds` includes communication and collation.
`session_startup_seconds` records the one-time setup also available as
`session.startup_seconds`. Startup outside a call is excluded from its
`total_seconds`. Session readiness is synchronized regardless of `profile_startup`;
the ordinary fresh-pool API keeps its existing scheduling defaults.

### Reusing a Factorization and Reconstructing Density Matrices

```python
from centrex_tlf.lindblad import prepare_dense_lindblad_propagator

propagator = prepare_dense_lindblad_propagator(
    prepared,
    initial_states,
    parameter_values={"rabi": rabi_value, "detuning": detuning_value},
)
counts = propagator.evaluate(
    initial_states, saveat,
    output="photon_integral",
    integral_weights=[(i, decay_rate) for i in excited_indices],
)
rho = propagator.density_matrices(initial_states, [0.0, 100e-6, 350e-6])
```

`evaluate` returns `(n_initial, n_times, output_width)`; density reconstruction
returns `(n_initial, n_times, n_states, n_states)`. Times are absolute and the
initial states apply at `t0` (default zero). Integrals start at `t0` and are
independent of the sample grid. Additional initial states and sample times reuse
the factorization. The propagator is a snapshot: changing prepared parameters
does not change it, and new power/detuning/field combinations require a new one.

Reduction uses exact graph reachability from the union of the initial states,
not a numerical coupling cutoff. Zero-feedback sink populations are reconstructed
analytically, including their initial population. Other unreachable packed
variables remain zero. A reduced propagator rejects later initial states with
support outside its retained subspace; prepare with all desired initial states,
or set `reduce=False` for arbitrary later initial coherences. Reconstruction
preserves the retained OBE model, including compact spectator states; it does
not undo state compaction.

### Optional CPU Decomposition with GPU Evaluation

Set `evolution_device="cuda"` and optionally `gpu_batch_size=8` on the dense solve
or scan. Decomposition and initial-state linear solves still run on CPU workers;
one GPU consumer receives projected coefficients and evaluates ready batches.
Time evaluation is chunked and task submission is bounded. Full packed density
reconstruction stays on the CPU; rates, integrals, populations, and selected
entries can use CUDA. `DenseLindbladPropagator.evaluate` also accepts the device
option. This is forward-only evaluation, without autograd.

PyTorch is imported only when CUDA is requested. Install the optional
`centrex-tlf[gpu]` extra with a CUDA-enabled PyTorch build appropriate for the
machine. Missing PyTorch or unavailable CUDA raises a clear error before worker
launch; CPU-only use does not require PyTorch.

The paired saved-matrix prototype measured about 15% higher throughput for the
hybrid path than eight-worker CPU-only evaluation on the Ryzen 7 9800X3D / RTX
5070 Ti system. This is a small photon-output benchmark, not a universal gain;
CPU-only remains the default. See `benchmarks/r2_f4_gpu_concurrency_results/REPORT.md`
and `benchmarks/dense_library_integration_results/REPORT.md` for scope, complete
system configuration, and library-level validation.

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

Integral outputs use `integral_method="solver"` by default. The solver
integrates the weighted population as an auxiliary state using the same
Runge-Kutta stages as the density matrix. This auxiliary state is excluded from
adaptive error control, so it does not change step selection for the physical
state. `saveat` only samples the cumulative integral; changing its spacing does
not change the terminal integral. `output_when="saveat"` returns the cumulative
integral at each requested time, and its last value agrees with a final-only
solve to solver accuracy. This option is available on `solve_lindblad`,
`solve_lindblad_batch`, and `grid_scan` (as a keyword forwarded to the batch
solver). For example, a two-point `saveat=[t0, t1]` returns the initial and
total integrated values without reducing the quadrature to one trapezoid.

Set `integral_method="sampled"` and provide `integral_saveat` to deliberately
use trapezoidal quadrature on a separate integration grid. The interval
endpoints are included automatically. `saveat` continues to control only the
times returned (`output_when="saveat"` requires explicit `saveat` values):
`output_when="saveat"` returns an array at those times, while
`output_when="final"` returns only the final value, regardless of the
integration method. Changing `saveat` does not change the sampled quadrature.
For example, keep output sparse while integrating on a finer grid:

```python
signal = solve_lindblad(
    prepared,
    rho0,
    (0.0, 100e-6),
    output="photon_integral",
    integral_weights=photon_weights,
    output_when="saveat",
    saveat=np.array([0.0, 100e-6]),
    integral_method="sampled",
    integral_saveat=np.linspace(0.0, 100e-6, 1001),
)
```

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
  `alternating_sign`, `linear`, `sine`. `variable_on_off_duty_invT` is an
  alias of `variable_on_off_duty` -- the Julia backend uses the `_invT`
  spelling, Python and Rust the shorter one, and both share a single
  `HelperFunctionId` so either lowers to the same helper.
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
