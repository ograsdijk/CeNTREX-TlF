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
    execution_mode="structured_sparse_dense_diag_soa_single_unpack",
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
    execution_mode="structured_sparse_dense_diag_soa_single_unpack",
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
| Full OBE Rust solvers | Non-stiff full OBE trajectories and scans | `prepare_lindblad_problem`, `solve_lindblad`, `grid_scan` | Recommended default. CPU ODE solves use D+S+U with `cpu_kernel="auto"`; CUDA ODE solves use `execution_mode="expanded_sparse"`. |
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
| `fixed_tsit5` | Optional batched CUDA solver | Prescribed fifth-order Tsitouras steps; explicit `dt`, no adaptive error control. |
| `fixed_ck54` | Optional batched CUDA solver | Five-stage fourth-order low-storage Carpenter-Kennedy; explicit `dt`. |
| `dense_eig` | Static-generator spectral propagation | CPU decomposition shared by all initial states and sample times; optional CUDA observable and full-density evaluation. |

Native Rust stiff/BDF solving is not implemented. Use `scipy_bdf` or
`scipy_radau` when a stiff fallback is needed.

## Sparse-Hermitian forward kernels

The sparse-Hermitian/dense-density kernels are available explicitly through
`execution_mode`. They evolve the full density matrix, including coherences;
they are arithmetic optimizations rather than physical approximations or
different ODE algorithms. CPU Rust ODE solves default to D+S+U; the other
combinations remain selectable for comparison. CUDA ODE solves continue to use
`expanded_sparse`.

| Execution mode | Optimization |
| --- | --- |
| `structured_sparse_dense` | Sparse Hermitian Hamiltonian acting on a dense working density matrix |
| `structured_sparse_dense_diag` | Direct diagonal-Hamiltonian contribution (D) |
| `structured_sparse_dense_soa` | Separate real/imaginary accumulation (S) |
| `structured_sparse_dense_single_unpack` | Unpack once per RHS evaluation (U) |
| `structured_sparse_dense_diag_soa` | D+S |
| `structured_sparse_dense_diag_single_unpack` | D+U |
| `structured_sparse_dense_soa_single_unpack` | S+U |
| `structured_sparse_dense_diag_soa_single_unpack` | D+S+U |

The full D+S+U combination is the default for CPU Rust ODE solves. Select it explicitly when you want the mode stated in code:

```python
result = solve_lindblad(
    prepared, rho0, t_span,
    solver="tsit5",
    execution_mode="structured_sparse_dense_diag_soa_single_unpack",
    output="full", output_when="final",
    reltol=1e-7, abstol=1e-9,
)
```

These names also apply to Rust batch/grid solves. Existing names beginning
`experimental_structured_sparse_dense` remain aliases to the same implementations.
The sparse-dense kernels accept both entrywise and decomposed prepared
Hamiltonians; decomposed preparation remains useful for scans. They retain the
current prepared solver's restriction to single-entry collapse operators;
general coherent collapse operators are not supported by this preparation path.
Validate tolerances for the requested
observables; the example values are not universal accuracy guarantees.

For compact static R2/R3, uncompacted RC, and the tested R2/R3 EOM controls,
D+S+U performed well. The additional general-system/default-promotion tests are
described in [the testing plan](benchmarks/LINDBLAD_KERNEL_PROMOTION_TEST_PLAN.md).
Selecting fewer final observables reduces output work; forward integration
still evolves the full density. Saving full density at many times can have
different memory and output costs from final-only solves.

## Opt-in adjoint observable responses

`prepare_lindblad_adjoint_response` computes a linear observable's response to
every initial state by integrating its adjoint backward in time. It uses a
native Rust sparse-Hamiltonian RHS and the native Dopri5/Tsit5 integrators.
Smooth time dependence is supported. Supply all known switching edges through
`discontinuities`, expressed in absolute physical time; the solver reverses the
segment schedule and preserves the drive's absolute-time phase.

```python
from centrex_tlf.lindblad import (
    population_observable,
    prepare_lindblad_adjoint_response,
)

n = prepared.layout.n
response = prepare_lindblad_adjoint_response(
    prepared, t_span,
    terminal_observables={
        "selected_group": population_observable(n, selected_indices),
    },
    integral_weights=[(i, float(GAMMA)) for i in excited_indices],
    integral_name="photons",
    solver="tsit5", reltol=1e-8, abstol=1e-10,
    # discontinuities=switch_edges_inside_t_span,
    # parameter_values={"detuning": detuning, "rabi": rabi},
)

photons_for_every_basis_state = response.basis_responses("photons")
photons_for_mixture = response.evaluate(rho0, objective="photons")
both_observables_for_initial_batch = response.evaluate(rho0_batch)
```

Terminal observables are named Hermitian matrices. `population_observable`
builds a projector for one population or a sum of selected populations; provide
separate named projectors to obtain each population independently. The photon
running integral is a separate objective, not implicitly added to a terminal
population. Each independent objective requires its own adjoint solve.

Response matrices contract with coherent initial density matrices as well as
population mixtures. Their diagonal entries alone suffice for basis-diagonal
mixtures. They are **observable-response matrices, not final density matrices**;
use forward or eligible static dense propagation when final coherences or the
full final density are required.

Responses are specific to the prepared model, parameter values, objectives,
time span, absolute entry phase and discontinuity schedule. Changing these
requires new responses. `parallel=True, threads=...` parallelizes independent
objectives within one preparation; it does not evolve initial states separately.
The native solve releases the Python GIL.

For detuning/power scans, `prepare_lindblad_adjoint_response_batch` returns an
ordered response object per parameter point and parallelizes those points:

```python
from centrex_tlf.lindblad import prepare_lindblad_adjoint_response_batch

responses = prepare_lindblad_adjoint_response_batch(
    prepared, t_span,
    parameter_values_batch=[{"detuning": float(d)} for d in detunings],
    integral_weights=[(i, float(GAMMA)) for i in excited_indices],
    integral_name="photons",
    parallel=True, threads=8,
    # discontinuities=switch_edges_inside_t_span,
)
photons_by_point_and_initial = np.stack(
    [response.basis_responses("photons") for response in responses]
)
```

Each worker solves objectives serially within its parameter point, avoiding
nested objective/parameter worker pools. `parameter_values` mappings override
base parameters only; compound expressions are reevaluated at the current
absolute physical time.

The current adjoint supports the prepared solver's structured, single-entry
collapse operators. General coherent collapse operators are rejected. Running
weights are constant weighted populations; terminal events, nonlinear/state-
dependent objectives, sampled density output and automatic discovery of
discontinuities are not supported. The benchmark prototype's speedups do not
establish the new native implementation's performance on every workload.
Existing forward defaults and output routing remain unchanged.

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

CPU full reconstruction solves the initial modal coefficients once per call,
reuses them for active evolution and analytic sink feed across time chunks, and
converts packed output to complex density matrices with bounded vectorized
batches. These optimizations are automatic; they change neither the generator
nor the density coordinates or accuracy settings. They accelerate reconstruction,
not eigendecomposition. Ordinary `LindbladResult.density_matrices()` calls use
the same vectorized conversion. See
[`benchmarks/dense_construction_reconstruction_results/REPORT.md`](benchmarks/dense_construction_reconstruction_results/REPORT.md)
for timing scope and validation.

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
reconstruction, rates, integrals, populations, and selected entries can use CUDA.
`DenseLindbladPropagator.evaluate` and `density_matrices` accept the device
option. Repeated full-density evaluations retain the eigenbasis on the GPU;
call `propagator.clear_cuda_cache()` to release those resident factors.
Initial states and sample times share one factorization and the reconstruction
uses bounded chunks of their Cartesian product. This is forward-only evaluation,
without autograd.

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

### Optional Batched CUDA ODE Integration

The single, batch, and scan APIs also support time-dependent CUDA ODEs:
`evolution_device="cuda", solver="fixed_tsit5"` or `"fixed_ck54"`.
Install `centrex-tlf[cuda]` for CUDA 12 or `centrex-tlf[cuda13]` for CUDA 13.
CuPy and a compatible NVIDIA driver are required; the existing static dense
CUDA path uses the separate PyTorch `gpu` extra. CPU-only imports require neither.
CuPy also needs matching CUDA runtime libraries, NVRTC and toolkit headers for
compiling the custom kernels. If these are not installed system-wide, install
CuPy's toolkit extra, for example `pip install "cupy-cuda13x[ctk]>=14"` for CUDA 13.
On Windows, when both are installed, the solver loads PyTorch before CuPy to
resolve bundled CUDA DLLs consistently. If importing these packages yourself,
use that same order before initializing CUDA.

```python
from centrex_tlf import lindblad

# prepared uses backend="rust", hamiltonian_representation="decomposed".
# Initial states are independent density matrices, not coherent superpositions.
# Prepack once for reuse across subsequent batch/grid calls.
initial_packed = prepared.layout.pack_batch(initial_states)
result = lindblad.grid_scan(
    prepared, None, (0.0, t_end),
    rho0_batch=initial_packed,
    scan={"detuning": detunings_rad_s},
    solver="fixed_tsit5", evolution_device="cuda",
    dt=validated_step_seconds, maxiters=1_000_000,
    output="populations", output_when="saveat", saveat=population_times,
    cuda_options=lindblad.CudaSolverOptions(
        batch_size=512, density_saveat=[t_end],
    ),
    collect_stats=True,
)
# Public packed Hermitian layout, shape (trajectories, snapshots, n_states**2).
exit_density = result.metadata["density_values"][:, 0]
# Unpack any snapshot into its full complex density matrix, including coherences.
rho_exit = prepared.layout.unpack(exit_density[0])
```

`pack_batch` accepts `(..., n_states, n_states)` matrices and returns an owned
real array with shape `(..., n_states**2)`, preserving complex upper-triangle
entries. It is a snapshot: regenerate it when the initial states change. Matrix
batch packing is vectorized automatically, and CUDA grids pack distinct initial
matrices once before expanding them over parameter points. Explicit prepacking
also avoids conversion work across separate solver calls. The same packed batch
input works with the CPU batch and dense solvers. Like scalar `layout.pack`,
`pack_batch` reads the upper triangle; callers must supply Hermitian densities.
CUDA validates matrix inputs before its automatic packing.

Both methods use synchronized, prescribed FP64 steps. **`abstol` and `reltol`
do not control their accuracy.** Validate `dt` against a stricter reference or
step refinement, checking complex density entries as well as populations.
A step appropriate for RC need not be appropriate for R(2). Tsit5 is fifth order;
CK54 is the five-stage fourth-order Carpenter-Kennedy low-storage method.
There is no automatic step-size or error controller. CPU defaults remain unchanged.

The implementation uses separate real/imaginary packed density arrays, an exact
sparse Hamiltonian commutator and structured decay, tiled products, CUDA graphs,
and optional precomputation of common drive waveforms. Hamiltonians are shared
across trajectories when only diagonal parameters differ. All density coordinates
are evolved by default even with photon-only output; reducing saved output saves
transfer and storage. Exact coordinate reduction is separately opt-in below.

All generic CUDA commutator paths (direct RHS, shared-component RHS, ordinary
sparse product and tiled product) use explicit FP64 FMA accumulation by default.
This changes floating-point rounding/association while preserving the equations,
matrix support and timestep; it does not enable fast math or reduce precision.
Both Tsit5 and CK54 use these kernels. See the
[generic FMA benchmark](benchmarks/cuda_generic_fma_results/REPORT.md) for
workload-specific measurements and the
[integration checks](benchmarks/CUDA_GENERIC_FMA_INTEGRATION.md) for the package defaults.

`CudaSolverOptions.batch_size` bounds simultaneous trajectories; scans larger
than this are chunked. Grid ordering is parameter point, then initial state.
Supported outputs include full packed density, populations, selected complex
entries, weighted/photon rates and integrals, and `full_and_integral`.
Integrals use the same Runge-Kutta stages. `density_saveat` saves additional full
density snapshots in result metadata independently of the main output grid.
Single-solve results also expose this metadata. Returned arrays are on the CPU.

Affine square waves, on/off gates, sawtooth resets and alternating-sign switches
are clipped automatically, including scanned frequencies/phases. Discontinuous
helpers use the interior value of each clipped step at all stages. Additional
jumps can be declared with `discontinuities=[...]`; nonlinear switching requires
an explicit list. `step_schedule=[(absolute_time, step_seconds), ...]` prescribes
piecewise steps for weak-drive wings or other known regimes. Requested sample
times also clip steps; no interpolation across discontinuities is performed.

This path requires a decomposed prepared Hamiltonian and
`execution_mode="expanded_sparse"`. It supports the built-in scalar expression
helpers, constant tuple multipass profiles, and constant real linear/PCHIP
interpolation tables. Unsupported expressions fail explicitly. Terminal events
and sampled trapezoidal quadrature are not supported. `parallel`/`threads` do
not create GPU workers; `cuda_options.batch_size` controls concurrency.
Compilation/graph setup costs matter for short or small batches. A GPU speedup
depends on the model, batch size, step and requested output.

Library validation against saved RC and R(2) benchmarks is recorded in
`benchmarks/CUDA_LIBRARY_INTEGRATION.md`.

### Automatic CUDA coefficient and Hamiltonian sharing

The fixed-step backend can separate shared time dependence from static
per-trajectory amplitude factors. For example, a common-height Gaussian
multipass profile can share its on-axis waveform across a height/power scan.
Static factors are evaluated once per chunk, rather than at every RK stage.
This preserves the existing pass order, losses, polarization and Doppler model.

`CudaSolverOptions(coefficient_sharing="auto", hamiltonian_sharing="auto")`
is the default and enables dependence-based selection. Set either option to `"off"` independently;
set both to `"off"` to use the previous coefficient/Hamiltonian calculation:

```python
cuda_options = lindblad.CudaSolverOptions(
    coefficient_sharing="off",
    hamiltonian_sharing="off",
)
```

Coefficient sharing works with both RHS choices. The additional Hamiltonian
component path uses the direct RHS: it shares one or several off-diagonal
matrices with finite real trajectory factors, including independently scaled
optical and microwave drives. Components may have different waveforms and
overlapping matrix support. A nonseparable or complex-factor remainder can retain
per-trajectory storage; the complete existing path is used when sharing is
unsupported or its estimated overhead/storage is unfavorable. Diagonal terms,
decay and photon accumulation are not scaled. The existing density-reduction
option still selects the direct RHS as documented below; sharing itself does
not change the requested RHS.

Auto examines expressions and the actual chunk parameters, without recognizing
particular systems or approximating factors from sampled values. It currently
considers up to four shared components and uses conservative storage estimates;
it also declines multiple-component scaling when added RHS arithmetic is large
relative to saved assembly. It does not run a timing autotuner.
`hamiltonian_sharing="on"` bypasses that arithmetic performance gate for explicit
benchmarking on other systems/devices; structural, real-factor and storage
eligibility checks still apply. Multiple components were slower for the tested
RC fixture on the RTX 5070 Ti, so auto retains normal H storage there.
Distinct temporal profiles, for example
different forward velocities, cannot automatically share one waveform. Normal
expression evaluation remains available for those cases.
Complex scalar parameter overrides conservatively use the previous coefficient
and H paths, preserving conjugation and real-helper argument validation.

`precompute_bytes` bounds the complete packed time table, including cached
subexpressions. `precompute=False` prevents time-table allocation; static-factor
caching and supported H sharing can still apply. Additional static coefficient
caches are bounded to 64 MB per chunk. Insufficient time-table budget falls back
to live evaluation. Sharing does not alter `dt`, step schedules or the need to
validate complex densities against a stricter reference.

With `collect_stats=True`, chunk statistics include `coefficient_sharing`,
`static_subexpressions`, `shared_subexpressions`, `waveform_table_bytes`,
`hamiltonian_sharing`, `shared_hamiltonian_components`,
`per_trajectory_hamiltonian_edges` and `hamiltonian_bytes`. The component count
describes the additional component kernel; an already-shared complete H is still
identified by `shared_offdiagonal_hamiltonian=True` and `already_shared`.

Integration measurements and limitations are recorded in
`benchmarks/spb_cu1_gpu_results/integrated_sharing/REPORT.md`.

### Specialized FP64 CUDA Tsit5 kernels

`CudaSolverOptions(rhs_specialization="auto")` now selects the benchmarked
prepared-sparsity FMA kernel for eligible direct Tsit5 chunks with one shared off-diagonal H
component and a real trajectory amplitude, no per-trajectory remainder, no photon-integral
state, and a provably stage-invariant diagonal. It fuses the next RK-stage
update and reuses the diagonal after the first fresh stage while retaining
Tsit5 FSAL/reset handling. States, coefficients, decay and stages remain FP64.
FMA accumulation changes rounding/association, not the equations or timestep.

Auto conservatively requires at least 512 trajectories and 64 grid steps.
Generated code is bounded to 32 levels, 528 density pairs, 256 Hamiltonian
edges, 6,000 complex terms and 1 MB of source. Other configurations retain
the generic kernel. CuPy caches generated source binaries using its normal
cache; the first eligible solve can require a substantial JIT compilation.
`rhs_specialization="off"` selects the generic implementation, which also uses
FP64 FMA. `"on"` bypasses the batch/step heuristic, raises for unsupported
structure/code size, and propagates compile errors. Auto falls back on CUDA
source compilation errors and records the reason in chunk statistics.

The specialized path uses 128 threads for full chunks and 64 for partial
chunks by default, based on cleanup-1 measurements at 1,024 and 548 rows.
These are launch heuristics, not universal optimal sizes. Set
`specialized_block_size=64/128/256/512` to override this choice. The generic
path continues to use `block_size`.

With `collect_stats=True`, chunk statistics include `rhs_specialization`
(selected kernel, eligibility/fallback reason, source SHA-256, source bytes,
complex term count, compilation/setup time and CUDA kernel attributes),
`diagonal_stage_reuse`, and the actual `block_size`.

The promoted improvements are the specialized FP64 FMA commutator/decay,
stage fusion, invariant-diagonal reuse and measured launch settings. Density
packing, Hamiltonian-envelope factoring, polarization-specific sparsity,
precision and numerical steps are unchanged. CPU kernels are unaffected.
The prior prototype passed full-flight/dark-tail density checks within
3.6e-15 and saved about 30% additional runtime against the already specialized
non-FMA prototype on cleanup-1. These are workload-specific benchmark results,
not a new measurement of this package integration. See
[the FMA report](benchmarks/cuda_fma_results/REPORT.md),
[block tuning and hardware counters](benchmarks/cuda_fma_blocks_results/REPORT.md),
and [integration notes](benchmarks/CUDA_SPECIALIZED_INTEGRATION.md).

### Opt-in exact CUDA density-coordinate reduction

Set `CudaSolverOptions(density_reduction=True)` in the usual single, batch,
or scan APIs. The default is `False`. This computes the exact union of reachable
populations/coherences from each chunk's initial states and every Hamiltonian
basis term, including all polarization configurations, plus directed decay
transfers. It uses no small-entry cutoff. Real/imaginary coherence pairs and
all ODE stage buffers are compacted; omitted entries reconstruct as zero.

This option selects the structured direct Hamiltonian/decay kernel even if
`rhs="tiled"` was supplied. It supports the same outputs, photon integrals,
additional density snapshots, switches, and fixed-step solvers. A new initial
state automatically recomputes the support. With `collect_stats=True`, inspect
`solver_stats["chunks"]` for `retained_density_pairs` and `full_density_pairs`.
Speed depends on the removed coordinates: RC improved about 12% in the tested
large batches; R(2) became slightly slower. No solver defaults change.

### Opt-in periodic CUDA cycle maps

For a strictly periodic generator, prepare its complete one-cycle density map
and reuse it for initial mixtures and durations (including forward velocities
that change only interaction time):

```python
cycle = lindblad.prepare_periodic_lindblad_propagator(
    prepared, period=5e-6, rho0_batch=initial_states,
    parameter_values={"detuning": 0.0},  # use this model's actual parameter names
    dt=validated_dt, solver="fixed_tsit5",
    cuda_options=lindblad.CudaSolverOptions(
        batch_size=1024, density_reduction=True,
    ),
)
times = field_length / forward_velocities
packed = cycle.evaluate(initial_states, times)  # (initial, time, n*n)
matrices = cycle.density_matrices(initial_states, times)
cycle.clear_cuda_cache()  # release GPU maps; a CPU copy permits later reuse
```

The declared period must cover all drives: e.g. 1 MHz switching and a 1.6 MHz
EOM share a 5 us period. The model must actually have constant envelopes for
that example. A conservative scalar-graph check rejects Gaussian time
envelopes, ramps, and noncommensurate drives. Very broad Gaussians do not count
as a proof of periodicity. Supported compositions of periodic helpers are
accepted; unproven expressions require an ordinary ODE solve.

`phase_origin` anchors the built cycle (default 0 s). Evaluation times are
absolute; `t0` defaults to that origin. Use `t0=` to change the entry phase;
the method integrates the prefix to a cycle boundary and any final fractional
cycle directly on CUDA. Complete cycles use cached powers of the map.
Unordered/repeated sample times are supported, as needed for a velocity scan.
Outputs are `full`, `populations`, or `selected`; photon integrals are not
implemented for the cycle-map API. ODE tolerances do not control map accuracy;
validate `dt` against the full complex density and intended duration.

The generator and parameter values are fixed in the propagator. Changed
detuning (including transverse Doppler shifts), power, or spatial envelopes
need a new map. Forward velocities that change Gaussian/multipass profiles in
time cannot directly share it. Mutations to the prepared model are rejected.
Supplying initial states enables exact invariant-support reduction; later
initial support outside it is rejected. Set `reduce=False` or omit the initial
batch to retain arbitrary initial coherences. This is distinct from physical
level compaction performed by the OBE builder.

`max_map_bytes` (default 1 GB) bounds retained GPU map/power storage, with ODE
and multiplication temporaries additional; a CPU snapshot of the map is also
kept. `density_saveat` and `step_schedule` are unsupported for this API; use
evaluation times and fixed `dt`. Explicit discontinuities are absolute times
within the reference cycle. Preparation can be costly, so compare construction
plus reuse against the actual number of initial states/durations per parameter
point. Existing ODE solver defaults and automatic selection remain unchanged.


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

For full-state batching, use `output="full"`. Values are packed real
Hermitian density states in `PreparedLindbladProblem.layout` order, with shape
`(n_trajectories, packed_len)` for `output_when="final"` and
`(n_trajectories, n_times, packed_len)` for `output_when="saveat"`. This lets
you carry coherent density matrices between piecewise-constant segments; use
`prepared.layout.unpack` to reconstruct an individual matrix.

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

### Packed D+S+U CPU kernels

`prepare_lindblad_problem(..., cpu_kernel="auto")` selects an optimized CPU
kernel for packed Rust D+S+U execution. Use the public execution-mode alias:

```python
prepared = prepare_lindblad_problem(system, params, cpu_kernel="auto")
print(prepared.cpu_kernel, prepared.resolved_cpu_kernel)
result = solve_lindblad(
    prepared, rho0, t_span,
    execution_mode="structured_sparse_dense_diag_soa_single_unpack",
    solver="tsit5",
)
```

| `cpu_kernel` | Selected work |
|---|---|
| `"auto"` (default) | AVX2 + FMA if supported, AVX2 otherwise, then the original kernel on other CPUs |
| `"original"` | Original D+S+U implementation, for opt-out and comparisons |
| `"avx2"` | AVX2 commutator and vectorized finalization |
| `"avx2_fma"` | AVX2/FMA commutator and vectorized finalization |
| `"avx512"` | AVX-512 commutator and AVX2 finalization |
| `"avx512_fma"` | AVX-512/FMA commutator and AVX2 finalization |

Every optimized choice also fuses structured decay and writes directly into the
solver's packed output. All density coordinates and complex coherences remain
present. FMA changes floating-point rounding slightly. Auto deliberately never
selects AVX-512: wider vectors were beneficial for RC but slower for smaller
systems on the benchmark CPU. Support alone does not guarantee a speedup on a
different CPU; the original and explicit choices remain available.

CPU/OS features are checked once when preparing the Rust plan, with no startup
timing calibration. Unsupported explicit choices raise `ValueError`. The current
AVX-512 implementation requires AVX512F, AVX2 and FMA. Compiling the extension
requires Rust 1.89 or newer; the binary does not require those ISA features on
every machine because intrinsics are isolated behind runtime dispatch.

Inspect `prepared.resolved_cpu_kernel` for the selected D+S+U kernel; Python-only
preparation reports `None`. The requested setting is serialized into single,
batch and grid payloads, and resolved on the receiving machine. Reprepare to
change it. Other execution modes, including `expanded_sparse`, matrix/split RHS
evaluation, Python and CUDA retain their existing computations.
Thus setting `cpu_kernel` does not switch the execution mode automatically.

In RHS profiling, the fused CPU pass is included in `commutator_seconds`;
`dissipator_seconds` and `pack_seconds` have no separate passes for this path.

Benchmarks and integration checks:
`benchmarks/rust_specialized_rhs_results/extended_simd_dsu/REPORT.md` and
`benchmarks/rust_specialized_rhs_results/cpu_kernel_integration/REPORT.md`.

Exact timings depend on system size, stiffness, Hamiltonian time dependence,
output mode, save-point count, scan dimensions, and hardware.

High-level guidance:

- Prepare once and reuse `PreparedLindbladProblem` or effective Rust plans.
- Decomposed Rust Hamiltonians cache their static upper triangle and numeric
  update offsets at preparation. Upper-triangle kernels, including D+S+U,
  reuse these updates on each RHS call. A shared scalar coefficient DAG also
  reuses identical expression subtrees; expressions without sharing or outside
  its supported scalar form retain the existing evaluator. These are generic
  prepared-data optimizations: changing the molecular system requires preparing
  new data, not compiling a new kernel. Runtime power, detuning and waveform
  parameters remain variable. See
  [integration validation](benchmarks/rust_specialized_rhs_results/PREPARED_HAMILTONIAN_INTEGRATION.md).
  Both optimizations default to enabled and can be disabled independently at
  preparation, without changing the Hamiltonian representation or execution mode:

  ```python
  prepared = prepare_lindblad_problem(
      system, params, backend="rust", hamiltonian_representation="decomposed",
      prepared_hamiltonian_assembly=False,
      shared_coefficient_dag=False,
  )
  ```

  Setting both to `False` uses the original upper-triangle assembly and
  coefficient evaluation paths. The choices are preserved in prepared payloads
  and apply to batch/grid solves; prepare a new problem to change them.
- Use the CPU default D+S+U mode as the general full-OBE Rust starting point,
  then benchmark the prepared model's available execution modes. For the tested
  70-state RC model with optical EOM and switched microwaves in short
  1–10 us pilots, `tsit5` with
  `execution_mode="structured"`, `reltol=1e-9`, and `abstol=1e-11` passed the
  full-density and photon checks and was about 3.2–3.4x faster than
  `expanded_sparse`. The intended RC interaction is at least 200 us; its
  timing calibration is not yet reference-validated, so these short results
  do not establish the actual-duration recommendation. This is case-specific
  evidence, not a library-wide default; a historical short constant-coefficient R2 comparison favored
  `expanded_sparse` (11.88 ms versus 16.24 ms) under a different 154-state
  setup. That old measurement has not been rerun and is not a production scan
  comparison. See the [historical R2 report](benchmarks/hamiltonian_representation_results/report.md)
  and the RC report for their respective scope and limits. See
  [`benchmarks/rc_propagation_results/README.md`](benchmarks/rc_propagation_results/README.md)
  for the benchmark settings, limits, and artifacts.
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
