# Lindblad Termination Events Plan

## Goal

Add terminal event support to the Lindblad solvers without changing current default solver behavior. A terminal event stops integration before `t_span[1]` when a user-defined condition is met, and the returned result includes the termination time as the final solver time.

## Event Types

### RuntimeExpression events

These depend on time and solver parameters, using the same RuntimeExpression machinery as Lindblad parameters.

Example:

```python
t = Time()
z = z0 + vz * t
stop_event = z - z_stop

result = solve_lindblad(..., stop_event=stop_event)
```

The solver stops when the event expression crosses zero.

### Population threshold events

These depend on the current packed density matrix state.

Examples:

```python
stop_event = population(3) - 0.1
stop_event = population([3, 4, 5]) - 0.5
```

The Rust solvers can evaluate these directly from the current or interpolated state because populations are stored in the diagonal entries of the packed state.

### Deferred: integral threshold events

Do not implement scattered-photon or other integral thresholds in the first version.

Although current output code supports weighted integrals, that integration is output bookkeeping rather than part of the adaptive ODE state. Accurate terminal events for accumulated quantities should be implemented later by adding internal auxiliary ODE variables, such as:

```text
dN/dt = emission_rate(t, y)
event = N - threshold
```

## User API

Start with one terminal event:

```python
solve_lindblad(..., stop_event=event)
```

Possible event helpers:

```python
from centrex_tlf.lindblad.events import population

stop_event = population(3, threshold=0.1)
stop_event = population([3, 4, 5], threshold=0.5)
```

For RuntimeExpression events, accept the expression directly.

Multiple events and non-terminal events can be added later if needed.

## Output Semantics

If an event triggers:

- `result.t[-1]` is the event time.
- The event time is appended as the final returned point even if it is not in `saveat`.
- `saveat` values after the event are skipped.
- `output_when="final"` returns exactly one time: the event time.
- `output="full"` returns the state at the event time.
- `output="populations"` and `output="selected"` return values evaluated at the event time.

If no event triggers, behavior remains unchanged.

## Solver Stats

When `collect_stats=True`, add event metadata:

```python
result.solver_stats["event_triggered"] = True
result.solver_stats["event_time"] = event_time
result.solver_stats["event_index"] = 0
result.solver_stats["event_name"] = event_name
```

If no event triggers:

```python
result.solver_stats["event_triggered"] = False
```

## Native Rust Solver Design

Add an event specification to the shared ODE options or to an adjacent event context.

For each accepted RK step:

1. Evaluate `g(t_old, y_old)`.
2. Evaluate `g(t_new, y_new)`.
3. If the sign crosses zero, locate the root inside the accepted step.
4. Use the solver dense interpolation to compute `y(t_event)`.
5. Push `t_event, y_event` to the configured output.
6. Terminate normally.

This applies to both native adaptive solvers:

- `dopri5`
- `tsit5`

Population events use interpolated `y(t)`.

RuntimeExpression events only need `t`, but should still use the same accepted-step root-finding path for consistent behavior.

## SciPy and Python Paths

For the `solve_ivp`-based paths:

- `backend="python", solver="python_rk45"`
- `backend="rust", solver="scipy_rk45"`
- `backend="rust", solver="scipy_bdf"`
- `backend="rust", solver="scipy_radau"`

Wrap the same event spec as a SciPy event function:

```python
def event(t, y):
    return evaluate_event(event_spec, t, y)

event.terminal = True
```

This keeps terminal event behavior aligned across solver backends.

## Batch and Grid Solves

Do not include batch/grid event support in the first implementation unless needed immediately.

Reason: current batch/grid outputs assume a shared time grid and rectangular output arrays. Per-trajectory events can produce different terminal times, which requires either ragged outputs or a restriction to `output_when="final"`.

Potential later scope:

- Allow common RuntimeExpression events for all trajectories.
- Allow per-trajectory events only with `output_when="final"`.
- Add ragged result containers for `output_when="saveat"` if needed.

## Validation Plan

Add tests for:

- RuntimeExpression event `z0 + vz * Time() - z_stop`.
- RuntimeExpression event with a nontrivial helper expression.
- Population threshold for one state.
- Population threshold for a set of states.
- `dopri5` terminal event behavior.
- `tsit5` terminal event behavior.
- SciPy/Python event behavior through `solve_ivp`.
- `output_when="final"` returning the event time and event state.
- `saveat` values after the event being skipped.
- No-event behavior staying unchanged.

## Initial Scope Recommendation

First implementation:

1. Single-trajectory `solve_lindblad`.
2. One terminal event.
3. RuntimeExpression events.
4. Population threshold events.
5. Native Rust plus SciPy/Python parity.

Later:

1. Multiple events.
2. Non-terminal event reporting.
3. Integral/scattered-photon thresholds through auxiliary ODE variables.
4. Batch/grid events.
