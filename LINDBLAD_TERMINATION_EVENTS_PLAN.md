# Lindblad Termination Events Remaining Work

Terminal-event v1 is implemented for single full OBE solves and final-only
batch/grid solves. The implemented API supports one terminal event through
`stop_event`, with RuntimeExpression events and population-threshold events.

This file tracks deferred work only.

## Deferred Event Features

- Support multiple events in one solve, including stable event ordering and
  result metadata for the triggering event.
- Support non-terminal event reporting without stopping integration.
- Add an explicit direction policy if use cases need rising-only or
  falling-only crossings instead of the current zero-crossing trigger.
- Consider richer event result containers if downstream analysis needs more
  than `solver_stats`, `result.t`, and batch/grid metadata.

## Deferred Integral Thresholds

Integral thresholds, such as scattered-photon count thresholds, are not part of
terminal-event v1. Current integral outputs are output bookkeeping rather than
adaptive ODE state, so accurate terminal events for accumulated quantities
should be implemented with internal auxiliary ODE variables.

Example future model:

```text
dN/dt = emission_rate(t, y)
event = N - threshold
```

This should share the existing root-finding path once the accumulated quantity
is part of the solver state.

## Deferred Batch/Grid Saveat Support

Batch and grid events currently support only `output_when="final"`, where each
trajectory can return one terminal or final time. Supporting events with
`output_when="saveat"` would require a representation for per-trajectory
terminal time grids.

Possible future approaches:

- Ragged per-trajectory outputs.
- Padded rectangular outputs plus validity masks.
- A dedicated event-aware batch result container.
