# Q1 Effective Hamiltonian Status

This document summarizes the current `Q1_F1_1o2_F0` fixed-basis compact-interpolated effective Hamiltonian work and the validation notebooks/scripts in this folder.

## Goal

The goal is to propagate a compact Lindblad/OBE model in one fixed reference basis while the electric field changes. This avoids rebuilding and re-diagonalizing the compact system at every RHS call, while keeping the model close to the direct compact calculation at each field.

The current focus is the Lindblad-safe fixed-basis compact interpolation for the `Q1_F1_1o2_F0` transition.

## Current Method

The model is built from compact direct OBE systems at a set of electric-field patch points.

For each patch field:

- Build the direct compact OBE system at that field.
- Embed/reorder the compact basis into a common fixed layout.
- Align the local compact basis to the fixed reference gauge.
- Shift the optical frame into one common laser-frequency reference.
- Store Hamiltonian pieces and dissipator data for interpolation.

At runtime:

- Interpolate the internal Hamiltonian block.
- Interpolate the optical coupling block.
- Interpolate the detuning block.
- Interpolate the Lindblad-safe full recycling kernel.
- Factor the interpolated recycling kernel back into collapse operators.
- Build an `OperatorBundle` with real `c_array` collapse operators.

The important point is that the propagating model now uses a real collapse-operator representation again, not a raw dissipator-superoperator override.

## Full Recycling Kernel

Earlier versions used per-target excited-state decay kernels. Those kernels preserved target-resolved decay rates and jump-rate observables, but they were too compressed: they did not encode all coherences in the lower manifold.

The current representation stores a full recycling kernel over collapse-operator matrix entries:

```text
(target_state, source_state)
```

This kernel is PSD-projected and factored back into collapse operators. That preserves the Lindblad-safe structure and the coherence information that per-target kernels can lose.

The old per-target decay kernels are still useful as diagnostics for rates, but they are no longer the source of the propagated dissipator.

## Static Validation

Static validation is in:

- `q1_static_compact_interpolated_validation.ipynb`
- `validate_q1_recycling_kernel.py`

The static comparison checks:

- `fixed_interpolated`: fixed-basis interpolated model evaluated at a field.
- `direct_compact_aligned`: direct compact model built at the same field and aligned into the fixed layout.
- `direct_compact_local_common_frame_rust`: direct compact local model solved through the Rust OBE path, shifted into the same common optical frame.
- `local_compact_bundle` / `local_compact_rust`: solver-path diagnostic comparing SciPy bundle propagation and Rust propagation for the same direct local compact system.

Relevant static diagnostics:

- `dissipator_rel_vs_aligned_c_array`: interpolated recycling-kernel dissipator versus the aligned direct compact collapse-operator dissipator.
- `jump_rate_rel`: jump-rate operator agreement.
- `h_internal_active_rel`, `h_opt_rel`, `h_det_rel`: active Hamiltonian component agreement.
- Observable agreement: photons, integrated excited population, final excited population, trace error.

Removed misleading diagnostics:

- Plain `dissipator_rel`, because it compared against an older superoperator override rather than the collapse-operator representation.
- `liouvillian_rel`, because it was dominated by full-space/reference-layout details not predictive of the propagated observables.
- `h_internal_full_rel`, for the same reason.

Current static script result:

```text
max dissipator_rel_vs_aligned_c_array ~ 1.45e-15
max jump_rate_rel                     ~ 9.29e-16
max h_internal_active_rel             ~ 3.21e-15
```

Static dynamics agreement is also at numerical precision when comparing the fixed-interpolated model to the aligned direct compact model at patch/static fields.

## Time-Dependent Validation

Time-dependent validation is in:

- `validate_q1_time_dependent_recycling_kernel.py`
- `q1_time_dependent_recycling_kernel_validation.ipynb`
- `validate_q1_instantaneous_sink_truth.py`
- `q1_instantaneous_sink_truth_validation.ipynb`

The candidate is:

```text
fixed-basis Lindblad-safe compact interpolation
```

using the production patch grid:

```text
[0, 5, 7, 7.5, 8, 10, 20, 30, 40, 50] V/cm
```

The time-dependent validation cases currently include:

- Linear ramp from `0` to `50 V/cm` over `2 us`.
- Sinusoidal field `25 +/- 20 V/cm` over `2 us`.

## Time-Dependent Reference Tiers

There are now two time-dependent reference tiers.

### Dense Aligned Compact Liouvillian Reference

The first time-dependent reference is not a fully continuous direct compact solve that rebuilds the exact compact model inside every RHS evaluation. That would be extremely expensive.

Instead, the script builds a denser direct compact reference grid:

```text
0 to 50 V/cm, default 41 points
```

At each reference field:

- Build the direct compact OBE system at that field.
- Align/embed it into the same fixed basis layout.
- Shift it into the same common optical frame.
- Build the direct compact Liouvillian and jump-rate operator.

During propagation, the reference RHS linearly interpolates these direct aligned compact Liouvillians on the dense grid.

So the base truth is:

```text
dense-grid direct compact aligned reference
```

not:

```text
exact continuous direct compact OBE rebuilt at every solver step
```

This means the time-dependent comparison measures the candidate production patch-grid interpolation against a much denser direct compact interpolation. It is a practical convergence reference, not an analytic exact solution.

Increasing `--reference-points` makes this reference closer to the continuous direct compact model but increases setup time.

### Instantaneous-Basis Sink Reference

The more physically meaningful compact reference is:

```text
dense-grid instantaneous compact basis + gauge Hamiltonian + field-dependent sink decay
```

This reference is implemented in:

```text
validate_q1_instantaneous_sink_truth.py
q1_instantaneous_sink_truth_validation.ipynb
```

At each reference field it builds the instantaneous compact basis and stores:

- instantaneous Hamiltonian terms,
- moving-basis gauge connection,
- dissipator and jump-rate operator,
- sink channels for omitted lower `J` manifolds.

During propagation it uses:

```text
H(t) = H_inst(E(t)) + H_gauge(E(t), dE/dt)
```

where the gauge term accounts for motion of the instantaneous basis.

For this Q1 setup, the sink channels correspond to omitted lower:

```text
X, J=0
X, J=2
```

The sink populations are part of the effective density matrix as dissipative-only states. This captures field-dependent decay into those omitted manifolds without coherently propagating their GHz-scale dynamics.

This is still not a full physical-basis all-`J` OBE. It is the best practical compact truth currently implemented.

## Current Time-Dependent Results

Using:

```text
validate_q1_time_dependent_recycling_kernel.py --reference-points 41
```

the current results are approximately:

```text
linear ramp 0 -> 50 V/cm:
  photons relative error          ~ 7.8e-4
  excited integral relative error ~ 7.8e-4
  max excited trace relative err  ~ 1.6e-3
  final excited relative error    ~ 1.4e-2

sinusoid 25 +/- 20 V/cm:
  photons relative error          ~ 1.1e-3
  excited integral relative error ~ 1.1e-3
  max excited trace relative err  ~ 1.2e-2
  final excited relative error    ~ 1.3e-3
```

The integrated observables are the most reliable validation metrics. Final excited-state relative error can look larger because the final excited population is small.

Using the instantaneous-basis sink reference:

```text
validate_q1_instantaneous_sink_truth.py --reference-points 41
```

the current results are approximately:

```text
linear ramp 0 -> 50 V/cm:
  photons relative error          ~ 2.6e-3
  excited integral relative error ~ 2.6e-3
  max excited trace relative err  ~ 2.9e-3
  final excited relative error    ~ 3.8e-2
  final sink absolute error       ~ 1.1e-7

sinusoid 25 +/- 20 V/cm:
  photons relative error          ~ 1.5e-3
  excited integral relative error ~ 1.5e-3
  max excited trace relative err  ~ 4.8e-2
  final excited relative error    ~ 2.5e-2
  final sink absolute error       ~ 6.0e-8
```

An ablation run against a 21-point instantaneous-basis sink reference:

```text
validate_q1_instantaneous_sink_truth.py --reference-points 21 \
  --candidate-grids production uniform_2p5 uniform_1p25 --gauge-ablation
```

showed:

```text
linear ramp 0 -> 50 V/cm:
  production photons/excited integral rel ~ 2.93e-3
  uniform_2p5 photons/excited integral rel ~ 1.84e-3
  uniform_1p25 photons/excited integral rel ~ 2.15e-3

sinusoid 25 +/- 20 V/cm:
  production photons/excited integral rel ~ 1.37e-3
  uniform_2p5 photons/excited integral rel ~ 4.51e-4
  uniform_1p25 photons/excited integral rel ~ 2.53e-4
```

The gauge ablation on the same 21-point instantaneous reference showed:

```text
with gauge vs without gauge:
  linear ramp photons/excited integral rel ~ 9.6e-3
  sinusoid photons/excited integral rel    ~ 2.8e-2
```

So the moving-basis gauge Hamiltonian is a percent-level effect for these trajectories and is not optional for the instantaneous-basis truth model.

## Interpretation

The static validation is now very strong: the full recycling-kernel path reproduces the direct aligned compact collapse-operator representation at numerical precision.

The time-dependent errors are now likely interpolation/reference-model differences from the production patch grid and the choice of fixed versus instantaneous basis. The candidate uses 10 production patch points; the dense references use up to 41 fields. Errors at the `1e-3` level in integrated observables are plausible for this grid.

Densifying the candidate grid clearly improves the sinusoidal trajectory. The ramp error is not monotonic across the tested candidate grids, so that case is not explained by candidate patch density alone. It likely also contains fixed-basis versus instantaneous-basis and reference interpolation effects.

## Next Steps

Likely next improvements:

- Increase or adapt the production patch grid where field dependence is strongest.
- Compare time-dependent results against multiple dense reference grids, e.g. 21, 41, 81 points, to estimate reference convergence.
- Add targeted patch points along actual experimental field trajectories.
- Route the fixed-basis interpolated `OperatorBundle` into the Rust OBE path if performance becomes the bottleneck.
- Audit the older superoperator override path separately, since it can differ from the stored collapse-operator representation at some fields.
