# ODE Helper Function Comparison

Compared sources:

- Julia extension: `../CeNTREX-TlF-julia-extension/centrex_tlf_julia_extension/lindblad_julia/julia_common.jl`
- Current Python helper registry: `centrex_tlf/lindblad/helper_functions.py`
- Current Rust runtime evaluator: `rust/src/lindblad/eval.rs`
- Current high-level expression constructors: `centrex_tlf/lindblad/parameters.py`

## Summary

For helper functions such as Gaussian profiles, waveforms, Rabi conversion, and multipass utilities, this repo has low-level parity with the Julia extension and in several cases has more functionality. The functions from `julia_common.jl` are implemented in `centrex_tlf/lindblad/helper_functions.py` and are also handled by the Rust runtime evaluator used by ODE solves.

What is missing here is mostly the high-level `RuntimeExpression` convenience layer. `parameters.py` exposes ergonomic constructors for only a subset: `linear`, `sine`, `gaussian`, `square_wave_profile`, `tabulated`, and `pchip_tabulated`. Most Julia-style ODE helper functions can be used by name inside parsed expressions, but do not yet have first-class Python builder functions.

## Julia Helper Coverage

| Julia helper | Current low-level Python helper | Current Rust ODE evaluator | High-level `parameters.py` constructor | Notes |
|---|---:|---:|---:|---|
| `gaussian_2d` | yes | yes | missing | Low-level support exists; no `RuntimeExpression` wrapper. |
| `gaussian_2d_rotated` | yes | yes | missing | Low-level support exists; no builder wrapper. |
| `phase_modulation` | yes | yes | missing | Used by legacy generated parameters by string expression, but no builder wrapper. |
| `square_wave` | yes | yes | partial | `square_wave_profile` wraps duty-cycle `variable_on_off_duty`, not exactly Julia `square_wave`. |
| `resonant_polarization_modulation` | yes | yes | missing | Low-level support exists; no builder wrapper. |
| `sawtooth_wave` | yes | yes | missing | Low-level support exists; no builder wrapper. |
| `variable_on_off` | yes | yes | missing | Low-level support exists; no builder wrapper. |
| `variable_on_off_duty_invT` | yes | yes | partial | Present as `variable_on_off_duty` and alias `variable_on_off_duty_invT`; `square_wave_profile` is the only public builder. |
| `multipass_2d_intensity` | yes | yes | missing | Tuple-valued parameters are supported, but no builder wrapper. |
| `rabi_from_intensity` | yes | yes | missing | Low-level support exists; no builder wrapper. |
| `multipass_2d_rabi` | yes | yes | missing | Low-level support exists; no builder wrapper. |
| `gaussian_beam_rabi` | yes | yes | missing | Low-level support exists; no builder wrapper. |
| `alternating_sign` | yes | yes | missing | Low-level support exists; no builder wrapper. |

## Extra Helpers Present Here

These are implemented in this repo but are not present in the Julia extension's `julia_common.jl` helper list:

- `gaussian_1d`
- `linear_interp`
- `pchip_interp`
- High-level `linear`
- High-level `sine`
- High-level 1D `gaussian`
- High-level `tabulated`
- High-level `pchip_tabulated`

## Missing Items To Add For Ergonomic Parity

If the goal is API parity for writing ODE parameter expressions in Python without string expressions, add high-level constructors in `centrex_tlf/lindblad/parameters.py` for:

- `gaussian_2d_profile(...)`
- `gaussian_2d_rotated_profile(...)`
- `phase_modulation_profile(...)`
- `resonant_polarization_modulation_profile(...)`
- `sawtooth_wave_profile(...)`
- `variable_on_off_profile(...)`
- `variable_on_off_duty_profile(...)`
- `multipass_2d_intensity_profile(...)`
- `rabi_from_intensity_profile(...)`
- `multipass_2d_rabi_profile(...)`
- `gaussian_beam_rabi_profile(...)`
- `alternating_sign_profile(...)`

These should be thin wrappers around the existing `_call_function(...)` mechanism, matching the helper names registered in `HELPER_FUNCTIONS`.

## Potential Compatibility Notes

- `square_wave` behavior should be verified numerically against Julia `Waveforms.squarewave`; this repo implements it using the sign of `sin(omega*t + phase)`.
- `sawtooth_wave` should also be verified against Julia `Waveforms.sawtoothwave`; endpoint conventions can differ between waveform libraries.
- `variable_on_off_duty_invT` is represented by the same helper ID as `variable_on_off_duty`, which is fine functionally but should be documented if public API parity matters.
- The Julia helper `alternating_sign` returns an `Int`; this repo returns float/real-valued `+1.0` or `-1.0`, which is generally compatible in ODE expressions.
