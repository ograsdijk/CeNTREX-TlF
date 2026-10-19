"""Low-cost native RHS/coefficient profiling for an already-built RC case.

No molecular model is constructed at import time.  Pass an existing
``RCCase``, one selected density matrix, and representative times (for example
switch-segment midpoints) to :func:`profile_rc_rhs`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py

DEFAULT_MODES = ("expanded_sparse", "structured")
PROFILE_FIELDS = (
    "unpack_seconds",
    "parameter_eval_seconds",
    "hamiltonian_fill_seconds",
    "commutator_seconds",
    "dissipator_seconds",
    "pack_seconds",
)


def profile_rc_rhs(
    case: Any,
    rho: np.ndarray,
    times_s: Sequence[float],
    *,
    modes: Sequence[str] = DEFAULT_MODES,
    repeats: int = 20,
) -> dict[str, Any]:
    """Profile Rust RHS and coefficient evaluation without integrating.

    The native evaluator exposes per-stage Rust timings.  This helper warms
    its workspace once, resets those counters, and evaluates the same packed
    state at each requested time ``repeats`` times.  Returned wall time also
    includes Python/FFI overhead, so use the native component timings to
    identify whether coefficient evaluation or RHS algebra dominates.

    ``times_s`` should be a small representative set spanning the desired
    coefficient behavior. For switched drives, use interior points from each
    switch state; the helper does not choose a switch schedule or propagate a
    state. ``case.prepared_dynamic`` is reused, so this performs no model or
    plan construction.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("modes must be a nonempty sequence without duplicates")
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)):
        raise ValueError("times_s must be a nonempty finite one-dimensional sequence")

    prepared = case.prepared_dynamic
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.shape != (prepared.layout.n, prepared.layout.n):
        raise ValueError(f"rho must have shape ({prepared.layout.n}, {prepared.layout.n})")
    packed = np.ascontiguousarray(prepared.layout.pack(matrix), dtype=np.float64)

    reference: list[np.ndarray] | None = None
    results: dict[str, Any] = {}
    for mode in modes:
        prepared.check_execution_mode(mode)
        evaluator = create_lindblad_rhs_evaluator_py(prepared.rust_plan, mode)
        # Initialize workspace and any static coefficient caches outside the
        # measured/profiled calls.
        evaluator.rhs_packed_py(packed, float(times[0]))
        evaluator.reset_profile_py()
        evaluator.enable_profile_py(True)

        outputs = [
            np.asarray(evaluator.rhs_packed_py(packed, float(t)), dtype=np.float64) for t in times
        ]
        # These vectors are retained for direct cross-mode equivalence checks.
        reference_outputs = outputs
        evaluator.reset_profile_py()
        started = time.perf_counter()
        for _ in range(repeats):
            for t in times:
                evaluator.rhs_packed_py(packed, float(t))
        wall_seconds = time.perf_counter() - started
        summary = dict(evaluator.profile_summary_py())
        calls = int(summary["calls"])
        evaluator.enable_profile_py(False)
        started = time.perf_counter()
        for _ in range(repeats):
            for t in times:
                evaluator.rhs_packed_py(packed, float(t))
        unprofiled_wall_seconds = time.perf_counter() - started
        row: dict[str, Any] = {
            "mode": mode,
            "calls": calls,
            "repeats": int(repeats),
            "sample_times_s": times.tolist(),
            "profiled_wall_seconds_including_python_ffi": wall_seconds,
            "unprofiled_wall_seconds_including_python_ffi": unprofiled_wall_seconds,
            "wall_seconds_per_call": unprofiled_wall_seconds / calls if calls else 0.0,
            "native_profile": summary,
            "native_component_fraction": {
                field: float(summary[field]) / float(summary["total_seconds"])
                if summary["total_seconds"] > 0.0
                else 0.0
                for field in PROFILE_FIELDS
            },
        }
        if reference is None:
            reference = reference_outputs
            row["max_abs_rhs_difference_vs_first_mode"] = 0.0
            row["max_rel_rhs_difference_vs_first_mode"] = 0.0
        else:
            differences = [
                np.abs(actual - expected)
                for actual, expected in zip(reference_outputs, reference, strict=True)
            ]
            max_abs = max(float(np.max(diff)) for diff in differences)
            ref_scale = max(float(np.max(np.abs(item))) for item in reference)
            row["max_abs_rhs_difference_vs_first_mode"] = max_abs
            row["max_rel_rhs_difference_vs_first_mode"] = max_abs / max(ref_scale, 1e-300)
        results[mode] = row

    return {
        "n_states": int(prepared.layout.n),
        "packed_len": int(prepared.layout.packed_len),
        "rho_trace_real": float(np.trace(matrix).real),
        "rho_trace_imag": float(np.trace(matrix).imag),
        "modes": results,
        "note": "native stage times are instrumented; wall time includes Python/FFI overhead",
    }


__all__ = ["DEFAULT_MODES", "profile_rc_rhs"]
