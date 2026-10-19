"""Reproducible RC Lindblad propagation benchmark driver.

The driver builds the current P(2), F1'=3/2, F'=1 optical system together
with X-state J=1<->2 and 2<->3 microwave drives.  `setup` and `smoke` are
deliberately small gates; `baseline` is an explicit request for the longer
solver comparison.  No timed workloads run at import time.

Switches are square waves at 1 MHz.  The ODE reference can either retain the
runtime square waves (useful for comparison) or freeze both switch values on
each half-cycle and carry the complete density matrix across each boundary.
The latter avoids evaluating an ambiguous square-wave endpoint in an RK
stage.  Optical phase modulation always uses absolute/global time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.lindblad.batch import solve_lindblad_batch
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import PreparedLindbladProblem, prepare_lindblad_problem

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "rc_propagation_results"
GAMMA = float(hamiltonian.Γ)
TWOPI = 2.0 * np.pi
E_FIELD_V_CM = 200.0
B_FIELD_G = 1.0e-2
F_EOM_HZ = 1.6e6
F_SWITCH_HZ = 1.0e6
BETA_EOM = 3.8
SWITCH_OFFSET_S = 250e-9
SWITCH_PHASE_OFFSET_RAD = TWOPI * F_SWITCH_HZ * SWITCH_OFFSET_S


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    optical_rabi_over_gamma: float
    microwave_rabi_over_gamma: float
    detuning_mhz: float
    duration_s: float
    entry_phase_s: float = 0.0

    @property
    def optical_rabi_rad_s(self) -> float:
        return self.optical_rabi_over_gamma * GAMMA

    @property
    def microwave_rabi_rad_s(self) -> float:
        return self.microwave_rabi_over_gamma * GAMMA

    @property
    def detuning_rad_s(self) -> float:
        return TWOPI * self.detuning_mhz * 1e6


CONFIGS: dict[str, BenchmarkConfig] = {
    # These are explicit numerical test points, not apparatus predictions.
    "weak": BenchmarkConfig("weak", 0.1, 0.1, 0.0, 2e-6),
    "nominal": BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 10e-6, 125e-9),
    "strong": BenchmarkConfig("strong", 4.0, 2.0, 2.0, 10e-6, 250e-9),
}


@dataclass
class RCCase:
    config: BenchmarkConfig
    amplitude_envelope: str
    system: Any
    selectors: list[Any]
    params_dynamic: LindbladParameters
    prepared_dynamic: PreparedLindbladProblem
    segmented_parameters: LindbladParameters | None = None
    prepared_segmented: PreparedLindbladProblem | None = None
    switch_override_slots: tuple[str, ...] = ()
    switch_parameters: dict[tuple[int, int], LindbladParameters] = field(default_factory=dict)
    prepared_switch_states: dict[tuple[int, int], PreparedLindbladProblem] = field(
        default_factory=dict
    )
    static_parameters: dict[tuple[int, int], LindbladParameters] = field(default_factory=dict)
    prepared_static_switch_states: dict[tuple[int, int], PreparedLindbladProblem] = field(
        default_factory=dict
    )
    state_labels: dict[str, int] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.system.QN)

    def rho0(self, label: str) -> np.ndarray:
        if label not in self.state_labels:
            raise KeyError(f"unknown RC initial state {label!r}; choices={list(self.state_labels)}")
        rho = np.zeros((self.n_states, self.n_states), dtype=np.complex128)
        rho[self.state_labels[label], self.state_labels[label]] = 1.0
        return rho


def _make_model(config: BenchmarkConfig) -> tuple[Any, list[Any]]:
    p2 = transitions.OpticalTransition(transitions.OpticalTransitionType.P, 2, 3 / 2, 1)
    j12 = transitions.MicrowaveTransition(1, 2)
    j23 = transitions.MicrowaveTransition(2, 3)
    # Retain two independent polarizations for each microwave transition.
    mw12_a = (couplings.polarization_X - couplings.polarization_Z).normalize()
    mw23_a = (couplings.polarization_X + couplings.polarization_Z).normalize()
    selectors = couplings.generate_transition_selectors(
        [p2, j12, j23],
        [
            [couplings.polarization_X],
            [mw12_a, couplings.polarization_Y],
            [mw23_a, couplings.polarization_Y],
        ],
        phase_modulations=[True, False, False],
    )
    system = lindblad.generate_OBE_system_transitions(
        [p2, j12, j23],
        selectors,
        qn_compact=False,
        E=np.array([0.0, 0.0, E_FIELD_V_CM]),
        B=np.array([0.0, 0.0, B_FIELD_G]),
        retain_opposite_parity_levels=True,
        normalize_pol=True,
        # Include two rotational levels above the highest MW target J=3.
        Jmax_X=5,
        Jmax_B=3,
    )
    return system, selectors


def _params_for(
    selectors: list[Any],
    config: BenchmarkConfig,
    *,
    dynamic_switches: bool,
    dynamic_eom: bool,
    amplitude_envelope: str = "constant",
) -> LindbladParameters:
    generated = lindblad.generate_lindblad_parameters(selectors)
    base = dict(generated.base_parameters)
    compound = dict(generated.compound_parameters)
    # Replace generated defaults with declared benchmark values.
    base["Ωt0"] = config.optical_rabi_rad_s
    base["β0"] = BETA_EOM
    base["ωphase0"] = TWOPI * F_EOM_HZ
    base["δ0"] = config.detuning_rad_s
    if amplitude_envelope == "gaussian":
        base["center0"] = config.entry_phase_s + config.duration_s / 2.0
        base["sigma0"] = config.duration_s / 4.0
        if dynamic_eom:
            compound["Ω0"] = "Ωt0*gaussian_1d(t, center0, sigma0)*phase_modulation(t, β0, ωphase0)"
        else:
            compound["Ω0"] = "Ωt0*gaussian_1d(t, center0, sigma0)"
    elif amplitude_envelope != "constant":
        raise ValueError("amplitude_envelope must be 'constant' or 'gaussian'")
    for idx in (1, 2):
        base[f"Ω{idx}"] = config.microwave_rabi_rad_s
        base[f"δ{idx}"] = 0.0
        base[f"ω{idx}"] = TWOPI * F_SWITCH_HZ
        base[f"φ{idx}"] = 0.0 if idx == 1 else np.pi / 2.0
    if not dynamic_eom and amplitude_envelope == "constant":
        # Same constant optical amplitude with phase modulation removed.
        compound.pop("Ω0", None)
        base["Ω0"] = config.optical_rabi_rad_s
    if not dynamic_switches:
        for selector in selectors[1:]:
            names = [str(symbol) for symbol in selector.polarization_symbols]
            for name in names:
                compound.pop(name, None)
                base[name] = 0.0
    return LindbladParameters(base, compound)


def _prepare(system: Any, params: LindbladParameters) -> PreparedLindbladProblem:
    return prepare_lindblad_problem(
        system, params, backend="rust", hamiltonian_representation="decomposed"
    )


def build_case(
    config: BenchmarkConfig,
    *,
    prepare_frozen: bool = False,
    prepare_static: bool = False,
    amplitude_envelope: str = "constant",
) -> RCCase:
    """Build the explicit RC model and requested prepared variants.

    `prepared_switch_states` and `prepared_static_switch_states` are lazy by
    default because each preparation lowers the full system independently.
    Keys are `(PA1, PA2)` integer values in `{0, 1}`.
    """
    system, selectors = _make_model(config)
    dynamic = _params_for(
        selectors,
        config,
        dynamic_switches=True,
        dynamic_eom=True,
        amplitude_envelope=amplitude_envelope,
    )
    case = RCCase(config, amplitude_envelope, system, selectors, dynamic, _prepare(system, dynamic))
    _initial_labels(case)
    if prepare_frozen:
        _prepare_segmented_plan(case)
    if prepare_static:
        _prepare_switch_plans(case, static=True)
    return case


def _prepare_segmented_plan(case: RCCase) -> None:
    """Reuse the dynamic plan and freeze each switch by omega/phase overrides."""
    if case.prepared_segmented is not None:
        return
    case.segmented_parameters = case.params_dynamic
    case.prepared_segmented = case.prepared_dynamic
    # square_wave(t, 0, +pi/2) is constantly 1; with -pi/2 it is 0.
    # These are base parameters, so the native batch API safely overrides them.
    case.switch_override_slots = ("ω1", "φ1", "ω2", "φ2")


def _prepare_switch_plans(case: RCCase, *, static: bool) -> None:
    if static and case.amplitude_envelope != "constant":
        raise ValueError("static switch generators require amplitude_envelope='constant'")
    target_params = case.static_parameters if static else case.switch_parameters
    target_plans = case.prepared_static_switch_states if static else case.prepared_switch_states
    for pa1 in (0, 1):
        for pa2 in (0, 1):
            key = (pa1, pa2)
            if key in target_plans:
                continue
            params = _params_for(
                case.selectors,
                case.config,
                dynamic_switches=False,
                dynamic_eom=not static,
                amplitude_envelope=case.amplitude_envelope,
            )
            for selector, value in zip(case.selectors[1:], (pa1, pa2), strict=True):
                first, second = [str(symbol) for symbol in selector.polarization_symbols]
                params.base_parameters[first] = float(value)
                params.base_parameters[second] = float(1 - value)
            target_params[key] = params
            target_plans[key] = _prepare(case.system, params)


def _initial_labels(case: RCCase) -> None:
    # Three independently labeled X hyperfine states connected by the two MWs.
    specifications = {
        "X_J1_F1_3o2_F1_mF0": (1, 1.5, 1.0),
        "X_J2_F1_5o2_F2_mF0": (2, 2.5, 2.0),
        "X_J3_F1_7o2_F3_mF0": (3, 3.5, 3.0),
    }
    for label, (j, f1, f) in specifications.items():
        sel = states.QuantumSelector(
            electronic=states.ElectronicState.X,
            J=j,
            F1=f1,
            F=f,
            mF=0,
            P=(-1) ** j,
        )
        matches = np.asarray(sel.get_indices(case.system.QN), dtype=int).ravel()
        if matches.size != 1:
            raise RuntimeError(f"initial label {label} resolved to {matches.size} states")
        case.state_labels[label] = int(matches[0])


def switch_values(t: float) -> tuple[int, int]:
    """Evaluate the repository square-wave convention at `t`.

    Segmented propagation evaluates this at an interior point of each segment,
    so the exact value assigned at a discontinuity does not affect a segment.
    """
    omega = TWOPI * F_SWITCH_HZ
    pa1 = int(np.sin(omega * t) >= 0.0)
    pa2 = int(np.sin(omega * t + np.pi / 2.0) >= 0.0)
    return pa1, pa2


def switch_boundaries(t0: float, t1: float) -> np.ndarray:
    """Sorted exact square-wave boundaries in `[t0,t1]`, including endpoints."""
    points = [float(t0), float(t1)]
    omega = TWOPI * F_SWITCH_HZ
    for phase in (0.0, np.pi / 2.0):
        # sin(omega*t + phase)=0 at integer multiples of pi.
        n0 = int(np.floor((omega * t0 + phase) / np.pi)) - 1
        n1 = int(np.ceil((omega * t1 + phase) / np.pi)) + 1
        for n in range(n0, n1 + 1):
            t = (n * np.pi - phase) / omega
            if t0 < t < t1:
                points.append(float(t))
    return np.asarray(sorted(set(points)), dtype=np.float64)


def propagate_segmented_batch(
    case: RCCase,
    rho0_batch: np.ndarray,
    t_span: tuple[float, float],
    *,
    solver: str = "dopri5",
    reltol: float = 1e-7,
    abstol: float = 1e-9,
    dt: float = 1e-10,
    execution_mode: str = "expanded_sparse",
    save_step_s: float | None = None,
    parallel: bool = True,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Propagate at switch boundaries using frozen polarization plans.

    Returns times, full densities with shape `(initial, time, N, N)`, and
    per-segment statistics (including one solver-native photon integral per
    initial condition). Initial density matrices are independent trajectories
    on one shared parameter graph and are carried across each boundary.
    """
    _prepare_segmented_plan(case)
    t0, t1 = map(float, t_span)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("t_span must contain finite endpoints with t1 > t0")
    edges = switch_boundaries(t0, t1)
    rho = np.asarray(rho0_batch, dtype=np.complex128)
    if rho.ndim == 2:
        rho = rho[np.newaxis, :, :]
    if rho.ndim != 3 or rho.shape[1:] != (case.n_states, case.n_states):
        raise ValueError(f"rho0_batch must have shape (n, {case.n_states}, {case.n_states})")
    if rho.shape[0] == 0:
        raise ValueError("rho0_batch must contain at least one initial state")
    matrices = [rho.copy()]
    times = [t0]
    stats: list[dict[str, Any]] = []
    for lo, hi in pairwise(edges):
        state = switch_values(float((lo + hi) / 2.0))
        assert case.prepared_segmented is not None
        prepared = case.prepared_segmented
        switch_row = np.asarray(
            [
                0.0,
                SWITCH_PHASE_OFFSET_RAD if state[0] else -SWITCH_PHASE_OFFSET_RAD,
                0.0,
                SWITCH_PHASE_OFFSET_RAD if state[1] else -SWITCH_PHASE_OFFSET_RAD,
            ],
            dtype=np.complex128,
        )
        switch_parameters = np.tile(switch_row, (rho.shape[0], 1))
        segment_saveat = np.array([float(hi)], dtype=np.float64)
        if save_step_s is not None:
            if save_step_s <= 0:
                raise ValueError("save_step_s must be positive")
            interior = np.arange(float(lo) + save_step_s, float(hi), save_step_s)
            segment_saveat = np.concatenate((interior, np.array([float(hi)])))
        rho_start = rho.copy()
        full_started = time.perf_counter()
        result = solve_lindblad_batch(
            prepared,
            rho,
            (float(lo), float(hi)),
            parameter_batch=switch_parameters,
            parameter_slots=case.switch_override_slots,
            solver=solver,
            execution_mode=execution_mode,
            reltol=reltol,
            abstol=abstol,
            dt=dt,
            saveat=segment_saveat,
            save_start=False,
            output="full",
            output_when="saveat",
            collect_stats=True,
            parallel=parallel,
            threads=threads,
        )
        packed_values = np.asarray(result.values, dtype=np.float64)
        result_matrices = np.empty(
            (rho.shape[0], packed_values.shape[1], case.n_states, case.n_states),
            dtype=np.complex128,
        )
        for trajectory in range(rho.shape[0]):
            for sample in range(packed_values.shape[1]):
                result_matrices[trajectory, sample] = prepared.layout.unpack(
                    packed_values[trajectory, sample]
                )
        rho = result_matrices[:, -1]
        full_elapsed = time.perf_counter() - full_started
        result_times = np.asarray(result.t, dtype=float)
        times.extend(result_times.tolist())
        matrices.extend(result_matrices.transpose(1, 0, 2, 3))
        # Use the Rust solver's integrated excited-state rate. Boundary-only
        # trapezoids badly underresolve the 99 ns lifetime and 625 ns EOM period.
        excited = [
            i
            for i, state_qn in enumerate(case.system.QN)
            if state_qn.largest.electronic_state == states.ElectronicState.B
        ]
        photon_started = time.perf_counter()
        photon = solve_lindblad_batch(
            prepared,
            rho_start,
            (float(lo), float(hi)),
            parameter_batch=switch_parameters,
            parameter_slots=case.switch_override_slots,
            solver=solver,
            execution_mode=execution_mode,
            reltol=reltol,
            abstol=abstol,
            dt=dt,
            output="photon_integral",
            output_when="final",
            integral_weights=[(int(i), GAMMA) for i in excited],
            parallel=parallel,
            threads=threads,
        )
        segment_photons = np.asarray(photon.values, dtype=float).reshape(-1).real
        photon_elapsed = time.perf_counter() - photon_started
        stats.append(
            {
                **(result.solver_stats or {}),
                "photon_integral": segment_photons,
                "full_density_wall_seconds": full_elapsed,
                "photon_integral_wall_seconds": photon_elapsed,
                "combined_wall_seconds": full_elapsed + photon_elapsed,
            }
        )
    return np.asarray(times), np.stack(matrices, axis=1), stats


def propagate_segmented(
    case: RCCase,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Single-initial-state compatibility wrapper around batch propagation."""
    times, matrices, stats = propagate_segmented_batch(
        case,
        np.asarray(rho0, dtype=np.complex128)[np.newaxis, :, :],
        t_span,
        parallel=False,
        **kwargs,
    )
    scalar_stats = [
        {**row, "photon_integral": float(np.asarray(row["photon_integral"]).ravel()[0])}
        for row in stats
    ]
    return times, matrices[0], scalar_stats


def full_density_metrics(rho: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(rho, dtype=np.complex128)
    herm = matrix - matrix.conj().T
    eig_min = float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0).min())
    return {
        "trace_real": float(np.trace(matrix).real),
        "trace_abs_imag": float(abs(np.trace(matrix).imag)),
        "hermiticity_max_abs": float(np.max(np.abs(herm))),
        "minimum_eigenvalue": eig_min,
    }


def photon_yield(segment_stats: list[dict[str, Any]]) -> float:
    """Sum solver-native integrated photon counts from the exact segments."""
    return float(sum(float(row.get("photon_integral", 0.0)) for row in segment_stats))


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=HERE.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def provenance(case: RCCase) -> dict[str, Any]:
    try:
        import scipy

        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    try:
        import threadpoolctl

        blas = threadpoolctl.threadpool_info()
    except Exception:
        blas = None
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        import centrex_tlf.centrex_tlf_rust as rust_extension

        extension_path = Path(rust_extension.__file__).resolve()
        extension_sha256 = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    except Exception:
        extension_path = None
        extension_sha256 = None
    try:
        import psutil

        memory = psutil.virtual_memory()
        resource_snapshot = {
            "available_memory_bytes": int(memory.available),
            "total_memory_bytes": int(memory.total),
            "system_cpu_percent": float(psutil.cpu_percent(interval=0.1)),
        }
    except Exception:
        resource_snapshot = None
    return {
        "source_revision": _git_revision(),
        "driver_sha256": source_hash,
        "rust_extension_path": str(extension_path) if extension_path else None,
        "rust_extension_sha256": extension_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "centrex_tlf": importlib.metadata.version("centrex-tlf"),
        "blas": blas,
        "cpu_count": os.cpu_count(),
        "resource_snapshot": resource_snapshot,
        "model": {
            "n_states": case.n_states,
            "n_qn": len(case.system.QN),
            "E_V_cm": E_FIELD_V_CM,
            "B_G": B_FIELD_G,
            "retain_opposite_parity_levels": True,
            "qn_compact": False,
            "Jmax_X": 5,
            "Jmax_B": 3,
            "selectors": [
                {
                    "transition": selector.description,
                    "ground_main": str(selector.ground_main),
                    "excited_main": str(selector.excited_main),
                    "polarization_vectors": [
                        np.asarray(v).tolist() for v in selector.polarizations
                    ],
                    "polarization_symbols": [str(s) for s in selector.polarization_symbols],
                }
                for selector in case.selectors
            ],
            "main_couplings": [
                str(getattr(coupling, "main_coupling", None)) for coupling in case.system.couplings
            ],
        },
        "config": asdict(case.config),
        "amplitude_envelope": case.amplitude_envelope,
        "rates": {
            "gamma_s-1": GAMMA,
            "optical_rabi_rad_s": case.config.optical_rabi_rad_s,
            "microwave_rabi_rad_s": case.config.microwave_rabi_rad_s,
            "detuning_rad_s": case.config.detuning_rad_s,
            "eom_beta": BETA_EOM,
            "eom_frequency_hz": F_EOM_HZ,
            "switch_frequency_hz": F_SWITCH_HZ,
            "switch_offset_s": SWITCH_OFFSET_S,
            "switch_phase_offset_rad": SWITCH_PHASE_OFFSET_RAD,
            "switch_phase_convention": "positive phase advances the square wave by 250 ns; second switch edges lead the zero-phase switch",
            "optical_rabi_normalization": "Ω amplitude in rad/s; normalized coupling matrix uses selected main pair",
        },
    }


def run_smoke(config_name: str = "weak", *, amplitude_envelope: str = "constant") -> dict[str, Any]:
    config = CONFIGS[config_name]
    started = time.perf_counter()
    case = build_case(config, prepare_frozen=True, amplitude_envelope=amplitude_envelope)
    build_seconds = time.perf_counter() - started
    label = next(iter(case.state_labels))
    rho0 = case.rho0(label)
    t0 = config.entry_phase_s
    # One switch cycle covers both complementary polarization states and
    # boundary continuation, while keeping this a setup/smoke gate.
    t1 = t0 + 1e-6
    started = time.perf_counter()
    ts, rhos, stats = propagate_segmented(
        case,
        rho0,
        (t0, t1),
        reltol=1e-9,
        abstol=1e-11,
    )
    solve_seconds = time.perf_counter() - started
    final_metrics = full_density_metrics(rhos[-1])
    if abs(final_metrics["trace_real"] - 1.0) > 2e-6:
        raise AssertionError(f"trace drifted in smoke solve: {final_metrics}")
    if final_metrics["hermiticity_max_abs"] > 1e-9:
        raise AssertionError(f"density lost Hermiticity in smoke solve: {final_metrics}")
    if final_metrics["minimum_eigenvalue"] < -1e-8:
        raise AssertionError(f"density lost positivity in smoke solve: {final_metrics}")
    return {
        "config": config_name,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "n_states": case.n_states,
        "segments": len(stats),
        "switch_boundaries_s": ts.tolist(),
        "solver_stats": stats,
        "final_density_metrics": final_metrics,
        "photon_yield": photon_yield(stats),
        "provenance": provenance(case),
    }


def run_baseline(
    config_name: str,
    repeats: int = 3,
    output_dir: Path = RESULTS_DIR,
    amplitude_envelope: str = "constant",
) -> list[dict[str, Any]]:
    config = CONFIGS[config_name]
    case = build_case(config, prepare_frozen=True, amplitude_envelope=amplitude_envelope)
    rows: list[dict[str, Any]] = []
    raw_labels: list[str] = []
    raw_methods = ["reference_dopri5_1e-10", "dopri5_1e-7", "tsit5_1e-7"]
    raw_times: np.ndarray | None = None
    raw_densities: list[np.ndarray] = []
    for label in case.state_labels:
        rho0 = case.rho0(label)
        t_span = (config.entry_phase_s, config.entry_phase_s + config.duration_s)
        times_ref, matrices_ref, stats_ref = propagate_segmented(
            case,
            rho0,
            t_span,
            solver="dopri5",
            reltol=1e-10,
            abstol=1e-12,
            save_step_s=100e-9,
        )
        if raw_times is None:
            raw_times = times_ref
        elif not np.array_equal(raw_times, times_ref):
            raise AssertionError("reference output grids differ across initial states")
        raw_labels.append(label)
        raw_densities.append(matrices_ref)
        for solver in ("dopri5", "tsit5"):
            elapsed: list[float] = []
            last: tuple[np.ndarray, np.ndarray, list[dict[str, Any]]] | None = None
            for _ in range(repeats):
                started = time.perf_counter()
                last = propagate_segmented(
                    case,
                    rho0,
                    (config.entry_phase_s, config.entry_phase_s + config.duration_s),
                    solver=solver,
                    reltol=1e-7,
                    abstol=1e-9,
                    save_step_s=100e-9,
                )
                elapsed.append(time.perf_counter() - started)
            assert last is not None
            times, matrices, stats = last
            if raw_times is None or not np.array_equal(raw_times, times):
                raise AssertionError("solver and reference output grids differ")
            raw_densities.append(matrices)
            candidate_photons = photon_yield(stats)
            reference_photons = photon_yield(stats_ref)
            photon_error = abs(candidate_photons - reference_photons)
            photon_tolerance = 1e-6 + 1e-5 * abs(reference_photons)
            rows.append(
                {
                    "config": config_name,
                    "initial_label": label,
                    "solver": solver,
                    "repeats": repeats,
                    "median_seconds": float(np.median(elapsed)),
                    "duration_s": config.duration_s,
                    "segments": len(stats),
                    "photon_yield": candidate_photons,
                    "reference_photon_yield": reference_photons,
                    "photon_yield_abs_error_vs_reference": photon_error,
                    "photon_yield_acceptance_tolerance": photon_tolerance,
                    "photon_yield_within_acceptance": photon_error <= photon_tolerance,
                    "max_abs_full_density_error_vs_reference": float(
                        np.max(np.abs(matrices - matrices_ref))
                    ),
                    "max_abs_population_error_vs_reference": float(
                        np.max(
                            np.abs(
                                np.real(np.diagonal(matrices, axis1=1, axis2=2))
                                - np.real(np.diagonal(matrices_ref, axis1=1, axis2=2))
                            )
                        )
                    ),
                    **full_density_metrics(matrices[-1]),
                }
            )
            if photon_error > photon_tolerance:
                raise AssertionError(
                    f"{solver} photon yield error {photon_error:.6g} exceeds "
                    f"acceptance tolerance {photon_tolerance:.6g} for {label}"
                )
    raw_path = output_dir / f"baseline_{config_name}_{amplitude_envelope}_full_density.npz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_array = np.asarray(raw_densities, dtype=np.complex128)
    # Order is label, then reference/dopri5/tsit5. Keep every full density
    # snapshot so later methods can be compared without rerunning this case.
    raw_array = raw_array.reshape(
        (len(case.state_labels), len(raw_methods), len(raw_times), case.n_states, case.n_states)
    )
    np.savez_compressed(
        raw_path,
        labels=np.asarray(raw_labels, dtype=str),
        methods=np.asarray(raw_methods, dtype=str),
        times=raw_times,
        density_matrices=raw_array,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("setup", "smoke", "baseline"), default="setup")
    parser.add_argument("--config", choices=tuple(CONFIGS), default="weak")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--envelope", choices=("constant", "gaussian"), default="constant")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "setup":
        started = time.perf_counter()
        case = build_case(CONFIGS[args.config], amplitude_envelope=args.envelope)
        payload = {
            "stage": "setup",
            "build_seconds": time.perf_counter() - started,
            "provenance": provenance(case),
            "initial_labels": case.state_labels,
        }
    elif args.stage == "smoke":
        payload = {"stage": "smoke", **run_smoke(args.config, amplitude_envelope=args.envelope)}
    else:
        rows = run_baseline(args.config, args.repeats, args.output_dir, args.envelope)
        payload = {"stage": "baseline", "rows": rows}
        with (args.output_dir / f"baseline_{args.config}_{args.envelope}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    out = args.output_dir / f"{args.stage}_{args.config}_{args.envelope}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
