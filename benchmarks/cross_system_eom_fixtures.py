"""Phase-modulated versions of the frozen compact R2/R3 fixtures.

The source fixture models (basis, fields, labels and initial state) are reused
unchanged. Only the optical Rabi symbol is rebound to a complex phase envelope;
building a fixture prepares the native plan but does not integrate an ODE.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from cross_system_fixtures import SPECS, build_fixture as build_static_fixture
except ModuleNotFoundError:
    from benchmarks.cross_system_fixtures import SPECS, build_fixture as build_static_fixture

from centrex_tlf.lindblad.parameters import LindbladParameters, phase_modulation
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem

HERE = Path(__file__).resolve().parent
EOM_FIXTURES = ("r2_f4", "r3_f1_7o2_f4", "r3_f1_9o2_f4")
ENTRY_PHASE_S = 125e-9
SCREEN_DURATION_S = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_parameters(*, rabi_rad_s: float, beta: float, omega_rad_s: float) -> tuple[LindbladParameters, Any]:
    """Return runtime parameters for Ω(t)=Ω₀ exp(i β sin(ωt))."""
    params = LindbladParameters()
    amplitude = params.real("Ωt0", float(rabi_rad_s))
    beta_param = params.real("β0", float(beta))
    omega_param = params.real("ωphase0", float(omega_rad_s))
    expression = amplitude * phase_modulation(params.time(), beta_param, omega_param)
    return params, expression


def build_fixture(name: str):
    """Build a phase-modulated compact R2/R3 fixture without solving it."""
    if name not in EOM_FIXTURES:
        raise ValueError(f"EOM fixture must be one of {EOM_FIXTURES}; got {name!r}")

    # The frozen model builder supplies exactly the same Hamiltonian, labels,
    # field configuration and initial density as its static counterpart.
    fixture = build_static_fixture(name)
    try:
        rc = importlib.import_module("bench_rc_propagation")
    except ModuleNotFoundError:
        rc = importlib.import_module("benchmarks.bench_rc_propagation")
    beta = float(rc.BETA_EOM)
    omega = float(2.0 * np.pi * rc.F_EOM_HZ)
    rabi = float(fixture.runtime_values["rabi_rad_s"])
    params, modulated_rabi = phase_parameters(
        rabi_rad_s=rabi, beta=beta, omega_rad_s=omega,
    )

    coupling_symbols = set(fixture.system.coupling_symbols)
    for symbol in sorted(fixture.system.H_symbolic.free_symbols, key=str):
        if symbol in coupling_symbols:
            params.bind(symbol, modulated_rabi, finalize=False)
        else:
            params.real(str(symbol), 0.0)
    for group in fixture.system.polarization_symbols:
        for symbol in group if isinstance(group, list | tuple) else [group]:
            params.bind(symbol, 1.0, finalize=False)
    params._finalize()

    prepared = prepare_lindblad_problem(
        fixture.system, params, backend="rust", hamiltonian_representation="decomposed",
    )
    spec = replace(fixture.spec, t_span=(ENTRY_PHASE_S, ENTRY_PHASE_S + SCREEN_DURATION_S))
    phase_at_entry = beta * float(np.sin(omega * ENTRY_PHASE_S))
    phase_factor = complex(np.cos(phase_at_entry), np.sin(phase_at_entry))
    if abs(phase_factor.imag) < 1e-3 or abs(abs(phase_factor) - 1.0) > 1e-14:
        raise RuntimeError("EOM entry point does not exercise a nontrivial unit-modulus optical phase")

    runtime_values = {
        **fixture.runtime_values,
        "phase_modulation_enabled": True,
        "eom_beta": beta,
        "eom_frequency_hz": float(rc.F_EOM_HZ),
        "eom_angular_frequency_rad_s": omega,
        "optical_phase_at_entry_rad": phase_at_entry,
        "optical_phase_factor_at_entry": [phase_factor.real, phase_factor.imag],
    }
    configuration = {
        **fixture.configuration,
        "t_span_s": list(spec.t_span),
        "phase_modulation": "exp(i*beta*sin(omega*t))",
        "phase_modulation_is_complex": True,
        "amplitude_envelope": "constant",
        "entry_phase_s": ENTRY_PHASE_S,
        "eom_beta": beta,
        "eom_frequency_hz": float(rc.F_EOM_HZ),
        "eom_angular_frequency_rad_s": omega,
        "solver": "tsit5",
    }
    metadata = {
        **fixture.metadata,
        "eom_fixture_builder_sha256": _sha256(Path(__file__).resolve()),
        "static_fixture_builder_sha256": fixture.metadata["fixture_builder_sha256"],
        "entry_phase_s": ENTRY_PHASE_S,
        "absolute_time_phase_evaluation": True,
        "phase_factor_at_entry": [phase_factor.real, phase_factor.imag],
    }
    return replace(
        fixture, spec=spec, prepared=prepared, runtime_values=runtime_values,
        configuration=configuration, metadata=metadata,
    )


def evaluate_phase_factor(t_s: float, *, beta: float, omega_rad_s: float) -> complex:
    """Small pure helper used by contract tests and provenance checks."""
    phase = float(beta) * float(np.sin(float(omega_rad_s) * float(t_s)))
    return complex(np.cos(phase), np.sin(phase))
