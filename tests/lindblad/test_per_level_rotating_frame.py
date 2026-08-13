"""Validation gates for `apply_per_level_rotating_frame`.

See `benchmarks/step_size_diagnostics_results/step_size_diagnostics_report.md`
section 4(a): the r2-in-static-E-field system is oscillation-limited by the
driven B J=3 manifold's 73.6 MHz static diagonal spread, not accuracy-limited.
`apply_per_level_rotating_frame` (in
`centrex_tlf/lindblad/generate_hamiltonian.py`) removes the numeric static
diagonal analytically via a per-level unitary `T = diag(exp(-i*E_i*t))`; this
module checks the physics is unchanged (populations are frame-invariant) both
on a 2-level toy system and on the actual r2 system used in the benchmark.

These are hard gates: if either test fails, the helper is not validated and
must not be advertised as usable -- do not loosen tolerances to make them
pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as smp

from centrex_tlf import hamiltonian
from centrex_tlf.lindblad.generate_hamiltonian import apply_per_level_rotating_frame
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.utils_setup import OBESystem

GAMMA = hamiltonian.Γ  # 2*pi*1.56e6 rad/s


def _load_diagnose_step_size():
    """Import `benchmarks/diagnose_step_size.py` as a module (not a package)."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "benchmarks" / "diagnose_step_size.py"
    spec = importlib.util.spec_from_file_location("diagnose_step_size", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("diagnose_step_size", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def two_level_obe_system() -> OBESystem:
    """2-level toy with a static diagonal splitting plus a symbolic detuning.

    H = [[0, Omega/2], [conj(Omega)/2, D0 + delta]] with numeric
    D0 = 2*pi*30e6 folded in as a numeric addition on the diagonal, and one
    decay operator (excited -> ground) at rate Gamma.
    """
    Omega, delta = smp.symbols("Omega delta", real=True)
    D0 = 2 * np.pi * 30e6
    H = smp.Matrix([[0, Omega / 2], [smp.conjugate(Omega) / 2, D0 + delta]])
    C_array = np.zeros((1, 2, 2), dtype=complex)
    C_array[0, 0, 1] = np.sqrt(GAMMA)  # excited (1) -> ground (0)

    return OBESystem(
        ground=[0],
        excited=[1],
        QN=[0, 1],
        H_int=np.diag([0.0, D0]).astype(complex),
        V_ref_int=np.zeros((2, 2), dtype=complex),
        couplings=[],
        H_symbolic=H,
        C_array=C_array,
        coupling_symbols=[Omega],
        polarization_symbols=[],
    )


def _two_level_params(rabi_value: float, detuning_value: float) -> LindbladParameters:
    params = LindbladParameters()
    omega = params.real("Omega", rabi_value)
    delta = params.real("delta", detuning_value)
    params.bind(smp.Symbol("Omega", real=True), omega, finalize=False)
    params.bind(smp.Symbol("delta", real=True), delta, finalize=False)
    params._finalize()
    return params


def test_two_level_toy_rotated_frame_matches_original(
    two_level_obe_system: OBESystem,
) -> None:
    obe = two_level_obe_system
    rotated = apply_per_level_rotating_frame(obe)

    # Sanity: the rotated Hamiltonian is now explicitly time-dependent, and
    # the numeric static part of the diagonal has been removed.
    t = smp.Symbol("t", real=True)
    assert t in rotated.H_symbolic.free_symbols
    for i in range(2):
        entry = rotated.H_symbolic[i, i]
        zero_subs = {s: 0 for s in entry.free_symbols}
        assert abs(complex(entry.subs(zero_subs)).real) < 1e-6
    # C_array must be untouched (identical object/values).
    np.testing.assert_array_equal(rotated.C_array, obe.C_array)

    rabi_value = 2 * np.pi * 0.3e6
    detuning_value = 0.0
    params = _two_level_params(rabi_value, detuning_value)

    prepared = prepare_lindblad_problem(
        obe, params, backend="rust", hamiltonian_representation="decomposed"
    )
    prepared_rotated = prepare_lindblad_problem(
        rotated, params, backend="rust", hamiltonian_representation="decomposed"
    )

    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    t_span = (0.0, 5e-6)
    saveat = np.linspace(t_span[0], t_span[1], 200)

    common_kwargs = dict(
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="saveat",
        saveat=saveat,
        dense_output=True,
        abstol=1e-10,
        reltol=1e-8,
    )

    result = solve_lindblad(prepared, rho0, t_span, **common_kwargs)
    result_rotated = solve_lindblad(prepared_rotated, rho0, t_span, **common_kwargs)

    np.testing.assert_allclose(result.t, result_rotated.t, atol=0.0, rtol=0.0)
    populations = result.values.reshape(saveat.size, 2)
    populations_rotated = result_rotated.values.reshape(saveat.size, 2)
    np.testing.assert_allclose(populations_rotated, populations, atol=1e-6)


@pytest.mark.parametrize("detuning_mhz", [0.0, 25.0])
def test_r2_system_rotated_frame_matches_original(detuning_mhz: float) -> None:
    diagnose = _load_diagnose_step_size()

    with pytest.warns(UserWarning, match="Low overlap detected"):
        system, ts = diagnose.build_system()
    rotated = apply_per_level_rotating_frame(system)

    rabi_value = diagnose.power_to_rabi_rectangular_beam(
        diagnose.POWER_W,
        abs(system.couplings[0].main_coupling),
        diagnose.BEAM_WX,
        diagnose.BEAM_WY,
    )
    detuning_rad = 2 * np.pi * detuning_mhz * 1e6
    params = diagnose.make_parameters(system, ts, rabi_value, detuning_rad)

    rho0 = diagnose.build_rho0(system)

    prepared = prepare_lindblad_problem(
        system, params, backend="rust", hamiltonian_representation="decomposed"
    )
    prepared_rotated = prepare_lindblad_problem(
        rotated, params, backend="rust", hamiltonian_representation="decomposed"
    )

    common_kwargs = dict(
        solver="dopri5",
        execution_mode="expanded_sparse",
        output="populations",
        output_when="final",
        dense_output=False,
        reltol=1e-7,
        abstol=1e-9,
    )

    result = solve_lindblad(prepared, rho0, (0.0, diagnose.T_END), **common_kwargs)
    result_rotated = solve_lindblad(
        prepared_rotated, rho0, (0.0, diagnose.T_END), **common_kwargs
    )

    np.testing.assert_allclose(
        result_rotated.values, result.values, atol=1e-5, err_msg=(
            f"rotated-frame final populations disagree with original frame at "
            f"detuning={detuning_mhz} MHz"
        )
    )
