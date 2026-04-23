from __future__ import annotations

from typing import Sequence

import numpy as np
import sympy as smp

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.effective_hamiltonian._utility import (
    _as_field_vector,
    _selector_indices,
    _symmetrize,
)
from centrex_tlf.effective_hamiltonian._decay import _sector_decay_kernel
from centrex_tlf.effective_hamiltonian.operator_bundle import OperatorBundle


def build_compact_reference_decomposed_bundle(
    *,
    transition: transitions.OpticalTransition,
    optical_polarization: couplings.Polarization,
    electric_field: Sequence[float] | np.ndarray = (0.0, 0.0, 100.0),
    magnetic_field: Sequence[float] | np.ndarray = (0.0, 0.0, 1e-5),
    polarization_scale: float = 1.0,
) -> tuple[lindblad.utils_setup.OBESystem, OperatorBundle]:
    electric = _as_field_vector(electric_field)
    magnetic = _as_field_vector(magnetic_field)
    transition_selectors = couplings.generate_transition_selectors(
        transitions=[transition],
        polarizations=[[optical_polarization]],
    )
    system = lindblad.generate_OBE_system_transitions(
        [transition],
        transition_selectors,
        E=electric,
        B=magnetic,
        qn_compact=True,
    )

    symbols = list(system.H_symbolic.free_symbols)
    ham_lambdify = smp.lambdify(symbols, system.H_symbolic, modules="numpy", cse=True)

    def evaluate_compact_hamiltonian(
        *,
        rabi_value: complex = 0.0,
        detuning_value: float = 0.0,
    ) -> np.ndarray:
        values: dict[str, float | complex] = {}
        for symbol in symbols:
            name = str(symbol)
            if name.startswith("\u03a9"):
                values[name] = complex(rabi_value)
            elif name.startswith("\u03b4"):
                values[name] = float(detuning_value)
            elif name.startswith("PZ"):
                values[name] = float(polarization_scale)
            elif name.startswith("PX") or name.startswith("PY"):
                values[name] = 0.0
            else:
                raise ValueError(f"Unsupported compact-reference symbol {name}")
        return np.asarray(ham_lambdify(**values), dtype=np.complex128)

    h_internal = _symmetrize(evaluate_compact_hamiltonian(rabi_value=0.0, detuning_value=0.0))
    h_with_rabi = _symmetrize(evaluate_compact_hamiltonian(rabi_value=1.0, detuning_value=0.0))
    h_with_detuning = _symmetrize(
        evaluate_compact_hamiltonian(rabi_value=0.0, detuning_value=1.0)
    )
    h_opt = _symmetrize(2.0 * (h_with_rabi - h_internal))
    h_det = _symmetrize(h_with_detuning - h_internal)
    c_array = np.asarray(system.C_array, dtype=np.complex128)
    excited_indices = _selector_indices(
        system.QN,
        states.QuantumSelector(electronic=states.ElectronicState.B),
    )
    ground_indices = _selector_indices(
        system.QN,
        states.QuantumSelector(J=0, electronic=states.ElectronicState.X),
    )
    sink_groups = tuple(
        _selector_indices(
            system.QN,
            states.QuantumSelector(J=J, electronic=states.ElectronicState.X),
        )
        for J in (1, 2, 3)
    )
    decay_kernel_ground = _sector_decay_kernel(c_array, excited_indices, ground_indices)
    decay_kernels_sinks = tuple(
        _sector_decay_kernel(c_array, excited_indices, sink_group)
        for sink_group in sink_groups
    )
    excited_to_ground_rates = np.real(np.diag(decay_kernel_ground)) / (2.0 * np.pi)
    excited_to_sink_rates = (
        np.sum(
            np.array([np.real(np.diag(kernel)) for kernel in decay_kernels_sinks], dtype=np.float64),
            axis=0,
        )
        / (2.0 * np.pi)
    )

    bundle = OperatorBundle(
        electric_field=electric,
        magnetic_field=magnetic,
        omega_reference=0.0,
        h_internal=h_internal,
        h_opt=h_opt,
        h_det=h_det,
        c_array=c_array,
        excited_indices=excited_indices,
        loss_operator=np.zeros_like(h_internal, dtype=np.complex128),
        h_full_internal=h_internal,
        h_lab_internal=h_internal,
        hermiticity_error=float(np.max(np.abs(h_internal - h_internal.conj().T))),
        excited_to_ground_rates_hz=excited_to_ground_rates,
        excited_to_sink_rates_hz=excited_to_sink_rates,
        decay_kernel_ground=decay_kernel_ground,
        decay_kernels_sinks=decay_kernels_sinks,
    )
    return system, bundle
