from __future__ import annotations

from typing import Any

import numpy as np

from .ir import evaluate_parameter_graph_py, fill_hamiltonian_py

__all__ = [
    "evaluate_hamiltonian_reference",
    "apply_dense_dissipator_reference",
    "apply_structured_dissipator_reference",
    "reference_rhs",
    "structured_rhs",
    "reference_jvp",
]


def evaluate_hamiltonian_reference(prepared: Any, t: float) -> np.ndarray:
    slot_values = evaluate_parameter_graph_py(prepared.parameter_graph, t)
    return fill_hamiltonian_py(prepared.hamiltonian_plan, slot_values, t)


def apply_dense_dissipator_reference(c_array: np.ndarray, rho: np.ndarray) -> np.ndarray:
    result = np.zeros_like(rho, dtype=np.complex128)
    for collapse in c_array:
        dagger = collapse.conjugate().T
        result += collapse @ rho @ dagger
        cdagger_c = dagger @ collapse
        result -= 0.5 * (cdagger_c @ rho + rho @ cdagger_c)
    return result


def apply_structured_dissipator_reference_legacy(
    structured_jumps: list[dict[str, Any]],
    source_decay_rates: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(rho, dtype=np.complex128)
    for jump in structured_jumps:
        result[jump["target"], jump["target"]] += jump["rate"] * rho[
            jump["source"], jump["source"]
        ]
    for source, rate in enumerate(source_decay_rates):
        if rate == 0.0:
            continue
        result[source, :] -= 0.5 * rate * rho[source, :]
        result[:, source] -= 0.5 * rate * rho[:, source]
    return result


def apply_structured_dissipator_reference(
    structured_jumps: list[dict[str, Any]],
    source_decay_rates: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(rho, dtype=np.complex128)
    n = rho.shape[0]

    for i, rate in enumerate(source_decay_rates):
        result[i, i] -= rate * rho[i, i]

    for i in range(n):
        rate_i = source_decay_rates[i]
        for j in range(n):
            if i == j:
                continue
            result[i, j] -= 0.5 * (rate_i + source_decay_rates[j]) * rho[i, j]

    for jump in structured_jumps:
        result[jump["target"], jump["target"]] += jump["rate"] * rho[
            jump["source"], jump["source"]
        ]

    return result


def reference_rhs(prepared: Any, packed_state: np.ndarray, t: float) -> np.ndarray:
    rho = prepared.layout.unpack(np.asarray(packed_state, dtype=np.float64))
    hamiltonian = evaluate_hamiltonian_reference(prepared, t)
    drho = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    drho += apply_dense_dissipator_reference(prepared.dense_c_array, rho)
    return prepared.layout.pack(drho)


def structured_rhs(prepared: Any, packed_state: np.ndarray, t: float) -> np.ndarray:
    rho = prepared.layout.unpack(np.asarray(packed_state, dtype=np.float64))
    hamiltonian = evaluate_hamiltonian_reference(prepared, t)
    drho = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    drho += apply_structured_dissipator_reference(
        prepared.structured_jumps,
        prepared.source_decay_rates,
        rho,
    )
    return prepared.layout.pack(drho)


def reference_jvp(prepared: Any, packed_vector: np.ndarray, t: float) -> np.ndarray:
    return reference_rhs(prepared, packed_vector, t)
