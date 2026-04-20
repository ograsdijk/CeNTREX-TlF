from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ir import lower_hamiltonian_upper_triangle, lower_parameter_graph
from .parameters import LindbladParameters, adapt_lindblad_parameters
from .state_layout import PackedHermitianLayout

__all__ = ["PreparedLindbladProblem", "prepare_lindblad_problem"]


@dataclass
class PreparedLindbladProblem:
    obe_system: Any
    parameters: LindbladParameters
    layout: PackedHermitianLayout
    parameter_graph: dict[str, Any]
    hamiltonian_plan: dict[str, Any]
    dense_c_array: np.ndarray
    structured_jumps: list[dict[str, Any]]
    source_decay_rates: np.ndarray
    incoming_transfers_by_target: list[list[dict[str, Any]]]
    blas_config: dict[str, Any] | None
    backend: str
    rust_plan: Any | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "n_states": int(self.layout.n),
            "parameter_graph": self.parameter_graph,
            "hamiltonian_plan": self.hamiltonian_plan,
            "structured_jumps": self.structured_jumps,
            "source_decay_rates": np.asarray(self.source_decay_rates, dtype=np.float64),
            "incoming_transfers_by_target": self.incoming_transfers_by_target,
            "dense_c_array": np.asarray(self.dense_c_array, dtype=np.complex128),
            "blas_config": self.blas_config,
        }


def _locate_scipy_openblas() -> dict[str, Any] | None:
    candidates: list[Path] = []
    try:
        import numpy

        candidates.extend(Path(numpy.__file__).resolve().parent.parent.glob("numpy.libs/libscipy_openblas*.dll"))
    except Exception:
        pass
    try:
        import scipy

        candidates.extend(Path(scipy.__file__).resolve().parent.parent.glob("scipy.libs/libscipy_openblas*.dll"))
    except Exception:
        pass
    if not candidates:
        return None
    symbol_candidates = [
        "scipy_cblas_zher2k64_",
        "scipy_cblas_zher2k",
        "cblas_zher2k64_",
        "cblas_zher2k",
    ]
    for candidate in candidates:
        try:
            dll = ctypes.WinDLL(str(candidate))
        except Exception:
            continue
        for symbol in symbol_candidates:
            try:
                getattr(dll, symbol)
            except AttributeError:
                continue
            return {"library_path": str(candidate), "zher2k_symbol": symbol}
    return None


def _lower_structured_jumps(
    c_array: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, list[list[dict[str, Any]]]]:
    jumps: list[dict[str, Any]] = []
    n_states = int(c_array.shape[1])
    source_decay_rates = np.zeros(n_states, dtype=np.float64)
    incoming_transfers_by_target: list[list[dict[str, Any]]] = [[] for _ in range(n_states)]
    for idx, collapse in enumerate(c_array):
        nz = np.argwhere(np.abs(collapse) > 0)
        if nz.shape[0] != 1:
            raise AssertionError(
                f"collapse operator {idx} is not a single-jump operator; found {nz.shape[0]} nonzero elements"
            )
        target, source = (int(nz[0, 0]), int(nz[0, 1]))
        amplitude = complex(collapse[target, source])
        rate = float((amplitude * np.conjugate(amplitude)).real)
        jumps.append(
            {
                "target": target,
                "source": source,
                "amp_re": float(amplitude.real),
                "amp_im": float(amplitude.imag),
                "rate": rate,
            }
        )
        incoming_transfers_by_target[target].append({"source": source, "rate": rate})
        source_decay_rates[source] += rate
    return jumps, source_decay_rates, incoming_transfers_by_target


def prepare_lindblad_problem(
    obe_system: Any,
    parameters: Any,
    backend: str = "rust",
    hamiltonian_representation: str = "auto",
) -> PreparedLindbladProblem:
    params = adapt_lindblad_parameters(parameters)
    params.check_hamiltonian_symbols(obe_system.H_symbolic)
    layout = PackedHermitianLayout(int(obe_system.H_symbolic.shape[0]))
    parameter_graph = lower_parameter_graph(params)
    tuple_value_names = {
        name for name, value in params.base_parameters.items() if isinstance(value, tuple)
    }
    hamiltonian_plan = lower_hamiltonian_upper_triangle(
        obe_system.H_symbolic,
        params.slot_index_by_name,
        tuple_value_names=tuple_value_names,
        representation=hamiltonian_representation,
    )
    dense_c_array = np.asarray(obe_system.C_array, dtype=np.complex128)
    structured_jumps, source_decay_rates, incoming_transfers_by_target = _lower_structured_jumps(
        dense_c_array
    )
    prepared = PreparedLindbladProblem(
        obe_system=obe_system,
        parameters=params,
        layout=layout,
        parameter_graph=parameter_graph,
        hamiltonian_plan=hamiltonian_plan,
        dense_c_array=dense_c_array,
        structured_jumps=structured_jumps,
        source_decay_rates=source_decay_rates,
        incoming_transfers_by_target=incoming_transfers_by_target,
        blas_config=_locate_scipy_openblas() if backend == "rust" else None,
        backend=backend,
    )
    if backend == "rust":
        from ..centrex_tlf_rust import prepare_lindblad_problem_py

        prepared.rust_plan = prepare_lindblad_problem_py(prepared.to_payload())
    elif backend != "python":
        raise ValueError(f"unsupported backend {backend!r}")
    return prepared
