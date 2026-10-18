"""Check isolated contributions of already compiled separate-RHS benchmarks.

Ground-only densities have zero dissipator; identity and H-squared densities
commute with H. These probe each contribution without recompiling the candidate.
"""

import copy
import importlib.util
import json

import bench_rust_specialized_rhs as base
import numpy as np
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.reference_dense import (
    apply_dense_dissipator_reference,
    evaluate_hamiltonian_reference,
)


def check(name):
    prepared, _, gates = base.model(name)
    module = "_rhs_separate_" + name
    spec = importlib.util.spec_from_file_location(
        module, base.CACHE / ("separate_" + name) / (module + ".pyd")
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    n = prepared.layout.n
    indices = [
        i for i, q in enumerate(prepared.obe_system.QN) if q.largest.electronic_state.name == "X"
    ]
    rng = np.random.default_rng(994)
    a = rng.normal(size=(len(indices), len(indices))) + 1j * rng.normal(
        size=(len(indices), len(indices))
    )
    a = a @ a.conj().T
    ground = np.zeros((n, n), complex)
    ground[np.ix_(indices, indices)] = a / np.trace(a)
    rows = []
    for t in np.array([125, 375, 625, 875]) * 1e-9:
        test = copy.copy(prepared)
        test.parameter_graph = copy.deepcopy(prepared.parameter_graph)
        for g in gates:
            idx = test.parameter_graph["slot_names"].index(g)
            gate = base.driver.switch_values(float(t))[0 if g.endswith("1") else 1]
            test.parameter_graph["base_values"][idx]["re"] = (
                gate if g.startswith("PA") else 1 - gate
            )
        h = evaluate_hamiltonian_reference(test, float(t))
        h2 = h @ h
        for label, rho in [
            ("H_only_ground_coherences", ground),
            ("D_only_identity", np.eye(n, dtype=complex) / n),
            ("D_only_H_squared", h2 / np.trace(h2)),
        ]:
            comm = -1j * (h @ rho - rho @ h)
            diss = apply_dense_dissipator_reference(prepared.dense_c_array, rho)
            expected, absent = (comm, diss) if label.startswith("H") else (diss, comm)
            actual = np.asarray(
                native.rhs_check(test.to_payload(), prepared.layout.pack(rho), float(t), 8)
            )
            scale = max(float(abs(expected).max()), 1.0)
            error = float(abs(actual - prepared.layout.pack(expected)).max() / scale)
            residual = float(abs(absent).max() / scale)
            rows.append(
                dict(
                    test=label,
                    time_s=float(t),
                    normalized_error=error,
                    absent_contribution_residual=residual,
                )
            )
            assert error < 1e-10 and residual < 1e-10, rows[-1]
    result = dict(system=name, passed=True, checks=rows)
    (base.RESULTS / (name + "_separate_parts_check.json")).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(name, "isolated contribution checks passed", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for name in ["r0", "rc", "r2", "q1"]:
            check(name)
