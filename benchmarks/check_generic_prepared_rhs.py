"""Check reusable prepared-data kernels after changing drive parameters."""

import copy
import importlib.util
import json

import bench_generic_prepared_rhs as bench
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    spec = importlib.util.spec_from_file_location(
        bench.MODULE, bench.FOLDER / (bench.MODULE + ".pyd")
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    records = []
    with threadpool_limits(limits=1):
        for name in ["r0", "rc", "r2", "q1"]:
            prepared, _, gates = bench.base.model(name)
            n = prepared.layout.n
            rng = np.random.default_rng(441)
            a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            a = a @ a.conj().T
            y = prepared.layout.pack(a / np.trace(a))
            for scale, detuning in [(0.0, -3.7e6), (0.3, 0.0), (2.0, 5.1e6)]:
                for t in np.array([125, 375, 625, 875]) * 1e-9:
                    payload = copy.deepcopy(prepared.to_payload())
                    graph = payload["parameter_graph"]
                    changed = {}
                    for idx, slot in enumerate(graph["slot_names"]):
                        if idx >= len(graph["base_values"]):
                            continue
                        entry = graph["base_values"][idx]
                        if slot.startswith("Ωt"):
                            entry["re"] *= scale
                            changed[slot] = entry["re"]
                        if slot.startswith("δ"):
                            entry["re"] = 2 * np.pi * detuning
                            changed[slot] = entry["re"]
                        if slot in gates:
                            gate = bench.base.driver.switch_values(float(t))[
                                0 if slot.endswith("1") else 1
                            ]
                            entry["re"] = gate if slot.startswith("PA") else 1 - gate
                    assert changed, "No runtime parameters changed"
                    reference = np.asarray(native.rhs_check(payload, y, float(t), 0))
                    error = 0.0
                    for mode in range(16, 80):
                        actual = np.asarray(native.rhs_check(payload, y, float(t), mode))
                        error = max(
                            error,
                            float(abs(actual - reference).max() / max(abs(reference).max(), 1.0)),
                        )
                    assert error < 1e-10, (name, scale, detuning, t, error)
                    records.append(
                        dict(
                            system=name,
                            rabi_scale=scale,
                            detuning_Hz=detuning,
                            time_s=float(t),
                            changed_parameters=changed,
                            normalized_error=error,
                        )
                    )
            print(name, "runtime-change checks passed", flush=True)
    (bench.base.RESULTS / "generic_prepared_runtime_checks.json").write_text(
        json.dumps(dict(passed=True, modes_per_check=64, checks=records), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
