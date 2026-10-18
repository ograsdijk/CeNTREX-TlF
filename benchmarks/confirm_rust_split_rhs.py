"""Paired timing confirmation using the already compiled split benchmarks."""

import importlib.util
import json

import bench_rust_specialized_rhs as base
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    for name in ["rc", "r2", "r0", "q1"]:
        base.emit("Paired split-function confirmation: " + name)
        prepared, rho0, _ = base.model(name)
        module = "_rhs_split_" + name
        path = base.CACHE / ("split_" + name) / (module + ".pyd")
        spec = importlib.util.spec_from_file_location(module, path)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        payload, y = prepared.to_payload(), prepared.layout.pack(rho0)
        # Warm both paths, then alternate order to reduce drift bias.
        for mode in [0, 6]:
            native.trajectory(payload, y, mode, 10e-6, 1e-11, 1e-9)
        measurements = {0: [], 6: []}
        outputs = {}
        for k in range(5):
            for mode in [0, 6] if k % 2 == 0 else [6, 0]:
                result = native.trajectory(payload, y, mode, 10e-6, 1e-11, 1e-9)
                measurements[mode].append(result[1])
                outputs[mode] = np.asarray(result[0])
        rhs = {0: [], 6: []}
        for k in range(3):
            for mode in [0, 6] if k % 2 == 0 else [6, 0]:
                rhs[mode].extend(native.rhs_benchmark(payload, y, mode, 2000))
        result = dict(
            system=name,
            duration_s=10e-6,
            paired_samples_s=measurements,
            baseline_median_s=float(np.median(measurements[0])),
            split_median_s=float(np.median(measurements[6])),
            wall_reduction_percent=float(
                100 * (1 - np.median(measurements[6]) / np.median(measurements[0]))
            ),
            rhs_median_s={key: float(np.median(value)) for key, value in rhs.items()},
            max_packed_difference=float(abs(outputs[0] - outputs[6]).max()),
        )
        (base.RESULTS / (name + "_split_paired.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
