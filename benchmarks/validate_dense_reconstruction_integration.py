"""Validate/time the integrated CPU reconstruction on the saved dense fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from check_dense_construction_reconstruction import density, time_call
from scipy import sparse
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.dense import _factor
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_construction_reconstruction_results"


def main():
    historical = json.loads((OUT / "results.json").read_text())
    results = []
    with threadpool_limits(limits=1):
        for point in range(3):
            with np.load(HERE / f"r2_f4_dense_threading_results/point_{point}_inputs.npz") as data:
                matrix, initial, n = data["L"], data["packed"].T.copy(), int(data["n"])
            factor = _factor(sparse.csr_matrix(matrix), PackedHermitianLayout(n), initial)
            sink = np.zeros((1, n * n))
            sink[0, factor.sinks[0]] = 1
            ground = np.flatnonzero(np.any(initial[:, :n], axis=0))
            psi = np.zeros(n, complex)
            psi[ground[:2]] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
            initial = np.concatenate(
                (initial, sink, factor.layout.pack(np.outer(psi, psi.conj()))[None])
            )
            for nt in [2, 65]:
                times = np.linspace(0, 350e-6, nt)
                reference = density(factor, initial, times, n)
                actual = factor.density_matrices(initial, times)
                np.testing.assert_allclose(actual, reference, atol=1e-13, rtol=1e-12)
                samples = []
                for _ in range(5):
                    _, elapsed = time_call(factor.density_matrices, initial, times)
                    samples.append(elapsed)
                before = next(
                    row for row in historical["reconstruction"]
                    if row["point"] == point and row["times"] == nt
                )["medians"]["density_original"]
                median = float(np.median(samples))
                row = dict(point=point, times=nt, seconds=samples, median_seconds=median,
                           historical_original_seconds=before, speedup=before / median,
                           max_complex_error=float(np.max(abs(actual - reference))))
                results.append(row)
                print(f"Point {point}, {nt} times: {median:.6f}s, {before / median:.2f}x", flush=True)
    (OUT / "library_validation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
