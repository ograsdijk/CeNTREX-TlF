"""Profile existing static dense stages and exact reconstruction prototypes.

Uses saved R(2) F'=4 matrices and the current public/native implementations.
No production solver changes or ODE scans.
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.dense import _factor, _phi, _StaticLiouvillian
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_construction_reconstruction_results"


def reconstruct(factor, initial, times):
    """Solve initial coefficients once; reuse modes for sink reconstruction."""
    C, s0 = factor._coefficients(initial)
    values = np.zeros((C.shape[1], len(times), factor.layout.packed_len))
    real, pos = factor.real, factor.positive
    pair_C = C[pos] - 1j * C[pos + 1]
    for start in range(0, len(times), 32):
        ts = times[start : start + 32]
        modes = np.empty((len(factor.w), C.shape[1], len(ts)))
        for order in [0, 1] if len(factor.sinks) else [0]:
            mul = _phi(factor.w, ts, order)
            modes[real] = C[real, :, None] * mul[real, None, :].real
            pair = pair_C[:, :, None] * mul[pos, None, :]
            modes[pos], modes[pos + 1] = pair.real, -pair.imag
            basis = factor.basis if order == 0 else factor.feed_basis
            indices = factor.keep if order == 0 else factor.sinks
            block = (basis @ modes.reshape(len(factor.w), -1)).reshape(
                len(indices), C.shape[1], len(ts)
            ).transpose(1, 2, 0)
            if order:
                block += s0.T[:, None, :]
            values[:, start : start + len(ts), indices] = block
    return values


def unpack(packed, n):
    result = np.empty((*packed.shape[:-1], n, n), complex)
    di = np.arange(n)
    ui, uj = np.triu_indices(n, 1)
    result[..., di, di] = packed[..., :n]
    upper = packed[..., n::2] + 1j * packed[..., n + 1 :: 2]
    result[..., ui, uj] = upper
    result[..., uj, ui] = upper.conj()
    return result


def density(factor, initial, times, n):
    return unpack(reconstruct(factor, initial, times), n)


def time_call(fn, *args):
    start = time.perf_counter()
    answer = fn(*args)
    return answer, time.perf_counter() - start


def main():
    OUT.mkdir(exist_ok=True)
    results = {"reconstruction": [], "construction": []}
    rng = np.random.default_rng(123)
    with threadpool_limits(limits=1):
        for point in range(3):
            with np.load(HERE / f"r2_f4_dense_threading_results/point_{point}_inputs.npz") as data:
                matrix, initial, n = data["L"], data["packed"].T.copy(), int(data["n"])
            factor, factor_s = time_call(
                _factor, sparse.csr_matrix(matrix), PackedHermitianLayout(n), initial
            )
            print(f"Point {point}: factor {factor_s:.3f}s, active {len(factor.w)}", flush=True)
            # Include initially occupied sinks and complex initial coherences.
            if len(factor.sinks):
                extra = np.zeros((1, n * n))
                extra[0, factor.sinks[0]] = 1
                initial = np.concatenate((initial, extra))
            # A PSD complex superposition within the supported driven block.
            ground = np.flatnonzero(np.any(initial[:, :n], axis=0))
            ground = [i for i in ground if i in factor.keep]
            psi = np.zeros(n, complex)
            psi[ground[:2]] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
            initial = np.concatenate((initial, factor.layout.pack(np.outer(psi, psi.conj()))[None]))
            for nt in [2, 65]:
                times = np.linspace(0, 350e-6, nt)
                reference = factor.density_matrices(initial, times)
                candidate = unpack(reconstruct(factor, initial, times), n)
                error = float(np.max(abs(candidate - reference)))
                np.testing.assert_allclose(candidate, reference, rtol=1e-8, atol=1e-10)
                packed = factor.evaluate(initial, times)
                timings = {name: [] for name in ["packed_original", "packed_prototype", "unpack_original", "unpack_vectorized", "density_original", "density_prototype"]}

                def original_unpack(packed=packed, n=n, nt=nt, factor=factor):
                    answer = np.empty((*packed.shape[:2], n, n), complex)
                    for i in range(len(packed)):
                        for j in range(nt):
                            answer[i, j] = factor.layout.unpack(packed[i, j])
                    return answer

                methods = {
                    "packed_original": partial(factor.evaluate, initial, times),
                    "packed_prototype": partial(reconstruct, factor, initial, times),
                    "unpack_original": original_unpack,
                    "unpack_vectorized": partial(unpack, packed, n),
                    "density_original": partial(factor.density_matrices, initial, times),
                    "density_prototype": partial(density, factor, initial, times, n),
                }
                for _ in range(5):
                    for name in rng.permutation(list(methods)):
                        _, elapsed = time_call(methods[name])
                        timings[name].append(elapsed)
                row = dict(point=point, n=n, initial_count=len(initial), times=nt,
                           factor_seconds=factor_s, max_complex_error=error,
                           timings=timings,
                           medians={key: float(np.median(value)) for key, value in timings.items()})
                results["reconstruction"].append(row)
                print(f"  {len(initial)} initials x {nt} times: {row['medians']}", flush=True)
                (OUT / "results.json").write_text(json.dumps(results, indent=2))

        sys.path.insert(0, str(HERE.parent / "reports/r2_f4_earth_field/obe"))
        import scan

        _, prepared, _, rabis, _, _, _ = scan.build(0.)
        extractor = _StaticLiouvillian(prepared.to_payload(), "expanded_sparse", prepared.rust_plan)
        for detuning in [6.8, 28.2, 80.]:
            params = {"detuning": 2 * np.pi * 1e6 * detuning, "rabi": rabis[1]}
            samples = []
            for _ in range(9):
                matrix, elapsed = time_call(extractor.matrix, params)
                samples.append(elapsed)
            results["construction"].append(dict(detuning_MHz=detuning, seconds=samples,
                                                median_seconds=float(np.median(samples)), nnz=matrix.nnz))
            print(f"Generator {detuning:g} MHz: {np.median(samples):.6f}s", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
