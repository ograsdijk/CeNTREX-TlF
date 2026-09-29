"""RHS-only block-diagonal comparison for identical-parameter initial bases."""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import sparse

from bench_shared_step_batch import RESULTS, prepare_model, write_row
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["two_level", "r0", "r2_compact"], required=True)
    parser.add_argument("--initials", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    model = prepare_model(args.system)
    evaluator = create_lindblad_rhs_evaluator_py(model["prepared"].rust_plan, "expanded_sparse")
    start = time.perf_counter()
    rows, cols, data = evaluator.jacobian_packed_sparse_py(0., method="analytic")
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(model["n"] ** 2,) * 2)
    assembly = time.perf_counter() - start
    for count in args.initials:
        if count > len(model["basis"]):
            continue
        state = np.zeros((count, model["n"] ** 2), dtype=float)
        for initial, index in enumerate(model["basis"][:count]):
            state[initial, index] = 1.
        repetitions = max(10, min(1000, 4000 // count))
        start = time.perf_counter()
        for _ in range(repetitions):
            independent = np.stack([matrix @ row for row in state])
        independent_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(repetitions):
            matrix_batch = (matrix @ state.T).T
        csr_matmat_seconds = time.perf_counter() - start
        start = time.perf_counter()
        block = sparse.block_diag([matrix] * count, format="csr")
        block_assembly = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(repetitions):
            result = block @ state.ravel()
        block_seconds = time.perf_counter() - start
        write_row(RESULTS / "block_diag_microbench.csv", dict(
            system=args.system, initial_count=count, repetitions=repetitions,
            matrix_assembly_seconds=assembly, block_assembly_seconds=block_assembly,
            independent_csr_seconds=independent_seconds, block_csr_seconds=block_seconds,
            csr_matmat_seconds=csr_matmat_seconds,
            speedup=independent_seconds / block_seconds,
            max_abs=float(np.max(np.abs(independent.ravel() - result))),
            max_abs_matmat=float(np.max(np.abs(independent - matrix_batch))),
            base_nnz=matrix.nnz, block_nnz=block.nnz,
            block_memory_bytes=block.data.nbytes + block.indices.nbytes + block.indptr.nbytes,
        ))
        print(args.system, count, independent_seconds / block_seconds, flush=True)


if __name__ == "__main__":
    main()
