"""Check full-density CUDA reconstruction with an archived R(2) eigensystem."""

import os

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
from pathlib import Path
import json
import numpy as np
import scipy.linalg as la
from centrex_tlf.lindblad import DenseLindbladPropagator
from centrex_tlf.lindblad.state_layout import PackedHermitianLayout
from centrex_tlf.lindblad.dense import _torch_device

_torch_device("cuda")
import run_r2_gpu_reuse as old

here = Path(__file__).resolve().parent
source = here / "r2_gpu_reuse_results/matched_cross_v1"
a = dict(np.load(source / "inputs.npz"))
f = dict(np.load(source / "factors.npz"))
n = int(a["n"])
keep = f["keep"]
sinks = f["sinks"]
a.update(keep=keep, sinks=sinks)
excluded = np.setdiff1d(np.arange(n * n), np.r_[keep, sinks])
p = DenseLindbladPropagator(
    PackedHermitianLayout(n),
    keep,
    sinks,
    excluded,
    f["w"],
    f["basis"],
    la.lu_factor(f["basis"]),
    f["real"],
    f["positive"],
    f["feed"],
    {},
)
times = 0.02 / np.linspace(50.0, 250.0, 21)
initial = a["packed"].T
actual = p.density_matrices(initial, times, evolution_device="cuda")
# Reuse the saved reference implementation and saved modal coefficients.
reference = old.project(
    f,
    a,
    np.tile(times, len(initial)),
    np.repeat(np.arange(len(initial)), len(times)),
    512,
    np,
    True,
)
reference = reference.T.reshape(actual.shape)
error = float(abs(actual - reference).max())
assert error < 1e-12
(here / "cuda_library_static_validation.json").write_text(
    json.dumps(
        dict(
            n=n,
            initials=len(initial),
            times=len(times),
            cases=len(initial) * len(times),
            max_complex_entry_difference=error,
            source=str(source),
            new_eigendecompositions=0,
        ),
        indent=2,
    )
)
print(n, actual.shape, error)
