"""Compare vectorized packing with the scalar implementation on RC/R2/R0 sizes."""
import json
import time
from pathlib import Path

import numpy as np

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

OUT=Path(__file__).resolve().parent/"cuda_remaining_results"


def main():
    OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(4)
    rows=[]
    for model,n in [("RC",70),("R2",40),("R0",18)]:
        layout=PackedHermitianLayout(n)
        matrices=rng.normal(size=(1024,n,n))+1j*rng.normal(size=(1024,n,n))
        matrices+=matrices.conj().swapaxes(-1,-2).copy()
        tick=time.perf_counter()
        scalar=np.stack([layout.pack(matrix) for matrix in matrices])
        scalar_seconds=time.perf_counter()-tick
        samples=[]
        for _ in range(3):
            tick=time.perf_counter()
            packed=layout.pack_batch(matrices)
            samples.append(time.perf_counter()-tick)
            np.testing.assert_array_equal(packed,scalar)
        row=dict(model=model,n=n,batch=1024,scalar_seconds=scalar_seconds,
                 vectorized_seconds=samples,median_vectorized_seconds=float(np.median(samples)),
                 max_difference=float(np.max(abs(packed-scalar))))
        rows.append(row)
        print(row,flush=True)
    (OUT/"batch_packing_integration.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
