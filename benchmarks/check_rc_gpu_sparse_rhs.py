import sys
sys.path.insert(0,'benchmarks')
from rc_gpu_sparse_methods import load,METHODS
from rc_gpu_sparse_job import rhs_check
c,_,_=load()
for m in METHODS:
    tests=rhs_check(c,m)
    print(m,'validated',len(tests),'max relative norm',max(t['relative_frobenius'] for t in tests),flush=True)
