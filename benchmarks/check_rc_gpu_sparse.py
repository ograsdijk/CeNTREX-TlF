import sys,time,numpy as np,cupy as cp
sys.path.insert(0,'benchmarks')
from rc_gpu_sparse_methods import *
c,_,_=load()
for method in METHODS:
    try:
        g=GPU(c,method,12,146e-6,147e-6,128,pattern='legacy')
        g.rhs(g.y,g.ks[0],0.)
        y=g.download(g.ks[0]); r=numpy_rhs(g.f,c,g.initial_host,g.det_host,146e-6)
        print(method,'RHS diff',np.max(np.abs(y-r)), 'rel-norm',np.linalg.norm(y-r)/np.linalg.norm(r),flush=True)
        g.capture()
        y,t,end=g.evolve(1)
        ref=np.load('benchmarks/rc_gpu_ode_screen_results/batch_12_substeps_128.npz')['rho']
        print('ODE',t,end,diagnostics(y,ref)['max_abs'],flush=True)
    except Exception:
        import traceback; traceback.print_exc()
