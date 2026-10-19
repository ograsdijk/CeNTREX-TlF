import sys,faulthandler
faulthandler.dump_traceback_later(20,repeat=True)
print(sys.executable,flush=True)
import cupy as cp
print('Imported',cp.__version__,flush=True)
print(cp.cuda.runtime.getDeviceProperties(0)['name'],flush=True)
k=cp.RawKernel('extern "C" __global__ void twice(double* x){x[threadIdx.x]*=2;}', 'twice')
print('RawKernel created',flush=True)
a=cp.ones(4)
print('Array created',flush=True)
k((1,),(4,),(a,))
print('Launched',flush=True)
print(a.get(),flush=True)
