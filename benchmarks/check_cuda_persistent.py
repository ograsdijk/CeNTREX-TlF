"""Trajectory-local persistent Tsit5 prototype, benchmark only."""
from __future__ import annotations

import argparse
import gc
import json
import time

import check_cuda_remaining as bench
import numpy as np

from centrex_tlf.lindblad import cuda

cp = bench.cp
OUT = bench.OUT

KERNEL = r"""
__device__ __forceinline__ void local_read(const double* y,int i,int j,int n,int U,double& r,double& v){
 int q=pk(i,j,n);r=y[q];v=(i>j?-1.:1.)*y[U+q];
}
extern "C" __global__ void persistent(double* y,double* ks,const C* params,const C* bases,const C* dbases,
 const int* ptr,const int* col,const int* hrows,const int* hcols,const int* rows,const int* cols,
 const double* rates,const int* tp,const int* tc,const double* tr,const double* grid,const int* resets,
 int B,int n,int E,int K,int start,int stop,int* error){
 int b=blockIdx.x,U=n*(n+1)/2;long P=(long)B*U;
 extern __shared__ double memory[];
 double* base=memory;double* stage=base+2*U;double* hr=stage+2*U;double* hi=hr+E;
 double* diagonal=hi+E;C* coeff=(C*)(diagonal+n);
 for(int q=threadIdx.x;q<2*U;q+=blockDim.x)base[q]=y[(q<U?q:(q-U))*B+b+(q<U?0:P)];
 __syncthreads();
 for(int step=start;step<stop;step++){
  double t=grid[2*step],h=grid[2*step+1];
  for(int q=threadIdx.x;q<2*U;q+=blockDim.x)stage[q]=base[q];
  __syncthreads();
  for(int s=0;s<7;s++){
   bool reuse=s==0 && !resets[step];
   if(reuse){
    for(int q=threadIdx.x;q<2*U;q+=blockDim.x)ks[((long)b)*2*U+q]=ks[((long)6*B+b)*2*U+q];
   }else{
    if(threadIdx.x==0){eval_common(params,B,b,t+PS_C[s]*h,t+.5*h,coeff,1,error);eval_variable(params,B,b,t+PS_C[s]*h,t+.5*h,coeff,1,error);}
    __syncthreads();
    for(int e=threadIdx.x;e<E;e+=blockDim.x){C z(0.,0.);bool lower=hrows[e]>hcols[e];
     for(int k=0;k<K;k++){C a=bases[k*E+e]*coeff[k];z+=lower?conj(a):a;}hr[e]=z.real();hi[e]=z.imag();}
    for(int i=threadIdx.x;i<n;i+=blockDim.x){C z(0.,0.);for(int k=0;k<K;k++)z+=dbases[k*n+i]*coeff[k];diagonal[i]=z.real();}
    __syncthreads();
    for(int q=threadIdx.x;q<U;q+=blockDim.x){
     int i=rows[q],j=cols[q];double ar=0.,ai=0.,jr=0.,ji=0.;
     for(int k=ptr[i];k<ptr[i+1];k++){double u,v;local_read(stage,col[k],j,n,U,u,v);ar+=hr[k]*u-hi[k]*v;ai+=hr[k]*v+hi[k]*u;}
     for(int k=ptr[j];k<ptr[j+1];k++){double u,v;local_read(stage,col[k],i,n,U,u,v);jr+=hr[k]*u-hi[k]*v;ji+=hr[k]*v+hi[k]*u;}
     double u=stage[q],v=stage[U+q],d=diagonal[i]-diagonal[j],loss=-.5*(rates[i]+rates[j]);
     double r=ai+ji+d*v+loss*u,w=jr-ar-d*u+loss*v;
     if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*stage[pk(tc[k],tc[k],n)];w=0.;}
     long a=((long)s*B+b)*2*U;ks[a+q]=r;ks[a+U+q]=w;
    }
   }
   __syncthreads();
   if(s<6){
    for(int q=threadIdx.x;q<2*U;q+=blockDim.x){double z=0.;
     for(int k=0;k<=s;k++)if(PS_A[(s+1)*7+k]!=0.)z+=PS_A[(s+1)*7+k]*ks[((long)k*B+b)*2*U+q];
     stage[q]=base[q]+h*z;}
    __syncthreads();
   }
  }
  for(int q=threadIdx.x;q<2*U;q+=blockDim.x)base[q]=stage[q];
  __syncthreads();
 }
 for(int q=threadIdx.x;q<2*U;q+=blockDim.x)y[(q<U?q:(q-U))*B+b+(q<U?0:P)]=base[q];
}
"""


def build(g, source, variant="fp64"):
    arrays = []
    weights = np.zeros((7, 7))
    for i, row in enumerate(cuda.TS_A):
        weights[i, :len(row)] = row
    for name, values in [("PS_C", cuda.TS_C), ("PS_A", weights.reshape(-1))]:
        arrays.append(f"__device__ __constant__ double {name}[]={{" + ",".join(f"{v:.17g}" for v in values) + "};")
    kernel = KERNEL
    if variant == "fp32_products":
        begin = kernel.index("int i=rows[q],j=cols[q];double ar")
        end = kernel.index("long a=((long)s*B+b)*2*U", begin)
        body = kernel[begin:end].replace("double ar=0.,ai=0.,jr=0.,ji=0.;", "float ar=0.,ai=0.,jr=0.,ji=0.;")
        for column in ["j", "i"]:
            body = body.replace(f"double u,v;local_read(stage,col[k],{column},n,U,u,v);",
                                f"double du,dv;local_read(stage,col[k],{column},n,U,du,dv);float u=du,v=dv;")
        body = body.replace("hr[k]", "(float)hr[k]").replace("hi[k]", "(float)hi[k]")
        kernel = kernel[:begin]+body+kernel[end:]
    code = source + "\n" + "\n".join(arrays) + kernel
    module = bench.RawModule(code=code, options=("--std=c++17",))
    function = module.get_function("persistent")
    shared = 8 * (4 * g.n * (g.n + 1) // 2 + 2 * g.E + g.n + 2 * g.K)
    function.max_dynamic_shared_size_bytes = shared
    scratch = cp.empty((7, g.B, g.n*(g.n+1)), cp.float64)
    return function, scratch, shared, code


def run(g, function, scratch, shared, span, chunk, threads):
    cp.copyto(g.y, g.initial)
    g.error.fill(0)
    cp.cuda.get_current_stream().synchronize()
    tick = time.perf_counter()
    for start in range(0, g.steps, chunk):
        function((g.B,), (threads,), (
            g.y, scratch, g.params, g.bases, g.diagonal_bases, g.ptr, g.col,
            g.hrows, g.hcols, g.rows, g.cols, g.rates, g.tp, g.tc, g.tr,
            g.grid, g.resets, *map(np.int32, (g.B, g.n, g.E, g.K, start, min(g.steps,start+chunk))), g.error,
        ), shared_mem=shared)
    host = g.y.get()
    return host, time.perf_counter()-tick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["RC", "R2_EOM"])
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--threads", type=int, nargs="+", default=[128,256])
    parser.add_argument("--chunks", type=int, nargs="+", default=[2,16])
    parser.add_argument("--arithmetic", choices=["fp64","fp32_products"], default="fp64")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    results = []
    for system in args.systems:
        factory = bench.previous.rc_fixture if system == "RC" else bench.previous.r2_fixture
        prepared, rhos, _, ids, pars0, names, span, dt, reference, _ = factory()
        select = np.arange(args.batch) % len(ids)
        g, answer, source = bench.setup(prepared,rhos[ids[select]],pars0[select],names,span,dt,"base_direct")
        function, scratch, shared, code = build(g, source, args.arithmetic)
        (OUT/f"{system}_persistent_{args.arithmetic}.cu").write_text(code)
        reference = reference[select % len(reference)]
        baseline = prepared.layout.unpack_batch(answer.values)
        rows = {}
        for threads in args.threads:
            for chunk in args.chunks:
                key=f"threads{threads}_steps{chunk}"
                print(f"{system} persistent {key}, shared={shared}", flush=True)
                host, _ = run(g,function,scratch,shared,span,chunk,threads)
                matrices = bench.internal_density(g,host,prepared.layout)
                errors, passed=bench.clean_error(matrices, reference)
                difference=float(np.max(abs(matrices-baseline)))
                rows[key]=dict(accuracy=errors,passed=passed,max_difference_from_baseline=difference,timings=[])
                print(f"  pass={passed}, baseline difference={difference:.4g}",flush=True)
                if not passed:
                    continue
                for _ in range(args.rounds):
                    _,seconds=run(g,function,scratch,shared,span,chunk,threads)
                    rows[key]["timings"].append(seconds)
                print(f"  median {np.median(rows[key]['timings']):.6f}s",flush=True)
        basetimes=[]
        for _ in range(args.rounds):
            _,seconds=bench.previous.replay(g,span)
            basetimes.append(seconds)
        results.append(dict(system=system,batch=args.batch,shared_bytes=shared,arithmetic=args.arithmetic,
                            kernel_attributes=function.attributes,baseline_times=basetimes,variants=rows))
        (OUT/f"persistent_{system}_{args.arithmetic}_b{args.batch}.json").write_text(json.dumps(results[-1:],indent=2))
        del g,scratch,function
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


if __name__=="__main__":
    main()
