"""Additional CUDA experiments, isolated from production solver code."""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import check_cuda_prepared_assembly as previous
import numpy as np

from centrex_tlf import lindblad
from centrex_tlf.lindblad import cuda

cp = previous.cp
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cuda_remaining_results"
RawModule = cp.RawModule
Integrator = cuda._Integrator

HYBRID = r"""
__global__ void direct_rhs(const double* y,double* out,const int* ptr,const int* col,const double* hr,const double* hi,
 const double* diag,const double* rates,const int* tp,const int* tc,const double* tr,const int* rows,const int* cols,
 int B,int n,int HB,const double* clock,int skip){
 if(skip&&clock[3]!=0.)return;
 const int T=BTILE,R=16;
 int j=blockIdx.y,b0=blockIdx.x*T;long P=(long)B*n*(n+1)/2;
 extern __shared__ double sm[];double* sr=sm;double* si=sm+n*T;
 for(int z=threadIdx.x;z<n*T;z+=blockDim.x){int k=z/T,b=b0+z%T;double u=0.,v=0.;
  if(b<B)readpk(y,k,j,b,B,n,P,u,v);sr[z]=u;si[z]=v;}
 __syncthreads();
 for(int z=threadIdx.x;z<R*T;z+=blockDim.x){
  int i=blockIdx.z*R+z/T,b=b0+z%T,t=z%T;if(i>j||i>=n||b>=B)continue;
  double ar=0.,ai=0.,jr=0.,ji=0.;
  for(int k=ptr[i];k<ptr[i+1];k++){int a=col[k]*T+t;long h=(long)k*HB+(HB==1?0:b);
   ar+=hr[h]*sr[a]-hi[h]*si[a];ai+=hr[h]*si[a]+hi[h]*sr[a];}
  for(int k=ptr[j];k<ptr[j+1];k++){double u,v;readpk(y,col[k],i,b,B,n,P,u,v);long h=(long)k*HB+(HB==1?0:b);
   jr+=hr[h]*u-hi[h]*v;ji+=hr[h]*v+hi[h]*u;}
  long x=pk(i,j,n)*B+b;double u=y[x],v=y[P+x],d=diag[i*B+b]-diag[j*B+b],loss=-.5*(rates[i]+rates[j]);
  double r=ai+ji+d*v+loss*u,w=jr-ar-d*u+loss*v;
  if(i==j){for(int k=tp[i];k<tp[i+1];k++)r+=tr[k]*y[pk(tc[k],tc[k],n)*B+b];w=0.;}
  out[x]=r;out[P+x]=w;
 }
}
"""


def transform(source, n, variant):
    if "hybrid" in variant:
        tile = 16 if "hybrid16" in variant else 32
        begin = source.index("__global__ void direct_rhs(")
        end = source.index("__global__ void integral_rhs(", begin)
        source = source[:begin] + HYBRID.replace("BTILE", str(tile)) + source[end:]
    if "gather" in variant:
        indices = [min(i, j) * n - min(i, j) * (min(i, j) + 1) // 2 + max(i, j)
                   for i in range(n) for j in range(n)]
        prefix = "__device__ __constant__ int GATHER[]={" + ",".join(map(str, indices)) + "};\n"
        old = "if(i>j){int t=i;i=j;j=t;}return (long)i*n-i*(i+1)/2+j;"
        assert old in source
        source = prefix + source.replace(old, "return GATHER[i*n+j];")
        if "gather_sign" in variant:
            signs = [-1 if i > j else 1 for i in range(n) for j in range(n)]
            source = "__device__ __constant__ signed char GATHER_SIGN[]={" + ",".join(map(str, signs)) + "};\n" + source
            source = source.replace("(i>j?-1.:1.)", "GATHER_SIGN[i*n+j]")
    if "fp32" in variant:
        names = ["product", "product_tile", "direct_rhs"]
        if "fp32_rhs" in variant:
            names.append("finish")
        for name in names:
            marker = f"__global__ void {name}("
            if marker not in source:
                continue
            begin = source.index(marker)
            body = source.index("{", begin)
            depth, end = 1, body + 1
            while depth:
                depth += (source[end] == "{") - (source[end] == "}")
                end += 1
            code = source[body:end]
            # FP64 pointers/storage and diagonal subtraction; FP32 arithmetic
            # for sparse products (and, in fp32_rhs, the final RHS arithmetic).
            code = code.replace("double ", "float ")
            for array in ["hr", "hi", "tr"]:
                code = re.sub(rf"\b{array}\[([^\]]+)\]", rf"(float){array}[\1]", code)
            # readpk takes double references: use temporary doubles at gathers.
            code = code.replace("float u,w;readpk(y,col[p],j,b,B,n,P,u,w);",
                                "double du,dw;readpk(y,col[p],j,b,B,n,P,du,dw);float u=du,w=dw;")
            code = code.replace("float u,v;readpk(y,col[k],j,b,B,n,P,u,v);",
                                "double du,dv;readpk(y,col[k],j,b,B,n,P,du,dv);float u=du,v=dv;")
            code = code.replace("float u,v;readpk(y,col[k],i,b,B,n,P,u,v);",
                                "double du,dv;readpk(y,col[k],i,b,B,n,P,du,dv);float u=du,v=dv;")
            # Existing tiled shared buffers retain FP64 storage.
            code = code.replace("__shared__ float ", "__shared__ double ")
            code = code.replace("float r=0.,v=0.;\n  if(b<B)readpk", "double r=0.,v=0.;\n  if(b<B)readpk")
            code = code.replace("(float)hr[h]*sr[k]", "(float)hr[h]*(float)sr[k]")
            code = code.replace("(float)hi[h]*si[k]", "(float)hi[h]*(float)si[k]")
            code = code.replace("(float)hr[h]*si[k]", "(float)hr[h]*(float)si[k]")
            code = code.replace("(float)hi[h]*sr[k]", "(float)hi[h]*(float)sr[k]")
            if name == "direct_rhs" and "fp32_product" in variant:
                code = code.replace("float u=y[x],v=y[P+x],d=", "double u=y[x],v=y[P+x],d=")
                code = code.replace("float r=ai+ji+d*v+loss*u,w=jr-ar-d*u+loss*v;",
                                    "double r=(double)ai+ji+d*v+loss*u,w=(double)jr-ar-d*u+loss*v;")
            source = source[:body] + code + source[end:]
    return source


def setup(prepared, initial, pars, names, span, dt, variant, precompute=True):
    held, sources = [], []

    class Experiment(Integrator):
        def __init__(self, *args, **kwargs):
            def compile_module(*a, **kw):
                kw["code"] = transform(kw["code"], prepared.layout.n, variant)
                sources.append(kw["code"])
                return RawModule(*a, **kw)

            cp.RawModule = compile_module
            try:
                super().__init__(*args, **kwargs)
            finally:
                cp.RawModule = RawModule
            held.append(self)

        def launch(self, name, args, count):
            if name == "direct_rhs" and "hybrid" in variant:
                tile = 16 if "hybrid16" in variant else 32
                self.kernels[name](((self.B + tile - 1) // tile, self.n, (self.n + 15) // 16),
                                   (256,), args, shared_mem=16 * self.n * tile)
            else:
                super().launch(name, args, count)

    cuda._Integrator = Experiment
    try:
        answer = lindblad.solve_lindblad_batch(
            prepared, initial, span, parameter_batch=pars, parameter_slots=names,
            solver="fixed_tsit5", evolution_device="cuda", dt=dt,
            output="full", collect_stats=True, maxiters=1000000,
            cuda_options=lindblad.CudaSolverOptions(batch_size=len(initial),
                rhs="direct" if "direct" in variant or "hybrid" in variant else "tiled",
                precompute=precompute),
        )
    finally:
        cuda._Integrator = Integrator
    return held[0], answer, sources[0]


def internal_density(g, host, layout):
    real = host[:g.P].reshape(-1, g.B)
    imag = host[g.P:2*g.P].reshape(-1, g.B)
    pp, pi = g.public_pk.get(), g.public_im.get().astype(bool)
    packed = np.empty((g.B, g.n*g.n))
    packed[:, ~pi] = real[pp[~pi]].T
    packed[:, pi] = imag[pp[pi]].T
    return layout.unpack_batch(packed)


def clean_error(matrices, reference):
    errors = previous.fixtures.diagnostics(matrices, reference)
    errors = {k: v for k, v in errors.items() if "trajectory" not in k}
    passed = errors["passes_1percent_above_cutoff"] and errors["near_zero_max_abs"] <= 1e-6
    passed = passed and errors["trace_error"] < 1e-6 and errors["min_eigenvalue"] >= -1e-7
    return errors, bool(passed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["RC", "R2_EOM"])
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=["base_tiled", "base_direct", "gather_tiled", "gather_direct", "hybrid16", "hybrid32", "gather_hybrid32", "fp32_product_tiled", "fp32_product_direct", "fp32_rhs_direct"])
    parser.add_argument("--name", default="screen")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    results = dict(device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), cases=[])
    rng = np.random.default_rng(429)
    for system in args.systems:
        factory = previous.rc_fixture if system == "RC" else previous.r2_fixture
        prepared, rhos, _, ids, pars0, names, span, dt, ref, refpath = factory()
        select = np.arange(args.batch) % len(ids)
        initial, pars = rhos[ids[select]], pars0[select]
        objects, rows = {}, {}
        for variant in args.variants:
            print(f"Preparing {system} {variant} batch={args.batch}", flush=True)
            tick = time.perf_counter()
            try:
                g, answer, source = setup(prepared, initial, pars, names, span, dt, variant)
                errors, passed = clean_error(prepared.layout.unpack_batch(answer.values), ref[select % len(ref)])
                rows[variant] = dict(accuracy=errors, passed=passed, setup_wall=time.perf_counter()-tick,
                                     stats=answer.solver_stats, timings=[])
                objects[variant] = g
                (OUT / f"{system}_{variant}.cu").write_text(source)
                print(f"  accuracy pass={passed}, relative={errors['max_relative_above_cutoff']:.6g}", flush=True)
            except Exception as exc:
                rows[variant] = dict(error=repr(exc), passed=False, timings=[])
                print(f"  FAILED: {exc}", flush=True)
        for repeat in range(args.rounds):
            for variant in rng.permutation(list(objects)):
                host, seconds = previous.replay(objects[variant], span)
                rows[variant]["timings"].append(seconds)
                matrices = internal_density(objects[variant], host, prepared.layout)
                errors, passed = clean_error(matrices, ref[select % len(ref)])
                rows[variant]["replay_accuracy"] = errors
                rows[variant]["passed"] &= passed
            print(f"{system} batch={args.batch}: round {repeat+1}/{args.rounds}", flush=True)
        case = dict(system=system, batch=args.batch, span=span, dt=dt, reference=refpath, variants=rows)
        results["cases"].append(case)
        (OUT / f"{args.name}.json").write_text(json.dumps(results, indent=2))
        print({v: (float(np.median(r["timings"])) if r["timings"] else None, r["passed"]) for v, r in rows.items()}, flush=True)
        del objects, g
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
