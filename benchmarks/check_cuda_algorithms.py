"""Full-density CUDA algorithm prototypes; no production solver modifications.

Reachability uses the union of every real/imaginary Hamiltonian basis and the
actual dissipator, without a small-element threshold. Exponential actions use
fixed-degree scaled Taylor polynomials within fourth-order CF Magnus steps.
Periodic maps are measured only for explicitly constant-envelope fixtures.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time
import types
from pathlib import Path

import check_cuda_remaining as bench
import numpy as np
import scipy.sparse as sp

from centrex_tlf import lindblad

cp = bench.cp
OUT = Path(__file__).resolve().parent / "cuda_algorithm_results"
TAG = ""


def save(name, data):
    OUT.mkdir(exist_ok=True)
    filename = Path(name)
    if TAG:
        filename = filename.with_stem(filename.stem + "_" + TAG)
    (OUT / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


def matrices(prepared):
    """Real packed generators for Re/Im of each upper-H coefficient."""
    n = prepared.layout.n
    off, diagonal, _, positions = bench.cuda._basis(prepared)
    # Unpack x into a row-major complex density: vec(rho) = T @ x.
    rows, cols, values = [], [], []
    for i in range(n):
        rows.append(i * n + i)
        cols.append(i)
        values.append(1.0)
    for i in range(n):
        for j in range(i + 1, n):
            r = prepared.layout.upper_real_index(i, j)
            rows.extend([i * n + j, j * n + i, i * n + j, j * n + i])
            cols.extend([r, r, r + 1, r + 1])
            values.extend([1.0, 1.0, 1j, -1j])
    T = sp.csr_matrix((values, (rows, cols)), shape=(n * n, n * n))
    Q = sp.diags(np.r_[np.ones(n), np.full(n * n - n, 0.5)]) @ T.conj().T
    identity = sp.eye(n, format="csr")
    components, labels = [], []
    for k in range(len(off)):
        upper = np.zeros((n, n), complex)
        mask = positions[0] < positions[1]
        upper[positions[0, mask], positions[1, mask]] = off[k, mask]
        for imag in (0, 1):
            u = upper * (1j if imag else 1.0)
            H = u + u.conj().T
            H[np.diag_indices(n)] = (diagonal[k] * (1j if imag else 1.0)).real
            L = -1j * (sp.kron(sp.csr_matrix(H), identity) - sp.kron(identity, sp.csr_matrix(H.T)))
            A = (Q @ L @ T).real.tocsr()
            A.eliminate_zeros()
            if A.nnz or (k == 0 and imag == 0):
                components.append(A)
                labels.append((k, imag))
    rates = np.asarray(prepared.source_decay_rates)
    D = sp.diags((-0.5 * (rates[:, None] + rates[None, :])).ravel()).tolil()
    for target, entries in enumerate(prepared.incoming_transfers_by_target):
        for entry in entries:
            source = entry["source"]
            D[target * n + target, source * n + source] += entry["rate"]
    components[0] = (components[0] + (Q @ D.tocsr() @ T).real).tocsr()
    components[0].eliminate_zeros()
    return components, labels


def closure(components, initial):
    support = sum(A.astype(bool).astype(np.int32) for A in components).tocsc()
    support.eliminate_zeros()
    reached = set(np.flatnonzero(np.any(initial != 0, axis=0)).tolist())
    frontier = list(reached)
    while frontier:
        j = frontier.pop()
        for i in support.indices[support.indptr[j] : support.indptr[j + 1]]:
            if int(i) not in reached:
                reached.add(int(i))
                frontier.append(int(i))
    return np.array(sorted(reached), dtype=np.int32)


SOURCE = r"""
extern "C" __global__ void sparse_rhs(const double* y,double* out,
 const int* ptr,const int* col,const int* slot,const int* imag,const double* value,
 const C* coeff,const int* varied,const double* clock,int B,int M,int skip){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(x>=(long)M*B||(skip&&clock[3]!=0.))return;
 int row=x/B,b=x%B;double z=0.;
 for(int q=ptr[row];q<ptr[row+1];q++){
  int k=slot[q];C c=coeff[k*B+(varied[k]?b:0)];
  z+=value[q]*(imag[q]?c.imag():c.real())*y[(long)col[q]*B+b];
 }out[x]=z;
}
extern "C" __global__ void gauss_coeff(const C* params,C* one,C* two,
 const double* clock,int B,int* error){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;
 double h=clock[1],t=clock[0],st=t+.5*h;
 if(b==0){eval_common(params,B,0,t+(.5-sqrt(3.)/6.)*h,st,one,B,error);
          eval_common(params,B,0,t+(.5+sqrt(3.)/6.)*h,st,two,B,error);}
 eval_variable(params,B,b,t+(.5-sqrt(3.)/6.)*h,st,one,B,error);
 eval_variable(params,B,b,t+(.5+sqrt(3.)/6.)*h,st,two,B,error);
}
extern "C" __global__ void mix_coeff(const C* a,const C* b,C* c,long N,int reverse,int scale){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=N)return;
 double p=(3.-2.*sqrt(3.))/12.,q=(3.+2.*sqrt(3.))/12.;
 c[x]=((reverse?p:q)*a[x]+(reverse?q:p)*b[x])/(double)scale;
}
extern "C" __global__ void taylor_add(const double* rhs,double* term,double* sum,
 const double* clock,long N,int degree){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=N)return;
 double v=clock[1]/degree*rhs[x];term[x]=v;sum[x]+=v;
}
"""


def sparse_arrays(components, labels, keep):
    entries = []
    for A, (slot, imag) in zip(components, labels, strict=True):
        a = A[keep][:, keep].tocoo()
        entries.extend(zip(a.row, a.col, [slot] * a.nnz, [imag] * a.nnz, a.data, strict=True))
    entries.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    a = np.asarray(entries)
    ptr = np.r_[0, np.cumsum(np.bincount(a[:, 0].astype(int), minlength=len(keep)))].astype(
        np.int32
    )
    return tuple(
        cp.asarray(v)
        for v in (
            ptr,
            a[:, 1].astype(np.int32),
            a[:, 2].astype(np.int32),
            a[:, 3].astype(np.int32),
            a[:, 4],
        )
    )


def projected(g, source, components, labels, keep, packed, span):
    module = cp.RawModule(code=source + SOURCE, options=("--std=c++17",))
    kernel = module.get_function("sparse_rhs")
    arrays = sparse_arrays(components, labels, keep)
    g.D = len(keep) * g.B
    g.initial = cp.asarray(np.ascontiguousarray(packed[:, keep].T))
    g.y = g.initial.copy()
    g.ks = [cp.empty_like(g.y) for _ in range(7)]
    g.temps = [cp.empty_like(g.y) for _ in range(2)]
    g.yn = cp.empty_like(g.y)

    def rhs(self, y, out, stage, fraction, skip=False):
        self.launch(
            "coefficients",
            (
                self.params,
                self.clock,
                self.coeff,
                self.table,
                np.int64(self.steps),
                np.int32(self.K),
                np.int32(self.B),
                np.int32(stage),
                np.float64(fraction),
                np.int32(skip),
                self.error,
            ),
            self.B,
        )
        kernel(
            ((self.D + 255) // 256,),
            (256,),
            (
                y,
                out,
                *arrays,
                self.coeff,
                self.varied,
                self.clock,
                np.int32(self.B),
                np.int32(len(keep)),
                np.int32(skip),
            ),
        )

    g.rhs = types.MethodType(rhs, g)
    g._capture(span)
    g.experimental_module, g.experimental_arrays = module, arrays
    return g


def replay(g, keep, total, span):
    cp.copyto(g.y, g.initial)
    g.clock.set(np.array([span[0], 0.0, 0.0, 0.0, 0.0, 0.0]))
    g.error.fill(0)
    cp.cuda.get_current_stream().synchronize()
    tick = time.perf_counter()
    g.advance(0, g.steps)
    host = g.y.get().reshape(len(keep), g.B).T
    elapsed = time.perf_counter() - tick
    if int(g.error.get()[0]):
        raise RuntimeError("CUDA scalar evaluation failed")
    full = np.zeros((g.B, total))
    full[:, keep] = host
    return full, elapsed


def compact_structured(g, source, prepared, keep, packed, span):
    """Compact complex upper entries, retaining the original H/dissipator split."""
    n = prepared.layout.n
    ui, uj = np.triu_indices(n)
    kept = set(keep.tolist())
    active = np.array(
        [
            i in kept
            if i == j
            else (
                prepared.layout.upper_real_index(i, j) in kept
                or prepared.layout.upper_imag_index(i, j) in kept
            )
            for i, j in zip(ui, uj, strict=True)
        ]
    )
    remap = np.full(len(ui), -1, dtype=np.int32)
    remap[active] = np.arange(active.sum())
    prefix = "__device__ __constant__ int REDUCE_MAP[]={" + ",".join(map(str, remap)) + "};\n"
    old = "long x=pk(i,j,n)*B+b;r=y[x];v=(i>j?-1.:1.)*y[P+x];"
    new = "long q=REDUCE_MAP[pk(i,j,n)];if(q<0){r=0.;v=0.;return;}long x=q*B+b;r=y[x];v=(i>j?-1.:1.)*y[P+x];"
    assert source.count(old) == 1
    source = prefix + source.replace(old, new)
    source = source.replace("long q=density_pk(i,i,n);", "long q=REDUCE_MAP[pk(i,i,n)];")
    start = source.index("__global__ void direct_rhs(")
    end = source.index("__global__ void integral_rhs(", start)
    code = source[start:end].replace("P=(long)B*n*(n+1)/2", f"P=(long)B*{active.sum()}")
    code = code.replace("P=(long)B*density_pairs(n)", f"P=(long)B*{active.sum()}")
    code = code.replace(
        "r+=tr[k]*y[pk(tc[k],tc[k],n)*B+b];",
        "{int q=REDUCE_MAP[pk(tc[k],tc[k],n)];if(q>=0)r+=tr[k]*y[(long)q*B+b];}",
    )
    source = source[:start] + code + source[end:]
    module = cp.RawModule(code=source, options=("--std=c++17",))
    g.kernels["direct_rhs"] = module.get_function("direct_rhs")
    g.options = dataclasses.replace(g.options, rhs="direct")
    g.P = int(active.sum()) * g.B
    g.D = 2 * g.P
    g.rows, g.cols = cp.asarray(ui[active], dtype=np.int32), cp.asarray(uj[active], dtype=np.int32)
    rho = prepared.layout.unpack_batch(packed)
    z = rho[:, ui[active], uj[active]].T
    g.initial = cp.asarray(np.r_[z.real.ravel(), z.imag.ravel()])
    g.y = g.initial.copy()
    g.ks = [cp.empty_like(g.y) for _ in range(7)]
    g.temps = [cp.empty_like(g.y) for _ in range(2)]
    g.yn = cp.empty_like(g.y)
    g._capture(span)
    g.experimental_module = module
    g.compact_pairs = ui[active], uj[active]
    return g


def structured_reduction(args):
    results = []
    rng = np.random.default_rng(73294)
    for name in args.systems:
        p, packed, pars, names, span, dt = fixture(name, args.batch, args.duration_us)
        components, _ = matrices(p)
        keep = closure(components, packed)
        gbase, ref, _ = bench.setup(p, packed, pars, names, span, dt, "base_tiled")
        gfull, fullsrc = allocate(p, packed, pars, names, span, dt)
        # A full-coordinate control has identical indexing machinery.
        compact_structured(gfull, fullsrc, p, np.arange(p.layout.packed_len), packed, span)
        gred, src = allocate(p, packed, pars, names, span, dt)
        compact_structured(gred, src, p, keep, packed, span)
        timings = {"current_tiled": [], "structured_full": [], "structured_reduced": []}
        checks = {}
        for repeat in range(args.repeats):
            for method in rng.permutation(list(timings)):
                g = {"current_tiled": gbase, "structured_full": gfull, "structured_reduced": gred}[
                    method
                ]
                host, elapsed = bench.previous.replay(g, span)
                if method == "current_tiled":
                    values = p.layout.pack_batch(bench.internal_density(g, host, p.layout))
                else:
                    i, j = g.compact_pairs
                    z = (host[: g.P] + 1j * host[g.P :]).reshape(-1, g.B).T
                    rho = np.zeros((g.B, g.n, g.n), complex)
                    rho[:, i, j] = z
                    rho[:, j, i] = z.conj()
                    values = p.layout.pack_batch(rho)
                timings[method].append(elapsed)
                checks[method] = compare(p.layout, values, ref.values)
            print(
                name,
                "structured reduction",
                repeat + 1,
                {k: round(v[-1], 4) for k, v in timings.items()},
                flush=True,
            )
        results.append(
            dict(
                system=name,
                batch=args.batch,
                span=span,
                real_coordinates=len(keep),
                complex_upper_entries=len(gred.compact_pairs[0]),
                timings=timings,
                checks=checks,
            )
        )
        save("structured_reduction.json", results)
        del gbase, gfull, gred, g
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return results


def compare(layout, packed, reference):
    # Compare complex entries, not the individual real and imaginary parts.
    if not np.isfinite(packed).all():
        return dict(passed=False, finite=False, reason="nonfinite density")
    got = np.array([layout.unpack(x) for x in packed])
    ref = np.array([layout.unpack(x) for x in reference])
    delta, mag = abs(got - ref), abs(ref)
    large = mag >= 1e-6
    relative = float(np.max(delta[large] / mag[large], initial=0))
    small = float(np.max(delta[~large], initial=0))
    trace = float(np.max(abs(np.trace(got, axis1=1, axis2=2) - 1)))
    eig = float(np.linalg.eigvalsh(got).min())
    return dict(
        max_abs=float(delta.max()),
        max_relative_above_1e6=relative,
        max_abs_below_1e6=small,
        trace_error=trace,
        min_eigenvalue=eig,
        passed=bool(relative <= 0.01 and small <= 1e-6 and trace <= 1e-6 and eig >= -1e-7),
    )


def fixture(name, batch, duration):
    factory = bench.previous.rc_fixture if name == "RC" else bench.previous.r2_fixture
    saved = bench.previous.fixtures.points, bench.previous.fixtures.fixture
    try:
        p, rhos, _, ids, pars, names, span, dt, _, _ = factory()
    finally:
        # The archived R2 fixture installer monkeypatches the shared RC helper.
        # Keep subsequent systems independent within this benchmark process.
        bench.previous.fixtures.points, bench.previous.fixtures.fixture = saved
    select = np.arange(batch) % len(ids)
    return (
        p,
        p.layout.pack_batch(rhos[ids[select]]),
        pars[select],
        names,
        (span[0], span[0] + duration * 1e-6),
        dt,
    )


def allocate(p, packed, pars, names, span, dt, solver="fixed_tsit5"):
    """Create/capture the library integrator without doing a complete solve."""
    from centrex_tlf.lindblad._cuda_lowering import ScalarCompiler

    graph = p.parameter_graph
    base = np.array(
        [complex(x["re"], x["im"]) if x["kind"] == "scalar" else 0 for x in graph["base_values"]]
    )
    parameters = np.broadcast_to(base, (len(packed), len(base))).copy()
    slots = [graph["slot_names"].index(name) for name in names]
    parameters[:, slots] = pars
    compiler = ScalarCompiler(graph, [c["expression"] for c in p.hamiltonian_plan["coefficients"]])
    varying = {k for k in slots if np.any(parameters[:, k] != parameters[0, k])}
    varied = np.array([bool(c.dependencies & varying) for c in compiler.coefficients])
    switches = compiler.discontinuities(parameters, span, None)
    grid, resets, _ = bench.cuda._steps(span, dt, switches, np.array([span[1]]), [], 1000000)
    sources = []
    original = cp.RawModule

    def module(*args, **kwargs):
        sources.append(kwargs["code"])
        return original(*args, **kwargs)

    cp.RawModule = module
    try:
        g = bench.Integrator(
            cp,
            p,
            compiler,
            packed,
            parameters,
            varied,
            bench.cuda._basis(p),
            grid,
            resets,
            lindblad.CudaSolverOptions(batch_size=len(packed)),
            solver,
            np.zeros(p.layout.n),
            False,
            span,
        )
    finally:
        cp.RawModule = original
    return g, sources[0]


def reduction(args):
    results = []
    rng = np.random.default_rng(8764)
    for name in args.systems:
        p, packed, pars, names, span, dt = fixture(name, args.batch, args.duration_us)
        tick = time.perf_counter()
        components, labels = matrices(p)
        keep = closure(components, packed)
        allcoords = np.arange(p.layout.packed_len, dtype=np.int32)
        structural_time = time.perf_counter() - tick
        print(f"{name}: exact reachable real coordinates {len(keep)}/{len(allcoords)}", flush=True)
        # The union closure must be invariant for EVERY scalar basis.
        excluded = np.setdiff1d(allcoords, keep)
        assert all(A[excluded][:, keep].nnz == 0 for A in components)
        gbase, answer, source = bench.setup(p, packed, pars, names, span, dt, "base_tiled")
        baseline = answer.values
        timings = {"current_tiled": [], "sparse_full": [], "sparse_reduced": []}
        probes = {}
        for method, coords in [("sparse_full", allcoords), ("sparse_reduced", keep)]:
            tick = time.perf_counter()
            g, src = allocate(p, packed, pars, names, span, dt)
            projected(g, src, components, labels, coords, packed, span)
            probes[method] = (g, coords, time.perf_counter() - tick)
        checks = {}
        for repeat in range(args.repeats):
            for method in rng.permutation(list(timings)):
                if method == "current_tiled":
                    host, elapsed = bench.previous.replay(gbase, span)
                    values = p.layout.pack_batch(bench.internal_density(gbase, host, p.layout))
                else:
                    g, coords, _ = probes[method]
                    values, elapsed = replay(g, coords, len(allcoords), span)
                timings[method].append(elapsed)
                checks[method] = compare(p.layout, values, baseline)
            print(
                name,
                "reduction round",
                repeat + 1,
                {k: round(v[-1], 4) for k, v in timings.items()},
                flush=True,
            )
        # Independently halve the reference step on the distinct batch members.
        unique = np.unique(np.c_[pars, packed], axis=0, return_index=True)[1]
        strict = lindblad.solve_lindblad_batch(
            p,
            packed[unique],
            span,
            parameter_batch=pars[unique],
            parameter_slots=names,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt / 2,
            output="full",
            maxiters=1000000,
            cuda_options=lindblad.CudaSolverOptions(batch_size=len(unique)),
        )
        refcheck = compare(p.layout, baseline[unique], strict.values)
        results.append(
            dict(
                system=name,
                batch=args.batch,
                span=span,
                dt=dt,
                coordinates=len(allcoords),
                retained=len(keep),
                preparation_seconds=structural_time,
                timings=timings,
                checks=checks,
                baseline_half_step_check=refcheck,
                prototype_setup={k: v[2] for k, v in probes.items()},
            )
        )
        save("reduction.json", results)
        del gbase, probes, g, strict
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return results


def exponential(g, source, components, labels, keep, packed, span, degree, scale, structured=True):
    if structured:
        module = cp.RawModule(code=source + SOURCE, options=("--std=c++17",))
        # Each CF4 exponential contains (a1+a2)*D = D/2. Scaling the
        # Hamiltonian coefficients alone would incorrectly double the decay.
        g.rates *= 0.5 / scale
        g.tr *= 0.5 / scale
        original_launch = g.launch
        frozen = [False]

        def launch(self, name, args, count):
            if name == "coefficients" or (name in {"build_h", "build_diag"} and frozen[0]):
                return
            original_launch(name, args, count)
            if name == "build_diag":
                frozen[0] = True

        g.launch = types.MethodType(launch, g)
    else:
        projected(g, source, components, labels, keep, packed, span)
        module = g.experimental_module
    sparse = module.get_function("sparse_rhs")
    gauss = module.get_function("gauss_coeff")
    mix = module.get_function("mix_coeff")
    add = module.get_function("taylor_add")
    one, two = cp.zeros_like(g.coeff), cp.zeros_like(g.coeff)
    term, out, total = [cp.empty_like(g.y) for _ in range(3)]
    arrays = () if structured else g.experimental_arrays

    def block(self, count):
        for _ in range(count):
            self.kernels["begin_step"](
                (1,), (1,), (self.clock, self.grid, self.resets, np.int64(self.steps))
            )
            gauss(
                ((self.B + 255) // 256,),
                (256,),
                (self.params, one, two, self.clock, np.int32(self.B), self.error),
            )
            for reverse in (0, 1):
                mix(
                    ((self.K * self.B + 255) // 256,),
                    (256,),
                    (
                        one,
                        two,
                        self.coeff,
                        np.int64(self.K * self.B),
                        np.int32(reverse),
                        np.int32(scale),
                    ),
                )
                if structured:
                    frozen[0] = False
                for _ in range(scale):
                    cp.copyto(term, self.y)
                    cp.copyto(total, self.y)
                    for m in range(1, degree + 1):
                        if structured:
                            self.rhs(term, out, 0, 0.0)
                        else:
                            sparse(
                                ((self.D + 255) // 256,),
                                (256,),
                                (
                                    term,
                                    out,
                                    *arrays,
                                    self.coeff,
                                    self.varied,
                                    self.clock,
                                    np.int32(self.B),
                                    np.int32(len(keep)),
                                    np.int32(0),
                                ),
                            )
                        add(
                            ((self.D + 255) // 256,),
                            (256,),
                            (out, term, total, self.clock, np.int64(self.D), np.int32(m)),
                        )
                    cp.copyto(self.y, total)
            self.kernels["end_step"]((1,), (1,), (self.clock, np.int32(0)))

    g.block = types.MethodType(block, g)
    g._capture(span)
    g.experimental_module = module
    g.exponential_buffers = one, two, term, out, total
    return g


def magnus(args):
    results = []
    for name in args.systems:
        p, packed, pars, names, span, dt = fixture(name, args.batch, args.duration_us)
        components, labels = matrices(p)
        keep = closure(components, packed)
        # Refine only the distinct physical points, then expand for comparison.
        unique, inverse = np.unique(
            np.c_[pars, packed], axis=0, return_index=False, return_inverse=True
        )
        ref = lindblad.solve_lindblad_batch(
            p,
            unique[:, len(names) :],
            span,
            parameter_batch=unique[:, : len(names)],
            parameter_slots=names,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt / 2,
            output="full",
            maxiters=1000000,
        )
        reference = ref.values[inverse]
        sampled = np.unique(np.linspace(0, len(unique) - 1, min(9, len(unique)), dtype=int))
        tighter = lindblad.solve_lindblad_batch(
            p,
            unique[sampled, len(names) :],
            span,
            parameter_batch=unique[sampled, : len(names)],
            parameter_slots=names,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt / 4,
            output="full",
            maxiters=1000000,
        )
        reference_refinement = compare(p.layout, ref.values[sampled], tighter.values)
        gbase, _, _ = bench.setup(p, packed, pars, names, span, dt, "base_tiled")
        baseline_times = [bench.previous.replay(gbase, span)[1] for _ in range(args.repeats)]
        del gbase
        rows = []
        for step_ns in args.steps_ns:
            for degree in args.degrees:
                print(f"{name} CF4 Taylor degree {degree}, step {step_ns} ns", flush=True)
                tick = time.perf_counter()
                g, source = allocate(p, packed, pars, names, span, step_ns * 1e-9)
                exponential(
                    g,
                    source,
                    components,
                    labels,
                    keep,
                    packed,
                    span,
                    degree,
                    args.scale,
                    structured=args.exponential_rhs == "structured",
                )
                setup_time = time.perf_counter() - tick
                timings = []
                for _ in range(args.repeats):
                    if args.exponential_rhs == "structured":
                        host, elapsed = bench.previous.replay(g, span)
                        values = p.layout.pack_batch(bench.internal_density(g, host, p.layout))
                    else:
                        values, elapsed = replay(g, keep, p.layout.packed_len, span)
                    timings.append(elapsed)
                check = compare(p.layout, values, reference)
                rows.append(
                    dict(
                        step_ns=step_ns,
                        degree=degree,
                        scale=args.scale,
                        rhs=args.exponential_rhs,
                        timings=timings,
                        setup_seconds=setup_time,
                        check=check,
                    )
                )
                print(rows[-1], flush=True)
                results_now = [
                    *results,
                    dict(
                        system=name,
                        batch=args.batch,
                        span=span,
                        baseline_seconds=baseline_times,
                        reference_refinement=reference_refinement,
                        rows=rows,
                    ),
                ]
                save("magnus.json", results_now)
                del g
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
        results.append(
            dict(
                system=name,
                batch=args.batch,
                span=span,
                baseline_seconds=baseline_times,
                reference_refinement=reference_refinement,
                rows=rows,
            )
        )
        save("magnus.json", results)
    return results


def periodic(args):
    results = []
    for name in args.systems:
        p, original, pars, names, _, dt = fixture(name, args.batch, 1.0)
        # Explicitly change the RC recipe to constant MW envelopes. At these
        # finite values the Gaussian rounds to exactly 1 in double precision.
        pars[:, 0] = 0.0
        if name == "RC":
            pars[:, names.index("center")] = 0.0
            pars[:, names.index("sigma_time")] = 1e6
        period = 5e-6 if name == "RC" else 0.625e-6
        span = (0.0, period * args.cycles)
        # Verify the actual compiled coefficient graph, not just its nominal
        # frequency labels, is periodic after the explicit envelope change.
        probe, _ = allocate(p, original[:1], pars[:1], names, (0.0, period), dt)
        periodicity_error = 0.0
        for fraction in (0.173, 0.431, 0.897):
            coefficients = []
            for offset in (0.0, period):
                probe.clock.set(np.array([fraction * period + offset, 0.0, 0.0, 0.0, 0.0, 0.0]))
                probe.launch(
                    "coefficients",
                    (
                        probe.params,
                        probe.clock,
                        probe.coeff,
                        probe.table,
                        np.int64(probe.steps),
                        np.int32(probe.K),
                        np.int32(1),
                        np.int32(0),
                        np.float64(0.0),
                        np.int32(0),
                        probe.error,
                    ),
                    1,
                )
                coefficients.append(probe.coeff.get()[:, 0])
            periodicity_error = max(
                periodicity_error,
                float(
                    np.max(
                        abs(coefficients[0] - coefficients[1])
                        / np.maximum(1.0, abs(coefficients[0]))
                    )
                ),
            )
        assert periodicity_error < 1e-10, periodicity_error
        del probe
        # Positive physical probes, including a coherent initial state.
        unique = np.unique(original, axis=0)
        components, _ = matrices(p)
        keep = closure(components, unique)
        for i in range(p.layout.n):
            found = False
            for j in range(i + 1, p.layout.n):
                if (
                    p.layout.upper_real_index(i, j) in keep
                    and p.layout.upper_imag_index(i, j) in keep
                    and unique[:, i].max() > 0
                    and unique[:, j].max() > 0
                ):
                    state = np.zeros(p.layout.n, complex)
                    state[i], state[j] = np.sqrt(0.5), 1j * np.sqrt(0.5)
                    unique = np.r_[unique, p.layout.pack(np.outer(state, state.conj()))[None]]
                    found = True
                    break
            if found:
                break
        B = len(original)
        initial = unique[np.arange(B) % len(unique)]
        print(
            f"{name} periodic map: {len(keep)} columns, period {period * 1e6:g} us, {args.cycles} cycles",
            flush=True,
        )
        reference = lindblad.solve_lindblad_batch(
            p,
            unique,
            span,
            parameter_batch=np.repeat(pars[:1], len(unique), axis=0),
            parameter_slots=names,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt / 2,
            output="full",
            maxiters=1000000,
        )
        check_period = (0.0, period)
        # End-to-end construction includes allocation, capture, result transfer,
        # and upload into the resident GPU map, not just the period ODE kernels.
        tick = time.perf_counter()
        M = cp.empty((len(keep), len(keep)), cp.float64)
        for lo in range(0, len(keep), args.batch):
            cols = keep[lo : lo + args.batch]
            basis = np.zeros((len(cols), p.layout.packed_len))
            basis[np.arange(len(cols)), cols] = 1.0
            answer = lindblad.solve_lindblad_batch(
                p,
                basis,
                check_period,
                parameter_batch=np.repeat(pars[:1], len(cols), axis=0),
                parameter_slots=names,
                solver="fixed_tsit5",
                evolution_device="cuda",
                dt=dt,
                output="full",
                maxiters=1000000,
                cuda_options=lindblad.CudaSolverOptions(batch_size=args.batch),
            )
            M[:, lo : lo + len(cols)] = cp.asarray(answer.values[:, keep].T)
            cp.cuda.get_current_stream().synchronize()
            print(
                f"{name} map columns {lo + len(cols)}/{len(keep)}, elapsed {time.perf_counter() - tick:.1f}s",
                flush=True,
            )
        construction = time.perf_counter() - tick

        # Power by squaring: a full density map, never a truncated Krylov basis.
        cp.cuda.get_current_stream().synchronize()
        tick = time.perf_counter()
        power = M.copy()
        powered = None
        exponent = args.cycles
        while exponent:
            if exponent & 1:
                powered = power.copy() if powered is None else power @ powered
            exponent >>= 1
            if exponent:
                power = power @ power
        cp.cuda.get_current_stream().synchronize()
        power_seconds = time.perf_counter() - tick
        input_device = cp.asarray(np.ascontiguousarray(initial[:, keep].T))
        timings = []
        for _ in range(args.repeats):
            cp.cuda.get_current_stream().synchronize()
            tick = time.perf_counter()
            values = (powered @ input_device).get().T
            timings.append(time.perf_counter() - tick)
        full = np.zeros_like(initial)
        full[:, keep] = values
        ref = reference.values[np.arange(B) % len(unique)]
        check = compare(p.layout, full, ref)
        direct_times = []
        for repeat in range(args.repeats):
            tick = time.perf_counter()
            answer = lindblad.solve_lindblad_batch(
                p,
                initial,
                span,
                parameter_batch=pars,
                parameter_slots=names,
                solver="fixed_tsit5",
                evolution_device="cuda",
                dt=dt,
                output="full",
                maxiters=1000000,
                cuda_options=lindblad.CudaSolverOptions(batch_size=args.batch),
            )
            direct_times.append(time.perf_counter() - tick)
            print(
                f"{name} direct periodic round {repeat + 1}/{args.repeats}: {direct_times[-1]:.3f}s",
                flush=True,
            )
        unique_tick = time.perf_counter()
        direct_unique = lindblad.solve_lindblad_batch(
            p,
            unique,
            span,
            parameter_batch=np.repeat(pars[:1], len(unique), axis=0),
            parameter_slots=names,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt,
            output="full",
            maxiters=1000000,
        )
        unique_seconds = time.perf_counter() - unique_tick
        # Long-duration accuracy test on the distinct physical states, without
        # charging another full 1024-trajectory timing run.
        long_cycles = max(40, args.cycles)
        long_reference = lindblad.solve_lindblad_batch(
            p,
            unique,
            (0.0, period * long_cycles),
            parameter_batch=np.repeat(pars[:1], len(unique), axis=0),
            parameter_slots=names,
            solver="fixed_tsit5",
            evolution_device="cuda",
            dt=dt / 2,
            output="full",
            maxiters=1000000,
        )
        long_power = cp.linalg.matrix_power(M, long_cycles)
        long_values = np.zeros_like(unique)
        long_values[:, keep] = (
            (long_power @ cp.asarray(np.ascontiguousarray(unique[:, keep].T))).get().T
        )
        long_check = compare(p.layout, long_values, long_reference.values)
        # The small physical-state batch is the relevant cold-cost comparator
        # when only the existing three J mixtures are needed at one RC key.
        row = dict(
            system=name,
            batch=B,
            detuning_MHz=0.0,
            period_seconds=period,
            cycles=args.cycles,
            span=span,
            constant_envelope=True,
            retained=len(keep),
            map_bytes=int(M.nbytes),
            construction_seconds=construction,
            power_seconds=power_seconds,
            reuse_timings=timings,
            direct_batch_timings=direct_times,
            physical_initial_states=len(unique),
            direct_unique_seconds=unique_seconds,
            check=check,
            direct_reference_check=compare(p.layout, direct_unique.values, reference.values),
            coefficient_periodicity_error=periodicity_error,
            long_cycles=long_cycles,
            long_check=long_check,
        )
        results.append(row)
        save("periodic.json", results)
        print(row, flush=True)
        del (
            M,
            power,
            powered,
            input_device,
            answer,
            direct_unique,
            reference,
            long_power,
            long_reference,
        )
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return results


def main():
    global TAG
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reduction", "structured_reduction", "magnus", "periodic"])
    parser.add_argument("--systems", nargs="+", choices=["RC", "R2_EOM"], default=["RC", "R2_EOM"])
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--duration-us", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps-ns", nargs="+", type=float, default=[8.0])
    parser.add_argument("--degrees", nargs="+", type=int, default=[8, 12])
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--exponential-rhs", choices=["structured", "sparse"], default="structured")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    TAG = args.tag
    if args.mode == "reduction":
        reduction(args)
    elif args.mode == "structured_reduction":
        structured_reduction(args)
    elif args.mode == "magnus":
        magnus(args)
    else:
        periodic(args)


if __name__ == "__main__":
    main()
