"""Bounded prepared-sparsity FP64 CUDA code generation.

Promoted from the cleanup-1 benchmarked single-component Tsit5 FMA path.
There are no molecular constants, numerical cutoffs, or benchmark imports.
CuPy is supplied by the integrator; importing this module does not load CUDA.
"""

from ._cuda_lowering import TIME, Node


def step_constant(node):
    """Prove stage invariance under ScalarCompiler's switch_t convention."""
    if isinstance(node, tuple):
        return all(step_constant(item) for item in node)
    if not isinstance(node, Node) or not node.symbolic.has(TIME):
        return True
    if node.op == "helper" and node.args[0] in {4, 6, 7, 12, 13}:
        return True
    if node.op == "time":
        return False
    return all(step_constant(arg) for arg in node.args)


def component_source(g, stages):
    """Emit the validated FP64 FMA derivative and fused Tsit5 stage updates."""
    fused = True
    rows, cols = g.rows.get(), g.cols.get()
    ptr, col = g.ptr.get(), g.col.get()
    rates = g.rates.get()
    tp, tc, tr = g.tp.get(), g.tc.get(), g.tr.get()
    remap = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols, strict=False))}

    def read(i, j):
        q = remap.get((min(i, j), max(i, j)), -1)
        if q < 0:
            return "0.", "0."
        return f"y[{q}L*B+b]", f"{'-' if i > j else ''}y[P+{q}L*B+b]"

    def term(edge, i, j, real, imag):
        u, v = read(i, j)
        address = str(edge)
        return (
            "{double u="
            + u
            + ",v="
            + v
            + ";double r=hr["
            + address
            + "],w=hi["
            + address
            + "];"
            + f"{real}=fma(r,u,{real});{real}=fma(-w,v,{real});"
            + f"{imag}=fma(r,v,{imag});{imag}=fma(w,u,{imag});}}"
        )

    args = """const double* y,double* out,const int* ptr,const int* col,
const double* hr,const double* hi,const int* groups,const long long* offsets,const double* amplitude,
const double* diag,const double* rates,const int* tp,const int* tc,const double* tr,
const int* rows,const int* cols,int B,int n,const double* clock,int skip"""
    if fused:
        args += (
            ",double* target,const double* base,"
            + ",".join(f"const double* k{i}" for i in range(7))
            + ",int next"
        )
    source = [
        'extern "C" __global__ void specialized_rhs(' + args + ") {",
        "int b=blockIdx.x*blockDim.x+threadIdx.x,p=blockIdx.y;",
        f"long P={g.upper_count}L*B;if(b>=B)return;",
        (
            "bool reuse=skip&&clock[3]!=0.;if(reuse&&next<0)return;if(!reuse){"
            if fused
            else "if(skip&&clock[3]!=0.)return;"
        ),
        "double r=0.,w=0.;switch(p){",
    ]
    for p, (ii, jj) in enumerate(zip(rows, cols, strict=False)):
        i, j = int(ii), int(jj)
        source.append(f"case {p}:{{double ar=0.,ai=0.,jr=0.,ji=0.;")
        for edge in range(int(ptr[i]), int(ptr[i + 1])):
            source.append(term(edge, int(col[edge]), j, "ar", "ai"))
        for edge in range(int(ptr[j]), int(ptr[j + 1])):
            source.append(term(edge, int(col[edge]), i, "jr", "ji"))
        loss = float(-0.5 * (rates[i] + rates[j]))
        source.append(
            "double scale=amplitude[b];"
            f"double u=y[{p}L*B+b],v=y[P+{p}L*B+b];"
            f"double d=diag[{i}L*B+b]-diag[{j}L*B+b],loss={loss:.17g};"
            "r=scale*(ai+ji)+d*v+loss*u;w=scale*(jr-ar)-d*u+loss*v;"
        )
        if i == j:
            for k in range(int(tp[i]), int(tp[i + 1])):
                pop, _ = read(int(tc[k]), int(tc[k]))
                source.append(f"r+=({float(tr[k]):.17g})*({pop});")
            source.append("w=0.;")
        source.append("break;}")
    source.append("}long x=(long)p*B+b;out[x]=r;out[P+x]=w;")
    if fused:
        source.append("}if(next>0){long x=(long)p*B+b;switch(next){")
        for row in range(1, 7):
            source.append(f"case {row}:")
            for index in ("x", "P+x"):
                terms = "+".join(
                    f"({a:.17g})*k{k}[{index}]" for k, a in enumerate(stages[row]) if a
                )
                source.append(f"target[{index}]=base[{index}]+clock[1]*({terms});")
            source.append("break;")
        source.append("}}")
    source.append("}")
    return "\n".join(source)
