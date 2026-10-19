"""Dependence-based CUDA coefficient caches and separable Hamiltonian plans."""

from __future__ import annotations

import numpy as np
import sympy as smp

from ._cuda_lowering import TIME, Node, ScalarCompiler, constant
from .ir import InstructionOp as Op


def _children(node):
    for arg in node.args:
        if isinstance(arg, Node):
            yield arg
        elif isinstance(arg, tuple):
            yield from arg


def _split(compiler, node, varying):
    """Return static amplitude and common coefficient without division by data."""
    if not node.dependencies & varying:
        return constant(1), node
    if node.op == "mul":
        left, right = (_split(compiler, n, varying) for n in node.args)
        if left is None or right is None:
            return None
        return tuple(compiler.binary(Op.MUL, a, b) for a, b in zip(left, right, strict=True))
    if node.op in {"neg", "conj"}:
        pair = _split(compiler, node.args[0], varying)
        if pair is None:
            return None
        a, t = pair
        if node.op == "neg":
            return a, compiler.binary(Op.MUL, constant(-1), t)
        return (
            compiler.make("conj", (a,), smp.conjugate(a.symbolic)),
            compiler.make("conj", (t,), smp.conjugate(t.symbolic)),
        )
    if not node.symbolic.has(TIME):
        return node, constant(1)
    return None


class CoefficientProgram:
    """One chunk's bounded cache layout and optional shared-H component model."""

    def __init__(self, prepared, original, parameters, varied, model, grid, options, fractions):
        complex_parameters = bool(np.any(parameters.imag != 0))
        enabled = (
            options.coefficient_sharing == "auto" or options.hamiltonian_sharing != "off"
        ) and not complex_parameters
        self.h_auto = options.hamiltonian_sharing == "auto"
        self.compiler = compiler = (
            ScalarCompiler(
                original.graph, [c["expression"] for c in prepared.hamiltonian_plan["coefficients"]]
            )
            if enabled
            else original
        )
        # Scalar IR symbols are real for algebra, but complex overrides are still
        # accepted by the public scan API. Keep real-helper validation intact.
        if enabled:
            compiler.factor_multipass()
        self.B, self.K = len(parameters), len(compiler.coefficients)
        varying = {
            k for k in range(parameters.shape[1]) if np.any(parameters[:, k] != parameters[0, k])
        }
        self.common = np.flatnonzero(~varied).tolist()
        self.variable = np.flatnonzero(varied).tolist()
        self.component_model = None
        self.amplitudes = np.empty((0, self.B))
        self.edge_groups = np.empty(0, np.int32)
        self.h_roots = []
        self.h_indices = []
        self.remainder_indices = []
        self.remainder_edges = np.empty(0, np.int32)
        self.h_offsets = np.empty(0, np.int64)
        self.h_storage = 0
        self.h_reason = "disabled" if options.hamiltonian_sharing == "off" else "already_shared"
        if complex_parameters and options.hamiltonian_sharing != "off":
            self.h_reason = "complex_parameter_override"
        offdiag, diagonal, ptr, positions = model
        used = np.flatnonzero(np.any(offdiag != 0, axis=1)).tolist()
        if (
            options.hamiltonian_sharing != "off"
            and not complex_parameters
            and options.rhs == "direct"
            and any(varied[k] for k in used)
        ):
            self._components(compiler, parameters, varying, model, used)
        elif options.hamiltonian_sharing != "off" and options.rhs != "direct":
            self.h_reason = "requires_direct_rhs"
        if self.component_model is not None:
            # Shared components replace off-diagonal coefficient evaluation.
            # Only the original diagonal still needs per-trajectory coefficients.
            needed = np.any(diagonal != 0, axis=1)
            needed[self.remainder_indices] = True
            self.common = [k for k in self.common if needed[k]]
            self.variable = [k for k in self.variable if needed[k]]

        static, shared, replacements = [], [], {}
        static_keys, shared_keys = {}, {}

        def visit(node, tn="stage_t"):
            if node.op in {"param", "const", "time"} or (node, tn) in replacements:
                return
            timed = node.symbolic.has(TIME)
            key = (node.op, node.symbolic, tn if timed else "static")
            if not timed:
                if key not in static_keys:
                    static_keys[key] = len(static)
                    static.append(node)
                replacements[node, tn] = ("static", static_keys[key])
            elif not node.dependencies & varying:
                if key not in shared_keys:
                    shared_keys[key] = len(shared)
                    shared.append((node, tn))
                replacements[node, tn] = ("shared", shared_keys[key])
            else:
                child_time = (
                    "switch_t" if node.op == "helper" and node.args[0] in {4, 6, 7, 12, 13} else tn
                )
                for child in _children(node):
                    visit(child, child_time)

        if options.coefficient_sharing == "auto" and not complex_parameters:
            for k in self.variable:
                visit(compiler.coefficients[k])
        # Bound additional static storage independently of the time-table budget.
        if len(static) * self.B * 16 > 64_000_000:
            replacements = {k: v for k, v in replacements.items() if v[0] != "static"}
            static = []
        self.static = static
        roots, times, lookup = [], [], {}

        def table_slot(node, tn="stage_t"):
            key = (node.op, node.symbolic, tn)
            if key not in lookup:
                lookup[key] = len(roots)
                roots.append(node)
                times.append(tn)
            return lookup[key]

        # Keep both-off layout compatible with the old path; auto packs live columns.
        legacy = not enabled
        common_slots = {k: table_slot(compiler.coefficients[k]) for k in self.common}
        if legacy:
            roots, times = list(compiler.coefficients), ["stage_t"] * self.K
            common_slots = {k: k for k in self.common}
        shared_slots = [table_slot(n, tn) for n, tn in shared]
        h_slots = {k: table_slot(self.h_roots[k]) for k in self.h_indices}
        self.table_width = len(roots)
        self.precompute = bool(
            options.precompute
            and len(grid)
            and len(grid) * 7 * self.table_width * 16 <= options.precompute_bytes
        )
        substitutions = {}
        for key, (kind, index) in replacements.items():
            if kind == "static":
                substitutions[key] = f"static_values[{index}*B+b]"
            elif self.precompute:
                substitutions[key] = f"shared_values[{shared_slots[index]}]"
        self.shared_count = len(shared) if self.precompute else 0
        self.cache_reason = (
            "disabled"
            if options.coefficient_sharing == "off"
            else "active"
            if substitutions
            else "no_reusable_subexpressions"
        )
        if options.coefficient_sharing == "auto" and shared and not self.precompute:
            self.cache_reason = (
                "static_only_precompute_unavailable" if static else "precompute_unavailable"
            )
        if complex_parameters and options.coefficient_sharing != "off":
            self.cache_reason = "complex_parameter_override"
        function = compiler.function
        self.functions = (
            function("eval_common", self.common)
            + function(
                "eval_variable",
                self.variable,
                substitutions=substitutions,
                extra_arguments=",const C* static_values,const C* shared_values",
            )
            + function("eval_static", list(range(len(static))), roots=static)
            + function("eval_unscaled", self.h_indices, roots=self.h_roots)
            + function(
                "eval_table",
                list(range(len(roots))) if not legacy else self.common,
                roots=roots,
                root_times=times,
            )
        )
        common_assign = "".join(f"out[{k}*B]=values[{v}];" for k, v in common_slots.items())
        h_assign = "".join(f"unscaled[{k}]=values[{v}];" for k, v in h_slots.items())
        frac_array = ",".join(f"{v:.17g}" for v in (*fractions, *([0.0] * (7 - len(fractions)))))
        self.kernels = (
            f"\n#define COEFFICIENT_TABLE_WIDTH {max(1, self.table_width)}\n"
            + r"""
extern "C" __global__ void coefficients(const C* params,double* clock,C* out,const C* table,const C* static_values,C* unscaled,long rows,int K,int B,int stage,double fraction,int skip,int* error){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B||clock[1]<=0.||(skip&&clock[3]!=0.))return;
 double stage_t=clock[0]+fraction*clock[1],switch_t=clock[0]+.5*clock[1];long row=(long)clock[4];
 const C* values=table+(PRECOMPUTE?(row*7+stage)*COEFFICIENT_TABLE_WIDTH:0);
 if(b==0){clock[5]+=1.;if(PRECOMPUTE&&row<rows){COMMON_ASSIGN H_ASSIGN}else{eval_common(params,B,0,stage_t,switch_t,out,B,error);eval_unscaled(params,B,0,stage_t,switch_t,unscaled,1,error);}}
 eval_variable(params,B,b,stage_t,switch_t,out,B,error,static_values,values);
}
extern "C" __global__ void initialize_static(const C* params,C* out,int B,int* error){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b<B)eval_static(params,B,b,0.,0.,out,B,error);}
extern "C" __global__ void precompute(const C* params,const double* grid,C* table,long rows,int K,int B,int* error){
 long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x>=rows*7)return;long row=x/7;int stage=x%7;
 const double fractions[7]={FRACTIONS};double h=grid[2*row+1],t=grid[2*row];
 eval_table(params,B,0,t+fractions[stage]*h,t+.5*h,table+x*COEFFICIENT_TABLE_WIDTH,1,error);
}
extern "C" __global__ void validate(const double* y,long D,int* error){long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D&&!isfinite(y[x]))atomicExch(error,1);}
""".replace("PRECOMPUTE", str(int(self.precompute)))
            .replace("COMMON_ASSIGN", common_assign)
            .replace("H_ASSIGN", h_assign)
            .replace("FRACTIONS", frac_array)
        )

    def _components(self, compiler, parameters, varying, model, used):
        offdiag, diagonal, _, positions = model
        pairs = {k: _split(compiler, compiler.coefficients[k], varying) for k in used}
        keys, mapping, amplitudes = {}, {}, []
        symbols = [smp.Symbol(f"p{k}", real=True) for k in range(parameters.shape[1])]
        for k, pair in pairs.items():
            mapping[k] = -1
            if pair is None:
                continue
            a, _ = pair
            key = a.symbolic
            if key not in keys:
                valid = not any(str(f.func).startswith("helper_") for f in key.atoms(smp.Function))
                if valid:
                    with np.errstate(all="ignore"):
                        value = np.broadcast_to(
                            np.asarray(smp.lambdify(symbols, key, "numpy")(*parameters.T), complex),
                            (self.B,),
                        )
                    valid = np.isfinite(value).all() and np.all(value.imag == 0)
                keys[key] = len(amplitudes) if valid else -1
                if valid:
                    amplitudes.append(value.real)
            mapping[k] = keys[key]
        if not amplitudes:
            self.h_reason = "nonseparable_or_nonreal_offdiagonal"
            return
        if len(amplitudes) > 4 or len(amplitudes) >= self.B:
            self.h_reason = "component_overhead"
            return
        entries = []
        remainder = [k for k in used if mapping[k] < 0]
        for g in (*range(len(amplitudes)), *([-1] if remainder else [])):
            indices = [k for k in used if mapping[k] == g]
            for edge in np.flatnonzero(np.any(offdiag[indices] != 0, axis=0)):
                entries.append((int(positions[0, edge]), int(positions[1, edge]), g, int(edge)))
        entries.sort()
        costs = np.asarray([self.B if g < 0 else 1 for _, _, g, _ in entries], np.int64)
        storage = int(costs.sum())
        if storage >= offdiag.shape[1] * self.B * (0.9 if remainder else 0.5):
            self.h_reason = "component_overhead"
            return
        # Per-edge scaling costs O(n) work per edge, versus O(K) saved assembly.
        # Avoid the measured RC regression; 'on' bypasses this performance gate.
        if self.h_auto and (len(amplitudes) > 1 or remainder) and len(diagonal[0]) > 3 * self.K:
            self.h_reason = "rhs_scaling_overhead"
            return
        basis = np.zeros((self.K, len(entries)), complex)
        for e, (_, _, g, old) in enumerate(entries):
            for k in used:
                if mapping[k] == g:
                    basis[k, e] = offdiag[k, old]
        rows, cols = np.asarray([(r, c) for r, c, _, _ in entries], np.int32).T
        ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(diagonal[0])))].astype(np.int32)
        self.component_model = basis, diagonal, ptr, np.stack((rows, cols))
        self.edge_groups = np.asarray([g for _, _, g, _ in entries], np.int32)
        self.amplitudes = np.asarray(amplitudes)
        self.remainder_indices = remainder
        self.remainder_edges = np.flatnonzero(self.edge_groups < 0).astype(np.int32)
        self.h_offsets = np.r_[0, np.cumsum(costs[:-1])].astype(np.int64)
        self.h_storage = storage
        self.h_indices = [k for k in used if mapping[k] >= 0]
        self.h_roots = [pairs[k][1] if k in self.h_indices else constant(0) for k in range(self.K)]
        self.h_reason = (
            "mixed_components"
            if remainder
            else "single_component"
            if len(amplitudes) == 1
            else "multiple_components"
        )
