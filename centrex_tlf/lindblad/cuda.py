"""Opt-in, synchronized FP64 CUDA integration of prepared Lindblad models.

The CPU defaults are unchanged. CuPy and CUDA are loaded only on execution.
Fixed steps have no tolerance controller: callers must check convergence for
their model, time interval and requested density/observable accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from itertools import pairwise
from numbers import Integral
from typing import Any

import numpy as np

from .integral_output import FULL_AND_INTEGRAL_OUTPUT, INTEGRAL_OUTPUTS, INTEGRAL_RATE_OUTPUTS

__all__ = ["CudaSolverOptions"]

TS_C = (0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)
TS_A = (
    (),
    (0.161,),
    (-0.008480655492356989, 0.335480655492357),
    (2.8971530571054935, -6.359448489975075, 4.3622954328695815),
    (5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525),
    (
        5.86145544294642,
        -12.92096931784711,
        8.159367898576159,
        -0.071584973281401,
        -0.028269050394068383,
    ),
    (
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
    ),
)
CK_A = (
    0.0,
    -567301805773 / 1357537059087,
    -2404267990393 / 2016746695238,
    -3550918686646 / 2091501179385,
    -1275806237668 / 842570457699,
)
CK_B = (
    1432997174477 / 9575080441755,
    5161836677717 / 13612068292357,
    1720146321549 / 2090206949498,
    3134564353537 / 4481467310338,
    2277821191437 / 14882151754819,
)
CK_C = (
    0.0,
    1432997174477 / 9575080441755,
    2526269341429 / 6820363962896,
    2006345519317 / 3224310063776,
    2802321613138 / 2924317926251,
)


@dataclass(frozen=True)
class CudaSolverOptions:
    """CUDA ODE controls; time values are absolute seconds.

    ``batch_size`` bounds concurrent trajectories, not the total scan size.
    Known affine square-wave/gate switches are clipped automatically. Supply
    ``discontinuities`` for additional jumps (required for nonlinear switches).
    ``step_schedule`` is an increasing sequence of ``(time, step_seconds)``;
    the solve's ``dt`` applies before its first entry. All trajectories share
    this prescribed schedule; there is no adaptive step-size controller.

    ``density_saveat`` optionally saves full packed density snapshots in result
    metadata while the main output is populations, photons, or selected entries.
    ``density_reduction=True`` computes exact reachable complex coordinates per
    chunk, compacts the ODE buffers and selects the structured direct kernel.
    Omitted coordinates reconstruct as zero. The default is False; benchmark
    before enabling it when little support can be removed.
    No RC geometry, modulation frequency, or beam envelope is hard-coded.

    ``coefficient_sharing`` and ``hamiltonian_sharing`` default to ``"auto"``;
    ``"off"`` disables either independently. Auto caches reusable scalar subexpressions and,
    with the direct RHS, shares separable off-diagonal H components. Unsupported
    factors use normal evaluation/storage. Precomputed subexpressions respect
    ``precompute_bytes``; ``precompute=False`` still permits static caching.
    ``hamiltonian_sharing="on"`` bypasses the arithmetic performance gate for
    explicit benchmarks; structural and storage eligibility still apply.
    """

    batch_size: int = 512
    block_size: int = 256
    tile_size: int = 32
    graph_steps: int = 16
    precompute: bool = True
    precompute_bytes: int = 128_000_000
    rhs: str = "tiled"
    device_id: int = 0
    discontinuities: Any = None
    step_schedule: Any = ()
    density_saveat: Any = ()
    density_reduction: bool = False
    coefficient_sharing: str = "auto"
    hamiltonian_sharing: str = "auto"

    def __post_init__(self) -> None:
        for name in ("coefficient_sharing", "hamiltonian_sharing"):
            choices = {"auto", "off", "on"} if name == "hamiltonian_sharing" else {"auto", "off"}
            if not isinstance(getattr(self, name), str) or getattr(self, name) not in choices:
                raise ValueError(f"CUDA {name} must be one of {sorted(choices)}")
        if not isinstance(self.density_reduction, bool):
            raise ValueError("CUDA density_reduction must be a bool")
        for name in [
            "batch_size",
            "block_size",
            "graph_steps",
            "precompute_bytes",
            "device_id",
            "tile_size",
        ]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"CUDA {name} must be an integer")
        if self.batch_size < 1 or self.precompute_bytes < 0 or self.device_id < 0:
            raise ValueError(
                "CUDA batch_size must be positive; memory budget and device_id must be nonnegative"
            )
        if self.block_size not in {64, 128, 256, 512} or self.tile_size not in {0, 16, 32}:
            raise ValueError("CUDA block_size must be 64/128/256/512 and tile_size 0/16/32")
        if self.graph_steps < 2 or self.graph_steps % 2 or self.graph_steps > 64:
            raise ValueError("CUDA graph_steps must be even and between 2 and 64")
        if self.rhs not in {"tiled", "direct"}:
            raise ValueError("CUDA rhs must be 'tiled' or 'direct'")


def _cupy() -> Any:
    # Windows resolves CUDA DLLs by first load. If both extras are installed,
    # load PyTorch's bundled DLLs before CuPy initializes CUDA; the reverse
    # order can crash a subsequent cuBLAS call in the static solver.
    import importlib.util
    import sys

    if sys.platform == "win32" and importlib.util.find_spec("torch") is not None:
        import torch  # noqa: F401
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "CUDA ODE integration requires optional CuPy; install centrex-tlf[cuda] or centrex-tlf[cuda13] for your CUDA runtime"
        ) from exc
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(
            "CUDA ODE integration requires an available CUDA device and compatible driver"
        ) from exc
    if not available:
        raise RuntimeError("CUDA ODE integration requires an available CUDA device")
    return cp


def _basis(prepared: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plan, n = prepared.hamiltonian_plan, prepared.layout.n
    if plan.get("kind") != "decomposed":
        raise ValueError("CUDA ODE integration requires hamiltonian_representation='decomposed'")
    bases = [np.triu(np.asarray(plan["static_matrix"], complex))]
    for coefficient in plan["coefficients"]:
        upper = np.zeros((n, n), complex)
        if "basis_terms" in coefficient:
            for term in coefficient["basis_terms"]:
                upper[term["i"], term["j"]] += complex(term["re"], term["im"])
        elif "basis_row_segments" in coefficient:
            for segment in coefficient["basis_row_segments"]:
                values = np.array(segment["values_re"]) + 1j * np.array(segment["values_im"])
                i, j = segment["row"], segment["start_col"]
                upper[i, j : j + len(values)] += values
        else:
            upper = np.asarray(coefficient["basis_upper"], complex)
        bases.append(upper)
    bases = np.asarray(bases)
    if not np.isfinite(bases).all():
        raise ValueError("CUDA Hamiltonian basis contains nonfinite values")
    support = np.any(bases != 0, axis=0)
    support |= support.T
    np.fill_diagonal(support, False)
    rows, cols = np.nonzero(support)
    ptr = np.r_[0, np.cumsum(support.sum(axis=1))].astype(np.int32)
    offdiag = bases[:, np.minimum(rows, cols), np.maximum(rows, cols)]
    diagonal = np.diagonal(bases, axis1=1, axis2=2).copy()
    return offdiag, diagonal, ptr, np.stack((rows, cols)).astype(np.int32)


def _steps(
    span: tuple[float, float],
    dt: float,
    switches: np.ndarray,
    outputs: np.ndarray,
    schedule: Any,
    maxiters: int,
) -> tuple[np.ndarray, np.ndarray, dict[float, int]]:
    knots = np.asarray(schedule, dtype=float)
    if knots.size == 0:
        knots = np.empty((0, 2))
    if (
        knots.ndim != 2
        or knots.shape[1] != 2
        or not np.isfinite(knots).all()
        or np.any(np.diff(knots[:, 0]) <= 0)
        or np.any(knots[:, 1] <= 0)
        or np.any(knots[:, 0] < span[0])
        or np.any(knots[:, 0] > span[1])
    ):
        raise ValueError(
            "CUDA step_schedule must contain increasing in-span (time, positive_step) pairs"
        )
    events = np.unique(np.r_[span[0], span[1], switches, outputs, knots[:, 0]])
    blocks, resets = [], []
    limits = {span[0]: 0}
    count = 0
    for a, b in pairwise(events):
        k = np.searchsorted(knots[:, 0], a, side="right") - 1
        h = dt if k < 0 else knots[k, 1]
        ratio = (b - a) / h
        if not np.isfinite(ratio) or ratio > maxiters:
            raise ValueError("CUDA fixed-step solve exceeds maxiters; increase maxiters or dt")
        n = max(1, int(np.ceil(ratio - 8 * np.finfo(float).eps * max(1, ratio))))
        starts = a + np.arange(n) * h
        starts = starts[starts < b]
        widths = np.diff(np.r_[starts, b])
        if np.any(widths <= 0):
            raise ValueError("CUDA step size cannot advance floating-point time")
        block = np.stack((starts, widths), axis=1)
        reset = np.zeros(len(block), np.int32)
        if len(block):
            reset[0] = int(count == 0 or np.any(switches == a))
        count += len(block)
        if count > maxiters:
            raise ValueError("CUDA fixed-step solve exceeds maxiters; increase maxiters or dt")
        blocks.append(block)
        resets.append(reset)
        limits[float(b)] = count
    return (
        np.concatenate(blocks) if blocks else np.empty((0, 2)),
        np.concatenate(resets) if resets else np.empty(0, np.int32),
        limits,
    )


def _stage_source() -> str:
    params = ",".join(f"const double* k{i}" for i in range(7))
    result = []
    for row in range(1, 7):
        terms = "+".join(f"({weight:.17g})*k{k}[x]" for k, weight in enumerate(TS_A[row]) if weight)
        result.append(
            f'extern "C" __global__ void stage_{row}(double* out,const double* y,{params},const double* clock,long D){{long x=(long)blockIdx.x*blockDim.x+threadIdx.x;if(x<D)out[x]=y[x]+clock[1]*({terms});}}'
        )
    return "\n".join(result)


class _Integrator:
    def __init__(
        self,
        cp: Any,
        prepared: Any,
        compiler: Any,
        packed: np.ndarray,
        parameters: np.ndarray,
        varied: np.ndarray,
        model: tuple,
        grid: np.ndarray,
        resets: np.ndarray,
        options: CudaSolverOptions,
        solver: str,
        weights: np.ndarray,
        integral: bool,
        span: tuple,
    ):
        from ._cuda_kernels import COMPONENT_KERNELS, HEADER, KERNELS, tile_kernel
        from ._density_support import reachable_upper

        self.cp, self.options, self.solver = cp, options, solver
        self.B, self.n = len(packed), prepared.layout.n
        upper_count = self.n * (self.n + 1) // 2
        active = (
            reachable_upper(prepared, packed, model)
            if options.density_reduction
            else np.arange(upper_count)
        )
        remap = np.full(upper_count, -1, dtype=np.int32)
        remap[active] = np.arange(len(active))
        self.upper_count = len(active)
        self.P = self.B * self.upper_count
        prefix = ""
        if options.density_reduction:
            # Keep the optimized H/decay split and compact ODE stage buffers.
            self.options = options = replace(options, rhs="direct")
            storage = "__constant__" if remap.nbytes <= 48_000 else "const"
            prefix = (
                f"#define REDUCED_PAIRS {len(active)}\n"
                f"__device__ {storage} int density_index[]={{" + ",".join(map(str, remap)) + "};\n"
            )
        self.D = 2 * self.P + (self.B if integral else 0)
        self.integral = integral
        self.grid_host = grid
        self.steps = len(grid)
        self.grid = cp.asarray(grid)
        self.resets = cp.asarray(resets)
        self.clock = cp.asarray(np.array([span[0], 0.0, 0.0, 0.0, 0.0, 0.0]))
        offdiag, diagonal, ptr, positions = model
        self.E = offdiag.shape[1]
        self.K = len(compiler.coefficients)
        self.HB = self.B if np.any(varied & np.any(offdiag != 0, axis=1)) else 1
        self.tile = options.tile_size if options.rhs == "tiled" else 0
        shared_limit = cp.cuda.runtime.getDeviceProperties(options.device_id)["sharedMemPerBlock"]
        if 16 * self.n * self.tile > shared_limit:
            self.tile = 0
        from ._cuda_sharing import CoefficientProgram

        self.program = program = CoefficientProgram(
            prepared,
            compiler,
            parameters,
            varied,
            model,
            grid,
            options,
            CK_C if solver == "fixed_ck54" else TS_C,
        )
        compiler = program.compiler
        self.components = program.component_model is not None
        if self.components:
            offdiag, diagonal, ptr, positions = program.component_model
            self.E, self.HB = offdiag.shape[1], 1
        self.precompute = program.precompute
        functions, coefficients = program.functions, program.kernels
        source = (
            prefix
            + HEADER
            + "\n".join(compiler.helpers)
            + functions
            + coefficients
            + KERNELS
            + (
                f"\n#define SINGLE_H_COMPONENT {int(len(program.amplitudes) == 1 and not len(program.remainder_edges))}\n"
                f"#define HYBRID_H_COMPONENTS {int(bool(len(program.remainder_edges)))}\n"
                + COMPONENT_KERNELS
                if self.components
                else ""
            )
            + _stage_source()
        )
        if self.tile:
            source += tile_kernel(self.n, self.tile)
        module = cp.RawModule(code=source, options=("--std=c++17",))
        names = [
            "set_limit",
            "begin_step",
            "end_step",
            "build_h",
            "build_diag",
            "product",
            "finish",
            "direct_rhs",
            "integral_rhs",
            "ls_update",
            "sample",
            "coefficients",
            "precompute",
            "validate",
            "initialize_static",
        ] + [f"stage_{i}" for i in range(1, 7)]
        if self.tile:
            names.append("product_tile")
        if self.components:
            names.extend(("build_components", "build_remainder", "component_rhs"))
        self.kernels = {name: module.get_function(name) for name in names}
        self.params = cp.asarray(np.ascontiguousarray(parameters.T))
        self.varied = cp.asarray(varied.astype(np.int32))
        self.coeff = cp.zeros((self.K, self.B), cp.complex128)
        self.table = cp.zeros(
            (self.steps, 7, program.table_width) if self.precompute else (1,), cp.complex128
        )
        self.static_values = cp.empty((max(1, len(program.static)), self.B), cp.complex128)
        self.unscaled = cp.zeros(self.K, cp.complex128)
        self.amplitudes = cp.asarray(program.amplitudes)
        self.edge_groups = cp.asarray(program.edge_groups)
        self.h_offsets = cp.asarray(program.h_offsets)
        self.remainder_edges = cp.asarray(program.remainder_edges)
        self.error = cp.zeros(1, np.int32)
        self.bases = cp.asarray(np.ascontiguousarray(offdiag))
        self.diagonal_bases = cp.asarray(np.ascontiguousarray(diagonal))
        self.ptr = cp.asarray(ptr)
        self.col = cp.asarray(positions[1])
        self.hrows = cp.asarray(positions[0])
        self.hcols = cp.asarray(positions[1])
        self.hr = cp.empty(program.h_storage if self.components else (self.E, self.HB))
        self.hi = cp.empty_like(self.hr)
        self.diag = cp.empty((self.n, self.B))
        self.rates = cp.asarray(prepared.source_decay_rates)
        tp, tc, tr = [0], [], []
        for entries in prepared.incoming_transfers_by_target:
            for entry in entries:
                tc.append(entry["source"])
                tr.append(entry["rate"])
            tp.append(len(tc))
        self.tp, self.tc, self.tr = (
            cp.asarray(tp, dtype=np.int32),
            cp.asarray(tc, dtype=np.int32),
            cp.asarray(tr, dtype=float),
        )
        ui, uj = np.triu_indices(self.n)
        ui, uj = ui[active], uj[active]
        self.rows, self.cols = cp.asarray(ui, dtype=np.int32), cp.asarray(uj, dtype=np.int32)
        public_pk = [i * self.n - i * (i - 1) // 2 for i in range(self.n)]
        public_im = [0] * self.n
        for i, j in zip(*np.triu_indices(self.n, 1), strict=False):
            p = i * self.n - i * (i + 1) // 2 + j
            public_pk.extend([p, p])
            public_im.extend([0, 1])
        public_pk = remap[np.asarray(public_pk)]
        self.public_pk, self.public_im = (
            cp.asarray(public_pk, dtype=np.int32),
            cp.asarray(public_im, dtype=np.int32),
        )
        host = np.zeros(self.D, float)
        real = host[: self.P].reshape(-1, self.B)
        imag = host[self.P : 2 * self.P].reshape(-1, self.B)
        # Vectorized conversion from public packed coordinates, without full
        # complex reconstruction on either device.
        pp = np.asarray(public_pk)
        pi = np.asarray(public_im, dtype=bool)
        re_keep, im_keep = (~pi) & (pp >= 0), pi & (pp >= 0)
        real[pp[re_keep]] = packed[:, re_keep].T
        imag[pp[im_keep]] = packed[:, im_keep].T
        self.initial = cp.asarray(host)
        self.y = self.initial.copy()
        self.weights = cp.asarray(weights)
        if solver == "fixed_ck54":
            self.residual = cp.empty_like(self.y)
            self.derivative = cp.empty_like(self.y)
        else:
            self.ks = [cp.empty_like(self.y) for _ in range(7)]
            self.temps = [cp.empty_like(self.y) for _ in range(2)]
            self.yn = cp.empty_like(self.y)
        if options.rhs != "direct":
            self.ar = cp.empty((self.n * self.n, self.B))
            self.ai = cp.empty_like(self.ar)
        if program.static:
            self.launch(
                "initialize_static",
                (self.params, self.static_values, np.int32(self.B), self.error),
                self.B,
            )
        if self.precompute:
            self.launch(
                "precompute",
                (
                    self.params,
                    self.grid,
                    self.table,
                    np.int64(self.steps),
                    np.int32(self.K),
                    np.int32(self.B),
                    self.error,
                ),
                self.steps * 7,
            )
        self._capture(span)

    def launch(self, name: str, args: tuple, count: int) -> None:
        if count:
            self.kernels[name](
                ((count + self.options.block_size - 1) // self.options.block_size,),
                (self.options.block_size,),
                args,
            )

    def rhs(self, y: Any, out: Any, stage: int, fraction: float, skip: bool = False) -> None:
        B, n, K, HB, E = map(np.int32, (self.B, self.n, self.K, self.HB, self.E))
        s = np.int32(skip)
        self.launch(
            "coefficients",
            (
                self.params,
                self.clock,
                self.coeff,
                self.table,
                self.static_values,
                self.unscaled,
                np.int64(self.steps),
                K,
                B,
                np.int32(stage),
                np.float64(fraction),
                s,
                self.error,
            ),
            self.B,
        )
        if self.components:
            self.launch(
                "build_components",
                (
                    self.unscaled,
                    self.bases,
                    self.hrows,
                    self.hcols,
                    self.edge_groups,
                    self.h_offsets,
                    self.hr,
                    self.hi,
                    E,
                    K,
                    self.clock,
                    s,
                ),
                self.E,
            )
            if len(self.program.remainder_edges):
                self.launch(
                    "build_remainder",
                    (
                        self.coeff,
                        self.bases,
                        self.varied,
                        self.hrows,
                        self.hcols,
                        self.remainder_edges,
                        self.h_offsets,
                        self.hr,
                        self.hi,
                        np.int32(len(self.program.remainder_edges)),
                        E,
                        K,
                        B,
                        self.clock,
                        s,
                    ),
                    len(self.program.remainder_edges) * self.B,
                )
        else:
            self.launch(
                "build_h",
                (
                    self.coeff,
                    self.bases,
                    self.varied,
                    self.hrows,
                    self.hcols,
                    self.hr,
                    self.hi,
                    E,
                    K,
                    B,
                    HB,
                    self.clock,
                    s,
                ),
                self.E * self.HB,
            )
        self.launch(
            "build_diag",
            (
                self.coeff,
                self.diagonal_bases,
                self.varied,
                self.diag,
                n,
                K,
                B,
                self.clock,
                s,
                self.error,
            ),
            self.n * self.B,
        )
        tail = (self.diag, self.rates, self.tp, self.tc, self.tr, self.rows, self.cols, B, n)
        if self.components:
            self.launch(
                "component_rhs",
                (
                    y,
                    out,
                    self.ptr,
                    self.col,
                    self.hr,
                    self.hi,
                    self.edge_groups,
                    self.h_offsets,
                    self.amplitudes,
                    *tail,
                    self.clock,
                    s,
                ),
                self.P,
            )
        elif self.options.rhs == "direct":
            self.launch(
                "direct_rhs",
                (y, out, self.ptr, self.col, self.hr, self.hi, *tail, HB, self.clock, s),
                self.P,
            )
        else:
            args = (
                y,
                self.ar,
                self.ai,
                self.ptr,
                self.col,
                self.hr,
                self.hi,
                B,
                n,
                HB,
                self.clock,
                s,
            )
            if self.tile:
                self.kernels["product_tile"](
                    ((self.B + self.tile - 1) // self.tile, self.n),
                    (self.options.block_size,),
                    args,
                )
            else:
                self.launch("product", args, self.n * self.n * self.B)
            self.launch("finish", (y, out, self.ar, self.ai, *tail, self.clock, s), self.P)
        if self.integral:
            self.launch("integral_rhs", (y, out, self.weights, B, n, self.clock, s), self.B)

    def block(self, count: int) -> None:
        if self.solver == "fixed_ck54":
            for _ in range(count):
                self.kernels["begin_step"](
                    (1,), (1,), (self.clock, self.grid, self.resets, np.int64(self.steps))
                )
                for k, (a, b, c) in enumerate(zip(CK_A, CK_B, CK_C, strict=False)):
                    self.rhs(self.y, self.derivative, k, c)
                    self.launch(
                        "ls_update",
                        (
                            self.y,
                            self.residual,
                            self.derivative,
                            self.clock,
                            np.int64(self.D),
                            np.float64(a),
                            np.float64(b),
                        ),
                        self.D,
                    )
                self.kernels["end_step"]((1,), (1,), (self.clock, np.int32(0)))
            return
        y, yn = self.y, self.yn
        ks = list(self.ks)
        for _ in range(count):
            self.kernels["begin_step"](
                (1,), (1,), (self.clock, self.grid, self.resets, np.int64(self.steps))
            )
            ks[0], ks[6] = ks[6], ks[0]
            stage_y = y
            for k, c in enumerate(TS_C):
                self.rhs(stage_y, ks[k], k, c, skip=k == 0)
                if k < 6:
                    target = yn if k == 5 else self.temps[k % 2]
                    self.launch(
                        f"stage_{k + 1}", (target, y, *ks, self.clock, np.int64(self.D)), self.D
                    )
                    stage_y = target
            self.kernels["end_step"]((1,), (1,), (self.clock, np.int32(1)))
            y, yn = yn, y

    def _capture(self, span: tuple) -> None:
        cp = self.cp
        self.clock[2] = min(2, self.steps)
        self.block(2)
        cp.cuda.get_current_stream().synchronize()
        cp.copyto(self.y, self.initial)
        self.clock.set(np.array([span[0], 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.graphs = {}
        for count in range(2, self.options.graph_steps + 1, 2):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                self.block(count)
                self.graphs[count] = stream.end_capture()
        cp.cuda.get_current_stream().synchronize()

    def advance(self, previous: int, target: int) -> None:
        self.kernels["set_limit"]((1,), (1,), (self.clock, np.int64(target)))
        full, tail = divmod(target - previous, self.options.graph_steps)
        for _ in range(full):
            self.graphs[self.options.graph_steps].launch()
        if tail:
            self.graphs[2 * ((tail + 1) // 2)].launch()


def _packed_initial_states(prepared: Any, rho0_batch: Any) -> np.ndarray:
    """Validate before packing, including before expanding initial states in grids."""
    from .batch import _pack_rho0_batch

    initial = np.asarray(rho0_batch)
    if not np.isfinite(initial).all():
        raise ValueError("CUDA initial states must be finite")
    if initial.ndim == 3 and not np.allclose(
        initial, initial.conj().swapaxes(-1, -2), atol=1e-12, rtol=1e-12
    ):
        raise ValueError("CUDA initial density matrices must be Hermitian")
    if initial.ndim == 2 and np.iscomplexobj(initial) and np.any(initial.imag != 0):
        raise ValueError("packed CUDA initial states must be real")
    packed = _pack_rho0_batch(prepared, initial)
    if not len(packed):
        raise ValueError("rho0_batch must contain at least one trajectory")
    return packed


def _solve_cuda_batch(
    prepared: Any,
    rho0_batch: Any,
    t_span: Any,
    *,
    solver: str,
    dt: float,
    parameter_batch: Any = None,
    parameter_slots: Any = None,
    saveat: Any = None,
    save_start: bool = True,
    output: str = "populations",
    output_when: str = "final",
    output_indices: Any = None,
    integral_weights: Any = None,
    integral_method: str = "solver",
    integral_saveat: Any = None,
    stop_event: Any = None,
    maxiters: int = 100_000,
    collect_stats: bool = False,
    metadata: Any = None,
    cuda_options: CudaSolverOptions | None = None,
    execution_mode: str = "expanded_sparse",
    dense_output: bool = True,
    **unused: Any,
) -> Any:
    from ._cuda_lowering import ScalarCompiler
    from .batch import (
        LindbladBatchResult,
        _parameter_slot_indices,
        _parameter_slot_names,
    )
    from .dense import _sample_times

    options = CudaSolverOptions() if cuda_options is None else cuda_options
    if not isinstance(options, CudaSolverOptions):
        raise TypeError("cuda_options must be CudaSolverOptions")
    if solver not in {"fixed_tsit5", "fixed_ck54"}:
        raise ValueError(
            "CUDA ODE solvers are 'fixed_tsit5' and 'fixed_ck54'; their step size is explicit dt, not adaptive tolerances"
        )
    if execution_mode != "expanded_sparse":
        raise ValueError("CUDA ODE integration currently supports execution_mode='expanded_sparse'")
    if stop_event is not None:
        raise ValueError("CUDA fixed-step batches do not support terminal events")
    if integral_method != "solver" or integral_saveat is not None:
        raise ValueError(
            "CUDA integrals use Runge-Kutta stage quadrature; sampled quadrature is not supported"
        )
    if not dense_output and output_when == "saveat":
        raise ValueError("CUDA save-at output requires dense_output=True")
    span = tuple(float(t) for t in t_span)
    if len(span) != 2 or not np.isfinite(span).all() or span[1] < span[0]:
        raise ValueError("t_span must be finite and increasing")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("CUDA dt must be finite and positive")
    if isinstance(maxiters, bool) or not isinstance(maxiters, Integral) or maxiters < 1:
        raise ValueError("maxiters must be a positive integer")
    model = _basis(prepared)
    packed = _packed_initial_states(prepared, rho0_batch)
    if output_when not in {"saveat", "final"}:
        raise ValueError("CUDA output_when must be saveat or final")
    times = _sample_times(span, saveat, save_start, output_when)
    density_times = np.asarray(options.density_saveat, float)
    if (
        density_times.ndim != 1
        or not np.isfinite(density_times).all()
        or np.any(np.diff(density_times) <= 0)
        or np.any(density_times < span[0])
        or np.any(density_times > span[1])
    ):
        raise ValueError(
            "CUDA density_saveat must be finite, strictly increasing and inside t_span"
        )
    modes = {"full": 0, "populations": 1, "selected": 2, FULL_AND_INTEGRAL_OUTPUT: 5}
    modes.update(dict.fromkeys(INTEGRAL_OUTPUTS, 3))
    modes.update(dict.fromkeys(INTEGRAL_RATE_OUTPUTS.values(), 4))
    if output not in modes:
        raise ValueError(f"unsupported CUDA output {output!r}")
    mode = modes[output]
    n = prepared.layout.n
    if (output_indices is not None) != (output == "selected"):
        raise ValueError("output_indices is required only for output='selected'")
    indices = [] if output_indices is None else list(output_indices)
    if output == "selected" and not indices:
        raise ValueError("output_indices must be nonempty")
    for i, j in indices:
        prepared.layout.diagonal_index(i)
        prepared.layout.diagonal_index(j)
    weighted = mode in {3, 4, 5}
    if (integral_weights is not None) != weighted:
        raise ValueError("integral_weights is required only for weighted CUDA outputs")
    weights = np.zeros(n)
    for i, w in [] if integral_weights is None else integral_weights:
        prepared.layout.diagonal_index(i)
        if not np.isfinite(w):
            raise ValueError("integral weights must be finite")
        weights[i] += float(w)
    if mode == 4 and output_when != "saveat":
        raise ValueError("weighted rate output requires output_when='saveat'")
    width = {0: n * n, 1: n, 2: 2 * len(indices), 3: 1, 4: 1, 5: n * n + 1}[mode]
    slots = _parameter_slot_indices(prepared, parameter_slots)
    graph = prepared.parameter_graph
    base = np.array(
        [complex(p["re"], p["im"]) if p["kind"] == "scalar" else 0 for p in graph["base_values"]]
    )
    parameters = np.broadcast_to(base, (len(packed), len(base))).copy()
    param_values = None
    if parameter_batch is not None:
        if parameter_slots is None:
            raise ValueError("parameter_slots are required with parameter_batch")
        param_values = np.asarray(parameter_batch, complex)
        if param_values.shape != (len(packed), len(slots)) or not np.isfinite(param_values).all():
            raise ValueError(
                "parameter_batch must be finite with shape (trajectories, parameter_slots)"
            )
        if any(graph["base_values"][slot]["kind"] != "scalar" for slot in slots):
            raise ValueError("CUDA scans support only scalar base parameters")
        parameters[:, slots] = param_values
    elif parameter_slots is not None:
        raise ValueError("parameter_slots were provided without parameter_batch")
    if not np.isfinite(parameters).all():
        raise ValueError("CUDA base parameters must be finite")
    compiler = ScalarCompiler(
        graph, [c["expression"] for c in prepared.hamiltonian_plan["coefficients"]]
    )
    varying = {k for k in slots if np.any(parameters[:, k] != parameters[0, k])}
    varied = np.array([bool(c.dependencies & varying) for c in compiler.coefficients])
    switches = compiler.discontinuities(parameters, span, options.discontinuities)
    events = np.union1d(times, density_times)
    grid, resets, limits = _steps(span, dt, switches, events, options.step_schedule, maxiters)
    cp = _cupy()
    if options.device_id >= cp.cuda.runtime.getDeviceCount():
        raise ValueError("CUDA device_id is unavailable")
    values = np.empty((len(packed), len(times), width), float)
    densities = np.empty((len(packed), len(density_times), n * n), float)
    start = time.perf_counter()
    stats = []
    with cp.cuda.Device(options.device_id):
        for lo in range(0, len(packed), options.batch_size):
            hi = min(lo + options.batch_size, len(packed))
            setup = time.perf_counter()
            g = _Integrator(
                cp,
                prepared,
                compiler,
                packed[lo:hi],
                parameters[lo:hi],
                varied,
                model,
                grid,
                resets,
                options,
                solver,
                weights,
                mode in {3, 5},
                span,
            )
            si = cp.asarray([i for i, j in indices], dtype=np.int32)
            sj = cp.asarray([j for i, j in indices], dtype=np.int32)
            # Separate bounded staging buffers for each requested output family.
            buffers = []
            for tt, ww, mm, host in [
                (times, width, mode, values),
                (density_times, n * n, 0, densities),
            ]:
                chunk = max(1, min(len(tt), 64_000_000 // max(1, g.B * ww * 8)))
                buffers.append(
                    dict(
                        times=tt,
                        width=ww,
                        mode=mm,
                        host=host,
                        buffer=cp.empty((chunk, g.B, ww)),
                        chunk=chunk,
                        used=0,
                        offset=0,
                    )
                )
            setup_s = time.perf_counter() - setup
            previous = 0
            for target in events:
                limit = limits[float(target)]
                g.advance(previous, limit)
                previous = limit
                for store in buffers:
                    k = store["offset"] + store["used"]
                    tt = store["times"]
                    if k < len(tt) and tt[k] == target:
                        g.launch(
                            "sample",
                            (
                                g.y,
                                store["buffer"],
                                g.public_pk,
                                g.public_im,
                                si,
                                sj,
                                g.weights,
                                np.int32(g.B),
                                np.int32(n),
                                np.int32(store["width"]),
                                np.int32(store["mode"]),
                                np.int32(store["used"]),
                            ),
                            g.B * store["width"],
                        )
                        store["used"] += 1
                        if store["used"] == store["chunk"] or k == len(tt) - 1:
                            data = store["buffer"][: store["used"]].get().transpose(1, 0, 2)
                            a = store["offset"]
                            store["host"][lo:hi, a : a + store["used"]] = data
                            store["offset"] += store["used"]
                            store["used"] = 0
            # Finish the declared interval even if its last requested output is earlier.
            g.advance(previous, len(grid))
            g.launch("validate", (g.y, np.int64(g.D), g.error), g.D)
            error = int(g.error.get()[0])
            if error:
                raise RuntimeError(
                    {
                        1: "nonfinite CUDA solution; reduce dt",
                        2: "complex argument passed to a real CUDA helper",
                        3: "CUDA Hamiltonian diagonal is not real",
                    }[error]
                )
            clock = g.clock.get()
            stats.append(
                dict(
                    trajectories=g.B,
                    steps_per_trajectory=int(clock[4]),
                    rhs_calls=int(clock[5]) * g.B,
                    shared_offdiagonal_hamiltonian=g.HB == 1 and not len(g.program.remainder_edges),
                    precomputed_waveforms=g.precompute,
                    tile_size=g.tile,
                    density_reduction=options.density_reduction,
                    retained_density_pairs=g.upper_count,
                    full_density_pairs=n * (n + 1) // 2,
                    setup_seconds=setup_s,
                    coefficient_sharing=g.program.cache_reason,
                    static_subexpressions=len(g.program.static),
                    shared_subexpressions=g.program.shared_count,
                    waveform_table_bytes=g.table.nbytes,
                    hamiltonian_bytes=g.hr.nbytes + g.hi.nbytes,
                    hamiltonian_sharing=g.program.h_reason,
                    shared_hamiltonian_components=len(g.program.amplitudes),
                    per_trajectory_hamiltonian_edges=len(g.program.remainder_edges),
                )
            )
            del g
    if not np.isfinite(values).all() or not np.isfinite(densities).all():
        raise RuntimeError("nonfinite CUDA output; reduce dt")
    if mode == 2:
        values = values[..., 0::2] + 1j * values[..., 1::2]
    if output_when == "final":
        values = values[:, 0]
    meta = {} if metadata is None else dict(metadata)
    meta.update(
        evolution_device="cuda",
        fixed_step=True,
        error_control=False,
        density_reduction=options.density_reduction,
    )
    if len(density_times):
        meta.update(density_times=density_times, density_values=densities)
    return LindbladBatchResult(
        times,
        values,
        output,
        None if output_indices is None else indices,
        len(packed),
        _parameter_slot_names(parameter_slots),
        param_values,
        dict(
            solver=solver,
            evolution_device="cuda",
            fixed_step=True,
            error_control=False,
            steps_per_trajectory=len(grid),
            accepted_steps=len(grid) * len(packed),
            rejected_steps=0,
            rhs_calls=sum(s["rhs_calls"] for s in stats),
            function_evaluations=sum(s["rhs_calls"] for s in stats),
            saved_points=len(times),
            total_seconds=time.perf_counter() - start,
            chunks=stats,
        )
        if collect_stats
        else None,
        meta,
    )


def _cuda_grid_scan(prepared: Any, rho0: Any, t_span: Any, scan: Any, kwargs: dict) -> Any:
    from .batch import _parameter_slot_names, solve_lindblad_batch

    axes = [np.asarray(a, complex).reshape(-1) for a in scan.values()]
    if not axes or any(not len(a) for a in axes):
        raise ValueError("scan axes must be nonempty")
    initial = kwargs.pop("rho0_batch", None)
    if initial is not None:
        if rho0 is not None:
            raise ValueError("pass rho0=None when supplying rho0_batch to a CUDA grid")
        packed = _packed_initial_states(prepared, initial)
    else:
        rho = np.asarray(rho0)
        packed = _packed_initial_states(prepared, rho[None])
    mesh = np.meshgrid(*axes, indexing="ij")
    parameters = np.stack([m.ravel() for m in mesh], axis=1)
    result = solve_lindblad_batch(
        prepared,
        np.tile(packed, (len(parameters), 1)),
        t_span,
        parameter_slots=list(scan),
        parameter_batch=np.repeat(parameters, len(packed), axis=0),
        **kwargs,
    )
    result.metadata.update(
        scan_kind="grid",
        grid_shape=tuple(len(a) for a in axes),
        grid_axes=dict(zip(_parameter_slot_names(list(scan)), axes, strict=False)),
        compact_grid=False,
        initial_condition_count=len(packed),
        trajectory_order="parameter_point_then_initial_state",
    )
    return result


def _solve_cuda_single(prepared: Any, rho0: Any, t_span: Any, **kwargs: Any) -> Any:
    from .batch import solve_lindblad_batch
    from .solve import LindbladObservableResult, LindbladResult

    rho = np.asarray(rho0)
    packed = rho[None]
    if kwargs.get("saveat") is None and kwargs.get("output_when") == "saveat":
        kwargs["saveat"] = np.unique(np.asarray(t_span, float))
    result = solve_lindblad_batch(prepared, packed, t_span, **kwargs)
    values = result.values[0]
    if result.output == "full":
        values = values.reshape(len(result.t), prepared.layout.packed_len)
        return LindbladResult(
            result.t, values, prepared.layout, result.solver_stats, metadata=result.metadata
        )
    answer = LindbladObservableResult(
        result.t,
        values,
        result.output,
        result.output_indices,
        result.solver_stats,
        metadata=result.metadata,
    )
    return answer
