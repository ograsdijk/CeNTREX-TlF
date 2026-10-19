"""Isolated GPU candidates and exact RC fixture; not a public solver API."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
SPB = ROOT.parent / "CeNTREX-SPB-testing"
QUARTER = 250e-9
METHODS = [
    "dense_two",
    "dense_one",
    "dense_split",
    "sparse_complex",
    "sparse_split",
    "fused_aos_major",
    "fused_aos_minor",
    "fused_soa_major",
    "fused_soa_minor",
    "staged_soa_major",
    "staged_soa_minor",
]


def load():
    pointer = json.loads(
        (SPB / ".cache/simulations/optical_bloch/native_rc_current.json").read_text(
            encoding="utf-8"
        )
    )
    source = Path(pointer["progress_path"]).parent / "prepared_context.pkl"
    return pickle.loads(source.read_bytes()), source, pointer


def points(count, pattern="grid"):
    if pattern == "grid":
        # Deterministic permutation: even small prefixes span the whole range.
        det = np.repeat(-20.0 + 0.25 * ((np.arange(161) * 61) % 161), 3)
    elif pattern == "full":
        det = np.repeat([-20.0, 0.0, 20.0], 3)
    else:
        det = np.repeat([-20.0, -1.0, 0.0, 20.0], 3)
    return np.resize(det, count), np.resize([0, 1, 2], count)


def fixture(context):
    q = context["prepared"]
    plan = q.hamiltonian_plan
    assert plan["kind"] == "decomposed" and len(plan["coefficients"]) == 8
    expected = [
        [(2, 3)],
        [(2, 5)],
        [(2, 17), (2, 20), (7, None)],
        [(2, 16), (2, 20), (7, None)],
        [(2, 18), (2, 21), (7, None)],
        [(2, 19), (2, 21), (7, None)],
        [(2, 4), (2, 15), (7, None)],
        [(2, 8)],
    ]
    actual = [
        [(x["op"], x.get("index")) for x in c["expression"]["instructions"]]
        for c in plan["coefficients"]
    ]
    if actual != expected:
        raise ValueError("Prepared coefficient mapping changed")
    basis = []
    for coefficient in plan["coefficients"]:
        b = np.zeros((q.layout.n, q.layout.n), complex)
        for term in coefficient["basis_terms"]:
            b[term["i"], term["j"]] += complex(term["re"], term["im"])
        basis.append(b)
    static = np.asarray(plan["static_matrix"])
    if np.any(static != np.diag(np.diag(static))):
        raise ValueError("Prototype requires diagonal static Hamiltonian")
    mask = np.diag(basis[0]).real
    if np.any(basis[0] != np.diag(mask)):
        raise ValueError("Detuning coefficient is not diagonal")
    full = np.asarray([b + np.triu(b, 1).conj().T for b in basis[2:7]])
    support = np.any(full != 0.0, axis=0)
    np.fill_diagonal(support, False)
    rows, columns = np.nonzero(support)
    ptr = np.r_[0, np.cumsum(support.sum(axis=1))].astype(np.int32)
    base = {
        name: complex(value["re"], value["im"]).real
        for name, value in zip(
            q.parameter_graph["slot_names"], q.parameter_graph["base_values"], strict=False
        )
    }
    tp, tc, tr = [0], [], []
    for entries in q.incoming_transfers_by_target:
        for entry in entries:
            tc.append(entry["source"])
            tr.append(entry["rate"])
        tp.append(len(tc))
    return {
        "basis": full,
        "diagonal": np.diag(static).real.copy(),
        "mask": mask,
        "rows": rows.astype(np.int32),
        "columns": columns.astype(np.int32),
        "ptr": ptr,
        "base": base,
        "tp": np.asarray(tp, np.int32),
        "tc": np.asarray(tc, np.int32),
        "tr": np.asarray(tr, float),
        "rates": np.asarray(q.source_decay_rates),
    }


def hamiltonian(f, context, t, velocity=150.0, switch_time=None):
    base = f["base"]
    duration = context["physics"]["uv_length_m"] / velocity
    sigma = np.sqrt(2) * context["sigma_I"] / velocity
    envelope = np.exp(-0.5 * ((t - duration / 2) / sigma) ** 2)
    quarter = int(np.floor((t if switch_time is None else switch_time) / QUARTER + 1e-8)) % 4
    a, b = ((1, 1), (1, 0), (0, 0), (0, 1))[quarter]
    coeff = [
        (1 - a) * base["mw_peak1"] * envelope,
        a * base["mw_peak1"] * envelope,
        b * base["mw_peak2"] * envelope,
        (1 - b) * base["mw_peak2"] * envelope,
        base["PUVZ0"] * base["Ωt0"] * np.exp(1j * base["β0"] * np.sin(base["ωphase0"] * t)),
    ]
    upper = sum(c * np.triu(matrix) for c, matrix in zip(coeff, f["basis"], strict=True))
    return np.diag(f["diagonal"]) + upper + np.triu(upper, 1).conj().T


def numpy_rhs(f, context, y, det, t):
    h = hamiltonian(f, context, t)
    hz = h[None] + 2 * np.pi * 1e6 * det[:, None, None] * np.diag(f["mask"])[None]
    result = -1j * (hz @ y - y @ hz) - 0.5 * (f["rates"][:, None] + f["rates"][None, :]) * y
    for i in range(len(f["tp"]) - 1):
        for p in range(f["tp"][i], f["tp"][i + 1]):
            result[:, i, i] += f["tr"][p] * y[:, f["tc"][p], f["tc"][p]]
    return result


class GPU:
    def __init__(
        self, context, method, batch, start, end, substeps=256, velocity=150.0, pattern="grid"
    ):
        if method not in METHODS:
            raise ValueError(method)
        self.context, self.method, self.batch = context, method, batch
        self.n = context["prepared"].layout.n
        self.size = batch * self.n * self.n
        self.soa = "soa" in method or method in ("dense_split", "sparse_split")
        self.minor = "minor" in method or method.startswith("sparse_")
        self.dense = method.startswith("dense_")
        self.f = f = fixture(context)
        self.det_host, js = points(batch, pattern)
        self.initial_host = context["rho0"][js]
        self.start, self.end = start, end
        self.dt = QUARTER / substeps
        self.clock_initial = np.array([start, self.dt, end, self.dt], float)
        self.clock = cp.asarray(self.clock_initial)
        n = self.n
        source = (ROOT / "benchmarks/rc_gpu_sparse_kernels.cu").read_text(encoding="utf-8")
        module = cp.RawModule(code=source, options=("--std=c++14",))
        self.kernels = {
            name: module.get_function(name)
            for name in [
                "begin_step",
                "advance",
                "coefficients",
                "build_h",
                "fused_rhs",
                "sparse_product",
                "finish_rhs",
                "finish_two",
                "combine",
            ]
        }
        self.br = cp.asarray(np.ascontiguousarray(f["basis"].real))
        self.bi = cp.asarray(np.ascontiguousarray(f["basis"].imag))
        self.diag, self.mask, self.rates = [cp.asarray(f[k]) for k in ["diagonal", "mask", "rates"]]
        self.ptr, self.col, self.tp, self.tc, self.tr = [
            cp.asarray(f[k]) for k in ["ptr", "columns", "tp", "tc", "tr"]
        ]
        self.det = cp.asarray(2 * np.pi * 1e6 * self.det_host)
        positions = np.arange(n * n, dtype=np.int32) if self.dense else f["rows"] * n + f["columns"]
        self.positions = cp.asarray(positions)
        self.hz = cp.empty(len(positions), cp.complex128)
        self.hr, self.hi = cp.empty(len(positions), float), cp.empty(len(positions), float)
        self.cr, self.ci = cp.empty(5, float), cp.empty(5, float)
        base = f["base"]
        self.p = cp.asarray(
            np.array(
                [
                    context["physics"]["uv_length_m"] / velocity / 2,
                    np.sqrt(2) * context["sigma_I"] / velocity,
                    base["mw_peak1"],
                    base["mw_peak2"],
                    base["PUVZ0"] * base["Ωt0"],
                    base["β0"],
                    base["ωphase0"],
                ]
            )
        )
        if method == "sparse_complex":
            self.sparse_z = csr_matrix((self.hz, self.col, self.ptr), shape=(n, n))
        if method == "sparse_split":
            self.sparse_r = csr_matrix((self.hr, self.col, self.ptr), shape=(n, n))
            self.sparse_i = csr_matrix((self.hi, self.col, self.ptr), shape=(n, n))
        self.ar, self.ai = cp.empty(self.size, float), cp.empty(self.size, float)
        self.y = self.upload(self.initial_host)
        self.initial = self.y.copy()
        self.temp = cp.empty_like(self.y)
        self.ks = [cp.empty_like(self.y) for _ in range(4)]
        self.reconstruct = (
            self.diag,
            self.mask,
            self.det,
            self.rates,
            self.tp,
            self.tc,
            self.tr,
            np.int32(batch),
            np.int32(n),
            np.int32(self.soa),
            np.int32(self.minor),
        )
        self.graph = None

    def launch(self, name, args, count=None):
        if count is None:
            count = self.size
        self.kernels[name](((count + 255) // 256,), (256,), args)

    def upload(self, host):
        host = host.transpose(1, 2, 0) if self.minor else host
        host = np.ascontiguousarray(host).reshape(-1)
        data = np.stack((host.real, host.imag)) if self.soa else host
        return cp.asarray(np.ascontiguousarray(data))

    def download(self, y=None):
        y = self.y if y is None else y
        a = cp.asnumpy(y)
        if self.soa:
            a = a[0] + 1j * a[1]
        if self.minor:
            return a.reshape(self.n, self.n, self.batch).transpose(2, 0, 1).copy()
        return a.reshape(self.batch, self.n, self.n)

    def reset(self):
        cp.copyto(self.y, self.initial)
        self.clock.set(self.clock_initial)

    def rhs(self, y, out, fraction):
        self.kernels["coefficients"](
            (1,), (1,), (self.clock, self.p, np.float64(fraction), self.cr, self.ci)
        )
        self.launch(
            "build_h",
            (
                self.positions,
                self.br,
                self.bi,
                self.diag,
                self.cr,
                self.ci,
                self.hz,
                self.hr,
                self.hi,
                np.int32(self.positions.size),
                np.int32(self.n),
                np.int32(self.dense),
            ),
            self.positions.size,
        )
        method = self.method
        if method.startswith("fused_"):
            self.launch(
                "fused_rhs", (y, out, self.ptr, self.col, self.hr, self.hi, *self.reconstruct)
            )
            return
        if method.startswith("staged_"):
            self.launch(
                "sparse_product",
                (
                    y,
                    self.ar,
                    self.ai,
                    self.ptr,
                    self.col,
                    self.hr,
                    self.hi,
                    np.int32(self.batch),
                    np.int32(self.n),
                    np.int32(self.soa),
                    np.int32(self.minor),
                ),
            )
            self.launch(
                "finish_rhs",
                (y, out, self.ar, self.ai, self.hz, np.int32(0), *self.reconstruct, np.int32(1)),
            )
            return
        if method in ("dense_two", "dense_one"):
            yy = y.reshape(self.batch, self.n, self.n)
            hh = self.hz.reshape(self.n, self.n)
            aa = hh @ yy
            if method == "dense_two":
                comm = cp.ascontiguousarray(aa - yy @ hh)
                self.launch(
                    "finish_two",
                    (
                        y,
                        out,
                        comm,
                        self.mask,
                        self.det,
                        self.rates,
                        self.tp,
                        self.tc,
                        self.tr,
                        np.int32(self.batch),
                        np.int32(self.n),
                    ),
                )
                return
            az = cp.ascontiguousarray(aa)
            self.launch(
                "finish_rhs",
                (y, out, self.ar, self.ai, az, np.int32(1), *self.reconstruct, np.int32(0)),
            )
            return
        if method == "dense_split":
            yr, yi = y.reshape(2, self.batch, self.n, self.n)
            hr, hi = self.hr.reshape(self.n, self.n), self.hi.reshape(self.n, self.n)
            ar = cp.ascontiguousarray(hr @ yr - hi @ yi)
            ai = cp.ascontiguousarray(hr @ yi + hi @ yr)
        elif method == "sparse_complex":
            az = cp.ascontiguousarray(self.sparse_z @ y.reshape(self.n, self.n * self.batch))
            self.launch(
                "finish_rhs",
                (y, out, self.ar, self.ai, az, np.int32(1), *self.reconstruct, np.int32(1)),
            )
            return
        elif method == "sparse_split":
            yr, yi = y.reshape(2, self.n, self.n * self.batch)
            ar = cp.ascontiguousarray(self.sparse_r @ yr - self.sparse_i @ yi)
            ai = cp.ascontiguousarray(self.sparse_r @ yi + self.sparse_i @ yr)
        else:
            raise ValueError(method)
        self.launch(
            "finish_rhs",
            (y, out, ar, ai, self.hz, np.int32(0), *self.reconstruct, np.int32(not self.dense)),
        )

    def block(self, steps=16):
        k1, k2, k3, k4 = self.ks
        for _ in range(steps):
            self.kernels["begin_step"]((1,), (1,), (self.clock,))
            self.rhs(self.y, k1, 0.0)
            for dest, weights, k_out, fraction in [
                (self.temp, (0.5, 0.0, 0.0, 0.0), k2, 0.5),
                (self.temp, (0.0, 0.5, 0.0, 0.0), k3, 0.5),
                (self.temp, (0.0, 0.0, 1.0, 0.0), k4, 1.0),
            ]:
                self.launch(
                    "combine",
                    (
                        dest,
                        self.y,
                        k1,
                        k2,
                        k3,
                        k4,
                        self.clock,
                        *[np.float64(x) for x in weights],
                        np.int64(2 * self.size),
                    ),
                    2 * self.size,
                )
                self.rhs(self.temp, k_out, fraction)
            self.launch(
                "combine",
                (
                    self.y,
                    self.y,
                    k1,
                    k2,
                    k3,
                    k4,
                    self.clock,
                    np.float64(1 / 6),
                    np.float64(1 / 3),
                    np.float64(1 / 3),
                    np.float64(1 / 6),
                    np.int64(2 * self.size),
                ),
                2 * self.size,
            )
            self.kernels["advance"]((1,), (1,), (self.clock,))

    def capture(self, steps=16, rhs_only=False):
        self.reset()
        if rhs_only:
            self.rhs(self.y, self.ks[0], 0.0)
        else:
            self.block(steps)
        cp.cuda.get_current_stream().synchronize()
        self.reset()
        # CuPy 14 explicitly rejects cuBLAS/cuSPARSE calls during graph capture.
        # Keep those library candidates as labeled eager implementations.
        self.eager = self.method.startswith(("dense_", "sparse_"))
        if self.eager:
            self.graph = None
            self.chunk = 1
            self.rhs_only = rhs_only
            return
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            if rhs_only:
                self.rhs(self.y, self.ks[0], 0.0)
            else:
                self.block(steps)
            self.graph = stream.end_capture()
        self.chunk = steps
        self.reset()
        cp.cuda.get_current_stream().synchronize()

    def evolve(self, repeats=3):
        results = []
        launches = int(np.ceil((self.end - self.start) / self.dt / self.chunk)) + 1
        for _ in range(repeats):
            self.reset()
            cp.cuda.get_current_stream().synchronize()
            tick = time.perf_counter()
            for _ in range(launches):
                if self.eager:
                    self.block(1)
                else:
                    self.graph.launch()
            cp.cuda.get_current_stream().synchronize()
            results.append(time.perf_counter() - tick)
        return self.download(), results, float(self.clock.get()[0])

    def full_transit(self):
        """Production's 201 population samples, field exit and post-decay density."""
        self.reset()
        populations = [np.diagonal(self.initial_host, axis1=-2, axis2=-1).real.copy()]
        started = time.perf_counter()
        previous = 0.0
        for target in np.linspace(0.0, self.end, 201)[1:]:
            self.clock.set(np.array([previous, self.dt, target, self.dt]))
            launches = int(np.ceil((target - previous) / self.dt / self.chunk)) + 2
            for _ in range(launches):
                self.graph.launch()
            actual = float(self.clock.get()[0])
            if abs(actual - target) > 1e-13:
                raise RuntimeError(f"Population sample incomplete: {actual} vs {target}")
            populations.append(np.diagonal(self.download(), axis1=-2, axis2=-1).real.copy())
            previous = target
        exit_rho = self.download()
        exit_wall = time.perf_counter() - started
        # Reuse the captured arithmetic but turn all amplitudes off at field exit.
        from centrex_tlf import hamiltonian

        end = self.end + float(self.context["physics"]["post_decay_lifetimes"]) / float(
            hamiltonian.Γ
        )
        p = self.p.get()
        p[2:5] = 0.0
        self.p.set(p)
        self.clock.set(np.array([self.end, self.dt, end, self.dt]))
        for _ in range(int(np.ceil((end - self.end) / self.dt / self.chunk)) + 2):
            self.graph.launch()
        final = self.download()
        return (
            np.stack((exit_rho, final), axis=1),
            np.stack(populations, axis=1),
            {
                "field_exit_wall_s_including_sampling": exit_wall,
                "full_wall_s_including_sampling_and_post_decay": time.perf_counter() - started,
                "post_decay_end_s": float(self.clock.get()[0]),
            },
        )


def diagnostics(y, reference):
    diff = np.abs(y - reference)
    mask = np.abs(reference) >= 1e-6
    off = mask.copy()
    off[:, np.arange(y.shape[-1]), np.arange(y.shape[-1])] = False
    ratios = np.where(mask, diff / np.maximum(np.abs(reference), 1e-300), 0.0)
    return {
        "max_abs": float(diff.max()),
        "max_relative_above_cutoff": float(ratios.max()),
        "max_coherence_relative_above_cutoff": float(ratios[off].max(initial=0.0)),
        "worst_per_trajectory_relative": ratios.max(axis=(1, 2)).tolist(),
        "near_zero_max_abs": float(diff[~mask].max(initial=0.0)),
        "relative_frobenius": float(
            np.max(
                np.linalg.norm(y - reference, axis=(1, 2)) / np.linalg.norm(reference, axis=(1, 2))
            )
        ),
        "trace_error": float(np.abs(np.trace(y, axis1=-2, axis2=-1) - 1).max()),
        "hermiticity_error": float(np.abs(y - y.conj().swapaxes(-1, -2)).max()),
        "min_eigenvalue": float(np.linalg.eigvalsh(y).min()),
        "passes_1percent_above_cutoff": bool(ratios.max() <= 0.01),
    }
