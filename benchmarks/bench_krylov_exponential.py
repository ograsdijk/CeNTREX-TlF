"""Krylov exponential-integrator spike (audit item 7 decision gate).

Audit item 7 proposes replacing dopri5 stepping with an exponential / Lawson
integrator that propagates the fast static linear part exactly, so the stepper
only resolves the slow envelope. The estimate attached to it was 10-25x.

This script tests the cost model behind that estimate, on the same
r2-in-static-E-field system used by ``diagnose_step_size.py``.

The claim under test
--------------------
Any exponential method must apply ``exp(L0*h)`` to a vector. Two routes:

* dense (eig / expm): two dense ``dim x dim`` matvecs per application,
* Krylov (``expm_multiply``): preserves sparsity, but the Krylov dimension
  needed scales with ``||L0*h||``.

If the Krylov dimension scales linearly with ``h``, the work to cover a fixed
span ``T`` is *independent of the step size* -- bigger steps just cost
proportionally more each -- and the method cannot beat explicit stepping by
more than a constant factor. Both scale as O(omega*T) matvecs.

So the decisive measurement is: **projected total wall time to cover T, as a
function of h.** Flat => the O(omega*T) floor is real and item 7's estimate is
wrong. Decreasing => the estimate may survive.

Matvecs are counted exactly (not inferred from timings) by wrapping L in a
counting ``LinearOperator``.

Usage::

    uv run python benchmarks/bench_krylov_exponential.py
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from centrex_tlf import hamiltonian, lindblad
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem

import diagnose_step_size as diag  # reuse the exact system definition

RESULTS_DIR = Path(__file__).parent / "krylov_exponential_results"
GAMMA = getattr(hamiltonian, "Γ")

DETUNING_MHZ = 25.0  # the opposite-parity peak: the physically interesting case
RELTOL = 1e-7
ABSTOL = 1e-9
T_END = diag.T_END

# step sizes to probe, from ~the current dopri5 step up to ~T/100
H_VALUES = [5e-9, 1e-8, 2.5e-8, 5e-8, 1e-7, 2.5e-7, 5e-7, 1e-6]


class CountingOperator(scipy.sparse.linalg.LinearOperator):
    """Wraps a sparse matrix and counts matvec / matmat columns applied."""

    def __init__(self, A):
        self.A = A
        self.AT = A.T.tocsr()
        self.count = 0
        self.rcount = 0  # norm-estimation traffic, kept out of the headline count
        super().__init__(dtype=A.dtype, shape=A.shape)

    def _matvec(self, x):
        self.count += 1
        return self.A @ x

    def _matmat(self, X):
        self.count += X.shape[1]
        return self.A @ X

    def _rmatvec(self, x):
        self.rcount += 1
        return self.AT @ x

    def _rmatmat(self, X):
        self.rcount += X.shape[1]
        return self.AT @ X


def build():
    system, ts = diag.build_system()
    rabi_value = diag.power_to_rabi_rectangular_beam(
        diag.POWER_W, abs(system.couplings[0].main_coupling), diag.BEAM_WX, diag.BEAM_WY
    )
    params = diag.make_parameters(system, ts, rabi_value, 2 * np.pi * 1e6 * DETUNING_MHZ)
    prepared = prepare_lindblad_problem(
        system, params, backend="rust", hamiltonian_representation="decomposed"
    )
    return system, prepared, rabi_value


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Building r2-in-static-E-field system ...")
    system, prepared, rabi_value = build()
    n = len(system.QN)
    rho0 = diag.build_rho0(system)

    ev = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "expanded_sparse", True)
    dim = prepared.layout.packed_len
    x0 = np.zeros(dim, dtype=np.float64)
    x0[:n] = np.real(np.diag(rho0))  # packed layout is diagonal-first
    print(f"  n_states = {n}, packed dim = {dim}, detuning = {DETUNING_MHZ} MHz")

    # ---- extract sparse L via the analytic Jacobian -------------------------
    t0 = time.perf_counter()
    rows, cols, vals = ev.jacobian_packed_sparse_py(0.0, 0.0, "analytic")
    L = scipy.sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(dim, dim)
    )
    t_extract = time.perf_counter() - t0
    print(f"  analytic L extract: {t_extract * 1e3:.2f} ms, nnz = {L.nnz}")

    rhs = np.asarray(ev.rhs_packed_py(x0, 0.0))
    err = np.max(np.abs(L @ x0 - rhs)) / max(np.max(np.abs(rhs)), 1.0)
    print(f"  L @ x0 vs rhs relative error: {err:.2e}")

    # Frequency content. NOTE: the packed-real diagonal of L holds decay rates,
    # not phases -- the oscillation frequencies (E_i - E_j) sit in the 2x2
    # re/im blocks of each coherence. So the relevant scale is max |Im lambda|,
    # and scipy's expm_multiply scaling parameter keys off ||A||_1.
    Ld = L.toarray()
    t0 = time.perf_counter()
    w, V = np.linalg.eig(Ld)
    t_eig = time.perf_counter() - t0
    omega = float(np.max(np.abs(w.imag)))
    norm1 = float(np.abs(Ld).sum(axis=0).max())
    # the "active" scale: spectators carry no coherence and never limit dopri5
    active = np.sort(np.abs(w.imag))[::-1]
    print(f"  max |Im lambda| = {omega:.3e} rad/s ({omega / (2 * np.pi * 1e6):.1f} MHz)")
    print(f"  ||L||_1         = {norm1:.3e} rad/s ({norm1 / (2 * np.pi * 1e6):.1f} MHz)")
    print(f"  eig({dim}) took {t_eig:.2f} s")

    # ---- single sparse matvec cost -----------------------------------------
    v = x0.copy()
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        L @ v
    t_matvec = (time.perf_counter() - t0) / reps
    print(f"  single sparse matvec: {t_matvec * 1e6:.2f} us")

    # ---- dopri5 baseline ----------------------------------------------------
    print("\ndopri5 baseline ...")
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = lindblad.solve_lindblad(
            prepared,
            rho0,
            (0.0, T_END),
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="populations",
            reltol=RELTOL,
            abstol=ABSTOL,
        )
        samples.append(time.perf_counter() - t0)
    t_dopri = statistics.median(samples)
    stats = getattr(result, "solver_stats", {}) or {}
    ref_pop = np.asarray(result.values)[-1].real
    print(f"  wall {t_dopri:.3f} s   stats: {stats}")

    # ---- Krylov cost vs step size ------------------------------------------
    print("\nexpm_multiply(L*h, v): cost per application vs h")
    header = (
        f"{'h [ns]':>10} {'omega*h':>9} {'matvecs':>9} {'t_apply[ms]':>12}"
        f" {'steps':>8} {'projected[s]':>13} {'vs dopri5':>10}"
    )
    print(header)
    traceA = float(L.diagonal().sum())
    rowsout = []
    for h in H_VALUES:
        Ah = (L * h).tocsr()
        # Timed with the plain sparse matrix (production-realistic: scipy then
        # uses the exact 1-norm rather than onenormest).
        t0 = time.perf_counter()
        scipy.sparse.linalg.expm_multiply(Ah, v, traceA=traceA * h)
        t_apply = time.perf_counter() - t0
        # Counted separately, only when that second pass is affordable.
        if t_apply < 20.0:
            op = CountingOperator(Ah)
            scipy.sparse.linalg.expm_multiply(op, v, traceA=traceA * h)
            matvecs = float(op.count)
        else:
            matvecs = float("nan")
        steps = T_END / h
        projected = steps * t_apply
        rowsout.append(
            {
                "h_s": h,
                "omega_h": omega * h,
                "matvecs": matvecs,
                "t_apply_s": t_apply,
                "steps": steps,
                "projected_s": projected,
                "speedup_vs_dopri5": t_dopri / projected,
            }
        )
        print(
            f"{h * 1e9:>10.1f} {omega * h:>9.1f} {matvecs:>9.0f} {t_apply * 1e3:>12.3f}"
            f" {steps:>8.0f} {projected:>13.1f} {t_dopri / projected:>10.3f}x"
        )
        if t_apply > 60.0:
            print("  (stopping scan: a single application now exceeds 60 s)")
            break

    # ---- accuracy of repeated-exp stepping at a moderate h ------------------
    # Done over a SHORT sub-span: repeated Krylov application over the full
    # 108.7 us would take ~20 min, which is itself the headline result.
    h_acc = 1e-8
    nsteps = 100
    t_acc_span = h_acc * nsteps
    print(f"\naccuracy check: {nsteps} x expm_multiply(L*{h_acc:.0e}) "
          f"= {t_acc_span * 1e6:.1f} us sub-span ...")
    short = lindblad.solve_lindblad(
        prepared, rho0, (0.0, t_acc_span), solver="dopri5",
        execution_mode="expanded_sparse", output="populations",
        reltol=1e-10, abstol=1e-12,
    )
    short_ref = np.asarray(short.values)[-1].real
    t0 = time.perf_counter()
    short_dopri = time.perf_counter()
    for _ in range(3):
        lindblad.solve_lindblad(
            prepared, rho0, (0.0, t_acc_span), solver="dopri5",
            execution_mode="expanded_sparse", output="populations",
            reltol=RELTOL, abstol=ABSTOL,
        )
    t_short_dopri = (time.perf_counter() - short_dopri) / 3

    Ah = (L * h_acc).tocsr()
    trace_h = float(Ah.diagonal().sum())
    t0 = time.perf_counter()
    x = x0.copy()
    for _ in range(nsteps):
        x = scipy.sparse.linalg.expm_multiply(Ah, x, traceA=trace_h)
    t_acc = time.perf_counter() - t0
    kry_pop = x[:n].copy()
    max_diff = float(np.max(np.abs(kry_pop - short_ref)))
    print(f"  krylov {t_acc:.2f} s vs dopri5 {t_short_dopri:.4f} s "
          f"-> {t_short_dopri / t_acc:.4f}x")
    print(f"  max |pop_krylov - pop_dopri5(reltol 1e-10)| = {max_diff:.3e}")

    # ---- dense eig route, for comparison ------------------------------------
    print("\ndense eig route (exact propagator machinery) ...")
    t0 = time.perf_counter()
    Vinv = np.linalg.inv(V)
    t_inv = time.perf_counter() - t0
    c = Vinv @ x0
    t0 = time.perf_counter()
    for _ in range(20):
        _ = (V @ (np.exp(w * T_END) * c)).real
    t_apply_dense = (time.perf_counter() - t0) / 20
    print(f"  eig {t_eig:.2f} s + inv {t_inv:.2f} s = setup {t_eig + t_inv:.2f} s")
    print(f"  per application (any t): {t_apply_dense * 1e3:.2f} ms")
    exact_pop = (V @ (np.exp(w * T_END) * c)).real[:n]
    dense_diff = float(np.max(np.abs(exact_pop - ref_pop)))
    print(f"  max |pop_exact - pop_dopri5| = {dense_diff:.3e}")

    summary = {
        "n_states": n,
        "packed_dim": dim,
        "nnz": int(L.nnz),
        "detuning_MHz": DETUNING_MHZ,
        "reltol": RELTOL,
        "T_end_s": T_END,
        "analytic_extract_s": t_extract,
        "L_rhs_rel_error": float(err),
        "max_abs_im_lambda_rad_s": omega,
        "L_norm1_rad_s": norm1,
        "top_10_abs_im_lambda_rad_s": [float(x) for x in active[:10]],
        "sparse_matvec_s": t_matvec,
        "dopri5_wall_s": t_dopri,
        "dopri5_stats": stats,
        "krylov_scan": rowsout,
        "krylov_stepping_h_s": h_acc,
        "krylov_stepping_nsteps": nsteps,
        "krylov_stepping_span_s": t_acc_span,
        "krylov_stepping_wall_s": t_acc,
        "krylov_stepping_dopri5_wall_s": t_short_dopri,
        "krylov_stepping_max_pop_diff": max_diff,
        "dense_eig_s": t_eig,
        "dense_inv_s": t_inv,
        "dense_apply_s": t_apply_dense,
        "dense_max_pop_diff": dense_diff,
    }
    (RESULTS_DIR / "results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    with (RESULTS_DIR / "krylov_cost_vs_step.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rowsout[0].keys()))
        wr.writeheader()
        wr.writerows(rowsout)
    print(f"\nWrote {RESULTS_DIR}")


if __name__ == "__main__":
    main()
