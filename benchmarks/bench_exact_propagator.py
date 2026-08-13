"""Exact-propagator feasibility for the r2-in-static-E-field system.

At fixed scan parameters the packed-real Liouvillian L of this system is
TIME-INDEPENDENT (see benchmarks/step_size_diagnostics_results/
step_size_diagnostics_report.md), so instead of ~24k dopri5 steps per
trajectory we can in principle:

  1. extract L once per grid point via the exact-Jacobian probe,
  2. eigendecompose it (L = V diag(w) V^-1),
  3. propagate any initial state to any set of times analytically:
     x(t) = V (exp(w t) * c),  c = V^-1 x0,
  4. get the photon integral analytically from Int_0^T exp(w t) dt.

This benchmark measures whether that is (a) numerically accurate for this
non-normal L and (b) faster than stepping. Also times a
scipy.linalg.expm(L*dt) + repeated-matvec alternative.

Writes CSVs and contributes to
benchmarks/scan_speedup_results/exact_propagator_and_threads.md.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnose_step_size as diag  # noqa: E402

from centrex_tlf import hamiltonian, lindblad, states  # noqa: E402
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py  # noqa: E402
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem  # noqa: E402
from centrex_tlf.utils.rabi import power_to_rabi_rectangular_beam  # noqa: E402

RESULTS_DIR = HERE / "scan_speedup_results"
GAMMA = float(getattr(hamiltonian, "Γ"))

DETUNINGS_MHZ = [0.0, 25.0]
N_SAVEAT = 801
T_END = diag.T_END  # ~108.7 us
REF_RELTOL = 1e-9
REF_ABSTOL = 1e-11
NOTEBOOK_RELTOL = 1e-7
NOTEBOOK_ABSTOL = 1e-9


def excited_indices(system) -> list[int]:
    return [
        idx
        for idx, state in enumerate(system.QN)
        if state.largest.electronic_state == states.ElectronicState.B
    ]


def extract_liouvillian(prepared, packed_rho0: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract the packed-real Liouvillian, verify time-independence and RHS parity."""
    ev = create_lindblad_rhs_evaluator_py(prepared.rust_plan, "expanded_sparse", True)
    dim = prepared.layout.packed_len

    t0 = time.perf_counter()
    rows, cols, vals = ev.jacobian_packed_sparse_py(0.0)
    L = scipy.sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(dim, dim)
    ).toarray()
    t_jac = time.perf_counter() - t0

    # time-independence probe at an arbitrary interior time
    rows2, cols2, vals2 = ev.jacobian_packed_sparse_py(3.7e-5)
    if not (
        np.array_equal(np.asarray(rows), np.asarray(rows2))
        and np.array_equal(np.asarray(cols), np.asarray(cols2))
        and np.array_equal(np.asarray(vals), np.asarray(vals2))
    ):
        raise AssertionError(
            "Liouvillian is NOT time-independent: jacobian at t=0 and t=3.7e-5 differ"
        )

    # sanity: L @ rho0 must equal the RHS (system is purely linear, no affine part)
    rhs = np.asarray(ev.rhs_packed_py(packed_rho0, 0.0))
    lhs = L @ packed_rho0
    scale = float(np.max(np.abs(rhs)))
    if not np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12 * max(scale, 1.0)):
        max_diff = float(np.max(np.abs(lhs - rhs)))
        raise AssertionError(
            f"L @ rho0 != rhs(rho0): max abs diff {max_diff:.3e} (rhs scale {scale:.3e})"
        )
    return L, t_jac


def phi_integral(w: np.ndarray, T: float) -> np.ndarray:
    """(exp(w*T) - 1)/w with a series fallback for |w*T| ~ 0."""
    wT = w * T
    small = np.abs(wT) < 1e-10
    out = np.empty_like(w)
    safe = np.where(small, 1.0, w)
    out = (np.exp(wT) - 1.0) / safe
    out[small] = T * (1.0 + 0.5 * wT[small])
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Building r2-in-E-field system (notebook-identical)...")
    t0 = time.perf_counter()
    system, ts = diag.build_system()
    n = len(system.QN)
    print(f"  built in {time.perf_counter() - t0:.2f} s, n_states={n}")

    rabi_value = power_to_rabi_rectangular_beam(
        diag.POWER_W, abs(system.couplings[0].main_coupling), diag.BEAM_WX, diag.BEAM_WY
    )
    rho0 = diag.build_rho0(system)
    exc = excited_indices(system)
    weights = [(int(idx), GAMMA) for idx in exc]
    saveat = np.linspace(0.0, T_END, N_SAVEAT)
    rows_out: list[dict] = []

    for detuning_mhz in DETUNINGS_MHZ:
        print(f"\n=== detuning {detuning_mhz:.1f} MHz ===")
        detuning_rad = 2 * np.pi * detuning_mhz * 1e6
        params = diag.make_parameters(system, ts, rabi_value, detuning_rad)
        prepared = prepare_lindblad_problem(
            system, params, backend="rust", hamiltonian_representation="decomposed"
        )
        dim = prepared.layout.packed_len
        # verify diagonal-first packed layout: population i lives at packed index i
        assert all(prepared.layout.diagonal_index(i) == i for i in range(n))
        packed_rho0 = prepared.layout.pack(rho0)

        L, t_jac = extract_liouvillian(prepared, packed_rho0)
        print(f"  L extracted: dim={dim}, nnz={np.count_nonzero(L)}, "
              f"jacobian+dense time {t_jac*1e3:.1f} ms; time-independence OK; RHS parity OK")

        # --- eigendecomposition ---
        t0 = time.perf_counter()
        w, V = np.linalg.eig(L)
        t_eig = time.perf_counter() - t0

        norm_L = float(np.linalg.norm(L))
        t0 = time.perf_counter()
        residual = float(np.linalg.norm(L @ V - V * w[None, :]) / norm_L)
        t_resid = time.perf_counter() - t0
        t0 = time.perf_counter()
        cond_V = float(np.linalg.cond(V))
        t_cond = time.perf_counter() - t0
        print(f"  eig: {t_eig:.2f} s; residual ||LV-Vw||/||L|| = {residual:.3e}; "
              f"cond(V) = {cond_V:.3e} (resid {t_resid:.2f} s, cond {t_cond:.2f} s)")

        t0 = time.perf_counter()
        lu, piv = scipy.linalg.lu_factor(V)
        t_lu = time.perf_counter() - t0

        # --- analytic propagation for one initial state, 801 saveat points ---
        t0 = time.perf_counter()
        c = scipy.linalg.lu_solve((lu, piv), packed_rho0.astype(np.complex128))
        t_solve_c = time.perf_counter() - t0

        t0 = time.perf_counter()
        E = np.exp(np.outer(w, saveat))  # (dim, n_t)
        X = V @ (E * c[:, None])  # (dim, n_t)
        pops_analytic = X[:n, :].T.real  # (n_t, n)
        t_prop = time.perf_counter() - t0

        max_imag = float(np.max(np.abs(X[:n, :].imag)))
        print(f"  analytic propagation: solve_c {t_solve_c*1e3:.1f} ms, "
              f"exp+matmul {t_prop*1e3:.1f} ms for {N_SAVEAT} points "
              f"(max |Im pop| = {max_imag:.2e})")

        # --- analytic photon integral ---
        t0 = time.perf_counter()
        phi = phi_integral(w, T_END)
        photon_analytic = GAMMA * float(np.real(V[exc, :] @ (phi * c)).sum())
        t_photon = time.perf_counter() - t0

        # --- reference dopri5 solves ---
        t0 = time.perf_counter()
        ref = lindblad.solve_lindblad(
            prepared,
            rho0,
            (0.0, T_END),
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="populations",
            output_when="saveat",
            saveat=saveat,
            abstol=REF_ABSTOL,
            reltol=REF_RELTOL,
            dt=1e-10,
        )
        t_ref_pops = time.perf_counter() - t0
        pops_ref = np.asarray(ref.values, dtype=np.float64).reshape(-1, n)
        # align lengths (save_start included in both)
        assert pops_ref.shape[0] == N_SAVEAT, pops_ref.shape

        t0 = time.perf_counter()
        ref_photon = lindblad.solve_lindblad(
            prepared,
            rho0,
            (0.0, T_END),
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="photon_integral",
            output_when="final",
            dense_output=False,
            integral_weights=weights,
            abstol=REF_ABSTOL,
            reltol=REF_RELTOL,
            dt=1e-10,
        )
        t_ref_photon = time.perf_counter() - t0
        photon_ref = float(np.asarray(ref_photon.values).reshape(-1)[0].real)

        pops_max_diff = float(np.max(np.abs(pops_analytic - pops_ref)))
        photon_rel_diff = abs(photon_analytic - photon_ref) / abs(photon_ref)
        print(f"  accuracy vs dopri5 reltol={REF_RELTOL:.0e}: "
              f"populations max abs diff = {pops_max_diff:.3e}; "
              f"photon integral: analytic {photon_analytic:.8f} vs ref {photon_ref:.8f} "
              f"(rel diff {photon_rel_diff:.3e})")

        # --- notebook-tolerance dopri5 timing (scan-shaped trajectory) ---
        dopri_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            lindblad.solve_lindblad(
                prepared,
                rho0,
                (0.0, T_END),
                solver="dopri5",
                execution_mode="expanded_sparse",
                output="photon_integral",
                output_when="final",
                dense_output=False,
                integral_weights=weights,
                abstol=NOTEBOOK_ABSTOL,
                reltol=NOTEBOOK_RELTOL,
                dt=1e-10,
            )
            dopri_times.append(time.perf_counter() - t0)
        t_dopri_notebook = float(np.median(dopri_times))
        print(f"  dopri5 per trajectory at reltol=1e-7 (photon_integral/final): "
              f"{t_dopri_notebook:.3f} s (median of 3)")

        # --- expm + repeated matvec alternative ---
        dt_save = T_END / (N_SAVEAT - 1)
        t0 = time.perf_counter()
        P = scipy.linalg.expm(L * dt_save)
        t_expm = time.perf_counter() - t0
        t0 = time.perf_counter()
        x = packed_rho0.copy()
        pops_expm = np.empty((N_SAVEAT, n))
        pops_expm[0] = x[:n]
        for k in range(1, N_SAVEAT):
            x = P @ x
            pops_expm[k] = x[:n]
        t_expm_prop = time.perf_counter() - t0
        pops_expm_max_diff = float(np.max(np.abs(pops_expm - pops_ref)))
        print(f"  expm(L*dt) alternative: expm {t_expm:.2f} s (once), "
              f"{N_SAVEAT-1} matvecs {t_expm_prop*1e3:.1f} ms, "
              f"populations max abs diff vs ref = {pops_expm_max_diff:.3e}")

        rows_out.append(
            {
                "detuning_MHz": detuning_mhz,
                "dim": dim,
                "nnz_L": int(np.count_nonzero(L)),
                "t_jacobian_extract_s": t_jac,
                "t_eig_s": t_eig,
                "t_lu_factor_V_s": t_lu,
                "t_cond_V_s": t_cond,
                "eig_residual_rel": residual,
                "cond_V": cond_V,
                "t_solve_c_per_state_s": t_solve_c,
                "t_propagate_801pts_s": t_prop,
                "t_photon_analytic_s": t_photon,
                "max_imag_population": max_imag,
                "pops_max_abs_diff_vs_ref": pops_max_diff,
                "photon_analytic": photon_analytic,
                "photon_ref_reltol1e-9": photon_ref,
                "photon_rel_diff": photon_rel_diff,
                "t_dopri5_ref_pops_reltol1e-9_s": t_ref_pops,
                "t_dopri5_ref_photon_reltol1e-9_s": t_ref_photon,
                "t_dopri5_notebook_reltol1e-7_s": t_dopri_notebook,
                "t_expm_s": t_expm,
                "t_expm_matvec_prop_s": t_expm_prop,
                "pops_expm_max_abs_diff_vs_ref": pops_expm_max_diff,
            }
        )

    out_path = RESULTS_DIR / "exact_propagator.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {out_path}")

    # summary verdict numbers
    r = rows_out[0]
    setup = r["t_jacobian_extract_s"] + r["t_eig_s"] + r["t_lu_factor_V_s"]
    per_state = r["t_solve_c_per_state_s"] + r["t_propagate_801pts_s"]
    print("\n=== summary (detuning 0 MHz) ===")
    print(f"  exact propagator: setup (jac+eig+LU) {setup:.2f} s/grid point, "
          f"+ {per_state*1e3:.1f} ms per initial state (801 saveat points)")
    print(f"  stepping:         {r['t_dopri5_notebook_reltol1e-7_s']:.3f} s per trajectory")
    ratio = setup / r["t_dopri5_notebook_reltol1e-7_s"]
    print(f"  break-even: setup costs {ratio:.1f}x one dopri5 trajectory")


if __name__ == "__main__":
    main()
