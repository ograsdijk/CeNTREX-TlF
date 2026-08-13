"""Step-size diagnostic for oscillatory OBE systems (plan step 7 pre-work).

Builds the r2-in-static-E-field system exactly as
``examples/lindblad/r2_peak_ratio_vs_z_polarization.ipynb`` does
(R2_F1_7o2_F3, Ez=171.6 V/cm, retain_opposite_parity_levels=True,
qn_compact=True, 60 mW rectangular beam, T = 0.02 m / 184 m/s) and answers:

1. What frequency content remains in the RWA Hamiltonian? Residual diagonal
   energies (per (electronic, J) manifold and globally), coupling magnitudes,
   and per-coupling detunings |H_ii - H_jj|.
2. What step size does dopri5 actually take (accepted steps over the span),
   and how does it compare with the fastest phase in H?
3. Is the step size accuracy-limited or oscillation/stability-limited?
   Empirically: accepted steps vs reltol (accuracy-limited explicit RK5 shows
   steps ~ reltol^(-1/5); oscillation-limited is flat).
4. Upper-bound estimate of the achievable step-count reduction if the static
   diagonal were removed analytically (interaction picture / per-level
   co-rotating frame): ratio of the full spectral radius to the spectral
   radius with the diagonal removed (plus Gamma as the incoherent floor).

Wall-clock only. Writes CSVs and a markdown report to
``benchmarks/step_size_diagnostics_results/``.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sympy as smp

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.couplings.polarization import Polarization
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.utils.rabi import power_to_rabi_rectangular_beam

RESULTS_DIR = Path(__file__).parent / "step_size_diagnostics_results"

GAMMA = getattr(hamiltonian, "Γ")  # 2*pi*1.56e6 rad/s

TRANSITION = transitions.R2_F1_7o2_F3
E_FIELD = np.array([0.0, 0.0, 171.6])
B_FIELD = np.array([0.0, 0.0, 1e-5])
POWER_W = 60.0e-3
BEAM_WX = BEAM_WY = 0.02
VELOCITY = 184.0
T_END = 0.02 / VELOCITY  # ~108.7 us
INITIAL_F1 = 5 / 2
INITIAL_F = 2
INITIAL_MF = 0

RELTOLS = [1e-5, 1e-7, 1e-9]  # notebook uses 1e-7
ABSTOL = 1e-9
DETUNINGS_MHZ = [0.0, 25.0]  # normal peak and opposite-parity peak


def mhz(x: float) -> float:
    return float(x) / (2 * np.pi * 1e6)


def build_system():
    pol = Polarization(
        vector=np.array([1.0, 0.0, 0.0], dtype=np.complex128), name="X"
    )
    ground_main = 1 * next(
        iter(
            states.generate_coupled_states_X(
                states.QuantumSelector(
                    electronic=states.ElectronicState.X,
                    J=TRANSITION.J_ground,
                    F1=INITIAL_F1,
                    F=INITIAL_F,
                    mF=INITIAL_MF,
                    P=TRANSITION.P_ground,
                )
            )
        )
    )
    excited_main = 1 * next(
        iter(
            states.generate_coupled_states_B(
                states.QuantumSelector(
                    electronic=states.ElectronicState.B,
                    J=TRANSITION.J_excited,
                    F1=TRANSITION.F1_excited,
                    F=TRANSITION.F_excited,
                    mF=1,
                    P=TRANSITION.P_excited,
                )
            )
        )
    )
    transition_selectors = couplings.generate_transition_selectors(
        transitions=[TRANSITION],
        polarizations=[[pol]],
        ground_mains=[ground_main],
        excited_mains=[excited_main],
    )
    system = lindblad.generate_OBE_system_transitions(
        [TRANSITION],
        transition_selectors,
        qn_compact=True,
        E=E_FIELD,
        B=B_FIELD,
        retain_opposite_parity_levels=True,
        normalize_pol=True,
    )
    return system, transition_selectors[0]


def make_parameters(system, ts, rabi_value: float, detuning_rad: float):
    params = LindbladParameters()
    rabi = params.real("rabi", rabi_value)
    detuning = params.real("detuning", detuning_rad)
    for symbol in system.H_symbolic.free_symbols:
        name = str(symbol)
        if symbol in system.coupling_symbols:
            params.bind(symbol, rabi, finalize=False)
        elif name == str(getattr(ts, "δ")):
            params.bind(symbol, detuning, finalize=False)
        else:
            params.real(name, 0.0)
    for group in system.polarization_symbols:
        symbols = group if isinstance(group, (list, tuple)) else [group]
        for symbol in symbols:
            params.bind(symbol, 1.0, finalize=False)
    params._finalize()
    return params


def numeric_hamiltonian(system, ts, rabi_value: float, detuning_rad: float):
    subs: dict = {}
    for symbol in system.H_symbolic.free_symbols:
        name = str(symbol)
        if symbol in system.coupling_symbols:
            subs[symbol] = rabi_value
        elif name == str(getattr(ts, "δ")):
            subs[symbol] = detuning_rad
        else:
            subs[symbol] = 0.0
    for group in system.polarization_symbols:
        symbols = group if isinstance(group, (list, tuple)) else [group]
        for symbol in symbols:
            subs[symbol] = 1.0
    H = system.H_symbolic.subs(subs)
    n = H.shape[0]
    return np.array(
        [[complex(H[i, j]) for j in range(n)] for i in range(n)], dtype=complex
    )


def analyze_hamiltonian(system, H, rows):
    n = H.shape[0]
    diag = np.real(np.diag(H))

    groups = defaultdict(list)
    for idx, state in enumerate(system.QN):
        largest = state.largest
        groups[(largest.electronic_state.name, int(largest.J))].append(idx)

    print("\nPer-manifold residual diagonal energies (MHz):")
    for key, indices in sorted(groups.items()):
        values = diag[indices]
        spread = values.max() - values.min()
        print(
            f"  {key[0]} J={key[1]}: {len(indices)} states, "
            f"min={mhz(values.min()):+9.3f}  max={mhz(values.max()):+9.3f}  "
            f"spread={mhz(spread):8.3f}"
        )
        rows.append(
            {
                "metric": f"manifold_spread_{key[0]}_J{key[1]}_MHz",
                "value": mhz(spread),
            }
        )

    global_spread = diag.max() - diag.min()
    offdiag = H - np.diag(np.diag(H))
    coupling_mags = np.abs(offdiag[np.triu_indices(n, 1)])
    coupling_mags = coupling_mags[coupling_mags > 0]

    # per-coupling detunings
    detunings = []
    iu, ju = np.nonzero(np.triu(np.abs(offdiag) > 0, 1))
    for i, j in zip(iu, ju):
        detunings.append(abs(diag[i] - diag[j]))
    detunings = np.array(detunings)

    eigs = np.linalg.eigvalsh(H)
    rho_full = np.abs(eigs).max()
    H_nodiag = H - np.diag(np.diag(H))
    rho_nodiag = np.abs(np.linalg.eigvalsh(H_nodiag)).max()
    # incoherent floor: decay + the coupling/detuning scales that must remain
    rho_after = max(rho_nodiag, GAMMA)

    print(f"\nGlobal diagonal spread: {mhz(global_spread):.3f} MHz")
    print(
        f"Couplings: {coupling_mags.size} nonzero, |Omega_ij|/2pi: "
        f"max={mhz(coupling_mags.max()):.4f} MHz, "
        f"median={mhz(np.median(coupling_mags)):.4f} MHz"
    )
    print(
        f"Coupling detunings |H_ii-H_jj|/2pi: max={mhz(detunings.max()):.3f} MHz, "
        f"median={mhz(np.median(detunings)):.3f} MHz"
    )
    print(f"Spectral radius rho(H)/2pi: {mhz(rho_full):.3f} MHz")
    print(f"Spectral radius without diagonal rho(H-diag)/2pi: {mhz(rho_nodiag):.4f} MHz")
    print(f"Gamma/2pi: {mhz(GAMMA):.3f} MHz")
    print(
        f"Upper-bound step-count reduction from removing the static diagonal: "
        f"~{rho_full / rho_after:.1f}x"
    )
    for name, value in [
        ("global_diagonal_spread_MHz", mhz(global_spread)),
        ("max_coupling_MHz", mhz(coupling_mags.max())),
        ("max_coupling_detuning_MHz", mhz(detunings.max())),
        ("spectral_radius_MHz", mhz(rho_full)),
        ("spectral_radius_nodiag_MHz", mhz(rho_nodiag)),
        ("gamma_MHz", mhz(GAMMA)),
        ("upper_bound_step_reduction", rho_full / rho_after),
    ]:
        rows.append({"metric": name, "value": value})
    return rho_full


def build_rho0(system):
    """Weighted initial density matrix, as in the notebook.

    Population is spread over F=2 mF=0 (weight 2/3) and F=3 mF=-1,0,+1
    (weight 1/9 each), matching the R2 initial-state preparation used
    throughout ``examples/lindblad/r2_peak_ratio_vs_z_polarization.ipynb``.

    Factored out of ``run_solves`` so tests can build the same rho0 without
    duplicating the selector/weight bookkeeping.
    """
    rho0 = np.zeros((len(system.QN), len(system.QN)), dtype=np.complex128)
    selectors = [
        states.QuantumSelector(J=2, F1=INITIAL_F1, F=INITIAL_F, mF=INITIAL_MF),
        states.QuantumSelector(J=2, F1=INITIAL_F1, F=INITIAL_F + 1, mF=-1),
        states.QuantumSelector(J=2, F1=INITIAL_F1, F=INITIAL_F + 1, mF=0),
        states.QuantumSelector(J=2, F1=INITIAL_F1, F=INITIAL_F + 1, mF=+1),
    ]
    weights = np.array([2 / 3, 1 / 9, 1 / 9, 1 / 9])
    for selector, weight in zip(selectors, weights):
        indices = np.asarray(selector.get_indices(system.QN), dtype=int).ravel()
        for idx in indices:
            rho0[idx, idx] += weight / indices.size
    rho0 /= np.trace(rho0)
    return rho0


def run_solves(system, ts, rabi_value, solve_rows):
    rho0 = build_rho0(system)

    for detuning_mhz in DETUNINGS_MHZ:
        detuning_rad = 2 * np.pi * detuning_mhz * 1e6
        params = make_parameters(system, ts, rabi_value, detuning_rad)
        prepared = prepare_lindblad_problem(
            system, params, backend="rust", hamiltonian_representation="decomposed"
        )
        for reltol in RELTOLS:
            start = time.perf_counter()
            result = lindblad.solve_lindblad(
                prepared,
                rho0,
                (0.0, T_END),
                solver="dopri5",
                execution_mode="expanded_sparse",
                output="populations",
                output_when="final",
                dense_output=False,
                abstol=ABSTOL,
                reltol=reltol,
                dt=1e-10,
                collect_stats=True,
            )
            elapsed = time.perf_counter() - start
            stats = result.solver_stats or {}
            accepted = int(stats.get("accepted_steps", 0))
            rejected = int(stats.get("rejected_steps", 0))
            rhs_calls = int(stats.get("rhs_calls", 0))
            mean_dt = T_END / accepted if accepted else float("nan")
            print(
                f"  detuning={detuning_mhz:5.1f} MHz reltol={reltol:.0e}: "
                f"accepted={accepted:7d} rejected={rejected:6d} rhs={rhs_calls:8d} "
                f"mean_dt={mean_dt*1e9:7.2f} ns wall={elapsed:6.2f} s"
            )
            solve_rows.append(
                {
                    "detuning_MHz": detuning_mhz,
                    "reltol": reltol,
                    "accepted_steps": accepted,
                    "rejected_steps": rejected,
                    "rhs_calls": rhs_calls,
                    "mean_dt_ns": mean_dt * 1e9,
                    "wall_seconds": elapsed,
                }
            )
    return solve_rows


def main():
    print("Building r2-in-E-field system (notebook-identical)...")
    t0 = time.perf_counter()
    system, ts = build_system()
    print(f"  built in {time.perf_counter() - t0:.2f} s, n_states={len(system.QN)}")

    rabi_value = power_to_rabi_rectangular_beam(
        POWER_W, abs(system.couplings[0].main_coupling), BEAM_WX, BEAM_WY
    )
    print(f"  Rabi rate: {mhz(rabi_value):.4f} MHz (2pi), Gamma: {mhz(GAMMA):.3f} MHz")
    print(f"  span T_END = {T_END*1e6:.2f} us")

    rows: list[dict] = [
        {"metric": "n_states", "value": len(system.QN)},
        {"metric": "rabi_MHz", "value": mhz(rabi_value)},
        {"metric": "T_END_us", "value": T_END * 1e6},
    ]
    H = numeric_hamiltonian(system, ts, rabi_value, 0.0)
    rho_full = analyze_hamiltonian(system, H, rows)

    print("\nSolves (dopri5 / expanded_sparse, abstol=1e-9):")
    solve_rows: list[dict] = []
    run_solves(system, ts, rabi_value, solve_rows)

    # derived: steps per fastest period at the notebook tolerance
    notebook = [r for r in solve_rows if r["reltol"] == 1e-7 and r["detuning_MHz"] == 0.0]
    if notebook:
        mean_dt = notebook[0]["mean_dt_ns"] * 1e-9
        f_max = rho_full / (2 * np.pi)
        steps_per_period = 1.0 / (mean_dt * f_max) if f_max > 0 else float("nan")
        print(
            f"\nAt reltol=1e-7: mean_dt={mean_dt*1e9:.2f} ns; fastest phase "
            f"{f_max/1e6:.2f} MHz -> {steps_per_period:.1f} steps per fastest period"
        )
        rows.append({"metric": "steps_per_fastest_period_reltol1e-7", "value": steps_per_period})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "hamiltonian_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS_DIR / "solve_step_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "detuning_MHz", "reltol", "accepted_steps", "rejected_steps",
                "rhs_calls", "mean_dt_ns", "wall_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(solve_rows)
    print(f"\nWrote CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
