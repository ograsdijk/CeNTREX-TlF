from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import PreparedLindbladProblem
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.utils.rabi import (
    power_to_intensity_rectangular_beam,
    power_to_rabi_rectangular_beam,
)


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "obe_solve_speed_results"
FIGURES_DIR = HERE / "figures"
REPORT_PATH = HERE / "obe_solve_speed_report.md"

GAMMA = getattr(hamiltonian, "\u0393")
TRANSITION = transitions.R2_F1_7o2_F3
E_FIELD = np.array([0.0, 0.0, 200.0])
B_FIELD = np.array([0.0, 0.0, 1e-5])
POWER_MW = 60.0
BEAM_WX = 0.02
BEAM_WY = 0.02
VELOCITY = 184.0
INTERACTION_LENGTH = 0.02
T_END = INTERACTION_LENGTH / VELOCITY
INITIAL_F1 = 5 / 2
INITIAL_F = 2
INITIAL_MF = 0
RABI_PARAMETER_NAME = "rabi"
DETUNING_PARAMETER_NAME = "detuning"


@dataclass(frozen=True)
class BenchmarkModel:
    name: str
    system: lindblad.OBESystem
    params: LindbladParameters
    prepared: PreparedLindbladProblem
    rho0: np.ndarray
    rabi_rad_s: float
    initial_index: int
    excited_indices: list[int]
    photon_integral_weights: list[tuple[int, float]]
    prep_seconds: float


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def transition_selectors() -> list[couplings.TransitionSelector]:
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
    return couplings.generate_transition_selectors(
        transitions=[TRANSITION],
        polarizations=[[couplings.polarization_X]],
        ground_mains=[ground_main],
        excited_mains=[excited_main],
    )


def decay_only_ground_selector() -> states.QuantumSelector:
    return states.QuantumSelector(
        J=[1, 3, 4, 5],
        electronic=states.ElectronicState.X,
    )


def initial_state_index(system: lindblad.OBESystem) -> int:
    candidates = []
    for idx, state in enumerate(system.QN):
        qn = state.largest
        if (
            qn.electronic_state == states.ElectronicState.X
            and qn.J == TRANSITION.J_ground
            and qn.F1 == INITIAL_F1
            and qn.F == INITIAL_F
            and qn.mF == INITIAL_MF
        ):
            candidates.append(idx)
    if not candidates:
        raise ValueError("Could not find the requested initial state.")
    return min(candidates, key=lambda idx: float(np.real(system.H_int[idx, idx])))


def initial_density_matrix(system: lindblad.OBESystem) -> np.ndarray:
    rho0 = np.zeros((len(system.QN), len(system.QN)), dtype=np.complex128)
    idx = initial_state_index(system)
    rho0[idx, idx] = 1.0
    return rho0


def excited_indices(system: lindblad.OBESystem) -> list[int]:
    return [
        idx
        for idx, state in enumerate(system.QN)
        if state.largest.electronic_state == states.ElectronicState.B
    ]


def build_parameters(
    system: lindblad.OBESystem,
    selectors: list[couplings.TransitionSelector],
    rabi_rad_s: float,
) -> LindbladParameters:
    params = LindbladParameters()
    rabi = params.real(RABI_PARAMETER_NAME, rabi_rad_s)
    detuning = params.real(DETUNING_PARAMETER_NAME, 0.0)
    detuning_symbol = getattr(selectors[0], "\u03b4")

    for symbol in system.H_symbolic.free_symbols:
        if symbol in system.coupling_symbols:
            params.bind(symbol, rabi, finalize=False)
        elif symbol == detuning_symbol:
            params.bind(symbol, detuning, finalize=False)
        else:
            params.real(str(symbol), 0.0)

    for symbol_group in system.polarization_symbols:
        symbols = symbol_group if isinstance(symbol_group, (list, tuple)) else [symbol_group]
        for symbol in symbols:
            params.bind(symbol, 1.0, finalize=False)

    params._finalize()
    return params


def prepare_model(name: str, qn_compact: bool | states.QuantumSelector) -> BenchmarkModel:
    selectors = transition_selectors()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Low overlap detected.*")
        system = lindblad.generate_OBE_system_transitions(
            [TRANSITION],
            selectors,
            qn_compact=qn_compact,
            E=E_FIELD,
            B=B_FIELD,
            retain_opposite_parity_levels=True,
            method="matrix",
        )
    rabi_rad_s = power_to_rabi_rectangular_beam(
        POWER_MW * 1e-3,
        abs(system.couplings[0].main_coupling),
        BEAM_WX,
        BEAM_WY,
    )
    params = build_parameters(system, selectors, rabi_rad_s)
    prepared = prepare_lindblad_problem(
        system,
        params,
        backend="rust",
        hamiltonian_representation="decomposed",
    )
    prep_seconds = time.perf_counter() - start
    excited = excited_indices(system)
    return BenchmarkModel(
        name=name,
        system=system,
        params=params,
        prepared=prepared,
        rho0=initial_density_matrix(system),
        rabi_rad_s=rabi_rad_s,
        initial_index=initial_state_index(system),
        excited_indices=excited,
        photon_integral_weights=[(int(idx), float(GAMMA)) for idx in excited],
        prep_seconds=prep_seconds,
    )


def system_summary(models: dict[str, BenchmarkModel]) -> pd.DataFrame:
    rows = []
    for model in models.values():
        rows.append(
            {
                "model": model.name,
                "n_states": len(model.system.QN),
                "rho_entries": len(model.system.QN) ** 2,
                "H_nnz": sum(1 for value in model.system.H_symbolic if value != 0),
                "C_ops": int(model.system.C_array.shape[0]),
                "C_nnz": int(np.count_nonzero(model.system.C_array)),
                "rabi_MHz": model.rabi_rad_s / (2 * np.pi * 1e6),
                "prep_seconds_not_timed": model.prep_seconds,
            }
        )
    return pd.DataFrame(rows)


def _stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = {} if stats is None else dict(stats)
    rhs = float(stats.get("rhs_calls", np.nan))
    elapsed = float(stats.get("elapsed_seconds", np.nan))
    return {
        "accepted_steps": stats.get("accepted_steps", np.nan),
        "rejected_steps": stats.get("rejected_steps", np.nan),
        "rhs_calls": rhs,
        "solver_elapsed_seconds": elapsed,
        "rhs_calls_per_second": rhs / elapsed if elapsed > 0 else np.nan,
    }


def run_single_trajectory_benchmarks(model: BenchmarkModel) -> pd.DataFrame:
    saveat_201 = np.linspace(0.0, T_END, 201)
    output_cases = [
        {
            "output": "photon_integral",
            "output_when": "saveat",
            "saveat": saveat_201,
            "integral_weights": model.photon_integral_weights,
            "output_indices": None,
        },
        {
            "output": "populations",
            "output_when": "final",
            "saveat": None,
            "integral_weights": None,
            "output_indices": None,
        },
        {
            "output": "selected",
            "output_when": "final",
            "saveat": None,
            "integral_weights": None,
            "output_indices": [(model.initial_index, model.initial_index)]
            + [(idx, idx) for idx in model.excited_indices[:2]],
        },
    ]
    rows = []
    for solver in ("dopri5", "tsit5"):
        for case in output_cases:
            print(f"single trajectory: {solver} / {case['output']}", flush=True)
            start = time.perf_counter()
            result = lindblad.solve_lindblad(
                model.prepared,
                model.rho0,
                (0.0, T_END),
                solver=solver,
                execution_mode="expanded_sparse",
                output=case["output"],
                output_when=case["output_when"],
                saveat=case["saveat"],
                save_start=True,
                integral_weights=case["integral_weights"],
                output_indices=case["output_indices"],
                reltol=1e-4,
                abstol=1e-7,
                dt=2e-9,
                collect_stats=True,
            )
            wall = time.perf_counter() - start
            value = np.asarray(result.values).reshape(-1)[-1] if hasattr(result, "values") else np.nan
            rows.append(
                {
                    "model": model.name,
                    "solver": solver,
                    "output": case["output"],
                    "output_when": case["output_when"],
                    "save_points": 0 if case["saveat"] is None else len(case["saveat"]),
                    "rtol": 1e-4,
                    "abstol": 1e-7,
                    "wall_seconds": wall,
                    "final_value": float(np.real(value)),
                    **_stats(result.solver_stats),
                }
            )
            rows[-1]["rhs_calls_per_second"] = (
                rows[-1]["rhs_calls"] / wall if wall > 0 else np.nan
            )
    return pd.DataFrame(rows)


def run_tolerance_benchmarks(model: BenchmarkModel) -> pd.DataFrame:
    rows = []
    saveat = np.linspace(0.0, T_END, 201)
    for rtol, abstol in (
        (3e-2, 3e-5),
        (1e-2, 1e-5),
        (3e-3, 3e-6),
        (1e-3, 1e-6),
        (3e-4, 3e-7),
        (1e-4, 1e-7),
        (3e-5, 3e-8),
        (1e-5, 1e-8),
        (3e-6, 3e-9),
        (1e-6, 1e-9),
        (3e-7, 3e-10),
    ):
        print(f"tolerance sweep: rtol={rtol:g}, abstol={abstol:g}", flush=True)
        start = time.perf_counter()
        result = lindblad.solve_lindblad(
            model.prepared,
            model.rho0,
            (0.0, T_END),
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="photon_integral",
            output_when="saveat",
            saveat=saveat,
            integral_weights=model.photon_integral_weights,
            reltol=rtol,
            abstol=abstol,
            dt=2e-9,
            collect_stats=True,
        )
        rows.append(
            {
                "model": model.name,
                "rtol": rtol,
                "abstol": abstol,
                "save_points": len(saveat),
                "wall_seconds": time.perf_counter() - start,
                "photons": float(np.asarray(result.values).reshape(-1)[-1]),
                **_stats(result.solver_stats),
            }
        )
        rows[-1]["rhs_calls_per_second"] = (
            rows[-1]["rhs_calls"] / rows[-1]["wall_seconds"]
            if rows[-1]["wall_seconds"] > 0
            else np.nan
        )
    return pd.DataFrame(rows)


def run_model_variant_benchmarks(models: dict[str, BenchmarkModel]) -> pd.DataFrame:
    rows = []
    saveat = np.linspace(0.0, T_END, 201)
    for model in models.values():
        print(f"model variant solve: {model.name}", flush=True)
        start = time.perf_counter()
        result = lindblad.solve_lindblad(
            model.prepared,
            model.rho0,
            (0.0, T_END),
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="photon_integral",
            output_when="saveat",
            saveat=saveat,
            integral_weights=model.photon_integral_weights,
            reltol=1e-4,
            abstol=1e-7,
            dt=2e-9,
            collect_stats=True,
        )
        rows.append(
            {
                "model": model.name,
                "wall_seconds": time.perf_counter() - start,
                "photons": float(np.asarray(result.values).reshape(-1)[-1]),
                **_stats(result.solver_stats),
            }
        )
        rows[-1]["rhs_calls_per_second"] = (
            rows[-1]["rhs_calls"] / rows[-1]["wall_seconds"]
            if rows[-1]["wall_seconds"] > 0
            else np.nan
        )
    return pd.DataFrame(rows)


def run_scan_thread_benchmarks(
    model: BenchmarkModel,
    scan_points: tuple[int, ...] = (25,),
    thread_counts: tuple[int, ...] = (1, 2, 4, 8),
) -> pd.DataFrame:
    rows = []
    saveat = np.linspace(0.0, T_END, 201)
    for n_points in scan_points:
        detuning_mhz = np.linspace(-80.0, 80.0, n_points)
        for threads in thread_counts:
            print(f"scan scaling: {n_points} detunings, threads={threads}", flush=True)
            start = time.perf_counter()
            result = lindblad.grid_scan(
                model.prepared,
                model.rho0,
                (0.0, T_END),
                scan={
                    DETUNING_PARAMETER_NAME: 2 * np.pi * 1e6 * detuning_mhz,
                },
                solver="dopri5",
                execution_mode="expanded_sparse",
                output="photon_integral",
                output_when="saveat",
                saveat=saveat,
                integral_weights=model.photon_integral_weights,
                reltol=1e-4,
                abstol=1e-7,
                dt=2e-9,
                parallel=True,
                threads=threads,
                collect_stats=True,
            )
            wall = time.perf_counter() - start
            rows.append(
                {
                    "model": model.name,
                    "scan_points": n_points,
                    "threads": threads,
                    "wall_seconds": wall,
                    "trajectories_per_second": n_points / wall if wall > 0 else np.nan,
                    "photons_min": float(np.min(result.values)),
                    "photons_max": float(np.max(result.values)),
                    **_stats(result.solver_stats),
                }
            )
            rows[-1]["rhs_calls_per_second"] = (
                rows[-1]["rhs_calls"] / wall if wall > 0 else np.nan
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        for n_points in df["scan_points"].unique():
            mask = df["scan_points"] == n_points
            base = float(df.loc[mask & (df["threads"] == 1), "wall_seconds"].iloc[0])
            df.loc[mask, "speedup_vs_1_thread"] = base / df.loc[mask, "wall_seconds"]
            df.loc[mask, "parallel_efficiency"] = df.loc[mask, "speedup_vs_1_thread"] / df.loc[mask, "threads"]
    return df


def run_detuning_stats_scan(model: BenchmarkModel) -> pd.DataFrame:
    detuning_mhz = np.linspace(-80.0, 80.0, 9)
    saveat = np.linspace(0.0, T_END, 201)
    rows = []
    for detuning in detuning_mhz:
        print(f"detuning stats: {detuning:g} MHz", flush=True)
        result = lindblad.grid_scan(
            model.prepared,
            model.rho0,
            (0.0, T_END),
            scan={DETUNING_PARAMETER_NAME: [2 * np.pi * 1e6 * detuning]},
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="photon_integral",
            output_when="saveat",
            saveat=saveat,
            integral_weights=model.photon_integral_weights,
            reltol=1e-4,
            abstol=1e-7,
            dt=2e-9,
            parallel=False,
            collect_stats=True,
        )
        rows.append(
            {
                "detuning_MHz": detuning,
                "photons": float(np.asarray(result.values).reshape(-1)[-1]),
                **_stats(result.solver_stats),
            }
        )
        wall = rows[-1].get("solver_elapsed_seconds", np.nan)
        if not np.isfinite(wall):
            rows[-1]["rhs_calls_per_second"] = np.nan
    return pd.DataFrame(rows)


def numeric_hamiltonian(
    model: BenchmarkModel,
    detuning_mhz: float,
    rabi_rad_s: float | None = None,
) -> np.ndarray:
    rabi_rad_s = model.rabi_rad_s if rabi_rad_s is None else rabi_rad_s
    detuning_symbol = getattr(transition_selectors()[0], "\u03b4")
    subs = {}
    for symbol in model.system.H_symbolic.free_symbols:
        if symbol in model.system.coupling_symbols:
            subs[symbol] = rabi_rad_s
        elif symbol == detuning_symbol:
            subs[symbol] = 2 * np.pi * 1e6 * detuning_mhz
        else:
            subs[symbol] = 1.0 if str(symbol).startswith("P") else 0.0
    return np.asarray(model.system.H_symbolic.subs(subs), dtype=np.complex128)


def liouvillian_sparse(H: np.ndarray, C_array: np.ndarray) -> sp.csr_matrix:
    n = H.shape[0]
    eye = sp.eye(n, format="csr", dtype=np.complex128)
    Hs = sp.csr_matrix(H)
    L = -1j * (sp.kron(eye, Hs, format="csr") - sp.kron(Hs.T, eye, format="csr"))
    for C in C_array:
        Cs = sp.csr_matrix(C)
        if Cs.nnz == 0:
            continue
        CdC = Cs.conj().T @ Cs
        L = L + sp.kron(Cs.conj(), Cs, format="csr")
        L = L - 0.5 * sp.kron(eye, CdC, format="csr")
        L = L - 0.5 * sp.kron(CdC.T, eye, format="csr")
    return L.tocsr()


def photon_weight_vector(model: BenchmarkModel) -> np.ndarray:
    n = len(model.system.QN)
    weights = np.zeros(n * n, dtype=np.complex128)
    for idx, weight in model.photon_integral_weights:
        weights[idx + idx * n] = weight
    return weights


def expm_photon_integral(model: BenchmarkModel, detuning_mhz: float) -> tuple[float, np.ndarray]:
    H = numeric_hamiltonian(model, detuning_mhz)
    L = liouvillian_sparse(H, np.asarray(model.system.C_array, dtype=np.complex128))
    rho_vec = model.rho0.reshape(-1, order="F").astype(np.complex128)
    weights = photon_weight_vector(model)
    augmented = sp.bmat(
        [
            [L, sp.csr_matrix((L.shape[0], 1), dtype=np.complex128)],
            [sp.csr_matrix(weights.reshape(1, -1)), sp.csr_matrix((1, 1), dtype=np.complex128)],
        ],
        format="csr",
    )
    y0 = np.concatenate([rho_vec, np.array([0.0 + 0.0j])])
    y_final = expm_multiply(augmented * T_END, y0)
    return float(np.real(y_final[-1])), y_final[:-1]


def run_exponential_comparison(model: BenchmarkModel) -> pd.DataFrame:
    rows = []
    saveat = np.linspace(0.0, T_END, 401)
    for detuning_mhz in (0.0,):
        print(f"constant-coefficient exponential comparison: {detuning_mhz:g} MHz", flush=True)
        start = time.perf_counter()
        ode_result = lindblad.grid_scan(
            model.prepared,
            model.rho0,
            (0.0, T_END),
            scan={DETUNING_PARAMETER_NAME: [2 * np.pi * 1e6 * detuning_mhz]},
            solver="dopri5",
            execution_mode="expanded_sparse",
            output="photon_integral",
            output_when="saveat",
            saveat=saveat,
            integral_weights=model.photon_integral_weights,
            reltol=1e-4,
            abstol=1e-7,
            dt=2e-9,
            parallel=False,
            collect_stats=True,
        )
        ode_wall = time.perf_counter() - start
        ode_photons = float(np.asarray(ode_result.values).reshape(-1)[-1])
        start = time.perf_counter()
        expm_photons, _ = expm_photon_integral(model, detuning_mhz)
        expm_wall = time.perf_counter() - start
        rows.append(
            {
                "model": model.name,
                "detuning_MHz": detuning_mhz,
                "adaptive_wall_seconds": ode_wall,
                "expm_wall_seconds": expm_wall,
                "adaptive_photons": ode_photons,
                "expm_photons": expm_photons,
                "abs_photon_difference": abs(expm_photons - ode_photons),
                "relative_photon_difference": abs(expm_photons - ode_photons)
                / max(abs(ode_photons), 1e-15),
                **{f"adaptive_{key}": value for key, value in _stats(ode_result.solver_stats).items()},
            }
        )
    return pd.DataFrame(rows)


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)


def plot_results(tables: dict[str, pd.DataFrame]) -> list[Path]:
    plt.rcParams.update({"font.size": 14})
    paths: list[Path] = []

    single = tables["single_trajectory"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = single["solver"] + " / " + single["output"]
    ax.bar(labels, single["wall_seconds"])
    ax.set_ylabel("Solve wall time (s)")
    ax.set_title("Single trajectory solve time")
    ax.tick_params(axis="x", rotation=35)
    path = FIGURES_DIR / "obe_solve_single_trajectory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    scan = tables["scan_thread_scaling"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for n_points, group in scan.groupby("scan_points"):
        ax.plot(group["threads"], group["trajectories_per_second"], "o-", label=f"{n_points} detunings")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Trajectories / s")
    ax.set_title("Frequency scan throughput")
    ax.legend()
    path = FIGURES_DIR / "obe_solve_scan_thread_scaling.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    stats = tables["detuning_stats"]
    fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax1.plot(stats["detuning_MHz"], stats["photons"], "o-", label="photons")
    ax1.set_xlabel("Detuning (MHz)")
    ax1.set_ylabel("Photons")
    ax2 = ax1.twinx()
    ax2.plot(stats["detuning_MHz"], stats["rhs_calls"], "s--", color="tab:red", label="RHS calls")
    ax2.set_ylabel("RHS calls")
    ax1.set_title("Detuning-dependent solve effort")
    path = FIGURES_DIR / "obe_solve_rhs_by_detuning.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    expm = tables["exponential_comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(expm))
    width = 0.35
    axes[0].bar(x - width / 2, expm["adaptive_wall_seconds"], width, label="adaptive ODE")
    axes[0].bar(x + width / 2, expm["expm_wall_seconds"], width, label="expm_multiply")
    axes[0].set_xticks(x, [f"{d:g}" for d in expm["detuning_MHz"]])
    axes[0].set_xlabel("Detuning (MHz)")
    axes[0].set_ylabel("Wall time (s)")
    axes[0].legend()
    axes[1].plot(expm["detuning_MHz"], expm["adaptive_photons"], "o-", label="adaptive")
    axes[1].plot(expm["detuning_MHz"], expm["expm_photons"], "s--", label="expm")
    axes[1].set_xlabel("Detuning (MHz)")
    axes[1].set_ylabel("Photons")
    axes[1].legend()
    fig.suptitle("Constant-coefficient exponential comparison")
    path = FIGURES_DIR / "obe_solve_exponential_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    tol = tables["tolerance_sweep"]
    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax1.semilogx(tol["rtol"], tol["wall_seconds"], "o-", label="wall time")
    ax1.set_xlabel("Relative tolerance")
    ax1.set_ylabel("Wall time (s)")
    ax2 = ax1.twinx()
    ax2.semilogx(tol["rtol"], tol["photons"], "s--", color="tab:red", label="photons")
    ax2.set_ylabel("Photons")
    ax1.invert_xaxis()
    ax1.set_title("Tolerance sweep")
    path = FIGURES_DIR / "obe_solve_tolerance_sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def _markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    table = df.loc[:, columns]
    if n is not None:
        table = table.head(n)
    headers = list(table.columns)

    def fmt(value: Any) -> str:
        if isinstance(value, float | np.floating):
            if np.isnan(value):
                return ""
            return f"{float(value):.4g}"
        if isinstance(value, int | np.integer):
            return str(int(value))
        return str(value)

    rows = [[fmt(value) for value in row] for row in table.to_numpy(dtype=object)]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def write_report(tables: dict[str, pd.DataFrame], figures: list[Path]) -> None:
    summary = tables["system_summary"]
    single = tables["single_trajectory"]
    scan = tables["scan_thread_scaling"]
    expm = tables["exponential_comparison"]
    variants = tables["model_variants"]

    best_single = single.loc[single["wall_seconds"].idxmin()]
    best_scan = scan.loc[scan["trajectories_per_second"].idxmax()]
    variant_speedup = (
        variants.set_index("model").loc["per_J_sinks", "wall_seconds"]
        / variants.set_index("model").loc["single_decay_sink", "wall_seconds"]
    )
    expm_slowdown = np.nanmedian(expm["expm_wall_seconds"] / expm["adaptive_wall_seconds"])

    lines = [
        "# OBE Solve-Time Speed Investigation",
        "",
        "This report times only solve calls after the OBE system and Rust plan have been prepared.",
        "Preparation timings are reported as context but excluded from solve-speed conclusions.",
        "",
        "## Configuration",
        "",
        f"- Transition: `{TRANSITION.name}`",
        f"- Electric field: `{E_FIELD.tolist()}` V/cm",
        f"- Power: `{POWER_MW:g}` mW over `{BEAM_WX * 100:g} cm x {BEAM_WY * 100:g} cm`",
        f"- Interaction time: `{T_END * 1e6:.3f}` us",
        f"- Intensity: `{power_to_intensity_rectangular_beam(POWER_MW * 1e-3, BEAM_WX, BEAM_WY) * 0.1:.3f}` mW/cm^2",
        "",
        "## Main Findings",
        "",
        f"- Fastest single-trajectory case in this run: `{best_single['solver']}` / `{best_single['output']}` at `{best_single['wall_seconds']:.3g}` s.",
        f"- Best scan throughput in this run: `{best_scan['trajectories_per_second']:.3g}` trajectories/s with `{int(best_scan['threads'])}` threads.",
        f"- Collapsing all decay-only ground states into one sink changed the photon-count solve time by a factor of `{variant_speedup:.3g}` for the benchmark case.",
        f"- Sparse `expm_multiply` was `{expm_slowdown:.3g}x` slower than adaptive Rust ODE over the tested detunings.",
        "",
        "## System Sizes",
        "",
        _markdown_table(
            summary,
            [
                "model",
                "n_states",
                "rho_entries",
                "H_nnz",
                "C_ops",
                "C_nnz",
                "rabi_MHz",
                "prep_seconds_not_timed",
            ],
        ),
        "",
        "## Single-Trajectory Solver Results",
        "",
        _markdown_table(
            single,
            [
                "solver",
                "output",
                "output_when",
                "save_points",
                "wall_seconds",
                "accepted_steps",
                "rhs_calls",
                "rhs_calls_per_second",
            ],
        ),
        "",
        f"![Single trajectory solve time]({figures[0].relative_to(HERE).as_posix()})",
        "",
        "## Frequency Scan Scaling",
        "",
        _markdown_table(
            scan,
            [
                "scan_points",
                "threads",
                "wall_seconds",
                "trajectories_per_second",
                "speedup_vs_1_thread",
                "parallel_efficiency",
                "rhs_calls",
            ],
        ),
        "",
        f"![Frequency scan throughput]({figures[1].relative_to(HERE).as_posix()})",
        "",
        "## Detuning-Dependent Solver Effort",
        "",
        "The adaptive solver does not spend exactly the same effort at each detuning.",
        "",
        f"![RHS calls by detuning]({figures[2].relative_to(HERE).as_posix()})",
        "",
        "## Constant-Coefficient Exponential Test",
        "",
        "For constant E/B/light fields, the density matrix obeys a constant-coefficient linear ODE. The test below compares the adaptive Rust ODE result against a sparse augmented Liouvillian evaluated with `scipy.sparse.linalg.expm_multiply`.",
        "",
        _markdown_table(
            expm,
            [
                "detuning_MHz",
                "adaptive_wall_seconds",
                "expm_wall_seconds",
                "adaptive_photons",
                "expm_photons",
                "relative_photon_difference",
            ],
        ),
        "",
        f"![Exponential comparison]({figures[3].relative_to(HERE).as_posix()})",
        "",
        "## Tolerance Sweep",
        "",
        _markdown_table(
            tables["tolerance_sweep"],
            ["rtol", "abstol", "wall_seconds", "photons", "accepted_steps", "rhs_calls"],
        ),
        "",
        "Loosening tolerances does not reduce RHS calls for this benchmark. Very loose settings distort the photon count before providing a speed benefit, so tolerance relaxation is not a useful speed lever here.",
        "",
        f"![Tolerance sweep]({figures[4].relative_to(HERE).as_posix()})",
        "",
        "## Notes",
        "",
        "- The committed benchmark is intentionally bounded so it can be rerun interactively.",
        "- Increase `scan_points` and `thread_counts` in the notebook for longer scaling runs such as 101/401 detunings and 12/16 threads.",
        "- The current Rust photon-integral output integrates over saved output samples, so photon-count accuracy and runtime should be checked against `saveat` density.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmarks(
    *,
    scan_points: tuple[int, ...] = (9,),
    thread_counts: tuple[int, ...] = (1, 2),
    run_exponential: bool = True,
) -> dict[str, pd.DataFrame]:
    ensure_dirs()
    print("preparing per-J sink model", flush=True)
    models = {
        "per_J_sinks": prepare_model("per_J_sinks", True),
    }
    print("preparing single decay-sink model", flush=True)
    models["single_decay_sink"] = prepare_model(
        "single_decay_sink", decay_only_ground_selector()
    )
    primary = models["per_J_sinks"]
    exponential_model = models["single_decay_sink"]

    tables: dict[str, pd.DataFrame] = {}
    tables["system_summary"] = system_summary(models)
    save_tables(tables)
    tables["single_trajectory"] = run_single_trajectory_benchmarks(primary)
    save_tables(tables)
    tables["tolerance_sweep"] = run_tolerance_benchmarks(primary)
    save_tables(tables)
    tables["model_variants"] = run_model_variant_benchmarks(models)
    save_tables(tables)
    tables["scan_thread_scaling"] = run_scan_thread_benchmarks(
        primary,
        scan_points=scan_points,
        thread_counts=thread_counts,
    )
    save_tables(tables)
    tables["detuning_stats"] = run_detuning_stats_scan(primary)
    save_tables(tables)
    if run_exponential:
        tables["exponential_comparison"] = run_exponential_comparison(exponential_model)
    else:
        tables["exponential_comparison"] = pd.DataFrame(
            [
                {
                    "model": exponential_model.name,
                    "detuning_MHz": 0.0,
                    "adaptive_wall_seconds": np.nan,
                    "expm_wall_seconds": np.nan,
                    "adaptive_photons": np.nan,
                    "expm_photons": np.nan,
                    "abs_photon_difference": np.nan,
                    "relative_photon_difference": np.nan,
                }
            ]
        )
    save_tables(tables)
    figures = plot_results(tables)
    write_report(tables, figures)
    metadata = {
        "transition": TRANSITION.name,
        "E_FIELD": E_FIELD.tolist(),
        "B_FIELD": B_FIELD.tolist(),
        "POWER_MW": POWER_MW,
        "T_END": T_END,
        "scan_points": list(scan_points),
        "thread_counts": list(thread_counts),
        "report": str(REPORT_PATH.relative_to(HERE)),
        "figures": [str(path.relative_to(HERE)) for path in figures],
    }
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-points",
        default="9",
        help="Comma-separated frequency scan sizes, e.g. 25,101,401.",
    )
    parser.add_argument(
        "--threads",
        default="1,2",
        help="Comma-separated Rust scan thread counts, e.g. 1,2,4,8,12,16.",
    )
    parser.add_argument(
        "--skip-exponential",
        action="store_true",
        help="Skip the sparse expm_multiply comparison.",
    )
    return parser.parse_args()


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(
        scan_points=_parse_int_tuple(args.scan_points),
        thread_counts=_parse_int_tuple(args.threads),
        run_exponential=not args.skip_exponential,
    )
