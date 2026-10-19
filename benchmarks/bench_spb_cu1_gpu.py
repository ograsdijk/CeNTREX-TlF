"""GPU-only CU1 comparison while unrelated CPU simulations continue.

Self-contained 26-level CU1 recipe; no phase EOM, alternating-pass Doppler.
No CPU ODEs or CPU timing benchmarks are run. References are GPU convergence
checks pending independent CPU verification when the machine is idle.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUPY_CACHE_DIR"] = str(ROOT / "benchmarks/.cupy_cache")

import numpy as np
import psutil

from centrex_tlf import couplings, lindblad, states, transitions
from centrex_tlf.lindblad import cuda

OUT = ROOT / "benchmarks/spb_cu1_gpu_results"
CACHE = ROOT / ".cache/benchmarks/spb_cu1_gpu"
T_END = 0.05 / 184.0
SLOTS = ["laser_detuning", "vx", "y0"]
CHECKPOINTS = np.array([0.0035, 0.0245, 0.0455, 0.05]) / 184.0


def atomic(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def report():
    data = json.loads((OUT / "performance.json").read_text())
    lines = [
        "# CU1 GPU comparison under concurrent CPU load",
        "",
        "26 levels, both excited parities, alternating-pass Doppler, no phase EOM. "
        "Uniform initial population across twelve J=1 states; 1024 distinct parameter combinations "
        "per full 271.739 us scan. This is GPU-only: the notebook CPU baseline and independent "
        "CPU density reference are deferred.",
        "",
        "CUDA event intervals cover integration and may include GPU idle gaps from delayed "
        "CPU graph submission. Public-call wall times include setup/capture and output transfer. "
        "The queue was shortened at the user's request. The 512-size runs are single scaling pilots; "
        "1024-size measurements have one to three randomized rounds, as shown in the table.",
        "",
        "| Scan | Solver / kernel | Step ns | Batch | Rounds | CUDA median s | Public-call median s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, c in enumerate(data["cases"]):
        for batch in [512, 1024]:
            samples = [r for r in data["timings"] if r["case"] == i and r["batch_size"] == batch]
            if not samples:
                continue
            label = f"{c['solver']} / {c['options']['rhs']}"
            if c["options"]["density_reduction"]:
                label += " + exact reduction"
            gpu = np.median([r["cuda_event_seconds"] for r in samples])
            wall = np.median([r["wall_seconds"] for r in samples])
            lines.append(
                f"| {c['pattern']} | {label} | {c['confirmation']['dt_ns']:g} | {batch} | {len(samples)} | {gpu:.3f} | {wall:.3f} |"
            )
    lines += [
        "",
        "Selected settings passed the 60-case full-transit density confirmation at "
        "first/middle/last pass centers and the endpoint. The reference is GPU Tsit5 at 0.5 ns, "
        "with a separate 1 ns versus 0.5 ns convergence check on twelve representative cases. "
        "This is not yet an independent CPU verification. Complex-entry accuracy is <=1% above "
        "magnitude 1e-6, absolute <=1e-6 below; J=1 residual, trace and positivity are checked too.",
        "",
        "The GPU control is the current library tiled Tsit5, not the historical notebook CPU "
        "method. No speedup versus the notebook CPU method is claimed from these results.",
        "",
        "Recipe and hashes: `provenance_performance.json`. Accuracy and rejected steps: "
        "`accuracy.json`. Kernel screen: `tune.json`. Full-transit confirmation: "
        "`confirmation.json`. Repeated measurements: `performance.json`.",
        "",
        "Reproduction: `bench_spb_cu1_gpu.py --stage fixture`, then `accuracy`, `refine`, "
        "`tune`, `confirm`, `performance`, `report`. Density reference arrays remain in "
        "the ignored `.cache/benchmarks/spb_cu1_gpu` directory. No production solver changes.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("Report written: " + str(OUT / "REPORT.md"))


def build():
    started = time.perf_counter()
    transition = transitions.R1_F1_5o2_F2
    X, B = states.ElectronicState.X, states.ElectronicState.B
    ground = states.CoupledBasisState(
        J=1, F=2, F1=1.5, mF=1, I1=0.5, I2=0.5, Ω=0, P=-1, electronic_state=X
    )
    excited = states.CoupledBasisState(
        J=2, F=2, F1=2.5, mF=2, I1=0.5, I2=0.5, Ω=1, P=1, electronic_state=B
    )
    selectors = couplings.generate_transition_selectors(
        [transition],
        [[couplings.polarization_X, couplings.polarization_Z]],
        ground_mains=[1 * ground],
        excited_mains=[1 * excited],
    )
    obe = lindblad.generate_OBE_system_transitions(
        [transition],
        selectors,
        qn_compact=True,
        E=np.array([0.0, 0.0, 150.0]),
        B=np.array([0.0, 0.0, 0.001]),
        retain_opposite_parity_levels=True,
    )
    params = lindblad.LindbladParameters()
    t = params.time()
    power = params.real("laser_power", 0.080)
    detuning = params.real("laser_detuning", 2 * np.pi * 1e6 * 19.0)
    vx, y0 = params.real("vx", 0.0), params.real("y0", 0.0)
    sigma_z, sigma_y = 0.00075, 0.0075
    order = __import__("centrex_tlf", fromlist=["utils"]).utils.multipass.multipass_prism_order(13)
    fractions = np.array([0.9 ** (n - 1) for n in order])
    intensities = params.real(
        "intensities_per_watt", tuple(fractions / (2 * np.pi * sigma_z * sigma_y))
    )
    centers = params.real("xlocs", tuple(0.0035 * np.arange(1, 14)))
    heights = params.real("ylocs", tuple(np.zeros(13)))
    main = float(abs(obe.couplings[0].main_coupling))
    spatial = lindblad.multipass_2d_rabi(
        184.0 * t, y0, intensities, centers, heights, sigma_z, sigma_y, main, 2.6675506e-30
    )
    params.bind(selectors[0].Ω, power**0.5 * spatial, finalize=False)
    # Freeze the external script's explicit carrier for an accurate recipe
    # comparison. This deliberately overrides the newer package default.
    shift = 2 * np.pi * 1.103e15 / 299792458.0
    sign = lindblad.alternating_sign(184.0 * t, 0.0035 / 2, 0.0035)
    params.bind(selectors[0].δ, detuning + vx * shift * sign, finalize=False)
    gate = lindblad.square_wave(t, 2 * np.pi * 0.5e6, 0.0)
    px, pz = selectors[0].polarization_symbols
    params.bind(px, 1.0 - gate, finalize=False)
    params.bind(pz, gate, finalize=False)
    params._finalize()
    p = lindblad.prepare_lindblad_problem(
        obe, params, backend="rust", hamiltonian_representation="decomposed"
    )
    idx = np.asarray(states.QuantumSelector(J=1, electronic=X).get_indices(obe.QN), int).ravel()
    rho = np.zeros((len(obe.QN),) * 2, complex)
    rho[idx, idx] = 1.0 / len(idx)
    assert len(obe.QN) == 26 and len(idx) == 12
    meta = dict(
        levels=len(obe.QN),
        initial_count=len(idx),
        main_coupling=main,
        initial="uniform over all twelve X J=1 states; no coherences",
        E_V_cm=[0, 0, 150],
        B_G=[0, 0, 0.001],
        power_W=0.08,
        sigma_z_m=sigma_z,
        sigma_y_m=sigma_y,
        centers_m=(0.0035 * np.arange(1, 14)).tolist(),
        relative_pass_powers=fractions.tolist(),
        dipole_C_m=2.6675506e-30,
        optical_reference_Hz=1.103e15,
        vz_m_s=184,
        t_end=T_END,
        retain_opposite_parity_levels=True,
        phase_EOM=False,
        polarization_Hz=500000,
        doppler="sign changes at spatial midpoints; compiler clips jumps",
        construction_seconds=time.perf_counter() - started,
    )
    return p, rho, idx, meta


def rows(detunings, velocities, heights):
    a = np.asarray(list(itertools.product(detunings, velocities, heights)), float)
    a[:, 0] *= 2 * np.pi * 1e6
    return a


def accuracy(p, values, reference, idx):
    got, ref = (
        p.layout.unpack_batch(values.reshape(-1, p.layout.packed_len)),
        p.layout.unpack_batch(reference.reshape(-1, p.layout.packed_len)),
    )
    difference, magnitude = abs(got - ref), abs(ref)
    large = magnitude >= 1e-6
    rel = float((difference[large] / magnitude[large]).max(initial=0))
    small = float(difference[~large].max(initial=0))
    trace = float(abs(np.trace(got, axis1=-2, axis2=-1) - 1).max())
    mineig = float(np.linalg.eigvalsh(got).min())
    population = np.diagonal(got, axis1=-2, axis2=-1).real[:, idx].sum(axis=-1)
    population_ref = np.diagonal(ref, axis1=-2, axis2=-1).real[:, idx].sum(axis=-1)
    remaining_scaled = float(
        (abs(population - population_ref) / np.maximum(0.01 * abs(population_ref), 1e-6)).max()
    )
    return dict(
        relative_above_1e6=rel,
        absolute_below_1e6=small,
        max_absolute=float(difference.max()),
        trace_error=trace,
        min_eigenvalue=mineig,
        remaining_population_scaled_error=remaining_scaled,
        passed=bool(
            rel <= 0.01
            and small <= 1e-6
            and trace <= 1e-6
            and mineig >= -1e-7
            and remaining_scaled <= 1
        ),
    )


def solve(
    p, rho, parameters, solver, dt_ns, option, *, full=False, span=(0.0, T_END), profile=False
):
    cp = cuda._cupy()
    checkpoints = np.union1d(
        CHECKPOINTS[(span[0] < CHECKPOINTS) & (span[1] >= CHECKPOINTS)], [span[1]]
    )
    events, submission = [], []
    original = cuda._Integrator

    class Timed(original):
        def advance(self, previous, target):
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            tick = time.perf_counter()
            super().advance(previous, target)
            submission.append(time.perf_counter() - tick)
            end.record()
            events.append((begin, end))

    if profile:
        cuda._Integrator = Timed
    before = psutil.cpu_percent()
    tick = time.perf_counter()
    try:
        answer = lindblad.parameter_scan(
            p,
            rho,
            span,
            parameter_slots=SLOTS,
            parameter_batch=parameters,
            solver=solver,
            evolution_device="cuda",
            dt=dt_ns * 1e-9,
            output="full" if full else "populations",
            saveat=checkpoints if full else None,
            output_when="saveat" if full else "final",
            maxiters=2_000_000,
            collect_stats=True,
            cuda_options=option,
        )
        cp.cuda.get_current_stream().synchronize()
    finally:
        cuda._Integrator = original
    wall = time.perf_counter() - tick
    record = dict(
        wall_seconds=wall,
        cuda_event_seconds=sum(cp.cuda.get_elapsed_time(a, b) / 1000 for a, b in events),
        host_submission_seconds=sum(submission),
        cpu_percent_during_call=psutil.cpu_percent(),
        cpu_percent_before=before,
        stats=answer.solver_stats,
    )
    return answer, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["fixture", "accuracy", "refine", "tune", "confirm", "performance", "report"],
        default="fixture",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Resume only the two reduced batch-512 pilots"
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        report()
        return
    # Only this process is deprioritized; other simulations are not modified.
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cp = cuda._cupy()
    p, rho, idx, meta = build()
    fixture_sha256 = hashlib.sha256(
        repr(
            (
                p.parameter_graph,
                p.hamiltonian_plan,
                p.source_decay_rates.tolist(),
                p.incoming_transfers_by_target,
            )
        ).encode()
    ).hexdigest()
    native = next((Path(__import__("centrex_tlf").__file__).parent).glob("*.pyd"))
    provenance = dict(
        recipe=meta,
        fixture_sha256=fixture_sha256,
        python=platform.python_version(),
        cupy=cp.__version__,
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        CPU_benchmarks_run=False,
        independent_CPU_reference="deferred until other CPU simulations finish",
        caveat="CUDA event spans include idle gaps if CPU graph submission is starved; concurrent CPU load is recorded.",
    )
    atomic(OUT / "provenance.json", provenance)
    atomic(OUT / f"provenance_{args.stage}.json", provenance)
    log(
        f"Built {meta['levels']} levels, uniform {meta['initial_count']}-state mixture, 271.7 us full transit"
    )
    if args.stage == "fixture":
        pars = rows([19.0], [-1, 0, 1], [0, 0.0085])
        a, r = solve(
            p,
            rho,
            pars,
            "fixed_tsit5",
            1.0,
            lindblad.CudaSolverOptions(batch_size=128),
            span=(18e-6, 20e-6),
            profile=True,
        )
        log(
            f"2 us smoke: trace error {abs(a.values.sum(axis=-1) - 1).max():.2g}, wall {r['wall_seconds']:.3f}s"
        )
        atomic(OUT / "smoke.json", r)
        return
    if args.stage == "accuracy":
        pars = rows([19.0, 20.2], [-1.0, 0.0, 1.0], [0.0, 0.0085])
        result = dict(
            parameters=pars.tolist(),
            checkpoint_seconds=CHECKPOINTS.tolist(),
            references={},
            candidates=[],
        )
        refs = {}
        for ns in [1.0, 0.5]:
            log(
                f"Reference full transit: Tsit5 {ns:g} ns, {len(pars)} cases, four density checkpoints"
            )
            a, r = solve(
                p,
                rho,
                pars,
                "fixed_tsit5",
                ns,
                lindblad.CudaSolverOptions(batch_size=128),
                full=True,
                profile=True,
            )
            refs[ns] = a.values
            result["references"][str(ns)] = r
            np.savez(CACHE / f"reference_{ns:g}ns.npz", parameters=pars, values=a.values)
            atomic(OUT / "accuracy.json", result)
        result["reference_convergence"] = accuracy(p, refs[1.0], refs[0.5], idx)
        log(f"Reference convergence: {result['reference_convergence']}")
        assert result["reference_convergence"]["passed"]
        for solver in ["fixed_tsit5", "fixed_ck54"]:
            for ns in [2.0, 4.0, 8.0]:
                log(f"Accuracy {solver} {ns:g} ns; full transit")
                row = dict(solver=solver, dt_ns=ns)
                try:
                    a, r = solve(
                        p,
                        rho,
                        pars,
                        solver,
                        ns,
                        lindblad.CudaSolverOptions(batch_size=128),
                        full=True,
                        profile=True,
                    )
                    row.update(timing=r, accuracy=accuracy(p, a.values, refs[0.5], idx))
                except (RuntimeError, FloatingPointError) as error:
                    row.update(error=str(error), accuracy={"passed": False})
                result["candidates"].append(row)
                atomic(OUT / "accuracy.json", result)
                log(f"{solver} {ns:g} ns: {row['accuracy']}")
        return
    verified = json.loads((OUT / "accuracy.json").read_text())
    if args.stage == "refine":
        saved = np.load(CACHE / "reference_0.5ns.npz")
        pars, reference = saved["parameters"], saved["values"]
        for ns in [1.0, 0.75]:
            log(f"Refine CK54 {ns:g} ns; full transit, four density checkpoints")
            a, r = solve(
                p,
                rho,
                pars,
                "fixed_ck54",
                ns,
                lindblad.CudaSolverOptions(batch_size=128),
                full=True,
                profile=True,
            )
            check = accuracy(p, a.values, reference, idx)
            verified["candidates"].append(
                dict(solver="fixed_ck54", dt_ns=ns, timing=r, accuracy=check)
            )
            atomic(OUT / "accuracy.json", verified)
            log(str(check))
            if check["passed"]:
                break
        return
    winners = {}
    for solver in ["fixed_tsit5", "fixed_ck54"]:
        passed = [
            x for x in verified["candidates"] if x["solver"] == solver and x["accuracy"]["passed"]
        ]
        if passed:
            winners[solver] = max(x["dt_ns"] for x in passed)
    assert winners, "No timestep passed; refine candidates before performance tests"
    patterns = {
        "shared_envelope": rows(np.linspace(17, 22, 32), np.linspace(-1, 1, 32), [0.0]),
        "varying_height": rows(
            np.linspace(17, 22, 16), np.linspace(-1, 1, 16), np.linspace(0, 0.0085, 4)
        ),
    }
    if args.stage == "tune":
        result = dict(winners=winners, cases=[])
        for pattern, pars in patterns.items():
            for solver, ns in winners.items():
                baseline = None
                for rhs, reduced, precompute in [
                    ("tiled", False, True),
                    ("direct", False, True),
                    ("direct", True, True),
                    ("tiled", False, False),
                ]:
                    option = lindblad.CudaSolverOptions(
                        batch_size=1024, rhs=rhs, density_reduction=reduced, precompute=precompute
                    )
                    label = f"{pattern} {solver} {ns:g}ns {rhs} reduced={reduced} precompute={precompute}"
                    log("Screen " + label)
                    a, r = solve(
                        p,
                        rho,
                        pars,
                        solver,
                        ns,
                        option,
                        span=(18e-6, 28e-6),
                        full=True,
                        profile=True,
                    )
                    assert np.isfinite(a.values).all()
                    if baseline is None:
                        baseline = a.values.copy()
                    check = accuracy(p, a.values, baseline, idx)
                    assert check["passed"], check
                    result["cases"].append(
                        dict(
                            pattern=pattern,
                            solver=solver,
                            dt_ns=ns,
                            options=asdict(option),
                            timing=r,
                            accuracy_vs_same_step_tiled=check,
                        )
                    )
                    atomic(OUT / "tune.json", result)
                    log(
                        f"{label}: CUDA {r['cuda_event_seconds']:.3f}s, wall {r['wall_seconds']:.3f}s"
                    )
        return
    tuning = json.loads((OUT / "tune.json").read_text())
    selected = []
    for pattern in patterns:
        cases = [c for c in tuning["cases"] if c["pattern"] == pattern]
        # Retain plain current Tsit5 as a GPU control and the fastest screened
        # option overall. A clearly slower solver need not repeat the full grid.
        control = next(
            c
            for c in cases
            if c["solver"] == "fixed_tsit5"
            and c["options"]["rhs"] == "tiled"
            and c["options"]["precompute"]
            and not c["options"]["density_reduction"]
        )
        chosen = [control, min(cases, key=lambda c: c["timing"]["cuda_event_seconds"])]
        seen = set()
        for c in chosen:
            identity = json.dumps([c["solver"], c["dt_ns"], c["options"]], sort_keys=True)
            if identity not in seen:
                seen.add(identity)
                selected.append(
                    dict(
                        c,
                        identity=identity,
                        role="GPU control" if c is control else "screen winner",
                    )
                )
    if args.stage == "confirm":
        pars = rows([17.0, 19.0, 20.2, 22.0], [-1.0, -0.5, 0.0, 0.5, 1.0], [0.0, 0.00425, 0.0085])
        reference_path = CACHE / "confirm_reference_0.5ns.npz"
        if reference_path.exists():
            saved = np.load(reference_path)
            assert str(saved["fixture_sha256"]) == fixture_sha256, (
                "Stale reference: fixture changed"
            )
            np.testing.assert_array_equal(saved["parameters"], pars)
            reference = saved["values"]
        else:
            log("Confirmation reference: 60 full-transit cases at 0.5 ns, four density checkpoints")
            a, r = solve(
                p,
                rho,
                pars,
                "fixed_tsit5",
                0.5,
                lindblad.CudaSolverOptions(batch_size=128),
                full=True,
                profile=True,
            )
            reference = a.values
            np.savez(
                reference_path, parameters=pars, values=reference, fixture_sha256=fixture_sha256
            )
            atomic(OUT / "confirm_reference.json", r)
        result = dict(cases=[], parameters=pars.tolist(), reference_step_ns=0.5)
        cache = {}
        for c in selected:
            if c["identity"] not in cache:
                ns = c["dt_ns"]
                option = lindblad.CudaSolverOptions(**c["options"])
                for _attempt in range(3):
                    log(
                        f"Confirm {c['solver']} {ns:g} ns {option.rhs}, reduction={option.density_reduction}"
                    )
                    a, r = solve(p, rho, pars, c["solver"], ns, option, full=True, profile=True)
                    check = accuracy(p, a.values, reference, idx)
                    log(str(check))
                    if check["passed"]:
                        break
                    ns *= 0.75
                assert check["passed"], check
                cache[c["identity"]] = dict(dt_ns=ns, accuracy=check, timing=r)
            result["cases"].append(dict(c, confirmation=cache[c["identity"]]))
            atomic(OUT / "confirmation.json", result)
        return
    confirmation = json.loads((OUT / "confirmation.json").read_text())
    result = dict(
        cases=confirmation["cases"],
        timings=[],
        CPU_baseline="not run: CPU simulations still active",
    )
    result_path = OUT / "performance.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        assert [c["identity"] for c in previous["cases"]] == [
            c["identity"] for c in result["cases"]
        ]
        result = previous
    jobs = []
    for i, _c in enumerate(result["cases"]):
        for batch in [512, 1024]:
            for repeat in range(1 if batch == 512 else 3):
                jobs.append((i, batch, repeat))
    rng = np.random.default_rng(14619)
    rng.shuffle(jobs)
    if args.compact:
        jobs = [
            j for j in jobs if j[1] == 512 and result["cases"][j[0]]["options"]["density_reduction"]
        ]
        result["queue_revision"] = dict(
            original_target=16,
            retained_target=10,
            reason="User requested fewer repetitions; preserve completed timings and finish only reduced batch-512 pilots.",
        )
    done = {(r["case"], r["batch_size"], r["repeat"]) for r in result["timings"]}
    jobs = [j for j in jobs if j not in done]
    atomic(result_path, result)
    for job, (i, batch, repeat) in enumerate(jobs):
        c = result["cases"][i]
        option = lindblad.CudaSolverOptions(**dict(c["options"], batch_size=batch))
        log(
            f"Full-transit timing {job + 1}/{len(jobs)}: {c['pattern']} {c['solver']} {c['confirmation']['dt_ns']:g} ns, "
            f"{option.rhs}, reduction={option.density_reduction}, batch={batch}, round={repeat + 1}"
        )
        a, r = solve(
            p,
            rho,
            patterns[c["pattern"]],
            c["solver"],
            c["confirmation"]["dt_ns"],
            option,
            profile=True,
        )
        assert np.isfinite(a.values).all()
        assert abs(a.values.sum(axis=-1) - 1).max() < 1e-6
        result["timings"].append(dict(case=i, batch_size=batch, repeat=repeat, **r))
        atomic(OUT / "performance.json", result)
        log(f"CUDA {r['cuda_event_seconds']:.3f}s; complete public call {r['wall_seconds']:.3f}s")


if __name__ == "__main__":
    main()
