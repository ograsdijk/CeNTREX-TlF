"""Frozen model builders for the cross-system Lindblad screen.

Importing this module is cheap; ``build_fixture`` builds model/prepared plans
but never propagates a physical trajectory.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.couplings.polarization import Polarization
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem

HERE = Path(__file__).resolve().parent
GAMMA = float(hamiltonian.Γ)
E_FIELD_V_CM = (0.0, 0.0, -170.0)
B_FIELD_G = (-0.01327, 0.19433, -0.48889)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    transition_name: str | None
    compact: bool
    t_span: tuple[float, float]
    rtol: float = 1e-7
    atol: float = 1e-9
    dt: float = 2e-9
    e_field_v_cm: tuple[float, float, float] = E_FIELD_V_CM
    b_field_g: tuple[float, float, float] = B_FIELD_G


SPECS = {
    "r2_f4": FixtureSpec("r2_f4", "R2_F1_7o2_F4", True, (0.0, 1.0e-6)),
    "r3_f1_7o2_f4": FixtureSpec("r3_f1_7o2_f4", "R3_F1_7o2_F4", True, (0.0, 1.0e-6)),
    "r3_f1_9o2_f4": FixtureSpec("r3_f1_9o2_f4", "R3_F1_9o2_F4", True, (0.0, 1.0e-6)),
    "rc": FixtureSpec("rc", None, False, (125e-9, 1.125e-6)),
}


@dataclass(frozen=True)
class CrossSystemFixture:
    spec: FixtureSpec
    system: Any
    prepared: Any
    rho0: np.ndarray
    initial_index: int
    initial_label: str
    selected_indices_1: tuple[int, ...]
    selected_indices_4: tuple[int, ...]
    selected_labels_4: tuple[str, ...]
    group_indices: tuple[int, ...]
    group_label: str
    excited_indices: tuple[int, ...]
    main_pair_labels: dict[str, str]
    main_coupling: float
    runtime_values: dict[str, Any]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    rc_case: Any = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_provenance() -> dict[str, str]:
    native = importlib.import_module("centrex_tlf.centrex_tlf_rust")
    native_path = Path(native.__file__).resolve()
    if native_path.suffix.lower() != ".pyd":
        raise RuntimeError(f"loaded native extension is not a .pyd binary: {native_path}")
    return {"native_extension_path": str(native_path),
            "native_extension_sha256": _sha256(native_path)}


def _source_provenance(*paths: Path) -> dict[str, str]:
    result = {"fixture_builder_sha256": _sha256(Path(__file__).resolve())}
    for path in paths:
        result[f"{path.stem}_sha256"] = _sha256(path.resolve())
    return result


def _selector_index(selector: Any, qn: Any, label: str) -> int:
    indices = np.asarray(selector.get_indices(qn), dtype=int).ravel()
    if indices.size != 1:
        raise RuntimeError(f"{label} resolves to {indices.size} states, expected exactly one")
    return int(indices[0])


def _state_label(state: Any) -> str:
    return state.largest.state_string_custom(
        ["electronic", "J", "F1", "F", "mF", "P", "Ω"]
    )


def _explicit_ground_selectors(j: int, f1: float, f: float) -> tuple[list[Any], list[str]]:
    # Four distinct mF labels from the same driven hyperfine manifold.
    ms = (-1, 0, 1, 2)
    selectors = [states.QuantumSelector(
        electronic=states.ElectronicState.X, J=j, F1=f1, F=int(f), mF=m,
        P=(-1) ** j,
    ) for m in ms]
    labels = [f"X_J{j}_F1_{str(f1).replace('.', 'o')}_F{str(f).replace('.', 'o')}_mF{m}"
              for m in ms]
    return selectors, labels


def _transition_fixture(spec: FixtureSpec) -> CrossSystemFixture:
    transition = getattr(transitions, str(spec.transition_name))
    j = int(transition.J_ground)
    f1g, fg = j + 0.5, j + 1.0
    ground_selector = states.QuantumSelector(
        electronic=states.ElectronicState.X, J=j, F1=f1g, F=int(fg), mF=0, P=(-1) ** j,
    )
    ground_main = list(states.generate_coupled_states_X(ground_selector))
    if len(ground_main) != 1:
        raise RuntimeError(f"{spec.name}: main ground label resolves to {len(ground_main)} basis states")
    ground_main_state = (1 * ground_main[0]).normalize()
    excited_selector = states.QuantumSelector(
        electronic=states.ElectronicState.B, J=int(transition.J_excited),
        F1=float(transition.F1_excited), F=int(transition.F_excited), mF=1,
        P=-((-1) ** j),
    )
    excited_main = list(states.generate_coupled_states_B(excited_selector))
    if len(excited_main) != 1:
        raise RuntimeError(f"{spec.name}: main excited label resolves to {len(excited_main)} basis states")
    excited_main_state = (1 * excited_main[0]).normalize()
    polarization = Polarization(np.array([1.0, 0.0, 0.0], dtype=np.complex128), name="X")
    selectors = couplings.generate_transition_selectors(
        [transition], [[polarization]], ground_mains=[ground_main_state],
        excited_mains=[excited_main_state],
    )
    j_excited = int(transition.J_excited)
    jmax_x, jmax_b = j + 2, j_excited + 2
    system = lindblad.generate_OBE_system_transitions(
        [transition], selectors, qn_compact=True,
        E=np.asarray(spec.e_field_v_cm), B=np.asarray(spec.b_field_g),
        normalize_pol=True, retain_opposite_parity_levels=True,
        Jmin_X=0, Jmax_X=jmax_x, Jmin_B=1, Jmax_B=jmax_b,
    )

    params = LindbladParameters()
    rabi = params.real("rabi", GAMMA)
    detuning = params.real("detuning", 0.0)
    for symbol in system.H_symbolic.free_symbols:
        if symbol in system.coupling_symbols:
            params.bind(symbol, rabi, finalize=False)
        elif str(symbol) == str(selectors[0].δ):
            params.bind(symbol, detuning, finalize=False)
        else:
            params.real(str(symbol), 0.0)
    for group in system.polarization_symbols:
        for symbol in group if isinstance(group, list | tuple) else [group]:
            params.bind(symbol, 1.0, finalize=False)
    params._finalize()
    prepared = prepare_lindblad_problem(
        system, params, backend="rust", hamiltonian_representation="decomposed",
    )

    initial_index = _selector_index(ground_selector, system.QN, "main initial label")
    # The main state's dressed label identifies one resolved system basis
    # state; initialize that state in the prepared model basis.
    rho0 = np.zeros((len(system.QN), len(system.QN)), dtype=np.complex128)
    rho0[initial_index, initial_index] = 1.0
    if rho0.shape != (len(system.QN), len(system.QN)) or not np.isclose(np.trace(rho0), 1.0):
        raise RuntimeError(f"{spec.name}: normalized main-state density matrix is invalid")

    four_selectors, four_labels = _explicit_ground_selectors(j, f1g, fg)
    four_indices = tuple(_selector_index(sel, system.QN, label)
                         for sel, label in zip(four_selectors, four_labels, strict=True))
    if len(set(four_indices)) != 4:
        raise RuntimeError(f"{spec.name}: selected four labels are not distinct resolved states")
    b_indices = tuple(i for i, qn in enumerate(system.QN)
                      if qn.largest.electronic_state == states.ElectronicState.B)
    if not b_indices:
        raise RuntimeError(f"{spec.name}: no retained B-manifold states")
    _selector_index(excited_selector, system.QN, "main excited label")
    main_coupling = float(abs(system.couplings[0].main_coupling))
    if not np.isfinite(main_coupling) or main_coupling <= 0:
        raise RuntimeError(f"{spec.name}: invalid main coupling {main_coupling}")

    configuration = {
        "transition": spec.transition_name,
        "compact": True,
        "retain_opposite_parity_levels": True,
        "E_V_cm": list(spec.e_field_v_cm), "B_G": list(spec.b_field_g),
        "Jmin_X": 0, "Jmax_X": jmax_x, "Jmin_B": 1, "Jmax_B": jmax_b,
        "polarization": [1.0, 0.0, 0.0], "rabi_rad_s": GAMMA,
        "detuning_rad_s": 0.0, "decay_rate_rad_s": GAMMA,
        "solver": "dopri5", "execution_mode": "structured",
        "t_span_s": list(spec.t_span), "dt_s": spec.dt,
        "rtol": spec.rtol, "atol": spec.atol,
        "hamiltonian_representation": "decomposed",
    }
    metadata = {
        "state_count": len(system.QN), "packed_length": prepared.layout.packed_len,
        "main_ground_label": _state_label(ground_main_state),
        "main_excited_label": _state_label(excited_main_state),
        "initial_label": _state_label(system.QN[initial_index]),
        "selected_labels_4": list(four_labels), "selected_indices_4": list(four_indices),
        "group_label": "B_total", "group_indices": list(b_indices),
        "excited_indices": list(b_indices), "main_coupling": main_coupling,
        "runtime_values": {"rabi_rad_s": GAMMA, "detuning_rad_s": 0.0,
                           "polarization_amplitude": 1.0},
    }
    metadata.update(_native_provenance())
    metadata.update(_source_provenance())
    return CrossSystemFixture(
        spec, system, prepared, rho0, initial_index, metadata["initial_label"],
        (initial_index,), four_indices, tuple(four_labels), b_indices, "B_total", b_indices,
        {"ground": metadata["main_ground_label"], "excited": metadata["main_excited_label"]},
        main_coupling, {"rabi_rad_s": GAMMA, "detuning_rad_s": 0.0,
                        "polarization_amplitude": 1.0}, configuration, metadata,
    )


def _rc_fixture(spec: FixtureSpec) -> CrossSystemFixture:
    try:
        rc = importlib.import_module("bench_rc_propagation")
    except ModuleNotFoundError:
        rc = importlib.import_module("benchmarks.bench_rc_propagation")

    config = rc.BenchmarkConfig("nominal", 1.0, 1.0, 0.0, 1.0e-6, 125e-9)
    case = rc.build_case(config, prepare_frozen=True)
    initial_label = "X_J2_F1_5o2_F2_mF0"
    initial_index = int(case.state_labels[initial_label])
    rho0 = case.rho0(initial_label)
    selectors, four_labels = _explicit_ground_selectors(2, 2.5, 2.0)
    four_indices = tuple(_selector_index(sel, case.system.QN, label)
                         for sel, label in zip(selectors, four_labels, strict=True))
    if len(set(four_indices)) != 4:
        raise RuntimeError("rc: selected four labels are not distinct resolved states")
    b_indices = tuple(i for i, qn in enumerate(case.system.QN)
                      if qn.largest.electronic_state == states.ElectronicState.B)
    main_coupling = float(abs(case.system.couplings[0].main_coupling))
    if not np.isfinite(main_coupling) or main_coupling <= 0:
        raise RuntimeError(f"rc: invalid main coupling {main_coupling}")
    config_payload = asdict(config)
    config_payload.update({
        "optical_rabi_rad_s": config.optical_rabi_rad_s,
        "microwave_rabi_rad_s": config.microwave_rabi_rad_s,
        "detuning_rad_s": config.detuning_rad_s,
        "E_V_cm": [0.0, 0.0, rc.E_FIELD_V_CM],
        "B_G": [0.0, 0.0, rc.B_FIELD_G],
        "eom_frequency_hz": rc.F_EOM_HZ, "eom_beta": rc.BETA_EOM,
        "switch_frequency_hz": rc.F_SWITCH_HZ,
        "switch_phase_offset_rad": rc.SWITCH_PHASE_OFFSET_RAD,
        "switch_override_slots": list(case.switch_override_slots),
    })
    configuration = {
        **config_payload, "compact": False, "retain_opposite_parity_levels": True,
        "Jmax_X": 5, "Jmax_B": 3, "solver": "dopri5",
        "execution_mode": "structured", "t_span_s": list(spec.t_span),
        "dt_s": spec.dt, "rtol": spec.rtol, "atol": spec.atol,
        "hamiltonian_representation": "decomposed",
        "amplitude_envelope": case.amplitude_envelope,
    }
    runtime_values = {
        "optical_rabi_rad_s": config.optical_rabi_rad_s,
        "microwave_rabi_rad_s": config.microwave_rabi_rad_s,
        "detuning_rad_s": config.detuning_rad_s,
        "eom_beta": rc.BETA_EOM, "eom_angular_frequency_rad_s": 2 * np.pi * rc.F_EOM_HZ,
        "switch_angular_frequency_rad_s": 2 * np.pi * rc.F_SWITCH_HZ,
        "switch_phase_offset_rad": rc.SWITCH_PHASE_OFFSET_RAD,
    }
    main_pair = {
        "ground": _state_label(case.system.couplings[0].ground_main),
        "excited": _state_label(case.system.couplings[0].excited_main),
    }
    metadata = {
        "state_count": len(case.system.QN), "packed_length": case.prepared_segmented.layout.packed_len,
        "initial_label": initial_label, "selected_labels_4": list(four_labels),
        "selected_indices_4": list(four_indices), "group_label": "B_total",
        "group_indices": list(b_indices), "excited_indices": list(b_indices),
        "main_pair_labels": main_pair, "main_coupling": main_coupling,
        "runtime_values": runtime_values, "configuration": configuration,
        "entry_phase_s": config.entry_phase_s,
        "switch_override_slots": list(case.switch_override_slots),
    }
    metadata.update(_native_provenance())
    metadata.update(_source_provenance(HERE / "bench_rc_propagation.py"))
    return CrossSystemFixture(
        spec, case.system, case.prepared_segmented, rho0, initial_index, initial_label,
        (initial_index,), four_indices, tuple(four_labels), b_indices, "B_total", b_indices,
        main_pair, main_coupling, runtime_values, configuration, metadata, rc_case=case,
    )


def build_fixture(name: str) -> CrossSystemFixture:
    """Build one named fixture and its Rust prepared plan; no integration is run."""
    try:
        spec = SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown fixture {name!r}; choose from {tuple(SPECS)}") from exc
    if name == "rc":
        return _rc_fixture(spec)
    return _transition_fixture(spec)
