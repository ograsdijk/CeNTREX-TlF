"""Tests for field-mixing-aware transition validation.

The E1 selection rules only describe couplings between *bare* basis states. In an electric
or magnetic field the eigenstates are superpositions, so nominally forbidden pairs can
become strongly driveable. Setting up an OBE system must therefore decide what is allowed
from the mixed-state dipole matrix element, not from the labels of the dominant components.
"""

import warnings

import numpy as np
import pytest

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.couplings.polarization import Polarization

# A pair that is forbidden under the bare E1 selection rules (ΔF = -2) but which is
# strongly driven at 200 V/cm once the Stark effect mixes the states. Both states lie
# inside the transition's own ground/excited manifolds, so it is a well-posed main pair.
TRANSITION = transitions.P2_F1_3o2_F1
POL_X = Polarization(np.array([1.0, 0.0, 0.0], dtype=np.complex128), name="X")
B_SMALL = np.array([0.0, 0.0, 1e-5])


def _forbidden_main_states():
    ground = 1 * next(
        iter(
            states.generate_coupled_states_X(
                states.QuantumSelector(
                    electronic=states.ElectronicState.X, J=2, F1=2.5, F=3, mF=-1, P=1
                )
            )
        )
    )
    excited = 1 * next(
        iter(
            states.generate_coupled_states_B(
                states.QuantumSelector(
                    electronic=states.ElectronicState.B, J=1, F1=1.5, F=1, mF=0, P=-1
                )
            )
        )
    )
    return ground, excited


def _selectors(ground_main, excited_main):
    return couplings.generate_transition_selectors(
        transitions=[TRANSITION],
        polarizations=[[POL_X]],
        ground_mains=[ground_main],
        excited_mains=[excited_main],
    )


def _build(ground_main, excited_main, E_z):
    return lindblad.generate_OBE_system_transitions(
        [TRANSITION],
        _selectors(ground_main, excited_main),
        qn_compact=True,
        E=np.array([0.0, 0.0, E_z]),
        B=B_SMALL,
        retain_opposite_parity_levels=True,
        normalize_pol=True,
        Jmax_X=4,
        Jmax_B=4,
    )


def test_bare_forbidden_transition_builds_in_field():
    """A ΔF = 2 pair is driveable once the E field mixes the states."""
    ground_main, excited_main = _forbidden_main_states()
    system = _build(ground_main, excited_main, E_z=200.0)

    main = abs(system.couplings[0].main_coupling)
    assert main > 0.1, f"expected a strong field-mixed coupling, got {main}"
    # A field-mixed element derives from the eigenvector composition of a strongly mixed
    # state, so it varies at the ~0.5% level across LAPACK/numpy versions (0.15854 on
    # some, 0.15781 on others). Pin the physics, not the last digits.
    np.testing.assert_allclose(main, 0.1585, rtol=2e-2)


def test_bare_forbidden_transition_still_rejected_without_field():
    """At zero field P, F and mF stay good, so the same pair has an identically zero ME."""
    ground_main, excited_main = _forbidden_main_states()
    with pytest.raises(ValueError) as excinfo:
        _build(ground_main, excited_main, E_z=0.0)

    message = str(excinfo.value)
    assert "is zero" in message
    # The selection rules survive as a diagnostic explaining *why* the element vanished.
    assert "|ΔF| must be ≤ 1" in message


def test_zero_field_rule_violating_couplings_vanish_identically():
    """The numeric test reduces exactly to the selection rules at zero field.

    Guards the claim that no field-strength branch is needed: at E = 0 every coupling that
    violates the bare rules is a hard zero, while at 200 V/cm one reaches a sizeable
    fraction of the strongest element in the matrix.
    """
    chk = couplings.utils.check_transition_coupled_allowed_polarization
    dmF = couplings.utils.ΔmF_allowed(POL_X.vector)

    worst = {}
    for E_z in (0.0, 200.0):
        H = hamiltonian.generate_reduced_hamiltonian_transitions(
            [TRANSITION], E=np.array([0.0, 0.0, E_z]), B=B_SMALL
        )
        C = np.abs(
            couplings.generate_coupling_matrix(
                H.QN,
                H.X_states,
                H.B_states,
                pol_vec=POL_X.vector,
                reduced=False,
                normalize_pol=True,
            )
        )
        violating = 0.0
        for i, si in enumerate(H.QN):
            for j, sj in enumerate(H.QN):
                if C[i, j] < 1e-12:
                    continue
                a, b = si.largest, sj.largest
                if a.electronic_state == b.electronic_state:
                    continue
                if a.electronic_state == states.ElectronicState.X:
                    g, e = a, b
                else:
                    g, e = b, a
                if not chk(g, e, dmF, return_err=False):
                    violating = max(violating, C[i, j])
        worst[E_z] = (violating, C.max())

    assert worst[0.0][0] == 0.0, "rule-violating couplings must vanish at zero field"
    # At field the effect is dominant, not a rounding-level artefact.
    assert worst[200.0][0] / worst[200.0][1] > 0.5


def _coupling_field(selector, H, **kwargs):
    return couplings.generate_coupling_field(
        ground_main_approx=selector.ground_main,
        excited_main_approx=selector.excited_main,
        ground_states_approx=selector.ground,
        excited_states_approx=selector.excited,
        QN_basis=H.QN_basis,
        H_rot=H.H_int,
        QN=H.QN,
        V_ref=H.V_ref_int,
        pol_main=POL_X.vector,
        pol_vecs=[POL_X.vector],
        **kwargs,
    )


def test_weak_main_coupling_warns():
    """main_coupling normalizes the Rabi rate, so a weak reference must be flagged."""
    ground_main, excited_main = _forbidden_main_states()
    selector = _selectors(ground_main, excited_main)[0]
    H = hamiltonian.generate_reduced_hamiltonian_transitions(
        [TRANSITION], E=np.array([0.0, 0.0, 200.0]), B=B_SMALL
    )

    # weak_main_fraction=1.0 forces the warning: this main pair is by construction not the
    # strongest element in the matrix.
    with pytest.warns(UserWarning, match="main coupling"):
        field = _coupling_field(selector, H, weak_main_fraction=1.0)
    assert abs(field.main_coupling) > 0

    # The default threshold (1e-2) must not fire for this comfortably strong pair.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _coupling_field(selector, H)


def test_field_mixed_transition_drives_population():
    """The newly permitted coupling actually drives population, and rho stays physical."""
    from centrex_tlf.utils import population

    ground_main, excited_main = _forbidden_main_states()
    system = _build(ground_main, excited_main, E_z=200.0)
    selector = _selectors(ground_main, excited_main)[0]

    params = lindblad.LindbladParameters()
    rabi = params.real("rabi", 2 * np.pi * 5e6)
    detuning = params.real("detuning", 0.0)
    for symbol in system.H_symbolic.free_symbols:
        if symbol in system.coupling_symbols:
            params.bind(symbol, rabi, finalize=False)
        elif str(symbol) == str(selector.δ):
            params.bind(symbol, detuning, finalize=False)
        else:
            params.real(str(symbol), 0.0)
    for group in system.polarization_symbols:
        for symbol in group if isinstance(group, (list, tuple)) else [group]:
            params.bind(symbol, 1.0, finalize=False)
    params._finalize()

    prepared = lindblad.prepare_lindblad_problem(
        system, params, backend="rust", hamiltonian_representation="decomposed"
    )

    n = len(system.QN)
    indices = np.asarray(
        states.QuantumSelector(J=2, F1=2.5, F=3, mF=-1).get_indices(system.QN), dtype=int
    ).ravel()
    rho0 = population.generate_uniform_population_state_indices(indices, n)

    t_end = 1e-6
    saveat = np.linspace(0.0, t_end, 51)
    result = lindblad.solve_lindblad(
        prepared,
        rho0,
        (0.0, t_end),
        solver="dopri5",
        saveat=saveat,
        dt=2e-9,
        reltol=1e-7,
        abstol=1e-9,
    )

    rho = np.asarray(result.density_matrices()).reshape(-1, n, n)
    traces = np.einsum("...ii->...", rho).real
    np.testing.assert_allclose(traces, 1.0, atol=1e-6)

    # Population must actually leave the initial level via the field-mixed coupling.
    initial_pop = rho[:, indices, indices].real.sum(axis=1)
    assert initial_pop.min() < 0.99


def test_automatic_main_selection_matches_bare_rules_at_any_field():
    """The automatic path must keep picking the bare-allowed main pair.

    Regression guard: the main pair is the Rabi normalization reference, so admitting
    mixing-only pairs as candidates must not let a weak one displace a properly allowed
    one. An earlier version ranked candidates by scan position, which silently switched
    to a ~7x weaker main pair at 200 V/cm.
    """
    selector = couplings.generate_transition_selectors(
        transitions=[TRANSITION], polarizations=[[POL_X]]
    )[0]

    for E_z in (0.0, 200.0, 1000.0):
        H = hamiltonian.generate_reduced_hamiltonian_transitions(
            [TRANSITION], E=np.array([0.0, 0.0, E_z]), B=B_SMALL
        )
        field = couplings.generate_coupling_field_automatic(
            selector.ground,
            selector.excited,
            H.QN_basis,
            H.H_int,
            H.QN,
            H.V_ref_int,
            pol_vecs=[POL_X.vector],
        )
        expected_g, expected_e = couplings.utils.select_main_states(
            selector.ground, selector.excited, POL_X.vector
        )
        assert str(field.ground_main.largest) == str(expected_g.largest), E_z
        assert str(field.excited_main.largest) == str(expected_e.largest), E_z

        # The chosen main must be a strong reference, not a mixing-only element.
        strongest = max(float(np.abs(c.field).max()) for c in field.fields)
        assert abs(field.main_coupling) > 0.1 * strongest, E_z


def test_automatic_main_selection_falls_back_to_field_mixed_pairs():
    """With no bare-allowed pair the automatic path takes the strongest mixed coupling."""
    ground = [
        1 * s
        for s in states.generate_coupled_states_X(
            states.QuantumSelector(
                electronic=states.ElectronicState.X, J=2, F1=2.5, F=3, P=1
            )
        )
    ]
    excited = [
        1 * s
        for s in states.generate_coupled_states_B(
            states.QuantumSelector(
                electronic=states.ElectronicState.B, J=1, F1=1.5, F=1, P=-1, Ω=1
            )
        )
    ]
    # Every pair here is ΔF = 2, so the bare selection rules admit none of them.
    with pytest.raises(ValueError):
        couplings.utils.select_main_states(ground, excited, POL_X.vector)

    H = hamiltonian.generate_reduced_hamiltonian_transitions(
        [TRANSITION], E=np.array([0.0, 0.0, 200.0]), B=B_SMALL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = couplings.generate_coupling_field_automatic(
            ground, excited, H.QN_basis, H.H_int, H.QN, H.V_ref_int,
            pol_vecs=[POL_X.vector],
        )

    # The fallback ranks by magnitude rather than by scan position, subject to the
    # inherited preference for an mF = 0 ground state.
    from centrex_tlf.couplings.coupling_matrix import _dress_states

    g = _dress_states(ground, H.QN_basis, H.QN, H.H_int, H.V_ref_int)
    e = _dress_states(excited, H.QN_basis, H.QN, H.H_int, H.V_ref_int)
    best_mF0 = max(
        abs(hamiltonian.generate_ED_ME_mixed_state(ex, gd, pol_vec=POL_X.vector))
        for ex in e
        for gd in g
        if gd.largest.mF == 0
    )
    np.testing.assert_allclose(abs(field.main_coupling), best_mF0, rtol=1e-6)
    assert field.ground_main.largest.mF == 0
