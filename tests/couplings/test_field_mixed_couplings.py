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
    # This element is dominated by the F=2/F=1 admixtures, not the nominal F=3 component,
    # and the ground state has a degenerate mF=+1 partner split by only ~6e-8 MHz at
    # B=1e-5 G. That gap is ~3000x machine epsilon relative to ||H||, so the mixing angle
    # within the pair — and hence this matrix element — depends on the BLAS build:
    # measured 0.15854 (scipy-openblas 0.3.29) vs 0.15781 (0.3.34). Pin the physics, not
    # digits that are not reproducible across platforms. See the B-field note in AGENTS.md.
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


def test_bare_state_rule_violating_matrix_elements_are_exactly_zero():
    """At true zero field (E = B = 0) the bare selection rules are exact, not a limit.

    This checks the *bare* dipole matrix element ⟨e|D̂·ε|g⟩ between undressed basis
    states directly (no Hamiltonian diagonalization, hence no near-degenerate
    eigenvectors to be uncertain about): every ground/excited pair the selection
    rules forbid must give a matrix element of exactly zero.
    """
    selector = couplings.generate_transition_selectors(
        transitions=[TRANSITION], polarizations=[[POL_X]]
    )[0]
    chk = couplings.utils.check_transition_coupled_allowed_polarization
    dmF = couplings.utils.ΔmF_allowed(POL_X.vector)

    checked_a_violation = False
    for ground in selector.ground:
        for excited in selector.excited:
            if chk(ground.largest, excited.largest, dmF, return_err=False):
                continue
            checked_a_violation = True
            ME = hamiltonian.generate_ED_ME_mixed_state(excited, ground, pol_vec=POL_X.vector)
            assert ME == 0j, (
                f"rule-forbidden bare pair {ground.largest} -> {excited.largest} "
                f"has nonzero matrix element {ME}"
            )
    assert checked_a_violation, "test setup must include at least one forbidden pair"


def test_near_zero_field_rule_violating_couplings_strongly_suppressed():
    """B_SMALL = [0, 0, 1e-5] G is a near-zero field, not exact zero.

    It only exists to lift exact degeneracies for numerical state identification (see
    AGENTS.md's B-field placeholder gotcha); it can still weakly mix F states, so
    nominally forbidden couplings need not be mathematically zero here, only strongly
    suppressed relative to the field-driven regime at 200 V/cm.
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

    # At near-zero field (E=0, B=B_SMALL) rule-violating couplings must still be
    # suppressed to at most a rounding-level artefact: B=[0,0,1e-5] G only just lifts
    # the ±mF degeneracy, so the mixing angle within a ±mF pair is only determined to
    # the ~1e-4..1e-2 level and is BLAS-build dependent (see AGENTS.md). This loose
    # tolerance still cleanly separates this regime from the field-driven one below.
    assert worst[0.0][0] / worst[0.0][1] < 1e-2, (
        "rule-violating couplings must be at most a rounding-level artefact at near-zero field"
    )
    # At 200 V/cm the effect is dominant, not a rounding-level artefact.
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
            states.QuantumSelector(electronic=states.ElectronicState.X, J=2, F1=2.5, F=3, P=1)
        )
    ]
    excited = [
        1 * s
        for s in states.generate_coupled_states_B(
            states.QuantumSelector(electronic=states.ElectronicState.B, J=1, F1=1.5, F=1, P=-1, Ω=1)
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
            ground,
            excited,
            H.QN_basis,
            H.H_int,
            H.QN,
            H.V_ref_int,
            pol_vecs=[POL_X.vector],
        )

    # The fallback ranks by magnitude rather than by scan position. The inherited
    # preference for an mF = 0 ground state only applies while that pair stays within
    # mF0_preference_fraction of the strongest coupling; here the best mF = 0 pair is
    # five times weaker, so the strongest pair wins.
    from centrex_tlf.couplings.coupling_matrix import (
        _dress_states,
        select_main_states_indices_coupling,
    )

    g = _dress_states(ground, H.QN_basis, H.QN, H.H_int, H.V_ref_int)
    e = _dress_states(excited, H.QN_basis, H.QN, H.H_int, H.V_ref_int)

    def me(gd, ex):
        return abs(hamiltonian.generate_ED_ME_mixed_state(ex, gd, pol_vec=POL_X.vector))

    best = max(me(gd, ex) for ex in e for gd in g)
    best_mF0 = max(me(gd, ex) for ex in e for gd in g if gd.largest.mF == 0)
    assert best_mF0 < 0.5 * best

    np.testing.assert_allclose(abs(field.main_coupling), best, rtol=1e-6)

    # the preference is a knob, not a rule: disabling the strength gate restores the
    # historical mF = 0 choice
    g_idx, e_idx = select_main_states_indices_coupling(
        g, e, POL_X.vector, mF0_preference_fraction=0.0
    )
    assert g[g_idx].largest.mF == 0
    np.testing.assert_allclose(me(g[g_idx], e[e_idx]), best_mF0, rtol=1e-6)
