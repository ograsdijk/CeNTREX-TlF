from dataclasses import replace

import numpy as np
import pytest

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.couplings.polarization import Polarization


@pytest.mark.parametrize("compact", [False, True])
def test_microwave_only_obe_retains_both_driven_manifolds(compact):
    transition = transitions.MicrowaveTransition(1, 2)
    selectors = couplings.generate_transition_selectors([transition], [[couplings.polarization_Z]])
    system = lindblad.generate_OBE_system_transitions([transition], selectors, qn_compact=compact)

    assert len(system.QN) == 32
    assert system.excited == []
    assert system.C_array.shape == (0, 32, 32)
    assert system.H_symbolic.shape == (32, 32)
    assert system.dissipator.is_zero_matrix
    assert any(symbol in system.H_symbolic.free_symbols for symbol in system.coupling_symbols)
    assert len(system.couplings[0].excited_states) > 0


def test_direct_microwave_only_obe_can_compact_an_undriven_manifold():
    transition = transitions.MicrowaveTransition(1, 2)
    selectors = couplings.generate_transition_selectors([transition], [[couplings.polarization_Z]])
    system = lindblad.generate_OBE_system(
        states.QuantumSelector(J=[0, 1, 2]),
        [],
        selectors,
        qn_compact=states.QuantumSelector(J=0, electronic=states.ElectronicState.X),
    )

    assert len(system.QN) == 33
    assert system.C_array.shape == (0, 33, 33)
    assert system.H_symbolic.shape == (33, 33)
    assert system.excited == []


def test_automatic_main_selection_honors_polarization_normalization():
    transition = transitions.R0_F1_1o2_F1
    polarization = Polarization(np.array([2.0, 0.0, 0.0], dtype=complex), name="ScaledX")
    selector = couplings.generate_transition_selectors([transition], [[polarization]])[0]
    automatic = replace(selector, ground_main=None, excited_main=None)
    unnormalized = lindblad.generate_OBE_system_transitions(
        [transition], [automatic], normalize_pol=False
    )
    normalized = lindblad.generate_OBE_system_transitions(
        [transition], [automatic], normalize_pol=True
    )

    np.testing.assert_allclose(
        abs(unnormalized.couplings[0].main_coupling),
        2 * abs(normalized.couplings[0].main_coupling),
    )
    assert np.linalg.norm(unnormalized.couplings[0].fields[0].polarization) == 2
    assert np.linalg.norm(normalized.couplings[0].fields[0].polarization) == 1


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("branchings", [[0.1], [0.1, 0.2]])
def test_extra_decay_channels_preserve_index_alignment_and_decay_rate(compact, branchings):
    transition = transitions.R0_F1_1o2_F1
    selectors = couplings.generate_transition_selectors([transition], [[couplings.polarization_X]])
    baseline = lindblad.generate_OBE_system_transitions([transition], selectors, qn_compact=compact)
    channels = [
        lindblad.DecayChannel(
            ground=1
            * states.CoupledBasisState(None, None, None, None, None, None, v=f"loss{index}"),
            excited=transition.qn_select_excited,
            branching=branching,
        )
        for index, branching in enumerate(branchings)
    ]
    system = lindblad.generate_OBE_system_transitions(
        [transition], selectors, qn_compact=compact, decay_channels=channels
    )
    excited_indices = np.asarray(transition.qn_select_excited.get_indices(system.QN)).ravel()
    loss_indices = [system.QN.index(channel.ground) for channel in channels]
    dimension = len(system.QN)
    assert dimension == len(baseline.QN) + len(channels)
    assert system.H_symbolic.shape == (dimension, dimension)
    assert system.C_array.shape[1:] == (dimension, dimension)
    original_dimension = len(system.QN_original) if compact else dimension
    assert system.H_int.shape == (original_dimension, original_dimension)
    assert system.V_ref_int.shape == (original_dimension, original_dimension)
    np.testing.assert_allclose(system.V_ref_int, np.eye(original_dimension))

    rates = np.sum(abs(system.C_array) ** 2, axis=0)
    np.testing.assert_allclose(rates[:, excited_indices].sum(axis=0), hamiltonian.Γ)
    for loss_index, branching in zip(loss_indices, branchings, strict=True):
        np.testing.assert_allclose(rates[loss_index, excited_indices], branching * hamiltonian.Γ)
        assert system.H_symbolic[loss_index, loss_index] == 0
        for coupling in system.couplings:
            for field in coupling.fields:
                assert not np.any(field.field[loss_index])
                assert not np.any(field.field[:, loss_index])

    retained = [index for index in range(dimension) if index not in loss_indices]
    for new_coupling, old_coupling in zip(system.couplings, baseline.couplings, strict=True):
        for new_field, old_field in zip(new_coupling.fields, old_coupling.fields, strict=True):
            np.testing.assert_allclose(new_field.field[np.ix_(retained, retained)], old_field.field)
    assert system.H_symbolic.extract(retained, retained) == baseline.H_symbolic
