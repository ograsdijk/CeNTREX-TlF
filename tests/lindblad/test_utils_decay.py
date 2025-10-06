import numpy as np
import sympy as smp

from centrex_tlf import lindblad, states


def get_insert_indices():
    X_states = states.generate_coupled_states_ground(Js=[0, 1, 2])
    B_states = states.generate_coupled_states_excited(Js=[1], Ps=-1, Omegas=1)
    qn_select_excited = states.QuantumSelector(J=1, electronic=states.ElectronicState.B)
    decay_channels = [
        lindblad.DecayChannel(
            1 * states.CoupledBasisState(None, None, None, None, None, None, v="other"),
            qn_select_excited,
            branching=1e-2,
            description="vibrational branching",
        ),
    ]
    indices = lindblad.utils_decay.get_insert_level_indices(
        decay_channels, list(X_states) + list(B_states), B_states
    )
    assert type(indices) == list
    assert indices == [36]


def test_add_level_symbolic_hamiltonian():
    original = smp.ones(4)
    new = lindblad.utils_decay.add_level_symbolic_hamiltonian(original, 2)
    assert new.shape == (5, 5)
    assert new[:, 2] == smp.zeros(5, 1)
    assert new[2, :] == smp.zeros(1, 5)


def test_add_levels_symbolic_hamiltonian():
    X_states = states.generate_coupled_states_ground(Js=[0, 1, 2])
    B_states = states.generate_coupled_states_excited(Js=[1], Ps=-1, Omegas=1)
    qn_select_excited = states.QuantumSelector(J=1, electronic=states.ElectronicState.B)

    size = len(X_states) + len(B_states)

    H = smp.ones(size)
    decay_channels = [
        lindblad.DecayChannel(
            1 * states.CoupledBasisState(None, None, None, None, None, None, v="other"),
            qn_select_excited,
            branching=1e-2,
            description="vibrational branching",
        ),
        lindblad.DecayChannel(
            1
            * states.CoupledBasisState(None, None, None, None, None, None, v="other 1"),
            qn_select_excited,
            branching=1e-2,
            description="vibrational branching",
        ),
    ]
    indices, arr = lindblad.utils_decay.add_levels_symbolic_hamiltonian(
        H, decay_channels, list(X_states) + list(B_states), B_states
    )
    assert indices == [36, 37]
    assert arr[:, 36] == smp.zeros(size + len(decay_channels), 1)
    assert arr[36, :] == smp.zeros(1, size + len(decay_channels))
    assert arr[:, 37] == smp.zeros(size + len(decay_channels), 1)
    assert arr[37, :] == smp.zeros(1, size + len(decay_channels))


def test_add_states_QN():
    X_states = states.generate_coupled_states_ground(Js=[0, 1, 2])
    B_states = states.generate_coupled_states_excited(Js=[1], Ps=-1, Omegas=1)
    qn_select_excited = states.QuantumSelector(J=1, electronic=states.ElectronicState.B)

    size = len(X_states) + len(B_states)

    decay_channels = [
        lindblad.DecayChannel(
            1 * states.CoupledBasisState(None, None, None, None, None, None, v="other"),
            qn_select_excited,
            branching=1e-2,
            description="vibrational branching",
        ),
        lindblad.DecayChannel(
            1
            * states.CoupledBasisState(None, None, None, None, None, None, v="other 1"),
            qn_select_excited,
            branching=1e-2,
            description="vibrational branching",
        ),
    ]

    indices = lindblad.utils_decay.get_insert_level_indices(
        decay_channels, list(X_states) + list(B_states), B_states
    )

    QN_new = lindblad.utils_decay.add_states_QN(
        decay_channels, list(X_states) + list(B_states), indices
    )

    assert len(QN_new) == size + len(decay_channels)
    assert QN_new[36] == decay_channels[0].ground
    assert QN_new[37] == decay_channels[1].ground


def test_add_decays_C_arrays():
    X_states = states.generate_coupled_states_ground(Js=[0, 1, 2])
    B_states = states.generate_coupled_states_excited(Js=[1], Ps=-1, Omegas=1)
    qn_select_excited = states.QuantumSelector(J=1, electronic=states.ElectronicState.B)

    size = len(X_states) + len(B_states)

    decay_channels = [
        lindblad.DecayChannel(
            1 * states.CoupledBasisState(None, None, None, None, None, None, v="other"),
            qn_select_excited,
            branching=1e-2,
            description="vibrational branching",
        ),
        lindblad.DecayChannel(
            1
            * states.CoupledBasisState(None, None, None, None, None, None, v="other 1"),
            qn_select_excited,
            branching=1e-2,
            description="vibrational branching 1",
        ),
    ]
    indices = lindblad.utils_decay.get_insert_level_indices(
        decay_channels, list(X_states) + list(B_states), B_states
    )

    C_array = np.zeros((3, size, size))
    C_array[0, -1, 0] = 1
    C_array[1, -2, 0] = 1
    C_array[2, -3, 0] = 1
    C_array_new = lindblad.utils_decay.add_decays_C_arrays(
        decay_channels, indices, list(X_states) + list(B_states), C_array, Γ=1
    )
    assert C_array_new.shape == (27, 50, 50)
    assert C_array_new.sum() == 5.4 + 0j
