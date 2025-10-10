import numpy as np
import pytest

from centrex_tlf import states


def test_state_initialization():
    s = states.CoupledBasisState(
        F=1, mF=0, F1=1 / 2, J=0, I1=1 / 2, I2=1 / 2, Omega=0, P=1
    )
    assert s == s
    assert isinstance(s, type(s))
    assert isinstance(s, states.CoupledBasisState)
    assert s.F == 1
    assert s.F1 == 1 / 2
    assert s.mF == 0
    assert s.J == 0
    assert s.I1 == 1 / 2
    assert s.I2 == 1 / 2
    assert s.Omega == 0
    assert s.P == 1


def test_state_matmul():
    QN_coupled = states.generate_coupled_states_ground(Js=[0, 1])
    QN_uncoupled = states.generate_uncoupled_states_ground(Js=[0, 1])
    assert QN_coupled[0] @ QN_coupled[0] == 1
    assert QN_coupled[0] @ QN_coupled[1] == 0
    assert QN_uncoupled[0] @ QN_uncoupled[0] == 1
    assert QN_uncoupled[0] @ QN_uncoupled[1] == 0
    assert QN_coupled[0] @ QN_uncoupled[1] == -0.7071067811865476 + 0j
    assert QN_coupled[0] @ QN_uncoupled[0] == 0 + 0j
    assert QN_uncoupled[1] @ QN_coupled[0] == -0.7071067811865476 + 0j
    assert QN_uncoupled[0] @ QN_coupled[0] == 0 + 0j
    with pytest.raises(TypeError):
        QN_coupled[0] @ 2


def test_basisstate_add():
    QN_coupled = states.generate_coupled_states_ground(Js=[0, 1])
    QN_uncoupled = states.generate_uncoupled_states_ground(Js=[0, 1])
    state1 = QN_coupled[0]
    state2 = QN_coupled[5]
    assert state1 + state2 == states.CoupledState([(1, state1), (1, state2)])
    with pytest.raises(TypeError):
        QN_coupled[0] + QN_uncoupled[5]


def test_CGc():
    assert states.CGc(1.0, -1.0, 1.0, 1.0, 1.0, 0.0) == -0.7071067811865475 + 0j


def test_parity_X():
    assert states.parity_X(0) == 1
    assert states.parity_X(1) == -1


def test_reorder_evecs():
    V_in = np.eye(10, dtype=np.complex128)
    E_in = np.random.rand(10).astype(np.complex128)
    V_ref = V_in.copy()
    ida = np.random.randint(0, 10)
    idb = np.random.randint(0, 10)
    _ = V_ref[ida, :].copy()
    V_ref[ida, :] = V_ref[idb, :]
    V_ref[idb, :] = _

    E_out, V_out = states.reorder_evecs(V_in, E_in, V_ref)
    assert E_out[ida] == E_in[idb]
    assert E_out[idb] == E_in[ida]


def test_quantumselector_get_indices():
    QN = states.generate_coupled_states_ground(Js=[0, 1])
    qn_select = states.QuantumSelector(J=1, mF=0, electronic=states.ElectronicState.X)
    assert np.all(
        qn_select.get_indices(QN, mode="python")
        == np.array([4, 6, 9, 13], dtype=np.int64)
    )
    assert np.all(
        qn_select.get_indices(QN, mode="julia")
        == np.array([4, 6, 9, 13], dtype=np.int64) + 1
    )


def test_find_closest_vector_idx():
    vector_array = np.eye(10).astype(np.complex128)
    idx_compare = np.random.randint(0, 10)
    state_vec = vector_array[idx_compare, :]
    idx = states.find_closest_vector_idx(state_vec, vector_array)
    assert idx == idx_compare


def test_check_approx_state_exact_state():
    QN = states.generate_coupled_states_ground(Js=[0, 1])
    ida = np.random.randint(0, len(QN))
    idb = np.random.randint(0, len(QN))
    approx = 1 * QN[ida]
    exact = 1 * QN[ida] + 0.1 * QN[idb]
    states.check_approx_state_exact_state(approx, exact)


def test_BasisStates_from_State():
    QN = states.generate_coupled_states_ground(Js=[0, 1])
    QN_State = 1 * QN
    QN_BasisState = states.BasisStates_from_State(QN_State)
    assert np.all(QN == QN_BasisState)


def test_get_indices_quantumnumbers_base():
    QN = states.generate_coupled_states_ground(Js=[0, 1])
    qn_select = states.QuantumSelector(J=1, mF=0, electronic=states.ElectronicState.X)
    indices = states.get_indices_quantumnumbers_base(qn_select, QN, mode="python")
    assert np.all(indices == np.array([4, 6, 9, 13], dtype=np.int64))
    indices = states.get_indices_quantumnumbers_base(qn_select, QN, mode="julia")
    assert np.all(indices == (np.array([4, 6, 9, 13], dtype=np.int64) + 1))


def test_get_indices_quantumnumbers():
    QN = states.generate_coupled_states_ground(Js=[0, 1])
    qn_select = [
        states.QuantumSelector(J=1, mF=0, electronic=states.ElectronicState.X),
        states.QuantumSelector(J=1, mF=1, electronic=states.ElectronicState.X),
    ]
    indices = states.get_indices_quantumnumbers(qn_select, QN)
    assert np.all(indices == np.array([4, 6, 7, 9, 10, 13, 14], dtype=np.int64))


def test_get_unique_basisstates():
    QN_original = states.generate_coupled_states_ground(Js=[0, 1])
    QN = np.append(QN_original, QN_original)
    QN_unique = states.get_unique_basisstates_from_basisstates(QN)
    assert np.all(QN_original == QN_unique)


def test_state_add():
    ss = states.generate_coupled_states_ground(Js=[0, 1])
    state1 = states.CoupledState([(1.0, s) for s in ss])
    ss = states.generate_coupled_states_ground(Js=[1, 2])
    state2 = states.CoupledState([(2.0, s) for s in ss])
    state_sum = state1 + state2
    state_sum = state_sum.order_by_amp()
    for a, s in state_sum.data:
        if s.J == 0:
            assert a == 1.0
        elif s.J == 1:
            assert a == 3.0
        elif s.J == 2:
            assert a == 2.0


def test_hash():
    # Test that all states have unique hashes (no collisions)
    s = states.generate_uncoupled_states_ground(np.arange(0, 15))
    assert (
        len([si.__hash__() for si in s]) == np.unique([si.__hash__() for si in s]).size
    )
    s = states.generate_coupled_states_ground(np.arange(0, 15))
    assert (
        len([si.__hash__() for si in s]) == np.unique([si.__hash__() for si in s]).size
    )

    s = states.generate_coupled_states_excited(
        np.arange(0, 15),
        Ps=None,
        Omegas=[-1, 1],  # type: ignore[arg-type]
    )
    assert (
        len([si.__hash__() for si in s]) == np.unique([si.__hash__() for si in s]).size
    )


def test_hash_stability_and_uniqueness():
    """Test that hash values are stable and unique for both CoupledBasisState and UncoupledBasisState.
    
    This test verifies:
    1. Hash values are consistent across multiple calls
    2. Equal states produce equal hashes
    3. Different states produce different hashes
    4. No collisions occur in large state spaces
    """
    # Test CoupledBasisState hash stability
    state1 = states.CoupledBasisState(
        F=1, mF=0, F1=1/2, J=0, I1=1/2, I2=1/2, Omega=0, P=1
    )
    hash1_call1 = hash(state1)
    hash1_call2 = hash(state1)
    assert hash1_call1 == hash1_call2, "Hash should be stable across calls"
    
    # Test equal states have equal hashes
    state1_copy = states.CoupledBasisState(
        F=1, mF=0, F1=1/2, J=0, I1=1/2, I2=1/2, Omega=0, P=1
    )
    assert hash(state1) == hash(state1_copy), "Equal states must have equal hashes"
    assert state1 == state1_copy, "States should be equal"
    
    # Test different states have different hashes
    state2 = states.CoupledBasisState(
        F=1, mF=1, F1=1/2, J=0, I1=1/2, I2=1/2, Omega=0, P=1
    )
    assert hash(state1) != hash(state2), "Different states should have different hashes"
    assert state1 != state2, "States should not be equal"
    
    # Test UncoupledBasisState hash stability
    ustate1 = states.UncoupledBasisState(
        J=0, mJ=0, I1=1/2, m1=1/2, I2=1/2, m2=-1/2, P=1, Omega=0,
        electronic_state=states.ElectronicState.X
    )
    uhash1_call1 = hash(ustate1)
    uhash1_call2 = hash(ustate1)
    assert uhash1_call1 == uhash1_call2, "Hash should be stable across calls"
    
    # Test equal uncoupled states have equal hashes
    ustate1_copy = states.UncoupledBasisState(
        J=0, mJ=0, I1=1/2, m1=1/2, I2=1/2, m2=-1/2, P=1, Omega=0,
        electronic_state=states.ElectronicState.X
    )
    assert hash(ustate1) == hash(ustate1_copy), "Equal states must have equal hashes"
    assert ustate1 == ustate1_copy, "States should be equal"
    
    # Test different uncoupled states have different hashes
    ustate2 = states.UncoupledBasisState(
        J=0, mJ=0, I1=1/2, m1=-1/2, I2=1/2, m2=-1/2, P=1, Omega=0,
        electronic_state=states.ElectronicState.X
    )
    assert hash(ustate1) != hash(ustate2), "Different states should have different hashes"
    assert ustate1 != ustate2, "States should not be equal"
    
    # Test no collisions in large state space for coupled states
    coupled_ground = states.generate_coupled_states_ground(np.arange(0, 15))
    coupled_hashes = [hash(s) for s in coupled_ground]
    assert len(coupled_hashes) == len(set(coupled_hashes)), (
        f"Found hash collisions in coupled ground states: "
        f"{len(coupled_hashes)} states, {len(set(coupled_hashes))} unique hashes"
    )
    
    # Test no collisions in large state space for uncoupled states
    uncoupled_ground = states.generate_uncoupled_states_ground(np.arange(0, 15))
    uncoupled_hashes = [hash(s) for s in uncoupled_ground]
    assert len(uncoupled_hashes) == len(set(uncoupled_hashes)), (
        f"Found hash collisions in uncoupled ground states: "
        f"{len(uncoupled_hashes)} states, {len(set(uncoupled_hashes))} unique hashes"
    )
    
    # Test no collisions in excited states with various quantum numbers
    coupled_excited = states.generate_coupled_states_excited(
        np.arange(0, 10), Ps=None, Omegas=[-1, 1]  # type: ignore[arg-type]
    )
    excited_hashes = [hash(s) for s in coupled_excited]
    assert len(excited_hashes) == len(set(excited_hashes)), (
        f"Found hash collisions in coupled excited states: "
        f"{len(excited_hashes)} states, {len(set(excited_hashes))} unique hashes"
    )
    
    # Test that states can be used in sets (requires proper __hash__ and __eq__)
    state_set = {state1, state1_copy, state2}
    assert len(state_set) == 2, "Set should contain only unique states"
    assert state1 in state_set
    assert state2 in state_set
    
    # Test that states can be used as dictionary keys
    state_dict = {state1: "first", state2: "second"}
    assert state_dict[state1_copy] == "first", "Equal states should access same dict entry"
    assert state_dict[state2] == "second"
