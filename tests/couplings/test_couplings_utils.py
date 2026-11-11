import numpy as np

from centrex_tlf import couplings, states


def test_check_transition_coupled_allowed():
    ground_state = states.CoupledBasisState(
        J=1,
        F1=1 / 2,
        F=1,
        mF=0,
        I1=1 / 2,
        I2=1 / 2,
        Omega=0,
        P=-1,
        electronic_state=states.ElectronicState.X,
    )
    excited_state = states.CoupledBasisState(
        J=1,
        F1=1 / 2,
        F=1,
        mF=0,
        I1=1 / 2,
        I2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )
    allowed = couplings.utils.check_transition_coupled_allowed_polarization(
        ground_state, excited_state, ΔmF_allowed=0, return_err=False
    )
    if isinstance(allowed, bool):
        assert not allowed

    excited_state = states.CoupledBasisState(
        J=1,
        F1=1 / 2,
        F=1,
        mF=-1,
        I1=1 / 2,
        I2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )
    allowed = couplings.utils.check_transition_coupled_allowed_polarization(
        ground_state, excited_state, ΔmF_allowed=0, return_err=False
    )
    if isinstance(allowed, bool):
        assert not allowed

    allowed = couplings.utils.check_transition_coupled_allowed_polarization(
        ground_state, excited_state, ΔmF_allowed=-1, return_err=False
    )
    print(allowed)
    if isinstance(allowed, bool):
        assert allowed

    excited_state = states.CoupledBasisState(
        J=1,
        F1=1 / 2,
        F=1,
        mF=1,
        I1=1 / 2,
        I2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )
    allowed = couplings.utils.check_transition_coupled_allowed_polarization(
        ground_state, excited_state, ΔmF_allowed=1, return_err=False
    )
    if isinstance(allowed, bool):
        assert allowed

    excited_state = states.CoupledBasisState(
        J=1,
        F1=1 / 2,
        F=1,
        mF=1,
        I1=1 / 2,
        I2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )
    allowed = couplings.utils.check_transition_coupled_allowed_polarization(
        ground_state, excited_state, ΔmF_allowed=-1, return_err=False, ΔmF_absolute=True
    )
    if isinstance(allowed, bool):
        assert allowed


def test_compact_collapse_operators_matches_dense_routine():
    """Single-element compaction gives exactly what the dense routine gives."""
    from centrex_tlf.couplings.utils_compact import (
        CollapseOperators,
        compact_C_array_indices,
        compact_collapse_operators,
    )

    rng = np.random.default_rng(0)
    n, n_ground, gamma = 14, 10, 1.3e7
    rows, cols = np.nonzero(rng.random((n_ground, n - n_ground)) < 0.7)
    cols = cols + n_ground
    values = np.sqrt(rng.random(len(rows)) * gamma)
    operators = CollapseOperators(rows, cols, values, n)

    dense = operators.to_dense()
    assert dense.shape == (len(rows), n, n)
    assert dense.dtype == np.float64

    # two compactions in a row; the second runs on the complex output of the first
    for indices in (np.array([2, 3, 4, 5]), np.array([0, 1])):
        dense = compact_C_array_indices(dense, gamma, indices)
        operators = compact_collapse_operators(operators, gamma, indices)
        assert operators is not None
        result = operators.to_dense()
        assert result.dtype == dense.dtype
        assert result.shape == dense.shape
        np.testing.assert_array_equal(result, dense)

    # compacting states no operator touches leaves the values untouched
    untouched = np.array([8, 9]) - 4
    dense = compact_C_array_indices(dense, gamma, untouched)
    operators = compact_collapse_operators(operators, gamma, untouched)
    np.testing.assert_array_equal(operators.to_dense(), dense)


def test_compact_collapse_operators_defers_special_cases_to_dense_routine():
    from centrex_tlf.couplings.utils_compact import (
        CollapseOperators,
        compact_collapse_operators,
    )

    empty = CollapseOperators(
        np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp), np.zeros(0), 4
    )
    assert compact_collapse_operators(empty, 1.0, np.array([0, 1])) is None

    duplicated = CollapseOperators(np.array([0, 0]), np.array([3, 3]), np.array([1.0, 2.0]), 4)
    assert compact_collapse_operators(duplicated, 1.0, np.array([0, 1])) is None

    with_zero = CollapseOperators(np.array([0, 1]), np.array([3, 3]), np.array([1.0, 0.0]), 4)
    assert compact_collapse_operators(with_zero, 1.0, np.array([0, 1])) is None
