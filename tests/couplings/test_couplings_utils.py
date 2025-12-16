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
