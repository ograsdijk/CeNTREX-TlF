from centrex_tlf import couplings, states


def test_ED_ME_coupled():
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
    dipole = couplings.ED_ME_coupled(ground_state, excited_state, rme_only=True)
    assert dipole == 0.816496580927726 + 0j

    dipole = couplings.ED_ME_coupled(ground_state, excited_state, rme_only=False)
    assert dipole == (0 + 0j)


def test_generate_ED_ME_mixed_state():
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
    ground_state = states.CoupledBasisState(
        J=1,
        F1=1 / 2,
        F=1,
        mF=1,
        I1=1 / 2,
        I2=1 / 2,
        Omega=0,
        P=-1,
        electronic_state=states.ElectronicState.X,
    )
    dipole = couplings.calculate_ED_ME_mixed_state(1 * ground_state, 1 * excited_state)
    assert dipole == (-0.19245008972987523 + 0.19245008972987523j)
