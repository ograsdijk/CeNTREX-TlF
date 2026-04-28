import numpy as np

from centrex_tlf import couplings, hamiltonian, states


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
    dipole = hamiltonian.ED_ME_coupled(ground_state, excited_state, rme_only=True)
    assert dipole == 0.816496580927726 + 0j

    dipole = hamiltonian.ED_ME_coupled(ground_state, excited_state, rme_only=False)
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
    dipole = hamiltonian.generate_ED_ME_mixed_state(1 * ground_state, 1 * excited_state)
    assert dipole == (0.2721655269759087 + 0j)


def test_ED_ME_uncoupled():
    bra = states.UncoupledBasisState(
        J=1,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=0,
        P=1,
        electronic_state=states.ElectronicState.X,
    )
    ket = states.UncoupledBasisState(
        J=0,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=0,
        P=1,
        electronic_state=states.ElectronicState.X,
    )

    dipole_rme = hamiltonian.ED_ME_uncoupled(bra, ket, rme_only=True)
    assert np.isclose(dipole_rme, (1 + 0j))

    dipole = hamiltonian.ED_ME_uncoupled(bra, ket, pol_vec=(0j, 0j, 1 + 0j), rme_only=False)
    assert np.isclose(dipole, (0.5773502691896257 + 0j))


def test_uncoupled_parity_to_omega_expansion_is_shared_convention():
    state = states.UncoupledBasisState(
        J=1,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )

    expanded = states.expand_uncoupled_parity_to_omega_components(state)
    transformed = state.transform_to_omega_basis()

    assert transformed.data == expanded
    assert expanded[0][1].Omega == 1
    assert expanded[1][1].Omega == -1
    assert np.isclose(expanded[0][0], 1 / np.sqrt(2))
    assert np.isclose(expanded[1][0], -1 / np.sqrt(2))


def test_generate_ED_ME_mixed_state_uncoupled():
    bra = states.UncoupledBasisState(
        J=1,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=0,
        P=1,
        electronic_state=states.ElectronicState.X,
    )
    ket = states.UncoupledBasisState(
        J=0,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=0,
        P=1,
        electronic_state=states.ElectronicState.X,
    )

    dipole = hamiltonian.generate_ED_ME_mixed_state_uncoupled(1 * bra, 1 * ket, pol_vec=[0, 0, 1])
    assert np.isclose(dipole, (0.5773502691896257 + 0j))


def test_ED_ME_uncoupled_selection_rules():
    bra = states.UncoupledBasisState(
        J=2,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=0,
        P=-1,
        electronic_state=states.ElectronicState.X,
    )
    ket = states.UncoupledBasisState(
        J=2,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )

    # nuclear spin projections must be conserved
    ket_bad_m1 = states.UncoupledBasisState(
        J=2,
        mJ=0,
        I1=1 / 2,
        m1=-1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )
    assert hamiltonian.ED_ME_uncoupled(bra, ket_bad_m1, rme_only=True) == 0j

    # for full ME, |ΔmJ| <= 1
    ket_bad_dmJ = states.UncoupledBasisState(
        J=2,
        mJ=2,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )
    assert hamiltonian.ED_ME_uncoupled(bra, ket_bad_dmJ, rme_only=False) == 0j


def test_generate_ED_ME_mixed_state_uncoupled_matches_basis_formula():
    # Pick a simple uncoupled pair and compare to the explicit formula.
    bra = states.UncoupledBasisState(
        J=1,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=0,
        P=-1,
        electronic_state=states.ElectronicState.X,
    )
    ket = states.UncoupledBasisState(
        J=1,
        mJ=1,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=1 / 2,
        Omega=1,
        P=1,
        electronic_state=states.ElectronicState.B,
    )

    pol = (0j, 0j, 1.0 + 0j)  # π (Ez)
    direct = hamiltonian.ED_ME_uncoupled(bra, ket, pol_vec=pol, rme_only=False)
    mixed = hamiltonian.generate_ED_ME_mixed_state_uncoupled(
        1 * bra, 1 * ket, pol_vec=np.array(pol, dtype=np.complex128)
    )
    assert direct == mixed


def test_generate_ED_ME_mixed_state_uncoupled_matches_coupled():
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

    expected = hamiltonian.generate_ED_ME_mixed_state(1 * ground_state, 1 * excited_state)

    ground_uncoupled = ground_state.transform_to_uncoupled()
    excited_uncoupled = excited_state.transform_to_uncoupled()

    got = hamiltonian.generate_ED_ME_mixed_state_uncoupled(
        ground_uncoupled, excited_uncoupled
    )
    assert got == expected


def test_ED_ME_uncoupled_matches_uncoupled_mixed_state_for_basis_states():
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

    # Take one uncoupled basis component from each transformed state
    ground_unc = ground_state.transform_to_uncoupled().data[0][1]
    excited_unc = excited_state.transform_to_uncoupled().data[0][1]

    pol = couplings.polarization_unpolarized.vector
    dipole_basis = hamiltonian.ED_ME_uncoupled(
        ground_unc, excited_unc, pol_vec=tuple(pol), rme_only=False
    )
    dipole_mixed = hamiltonian.generate_ED_ME_mixed_state_uncoupled(
        1 * ground_unc,
        1 * excited_unc,
        pol_vec=pol,
        normalize_pol=False,
    )
    assert dipole_basis == dipole_mixed
