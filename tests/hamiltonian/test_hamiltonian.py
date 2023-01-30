import numpy as np
from scipy.constants import physical_constants

from centrex_tlf import hamiltonian, states
from centrex_tlf.states.states import ElectronicState


def test_gfactor_B_hamiltonian():
    # generate the hyperfine sublevels in J=0 to J=3
    QN = states.generate_coupled_states_excited(
        Js=np.arange(1, 4), Ps=[-1, 1], Omegas=1, basis=states.Basis.CoupledP
    )

    # generate the X hamiltonian terms
    H = hamiltonian.generate_coupled_hamiltonian_B(QN)

    # create a function outputting the hamiltonian as a function of E and B
    Hfunc = hamiltonian.generate_coupled_hamiltonian_B_function(H)

    # generate the Hamiltonian for (almost) zero field, add a small field to make states
    # non-degenerate
    H_0G = Hfunc(E=[0.0, 0.0, 0.0], B=[0.0, 0.0, 0.000001])
    E_0G, V_0G = np.linalg.eigh(H_0G)

    # get the true superposition-states of the system
    QN_states = hamiltonian.matrix_to_states(V_0G, QN)

    H_1G = Hfunc(E=[0.0, 0.0, 0.0], B=[0.0, 0.0, 1.0])
    E_1G, V_1G = np.linalg.eigh(H_1G)

    # sort indices to keep the state order the same
    indices = np.argmax(np.abs(V_0G.conj().T @ V_1G), axis=1)
    E_1G = E_1G[indices]
    V_1G[:, :] = V_1G[:, indices]

    # convert J/T to Hz/G with factor 1.509e29
    gFactors = -(E_1G - E_0G) / (
        2 * np.pi * physical_constants["Bohr magneton"][0] * 1.509 * 1e29
    )

    states_to_check = [
        states.CoupledBasisState(
            J=1,
            F1=1 / 2,
            F=0,
            mF=0,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
        states.CoupledBasisState(
            J=1,
            F1=1 / 2,
            F=1,
            mF=1,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
        states.CoupledBasisState(
            J=1,
            F1=3 / 2,
            F=1,
            mF=1,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
        states.CoupledBasisState(
            J=1,
            F1=3 / 2,
            F=2,
            mF=1,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
        states.CoupledBasisState(
            J=2,
            F1=5 / 2,
            F=2,
            mF=1,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
        states.CoupledBasisState(
            J=2,
            F1=5 / 2,
            F=3,
            mF=1,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
        states.CoupledBasisState(
            J=2,
            F1=3 / 2,
            F=1,
            mF=1,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
        states.CoupledBasisState(
            J=2,
            F1=3 / 2,
            F=2,
            mF=1,
            I1=1 / 2,
            I2=1 / 2,
            Omega=1,
            P=1,
            electronic_state=ElectronicState.B,
        ),
    ]
    indices_states = [
        ids
        for ids, s in enumerate(QN_states)
        for sc in states_to_check
        if s.largest == sc
    ]

    assert np.allclose(
        gFactors[indices_states],
        np.array(
            [
                3.14075269e-04,
                3.21404093e-01,
                2.53551426e-01,
                1.41945999e-01,
                1.10764994e-01,
                7.64882501e-02,
                4.25547758e-01,
                2.54489133e-01,
            ]
        ),
    )
