import numpy as np
import sympy as smp

from centrex_tlf import couplings, hamiltonian, lindblad, states


def test_generate_symbolic_hamiltonian():
    x_select = states.QuantumSelector(J=1)
    b_select = states.QuantumSelector(J=1, F=1, F1=1 / 2, P=1)
    transitions = [
        couplings.TransitionSelector(
            ground=1 * states.generate_coupled_states_X(x_select),
            excited=1 * states.generate_coupled_states_B(b_select),
            polarizations=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            polarization_symbols=smp.symbols("Plx Plz"),
            Ω=smp.Symbol("Ωl", comlex=True),
            δ=smp.Symbol("δl"),
            description="Q1 F1'=1/2 F'=1",
        )
    ]
    H_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        X_states_approx=list(states.generate_coupled_states_X(x_select)),
        B_states_approx=list(states.generate_coupled_states_B(b_select)),
    )
    QN = [1 * s for s in states.generate_coupled_states_X(x_select)] + [
        1 * s for s in states.generate_coupled_states_B(b_select)
    ]
    coupl = []
    for transition in transitions:
        coupl.append(
            couplings.generate_coupling_field_automatic(
                [1 * s for s in transition.ground],
                [1 * s for s in transition.excited],
                QN,
                H_reduced.H_int,
                H_reduced.QN,
                H_reduced.V_ref_int,
                pol_vecs=transition.polarizations,
            )
        )
    hamiltonian_symbolic = lindblad.generate_rwa_symbolic_hamiltonian(
        QN=H_reduced.QN,
        H_int=H_reduced.H_int,
        couplings=coupl,
        Ωs=[smp.Symbol("Ωl", complex=True)],
        δs=[smp.Symbol("δl")],
        pols=[[smp.Symbol("Plx"), smp.Symbol("Plz")]],
    )
    δl = smp.Symbol("δl")
    true_values = [
        1.0 * δl - 1336622.01036072,
        1.0 * δl - 1196891.60754395,
        1.0 * δl - 1196891.61817932,
        1.0 * δl - 1196891.62893677,
        1.0 * δl - 91349.8655090332,
        1.0 * δl - 91349.8653564453,
        1.0 * δl - 91349.8651580811,
        1.0 * δl + 0.0206298828125,
        1.0 * δl + 0.0102691650390625,
        1.0 * δl,
        1.0 * δl - 0.0102996826171875,
        1.0 * δl - 0.020599365234375,
        5.65187168121338,
        2.82593345642090,
        0,
    ]
    for dh, dtv in zip(np.diag(hamiltonian_symbolic), true_values):
        assert np.abs(dh - dtv) <= 1e-3


def test_generate_total_symbolic_hamiltonian():
    x_select = states.QuantumSelector(J=1)
    x_select_compact = states.QuantumSelector(J=3, electronic=states.ElectronicState.X)
    b_select = states.QuantumSelector(J=1, F=1, F1=1 / 2, P=1)
    transitions = [
        couplings.TransitionSelector(
            ground=[1 * s for s in states.generate_coupled_states_X(x_select)],
            excited=[1 * s for s in states.generate_coupled_states_B(b_select)],
            polarizations=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            polarization_symbols=smp.symbols("Plx Plz"),
            Ω=smp.Symbol("Ωl", comlex=True),
            δ=smp.Symbol("δl"),
            description="Q1 F1'=1/2 F'=1",
        )
    ]
    H_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        X_states_approx=list(states.generate_coupled_states_X(x_select))
        + list(states.generate_coupled_states_X(x_select_compact)),
        B_states_approx=list(states.generate_coupled_states_B(b_select)),
    )
    QN = (
        [1 * s for s in states.generate_coupled_states_X(x_select)]
        + [1 * s for s in states.generate_coupled_states_X(x_select_compact)]
        + [1 * s for s in states.generate_coupled_states_B(b_select)]
    )
    coupl = []
    for transition in transitions:
        coupl.append(
            couplings.generate_coupling_field_automatic(
                transition.ground,
                transition.excited,
                QN,
                H_reduced.H_int,
                H_reduced.QN,
                H_reduced.V_ref_int,
                pol_vecs=transition.polarizations,
            )
        )
    hamiltonian_symbolic, QN_compact = lindblad.generate_total_symbolic_hamiltonian(
        QN=H_reduced.QN,
        H_int=H_reduced.H_int,
        couplings=coupl,
        transitions=transitions,
        qn_compact=x_select_compact,
    )
    assert hamiltonian_symbolic.shape == (16, 16)
    assert states.QuantumSelector(J=3, electronic=states.ElectronicState.X).get_indices(
        QN_compact
    ) == np.array([12])

    δl = smp.Symbol("δl")
    true_values = [
        1.0 * δl - 1336622.00978088,
        1.0 * δl - 1196891.60691833,
        1.0 * δl - 1196891.61756897,
        1.0 * δl - 1196891.62826538,
        1.0 * δl - 91349.8648376465,
        1.0 * δl - 91349.8646697998,
        1.0 * δl - 91349.8645477295,
        1.0 * δl + 0.0206451416015625,
        1.0 * δl + 0.0102691650390625,
        1.0 * δl,
        1.0 * δl - 0.0103607177734375,
        1.0 * δl - 0.0205841064453125,
        502704902817.902,
        5.65187168121338,
        2.82593345642090,
        0,
    ]
    for dh, dtv in zip(np.diag(hamiltonian_symbolic), true_values):
        _dtv = float(dtv.subs(δl, 0)) if not isinstance(dtv, (int, float)) else dtv
        _dtv = 1.0 if _dtv == 0 else _dtv
        assert abs(dh - dtv) / abs(_dtv) < 1e-2
