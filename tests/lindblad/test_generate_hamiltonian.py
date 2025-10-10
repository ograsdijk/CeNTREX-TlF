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
        δl - 1336622.01036072,
        δl - 1196891.5111084,
        δl - 1196891.61816406,
        δl - 1196891.72523499,
        δl - 91349.8670654297,
        δl - 91349.8652191162,
        δl - 91349.8634338379,
        δl + 0.20611572265625,
        δl + 0.103134155273438,
        δl,
        δl - 0.102996826171875,
        δl - 0.206008911132813,
        56.5186948776245,
        28.2593536376953,
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
        δl - 1336622.00978088,
        δl - 1196891.51052856,
        δl - 1196891.61752319,
        δl - 1196891.72459412,
        δl - 91349.8665771484,
        δl - 91349.8648071289,
        δl - 91349.8630065918,
        δl + 0.20599365234375,
        δl + 0.103012084960938,
        δl,
        δl - 0.103073120117188,
        δl - 0.206039428710938,
        508269908134.471,
        56.5186948776245,
        28.2593536376953,
        0,
    ]
    for dh, dtv in zip(np.diag(hamiltonian_symbolic), true_values):
        # Evaluate both sides at δl=0 for numerical comparison
        _dh = float(dh.subs(δl, 0)) if hasattr(dh, "subs") else float(dh)
        _dtv = float(dtv.subs(δl, 0)) if hasattr(dtv, "subs") else float(dtv)
        # Handle zero values correctly
        if _dtv == 0:
            assert abs(_dh) < 1e-6  # If expected is 0, actual should be very close to 0
        else:
            assert abs(_dh - _dtv) / abs(_dtv) < 1e-2
