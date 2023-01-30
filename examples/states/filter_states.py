from centrex_tlf import states

QN = states.generate_uncoupled_states_ground(Js=[0, 1])

QN = states.generate_coupled_states_ground(Js=[0, 1])
qn_select = states.QuantumSelector(J=1, mF=0, electronic=states.ElectronicState.X)

print(qn_select)
print(qn_select.get_indices(QN))
