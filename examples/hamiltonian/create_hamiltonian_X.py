from centrex_tlf import hamiltonian, states

# generate the hyperfine sublevels in J=0 and J=1
QN = states.generate_uncoupled_states_ground(Js=[0, 1, 2, 3, 4, 5, 6])

# generate a dictionary with X hamiltonian terms
H = hamiltonian.generate_uncoupled_hamiltonian_X(QN)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_uncoupled_hamiltonian_X_function(H)

# print(Hfunc([0, 0, 0], [0, 0, 0]))
