from centrex_tlf import hamiltonian, states

# generate the hyperfine sublevels in J=0 and J=1
QN = states.generate_coupled_states_excited(Js=[0, 1, 2, 3])

# generate a dictionary with X hamiltonian terms
H = hamiltonian.generate_coupled_hamiltonian_B(QN)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_coupled_hamiltonian_B_function(H)

# print(QN)

# print(Hfunc([0, 0, 0], [0, 0, 0]))
