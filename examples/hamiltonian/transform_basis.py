from centrex_tlf import hamiltonian, states

# generate the hyperfine sublevels in J=0 and J=1
QN = states.generate_uncoupled_states_ground(Js=[0, 1])
# generate the coupled hyperfine sublevels in J=0 and J=1
QNc = states.generate_coupled_states_ground(Js=[0, 1])

# generate a dictionary with X hamiltonian terms
H = hamiltonian.generate_uncoupled_hamiltonian_X(QN)
Hfunc = hamiltonian.generate_uncoupled_hamiltonian_X_function(H)
H0 = Hfunc(E=[0, 0, 0], B=[0, 0, 1e-3])

# generate the transformation matrix
transform = hamiltonian.generate_transform_matrix(QN, QNc)

# calculate the transformed matrix
H0c = transform.conj().T @ H0 @ transform
