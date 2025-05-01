from centrex_tlf import hamiltonian, transitions

trans = transitions.OpticalTransition(
    transitions.OpticalTransitionType.R, J_ground=0, F1_excited=3 / 2, F_excited=2
)

H_red = hamiltonian.generate_reduced_hamiltonian_transitions([trans])
