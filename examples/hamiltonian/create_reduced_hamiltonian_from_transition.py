from centrex_tlf import hamiltonian, transitions

trans = transitions.OpticalTransition(
    transitions.OpticalTransitionType.R, J_ground=0, F1=3 / 2, F=2
)

H_red = hamiltonian.generate_reduced_hamiltonian_transitions([trans])
