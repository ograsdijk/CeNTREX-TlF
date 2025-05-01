import numpy as np

from centrex_tlf import hamiltonian, transitions

trans = transitions.OpticalTransition(
    transitions.OpticalTransitionType.R, J_ground=0, F1_excited=3 / 2, F_excited=2
)

# include Jmax_X to include mixing of ground states. The B state calculation
# automatically uses Jmax_B = the largest J included + 2
H_red_electric = hamiltonian.generate_reduced_hamiltonian_transitions(
    [trans], E=np.array([0, 0, 500]), Jmax_X=4
)
