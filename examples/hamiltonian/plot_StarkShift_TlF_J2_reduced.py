import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from centrex_tlf import hamiltonian
from centrex_tlf.constants import TlFNuclearSpins
from centrex_tlf.hamiltonian.basis_transformations import generate_transform_matrix
from centrex_tlf.states.find_states import QuantumSelector
from centrex_tlf.states.generate_states import (
    generate_coupled_states_ground,
    generate_coupled_states_X,
    generate_uncoupled_states_ground,
)

# generate the hyperfine sublevels in J=2
qn_select = QuantumSelector(J=[2])
QN = generate_coupled_states_X(qn_select)

# generate the transformation matrix
qn = generate_uncoupled_states_ground(
    Js=np.arange(0, 6 + 1), nuclear_spins=TlFNuclearSpins()
)
qnc = generate_coupled_states_ground(
    Js=np.arange(0, 6 + 1), nuclear_spins=TlFNuclearSpins()
)
transform = generate_transform_matrix(qn, qnc)

# generate the X hamiltonian terms
H = hamiltonian.generate_uncoupled_hamiltonian_X(qn)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_uncoupled_hamiltonian_X_function(H)

# generate the reduced X hamiltonian
QN_states, H = hamiltonian.generate_reduced_X_hamiltonian(
    QN, Jmin=0, Jmax=6, transform=transform, H_func=Hfunc
)

E, V = np.linalg.eigh(H)

# V/cm
Ez = np.linspace(0, 15e3, 51)

# original eigenvectors used in tracking states as energies change order
V_track = V.copy()

# empty array for storing energies
energy = np.zeros([Ez.size, len(QN)], dtype=np.complex128)

# iterate over the electric field values
for idx, Ei in tqdm(enumerate(Ez), total=len(Ez)):
    _, H = hamiltonian.generate_reduced_X_hamiltonian(
        QN,
        E=[0.0, 0.0, Ei],
        B=[0.0, 0.0, 1e-3],
        Jmin=0,
        Jmax=6,
        transform=transform,
        H_func=Hfunc,
    )
    E, V = np.linalg.eigh(H)

    # sort indices to keep the state order the same
    indices = np.argmax(np.abs(V_track.conj().T @ V), axis=1)
    energy[idx, :] = E[indices]
    V_track[:, :] = V[:, indices]

# plot the J=2, mJ=0 Stark curves
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(Ez, (energy.real - energy.real[0, 0]) / (2 * np.pi * 1e9), lw=3, color="k")
ax.set_xlabel("E [V/cm]")
ax.set_ylabel("Energy [GHz]")
ax.set_title("|J=2> Stark Curve")
ax.grid(True)

plt.show()
