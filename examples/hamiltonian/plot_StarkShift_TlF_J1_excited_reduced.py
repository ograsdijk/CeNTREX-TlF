import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from centrex_tlf import hamiltonian, states
from centrex_tlf.states.generate_states import generate_coupled_states_B

# generate the hyperfine sublevels in J=1
qn_select = states.QuantumSelector(J=1, F1=1 / 2, F=1, P=+1, Ω=1)
QN = generate_coupled_states_B(qn_select)

# generate the B hamiltonian terms
qnc = states.generate_coupled_states_excited(
    Js=np.arange(1, 6), Ps=[-1, 1], Omegas=1, basis=states.Basis.CoupledP
)
H = hamiltonian.generate_coupled_hamiltonian_B(qnc)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_coupled_hamiltonian_B_function(H)

# generate the zero field hamiltonion
QN_states, H = hamiltonian.generate_reduced_B_hamiltonian(
    QN, H_func=Hfunc, Jmin=1, Jmax=5
)

# original eigenvectors used in tracking states as energies change order
E, V = np.linalg.eigh(H)
V_track = V.copy()

# V/cm
Ez = np.linspace(0, 100, 21)

# empty array for storing energies
energy = np.empty([Ez.size, len(QN)], dtype=np.float64)

# iterate over the electric field values
for idx, Ei in tqdm(enumerate(Ez), total=len(Ez)):
    _, H = hamiltonian.generate_reduced_B_hamiltonian(
        QN, E=[0.0, 0.0, Ei], B=[0.0, 0.0, 1e-3], Jmin=1, Jmax=5, H_func=Hfunc
    )
    E, V = np.linalg.eigh(H)

    # sort indices to keep the state order the same
    indices = np.argmax(np.abs(V_track.conj().T @ V), axis=1)
    energy[idx, :] = E[indices]
    V_track[:, :] = V[:, indices]

# plot the J'=2 Stark curves
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(Ez, (energy.real - energy.real[0, 0]) / (2 * np.pi * 1e9), lw=3, color="k")
ax.set_xlabel("E [V/cm]")
ax.set_ylabel("Energy [GHz]")
ax.set_title("|J'=1> Stark Curve")
ax.grid(True)

plt.show()
